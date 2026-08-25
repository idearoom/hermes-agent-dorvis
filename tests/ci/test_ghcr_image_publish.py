"""Contracts for the IdeaRoom GHCR immutable-image publisher.

The publisher's public boundary is the registry manifest: a local Docker image
ID addresses its config JSON, while the digest consumers pin addresses the
registry manifest.  These tests keep that distinction explicit and exercise
the fail-closed decisions without requiring Docker or network access.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[2]
_PATH = _REPO / "scripts" / "ci" / "publish_immutable_image.py"
_spec = importlib.util.spec_from_file_location("publish_immutable_image", _PATH)
if _spec is None or _spec.loader is None:
    raise ImportError("Failed to load publish_immutable_image.py")
_mod = importlib.util.module_from_spec(_spec)
sys.modules["publish_immutable_image"] = _mod
_spec.loader.exec_module(_mod)

LOCAL_IMAGE = "ghcr.io/idearoom/hermes-agent-dorvis:candidate-123-1"
REVISION = "a" * 40
IMMUTABLE_IMAGE = f"ghcr.io/idearoom/hermes-agent-dorvis:sha-{REVISION}"
SOURCE = "https://github.com/idearoom/hermes-agent-dorvis"
LOCAL_CONFIG = "sha256:" + "1" * 64
OTHER_CONFIG = "sha256:" + "2" * 64


def _completed(
    args: list[str],
    *,
    stdout: str = "",
    stderr: str = "",
    returncode: int = 0,
) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(args, returncode, stdout, stderr)


class ScriptedDocker:
    """Small command seam that requires every expected Docker call in order."""

    def __init__(
        self, responses: list[tuple[list[str], subprocess.CompletedProcess[str]]]
    ):
        self.responses = list(responses)
        self.calls: list[list[str]] = []

    def run(
        self,
        args: list[str],
        *,
        capture: bool = True,
        check: bool = True,
    ) -> subprocess.CompletedProcess[str]:
        self.calls.append(args)
        assert self.responses, f"unexpected Docker call: {args!r}"
        expected, result = self.responses.pop(0)
        assert args == expected
        if check and result.returncode:
            raise subprocess.CalledProcessError(
                result.returncode,
                args,
                output=result.stdout,
                stderr=result.stderr,
            )
        return result

    def assert_finished(self) -> None:
        assert self.responses == []


def _json_result(args: list[str], value: object) -> subprocess.CompletedProcess[str]:
    return _completed(args, stdout=json.dumps(value, separators=(",", ":")))


def _descriptor(raw: str, media_type: str) -> dict[str, object]:
    return {
        "schemaVersion": 2,
        "mediaType": media_type,
        "digest": "sha256:" + hashlib.sha256(raw.encode()).hexdigest(),
        "size": len(raw.encode()),
    }


def _image_manifest(config_digest: str) -> tuple[str, dict[str, object]]:
    raw = json.dumps(
        {
            "schemaVersion": 2,
            "mediaType": _mod.OCI_IMAGE_MANIFEST,
            "config": {
                "mediaType": _mod.OCI_IMAGE_CONFIG,
                "size": 123,
                "digest": config_digest,
            },
            "layers": [],
        },
        separators=(",", ":"),
    )
    return raw, _descriptor(raw, _mod.OCI_IMAGE_MANIFEST)


def _local_inspect_args() -> list[str]:
    return ["docker", "image", "inspect", LOCAL_IMAGE]


def _local_inspect_result() -> subprocess.CompletedProcess[str]:
    payload = [
        {
            "Id": LOCAL_CONFIG,
            "Architecture": "arm64",
            "Os": "linux",
            "Config": {
                "Labels": {
                    "org.opencontainers.image.revision": REVISION,
                    "org.opencontainers.image.source": SOURCE,
                }
            },
        }
    ]
    return _json_result(_local_inspect_args(), payload)


def _remote_commands(reference: str) -> tuple[list[str], list[str]]:
    return (
        [
            "docker",
            "buildx",
            "imagetools",
            "inspect",
            reference,
            "--format",
            "{{json .Manifest}}",
        ],
        ["docker", "buildx", "imagetools", "inspect", reference, "--raw"],
    )


def test_inspect_remote_resolves_arm64_from_an_index_and_verifies_digests():
    child_raw, child_descriptor = _image_manifest(LOCAL_CONFIG)
    index_raw = json.dumps(
        {
            "schemaVersion": 2,
            "mediaType": _mod.OCI_IMAGE_INDEX,
            "manifests": [
                {
                    **child_descriptor,
                    "platform": {
                        "os": "linux",
                        "architecture": "arm64",
                        "variant": "v8",
                    },
                },
                {
                    "mediaType": _mod.OCI_IMAGE_MANIFEST,
                    "digest": "sha256:" + "9" * 64,
                    "size": 99,
                    "platform": {"os": "unknown", "architecture": "unknown"},
                    "annotations": {
                        "vnd.docker.reference.type": "attestation-manifest"
                    },
                },
            ],
        },
        separators=(",", ":"),
    )
    index_descriptor = _descriptor(index_raw, _mod.OCI_IMAGE_INDEX)
    format_args, raw_args = _remote_commands(IMMUTABLE_IMAGE)
    child_ref = f"ghcr.io/idearoom/hermes-agent-dorvis@{child_descriptor['digest']}"
    cli = ScriptedDocker([
        (format_args, _json_result(format_args, index_descriptor)),
        (raw_args, _completed(raw_args, stdout=index_raw)),
        (
            ["docker", "buildx", "imagetools", "inspect", child_ref, "--raw"],
            _completed([], stdout=child_raw),
        ),
    ])

    remote = _mod.inspect_remote_image(IMMUTABLE_IMAGE, cli)

    assert remote is not None
    assert remote.digest == index_descriptor["digest"]
    assert remote.config_digest == LOCAL_CONFIG
    cli.assert_finished()


def test_arm64_index_descriptor_must_claim_an_image_manifest():
    child_raw, child_descriptor = _image_manifest(LOCAL_CONFIG)
    child_descriptor["mediaType"] = _mod.OCI_IMAGE_INDEX
    index_raw = json.dumps(
        {
            "schemaVersion": 2,
            "mediaType": _mod.OCI_IMAGE_INDEX,
            "manifests": [
                {
                    **child_descriptor,
                    "platform": {"os": "linux", "architecture": "arm64"},
                }
            ],
        },
        separators=(",", ":"),
    )
    index_descriptor = _descriptor(index_raw, _mod.OCI_IMAGE_INDEX)
    format_args, raw_args = _remote_commands(IMMUTABLE_IMAGE)
    cli = ScriptedDocker([
        (format_args, _json_result(format_args, index_descriptor)),
        (raw_args, _completed(raw_args, stdout=index_raw)),
    ])

    with pytest.raises(_mod.PublishError, match="arm64 descriptor media type"):
        _mod.inspect_remote_image(IMMUTABLE_IMAGE, cli)

    cli.assert_finished()


def test_arm64_descriptor_and_manifest_media_types_must_agree():
    child_raw, child_descriptor = _image_manifest(LOCAL_CONFIG)
    child_descriptor["mediaType"] = _mod.DOCKER_IMAGE_MANIFEST
    index_raw = json.dumps(
        {
            "schemaVersion": 2,
            "mediaType": _mod.OCI_IMAGE_INDEX,
            "manifests": [
                {
                    **child_descriptor,
                    "platform": {"os": "linux", "architecture": "arm64"},
                }
            ],
        },
        separators=(",", ":"),
    )
    index_descriptor = _descriptor(index_raw, _mod.OCI_IMAGE_INDEX)
    format_args, raw_args = _remote_commands(IMMUTABLE_IMAGE)
    child_ref = f"ghcr.io/idearoom/hermes-agent-dorvis@{child_descriptor['digest']}"
    cli = ScriptedDocker([
        (format_args, _json_result(format_args, index_descriptor)),
        (raw_args, _completed(raw_args, stdout=index_raw)),
        (
            ["docker", "buildx", "imagetools", "inspect", child_ref, "--raw"],
            _completed([], stdout=child_raw),
        ),
    ])

    with pytest.raises(_mod.PublishError, match="descriptor and manifest media types"):
        _mod.inspect_remote_image(IMMUTABLE_IMAGE, cli)

    cli.assert_finished()


def test_inspect_remote_treats_only_reference_not_found_as_absent():
    format_args, _ = _remote_commands(IMMUTABLE_IMAGE)
    missing = ScriptedDocker([
        (
            format_args,
            _completed(
                format_args,
                returncode=1,
                stderr=f"ERROR: {IMMUTABLE_IMAGE}: not found",
            ),
        )
    ])
    assert _mod.inspect_remote_image(IMMUTABLE_IMAGE, missing) is None

    dns_failure = ScriptedDocker([
        (
            format_args,
            _completed(
                format_args,
                returncode=1,
                stderr="failed to do request: dial tcp: lookup ghcr.io: no such host",
            ),
        )
    ])
    with pytest.raises(_mod.PublishError, match="could not determine whether"):
        _mod.inspect_remote_image(IMMUTABLE_IMAGE, dns_failure)

    unscoped_unknown = ScriptedDocker([
        (
            format_args,
            _completed(
                format_args,
                returncode=1,
                stderr="registry error: manifest unknown",
            ),
        )
    ])
    with pytest.raises(_mod.PublishError, match="could not determine whether"):
        _mod.inspect_remote_image(IMMUTABLE_IMAGE, unscoped_unknown)


def test_image_manifest_requires_a_container_config_descriptor():
    manifest = {
        "schemaVersion": 2,
        "mediaType": _mod.OCI_IMAGE_MANIFEST,
        "config": {
            "mediaType": "application/octet-stream",
            "size": 123,
            "digest": LOCAL_CONFIG,
        },
        "layers": [],
    }
    raw = json.dumps(manifest, separators=(",", ":"))
    descriptor = _descriptor(raw, _mod.OCI_IMAGE_MANIFEST)
    format_args, raw_args = _remote_commands(IMMUTABLE_IMAGE)
    cli = ScriptedDocker([
        (format_args, _json_result(format_args, descriptor)),
        (raw_args, _completed(raw_args, stdout=raw)),
    ])

    with pytest.raises(_mod.PublishError, match="config media type"):
        _mod.inspect_remote_image(IMMUTABLE_IMAGE, cli)

    cli.assert_finished()


def test_existing_identical_tag_is_an_idempotent_no_push():
    raw, descriptor = _image_manifest(LOCAL_CONFIG)
    format_args, raw_args = _remote_commands(IMMUTABLE_IMAGE)
    tag_args = ["docker", "image", "tag", LOCAL_IMAGE, IMMUTABLE_IMAGE]
    cli = ScriptedDocker([
        (_local_inspect_args(), _local_inspect_result()),
        (tag_args, _completed(tag_args)),
        (format_args, _json_result(format_args, descriptor)),
        (raw_args, _completed(raw_args, stdout=raw)),
    ])

    result = _mod.publish_immutable_image(
        LOCAL_IMAGE,
        IMMUTABLE_IMAGE,
        REVISION,
        SOURCE,
        cli,
        sleep=lambda _: None,
    )

    assert result.published is False
    assert result.digest == descriptor["digest"]
    assert result.image == (
        "ghcr.io/idearoom/hermes-agent-dorvis@" + str(descriptor["digest"])
    )
    assert not any(call[:3] == ["docker", "image", "push"] for call in cli.calls)
    cli.assert_finished()


def test_existing_conflicting_tag_fails_before_any_push():
    raw, descriptor = _image_manifest(OTHER_CONFIG)
    format_args, raw_args = _remote_commands(IMMUTABLE_IMAGE)
    tag_args = ["docker", "image", "tag", LOCAL_IMAGE, IMMUTABLE_IMAGE]
    cli = ScriptedDocker([
        (_local_inspect_args(), _local_inspect_result()),
        (tag_args, _completed(tag_args)),
        (format_args, _json_result(format_args, descriptor)),
        (raw_args, _completed(raw_args, stdout=raw)),
    ])

    with pytest.raises(_mod.PublishError, match="immutable tag conflict"):
        _mod.publish_immutable_image(
            LOCAL_IMAGE,
            IMMUTABLE_IMAGE,
            REVISION,
            SOURCE,
            cli,
            sleep=lambda _: None,
        )

    assert not any(call[:3] == ["docker", "image", "push"] for call in cli.calls)
    cli.assert_finished()


def test_absent_tag_pushes_the_smoked_local_image_then_attests_registry_content():
    raw, descriptor = _image_manifest(LOCAL_CONFIG)
    format_args, raw_args = _remote_commands(IMMUTABLE_IMAGE)
    tag_args = ["docker", "image", "tag", LOCAL_IMAGE, IMMUTABLE_IMAGE]
    push_args = ["docker", "image", "push", IMMUTABLE_IMAGE]
    cli = ScriptedDocker([
        (_local_inspect_args(), _local_inspect_result()),
        (tag_args, _completed(tag_args)),
        (
            format_args,
            _completed(
                format_args,
                returncode=1,
                stderr=f"ERROR: {IMMUTABLE_IMAGE}: not found",
            ),
        ),
        (push_args, _completed(push_args)),
        (format_args, _json_result(format_args, descriptor)),
        (raw_args, _completed(raw_args, stdout=raw)),
    ])

    result = _mod.publish_immutable_image(
        LOCAL_IMAGE,
        IMMUTABLE_IMAGE,
        REVISION,
        SOURCE,
        cli,
        sleep=lambda _: None,
    )

    assert result.published is True
    assert result.config_digest == LOCAL_CONFIG
    assert result.digest == descriptor["digest"]
    assert cli.calls.count(push_args) == 1
    assert not any("build" in call for call in cli.calls)
    cli.assert_finished()


def test_post_push_readback_refuses_registry_content_that_changed():
    raw, descriptor = _image_manifest(OTHER_CONFIG)
    format_args, raw_args = _remote_commands(IMMUTABLE_IMAGE)
    tag_args = ["docker", "image", "tag", LOCAL_IMAGE, IMMUTABLE_IMAGE]
    push_args = ["docker", "image", "push", IMMUTABLE_IMAGE]
    cli = ScriptedDocker([
        (_local_inspect_args(), _local_inspect_result()),
        (tag_args, _completed(tag_args)),
        (
            format_args,
            _completed(
                format_args,
                returncode=1,
                stderr=f"ERROR: {IMMUTABLE_IMAGE}: not found",
            ),
        ),
        (push_args, _completed(push_args)),
        (format_args, _json_result(format_args, descriptor)),
        (raw_args, _completed(raw_args, stdout=raw)),
    ])

    with pytest.raises(_mod.PublishError, match="does not match the smoke-tested"):
        _mod.publish_immutable_image(
            LOCAL_IMAGE,
            IMMUTABLE_IMAGE,
            REVISION,
            SOURCE,
            cli,
            sleep=lambda _: None,
        )

    cli.assert_finished()


def test_remote_descriptor_digest_must_hash_the_exact_manifest_bytes():
    raw, descriptor = _image_manifest(LOCAL_CONFIG)
    descriptor["digest"] = "sha256:" + "f" * 64
    format_args, raw_args = _remote_commands(IMMUTABLE_IMAGE)
    cli = ScriptedDocker([
        (format_args, _json_result(format_args, descriptor)),
        (raw_args, _completed(raw_args, stdout=raw)),
    ])

    with pytest.raises(_mod.PublishError, match="digest mismatch"):
        _mod.inspect_remote_image(IMMUTABLE_IMAGE, cli)

    cli.assert_finished()


def test_evidence_records_the_registry_digest_and_not_the_local_image_id(tmp_path):
    result = _mod.PublishResult(
        repository="ghcr.io/idearoom/hermes-agent-dorvis",
        tag=IMMUTABLE_IMAGE,
        digest="sha256:" + "8" * 64,
        config_digest=LOCAL_CONFIG,
        revision=REVISION,
        published=True,
    )
    github_output = tmp_path / "github-output"
    summary = tmp_path / "summary.md"
    artifact = tmp_path / "ghcr-image.json"

    _mod.write_evidence(result, github_output, summary, artifact)

    evidence = json.loads(artifact.read_text(encoding="utf-8"))
    assert evidence["image"] == (
        "ghcr.io/idearoom/hermes-agent-dorvis@sha256:" + "8" * 64
    )
    assert evidence["digest"] != evidence["config_digest"]
    outputs = github_output.read_text(encoding="utf-8")
    assert f"digest={evidence['digest']}" in outputs
    assert f"image={evidence['image']}" in outputs
    assert evidence["image"] in summary.read_text(encoding="utf-8")


def test_workflow_builds_once_smokes_that_image_and_exports_publish_evidence():
    yaml = pytest.importorskip("yaml")
    workflow = yaml.safe_load(
        (_REPO / ".github/workflows/idearoom-ghcr-publish.yml").read_text(
            encoding="utf-8"
        )
    )
    validation = workflow["jobs"]["validate-pr"]
    assert validation["permissions"] == {"contents": "read"}
    assert "pull_request" in validation["if"]
    for checkout in (
        step
        for step in validation["steps"]
        if str(step.get("uses", "")).startswith("actions/checkout@")
    ):
        assert checkout["with"]["persist-credentials"] is False

    job = workflow["jobs"]["build-publish"]
    assert job["permissions"] == {"contents": "read", "packages": "write"}
    assert "github.event_name == 'push'" in job["if"]
    assert "github.ref == 'refs/heads/main'" in job["if"]
    assert "!=" not in job["if"]
    steps = job["steps"]
    builds = [
        step
        for step in steps
        if str(step.get("uses", "")).startswith("docker/build-push-action@")
    ]

    assert len(builds) == 1, "the smoke image and pushed image must be one build"
    build = builds[0]["with"]
    assert build["load"] is True
    assert build.get("push") is not True
    assert build["platforms"] == "linux/arm64"
    assert "org.opencontainers.image.revision=" in build["labels"]
    assert "HERMES_GIT_SHA=" in build["build-args"]

    smoke = next(step for step in steps if step.get("name") == "Run docker smoke tests")
    assert smoke["env"]["HERMES_TEST_IMAGE"] == builds[0]["with"]["tags"]

    publish = next(step for step in steps if step.get("id") == "publish")
    assert "../.ci-control/scripts/ci/publish_immutable_image.py" in publish["run"]
    assert job["outputs"]["digest"] == "${{ steps.publish.outputs.digest }}"
    assert job["outputs"]["image"] == "${{ steps.publish.outputs.image }}"

    checkouts = [
        step
        for step in steps
        if str(step.get("uses", "")).startswith("actions/checkout@")
    ]
    assert len(checkouts) == 2
    assert {step["with"]["path"] for step in checkouts} == {
        ".ci-control",
        "candidate",
    }
    assert all(step["with"]["persist-credentials"] is False for step in checkouts)
    assert all(step["with"]["ref"] == "${{ github.sha }}" for step in checkouts)

    identity = next(step for step in steps if step.get("id") == "image")
    assert "controller_revision" in identity["run"]
    assert "does not match event SHA" in identity["run"]

    login_index = next(
        index
        for index, step in enumerate(steps)
        if str(step.get("uses", "")).startswith("docker/login-action@")
    )
    assert steps.index(publish) > login_index
    for step in steps[login_index + 1 :]:
        command = str(step.get("run", ""))
        assert "candidate/scripts/" not in command
        assert "scripts/run_tests.sh" not in command

    artifact = next(
        step
        for step in steps
        if str(step.get("uses", "")).startswith("actions/upload-artifact@")
    )
    assert artifact["with"]["path"] == "candidate/ghcr-image.json"
    assert artifact["with"]["if-no-files-found"] == "error"
    workflow_source = (_REPO / ".github/workflows/idearoom-ghcr-publish.yml").read_text(
        encoding="utf-8"
    )
    assert "workflow_dispatch" not in workflow_source
    assert ":latest" not in workflow_source
    assert "imagetools create" not in workflow_source


def test_both_debian_stages_share_one_immutable_official_image_digest():
    dockerfile = (_REPO / "Dockerfile").read_text(encoding="utf-8")
    bases = re.findall(
        r"^FROM (debian:13\.4@sha256:[0-9a-f]{64})(?:\s|$)",
        dockerfile,
        re.MULTILINE,
    )

    assert len(bases) == 2
    assert len(set(bases)) == 1, "build and runtime must use the same Debian root"
