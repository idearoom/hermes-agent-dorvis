#!/usr/bin/env python3
"""Publish one smoke-tested Docker image under a conflict-safe immutable tag.

Docker's local image ID is the digest of the image config JSON.  A registry
digest is instead the digest of a manifest (or index) that points at that
config and the image layers.  Treating those two values as interchangeable can
make a publisher claim it tested content that it rebuilt or replaced later.

This publisher never builds.  It accepts the already-built, already-smoked
local image, checks the OCI labels and platform, and then:

* skips an existing ``sha-<commit>`` tag only when its linux/arm64 manifest
  points at the exact local config digest;
* refuses to overwrite any conflicting or unverifiable remote tag;
* pushes the local image once when the tag is definitely absent;
* reads the registry back and emits its manifest digest for downstream pins.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

OCI_IMAGE_INDEX = "application/vnd.oci.image.index.v1+json"
DOCKER_MANIFEST_LIST = "application/vnd.docker.distribution.manifest.list.v2+json"
OCI_IMAGE_MANIFEST = "application/vnd.oci.image.manifest.v1+json"
DOCKER_IMAGE_MANIFEST = "application/vnd.docker.distribution.manifest.v2+json"
OCI_IMAGE_CONFIG = "application/vnd.oci.image.config.v1+json"
DOCKER_IMAGE_CONFIG = "application/vnd.docker.container.image.v1+json"

_INDEX_TYPES = {OCI_IMAGE_INDEX, DOCKER_MANIFEST_LIST}
_MANIFEST_TYPES = {OCI_IMAGE_MANIFEST, DOCKER_IMAGE_MANIFEST}
_CONFIG_TYPES = {OCI_IMAGE_CONFIG, DOCKER_IMAGE_CONFIG}
_DIGEST = re.compile(r"^sha256:[0-9a-f]{64}$")
_REVISION = re.compile(r"^(?:[0-9a-f]{40}|[0-9a-f]{64})$")
_REVISION_LABEL = "org.opencontainers.image.revision"
_SOURCE_LABEL = "org.opencontainers.image.source"
_POST_PUSH_ATTEMPTS = 4


class PublishError(RuntimeError):
    """A fail-closed image identity, registry, or publishing failure."""


@dataclass(frozen=True)
class LocalImage:
    reference: str
    config_digest: str
    os: str
    architecture: str
    labels: dict[str, str]


@dataclass(frozen=True)
class RemoteImage:
    reference: str
    digest: str
    config_digest: str


@dataclass(frozen=True)
class PublishResult:
    repository: str
    tag: str
    digest: str
    config_digest: str
    revision: str
    published: bool

    @property
    def image(self) -> str:
        return f"{self.repository}@{self.digest}"


class DockerCLI:
    """Production command boundary; tests provide a deterministic replacement."""

    def run(
        self,
        args: list[str],
        *,
        capture: bool = True,
        check: bool = True,
    ) -> subprocess.CompletedProcess[str]:
        try:
            result = subprocess.run(
                args,
                text=True,
                stdout=subprocess.PIPE if capture else None,
                stderr=subprocess.PIPE if capture else None,
                check=False,
            )
        except OSError as exc:
            raise PublishError(f"could not execute {args[0]!r}: {exc}") from exc
        if check and result.returncode:
            raise PublishError(_command_failure(args, result))
        return result


def _command_failure(args: list[str], result: subprocess.CompletedProcess[str]) -> str:
    detail = "\n".join(
        part.strip()
        for part in (result.stdout or "", result.stderr or "")
        if part.strip()
    )
    command = " ".join(args)
    return f"command failed ({result.returncode}): {command}" + (
        f"\n{detail}" if detail else ""
    )


def _json_object(raw: str, description: str) -> dict[str, Any]:
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise PublishError(f"{description} was not valid JSON") from exc
    if not isinstance(value, dict):
        raise PublishError(f"{description} was not a JSON object")
    return value


def _sha256_digest(raw: str) -> str:
    return "sha256:" + hashlib.sha256(raw.encode()).hexdigest()


def _require_digest(value: object, description: str) -> str:
    if not isinstance(value, str) or not _DIGEST.fullmatch(value):
        raise PublishError(f"{description} did not contain a sha256 digest")
    return value


def _repository(reference: str) -> str:
    """Remove a tag or digest without mistaking a registry port for a tag."""
    without_digest = reference.split("@", 1)[0]
    last_colon = without_digest.rfind(":")
    last_slash = without_digest.rfind("/")
    if last_colon > last_slash:
        without_digest = without_digest[:last_colon]
    if "/" not in without_digest:
        raise PublishError(f"image reference is not fully qualified: {reference}")
    return without_digest


def _verify_manifest_bytes(
    raw: str,
    expected_digest: str,
    expected_size: object,
    description: str,
) -> dict[str, Any]:
    actual_digest = _sha256_digest(raw)
    if actual_digest != expected_digest:
        raise PublishError(
            f"{description} digest mismatch: descriptor {expected_digest}, "
            f"content {actual_digest}"
        )
    if not isinstance(expected_size, int) or expected_size != len(raw.encode()):
        raise PublishError(f"{description} size did not match its descriptor")
    return _json_object(raw, description)


def inspect_local_image(reference: str, cli: Any) -> LocalImage:
    args = ["docker", "image", "inspect", reference]
    result = cli.run(args, capture=True, check=False)
    if result.returncode:
        raise PublishError(_command_failure(args, result))
    try:
        payload = json.loads(result.stdout or "")
    except json.JSONDecodeError as exc:
        raise PublishError("docker image inspect did not return JSON") from exc
    if (
        not isinstance(payload, list)
        or len(payload) != 1
        or not isinstance(payload[0], dict)
    ):
        raise PublishError("docker image inspect did not resolve exactly one image")

    data = payload[0]
    config_digest = _require_digest(data.get("Id"), "local image")
    config = data.get("Config")
    raw_labels = config.get("Labels") if isinstance(config, dict) else None
    labels = raw_labels if isinstance(raw_labels, dict) else {}
    if not all(
        isinstance(key, str) and isinstance(value, str) for key, value in labels.items()
    ):
        raise PublishError("local image labels were malformed")
    return LocalImage(
        reference=reference,
        config_digest=config_digest,
        os=str(data.get("Os", "")),
        architecture=str(data.get("Architecture", "")),
        labels=dict(labels),
    )


def _definitely_absent(reference: str, output: str) -> bool:
    """Recognize only registry responses that identify this exact ref as absent."""
    lowered = output.lower()
    if "manifest unknown" in lowered and reference.lower() in lowered:
        return True
    return (
        re.search(rf"{re.escape(reference)}:\s*not found(?:\s|$)", output, re.I)
        is not None
    )


def _raw_manifest(reference: str, cli: Any) -> str:
    args = ["docker", "buildx", "imagetools", "inspect", reference, "--raw"]
    result = cli.run(args, capture=True, check=False)
    if result.returncode:
        raise PublishError(_command_failure(args, result))
    if not isinstance(result.stdout, str) or not result.stdout:
        raise PublishError(f"registry returned an empty manifest for {reference}")
    return result.stdout


def _manifest_config_digest(manifest: dict[str, Any], description: str) -> str:
    media_type = manifest.get("mediaType")
    if media_type not in _MANIFEST_TYPES:
        raise PublishError(f"{description} was not an OCI/Docker image manifest")
    config = manifest.get("config")
    if not isinstance(config, dict):
        raise PublishError(f"{description} did not contain an image config")
    if config.get("mediaType") not in _CONFIG_TYPES:
        raise PublishError(f"{description} had an unsupported config media type")
    return _require_digest(config.get("digest"), f"{description} config")


def inspect_remote_image(reference: str, cli: Any) -> RemoteImage | None:
    """Resolve one registry reference to its top digest and arm64 config digest."""
    format_args = [
        "docker",
        "buildx",
        "imagetools",
        "inspect",
        reference,
        "--format",
        "{{json .Manifest}}",
    ]
    described = cli.run(format_args, capture=True, check=False)
    if described.returncode:
        output = "\n".join((described.stdout or "", described.stderr or ""))
        if _definitely_absent(reference, output):
            return None
        raise PublishError(
            "could not determine whether the immutable registry tag exists; "
            "refusing to push\n" + _command_failure(format_args, described)
        )

    descriptor = _json_object(
        described.stdout or "", f"registry descriptor for {reference}"
    )
    top_digest = _require_digest(descriptor.get("digest"), "registry descriptor")
    raw = _raw_manifest(reference, cli)
    manifest = _verify_manifest_bytes(
        raw,
        top_digest,
        descriptor.get("size"),
        f"registry manifest for {reference}",
    )
    media_type = manifest.get("mediaType")
    if descriptor.get("mediaType") != media_type:
        raise PublishError("registry descriptor and manifest media types disagreed")

    if media_type in _MANIFEST_TYPES:
        config_digest = _manifest_config_digest(manifest, reference)
    elif media_type in _INDEX_TYPES:
        entries = manifest.get("manifests")
        if not isinstance(entries, list):
            raise PublishError(f"registry index for {reference} had no manifests")
        arm64_entries = []
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            platform = entry.get("platform")
            annotations = entry.get("annotations")
            if (
                isinstance(platform, dict)
                and platform.get("os") == "linux"
                and platform.get("architecture") == "arm64"
                and not (
                    isinstance(annotations, dict)
                    and annotations.get("vnd.docker.reference.type")
                    == "attestation-manifest"
                )
            ):
                arm64_entries.append(entry)
        if len(arm64_entries) != 1:
            raise PublishError(
                f"registry index for {reference} resolved {len(arm64_entries)} "
                "linux/arm64 image manifests"
            )
        child = arm64_entries[0]
        child_media_type = child.get("mediaType")
        if child_media_type not in _MANIFEST_TYPES:
            raise PublishError(
                f"arm64 descriptor media type for {reference} was not an "
                "OCI/Docker image manifest"
            )
        child_digest = _require_digest(child.get("digest"), "arm64 descriptor")
        child_reference = f"{_repository(reference)}@{child_digest}"
        child_raw = _raw_manifest(child_reference, cli)
        child_manifest = _verify_manifest_bytes(
            child_raw,
            child_digest,
            child.get("size"),
            f"arm64 manifest for {reference}",
        )
        if child_manifest.get("mediaType") != child_media_type:
            raise PublishError(
                f"arm64 descriptor and manifest media types disagreed for {reference}"
            )
        config_digest = _manifest_config_digest(child_manifest, child_reference)
    else:
        raise PublishError(
            f"unsupported registry media type for {reference}: {media_type}"
        )

    return RemoteImage(reference, top_digest, config_digest)


def _validate_identity(
    local: LocalImage,
    immutable_reference: str,
    expected_revision: str,
    expected_source: str,
) -> None:
    if not _REVISION.fullmatch(expected_revision):
        raise PublishError("expected revision must be a full lowercase Git SHA")
    expected_reference = f"{_repository(immutable_reference)}:sha-{expected_revision}"
    if immutable_reference != expected_reference:
        raise PublishError(
            f"immutable tag must be sha-{expected_revision}: {immutable_reference}"
        )
    if local.os != "linux" or local.architecture != "arm64":
        raise PublishError(
            f"local image platform is {local.os}/{local.architecture}, expected linux/arm64"
        )
    if local.labels.get(_REVISION_LABEL) != expected_revision:
        raise PublishError(
            "local image revision label does not match the target commit"
        )
    if local.labels.get(_SOURCE_LABEL) != expected_source:
        raise PublishError(
            "local image source label does not match the fork repository"
        )


def _result(
    remote: RemoteImage,
    immutable_reference: str,
    revision: str,
    published: bool,
) -> PublishResult:
    return PublishResult(
        repository=_repository(immutable_reference),
        tag=immutable_reference,
        digest=remote.digest,
        config_digest=remote.config_digest,
        revision=revision,
        published=published,
    )


def publish_immutable_image(
    local_reference: str,
    immutable_reference: str,
    expected_revision: str,
    expected_source: str,
    cli: Any,
    *,
    sleep: Callable[[float], None] = time.sleep,
) -> PublishResult:
    """Publish the local image once, or prove an existing tag is identical."""
    local = inspect_local_image(local_reference, cli)
    _validate_identity(local, immutable_reference, expected_revision, expected_source)

    tag_args = ["docker", "image", "tag", local_reference, immutable_reference]
    cli.run(tag_args, capture=False, check=True)

    remote = inspect_remote_image(immutable_reference, cli)
    if remote is not None:
        if remote.config_digest != local.config_digest:
            raise PublishError(
                "immutable tag conflict: remote config "
                f"{remote.config_digest} does not match smoke-tested local config "
                f"{local.config_digest}; refusing to overwrite {immutable_reference}"
            )
        return _result(remote, immutable_reference, expected_revision, False)

    push_args = ["docker", "image", "push", immutable_reference]
    cli.run(push_args, capture=False, check=True)

    last_error: PublishError | None = None
    remote = None
    for attempt in range(_POST_PUSH_ATTEMPTS):
        try:
            remote = inspect_remote_image(immutable_reference, cli)
        except PublishError as exc:
            last_error = exc
        if remote is not None:
            break
        if attempt + 1 < _POST_PUSH_ATTEMPTS:
            sleep(float(2**attempt))
    if remote is None:
        detail = f": {last_error}" if last_error is not None else ""
        raise PublishError(
            f"pushed {immutable_reference}, but could not read it back{detail}"
        )
    if remote.config_digest != local.config_digest:
        raise PublishError(
            "pushed registry content does not match the smoke-tested local image: "
            f"remote config {remote.config_digest}, local config {local.config_digest}"
        )
    return _result(remote, immutable_reference, expected_revision, True)


def write_evidence(
    result: PublishResult,
    github_output: Path | None,
    summary_file: Path | None,
    artifact_file: Path,
) -> None:
    """Persist machine-readable and reviewer-readable registry identity evidence."""
    evidence = {
        "schema_version": 1,
        **asdict(result),
        "image": result.image,
    }
    artifact_file.parent.mkdir(parents=True, exist_ok=True)
    temporary = artifact_file.with_suffix(artifact_file.suffix + ".tmp")
    temporary.write_text(
        json.dumps(evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(artifact_file)

    if github_output is not None:
        with github_output.open("a", encoding="utf-8") as output:
            output.write(
                f"digest={result.digest}\n"
                f"image={result.image}\n"
                f"tag={result.tag}\n"
                f"config_digest={result.config_digest}\n"
                f"published={str(result.published).lower()}\n"
            )
    if summary_file is not None:
        disposition = "pushed" if result.published else "already existed identically"
        with summary_file.open("a", encoding="utf-8") as summary:
            summary.write(
                "## GHCR image identity\n\n"
                f"- Immutable tag: `{result.tag}`\n"
                f"- Digest pin: `{result.image}`\n"
                f"- Local/remote config digest: `{result.config_digest}`\n"
                f"- Result: {disposition}\n"
            )


def _path_from_env(name: str) -> Path | None:
    value = os.environ.get(name)
    return Path(value) if value else None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--local-image", required=True)
    parser.add_argument("--immutable-image", required=True)
    parser.add_argument("--expected-revision", required=True)
    parser.add_argument("--expected-source", required=True)
    parser.add_argument("--artifact-file", type=Path, default=Path("ghcr-image.json"))
    args = parser.parse_args(argv)

    try:
        result = publish_immutable_image(
            args.local_image,
            args.immutable_image,
            args.expected_revision,
            args.expected_source,
            DockerCLI(),
        )
        write_evidence(
            result,
            _path_from_env("GITHUB_OUTPUT"),
            _path_from_env("GITHUB_STEP_SUMMARY"),
            args.artifact_file,
        )
    except PublishError as exc:
        print(f"::error::{exc}", file=sys.stderr)
        return 1

    print(json.dumps({"image": result.image, "published": result.published}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
