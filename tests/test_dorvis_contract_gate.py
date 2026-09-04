"""The release evidence guard must fail closed, not just run pytest."""

import json
from types import SimpleNamespace

import pytest

from scripts.ci import dorvis_contracts as gate


@pytest.fixture
def evidence(tmp_path):
    file = "tests/test_one.py"
    node = file + "::test_contract"
    manifest = {"files": [file], "minimum_passed": 1, "required_tests": [node]}
    report = {
        "file": file,
        "exitstatus": 0,
        "collection_errors": [],
        "collected": [node],
        "outcomes": {node: {"status": "passed"}},
    }

    def save():
        (tmp_path / gate.report_name(file)).write_text(json.dumps(report))

    save()
    return manifest, report, save, tmp_path


def test_valid_evidence(evidence):
    manifest, _, _, directory = evidence
    assert gate.verify_reports(manifest, directory)["passed"] == 1


@pytest.mark.parametrize(
    "change",
    [
        "missing",
        "wrong_file",
        "exit",
        "collection_skip",
        "empty",
        "failed",
        "skip",
        "dependency_skip",
        "xfail",
        "under_collected",
        "missing_required",
    ],
)
def test_refuses_incomplete_or_false_green_evidence(evidence, change):
    manifest, report, save, directory = evidence
    node = manifest["required_tests"][0]
    if change == "missing":
        (directory / gate.report_name(report["file"])).unlink()
    else:
        if change == "wrong_file":
            report["file"] = "tests/wrong.py"
        elif change == "exit":
            report["exitstatus"] = 1
        elif change == "collection_skip":
            report["collection_errors"] = ["missing SDK"]
        elif change == "empty":
            report["outcomes"] = {}
        elif change == "failed":
            report["outcomes"][node]["status"] = "failed"
        elif change in {"skip", "dependency_skip", "xfail"}:
            report["outcomes"][node] = {
                "status": "skipped",
                "reason": "missing SDK",
                "off_host_windows": change == "dependency_skip",
            }
        elif change == "under_collected":
            manifest["minimum_passed"] = 2
        elif change == "missing_required":
            manifest["required_tests"] = [node + "_gone"]
        save()
    with pytest.raises((ValueError, FileNotFoundError)):
        gate.verify_reports(manifest, directory)


def test_only_off_host_windows_marker_is_an_allowed_skip(evidence):
    manifest, report, save, directory = evidence
    report["outcomes"][report["file"] + "::test_windows"] = {
        "status": "skipped",
        "reason": "Windows-only test (marked windows_only); host is linux",
        "off_host_windows": True,
    }
    report["collected"].append(report["file"] + "::test_windows")
    save()
    assert gate.verify_reports(manifest, directory)["off_host_windows_skipped"] == 1


@pytest.mark.parametrize(
    "change",
    [
        "missing_file",
        "uncovered_inventory",
        "empty",
        "duplicate",
        "fixture",
        "missing_required_file",
    ],
)
def test_manifest_cannot_drop_inventory(evidence, change):
    manifest, _, _, root = evidence
    (root / "tests").mkdir()
    (root / manifest["files"][0]).touch()
    (root / "PATCHES.md").write_text("`tests/test_one.py`")
    if change == "missing_file":
        (root / manifest["files"][0]).unlink()
    elif change == "uncovered_inventory":
        (root / "PATCHES.md").write_text("`tests/test_new.py`")
    elif change == "empty":
        manifest["files"] = []
    elif change == "duplicate":
        manifest["files"] *= 2
    elif change == "fixture":
        manifest["files"] = ["tests/conftest.py"]
    elif change == "missing_required_file":
        manifest["required_tests"] = ["tests/missing.py::test_one"]
    with pytest.raises(ValueError):
        gate.validate_manifest(manifest, root)


def test_teardown_failure_cannot_be_hidden_by_success(monkeypatch):
    monkeypatch.setattr(gate, "_outcomes", {})
    node = "tests/test_one.py::test_contract"
    for when, outcome in [
        ("setup", "passed"),
        ("call", "passed"),
        ("teardown", "failed"),
    ]:
        gate.pytest_runtest_logreport(
            SimpleNamespace(
                nodeid=node,
                when=when,
                outcome=outcome,
                failed=outcome == "failed",
                skipped=False,
                longrepr="",
            )
        )
    assert gate._outcomes[node]["status"] == "failed"
    # This module is also the live reporter for this test process. Restore it
    # before pytest emits this test's own call report.
    monkeypatch.undo()
