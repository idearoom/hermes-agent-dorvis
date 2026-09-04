"""Bounded, offline Dorvis contracts through the canonical isolated runner.

Also a pytest plugin: each per-file process writes independent execution
evidence. The outer command verifies it, including tests that silently skip.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = Path(__file__).with_name("dorvis-contracts.json")
PG_FILES = {
    "tests/gateway/test_session_store_pg.py",
    "tests/gateway/test_session_store_pg_unit.py",
    "tests/gateway/test_async_delegation_pg.py",
    "tests/gateway/test_response_store_pg.py",
}


def validate_manifest(manifest, root=ROOT):
    files = manifest["files"]
    if not files or len(files) != len(set(files)):
        raise ValueError("Contract files must be nonempty and unique")
    for file in files:
        if (
            not file.startswith("tests/")
            or ".." in Path(file).parts
            or not (root / file).is_file()
        ):
            raise ValueError(f"Missing or invalid contract file: {file}")
    inventory = set(
        re.findall(r"tests/[A-Za-z0-9_./-]+\.py", (root / "PATCHES.md").read_text())
    )
    missing = inventory - set(files) - PG_FILES - {"tests/conftest.py"}
    if missing:
        raise ValueError(
            f"PATCHES.md contracts have no execution lane: {sorted(missing)}"
        )
    if manifest["minimum_passed"] < 1 or not manifest["required_tests"]:
        raise ValueError("Contract execution floor and required tests must be nonempty")
    for node in manifest["required_tests"]:
        if node.split("::")[0] not in files:
            raise ValueError(f"Required test not in the selected files: {node}")


def report_name(file):
    return hashlib.sha256(file.encode()).hexdigest() + ".json"


def verify_reports(manifest, directory):
    passed = skipped = 0
    passing_nodes = set()
    for file in manifest["files"]:
        report = json.loads((directory / report_name(file)).read_text())
        if (
            report["file"] != file
            or report["exitstatus"] != 0
            or report["collection_errors"]
        ):
            raise ValueError(f"Contract execution failed: {file}")
        if set(report["collected"]) != set(report["outcomes"]):
            raise ValueError(
                f"Some collected contracts have no execution outcome: {file}"
            )
        file_passed = 0
        for node, outcome in report["outcomes"].items():
            if not node.startswith(file + "::"):
                raise ValueError(f"Unexpected test identity: {node}")
            if outcome["status"] == "passed":
                file_passed += 1
                passing_nodes.add(node)
            elif (
                outcome["status"] == "skipped"
                and outcome.get("off_host_windows")
                and "Windows-only test (marked windows_only); host is "
                in outcome.get("reason", "")
            ):
                # Only the conftest's host marker is exempt. importorskip,
                # xfail, dependency skips, and unmarked skips fail closed.
                skipped += 1
            else:
                raise ValueError(f"Unexpected test outcome: {node}: {outcome}")
        if file_passed == 0:
            raise ValueError(f"No contracts passed in {file}")
        passed += file_passed
    if passed < manifest["minimum_passed"]:
        raise ValueError(
            f"Contract under-collection: {passed} passed, floor {manifest['minimum_passed']}"
        )
    for required in manifest["required_tests"]:
        if not any(
            n == required or n.startswith(required + "[") for n in passing_nodes
        ):
            raise ValueError(f"Required contract did not pass: {required}")
    return {
        "passed": passed,
        "off_host_windows_skipped": skipped,
        "files": len(manifest["files"]),
    }


# Pytest plugin hooks. Explicit CLI activation keeps ordinary upstream runs
# unchanged. Each process owns one file and gets a fresh outer report dir.
_outcomes = {}
_off_host_windows = set()
_collection_errors = []
_collected = []


def pytest_addoption(parser):
    parser.addoption("--dorvis-contract-report-dir")


def pytest_collection_modifyitems(items):
    for item in items:
        _collected.append(item.nodeid)
        if sys.platform != "win32" and item.get_closest_marker("windows_only"):
            _off_host_windows.add(item.nodeid)


def pytest_collectreport(report):
    if report.failed or report.skipped:
        _collection_errors.append(str(report.longrepr))


def pytest_runtest_logreport(report):
    if report.failed or report.skipped or report.when == "call":
        previous = _outcomes.get(report.nodeid)
        if previous and previous["status"] != "passed":
            return
        _outcomes[report.nodeid] = {
            "status": "xpass" if getattr(report, "wasxfail", None) else report.outcome,
            "off_host_windows": report.nodeid in _off_host_windows,
            "reason": str(report.longrepr) if report.longrepr else "",
        }


def pytest_sessionfinish(session, exitstatus):
    directory = session.config.getoption("--dorvis-contract-report-dir")
    if directory is None:
        return
    args = session.config.args
    if len(args) != 1:
        raise ValueError("Dorvis reports require one isolated test file per process")
    file = Path(args[0]).resolve().relative_to(ROOT).as_posix()
    (Path(directory) / report_name(file)).write_text(
        json.dumps({
            "file": file,
            "exitstatus": int(exitstatus),
            "outcomes": _outcomes,
            "collection_errors": _collection_errors,
            "collected": _collected,
        })
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-manifest", action="store_true")
    parser.add_argument("--summary", type=Path)
    args = parser.parse_args()
    manifest = json.loads(MANIFEST.read_text())
    validate_manifest(manifest)
    if args.check_manifest:
        print(f"Validated {len(manifest['files'])} contract files against PATCHES.md")
        return
    started = time.monotonic()
    with tempfile.TemporaryDirectory(prefix="dorvis-contracts-") as directory:
        command = [
            "scripts/run_tests.sh",
            "-j",
            "4",
            "--file-retries",
            "0",
            "--file-timeout",
            "120",
            "--files",
            ":".join(manifest["files"]),
            "--",
            "-q",
            "-rs",
            "--tb=short",
            "-p",
            "scripts.ci.dorvis_contracts",
            "--dorvis-contract-report-dir",
            directory,
        ]
        result = subprocess.run(command, cwd=ROOT, check=False, timeout=600)
        summary = verify_reports(manifest, Path(directory))
        if result.returncode:
            raise ValueError(f"Contract runner failed: exit {result.returncode}")
    summary["elapsed_seconds"] = round(time.monotonic() - started, 2)
    print(json.dumps(summary, indent=2))
    if args.summary:
        args.summary.write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
