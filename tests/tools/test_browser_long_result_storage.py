"""Behavior contracts for deterministic browser snapshot storage."""

import json
from pathlib import Path

import pytest

from agent.redact import redact_sensitive_text
from tools import browser_camofox, browser_tool


@pytest.fixture(autouse=True)
def isolated_browser_storage(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    monkeypatch.setattr("agent.redact._REDACT_ENABLED", False)
    return tmp_path / ".hermes"


def _oversized_snapshot(secret: str) -> str:
    return (
        "heading: Account settings\n"
        f"text: API key {secret}\n"
        "button [ref=e1]: Save\n"
        + "\n".join(f"text: ordinary row {index}" for index in range(300))
    )


def test_stored_snapshot_force_redacts_caps_and_reuses_identical_path():
    secret = "sk-" + "S" * 32
    snapshot = (
        "heading: Storage fidelity\n"
        f"text: credential {secret}\n"
        "text: preserved prefix\n"
        + "x"
        * (browser_tool.MAX_STORED_SNAPSHOT_CHARS + 512)
    )
    expected = redact_sensitive_text(snapshot, force=True)
    marker = (
        "\n\n[... stored copy truncated at "
        f"{browser_tool.MAX_STORED_SNAPSHOT_CHARS:,} chars of "
        f"{len(expected):,} ...]"
    )

    first_path = browser_tool._store_full_snapshot(snapshot)
    second_path = browser_tool._store_full_snapshot(snapshot)

    assert first_path is not None
    assert second_path is not None
    assert Path(first_path).is_absolute()
    assert second_path == first_path
    stored = Path(second_path).read_text(encoding="utf-8")
    assert stored == expected[: browser_tool.MAX_STORED_SNAPSHOT_CHARS] + marker
    assert secret not in stored
    assert "heading: Storage fidelity" in stored
    assert "text: preserved prefix" in stored


def test_main_browser_returned_snapshot_force_redacts_when_globally_disabled(
    monkeypatch, isolated_browser_storage
):
    secret = "ghp_" + "M" * 32
    snapshot = _oversized_snapshot(secret)
    monkeypatch.setattr(browser_tool, "_is_camofox_mode", lambda: False)
    monkeypatch.setattr(browser_tool, "_is_local_backend", lambda: True)
    monkeypatch.setattr(browser_tool, "_last_session_key", lambda task_id: task_id)
    monkeypatch.setattr(browser_tool, "get_browser_snapshot_threshold", lambda: 1000)
    monkeypatch.setattr(
        browser_tool,
        "_run_browser_command",
        lambda *_args, **_kwargs: {
            "success": True,
            "data": {"snapshot": snapshot, "refs": {"e1": {}}},
        },
    )

    tool_result = browser_tool.browser_snapshot(task_id="main-redaction")
    result = json.loads(tool_result)

    assert result["success"] is True
    assert "more lines truncated" in result["snapshot"]
    assert secret not in result["snapshot"]
    assert "Account settings" in result["snapshot"]

    from tools.tool_result_storage import maybe_persist_tool_result

    handed_off = maybe_persist_tool_result(
        content=tool_result,
        tool_name="browser_snapshot",
        tool_use_id="browser-large-handoff",
    )
    assert handed_off == tool_result
    assert not (isolated_browser_storage / "cache" / "spillover").exists()


def test_camofox_returned_snapshot_force_redacts_when_globally_disabled(
    monkeypatch,
):
    secret = "ghp_" + "C" * 32
    snapshot = _oversized_snapshot(secret)
    session = {"tab_id": "tab-1", "user_id": "user-1"}
    monkeypatch.setattr(browser_camofox, "_get_session", lambda _task_id: session)
    monkeypatch.setattr(
        browser_camofox,
        "_camofox_private_page_block",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        browser_camofox,
        "_get",
        lambda *_args, **_kwargs: {"snapshot": snapshot, "refsCount": 1},
    )
    monkeypatch.setattr(browser_tool, "get_browser_snapshot_threshold", lambda: 1000)

    result = json.loads(browser_camofox.camofox_snapshot(task_id="camofox-redaction"))

    assert result["success"] is True
    assert "more lines truncated" in result["snapshot"]
    assert secret not in result["snapshot"]
    assert "Account settings" in result["snapshot"]


def test_camofox_navigation_snapshot_force_redacts_when_globally_disabled(
    monkeypatch,
):
    secret = "ghp_" + "N" * 32
    snapshot = _oversized_snapshot(secret)
    session = {"tab_id": "tab-1", "user_id": "user-1"}
    monkeypatch.setattr(
        browser_camofox,
        "_rewrite_loopback_url_for_camofox",
        lambda url: (url, None),
    )
    monkeypatch.setattr(browser_camofox, "_get_session", lambda _task_id: session)
    monkeypatch.setattr(
        browser_camofox,
        "_post",
        lambda *_args, **_kwargs: {
            "url": "https://example.com",
            "title": "Example",
        },
    )
    monkeypatch.setattr(
        browser_camofox,
        "_get",
        lambda *_args, **_kwargs: {"snapshot": snapshot, "refsCount": 1},
    )
    monkeypatch.setattr(browser_camofox, "get_vnc_url", lambda: None)
    monkeypatch.setattr(browser_tool, "get_browser_snapshot_threshold", lambda: 1000)

    result = json.loads(
        browser_camofox.camofox_navigate(
            "https://example.com",
            task_id="camofox-navigation-redaction",
        )
    )

    assert result["success"] is True
    assert "more lines truncated" in result["snapshot"]
    assert secret not in result["snapshot"]
    assert "Account settings" in result["snapshot"]
