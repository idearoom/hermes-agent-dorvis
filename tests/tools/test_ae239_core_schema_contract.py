"""AE-239 contracts for Dorvis's configured, schema-dieted core tools."""

from __future__ import annotations

import json

import pytest

from tools import (
    browser_use_cli,
    clarify_tool,
    cronjob_tools,
    session_search_tool,
    skill_manager_tool,
    terminal_tool,
)
from tools.registry import registry


_REGISTERED_SCHEMAS = {
    "browser_exec": browser_use_cli.BROWSER_EXEC_SCHEMA,
    "terminal": terminal_tool.TERMINAL_SCHEMA,
    "clarify": clarify_tool.CLARIFY_SCHEMA,
    "skill_manage": skill_manager_tool.SKILL_MANAGE_SCHEMA,
    "session_search": session_search_tool.SESSION_SEARCH_SCHEMA,
    "cronjob": cronjob_tools.CRONJOB_SCHEMA,
}

_SCHEMA_SURFACES = {
    "browser_exec": (
        ["code"],
        {"code", "session", "timeout_s"},
    ),
    "terminal": (
        ["command"],
        {"command", "background", "timeout", "workdir", "pty", "notify"},
    ),
    "clarify": (
        ["questions"],
        {"questions"},
    ),
    "skill_manage": (
        ["action", "name"],
        {
            "action",
            "name",
            "content",
            "old_string",
            "new_string",
            "replace_all",
            "category",
            "file_path",
            "file_content",
        },
    ),
    "session_search": (
        [],
        {
            "query",
            "limit",
            "sort",
            "detail",
            "session_id",
            "around_message_id",
            "window",
            "role_filter",
            "profile",
        },
    ),
    "cronjob": (
        ["action"],
        {
            "action",
            "job_id",
            "prompt",
            "schedule",
            "name",
            "repeat",
            "deliver",
            "skills",
            "script",
            "monitor",
            "no_agent",
            "context_from",
            "continuity",
            "enabled_toolsets",
            "workdir",
            "attach_to_session",
        },
    ),
}


@pytest.fixture
def model_facing_schemas(monkeypatch):
    """Resolve the definitions exactly where model_tools consumes them."""
    monkeypatch.setattr("tools.registry._check_fn_cached", lambda _check_fn: True)
    monkeypatch.setattr(browser_use_cli, "_real_profile_consented", lambda: False)

    definitions = registry.get_definitions(set(_SCHEMA_SURFACES), quiet=True)
    schemas = {
        definition["function"]["name"]: definition["function"]
        for definition in definitions
    }
    assert set(schemas) == set(_SCHEMA_SURFACES)
    return schemas


@pytest.mark.parametrize(
    ("tool_name", "required", "properties"),
    [
        (tool_name, *contract)
        for tool_name, contract in _SCHEMA_SURFACES.items()
    ],
)
def test_configured_tool_advertises_exact_parameter_surface(
    model_facing_schemas, tool_name, required, properties
):
    entry = registry.get_entry(tool_name)

    assert entry is not None
    assert entry.schema is _REGISTERED_SCHEMAS[tool_name]
    parameters = model_facing_schemas[tool_name]["parameters"]
    assert parameters["required"] == required
    assert set(parameters["properties"]) == properties


def test_clarify_question_rows_have_one_exact_shape():
    questions = registry.get_entry("clarify").schema["parameters"]["properties"][
        "questions"
    ]
    row = questions["items"]

    assert row["required"] == ["question"]
    assert set(row["properties"]) == {"question", "choices", "multi_select"}


def test_skill_manage_advertises_exact_action_enum():
    action = registry.get_entry("skill_manage").schema["parameters"]["properties"][
        "action"
    ]

    assert action["enum"] == [
        "create",
        "patch",
        "delete",
        "write_file",
        "remove_file",
    ]


@pytest.mark.parametrize(
    ("legacy_args", "notify_on_complete", "watch_patterns"),
    [
        ({"notify_on_complete": True}, True, None),
        ({"watch_patterns": ["READY"]}, False, ["READY"]),
    ],
)
def test_terminal_handler_keeps_legacy_notification_aliases(
    monkeypatch, legacy_args, notify_on_complete, watch_patterns
):
    calls = []
    monkeypatch.setattr(
        terminal_tool,
        "terminal_tool",
        lambda **kwargs: calls.append(kwargs) or '{"success": true}',
    )

    result = registry.get_entry("terminal").handler(
        {"command": "build", "background": True, **legacy_args},
        task_id="task-1",
        session_id="session-1",
    )

    assert json.loads(result) == {"success": True}
    assert calls[0]["notify_on_complete"] is notify_on_complete
    assert calls[0]["watch_patterns"] == watch_patterns


def test_clarify_handler_keeps_the_legacy_single_question_shape():
    calls = []

    def callback(question, choices, multi_select=False):
        calls.append((question, choices, multi_select))
        return ["Second"]

    result = registry.get_entry("clarify").handler(
        {
            "question": "Pick one or more",
            "choices": ["First", "Second"],
            "multi_select": True,
        },
        callback=callback,
    )

    assert calls == [
        (
            "Pick one or more",
            ["First (Recommended)", "Second"],
            True,
        )
    ]
    assert json.loads(result)["user_response"] == ["Second"]


def test_skill_manage_handler_keeps_unadvertised_legacy_arguments(monkeypatch):
    calls = []
    monkeypatch.setattr(
        skill_manager_tool,
        "skill_manage",
        lambda **kwargs: calls.append(kwargs) or '{"success": true}',
    )
    handler = registry.get_entry("skill_manage").handler

    handler(
        {"action": "edit", "name": "legacy", "content": "replacement"},
        task_id="task-1",
        session_id="session-1",
    )
    handler(
        {
            "action": "delete",
            "name": "absorbed",
            "absorbed_into": "umbrella",
        }
    )

    assert calls[0]["action"] == "edit"
    assert calls[0]["content"] == "replacement"
    assert calls[1]["action"] == "delete"
    assert calls[1]["absorbed_into"] == "umbrella"


@pytest.mark.parametrize(
    ("tool_args", "monitor_script", "monitor_url"),
    [
        ({"monitor": "check.py"}, "check.py", ""),
        (
            {"monitor": "https://example.test/health"},
            "",
            "https://example.test/health",
        ),
        ({"monitor_script": "legacy.py"}, "legacy.py", None),
        (
            {"monitor_url": "https://legacy.test/health"},
            None,
            "https://legacy.test/health",
        ),
        (
            {
                "monitor": "new.py",
                "monitor_script": "old.py",
                "monitor_url": "https://old.test/health",
            },
            "new.py",
            "",
        ),
    ],
)
def test_cronjob_handler_adapts_unified_and_legacy_monitor_arguments(
    monkeypatch, tool_args, monitor_script, monitor_url
):
    calls = []
    monkeypatch.setattr(
        cronjob_tools,
        "cronjob",
        lambda **kwargs: calls.append(kwargs) or '{"success": true}',
    )

    result = registry.get_entry("cronjob").handler(
        {"action": "update", **tool_args},
        task_id="task-1",
        session_id="session-1",
    )

    assert json.loads(result) == {"success": True}
    assert calls[0]["monitor_script"] == monitor_script
    assert calls[0]["monitor_url"] == monitor_url
