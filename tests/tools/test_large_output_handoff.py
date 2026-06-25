import json
import os
from pathlib import Path
import tempfile

import tools.large_output_handoff as large_output_handoff
from tools.large_output_handoff import (
    HANDOFF_TYPE,
    maybe_transform_large_output,
    sanitize_output_text,
    write_large_output_handoff,
)


def test_write_large_output_handoff_returns_parseable_reference(monkeypatch, tmp_path):
    handoff_dir = tmp_path / "handoffs"
    monkeypatch.setenv("HERMES_LARGE_OUTPUT_DIR", str(handoff_dir))

    payload = [{"row": i, "value": "x" * 80} for i in range(100)]
    output = json.dumps(payload)
    reference = json.loads(
        write_large_output_handoff(
            output,
            max_inline_chars=1000,
            task_id="task/example",
            producer="terminal",
            source="stdout",
        )
    )

    assert reference["type"] == HANDOFF_TYPE
    assert reference["truncated"] is True
    assert reference["total_chars"] == len(output)
    assert reference["max_inline_chars"] == 1000
    assert "OUTPUT TRUNCATED" not in json.dumps(reference)

    full_output_path = reference["full_output_path"]
    with open(full_output_path, encoding="utf-8") as handle:
        assert json.loads(handle.read()) == payload

    manifest = handoff_dir / "manifest.jsonl"
    assert manifest.exists()
    with open(manifest, encoding="utf-8") as handle:
        manifest_entry = json.loads(handle.readline())
    assert manifest_entry["full_output_path"] == full_output_path


def test_maybe_transform_large_output_redacts_before_handoff(monkeypatch, tmp_path):
    handoff_dir = tmp_path / "handoffs"
    monkeypatch.setenv("HERMES_LARGE_OUTPUT_DIR", str(handoff_dir))
    monkeypatch.setattr("agent.redact._REDACT_ENABLED", True)

    secret = "sk-proj-abc123def456ghi789jkl012mno345"
    output = f"\x1b[31mOPENAI_API_KEY={secret}\x1b[0m\n" + ("x" * 2000)
    transformed = maybe_transform_large_output(
        output,
        max_inline_chars=100,
        task_id="secret-task",
        producer="terminal",
    )
    reference = json.loads(transformed)

    with open(reference["full_output_path"], encoding="utf-8") as handle:
        persisted = handle.read()
    assert "\x1b" not in persisted
    assert secret not in persisted
    assert "OPENAI_API_KEY=" in persisted
    assert "***" in persisted


def test_write_large_output_handoff_falls_back_when_first_dir_is_unwritable(
    monkeypatch,
    tmp_path,
):
    primary = tmp_path / "primary"
    fallback_root = tmp_path / "tmp"
    fallback_dir = fallback_root / f"hermes-large-outputs-{os.getuid()}"
    real_ensure = large_output_handoff._ensure_writable_dir

    def fail_primary_once(directory):
        if Path(directory) == primary:
            raise PermissionError("primary handoff dir is not writable")
        return real_ensure(directory)

    monkeypatch.setenv("HERMES_LARGE_OUTPUT_DIR", str(primary))
    monkeypatch.setattr(tempfile, "gettempdir", lambda: str(fallback_root))
    monkeypatch.setattr(
        large_output_handoff,
        "_ensure_writable_dir",
        fail_primary_once,
    )

    reference = json.loads(
        write_large_output_handoff(
            json.dumps([{"row": 1}]),
            max_inline_chars=1,
            task_id="fallback-test",
            producer="execute_code",
        )
    )

    full_output_path = Path(reference["full_output_path"])
    assert full_output_path.parent == fallback_dir
    assert json.loads(full_output_path.read_text(encoding="utf-8")) == [{"row": 1}]


def test_sanitize_output_text_keeps_small_clean_text_unchanged():
    assert sanitize_output_text("plain output") == "plain output"
