"""Helpers for replacing oversized tool output with a file handoff.

Terminal-like tools need to keep model context bounded without corrupting
structured stdout. These helpers write the full redacted text to a local file
and return a small JSON reference that callers can parse deterministically.
"""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
from typing import Any


DEFAULT_HANDOFF_DIR = "/tmp/hermes-large-outputs"
HANDOFF_TYPE = "hermes_large_output_handoff"

_SAFE_COMPONENT_RE = re.compile(r"[^A-Za-z0-9_.-]+")


def sanitize_output_text(text: str) -> str:
    """Apply the normal terminal-output cleanup before persisting output."""
    from agent.redact import redact_sensitive_text
    from tools.ansi_strip import strip_ansi

    return redact_sensitive_text(strip_ansi(text or ""))


def maybe_transform_large_output(
    output: str,
    *,
    max_inline_chars: int,
    task_id: str = "",
    producer: str,
    source: str = "stdout",
) -> str | None:
    """Return a replacement string when ``output`` would be truncated.

    If cleanup shrinks the output below the cap, the cleaned output is returned
    so callers avoid running the older head/tail truncation against raw text.
    """
    cleaned = sanitize_output_text(output)
    if len(cleaned) > max_inline_chars:
        return write_large_output_handoff(
            cleaned,
            max_inline_chars=max_inline_chars,
            task_id=task_id,
            producer=producer,
            source=source,
        )
    if len(output or "") > max_inline_chars:
        return cleaned
    return None


def write_large_output_handoff(
    output: str,
    *,
    max_inline_chars: int,
    task_id: str = "",
    producer: str,
    source: str = "stdout",
) -> str:
    """Write sanitized ``output`` to disk and return a parseable JSON reference."""
    text = sanitize_output_text(output or "")
    encoded = text.encode("utf-8", errors="replace")
    digest = hashlib.sha256(encoded).hexdigest()
    directory = _handoff_dir()
    directory.mkdir(mode=0o700, parents=True, exist_ok=True)

    safe_task = _safe_component(task_id or "default")
    safe_producer = _safe_component(producer)
    path = directory / f"{safe_task}-{safe_producer}-{digest[:16]}.txt"
    _write_bytes_atomic(path, encoded)

    record: dict[str, Any] = {
        "type": HANDOFF_TYPE,
        "producer": producer,
        "source": source,
        "truncated": True,
        "inline_output_replaced": True,
        "full_output_path": str(path),
        "total_chars": len(text),
        "total_bytes": len(encoded),
        "max_inline_chars": max_inline_chars,
        "sha256": digest,
        "encoding": "utf-8",
        "sanitized": True,
        "message": (
            "Output exceeded the inline tool cap and was written to "
            "full_output_path. Read that file before parsing the command "
            "or script output."
        ),
    }
    _append_manifest(directory, record)
    return json.dumps(record, ensure_ascii=False, sort_keys=True)


def _handoff_dir() -> Path:
    return Path(os.getenv("HERMES_LARGE_OUTPUT_DIR") or DEFAULT_HANDOFF_DIR)


def _safe_component(value: str) -> str:
    safe = _SAFE_COMPONENT_RE.sub("-", value.strip())[:80].strip(".-")
    return safe or "default"


def _write_bytes_atomic(path: Path, data: bytes) -> None:
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with open(tmp, "wb") as handle:
        handle.write(data)
    os.chmod(tmp, 0o600)
    os.replace(tmp, path)
    try:
        os.chmod(path, 0o600)
    except OSError:
        pass


def _append_manifest(directory: Path, record: dict[str, Any]) -> None:
    entry = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        **record,
    }
    manifest = directory / "manifest.jsonl"
    with open(manifest, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry, ensure_ascii=False, sort_keys=True) + "\n")
    try:
        os.chmod(manifest, 0o600)
    except OSError:
        pass
