"""Integration contracts for structured documents read through ``read_file``."""

from __future__ import annotations

import base64
import json
import os
import zipfile
from contextlib import contextmanager
from pathlib import Path
from unittest import mock

import pytest

from tools import file_state, file_tools
from tools.file_tools import clear_file_ops_cache, read_file_tool
from tools.file_operations import ReadResult


_WORD_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
_SHEET_NS = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
_OFFICE_REL_NS = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
_PACKAGE_REL_NS = "http://schemas.openxmlformats.org/package/2006/relationships"


def _write_notebook(path: Path, text: str) -> None:
    path.write_text(
        json.dumps({
            "cells": [{"cell_type": "markdown", "source": text}],
            "metadata": {},
            "nbformat": 4,
            "nbformat_minor": 5,
        }),
        encoding="utf-8",
    )


def _write_docx(path: Path, text: str) -> None:
    document = (
        f'<w:document xmlns:w="{_WORD_NS}"><w:body><w:p><w:r>'
        f"<w:t>{text}</w:t>"
        "</w:r></w:p></w:body></w:document>"
    )
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("word/document.xml", document)


def _write_xlsx(path: Path, text: str) -> None:
    workbook = (
        f'<workbook xmlns="{_SHEET_NS}" xmlns:r="{_OFFICE_REL_NS}">'
        '<sheets><sheet name="Data" sheetId="1" r:id="rId1"/></sheets>'
        "</workbook>"
    )
    relationships = (
        f'<Relationships xmlns="{_PACKAGE_REL_NS}">'
        '<Relationship Id="rId1" Target="worksheets/sheet1.xml" Type="x"/>'
        "</Relationships>"
    )
    worksheet = (
        f'<worksheet xmlns="{_SHEET_NS}"><sheetData><row r="1">'
        f'<c r="A1" t="str"><v>{text}</v></c>'
        "</row></sheetData></worksheet>"
    )
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("xl/workbook.xml", workbook)
        archive.writestr("xl/_rels/workbook.xml.rels", relationships)
        archive.writestr("xl/worksheets/sheet1.xml", worksheet)


@pytest.fixture(autouse=True)
def _reset_file_read_state():
    file_state.get_registry().clear()
    yield
    file_state.get_registry().clear()


@pytest.mark.parametrize(
    ("extension", "writer"),
    [
        ("ipynb", _write_notebook),
        ("docx", _write_docx),
        ("xlsx", _write_xlsx),
    ],
)
def test_identical_structured_reads_use_normal_tracking_contract(
    tmp_path: Path,
    extension: str,
    writer,
) -> None:
    """A native document read is deduplicated, loop-limited, and stateful."""
    path = tmp_path / f"tracked.{extension}"
    writer(path, "TRACKED_CONTENT")
    task_id = f"structured-tracking-{extension}"

    try:
        first = json.loads(read_file_tool(str(path), task_id=task_id))
        second = json.loads(read_file_tool(str(path), task_id=task_id))
        third = json.loads(read_file_tool(str(path), task_id=task_id))

        assert "TRACKED_CONTENT" in first["content"]
        assert second == {
            "status": "unchanged",
            "message": (
                "File unchanged since last read. The content from the earlier "
                "read_file result in this conversation is still current — "
                "refer to that instead of re-reading."
            ),
            "path": str(path),
            "dedup": True,
            "content_returned": False,
        }
        assert "BLOCKED" in third["error"]
        assert third["already_read"] == 3
        assert file_state.known_reads(task_id) == [str(path)]

        stat = path.stat()
        os.utime(
            path,
            ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000_000),
        )
        stale = file_state.check_stale(task_id, str(path))
        assert stale is not None
        assert "modified since you last read it" in stale
    finally:
        clear_file_ops_cache(task_id)


def test_native_reader_slot_covers_rendering_and_serialization(
    tmp_path: Path,
) -> None:
    """The allocation bound must include post-parse retained-data phases."""
    path = tmp_path / "slot-scope.docx"
    _write_docx(path, "SLOT_SCOPE_CONTENT")
    task_id = "structured-slot-scope"
    depth = 0

    @contextmanager
    def tracked_slot(_path):
        nonlocal depth
        depth += 1
        try:
            yield
        finally:
            depth -= 1

    real_page_result = file_tools._document_page_result
    real_finish_read = file_tools._finish_tracked_read

    def asserted_page_result(*args, **kwargs):
        assert depth > 0, "native reader slot released before pagination/redaction"
        return real_page_result(*args, **kwargs)

    def asserted_finish_read(*args, **kwargs):
        assert depth > 0, "native reader slot released before serialization/tracking"
        return real_finish_read(*args, **kwargs)

    try:
        with (
            mock.patch("tools.read_extract.native_extraction_slot", tracked_slot),
            mock.patch.object(
                file_tools,
                "_document_page_result",
                side_effect=asserted_page_result,
            ),
            mock.patch.object(
                file_tools,
                "_finish_tracked_read",
                side_effect=asserted_finish_read,
            ),
        ):
            result = json.loads(read_file_tool(str(path), task_id=task_id))

        assert "SLOT_SCOPE_CONTENT" in result["content"]
        assert depth == 0
    finally:
        clear_file_ops_cache(task_id)


def test_wide_structured_row_is_losslessly_retrievable_by_offset(
    tmp_path: Path,
) -> None:
    """Per-line display limits split wide extracted rows instead of dropping tails."""
    path = tmp_path / "wide-row.xlsx"
    tail = "UNIQUE_TAIL_AFTER_THE_OLD_LINE_CLAMP"
    wide_value = "A" * 4_500 + tail
    _write_xlsx(path, wide_value)
    task_id = "structured-wide-row"

    try:
        complete = json.loads(read_file_tool(str(path), task_id=task_id))

        assert tail in complete["content"]
        assert "... [truncated]" not in complete["content"]
        assert complete["total_lines"] > complete["logical_total_lines"]
        assert complete["wrapped_logical_lines"] == 1
        assert complete["truncated"] is False

        tail_pages = []
        row_parts = []
        for offset in range(1, complete["total_lines"] + 1):
            page = json.loads(
                read_file_tool(
                    str(path),
                    offset=offset,
                    limit=1,
                    task_id=task_id,
                )
            )
            content = page.get("content", "")
            assert "not retrievable" not in page.get("hint", "")
            if tail in content:
                tail_pages.append((offset, page))
            _gutter, separator, body = content.partition("|")
            if separator and (body.startswith("A") or body.startswith("↳ ")):
                row_parts.append(body.removeprefix("↳ "))

        assert "".join(row_parts) == wide_value
        assert len(tail_pages) == 1
        tail_offset, tail_page = tail_pages[0]
        assert tail_offset > 1
        assert tail_page["truncated"] is (tail_offset < complete["total_lines"])
    finally:
        clear_file_ops_cache(task_id)


def test_structured_read_redacts_secret_before_lossless_line_wrapping(
    tmp_path: Path,
) -> None:
    """A continuation boundary must not split a token before redaction."""
    path = tmp_path / "boundary-secret.xlsx"
    secret_body = "proj-abc123def456ghi789jkl012mno345"
    secret = f"sk-{secret_body}"
    # The default extracted-line cap is 2,000 characters.  Place ``sk`` at
    # the end of that window so post-wrap redaction would see neither the
    # complete prefix nor the complete token.
    _write_xlsx(path, "A" * 1_997 + " " + secret + " SAFE_TAIL")
    task_id = "structured-boundary-secret"

    try:
        result = json.loads(read_file_tool(str(path), task_id=task_id))
        reconstructed = "".join(
            line.partition("|")[2].removeprefix("↳ ")
            for line in result["content"].splitlines()
        )

        assert "«redacted:sk-…»" in reconstructed
        assert secret not in reconstructed
        assert secret_body not in reconstructed
        assert "SAFE_TAIL" in reconstructed
    finally:
        clear_file_ops_cache(task_id)


def test_structured_char_budget_continues_at_lossless_display_line(
    tmp_path: Path,
) -> None:
    """Char-budget truncation reports a recoverable display-line cursor."""
    path = tmp_path / "budgeted-wide-row.xlsx"
    wide_value = "B" * 4_500 + "BUDGETED_TAIL"
    _write_xlsx(path, wide_value)
    task_id = "structured-budgeted-wide-row"
    row_parts = []
    offset = 1

    try:
        with mock.patch("tools.file_tools._max_read_chars_cached", 2_100):
            for _ in range(10):
                page = json.loads(
                    read_file_tool(
                        str(path),
                        offset=offset,
                        limit=2_000,
                        task_id=task_id,
                    )
                )
                for line in page["content"].splitlines():
                    _gutter, separator, body = line.partition("|")
                    if separator and (body.startswith("B") or body.startswith("↳ ")):
                        row_parts.append(body.removeprefix("↳ "))

                assert "not retrievable" not in page.get("hint", "")
                if not page["truncated"]:
                    break
                assert page["truncated_by"] == "characters"
                assert page["next_offset"] > offset
                offset = page["next_offset"]
            else:
                pytest.fail("structured continuation did not terminate")

        assert "".join(row_parts) == wide_value
    finally:
        clear_file_ops_cache(task_id)


@pytest.mark.parametrize("backend", ["docker", "ssh"])
@pytest.mark.parametrize("backend_path", ["~/report.ipynb", "~service/report.ipynb"])
def test_structured_read_preserves_backend_tilde_path(
    backend: str,
    backend_path: str,
) -> None:
    """Remote/container home expansion belongs to the target shell, not the host."""
    payload = json.dumps({
        "cells": [{"cell_type": "markdown", "source": "REMOTE_NOTEBOOK"}],
        "metadata": {},
        "nbformat": 4,
        "nbformat_minor": 5,
    }).encode("utf-8")

    class _RemoteEnvironment:
        cwd = "/workspace"

    class _RemoteFileOps:
        env = _RemoteEnvironment()

        def __init__(self) -> None:
            self.seen_paths = []

        def read_file_bytes(self, path, max_bytes=None):
            self.seen_paths.append(path)
            return ReadResult(
                base64_content=base64.b64encode(payload).decode("ascii"),
                file_size=len(payload),
                is_binary=False,
            )

        @staticmethod
        def _add_line_numbers(content, start_line=1):
            return "\n".join(
                f"{number}|{line}"
                for number, line in enumerate(content.split("\n"), start_line)
            )

    task_id = f"structured-{backend}-{backend_path}"
    file_ops = _RemoteFileOps()
    try:
        with (
            mock.patch(
                "tools.file_tools._terminal_env_type_for_task",
                return_value=backend,
            ),
            mock.patch("tools.file_tools._get_file_ops", return_value=file_ops),
        ):
            result = json.loads(read_file_tool(backend_path, task_id=task_id))

        assert "REMOTE_NOTEBOOK" in result["content"]
        assert file_ops.seen_paths == [backend_path]
    finally:
        clear_file_ops_cache(task_id)
