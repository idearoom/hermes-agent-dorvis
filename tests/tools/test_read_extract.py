#!/usr/bin/env python3
"""
Tests for structured-document extraction in the read_file tool.

Covers .ipynb / .docx / .xlsx extraction (ported from Kilo-Org/kilocode
#10733, #10737, #10740) and the read_file_tool integration: pagination,
line-numbering, graceful fallback on malformed input, and hidden-sheet
omission.

Run with:  python -m pytest tests/tools/test_read_extract.py -v
"""

import base64
import concurrent.futures
import json
import os
import struct
import tempfile
import threading
import unittest
import warnings
import zipfile
from pathlib import Path
from unittest import mock

from tools.read_extract import (
    ExtractionError,
    extract_document_text,
    is_extractable_document,
)
from tools.file_tools import read_file_tool


# ---------------------------------------------------------------------------
# Fixture builders — construct minimal valid OOXML / notebook files.
# ---------------------------------------------------------------------------

def _write_notebook(path, cells, nbformat=4):
    nb = {"cells": cells, "metadata": {}, "nbformat": nbformat, "nbformat_minor": 5}
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(nb, fh)


def _write_docx(path, document_xml):
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("[Content_Types].xml", "<Types/>")
        z.writestr("word/document.xml", document_xml)


def _write_xlsx(path, *, workbook, rels, shared, sheets):
    """sheets: dict of part-name -> xml string."""
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("xl/workbook.xml", workbook)
        z.writestr("xl/_rels/workbook.xml.rels", rels)
        if shared is not None:
            z.writestr("xl/sharedStrings.xml", shared)
        for part, xml in sheets.items():
            z.writestr(part, xml)


def _set_encrypted_flag(path):
    """Mark every local/central ZIP entry encrypted without encrypting data.

    The extractor must reject the metadata before attempting a member read,
    so a synthetically-set general-purpose flag is sufficient for the guard
    contract and avoids depending on a non-stdlib encrypted-ZIP writer.
    """
    with open(path, "rb") as fh:
        data = bytearray(fh.read())
    for signature, flag_offset in ((b"PK\x03\x04", 6), (b"PK\x01\x02", 8)):
        start = 0
        while True:
            start = data.find(signature, start)
            if start < 0:
                break
            flags = struct.unpack_from("<H", data, start + flag_offset)[0]
            struct.pack_into("<H", data, start + flag_offset, flags | 0x1)
            start += 4
    with open(path, "wb") as fh:
        fh.write(data)


def _replace_zip_member_name_bytes(path, old, new):
    """Replace equal-width local/central filenames in a synthetic archive."""
    if len(old) != len(new):
        raise AssertionError("replacement ZIP member names must have equal widths")
    with open(path, "rb") as fh:
        data = fh.read()
    if data.count(old) != 2:
        raise AssertionError("expected one local and one central ZIP filename")
    with open(path, "wb") as fh:
        fh.write(data.replace(old, new))


_NS_W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
_NS_S = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"


# ---------------------------------------------------------------------------
# is_extractable_document
# ---------------------------------------------------------------------------

class TestIsExtractable(unittest.TestCase):
    def test_recognized_extensions(self):
        self.assertTrue(is_extractable_document("a.ipynb"))
        self.assertTrue(is_extractable_document("/x/B.DOCX"))
        self.assertTrue(is_extractable_document("report.xlsx"))

    def test_unrecognized_extensions(self):
        self.assertFalse(is_extractable_document("a.py"))
        self.assertFalse(is_extractable_document("a.txt"))
        self.assertFalse(is_extractable_document("a.mp4"))

    def test_anydoc_extensions_track_availability(self):
        """PDF (and the other anydoc formats) are extractable exactly when
        the optional `anydoc` converter is importable."""
        from tools import read_extract

        available = read_extract._anydoc() is not None
        self.assertEqual(is_extractable_document("a.pdf"), available)
        self.assertEqual(is_extractable_document("a.odt"), available)
        self.assertEqual(is_extractable_document("a.epub"), available)


# ---------------------------------------------------------------------------
# Optional anydoc-backed formats (PDF, legacy Office, ODF, RTF, EPUB)
# ---------------------------------------------------------------------------

class TestAnydocExtraction(unittest.TestCase):
    """Real-binding tests — skipped when firecrawl-anydoc is not installed."""

    @classmethod
    def setUpClass(cls):
        from tools import read_extract

        cls.mod = read_extract._anydoc()
        if cls.mod is None:
            raise unittest.SkipTest("firecrawl-anydoc not installed")

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="rex_anydoc_")

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_rtf_extracts_markdown(self):
        p = os.path.join(self.tmp, "doc.rtf")
        with open(p, "w", encoding="ascii") as fh:
            fh.write(r"{\rtf1\ansi {\b Bold title}\par plain body\par}")
        text = extract_document_text(p)
        self.assertIn("Bold title", text)
        self.assertIn("plain body", text)
        self.assertTrue(text.endswith("\n"))

    def test_malformed_file_raises_extraction_error(self):
        p = os.path.join(self.tmp, "junk.pdf")
        with open(p, "wb") as fh:
            fh.write(b"\x00\x01 not a pdf at all")
        with self.assertRaises(ExtractionError):
            extract_document_text(p)

    def test_stdlib_docx_path_still_authoritative(self):
        """A .docx keeps using the stdlib extractor even with anydoc
        installed — behavior must be identical either way."""
        p = os.path.join(self.tmp, "d.docx")
        _write_docx(
            p,
            f'<w:document xmlns:w="{_NS_W}"><w:body>'
            "<w:p><w:r><w:t>hello</w:t></w:r></w:p>"
            "</w:body></w:document>",
        )
        text = extract_document_text(p)
        self.assertEqual(text, "hello\n")


class TestAnydocSizeCap(unittest.TestCase):
    """Oversized inputs must be rejected before anydoc converts them.
    Uses a fake binding so it runs regardless of local install state."""

    def setUp(self):
        from tools import read_extract

        self.rex = read_extract
        self._saved_module = read_extract._anydoc_module
        self._saved_cap = read_extract.MAX_ANYDOC_BYTES
        self.tmp = tempfile.mkdtemp(prefix="rex_cap_")
        self.calls = []

        class _FakeAnydoc:
            def to_markdown(_self, path):
                self.calls.append(path)
                return "converted\n"

        read_extract._anydoc_module = _FakeAnydoc()

    def tearDown(self):
        import shutil

        self.rex._anydoc_module = self._saved_module
        self.rex.MAX_ANYDOC_BYTES = self._saved_cap
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _write(self, name, size):
        p = os.path.join(self.tmp, name)
        with open(p, "wb") as fh:
            fh.write(b"x" * size)
        return p

    def test_oversized_file_rejected_before_conversion(self):
        from tools.read_extract import _extract_anydoc

        self.rex.MAX_ANYDOC_BYTES = 10
        p = self._write("big.pdf", 11)
        with self.assertRaises(ExtractionError) as ctx:
            _extract_anydoc(p)
        self.assertIn("too large", str(ctx.exception))
        self.assertEqual(self.calls, [])

    def test_file_at_limit_converts(self):
        from tools.read_extract import _extract_anydoc

        self.rex.MAX_ANYDOC_BYTES = 10
        p = self._write("ok.pdf", 10)
        self.assertEqual(_extract_anydoc(p), "converted\n")
        self.assertEqual(self.calls, [p])

    def test_missing_file_raises_extraction_error(self):
        from tools.read_extract import _extract_anydoc

        with self.assertRaises(ExtractionError):
            _extract_anydoc(os.path.join(self.tmp, "gone.pdf"))
        self.assertEqual(self.calls, [])


class TestAnydocAbsent(unittest.TestCase):
    """The absent-dep contract, verified regardless of local install state
    by forcing the cached module handle to None."""

    def setUp(self):
        from tools import read_extract

        self._saved = read_extract._anydoc_module
        read_extract._anydoc_module = None

    def tearDown(self):
        from tools import read_extract

        read_extract._anydoc_module = self._saved

    def test_pdf_not_extractable_without_anydoc(self):
        self.assertFalse(is_extractable_document("a.pdf"))
        self.assertFalse(is_extractable_document("a.rtf"))

    def test_extract_raises_unsupported_without_anydoc(self):
        from tools.read_extract import _extract_anydoc

        with self.assertRaises(ExtractionError):
            _extract_anydoc("/tmp/whatever.pdf")

    def test_stdlib_formats_unaffected(self):
        self.assertTrue(is_extractable_document("a.ipynb"))
        self.assertTrue(is_extractable_document("a.docx"))
        self.assertTrue(is_extractable_document("a.xlsx"))


class TestAnydocManagedQuarantine(unittest.TestCase):
    """A managed image can quarantine AnyDoc without disabling other deps."""

    def test_disable_env_blocks_cached_or_lazy_anydoc_only(self):
        from tools import read_extract

        saved_module = read_extract._anydoc_module
        read_extract._anydoc_module = object()
        try:
            with mock.patch.dict(
                os.environ,
                {"HERMES_DISABLE_ANYDOC": "1"},
            ), mock.patch("tools.lazy_deps.ensure") as ensure:
                self.assertIsNone(read_extract._anydoc())
                self.assertFalse(is_extractable_document("report.pdf"))
                self.assertTrue(is_extractable_document("report.docx"))
                self.assertTrue(is_extractable_document("report.xlsx"))
                self.assertTrue(is_extractable_document("report.ipynb"))

            ensure.assert_not_called()
        finally:
            read_extract._anydoc_module = saved_module

    def test_disable_env_blocks_missing_module_install_and_import(self):
        from tools import read_extract

        saved_module = read_extract._anydoc_module
        saved_failed_at = read_extract._anydoc_failed_at
        read_extract._anydoc_module = read_extract._ANYDOC_UNSET
        read_extract._anydoc_failed_at = None
        try:
            with mock.patch.dict(
                os.environ,
                {"HERMES_DISABLE_ANYDOC": "1"},
            ), mock.patch("tools.lazy_deps.ensure") as ensure, mock.patch(
                "importlib.import_module",
            ) as import_module:
                self.assertIsNone(read_extract._anydoc())

            ensure.assert_not_called()
            import_module.assert_not_called()
            self.assertIs(read_extract._anydoc_module, read_extract._ANYDOC_UNSET)
            self.assertIsNone(read_extract._anydoc_failed_at)
        finally:
            read_extract._anydoc_module = saved_module
            read_extract._anydoc_failed_at = saved_failed_at

    def test_security_config_can_disable_anydoc_without_internal_env(self):
        from tools import read_extract

        with mock.patch.dict(os.environ, {}, clear=True), mock.patch(
            "hermes_cli.config.load_config_readonly",
            return_value={"security": {"allow_anydoc": False}},
        ):
            self.assertTrue(read_extract._anydoc_disabled())


class TestAnydocInitLifecycle(unittest.TestCase):
    """First-load lifecycle: one failed load must not disable extraction
    for the rest of the process, and concurrent first use must not race."""

    def setUp(self):
        from tools import read_extract

        self.rex = read_extract
        self._saved_module = read_extract._anydoc_module
        self._saved_failed_at = read_extract._anydoc_failed_at
        self._saved_retry = read_extract.ANYDOC_RETRY_SECONDS
        read_extract._anydoc_module = read_extract._ANYDOC_UNSET
        read_extract._anydoc_failed_at = None
        self._ensure = mock.patch("tools.lazy_deps.ensure", return_value=None)
        self._ensure.start()

    def tearDown(self):
        self._ensure.stop()
        self.rex._anydoc_module = self._saved_module
        self.rex._anydoc_failed_at = self._saved_failed_at
        self.rex.ANYDOC_RETRY_SECONDS = self._saved_retry

    def test_successful_load_is_cached(self):
        fake = object()
        calls = []

        def fake_import(name):
            calls.append(name)
            return fake

        with mock.patch("importlib.import_module", side_effect=fake_import):
            self.assertIs(self.rex._anydoc(), fake)
            self.assertIs(self.rex._anydoc(), fake)
        self.assertEqual(calls, ["anydoc"])

    def test_failed_reconciliation_does_not_import_unverified_binding(self):
        with mock.patch(
            "tools.lazy_deps.ensure", side_effect=RuntimeError("wrong version")
        ), mock.patch("importlib.import_module") as import_module:
            self.assertIsNone(self.rex._anydoc())
        import_module.assert_not_called()

    def test_failed_load_is_retried_after_cooldown(self):
        fake = object()
        calls = []

        def fake_import(name):
            calls.append(name)
            if len(calls) == 1:
                raise ImportError("boom")
            return fake

        self.rex.ANYDOC_RETRY_SECONDS = 0.0
        with mock.patch("importlib.import_module", side_effect=fake_import):
            self.assertIsNone(self.rex._anydoc())
            self.assertIs(self.rex._anydoc(), fake)
        self.assertEqual(calls, ["anydoc", "anydoc"])

    def test_failed_load_not_retried_within_cooldown(self):
        calls = []

        def fake_import(name):
            calls.append(name)
            raise ImportError("boom")

        self.rex.ANYDOC_RETRY_SECONDS = 3600.0
        with mock.patch("importlib.import_module", side_effect=fake_import):
            self.assertIsNone(self.rex._anydoc())
            self.assertIsNone(self.rex._anydoc())
        # One import attempt total, and the handle stays UNSET so a retry
        # remains possible once the cooldown expires.
        self.assertEqual(calls, ["anydoc"])
        self.assertIs(self.rex._anydoc_module, self.rex._ANYDOC_UNSET)

    def test_concurrent_first_load_imports_once(self):
        import threading

        fake = object()
        calls = []
        barrier = threading.Barrier(4)

        def fake_import(name):
            calls.append(name)
            return fake

        def worker(out):
            barrier.wait(5)
            out.append(self.rex._anydoc())

        with mock.patch("importlib.import_module", side_effect=fake_import):
            results = []
            threads = [threading.Thread(target=worker, args=(results,)) for _ in range(3)]
            for t in threads:
                t.start()
            barrier.wait(5)
            for t in threads:
                t.join(5)
        self.assertEqual(calls, ["anydoc"])
        self.assertEqual(results, [fake, fake, fake])


# ---------------------------------------------------------------------------
# Notebooks (.ipynb) — #10733
# ---------------------------------------------------------------------------

class TestNotebookExtraction(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="rex_nb_")

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_markdown_and_code_in_order(self):
        p = os.path.join(self.tmp, "nb.ipynb")
        _write_notebook(p, [
            {"cell_type": "markdown", "source": ["# Title\n", "para"]},
            {"cell_type": "code", "source": "x = 1\nprint(x)",
             "outputs": [{"output_type": "stream", "text": ["1\n"]}],
             "execution_count": 1},
        ])
        text = extract_document_text(p)
        self.assertIn("# Title", text)
        self.assertIn("print(x)", text)
        # Output payloads must NOT leak into the extracted text.
        self.assertNotIn("output_type", text)
        self.assertNotIn("execution_count", text)
        # Order preserved: markdown before code.
        self.assertLess(text.index("Title"), text.index("print(x)"))


    def test_empty_cells_raises(self):
        p = os.path.join(self.tmp, "empty.ipynb")
        _write_notebook(p, [])
        with self.assertRaises(ExtractionError):
            extract_document_text(p)

    def test_stream_output_rendered(self):
        p = os.path.join(self.tmp, "nb_out.ipynb")
        _write_notebook(p, [
            {"cell_type": "code", "source": "print('epoch done')",
             "outputs": [{"output_type": "stream", "name": "stdout",
                          "text": ["epoch done\n", "loss=0.42\n"]}]},
        ])
        text = extract_document_text(p)
        self.assertIn("Output (cell 1)", text)
        self.assertIn("loss=0.42", text)

    def test_error_output_keeps_traceback_strips_ansi(self):
        p = os.path.join(self.tmp, "nb_err.ipynb")
        _write_notebook(p, [
            {"cell_type": "code", "source": "1/0",
             "outputs": [{"output_type": "error", "ename": "ZeroDivisionError",
                          "evalue": "division by zero",
                          "traceback": ["\x1b[31mZeroDivisionError\x1b[0m: division by zero"]}]},
        ])
        text = extract_document_text(p)
        self.assertIn("Error: ZeroDivisionError: division by zero", text)
        self.assertNotIn("\x1b", text)

    def test_image_output_replaced_with_placeholder(self):
        payload = "A" * 4096  # ~3 KB decoded
        p = os.path.join(self.tmp, "nb_img.ipynb")
        _write_notebook(p, [
            {"cell_type": "code", "source": "plot()",
             "outputs": [{"output_type": "display_data",
                          "data": {"image/png": payload}}]},
        ])
        text = extract_document_text(p)
        self.assertIn("[image/png output — 3 KB, omitted]", text)
        self.assertNotIn(payload, text)

    def test_execute_result_prefers_text_plain_over_html(self):
        p = os.path.join(self.tmp, "nb_df.ipynb")
        _write_notebook(p, [
            {"cell_type": "code", "source": "df.head()",
             "outputs": [{"output_type": "execute_result",
                          "data": {"text/html": "<table><tr><td>1</td></tr></table>",
                                   "text/plain": "   col\n0    1"}}]},
        ])
        text = extract_document_text(p)
        self.assertIn("   col", text)
        self.assertNotIn("<table>", text)

    def test_carriage_return_progress_collapsed(self):
        p = os.path.join(self.tmp, "nb_tqdm.ipynb")
        _write_notebook(p, [
            {"cell_type": "code", "source": "train()",
             "outputs": [{"output_type": "stream",
                          "text": [" 10%|█\r 50%|█████\r100%|██████████\n"]}]},
        ])
        text = extract_document_text(p)
        self.assertIn("100%|██████████", text)
        self.assertNotIn("50%", text)

    def test_output_cleanup_receives_only_progressively_bounded_raw_input(self):
        """ANSI cleanup never receives an attacker-sized notebook output."""
        from tools import read_extract

        p = os.path.join(self.tmp, "adversarial-output.ipynb")
        _write_notebook(p, [
            {
                "cell_type": "code",
                "source": "emit_hostile_progress()",
                "outputs": [
                    {
                        "output_type": "display_data",
                        "data": {"text/plain": "z" * 64},
                    },
                    {
                        "output_type": "stream",
                        # Repeated unterminated OSC introducers make the
                        # former ANSI regex rescan the remaining suffix for
                        # every introducer: the reproduced superlinear shape.
                        "text": "\x1b]" * 50_000,
                    },
                ],
            },
        ])
        cleaned_lengths = []

        def guarded_strip_ansi(value):
            cleaned_lengths.append(len(value))
            self.assertLessEqual(
                sum(cleaned_lengths),
                read_extract._MAX_NOTEBOOK_OUTPUT_RAW_CHARS,
            )
            return value

        with mock.patch.object(
            read_extract, "_MAX_NOTEBOOK_OUTPUT_RAW_CHARS", 128
        ), mock.patch(
            "tools.ansi_strip.strip_ansi", side_effect=guarded_strip_ansi
        ):
            text = extract_document_text(p)

        self.assertTrue(cleaned_lengths)
        self.assertEqual(cleaned_lengths, [64, 64])
        self.assertLessEqual(sum(cleaned_lengths), 128)
        self.assertIn("raw output omitted before display sanitization", text)

    def test_widget_output_placeholder(self):
        p = os.path.join(self.tmp, "nb_widget.ipynb")
        _write_notebook(p, [
            {"cell_type": "code", "source": "slider",
             "outputs": [{"output_type": "display_data",
                          "data": {"application/vnd.jupyter.widget-view+json": {"model_id": "abc"},
                                   "text/plain": "IntSlider(value=0)"}}]},
        ])
        text = extract_document_text(p)
        self.assertIn("[interactive widget — omitted]", text)

    def test_oversized_outputs_truncated(self):
        from tools.read_extract import _MAX_OUTPUT_CHARS
        p = os.path.join(self.tmp, "nb_big.ipynb")
        _write_notebook(p, [
            {"cell_type": "markdown", "source": "# intro"},
            {"cell_type": "code", "source": "spam()",
             "outputs": [{"output_type": "stream",
                          "text": "x" * (_MAX_OUTPUT_CHARS + 5000)}]},
        ])
        text = extract_document_text(p)
        self.assertIn("output chars truncated", text)
        self.assertIn(
            '— full output location: notebook file "nb_big.ipynb", '
            'JSON path ".cells[1].outputs"]',
            text,
        )
        self.assertLess(len(text), _MAX_OUTPUT_CHARS + 2000)

    def test_oversized_outputs_truncated_v3_location_hint(self):
        from tools.read_extract import _MAX_OUTPUT_CHARS
        p = os.path.join(self.tmp, "nb_v3_big.ipynb")
        nb = {"worksheets": [{"cells": [
            {"cell_type": "markdown", "source": "# intro"},
            {"cell_type": "code", "source": "spam()",
             "outputs": [{"output_type": "stream",
                          "text": "x" * (_MAX_OUTPUT_CHARS + 5000)}]},
        ]}], "nbformat": 3}
        with open(p, "w") as fh:
            json.dump(nb, fh)
        text = extract_document_text(p)
        self.assertIn("output chars truncated", text)
        self.assertIn(
            '— full output location: notebook file "nb_v3_big.ipynb", '
            'JSON path ".worksheets[0].cells[1].outputs"]',
            text,
        )

    def test_long_output_guidance_is_non_executable_for_arbitrary_filename(self):
        from tools.read_extract import _MAX_OUTPUT_CHARS

        filename = "- report '$(touch owned)' ;\nnext line.ipynb"
        p = os.path.join(self.tmp, filename)
        _write_notebook(p, [
            {
                "cell_type": "code",
                "source": "spam()",
                "outputs": [
                    {
                        "output_type": "stream",
                        "text": "x" * (_MAX_OUTPUT_CHARS + 1),
                    },
                ],
            },
        ])

        from tools import read_extract

        with mock.patch.object(
            read_extract,
            "_MAX_NOTEBOOK_OUTPUT_RAW_CHARS",
            _MAX_OUTPUT_CHARS * 2,
        ):
            text = extract_document_text(p)

        self.assertIn("output chars truncated", text)
        self.assertIn(json.dumps(filename, ensure_ascii=True), text)
        self.assertNotIn(filename, text)
        self.assertNotIn("jq -r", text)
        self.assertNotIn("`jq", text)

    def test_legacy_v3_pyout_flat_fields(self):
        p = os.path.join(self.tmp, "nb_v3.ipynb")
        nb = {"worksheets": [{"cells": [
            {"cell_type": "code", "source": "1+1",
             "outputs": [{"output_type": "pyout", "text": ["2"]}]},
        ]}], "nbformat": 3}
        with open(p, "w") as fh:
            json.dump(nb, fh)
        text = extract_document_text(p)
        self.assertIn("Output (cell 1)", text)
        self.assertIn("2", text)

    def test_valid_v3_code_input_and_heading_are_preserved(self):
        p = os.path.join(self.tmp, "valid-v3.ipynb")
        nb = {
            "worksheets": [{
                "cells": [
                    {
                        "cell_type": "heading",
                        "level": 2,
                        "source": ["Legacy ", "Heading"],
                    },
                    {
                        "cell_type": "code",
                        "input": ["value = 40\n", "print(value + 2)"],
                        "outputs": [
                            {"output_type": "stream", "text": "42\n"},
                        ],
                    },
                ],
            }],
            "metadata": {},
            "nbformat": 3,
            "nbformat_minor": 0,
        }
        with open(p, "w", encoding="utf-8") as fh:
            json.dump(nb, fh)

        text = extract_document_text(p)

        self.assertIn("## Legacy Heading", text)
        self.assertIn("value = 40", text)
        self.assertIn("print(value + 2)", text)
        self.assertIn("Output (cell 1)", text)

    def test_malformed_outputs_ignored(self):
        p = os.path.join(self.tmp, "nb_bad_out.ipynb")
        _write_notebook(p, [
            {"cell_type": "code", "source": "ok()",
             "outputs": ["not-a-dict", {"output_type": "bogus"}, None]},
            {"cell_type": "code", "source": "also_ok()", "outputs": "not-a-list"},
        ])
        text = extract_document_text(p)
        self.assertIn("ok()", text)
        self.assertIn("also_ok()", text)
        self.assertNotIn("Output (cell", text)

    def test_oversized_notebook_rejected_before_json_parse(self):
        from tools import read_extract

        p = os.path.join(self.tmp, "oversized.ipynb")
        _write_notebook(p, [
            {"cell_type": "markdown", "source": "x" * 512},
        ])
        with mock.patch.object(read_extract, "MAX_NOTEBOOK_BYTES", 128), \
                mock.patch.object(read_extract.json, "loads") as json_load:
            with self.assertRaisesRegex(
                ExtractionError, "Notebook too large to parse"
            ):
                extract_document_text(p)
        json_load.assert_not_called()

    def test_malformed_legacy_container_shapes_raise_extraction_error(self):
        malformed = (
            {"worksheets": None, "nbformat": 3},
            {"worksheets": [{"cells": None}], "nbformat": 3},
            {"worksheets": [{"cells": 42}], "nbformat": 3},
        )
        for index, notebook in enumerate(malformed):
            with self.subTest(notebook=notebook):
                p = os.path.join(self.tmp, f"malformed-{index}.ipynb")
                with open(p, "w", encoding="utf-8") as fh:
                    json.dump(notebook, fh)
                with self.assertRaisesRegex(ExtractionError, "contains no cells"):
                    extract_document_text(p)

    def test_notebook_at_input_limit_extracts(self):
        from tools import read_extract

        p = os.path.join(self.tmp, "at-limit.ipynb")
        _write_notebook(p, [
            {"cell_type": "markdown", "source": "near boundary"},
        ])
        size = os.path.getsize(p)
        with mock.patch.object(read_extract, "MAX_NOTEBOOK_BYTES", size):
            text = extract_document_text(p)
        self.assertIn("near boundary", text)

    def test_notebook_extracted_text_is_bounded(self):
        from tools import read_extract

        p = os.path.join(self.tmp, "long-source.ipynb")
        _write_notebook(p, [
            {"cell_type": "markdown", "source": "x" * 2048},
        ])
        with mock.patch.object(
            read_extract, "MAX_EXTRACTED_TEXT_CHARS", 256
        ):
            text = extract_document_text(p)
        self.assertLessEqual(len(text), 256)
        self.assertIn("Extraction truncated", text)

    def test_ten_concurrent_notebook_reads_remain_isolated(self):
        paths = []
        for index in range(10):
            p = os.path.join(self.tmp, f"concurrent-{index}.ipynb")
            _write_notebook(p, [
                {"cell_type": "markdown", "source": f"reader {index}"},
            ])
            paths.append(p)

        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as pool:
            texts = list(pool.map(extract_document_text, paths))

        self.assertEqual(len(texts), 10)
        for index, text in enumerate(texts):
            self.assertIn(f"reader {index}", text)

    def test_native_reader_capacity_refuses_without_waiting(self):
        from tools import read_extract

        slots = mock.Mock()
        slots.acquire.return_value = False
        with mock.patch.object(read_extract, "_native_read_slots", slots):
            with self.assertRaisesRegex(
                read_extract.ExtractionBusyError, "retry this read shortly"
            ):
                with read_extract.native_extraction_slot("busy.ipynb"):
                    self.fail("a saturated reader must not enter")

        slots.acquire.assert_called_once_with(blocking=False)
        slots.release.assert_not_called()

    def test_native_reader_slot_is_reentrant_across_byte_materialization(self):
        from tools import read_extract

        payload = json.dumps({
            "cells": [{"cell_type": "markdown", "source": "reentrant"}],
            "metadata": {},
            "nbformat": 4,
            "nbformat_minor": 5,
        }).encode("utf-8")
        slots = threading.BoundedSemaphore(1)
        with mock.patch.object(read_extract, "_native_read_slots", slots):
            text = read_extract.extract_document_bytes(payload, "remote.ipynb")
        self.assertIn("reentrant", text)

    def test_native_reader_allocation_proof_fits_concurrency_budget(self):
        from tools import read_extract

        notebook_paths = (
            read_extract.ESTIMATED_NOTEBOOK_PARSED_WORKING_SET_BYTES,
            read_extract.ESTIMATED_NOTEBOOK_FALLBACK_WORKING_SET_BYTES,
        )
        for estimate in notebook_paths:
            self.assertLessEqual(
                estimate,
                read_extract.MAX_NATIVE_READ_WORKING_SET_BYTES,
            )
        self.assertEqual(
            read_extract.ESTIMATED_NOTEBOOK_WORKING_SET_BYTES,
            max(notebook_paths),
        )
        per_request = max(
            read_extract.ESTIMATED_OOXML_WORKING_SET_BYTES,
            read_extract.ESTIMATED_NOTEBOOK_WORKING_SET_BYTES,
        )
        self.assertLessEqual(
            per_request,
            read_extract.MAX_NATIVE_READ_WORKING_SET_BYTES,
        )
        self.assertLessEqual(
            per_request * read_extract.MAX_CONCURRENT_NATIVE_READS,
            5 * 1024 * 1024 * 1024 // 2,
        )

    def test_exact_output_limit_does_not_add_a_newline_past_cap(self):
        from tools import read_extract

        with mock.patch.object(read_extract, "MAX_EXTRACTED_TEXT_CHARS", 8):
            output = read_extract._BoundedText("test")
            self.assertTrue(output.append("12345678"))
            rendered = output.render()

        self.assertEqual(rendered, "12345678")
        self.assertEqual(len(rendered), 8)


# ---------------------------------------------------------------------------
# Word documents (.docx) — #10737
# ---------------------------------------------------------------------------

class TestDocxExtraction(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="rex_docx_")

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _doc(self, body):
        return (f'<?xml version="1.0"?><w:document xmlns:w="{_NS_W}">'
                f'<w:body>{body}</w:body></w:document>')

    def test_paragraphs_and_runs(self):
        p = os.path.join(self.tmp, "d.docx")
        _write_docx(p, self._doc(
            '<w:p><w:r><w:t>Hello </w:t></w:r><w:r><w:t>World</w:t></w:r></w:p>'
            '<w:p><w:r><w:t>Second</w:t></w:r></w:p>'))
        text = extract_document_text(p)
        self.assertTrue(text.startswith("[Extraction view: main document body only"))
        self.assertIn("Hello World", text)
        self.assertIn("Second", text)

    def test_strict_ooxml_wordprocessing_namespace_extracts(self):
        strict_w = "http://purl.oclc.org/ooxml/wordprocessingml/main"
        p = os.path.join(self.tmp, "strict.docx")
        _write_docx(
            p,
            f'<w:document xmlns:w="{strict_w}"><w:body>'
            "<w:p><w:r><w:t>Strict Word text</w:t></w:r></w:p>"
            "</w:body></w:document>",
        )

        text = extract_document_text(p)

        self.assertIn("Strict Word text", text)

    def test_non_body_parts_are_disclosed_and_not_silently_implied(self):
        p = os.path.join(self.tmp, "body-only.docx")
        _write_docx(
            p,
            self._doc(
                "<w:p><w:r><w:t>BODY_TEXT</w:t></w:r></w:p>"
            ),
        )
        with zipfile.ZipFile(p, "a") as z:
            z.writestr(
                "word/header1.xml",
                f'<w:hdr xmlns:w="{_NS_W}"><w:p><w:r>'
                "<w:t>HEADER_TEXT_NOT_IN_VIEW</w:t></w:r></w:p></w:hdr>",
            )

        text = extract_document_text(p)

        self.assertIn("headers, footers, footnotes, endnotes, and comments", text)
        self.assertIn("BODY_TEXT", text)
        self.assertNotIn("HEADER_TEXT_NOT_IN_VIEW", text)


    def test_missing_document_xml_raises(self):
        p = os.path.join(self.tmp, "nodoc.docx")
        with zipfile.ZipFile(p, "w") as z:
            z.writestr("other.xml", "<x/>")
        with self.assertRaises(ExtractionError):
            extract_document_text(p)

    def test_dtd_and_entity_declaration_is_rejected(self):
        p = os.path.join(self.tmp, "entity.docx")
        xml = (
            '<?xml version="1.0"?>'
            '<!DOCTYPE w:document [<!ENTITY amplified "unsafe">]>'
            f'<w:document xmlns:w="{_NS_W}"><w:body><w:p>'
            "<w:r><w:t>&amplified;</w:t></w:r>"
            "</w:p></w:body></w:document>"
        )
        _write_docx(p, xml)
        with self.assertRaisesRegex(ExtractionError, "Unsafe DTD/entity"):
            extract_document_text(p)

    def test_dtd_split_across_reader_chunks_is_rejected(self):
        from tools import read_extract

        p = os.path.join(self.tmp, "split-doctype.docx")
        declaration = '<?xml version="1.0"?>'
        chunk_size = 64
        # End the first bounded read with "<!DO" so only the reader's tail
        # scan can recognize the completed declaration on the next read.
        padding = " " * (chunk_size - len(declaration.encode("utf-8")) - 4)
        xml = (
            declaration
            + padding
            + '<!DOCTYPE w:document [<!ENTITY x "unsafe">]>'
            + f'<w:document xmlns:w="{_NS_W}"><w:body><w:p>'
            + "<w:r><w:t>&x;</w:t></w:r></w:p></w:body></w:document>"
        )
        self.assertEqual(xml.encode("utf-8")[chunk_size - 4:chunk_size], b"<!DO")
        _write_docx(p, xml)
        with mock.patch.object(read_extract, "_XML_READ_CHUNK_BYTES", chunk_size):
            with self.assertRaisesRegex(ExtractionError, "Unsafe DTD/entity"):
                extract_document_text(p)

    def test_oversized_document_xml_rejected_before_member_read(self):
        from tools import read_extract

        p = os.path.join(self.tmp, "oversized.docx")
        xml = self._doc("<w:p><w:r><w:t>" + "x" * 512 + "</w:t></w:r></w:p>")
        _write_docx(p, xml)
        with mock.patch.object(read_extract, "MAX_OOXML_MEMBER_BYTES", 128), \
                mock.patch.object(
                    zipfile.ZipFile,
                    "open",
                    side_effect=AssertionError("member must not be opened"),
                ):
            with self.assertRaisesRegex(
                ExtractionError, "word/document.xml.*uncompressed"
            ):
                extract_document_text(p)

    def test_document_xml_at_member_limit_extracts(self):
        from tools import read_extract

        p = os.path.join(self.tmp, "at-limit.docx")
        xml = self._doc("<w:p><w:r><w:t>boundary</w:t></w:r></w:p>")
        with zipfile.ZipFile(p, "w", compression=zipfile.ZIP_STORED) as z:
            z.writestr("word/document.xml", xml)
        with mock.patch.object(
            read_extract, "MAX_OOXML_MEMBER_BYTES", len(xml.encode("utf-8"))
        ):
            text = extract_document_text(p)
        self.assertTrue(text.startswith("[Extraction view: main document body only"))
        self.assertTrue(text.endswith("boundary\n"))

    def test_docx_zip_bomb_rejected_before_member_read(self):
        from tools import read_extract

        p = os.path.join(self.tmp, "bomb.docx")
        body = "A" * (read_extract.MIN_ZIP_RATIO_MEMBER_BYTES + 128_000)
        xml = self._doc(f"<w:p><w:r><w:t>{body}</w:t></w:r></w:p>")
        with zipfile.ZipFile(p, "w", compression=zipfile.ZIP_DEFLATED) as z:
            z.writestr("word/document.xml", xml)
        with mock.patch.object(
            zipfile.ZipFile,
            "open",
            side_effect=AssertionError("member must not be opened"),
        ):
            with self.assertRaisesRegex(ExtractionError, "compression ratio"):
                extract_document_text(p)

    def test_highly_repetitive_valid_docx_does_not_hit_ratio_guard(self):
        """Real OOXML repetition can exceed 200:1 without being a bomb."""
        p = os.path.join(self.tmp, "repetitive-valid.docx")
        paragraph = (
            "<w:p><w:r><w:t>Quarterly recurring status line"
            "</w:t></w:r></w:p>"
        )
        xml = self._doc(paragraph * 20_000)
        with zipfile.ZipFile(p, "w", compression=zipfile.ZIP_DEFLATED) as z:
            z.writestr("word/document.xml", xml)
        with zipfile.ZipFile(p) as z:
            info = z.getinfo("word/document.xml")
            ratio = info.file_size / info.compress_size
        self.assertGreater(ratio, 200)
        text = extract_document_text(p)
        self.assertIn("Quarterly recurring status line", text)
        self.assertNotIn("Extraction truncated", text)

    def test_small_high_ratio_single_run_is_allowed(self):
        """Ratio screening starts only where expansion is resource-relevant."""
        from tools import read_extract

        p = os.path.join(self.tmp, "small-high-ratio.docx")
        body = "A" * (1024 * 1024 + 128_000)
        xml = self._doc(f"<w:p><w:r><w:t>{body}</w:t></w:r></w:p>")
        self.assertLess(len(xml.encode("utf-8")), read_extract.MIN_ZIP_RATIO_MEMBER_BYTES)
        with zipfile.ZipFile(p, "w", compression=zipfile.ZIP_DEFLATED) as z:
            z.writestr("word/document.xml", xml)
        with zipfile.ZipFile(p) as z:
            info = z.getinfo("word/document.xml")
            ratio = info.file_size / info.compress_size
        self.assertGreater(ratio, read_extract.MAX_ZIP_COMPRESSION_RATIO)
        text = extract_document_text(p)
        self.assertIn("A" * 100, text)

    def test_too_many_zip_entries_rejected_before_zipfile_construction(self):
        from tools import read_extract

        p = os.path.join(self.tmp, "many-entries.docx")
        with zipfile.ZipFile(p, "w") as z:
            z.writestr("word/document.xml", self._doc("<w:p/>"))
            for index in range(3):
                z.writestr(f"customXml/item{index}.xml", "<x/>")
        with mock.patch.object(read_extract, "MAX_OOXML_ZIP_ENTRIES", 3), \
                mock.patch.object(
                    zipfile,
                    "ZipFile",
                    side_effect=AssertionError("central directory must not load"),
                ):
            with self.assertRaisesRegex(ExtractionError, "4 entries.*limit is 3"):
                extract_document_text(p)

    def test_central_directory_size_rejected_before_zipfile_construction(self):
        from tools import read_extract

        p = os.path.join(self.tmp, "large-central-directory.docx")
        _write_docx(p, self._doc("<w:p/>"))
        with mock.patch.object(
            read_extract, "MAX_OOXML_CENTRAL_DIRECTORY_BYTES", 32
        ), mock.patch.object(
            zipfile,
            "ZipFile",
            side_effect=AssertionError("central directory must not load"),
        ):
            with self.assertRaisesRegex(ExtractionError, "central directory.*limit"):
                extract_document_text(p)

    def test_forged_low_entry_count_rejected_before_zipfile_construction(self):
        p = os.path.join(self.tmp, "forged-count.docx")
        with zipfile.ZipFile(p, "w") as z:
            z.writestr("word/document.xml", self._doc("<w:p/>"))
            z.writestr("customXml/item1.xml", "<x/>")
        with open(p, "rb") as fh:
            data = bytearray(fh.read())
        eocd = data.rfind(b"PK\x05\x06")
        self.assertGreaterEqual(eocd, 0)
        struct.pack_into("<H", data, eocd + 8, 1)
        struct.pack_into("<H", data, eocd + 10, 1)
        with open(p, "wb") as fh:
            fh.write(data)
        with mock.patch.object(
            zipfile,
            "ZipFile",
            side_effect=AssertionError("central directory must not load"),
        ):
            with self.assertRaisesRegex(ExtractionError, "declares 1 entries.*contains 2"):
                extract_document_text(p)

    def test_zip64_metadata_is_rejected_before_zipfile_construction(self):
        p = os.path.join(self.tmp, "zip64.docx")
        _write_docx(p, self._doc("<w:p/>"))
        with open(p, "rb") as fh:
            data = bytearray(fh.read())
        eocd = data.rfind(b"PK\x05\x06")
        self.assertGreaterEqual(eocd, 0)
        struct.pack_into("<H", data, eocd + 8, 0xFFFF)
        struct.pack_into("<H", data, eocd + 10, 0xFFFF)
        with open(p, "wb") as fh:
            fh.write(data)
        with mock.patch.object(
            zipfile,
            "ZipFile",
            side_effect=AssertionError("central directory must not load"),
        ):
            with self.assertRaisesRegex(ExtractionError, "ZIP64.*not supported"):
                extract_document_text(p)

    def test_missing_private_zip_preflight_api_fails_cleanly(self):
        p = os.path.join(self.tmp, "runtime-compat.docx")
        _write_docx(p, self._doc("<w:p/>"))
        with mock.patch.object(zipfile, "_EndRecData", None):
            with self.assertRaisesRegex(
                ExtractionError, "runtime lacks the bounded ZIP preflight API"
            ):
                extract_document_text(p)

    def test_multidisk_metadata_is_rejected_before_zipfile_construction(self):
        p = os.path.join(self.tmp, "multidisk.docx")
        _write_docx(p, self._doc("<w:p/>"))
        with open(p, "rb") as fh:
            data = bytearray(fh.read())
        eocd = data.rfind(b"PK\x05\x06")
        self.assertGreaterEqual(eocd, 0)
        struct.pack_into("<H", data, eocd + 4, 1)
        with open(p, "wb") as fh:
            fh.write(data)
        with mock.patch.object(
            zipfile,
            "ZipFile",
            side_effect=AssertionError("central directory must not load"),
        ):
            with self.assertRaisesRegex(ExtractionError, "multi-disk"):
                extract_document_text(p)

    def test_unsupported_member_compression_is_rejected_before_read(self):
        p = os.path.join(self.tmp, "bzip2.docx")
        with zipfile.ZipFile(p, "w", compression=zipfile.ZIP_BZIP2) as z:
            z.writestr("word/document.xml", self._doc("<w:p/>"))
        with mock.patch.object(
            zipfile.ZipFile,
            "open",
            side_effect=AssertionError("member must not be opened"),
        ):
            with self.assertRaisesRegex(ExtractionError, "compression method"):
                extract_document_text(p)

    def test_duplicate_member_is_rejected(self):
        p = os.path.join(self.tmp, "duplicate.docx")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            with zipfile.ZipFile(p, "w") as z:
                z.writestr("word/document.xml", self._doc("<w:p/>"))
                z.writestr("word/document.xml", self._doc("<w:p/>"))
        with mock.patch.object(
            zipfile.ZipFile,
            "open",
            side_effect=AssertionError("member must not be opened"),
        ):
            with self.assertRaisesRegex(ExtractionError, "Duplicate ZIP member"):
                extract_document_text(p)

    def test_unsafe_member_path_is_rejected(self):
        p = os.path.join(self.tmp, "unsafe.docx")
        with zipfile.ZipFile(p, "w") as z:
            z.writestr("word/document.xml", self._doc("<w:p/>"))
            z.writestr("word/../custom.xml", "<x/>")
        with mock.patch.object(
            zipfile.ZipFile,
            "open",
            side_effect=AssertionError("member must not be opened"),
        ):
            with self.assertRaisesRegex(ExtractionError, "Unsafe ZIP member path"):
                extract_document_text(p)

    def test_nul_in_original_member_name_is_rejected_before_read(self):
        p = os.path.join(self.tmp, "nul-name.docx")
        with zipfile.ZipFile(p, "w") as z:
            z.writestr("word/document.xml", self._doc("<w:p/>"))
            z.writestr("word/evilX.xml", "<x/>")
        _replace_zip_member_name_bytes(
            p,
            b"word/evilX.xml",
            b"word/evil\x00.xml",
        )
        with mock.patch.object(
            zipfile.ZipFile,
            "open",
            side_effect=AssertionError("member must not be opened"),
        ):
            with self.assertRaisesRegex(ExtractionError, "Unsafe ZIP member path"):
                extract_document_text(p)

    def test_windows_member_paths_are_rejected_before_read(self):
        for index, unsafe_name in enumerate((r"word\evil.xml", "C:/evil.xml")):
            with self.subTest(name=unsafe_name):
                p = os.path.join(self.tmp, f"windows-name-{index}.docx")
                with zipfile.ZipFile(p, "w") as z:
                    z.writestr("word/document.xml", self._doc("<w:p/>"))
                    z.writestr(unsafe_name, "<x/>")
                with mock.patch.object(
                    zipfile.ZipFile,
                    "open",
                    side_effect=AssertionError("member must not be opened"),
                ):
                    with self.assertRaisesRegex(
                        ExtractionError, "Unsafe ZIP member path"
                    ):
                        extract_document_text(p)

    def test_deeply_nested_xml_is_rejected(self):
        from tools import read_extract

        depth = read_extract.MAX_OOXML_XML_DEPTH + 1
        p = os.path.join(self.tmp, "deep.docx")
        _write_docx(
            p,
            self._doc("<w:r>" * depth + "<w:t>x</w:t>" + "</w:r>" * depth),
        )
        with self.assertRaisesRegex(ExtractionError, "maximum nesting depth"):
            extract_document_text(p)

    def test_document_element_bound_returns_bounded_body_prefix(self):
        from tools import read_extract

        p = os.path.join(self.tmp, "many-empty-elements.docx")
        _write_docx(
            p,
            self._doc(
                "<w:p><w:r><w:t>start</w:t></w:r>"
                + "<w:r/>" * 2_000
                + "</w:p>"
            ),
        )
        with mock.patch.object(read_extract, "MAX_DOCX_XML_ELEMENTS", 1_000):
            text = extract_document_text(p)

        self.assertIn("start", text)
        self.assertIn("Extraction truncated", text)
        self.assertIn("1,000-element DOCX body safety limit", text)

    def test_encrypted_member_is_rejected_before_member_read(self):
        p = os.path.join(self.tmp, "encrypted.docx")
        _write_docx(p, self._doc("<w:p><w:r><w:t>secret</w:t></w:r></w:p>"))
        _set_encrypted_flag(p)
        with mock.patch.object(
            zipfile.ZipFile,
            "open",
            side_effect=AssertionError("member must not be opened"),
        ):
            with self.assertRaisesRegex(ExtractionError, "Encrypted ZIP member"):
                extract_document_text(p)

    def test_docx_extracted_text_is_bounded(self):
        from tools import read_extract

        p = os.path.join(self.tmp, "long.docx")
        _write_docx(
            p,
            self._doc(
                "<w:p><w:r><w:t>" + "x" * 2048 + "</w:t></w:r></w:p>"
            ),
        )
        with mock.patch.object(
            read_extract, "MAX_EXTRACTED_TEXT_CHARS", 256
        ):
            text = extract_document_text(p)
        self.assertLessEqual(len(text), 256)
        self.assertIn("Extraction truncated", text)


# ---------------------------------------------------------------------------
# Excel workbooks (.xlsx) — #10740
# ---------------------------------------------------------------------------

class TestXlsxExtraction(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="rex_xlsx_")

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _build(self, path, *, include_hidden=True):
        r = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
        hidden_sheet = (f'<sheet name="Hidden" sheetId="2" state="hidden" '
                        f'xmlns:r="{r}" r:id="rId2"/>') if include_hidden else ""
        workbook = (
            f'<workbook xmlns="{_NS_S}" xmlns:r="{r}"><sheets>'
            f'<sheet name="Data" sheetId="1" r:id="rId1"/>{hidden_sheet}'
            f'</sheets></workbook>')
        rels = (
            '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
            '<Relationship Id="rId1" Target="worksheets/sheet1.xml" Type="x"/>'
            '<Relationship Id="rId2" Target="worksheets/sheet2.xml" Type="x"/>'
            '</Relationships>')
        shared = (f'<sst xmlns="{_NS_S}"><si><t>Name</t></si><si><t>Score</t></si>'
                  f'<si><t>Alice</t></si></sst>')
        sheet1 = (
            f'<worksheet xmlns="{_NS_S}"><sheetData>'
            '<row r="1"><c r="A1" t="s"><v>0</v></c><c r="B1" t="s"><v>1</v></c></row>'
            '<row r="2"><c r="A2" t="s"><v>2</v></c><c r="B2"><v>95</v></c></row>'
            '</sheetData></worksheet>')
        sheet2 = (f'<worksheet xmlns="{_NS_S}"><sheetData>'
                  '<row r="1"><c r="A1" t="str"><v>SECRETDATA</v></c></row>'
                  '</sheetData></worksheet>')
        _write_xlsx(path, workbook=workbook, rels=rels, shared=shared,
                    sheets={"xl/worksheets/sheet1.xml": sheet1,
                            "xl/worksheets/sheet2.xml": sheet2})

    def test_visible_sheet_content(self):
        p = os.path.join(self.tmp, "wb.xlsx")
        self._build(p)
        text = extract_document_text(p)
        self.assertIn("Data", text)        # sheet label
        self.assertIn("up to 5,000 rows and the first 256 columns", text)
        self.assertIn("Name\tScore", text)  # shared-string header row
        self.assertIn("Alice\t95", text)    # string + numeric cells

    def test_strict_ooxml_spreadsheet_and_relationship_namespaces_extract(self):
        strict_s = "http://purl.oclc.org/ooxml/spreadsheetml/main"
        strict_r = "http://purl.oclc.org/ooxml/officeDocument/relationships"
        strict_pkg_r = "http://purl.oclc.org/ooxml/package/relationships"
        p = os.path.join(self.tmp, "strict.xlsx")
        workbook = (
            f'<workbook xmlns="{strict_s}" xmlns:r="{strict_r}"><sheets>'
            '<sheet name="Strict Sheet" sheetId="1" r:id="rId1"/>'
            "</sheets></workbook>"
        )
        rels = (
            f'<Relationships xmlns="{strict_pkg_r}">'
            '<Relationship Id="rId1" Target="worksheets/sheet1.xml" Type="x"/>'
            "</Relationships>"
        )
        shared = (
            f'<sst xmlns="{strict_s}"><si><t>Strict shared text</t></si></sst>'
        )
        sheet = (
            f'<worksheet xmlns="{strict_s}"><sheetData><row r="1">'
            '<c r="A1" t="s"><v>0</v></c>'
            "</row></sheetData></worksheet>"
        )
        _write_xlsx(
            p,
            workbook=workbook,
            rels=rels,
            shared=shared,
            sheets={"xl/worksheets/sheet1.xml": sheet},
        )

        text = extract_document_text(p)

        self.assertIn("Strict Sheet", text)
        self.assertIn("Strict shared text", text)

    def test_generic_xml_event_budget_accumulates_across_xlsx_members(self):
        """Foreign elements cannot reset the CPU budget at each XML part."""
        from tools import read_extract

        p = os.path.join(self.tmp, "foreign-element-stress.xlsx")
        workbook, rels, sheet = self._single_sheet_parts()
        foreign = '<f:noise xmlns:f="urn:foreign"/>' * 4
        workbook = workbook.replace("<sheets>", foreign + "<sheets>")
        shared = (
            f'<sst xmlns="{_NS_S}">{foreign}<si><t>value</t></si></sst>'
        )
        _write_xlsx(
            p,
            workbook=workbook,
            rels=rels,
            shared=shared,
            sheets={"xl/worksheets/sheet1.xml": sheet},
        )

        # Each member is below 30 events on its own. Only one extraction-wide
        # budget catches the cumulative foreign-element work.
        with mock.patch.object(read_extract, "MAX_OOXML_XML_EVENTS", 30):
            with self.assertRaisesRegex(
                ExtractionError,
                "XLSX XML exceeds the extraction-wide budget of 30 events",
            ):
                extract_document_text(p)

    def test_row_beyond_extraction_view_is_disclosed(self):
        p = os.path.join(self.tmp, "beyond-row-view.xlsx")
        workbook, rels, _ = self._single_sheet_parts()
        rows = "".join(
            f'<row r="{row_number}"/>'
            for row_number in range(1, 5_001)
        )
        rows += (
            '<row r="5001"><c r="A5001" t="str">'
            '<v>OUTSIDE_ROW_VIEW</v></c></row>'
        )
        sheet = (
            f'<worksheet xmlns="{_NS_S}"><sheetData>'
            f"{rows}</sheetData></worksheet>"
        )
        _write_xlsx(
            p,
            workbook=workbook,
            rels=rels,
            shared=None,
            sheets={"xl/worksheets/sheet1.xml": sheet},
        )

        text = extract_document_text(p)

        self.assertIn("up to 5,000 rows and the first 256 columns", text)
        self.assertIn("cells outside this view are not included", text)
        self.assertNotIn("OUTSIDE_ROW_VIEW", text)

    def test_column_beyond_extraction_view_is_disclosed(self):
        p = os.path.join(self.tmp, "beyond-column-view.xlsx")
        workbook, rels, _ = self._single_sheet_parts()
        sheet = (
            f'<worksheet xmlns="{_NS_S}"><sheetData><row r="1">'
            '<c r="A1" t="str"><v>INSIDE_COLUMN_VIEW</v></c>'
            '<c r="IW1" t="str"><v>OUTSIDE_COLUMN_VIEW</v></c>'
            "</row></sheetData></worksheet>"
        )
        _write_xlsx(
            p,
            workbook=workbook,
            rels=rels,
            shared=None,
            sheets={"xl/worksheets/sheet1.xml": sheet},
        )

        text = extract_document_text(p)

        self.assertIn("up to 5,000 rows and the first 256 columns", text)
        self.assertIn("cells outside this view are not included", text)
        self.assertIn("INSIDE_COLUMN_VIEW", text)
        self.assertNotIn("OUTSIDE_COLUMN_VIEW", text)


    def test_not_a_zip_raises(self):
        p = os.path.join(self.tmp, "bad.xlsx")
        with open(p, "wb") as fh:
            fh.write(b"nope")
        with self.assertRaises(ExtractionError):
            extract_document_text(p)

    def test_malformed_shared_strings_fails_contextually(self):
        p = os.path.join(self.tmp, "malformed-shared.xlsx")
        workbook, rels, sheet = self._single_sheet_parts()
        _write_xlsx(
            p,
            workbook=workbook,
            rels=rels,
            shared=f'<sst xmlns="{_NS_S}"><si><t>broken',
            sheets={"xl/worksheets/sheet1.xml": sheet},
        )
        with self.assertRaisesRegex(
            ExtractionError, "Malformed XML in xl/sharedStrings.xml"
        ):
            extract_document_text(p)

    def test_malformed_workbook_relationships_fails_contextually(self):
        p = os.path.join(self.tmp, "malformed-rels.xlsx")
        workbook, _, sheet = self._single_sheet_parts()
        _write_xlsx(
            p,
            workbook=workbook,
            rels=(
                '<Relationships xmlns="http://schemas.openxmlformats.org/'
                'package/2006/relationships"><Relationship'
            ),
            shared=None,
            sheets={"xl/worksheets/sheet1.xml": sheet},
        )
        with self.assertRaisesRegex(
            ExtractionError, "Malformed XML in xl/_rels/workbook.xml.rels"
        ):
            extract_document_text(p)

    def test_malformed_only_visible_sheet_fails_contextually(self):
        p = os.path.join(self.tmp, "malformed-sheet.xlsx")
        workbook, rels, _ = self._single_sheet_parts()
        _write_xlsx(
            p,
            workbook=workbook,
            rels=rels,
            shared=None,
            sheets={
                "xl/worksheets/sheet1.xml": (
                    f'<worksheet xmlns="{_NS_S}"><sheetData><row>'
                )
            },
        )
        with self.assertRaisesRegex(
            ExtractionError, "Malformed XML in xl/worksheets/sheet1.xml"
        ):
            extract_document_text(p)

    def test_malformed_visible_sheet_prevents_partial_workbook_result(self):
        relationship_ns = (
            "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
        )
        p = os.path.join(self.tmp, "partial-sheet.xlsx")
        workbook = (
            f'<workbook xmlns="{_NS_S}" xmlns:r="{relationship_ns}"><sheets>'
            '<sheet name="Good" sheetId="1" r:id="rId1"/>'
            '<sheet name="Broken" sheetId="2" r:id="rId2"/>'
            "</sheets></workbook>"
        )
        rels = (
            '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
            '<Relationship Id="rId1" Target="worksheets/sheet1.xml" Type="x"/>'
            '<Relationship Id="rId2" Target="worksheets/sheet2.xml" Type="x"/>'
            "</Relationships>"
        )
        valid_sheet = (
            f'<worksheet xmlns="{_NS_S}"><sheetData><row>'
            '<c t="str"><v>valid content</v></c>'
            "</row></sheetData></worksheet>"
        )
        malformed_sheet = f'<worksheet xmlns="{_NS_S}"><sheetData><row>'
        _write_xlsx(
            p,
            workbook=workbook,
            rels=rels,
            shared=None,
            sheets={
                "xl/worksheets/sheet1.xml": valid_sheet,
                "xl/worksheets/sheet2.xml": malformed_sheet,
            },
        )
        with self.assertRaisesRegex(
            ExtractionError, "sheet2.xml.*Broken"
        ):
            extract_document_text(p)

    def test_invalid_shared_string_reference_fails_contextually(self):
        for index, invalid_ref in enumerate(("not-an-index", "7", "-1")):
            with self.subTest(shared_string_ref=invalid_ref):
                p = os.path.join(self.tmp, f"invalid-shared-ref-{index}.xlsx")
                workbook, rels, _ = self._single_sheet_parts()
                shared = f'<sst xmlns="{_NS_S}"><si><t>only</t></si></sst>'
                sheet = (
                    f'<worksheet xmlns="{_NS_S}"><sheetData><row r="1">'
                    f'<c r="A1" t="s"><v>{invalid_ref}</v></c>'
                    "</row></sheetData></worksheet>"
                )
                _write_xlsx(
                    p,
                    workbook=workbook,
                    rels=rels,
                    shared=shared,
                    sheets={"xl/worksheets/sheet1.xml": sheet},
                )

                with self.assertRaisesRegex(
                    ExtractionError,
                    "Invalid XLSX shared-string index.*sheet1.xml cell A1",
                ):
                    extract_document_text(p)

    def test_invalid_boolean_cell_fails_instead_of_becoming_false(self):
        p = os.path.join(self.tmp, "invalid-boolean.xlsx")
        workbook, rels, _ = self._single_sheet_parts()
        sheet = (
            f'<worksheet xmlns="{_NS_S}"><sheetData><row r="1">'
            '<c r="B1" t="b"><v>not-a-boolean</v></c>'
            "</row></sheetData></worksheet>"
        )
        _write_xlsx(
            p,
            workbook=workbook,
            rels=rels,
            shared=None,
            sheets={"xl/worksheets/sheet1.xml": sheet},
        )

        with self.assertRaisesRegex(
            ExtractionError,
            "Invalid XLSX boolean value.*sheet1.xml cell B1",
        ):
            extract_document_text(p)

    def test_missing_visible_sheet_relationship_prevents_partial_result(self):
        relationship_ns = (
            "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
        )
        p = os.path.join(self.tmp, "missing-visible-relationship.xlsx")
        workbook = (
            f'<workbook xmlns="{_NS_S}" xmlns:r="{relationship_ns}"><sheets>'
            '<sheet name="Good" sheetId="1" r:id="rId1"/>'
            '<sheet name="Missing" sheetId="2" r:id="rId2"/>'
            "</sheets></workbook>"
        )
        rels = (
            '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
            '<Relationship Id="rId1" Target="worksheets/sheet1.xml" Type="x"/>'
            "</Relationships>"
        )
        good_sheet = (
            f'<worksheet xmlns="{_NS_S}"><sheetData><row>'
            '<c t="str"><v>GOOD_CONTENT_MUST_NOT_RETURN_PARTIALLY</v></c>'
            "</row></sheetData></worksheet>"
        )
        _write_xlsx(
            p,
            workbook=workbook,
            rels=rels,
            shared=None,
            sheets={"xl/worksheets/sheet1.xml": good_sheet},
        )

        with self.assertRaisesRegex(
            ExtractionError, "sheet 'Missing'.*relationship 'rId2'"
        ):
            extract_document_text(p)

    def test_visible_sheet_with_empty_relationship_target_fails(self):
        p = os.path.join(self.tmp, "empty-sheet-target.xlsx")
        workbook, _, _ = self._single_sheet_parts()
        rels = (
            '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
            '<Relationship Id="rId1" Target="" Type="x"/>'
            "</Relationships>"
        )
        _write_xlsx(
            p,
            workbook=workbook,
            rels=rels,
            shared=None,
            sheets={},
        )

        with self.assertRaisesRegex(
            ExtractionError, "sheet 'Data'.*does not target a worksheet"
        ):
            extract_document_text(p)

    def test_visible_sheet_with_missing_worksheet_part_fails(self):
        p = os.path.join(self.tmp, "missing-sheet-part.xlsx")
        workbook, _, _ = self._single_sheet_parts()
        rels = (
            '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
            '<Relationship Id="rId1" Target="worksheets/missing.xml" Type="x"/>'
            "</Relationships>"
        )
        _write_xlsx(
            p,
            workbook=workbook,
            rels=rels,
            shared=None,
            sheets={},
        )

        with self.assertRaisesRegex(
            ExtractionError, "sheet 'Data'.*worksheet part is missing.*missing.xml"
        ):
            extract_document_text(p)

    def test_duplicate_nonempty_relationship_ids_are_rejected(self):
        p = os.path.join(self.tmp, "duplicate-relationship-id.xlsx")
        workbook, _, sheet = self._single_sheet_parts()
        rels = (
            '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
            '<Relationship Id="rId1" Target="worksheets/sheet1.xml" Type="x"/>'
            '<Relationship Id="rId1" Target="worksheets/sheet2.xml" Type="x"/>'
            "</Relationships>"
        )
        _write_xlsx(
            p,
            workbook=workbook,
            rels=rels,
            shared=None,
            sheets={"xl/worksheets/sheet1.xml": sheet},
        )

        with self.assertRaisesRegex(
            ExtractionError, "Duplicate XLSX workbook relationship Id: 'rId1'"
        ):
            extract_document_text(p)

    def test_oversized_shared_strings_rejected_before_member_read(self):
        from tools import read_extract

        p = os.path.join(self.tmp, "oversized-shared.xlsx")
        workbook, rels, sheet = self._single_sheet_parts()
        _write_xlsx(
            p,
            workbook=workbook,
            rels=rels,
            shared=f'<sst xmlns="{_NS_S}"><si><t>{"x" * 512}</t></si></sst>',
            sheets={"xl/worksheets/sheet1.xml": sheet},
        )
        with mock.patch.object(read_extract, "MAX_OOXML_MEMBER_BYTES", 256), \
                mock.patch.object(
                    zipfile.ZipFile,
                    "open",
                    side_effect=AssertionError("member must not be opened"),
                ):
            with self.assertRaisesRegex(
                ExtractionError, "xl/sharedStrings.xml.*uncompressed"
            ):
                extract_document_text(p)

    def test_oversized_worksheet_rejected_before_member_read(self):
        from tools import read_extract

        p = os.path.join(self.tmp, "oversized-sheet.xlsx")
        workbook, rels, _ = self._single_sheet_parts()
        sheet = (
            f'<worksheet xmlns="{_NS_S}"><sheetData><row><c t="str"><v>'
            + "x" * 512
            + "</v></c></row></sheetData></worksheet>"
        )
        _write_xlsx(
            p,
            workbook=workbook,
            rels=rels,
            shared=None,
            sheets={"xl/worksheets/sheet1.xml": sheet},
        )
        with mock.patch.object(read_extract, "MAX_OOXML_MEMBER_BYTES", 256), \
                mock.patch.object(
                    zipfile.ZipFile,
                    "open",
                    side_effect=AssertionError("member must not be opened"),
                ):
            with self.assertRaisesRegex(
                ExtractionError, "xl/worksheets/sheet1.xml.*uncompressed"
            ):
                extract_document_text(p)

    def test_aggregate_relevant_member_size_is_bounded(self):
        from tools import read_extract

        p = os.path.join(self.tmp, "aggregate.xlsx")
        workbook, rels, sheet = self._single_sheet_parts()
        _write_xlsx(
            p,
            workbook=workbook,
            rels=rels,
            shared=f'<sst xmlns="{_NS_S}"><si><t>Name</t></si></sst>',
            sheets={"xl/worksheets/sheet1.xml": sheet},
        )
        with mock.patch.object(read_extract, "MAX_OOXML_MEMBER_BYTES", 4096), \
                mock.patch.object(read_extract, "MAX_OOXML_RELEVANT_BYTES", 128), \
                mock.patch.object(
                    zipfile.ZipFile,
                    "open",
                    side_effect=AssertionError("member must not be opened"),
                ):
            with self.assertRaisesRegex(
                ExtractionError, "relevant XML expands to.*limit"
            ):
                extract_document_text(p)

    def test_xlsx_zip_bomb_rejected_before_member_read(self):
        from tools import read_extract

        p = os.path.join(self.tmp, "bomb.xlsx")
        workbook, rels, _ = self._single_sheet_parts()
        payload = "A" * (read_extract.MIN_ZIP_RATIO_MEMBER_BYTES + 128_000)
        sheet = (
            f'<worksheet xmlns="{_NS_S}"><sheetData><row><c t="str"><v>'
            f"{payload}</v></c></row></sheetData></worksheet>"
        )
        with zipfile.ZipFile(p, "w", compression=zipfile.ZIP_DEFLATED) as z:
            z.writestr("xl/workbook.xml", workbook)
            z.writestr("xl/_rels/workbook.xml.rels", rels)
            z.writestr("xl/worksheets/sheet1.xml", sheet)
        with mock.patch.object(
            zipfile.ZipFile,
            "open",
            side_effect=AssertionError("member must not be opened"),
        ):
            with self.assertRaisesRegex(ExtractionError, "compression ratio"):
                extract_document_text(p)

    def test_many_sheets_are_rejected_during_streaming_parse(self):
        from tools import read_extract

        p = os.path.join(self.tmp, "many-sheets.xlsx")
        r = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
        sheets = "".join(
            f'<sheet name="S{i}" sheetId="{i}" xmlns:r="{r}" r:id="rId{i}"/>'
            for i in range(1, 4)
        )
        workbook = f'<workbook xmlns="{_NS_S}"><sheets>{sheets}</sheets></workbook>'
        rels = (
            '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
            + "".join(
                f'<Relationship Id="rId{i}" Target="worksheets/sheet{i}.xml" Type="x"/>'
                for i in range(1, 4)
            )
            + "</Relationships>"
        )
        worksheet = f'<worksheet xmlns="{_NS_S}"><sheetData/></worksheet>'
        _write_xlsx(
            p,
            workbook=workbook,
            rels=rels,
            shared=None,
            sheets={f"xl/worksheets/sheet{i}.xml": worksheet for i in range(1, 4)},
        )
        with mock.patch.object(read_extract, "MAX_XLSX_SHEETS", 2):
            with self.assertRaisesRegex(ExtractionError, "more than 2 sheets"):
                extract_document_text(p)

    def test_relationships_without_ids_still_count_toward_limit(self):
        from tools import read_extract

        p = os.path.join(self.tmp, "many-duplicate-rels.xlsx")
        workbook, _, sheet = self._single_sheet_parts()
        rels = (
            '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
            + '<Relationship Target="worksheets/sheet1.xml" Type="x"/>'
            * 5
            + "</Relationships>"
        )
        _write_xlsx(
            p,
            workbook=workbook,
            rels=rels,
            shared=None,
            sheets={"xl/worksheets/sheet1.xml": sheet},
        )
        with mock.patch.object(read_extract, "MAX_XLSX_SHEETS", 1):
            with self.assertRaisesRegex(ExtractionError, "too many.*relationships"):
                extract_document_text(p)

    def test_oversized_cell_count_in_one_row_is_rejected(self):
        from tools import read_extract

        p = os.path.join(self.tmp, "many-cells-one-row.xlsx")
        workbook, rels, _ = self._single_sheet_parts()
        cells = '<c t="str"><v>x</v></c>' * (
            read_extract.MAX_XLSX_CELLS_PER_ROW + 1
        )
        sheet = (
            f'<worksheet xmlns="{_NS_S}"><sheetData><row>'
            f"{cells}</row></sheetData></worksheet>"
        )
        _write_xlsx(
            p,
            workbook=workbook,
            rels=rels,
            shared=None,
            sheets={"xl/worksheets/sheet1.xml": sheet},
        )
        with self.assertRaisesRegex(ExtractionError, "more than 16,384 cells"):
            extract_document_text(p)

    def test_worksheet_at_member_limit_extracts(self):
        from tools import read_extract

        p = os.path.join(self.tmp, "at-limit.xlsx")
        workbook, rels, sheet = self._single_sheet_parts()
        sheet = sheet.replace("boundary", "boundary" + "x" * 256)
        self.assertGreater(len(sheet.encode("utf-8")), len(workbook.encode("utf-8")))
        _write_xlsx(
            p,
            workbook=workbook,
            rels=rels,
            shared=None,
            sheets={"xl/worksheets/sheet1.xml": sheet},
        )
        with mock.patch.object(
            read_extract, "MAX_OOXML_MEMBER_BYTES", len(sheet.encode("utf-8"))
        ):
            text = extract_document_text(p)
        self.assertIn("boundary", text)

    def test_xlsx_extracted_text_is_bounded(self):
        from tools import read_extract

        p = os.path.join(self.tmp, "long.xlsx")
        workbook, rels, _ = self._single_sheet_parts()
        sheet = (
            f'<worksheet xmlns="{_NS_S}"><sheetData><row><c t="str"><v>'
            + "x" * 2048
            + "</v></c></row></sheetData></worksheet>"
        )
        _write_xlsx(
            p,
            workbook=workbook,
            rels=rels,
            shared=None,
            sheets={"xl/worksheets/sheet1.xml": sheet},
        )
        with mock.patch.object(
            read_extract, "MAX_EXTRACTED_TEXT_CHARS", 256
        ):
            text = extract_document_text(p)
        self.assertLessEqual(len(text), 256)
        self.assertIn("Extraction truncated", text)

    @staticmethod
    def _single_sheet_parts():
        r = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
        workbook = (
            f'<workbook xmlns="{_NS_S}" xmlns:r="{r}"><sheets>'
            '<sheet name="Data" sheetId="1" r:id="rId1"/>'
            "</sheets></workbook>"
        )
        rels = (
            '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
            '<Relationship Id="rId1" Target="worksheets/sheet1.xml" Type="x"/>'
            "</Relationships>"
        )
        sheet = (
            f'<worksheet xmlns="{_NS_S}"><sheetData>'
            '<row r="1"><c r="A1" t="str"><v>boundary</v></c></row>'
            "</sheetData></worksheet>"
        )
        return workbook, rels, sheet


# ---------------------------------------------------------------------------
# read_file_tool integration
# ---------------------------------------------------------------------------

class TestReadFileToolIntegration(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="rex_int_")

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_notebook_read_is_line_numbered(self):
        p = os.path.join(self.tmp, "nb.ipynb")
        _write_notebook(p, [
            {"cell_type": "markdown", "source": "# H"},
            {"cell_type": "code", "source": "print(1)"},
        ])
        res = json.loads(read_file_tool(p))
        self.assertTrue(res.get("extracted_document"))
        self.assertIn("1|", res["content"])  # line-number gutter
        self.assertIn("print(1)", res["content"])


    def test_corrupt_docx_surfaces_extraction_error(self):
        p = os.path.join(self.tmp, "bad.docx")
        with open(p, "wb") as fh:
            fh.write(b"not a zip")
        res = json.loads(read_file_tool(p))
        # Should NOT crash; the binary guard fires but surfaces the
        # specific extraction failure instead of the generic message.
        self.assertIn("error", res)
        self.assertIn("extraction failed", res["error"].lower())
        self.assertIn("docx", res["error"].lower())

    def test_oversized_anydoc_read_surfaces_size_error(self):
        import tools.read_extract as rex

        saved_cap = rex.MAX_ANYDOC_BYTES
        saved_module = rex._anydoc_module

        class _FakeAnydoc:
            def to_markdown(self, path):  # pragma: no cover - must not be called
                raise AssertionError("conversion should be rejected before call")

        rex._anydoc_module = _FakeAnydoc()
        rex.MAX_ANYDOC_BYTES = 10
        try:
            p = os.path.join(self.tmp, "big.pdf")
            with open(p, "wb") as fh:
                fh.write(b"x" * 11)
            res = json.loads(read_file_tool(p))
            self.assertIn("error", res)
            self.assertIn("too large", res["error"].lower())
            # The size hint reaches the agent instead of a generic binary error.
            self.assertNotIn("cannot read binary file", res["error"].lower())
        finally:
            rex.MAX_ANYDOC_BYTES = saved_cap
            rex._anydoc_module = saved_module

    def test_unavailable_converter_falls_back_to_raw_read(self):
        import time

        import tools.read_extract as rex

        saved_module = rex._anydoc_module
        saved_failed_at = rex._anydoc_failed_at
        # Simulate "converter unavailable and in cooldown": _anydoc() returns
        # None, the .pdf is not treated as extractable, and read_file keeps
        # its historical raw-read fallthrough (no extraction error surfaced).
        rex._anydoc_module = None
        rex._anydoc_failed_at = time.monotonic()
        try:
            p = os.path.join(self.tmp, "doc.pdf")
            with open(p, "wb") as fh:
                fh.write(b"%PDF-1.4 fake")
            res = json.loads(read_file_tool(p))
            self.assertNotIn("error", res)
            self.assertIn("%PDF-1.4 fake", res.get("content", ""))
        finally:
            rex._anydoc_module = saved_module
            rex._anydoc_failed_at = saved_failed_at

    def test_docx_read_extracts(self):
        p = os.path.join(self.tmp, "d.docx")
        _write_docx(p, (f'<?xml version="1.0"?><w:document xmlns:w="{_NS_W}">'
                        '<w:body><w:p><w:r><w:t>Report body</w:t></w:r></w:p>'
                        '</w:body></w:document>'))
        res = json.loads(read_file_tool(p))
        self.assertTrue(res.get("extracted_document"))
        self.assertIn("Report body", res["content"])

    def test_backend_notebook_uses_notebook_transport_cap(self):
        from tools import file_tools, read_extract
        from tools.file_operations import ReadResult

        payload = json.dumps({
            "cells": [{"cell_type": "markdown", "source": "remote notebook"}],
            "metadata": {},
            "nbformat": 4,
            "nbformat_minor": 5,
        }).encode("utf-8")

        class FakeFileOps:
            def read_file_bytes(self, path, max_bytes=None):
                self.path = path
                self.max_bytes = max_bytes
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

        fake_ops = FakeFileOps()
        with mock.patch.object(file_tools, "_get_file_ops", return_value=fake_ops), \
                mock.patch.object(
                    file_tools,
                    "_resolve_path_for_task",
                    return_value=file_tools.PurePosixPath("/workspace/remote.ipynb"),
                ):
            res = json.loads(
                read_file_tool("/workspace/remote.ipynb", task_id="remote")
            )

        self.assertTrue(res.get("extracted_document"))
        self.assertIn("remote notebook", res["content"])
        self.assertEqual(fake_ops.max_bytes, read_extract.MAX_NOTEBOOK_BYTES)
        self.assertEqual(fake_ops.path, "/workspace/remote.ipynb")

    def test_oversized_notebook_transport_never_falls_through_to_raw_read(self):
        from tools import file_tools, read_extract
        from tools.file_operations import ReadResult

        class FakeFileOps:
            env = object()

            def read_file_bytes(self, path, max_bytes=None):
                self.max_bytes = max_bytes
                return ReadResult(
                    file_size=read_extract.MAX_NOTEBOOK_BYTES + 1,
                    error=(
                        "File is too large "
                        f"({read_extract.MAX_NOTEBOOK_BYTES + 1:,} bytes, "
                        f"limit is {read_extract.MAX_NOTEBOOK_BYTES:,})"
                    ),
                )

            def read_file(self, path, offset, limit):
                raise AssertionError("oversized notebook path must not be re-read")

        fake_ops = FakeFileOps()
        with mock.patch.object(file_tools, "_get_file_ops", return_value=fake_ops), \
                mock.patch.object(
                    file_tools,
                    "_resolve_path_for_task",
                    return_value=file_tools.PurePosixPath("/workspace/large.ipynb"),
                ):
            result = json.loads(
                read_file_tool("/workspace/large.ipynb", task_id="remote")
            )

        self.assertIn("error", result)
        self.assertIn("bounded notebook", result["error"])
        self.assertIn("too large", result["error"])
        self.assertEqual(fake_ops.max_bytes, read_extract.MAX_NOTEBOOK_BYTES)

    def test_invalid_binary_document_transport_is_actionable(self):
        from tools import file_tools
        from tools.file_operations import ReadResult

        class FakeFileOps:
            env = object()

            def read_file_bytes(self, path, max_bytes=None):
                return ReadResult(
                    base64_content="not valid base64!",
                    file_size=12,
                    is_binary=True,
                )

            def read_file(self, path, offset, limit):
                raise AssertionError("invalid transport must not fall through")

        with mock.patch.object(file_tools, "_get_file_ops", return_value=FakeFileOps()), \
                mock.patch.object(
                    file_tools,
                    "_resolve_path_for_task",
                    return_value=file_tools.PurePosixPath("/workspace/broken.docx"),
                ):
            result = json.loads(
                read_file_tool("/workspace/broken.docx", task_id="remote")
            )

        self.assertIn("error", result)
        self.assertIn("bounded document transport", result["error"])
        self.assertIn("invalid byte data", result["error"])
        self.assertNotIn("Cannot read binary file", result["error"])

    def test_malformed_notebook_uses_bounded_snapshot_for_raw_pagination(self):
        from tools import file_tools
        from tools.file_operations import ReadResult

        payload = b'{\n  "cells": [\n    BROKEN\n  ]\n}\n'

        class FakeFileOps:
            env = object()

            def read_file_bytes(self, path, max_bytes=None):
                return ReadResult(
                    base64_content=base64.b64encode(payload).decode("ascii"),
                    file_size=len(payload),
                    is_binary=False,
                )

            def read_file(self, path, offset, limit):
                raise AssertionError("malformed notebook path must not be re-read")

            @staticmethod
            def _add_line_numbers(content, start_line=1):
                return "\n".join(
                    f"{number}|{line}"
                    for number, line in enumerate(content.split("\n"), start_line)
                )

        with mock.patch.object(file_tools, "_get_file_ops", return_value=FakeFileOps()), \
                mock.patch.object(
                    file_tools,
                    "_resolve_path_for_task",
                    return_value=file_tools.PurePosixPath("/workspace/broken.ipynb"),
                ):
            result = json.loads(
                read_file_tool(
                    "/workspace/broken.ipynb",
                    offset=2,
                    limit=2,
                    task_id="remote",
                )
            )

        self.assertTrue(result.get("raw_notebook_fallback"))
        self.assertFalse(result.get("extracted_document", False))
        self.assertIn('2|  "cells": [', result["content"])
        self.assertIn("3|    BROKEN", result["content"])
        self.assertTrue(result["truncated"])
        self.assertIn("offset=4", result["hint"])
        self.assertIn("bounded raw snapshot", result["note"])

    def test_semantically_unreadable_notebooks_use_bounded_raw_fallback(self):
        cases = (
            ("root", [], "Notebook root is not an object"),
            (
                "no-cells",
                {"metadata": {}, "nbformat": 4},
                "Notebook contains no cells",
            ),
            (
                "no-readable-cells",
                {
                    "cells": [{}],
                    "metadata": {},
                    "nbformat": 4,
                    "nbformat_minor": 5,
                },
                "Notebook contains no readable cells",
            ),
        )

        for name, notebook, reason in cases:
            with self.subTest(case=name):
                path = os.path.join(self.tmp, f"{name}.ipynb")
                with open(path, "w", encoding="utf-8") as fh:
                    json.dump(notebook, fh)

                result = json.loads(
                    read_file_tool(path, task_id=f"semantic-fallback-{name}")
                )

                self.assertNotIn("error", result)
                self.assertTrue(result.get("raw_notebook_fallback"))
                self.assertFalse(result.get("extracted_document", False))
                self.assertIn(reason, result["note"])
                self.assertIn("bounded raw snapshot", result["note"])
                self.assertTrue(result["content"])

    def test_busy_native_reader_rejects_before_backend_transport(self):
        from tools import file_tools, read_extract

        class FakeFileOps:
            env = object()

            def read_file_bytes(self, path, max_bytes=None):
                raise AssertionError("saturated reads must not start transport")

        slots = mock.Mock()
        slots.acquire.return_value = False
        with mock.patch.object(file_tools, "_get_file_ops", return_value=FakeFileOps()), \
                mock.patch.object(
                    file_tools,
                    "_resolve_path_for_task",
                    return_value=file_tools.PurePosixPath("/workspace/busy.ipynb"),
                ), mock.patch.object(read_extract, "_native_read_slots", slots):
            result = json.loads(
                read_file_tool("/workspace/busy.ipynb", task_id="remote")
            )

        self.assertIn("error", result)
        self.assertIn("Native document extraction is busy", result["error"])
        self.assertIn("retry this read shortly", result["error"])

    def test_denied_document_paths_are_blocked_before_byte_transport(self):
        from agent import file_safety
        from tools import file_tools, read_extract

        hermes_home = os.path.join(self.tmp, "hermes-home")
        blocked_paths = (
            os.path.join(hermes_home, "mcp-tokens", "secret.ipynb"),
            os.path.join(hermes_home, "skills", ".hub", "payload.docx"),
        )

        class FakeFileOps:
            env = object()

            def read_file_bytes(self, path, max_bytes=None):
                raise AssertionError("denied path must not start byte transport")

        for blocked_path in blocked_paths:
            with self.subTest(path=blocked_path):
                os.makedirs(os.path.dirname(blocked_path), exist_ok=True)
                with open(blocked_path, "wb") as fh:
                    fh.write(b"blocked")
                with mock.patch.object(
                    file_safety, "_hermes_home_path", return_value=Path(hermes_home)
                ), mock.patch.object(
                    file_safety, "_hermes_root_path", return_value=Path(hermes_home)
                ), mock.patch.object(
                    file_tools, "_get_file_ops", return_value=FakeFileOps()
                ), mock.patch.object(
                    file_tools,
                    "_resolve_path_for_task",
                    return_value=file_tools.PurePosixPath(blocked_path),
                ), mock.patch.object(read_extract, "extract_document_bytes") as extract:
                    result = json.loads(
                        read_file_tool(blocked_path, task_id="remote")
                    )

                self.assertIn("error", result)
                self.assertIn("Access denied", result["error"])
                extract.assert_not_called()

    def test_pdf_coverage_description_matches_prepended_warning(self):
        from tools.file_tools import READ_FILE_SCHEMA

        description = READ_FILE_SCHEMA["description"]
        self.assertIn("begins with an EXTRACTION COVERAGE WARNING", description)
        self.assertNotIn("ends with an EXTRACTION COVERAGE WARNING", description)

    def test_backend_only_anydoc_path_uses_transferred_bytes(self):
        from tools import file_tools, read_extract
        from tools.file_operations import ReadResult

        payload = br"{\rtf1\ansi Remote body\par}"

        class FakeAnydoc:
            def to_markdown_bytes(self, data):
                self.seen = data
                return "Remote body\n"

        class FakeFileOps:
            def read_file_bytes(self, path, max_bytes=None):
                self.path = path
                return ReadResult(
                    base64_content=base64.b64encode(payload).decode("ascii"),
                    file_size=len(payload),
                    is_binary=True,
                )

            @staticmethod
            def _add_line_numbers(content, start_line=1):
                return "\n".join(
                    f"{number}|{line}"
                    for number, line in enumerate(content.split("\n"), start_line)
                )

        fake_anydoc = FakeAnydoc()
        fake_ops = FakeFileOps()
        saved_module = read_extract._anydoc_module
        read_extract._anydoc_module = fake_anydoc
        try:
            with mock.patch.object(file_tools, "_get_file_ops", return_value=fake_ops), \
                    mock.patch.object(
                        file_tools,
                        "_resolve_path_for_task",
                        return_value=file_tools.PurePosixPath("/workspace/remote.rtf"),
                    ), mock.patch("os.path.getsize", side_effect=AssertionError("host read")):
                res = json.loads(read_file_tool("/workspace/remote.rtf", task_id="remote"))
        finally:
            read_extract._anydoc_module = saved_module

        self.assertTrue(res.get("extracted_document"))
        self.assertIn("Remote body", res["content"])
        self.assertEqual(fake_anydoc.seen, payload)
        self.assertEqual(fake_ops.path, "/workspace/remote.rtf")


# ---------------------------------------------------------------------------
# Scanned-PDF coverage warning
# ---------------------------------------------------------------------------

class TestPdfCoverageNote(unittest.TestCase):
    """The coverage footer flags PDFs whose pages yielded no text."""

    def _note_with_counts(self, counts):
        """Drive _pdf_coverage_note with synthetic per-page texts whose
        stripped lengths equal ``counts``."""
        from tools import read_extract
        texts = None if counts is None else ["x" * n for n in counts]
        with mock.patch.object(read_extract, "_pdf_page_texts",
                               return_value=texts):
            return read_extract._pdf_coverage_note("/x/doc.pdf")

    def test_mostly_scanned_pdf_warns_with_page_ranges(self):
        # 3 text pages then 6 empty ones (scanned) — well past the ratio.
        note = self._note_with_counts([900, 800, 700, 0, 0, 3, 0, 0, 0])
        self.assertIn("EXTRACTION COVERAGE WARNING", note)
        self.assertIn("6 of 9 pages", note)
        self.assertIn("pages 4-9", note)        # contiguous empty gap
        self.assertIn("(6 pages)", note)        # gap size stated
        self.assertIn("vision_analyze", note)   # recovery path is named
        self.assertIn("ocr-and-documents", note)
        self.assertIn("do NOT OCR or render everything", note)

    def test_pdf_recovery_guidance_is_non_executable_for_arbitrary_path(self):
        from tools import read_extract

        display_path = "- report '$(touch owned)' ;\nnext line.pdf"
        with mock.patch.object(
            read_extract,
            "_pdf_page_texts",
            return_value=["Readable section heading", "", ""],
        ):
            note = read_extract._pdf_coverage_note(
                "/private/host-temp.pdf",
                display_path=display_path,
            )

        self.assertIn("EXTRACTION COVERAGE WARNING", note)
        self.assertIn(json.dumps(display_path, ensure_ascii=True), note)
        self.assertNotIn(display_path, note)
        self.assertNotIn("`pdftoppm ", note)
        self.assertNotIn("pdftoppm -jpeg", note)

    def test_gap_labels_carry_preceding_section_text(self):
        """Each gap is labeled with the last text page before it (usually
        a section divider), so the agent can pick which gaps to read."""
        from tools import read_extract
        texts = (
            ["Section One: Bylaws of the Corporation"] + [""] * 5
            + ["Section Two: Budget details here"] + [""] * 4
        )
        with mock.patch.object(read_extract, "_pdf_page_texts",
                               return_value=texts):
            note = read_extract._pdf_coverage_note("/x/doc.pdf")
        self.assertIn(
            'pages 2-6 (5 pages) — after "Section One: Bylaws of the Corporation" (p1)',
            note,
        )
        self.assertIn(
            'pages 8-11 (4 pages) — after "Section Two: Budget details here" (p7)',
            note,
        )

    def test_gap_map_caps_pathological_alternation(self):
        """Hundreds of alternating text/scan pages must not balloon the
        warning — gaps beyond the cap collapse to one summary line."""
        from tools import read_extract
        texts = []
        for i in range(60):  # 60 gaps of 1 page each
            texts.extend([f"Divider page number {i} with enough text", ""])
        with mock.patch.object(read_extract, "_pdf_page_texts",
                               return_value=texts):
            note = read_extract._pdf_coverage_note("/x/doc.pdf")
        gap_lines = [ln for ln in note.splitlines() if ln.startswith("  ")]
        self.assertEqual(
            len(gap_lines), read_extract.PDF_GAP_MAP_MAX_ENTRIES + 1
        )
        self.assertIn("more gaps", gap_lines[-1])
        self.assertIn("(40 pages)", gap_lines[-1])

    def test_full_text_pdf_is_silent(self):
        self.assertEqual(self._note_with_counts([500] * 20), "")

    def test_one_blank_page_is_tolerated(self):
        # A single separator/blank page in a text PDF should not warn.
        self.assertEqual(self._note_with_counts([500, 0, 500, 500]), "")

    def test_small_share_below_ratio_and_absolute_is_silent(self):
        # 3 empty of 40 (7.5% < 20%, and < absolute threshold of 10).
        counts = [400] * 37 + [0, 0, 0]
        self.assertEqual(self._note_with_counts(counts), "")

    def test_large_absolute_count_warns_even_below_ratio(self):
        # 12 empty of 100 (12% < 20% ratio) still warns: 12 lost pages
        # is real data loss regardless of document size.
        counts = [400] * 88 + [0] * 12
        note = self._note_with_counts(counts)
        self.assertIn("12 of 100 pages", note)

    def test_undeterminable_counts_are_silent(self):
        self.assertEqual(self._note_with_counts(None), "")
        self.assertEqual(self._note_with_counts([0]), "")  # single page

    def test_page_ranges_compact(self):
        from tools.read_extract import _page_ranges
        self.assertEqual(_page_ranges([2, 3, 4, 7, 9, 10]), "2-4, 7, 9-10")
        self.assertEqual(_page_ranges([5]), "5")

    def test_page_char_counts_missing_pdftotext(self):
        from tools import read_extract
        with mock.patch.object(read_extract.shutil, "which", return_value=None):
            self.assertIsNone(read_extract._pdf_page_char_counts("/x/doc.pdf"))

    def test_page_char_counts_parses_formfeeds(self):
        from tools import read_extract
        fake = mock.Mock(returncode=0, stdout=b"alpha beta\fgamma\f\f")
        with mock.patch.object(read_extract.shutil, "which",
                               return_value="/usr/bin/pdftotext"), \
             mock.patch.object(read_extract.subprocess, "run",
                               return_value=fake):
            counts = read_extract._pdf_page_char_counts("/x/doc.pdf")
        # Trailing empty segment after the final \f is dropped; the real
        # empty page between the two \f markers is preserved.
        self.assertEqual(counts, [len("alpha beta"), len("gamma"), 0])

    def test_extract_anydoc_prepends_note_for_pdf(self):
        """The warning leads the extracted text for .pdf inputs (a trailing
        footer would land on a page the model may never fetch)."""
        from tools import read_extract
        fake_mod = mock.Mock()
        fake_mod.to_markdown.return_value = "# Title\n\nBody"
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as fh:
            fh.write(b"%PDF-1.4 fake")
            p = fh.name
        try:
            with mock.patch.object(read_extract, "_anydoc",
                                   return_value=fake_mod), \
                 mock.patch.object(read_extract, "_pdf_coverage_note",
                                   return_value="[EXTRACTION COVERAGE WARNING: test]\n"):
                text = read_extract._extract_anydoc(p)
        finally:
            os.unlink(p)
        self.assertTrue(text.startswith("[EXTRACTION COVERAGE WARNING"))
        self.assertIn("# Title", text)

    def test_extract_anydoc_no_note_for_non_pdf(self):
        from tools import read_extract
        fake_mod = mock.Mock()
        fake_mod.to_markdown.return_value = "converted"
        with tempfile.NamedTemporaryFile(suffix=".rtf", delete=False) as fh:
            fh.write(b"{\\rtf1 fake}")
            p = fh.name
        try:
            with mock.patch.object(read_extract, "_anydoc",
                                   return_value=fake_mod), \
                 mock.patch.object(read_extract, "_pdf_coverage_note") as note:
                text = read_extract._extract_anydoc(p)
        finally:
            os.unlink(p)
        note.assert_not_called()
        self.assertEqual(text, "converted\n")

    def test_bytes_path_prepends_note_with_display_path(self):
        """Backend-transferred PDF bytes get the same warning, and the
        recovery command names the backend-visible path, not the host
        temp file the scan ran against."""
        from tools import read_extract
        fake_mod = mock.Mock()
        fake_mod.to_markdown_bytes.return_value = "# Title\n\nBody"
        seen = {}

        def fake_note(path, display_path=None):
            seen["scan_path"] = path
            seen["display_path"] = display_path
            return f"[EXTRACTION COVERAGE WARNING: test '{display_path}']\n"

        with mock.patch.object(read_extract, "_anydoc",
                               return_value=fake_mod), \
             mock.patch.object(read_extract, "_pdf_coverage_note",
                               side_effect=fake_note):
            text = read_extract._extract_anydoc_bytes(
                b"%PDF-1.4 fake", "/workspace/remote.pdf"
            )
        self.assertTrue(text.startswith("[EXTRACTION COVERAGE WARNING"))
        self.assertIn("/workspace/remote.pdf", text)
        self.assertEqual(seen["display_path"], "/workspace/remote.pdf")
        # The scanned file is a host temp materialization, already removed.
        self.assertNotEqual(seen["scan_path"], "/workspace/remote.pdf")
        self.assertFalse(os.path.exists(seen["scan_path"]))

    def test_bytes_path_no_note_for_non_pdf(self):
        from tools import read_extract
        fake_mod = mock.Mock()
        fake_mod.to_markdown_bytes.return_value = "converted"
        with mock.patch.object(read_extract, "_anydoc",
                               return_value=fake_mod), \
             mock.patch.object(read_extract,
                               "_pdf_coverage_note_from_bytes") as note:
            text = read_extract._extract_anydoc_bytes(
                b"{\\rtf1 fake}", "/workspace/remote.rtf"
            )
        note.assert_not_called()
        self.assertEqual(text, "converted\n")


if __name__ == "__main__":
    unittest.main()
