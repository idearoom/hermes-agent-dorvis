"""Stdlib document-to-text extraction for ``read_file``.

Supports Jupyter notebooks, DOCX, and XLSX without adding hard dependencies.
When the optional ``firecrawl-anydoc`` package is installed (``pip install
firecrawl-anydoc``, imports as ``anydoc``), coverage widens to legacy Office
(.doc/.ppt/.xls), OpenDocument, RTF, EPUB, and PDF — converted to Markdown by
its Rust core. The stdlib extractors remain authoritative for their three
formats so behavior is identical whether or not anydoc is present.
Malformed documents raise :class:`ExtractionError`. Callers surface binary
document failures; notebook JSON syntax failures and parsed notebooks with no
renderable cell structure may show only the same bounded byte snapshot as raw
text.
"""

from __future__ import annotations

import importlib
import json
import os
import posixpath
import re
import shutil
import struct
import subprocess
import tempfile
import threading
import time
import zipfile
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path
from typing import Any, BinaryIO, Callable, Iterator, Optional
from xml.etree import ElementTree as ET

__all__ = [
    "EXTRACTABLE_EXTENSIONS",
    "ExtractionBusyError",
    "ExtractionError",
    "NotebookFallbackError",
    "NotebookSyntaxError",
    "document_size_limit",
    "extract_document_bytes",
    "extract_document_text",
    "is_extractable_document",
    "native_extraction_slot",
]

EXTRACTABLE_EXTENSIONS = frozenset({".ipynb", ".docx", ".xlsx"})
# Formats handled only when the optional anydoc converter is installed.
ANYDOC_EXTENSIONS = frozenset({
    ".doc", ".docm",
    ".ppt", ".pps", ".pot", ".pptx", ".pptm", ".ppsx", ".ppsm",
    ".xls", ".xlsm", ".xlsb",
    ".odt", ".ods", ".odp",
    ".rtf", ".epub", ".pdf",
})
# Refuse to convert huge documents. anydoc loads the whole file through its
# Rust core with no streaming, and the read_file char budget only applies
# after conversion, so an unbounded input can pin a tool turn and spike RAM.
# Its returned Markdown is still materialized by the optional converter and is
# intentionally outside the native-reader guarantees below. Dorvis does not
# bake AnyDoc in this release; bounded converter output remains follow-up work
# under AE-226 rather than a behavior change in the upstream parity update.
MAX_ANYDOC_BYTES = 50 * 1024 * 1024
MAX_DOCUMENT_BYTES = 32 * 1024 * 1024
MAX_NOTEBOOK_BYTES = 8 * 1024 * 1024

# The always-on stdlib DOCX/XLSX/IPYNB readers run inside the long-lived
# gateway. A process-wide, fail-fast semaphore below limits native extraction
# independently of the gateway's broader API-run admission and delegation.
# The limits are sized for ten simultaneous document reads in an 8 GiB task
# while leaving most of the task for the agent, browser, and gateway itself.
# At the worst transport
# shape an OOXML request retains roughly three copies of its 32 MiB archive
# (backend/base64/decoded-temp), one streaming XML window, and bounded lookup /
# output state: comfortably below 256 MiB per request. An 8 MiB notebook stays
# below that same envelope with a 26x allowance for Python's parsed-JSON
# object overhead (measured adversarial empty-cell JSON peaks at ~24.2x).
MAX_CONCURRENT_NATIVE_READS = 10
MAX_NATIVE_READ_WORKING_SET_BYTES = 256 * 1024 * 1024

# OOXML ZIP metadata is validated before ZipFile constructs its ZipInfo list or
# opens any member. ZIP64 is unnecessary under the 32 MiB transport cap and is
# rejected so forged 64-bit sizes cannot bypass these checks.
MAX_OOXML_ZIP_ENTRIES = 4096
MAX_OOXML_CENTRAL_DIRECTORY_BYTES = 8 * 1024 * 1024
MAX_OOXML_MEMBER_BYTES = 16 * 1024 * 1024
MAX_OOXML_RELEVANT_BYTES = 64 * 1024 * 1024
MIN_ZIP_RATIO_MEMBER_BYTES = 8 * 1024 * 1024
MAX_ZIP_COMPRESSION_RATIO = 500.0
MAX_OOXML_XML_DEPTH = 256
# One extraction-wide event budget covers every XML member actually parsed.
# It is intentionally separate from the format-specific semantic limits below:
# foreign-namespace elements still consume parser CPU and must remain bounded.
MAX_OOXML_XML_EVENTS = 1_000_000

# Parsing/output state stays bounded after ZIP metadata passes. Two million
# characters is twenty normal read_file pages while consuming at most ~8 MiB
# for a wide Python Unicode representation. Excel's documented cell text limit
# is 32,767 characters; larger values are malformed rather than useful cells.
MAX_EXTRACTED_TEXT_CHARS = 2_000_000
MAX_DOCX_XML_ELEMENTS = 250_000
MAX_XLSX_SHEETS = 256
MAX_XLSX_SHARED_STRINGS = 100_000
MAX_XLSX_SHARED_STRING_CHARS = 2_000_000
MAX_XLSX_CELL_CHARS = 32_767
MAX_XLSX_CELLS_PER_ROW = 16_384
_XML_READ_CHUNK_BYTES = 64 * 1024
_MAX_XLSX_ROWS_PER_SHEET = 5000
_MAX_XLSX_COLS = 256
_XLSX_COVERAGE_NOTE = (
    "[Extraction view: up to 5,000 rows and the first 256 columns per sheet; "
    "cells outside this view are not included.]\n"
)

# Conservative retained-allocation proofs (not measured averages): three
# transport-sized copies cover terminal stdout/base64/decoded+temp overlap;
# XML text uses the four-byte Unicode worst case; parsed notebook JSON gets a
# 26x input allowance; output gets a second wide-string copy for final join.
# Notebook fallback-eligible failures take a separate bounded-raw path. Syntax
# failures never retain a parsed object; semantic failures clear their complete
# exception traceback before raw decoding, releasing the acyclic JSON graph.
# Six wide input-sized copies conservatively cover decode, pagination, page
# join, line numbering, redaction, and serialization. Keeping the mutually
# exclusive paths explicit prevents a parsed object and raw fallback from being
# counted as coexisting.
ESTIMATED_OOXML_WORKING_SET_BYTES = (
    3 * MAX_DOCUMENT_BYTES
    + 4 * MAX_OOXML_MEMBER_BYTES
    + 2 * 4 * MAX_EXTRACTED_TEXT_CHARS
    + 8 * MAX_XLSX_SHARED_STRING_CHARS
)
ESTIMATED_NOTEBOOK_PARSED_WORKING_SET_BYTES = (
    3 * MAX_NOTEBOOK_BYTES
    + 26 * MAX_NOTEBOOK_BYTES
    + 2 * 4 * MAX_EXTRACTED_TEXT_CHARS
)
ESTIMATED_NOTEBOOK_FALLBACK_WORKING_SET_BYTES = (
    3 * MAX_NOTEBOOK_BYTES
    + 6 * 4 * MAX_NOTEBOOK_BYTES
    + 2 * 4 * MAX_EXTRACTED_TEXT_CHARS
)
ESTIMATED_NOTEBOOK_WORKING_SET_BYTES = max(
    ESTIMATED_NOTEBOOK_PARSED_WORKING_SET_BYTES,
    ESTIMATED_NOTEBOOK_FALLBACK_WORKING_SET_BYTES,
)

_NS_W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
_NS_S = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
_NS_REL = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
_NS_PKG_REL = "http://schemas.openxmlformats.org/package/2006/relationships"
_NS_W_STRICT = "http://purl.oclc.org/ooxml/wordprocessingml/main"
_NS_S_STRICT = "http://purl.oclc.org/ooxml/spreadsheetml/main"
_NS_REL_STRICT = "http://purl.oclc.org/ooxml/officeDocument/relationships"
_NS_PKG_REL_STRICT = "http://purl.oclc.org/ooxml/package/relationships"
_NS_W_FAMILY = (_NS_W, _NS_W_STRICT)
_NS_S_FAMILY = (_NS_S, _NS_S_STRICT)
_NS_REL_FAMILY = (_NS_REL, _NS_REL_STRICT)
_NS_PKG_REL_FAMILY = (_NS_PKG_REL, _NS_PKG_REL_STRICT)


def _namespace_tags(namespaces: tuple[str, ...], local_name: str) -> frozenset[str]:
    return frozenset(f"{{{namespace}}}{local_name}" for namespace in namespaces)


class ExtractionError(Exception):
    """Raised when a supported-looking document cannot be rendered as text."""


class NotebookFallbackError(ExtractionError):
    """Notebook failure safe to expose through its bounded raw snapshot."""


class NotebookSyntaxError(NotebookFallbackError):
    """Raised before notebook JSON has produced a parsed object graph."""


class NotebookSemanticError(NotebookFallbackError):
    """Raised after parsing when no structured notebook view can be rendered."""


class ExtractionBusyError(ExtractionError):
    """Raised when the bounded native-reader capacity is already occupied."""


class _XmlEventBudgetExceeded(ExtractionError):
    """Internal signal that bounded OOXML content parsing must stop."""


_native_read_slots = threading.BoundedSemaphore(MAX_CONCURRENT_NATIVE_READS)
_native_read_depth: ContextVar[int] = ContextVar("native_read_depth", default=0)


@contextmanager
def native_extraction_slot(path: str) -> Iterator[None]:
    """Bound native document transport + parsing without queueing requests."""
    if Path(path).suffix.lower() not in EXTRACTABLE_EXTENSIONS:
        yield
        return

    depth = _native_read_depth.get()
    if depth:
        token = _native_read_depth.set(depth + 1)
        try:
            yield
        finally:
            _native_read_depth.reset(token)
        return

    if not _native_read_slots.acquire(blocking=False):
        raise ExtractionBusyError(
            "Native document extraction is busy "
            f"({MAX_CONCURRENT_NATIVE_READS} readers already active); "
            "retry this read shortly"
        )
    token = _native_read_depth.set(1)
    try:
        yield
    finally:
        _native_read_depth.reset(token)
        _native_read_slots.release()


def _extension(path: str) -> str:
    ext = Path(path).suffix.lower()
    if ext in EXTRACTABLE_EXTENSIONS:
        return ext
    if ext in ANYDOC_EXTENSIONS and _anydoc() is not None:
        return ext
    return ""


_ANYDOC_UNSET = object()
_anydoc_module: Any = _ANYDOC_UNSET
_anydoc_lock = threading.Lock()
# After a failed first load, wait this long before trying again. The attempt
# can shell out to pip, so retrying on every call would hammer the network
# in environments where the install can never succeed.
ANYDOC_RETRY_SECONDS = 300.0
_anydoc_failed_at: Optional[float] = None


def _anydoc_disabled() -> bool:
    """Whether the embedding runtime explicitly quarantines AnyDoc.

    Generic Hermes/fork images retain upstream's optional lazy-install path.
    A managed overlay can disable only this converter while preserving the
    existing allowlisted lazy-dependency contract for unrelated capabilities.
    Any non-empty value other than a conventional false token opts out.
    """
    value = os.environ.get("HERMES_DISABLE_ANYDOC")
    if value is not None:
        return value.strip().lower() not in {"", "0", "false", "no", "off"}
    try:
        from hermes_cli.config import load_config_readonly

        security = load_config_readonly().get("security") or {}
        return not bool(security.get("allow_anydoc", True))
    except Exception:
        # Preserve the upstream default if config is unavailable. The managed
        # runtime seal above remains a fail-closed pre-import override.
        return False


def _anydoc() -> Optional[Any]:
    """Lazily import the optional anydoc converter; None when unavailable.

    A failed load is retried after :data:`ANYDOC_RETRY_SECONDS` rather than
    disabling extraction for the rest of the process, so one transient
    failure (network blip, pip race) does not stick in long-lived workers.
    """
    global _anydoc_module, _anydoc_failed_at
    if _anydoc_disabled():
        return None
    if _anydoc_module is not _ANYDOC_UNSET:
        return _anydoc_module
    with _anydoc_lock:
        if _anydoc_module is not _ANYDOC_UNSET:
            return _anydoc_module
        if (
            _anydoc_failed_at is not None
            and time.monotonic() - _anydoc_failed_at < ANYDOC_RETRY_SECONDS
        ):
            return None
        try:
            from tools.lazy_deps import ensure as _lazy_ensure

            # prompt=False: read_file must never block on an install prompt.
            _lazy_ensure("tool.doc_extract", prompt=False)
        except Exception:
            _anydoc_failed_at = time.monotonic()
            return None
        try:
            _anydoc_module = importlib.import_module("anydoc")
        except Exception:  # ImportError or a broken native binding
            _anydoc_failed_at = time.monotonic()
            return None
        _anydoc_failed_at = None
    return _anydoc_module  # type: ignore[return-value]


def is_extractable_document(path: str) -> bool:
    return bool(_extension(path))


def document_size_limit(path: str) -> int:
    """Maximum transferred input bytes for the document's native parser."""
    ext = Path(path).suffix.lower()
    if ext == ".ipynb":
        return MAX_NOTEBOOK_BYTES
    if ext in ANYDOC_EXTENSIONS:
        return MAX_ANYDOC_BYTES
    return MAX_DOCUMENT_BYTES


def extract_document_text(path: str) -> str:
    ext = _extension(path)
    with native_extraction_slot(path):
        if ext == ".ipynb":
            return _extract_notebook(path)
        if ext == ".docx":
            return _extract_docx(path)
        if ext == ".xlsx":
            return _extract_xlsx(path)
        if ext in ANYDOC_EXTENSIONS:
            return _extract_anydoc(path)
    raise ExtractionError(f"Unsupported document type: {path!r}")


def extract_document_bytes(data: bytes, path: str) -> str:
    """Extract a document already fetched across a file backend boundary."""
    limit = document_size_limit(path)
    if len(data) > limit:
        label = "Notebook" if Path(path).suffix.lower() == ".ipynb" else "Document"
        raise ExtractionError(
            f"{label} too large to parse ({len(data):,} bytes, limit is {limit:,})"
        )
    ext = _extension(path)
    if ext in ANYDOC_EXTENSIONS:
        return _extract_anydoc_bytes(data, path)
    if ext not in EXTRACTABLE_EXTENSIONS:
        raise ExtractionError(f"Unsupported document type: {path!r}")

    with native_extraction_slot(path):
        # The stdlib extractors are path-oriented. Materialize backend bytes in
        # a private host temp file, then remove it even when parsing fails.
        temp_path = ""
        try:
            with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as fh:
                fh.write(data)
                temp_path = fh.name
            return extract_document_text(temp_path)
        finally:
            if temp_path:
                try:
                    os.unlink(temp_path)
                except OSError:
                    pass


def _extract_anydoc(path: str) -> str:
    mod = _anydoc()
    if mod is None:
        raise ExtractionError(f"Unsupported document type: {path!r}")
    try:
        size = os.path.getsize(path)
    except OSError as exc:
        raise ExtractionError(str(exc)) from exc
    if size > MAX_ANYDOC_BYTES:
        raise ExtractionError(
            f"Document too large to convert ({size:,} bytes, limit is {MAX_ANYDOC_BYTES:,})"
        )
    try:
        text = mod.to_markdown(path)
    except OSError as exc:
        raise ExtractionError(str(exc)) from exc
    except Exception as exc:
        # anydoc raises one ConvertError subclass per failure mode
        # (Unsupported, Malformed, Encrypted, ResourceLimit, MissingPart).
        # Any of them means "no meaningful text": fall back to the normal
        # path/binary handling rather than crash read_file.
        raise ExtractionError(f"{type(exc).__name__}: {exc}") from exc
    if not isinstance(text, str) or not text.strip():
        raise ExtractionError("Document contains no extractable text")
    text = text.rstrip("\n") + "\n"
    if Path(path).suffix.lower() == ".pdf":
        note = _pdf_coverage_note(path)
        if note:
            # Prepend: read_file paginates the extraction, so a footer on a
            # long document would sit on a page the model may never fetch.
            text = note + text
    return text


# ── Scanned-PDF coverage detection ──────────────────────────────────
#
# anydoc (like every text-layer extractor) returns nothing for scanned
# image pages and emits no image placeholders or page markers, so a
# mostly-scanned PDF converts "successfully" into a few headers with
# empty bodies — silent data loss the model cannot detect. Count per-page
# text via poppler's pdftotext (form-feed page separators) and append a
# loud footer when a meaningful share of pages yielded no text.

# A page with fewer extracted characters than this is considered empty.
PDF_EMPTY_PAGE_CHARS = 20
# Warn when at least this many pages are empty AND they exceed the ratio,
# or when the absolute count alone is overwhelming.
PDF_COVERAGE_MIN_EMPTY = 2
PDF_COVERAGE_MIN_RATIO = 0.2
PDF_COVERAGE_ABSOLUTE_EMPTY = 10
PDF_PAGE_SCAN_TIMEOUT = 20.0


def _pdf_page_texts(path: str) -> Optional[list[str]]:
    """Per-page extracted text, or None when undeterminable."""
    if shutil.which("pdftotext") is None:
        return None
    try:
        proc = subprocess.run(
            ["pdftotext", path, "-"],
            capture_output=True,
            timeout=PDF_PAGE_SCAN_TIMEOUT,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if proc.returncode != 0:
        return None
    pages = proc.stdout.decode("utf-8", errors="replace").split("\f")
    if pages and not pages[-1].strip():
        pages.pop()  # trailing form-feed artifact
    return pages or None


def _pdf_page_char_counts(path: str) -> Optional[list[int]]:
    """Per-page extracted-text char counts, or None when undeterminable."""
    pages = _pdf_page_texts(path)
    if pages is None:
        return None
    return [len(page.strip()) for page in pages]


def _page_ranges(pages: list[int]) -> str:
    """Compact 1-based range list, e.g. '2-29, 33-35, 42'."""
    parts = [f"{a}-{b}" if a != b else str(a) for a, b in _group_ranges(pages)]
    if len(parts) > 12:
        parts = parts[:12] + ["…"]
    return ", ".join(parts)


def _group_ranges(pages: list[int]) -> list[list[int]]:
    """Group sorted 1-based page numbers into [start, end] runs."""
    ranges: list[list[int]] = []
    for p in pages:
        if ranges and p == ranges[-1][1] + 1:
            ranges[-1][1] = p
        else:
            ranges.append([p, p])
    return ranges


# Cap the per-gap breakdown so a pathological PDF (hundreds of alternating
# text/scan pages) cannot balloon the warning. Ranges beyond the cap are
# summarized in one line.
PDF_GAP_MAP_MAX_ENTRIES = 20
_GAP_CONTEXT_CHARS = 60


def _gap_map(counts: list[int], texts: list[str], empty: list[int]) -> str:
    """Per-gap breakdown: each empty range labeled with the last text seen
    before it (usually a section divider/header page), so the agent can
    decide WHICH gaps it actually needs to read instead of OCRing all of
    them."""
    ranges = _group_ranges(empty)
    lines: list[str] = []
    for a, b in ranges[:PDF_GAP_MAP_MAX_ENTRIES]:
        label = ""
        # Walk back to the nearest preceding page with text.
        for prev in range(a - 2, -1, -1):
            if counts[prev] >= PDF_EMPTY_PAGE_CHARS:
                snippet = " ".join(texts[prev].split())[:_GAP_CONTEXT_CHARS]
                label = f' — after "{snippet}" (p{prev + 1})'
                break
        span = f"page {a}" if a == b else f"pages {a}-{b}"
        n = b - a + 1
        lines.append(f"  {span} ({n} page{'s' if n != 1 else ''}){label}")
    if len(ranges) > PDF_GAP_MAP_MAX_ENTRIES:
        rest = ranges[PDF_GAP_MAP_MAX_ENTRIES:]
        rest_pages = sum(b - a + 1 for a, b in rest)
        lines.append(f"  … {len(rest)} more gaps ({rest_pages} pages)")
    return "\n".join(lines)


def _pdf_coverage_note(path: str, display_path: Optional[str] = None) -> str:
    """A warning header when many PDF pages produced no text, else ''.

    ``path`` is the file scanned with pdftotext (may be a host temp file
    for backend-transferred bytes); ``display_path`` is the path shown in
    the recovery command — the one the agent's terminal can actually see.
    """
    texts = _pdf_page_texts(path)
    if not texts or len(texts) < 2:
        return ""
    counts = [len(page.strip()) for page in texts]
    empty = [i + 1 for i, n in enumerate(counts) if n < PDF_EMPTY_PAGE_CHARS]
    total = len(counts)
    if len(empty) < PDF_COVERAGE_MIN_EMPTY:
        return ""
    if (
        len(empty) / total < PDF_COVERAGE_MIN_RATIO
        and len(empty) < PDF_COVERAGE_ABSOLUTE_EMPTY
    ):
        return ""
    shown = display_path or path
    shown_json = json.dumps(shown, ensure_ascii=True)
    return (
        "[EXTRACTION COVERAGE WARNING: "
        f"{len(empty)} of {total} pages in this PDF yielded no text. "
        "Those pages are likely scanned images (or blank) — their content "
        "is MISSING from the extracted text below, even where section "
        "headers appear with empty bodies. Unreadable gaps, each labeled "
        "with the last text extracted before it:\n"
        f"{_gap_map(counts, texts, empty)}\n"
        "Decide which gaps you actually need — do NOT OCR or render "
        "everything. For the gaps that matter, use `pdftoppm` at 150 DPI "
        "with JPEG output and explicit first/last page arguments. Pass this "
        "JSON-escaped input path as data (never interpolate it into a shell "
        f"command): {shown_json}. Write the images under a private temporary "
        "directory, "
        "and inspect each image with the vision_analyze tool, or use the "
        "ocr-and-documents skill (marker-pdf) for bulk OCR of large "
        "ranges.]\n"
    )


def _extract_anydoc_bytes(data: bytes, path: str) -> str:
    mod = _anydoc()
    if mod is None:
        raise ExtractionError(f"Unsupported document type: {path!r}")
    if len(data) > MAX_ANYDOC_BYTES:
        raise ExtractionError(
            f"Document too large to convert ({len(data):,} bytes, limit is {MAX_ANYDOC_BYTES:,})"
        )
    try:
        text = mod.to_markdown_bytes(data)
    except Exception as exc:
        raise ExtractionError(f"{type(exc).__name__}: {exc}") from exc
    if not isinstance(text, str) or not text.strip():
        raise ExtractionError("Document contains no extractable text")
    text = text.rstrip("\n") + "\n"
    if Path(path).suffix.lower() == ".pdf":
        note = _pdf_coverage_note_from_bytes(data, path)
        if note:
            # Prepend: read_file paginates the extraction, so a footer on a
            # long document would sit on a page the model may never fetch.
            text = note + text
    return text


def _pdf_coverage_note_from_bytes(data: bytes, display_path: str) -> str:
    """Coverage note for backend-transferred PDF bytes.

    pdftotext is path-oriented, so materialize the bytes in a private host
    temp file for the scan; the recovery command still names
    ``display_path`` — the path the agent's terminal backend can see.
    """
    temp_path = ""
    try:
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as fh:
            fh.write(data)
            temp_path = fh.name
        return _pdf_coverage_note(temp_path, display_path=display_path)
    except OSError:
        return ""
    finally:
        if temp_path:
            try:
                os.unlink(temp_path)
            except OSError:
                pass


def _source_text(source) -> str:
    if isinstance(source, str):
        return source
    if isinstance(source, list):
        return "".join(item for item in source if isinstance(item, str))
    return ""


class _BoundedText:
    """Progressive extracted-text builder with a hard retained-char ceiling."""

    def __init__(self, document_type: str):
        self._document_type = document_type
        self._parts: list[str] = []
        self._chars = 0
        self.truncated = False
        self._truncation_reason: Optional[str] = None

    def mark_truncated(self, reason: str) -> None:
        """Stop content extraction and retain an accurate bounded-view note."""
        if not self.truncated:
            self.truncated = True
            self._truncation_reason = reason

    def append(self, text: str) -> bool:
        """Append up to the budget; return False once parsing should stop."""
        if self.truncated:
            return False
        remaining = MAX_EXTRACTED_TEXT_CHARS - self._chars
        if remaining <= 0:
            self.mark_truncated(
                f"the {MAX_EXTRACTED_TEXT_CHARS:,}-character safety limit"
            )
            return False
        if len(text) <= remaining:
            self._parts.append(text)
            self._chars += len(text)
            return True
        self._parts.append(text[:remaining])
        self._chars += remaining
        self.mark_truncated(
            f"the {MAX_EXTRACTED_TEXT_CHARS:,}-character safety limit"
        )
        return False

    def checkpoint(self) -> tuple[int, int, bool, Optional[str]]:
        return (
            len(self._parts),
            self._chars,
            self.truncated,
            self._truncation_reason,
        )

    def rollback(
        self,
        checkpoint: tuple[int, int, bool, Optional[str]],
    ) -> None:
        (
            part_count,
            self._chars,
            self.truncated,
            self._truncation_reason,
        ) = checkpoint
        del self._parts[part_count:]

    def render(self) -> str:
        text = "".join(self._parts)
        if self.truncated:
            reason = self._truncation_reason or (
                f"the {MAX_EXTRACTED_TEXT_CHARS:,}-character safety limit"
            )
            notice = (
                "\n[Extraction truncated at "
                f"{reason} for "
                f"{self._document_type}. Split or convert the document to "
                "inspect the remaining content.]\n"
            )
            keep = max(0, MAX_EXTRACTED_TEXT_CHARS - len(notice))
            text = text[:keep].rstrip("\n") + notice
            return text[:MAX_EXTRACTED_TEXT_CHARS]
        text = text.rstrip("\n")
        if len(text) >= MAX_EXTRACTED_TEXT_CHARS:
            return text[:MAX_EXTRACTED_TEXT_CHARS]
        return text + "\n"


def _human_size(n_bytes: int) -> str:
    return f"{round(n_bytes / 1024)} KB" if n_bytes >= 1024 else f"{n_bytes} B"


def _base64_bytes(payload: str) -> int:
    """Approximate decoded size of a base64 payload (whitespace ignored)."""
    clean = re.sub(r"[^0-9+/=A-Za-z]", "", payload)
    padding = min(2, len(clean) - len(clean.rstrip("=")))
    return max(0, (len(clean) * 3) // 4 - padding)


def _clean_stream_text(text: str) -> str:
    """Strip ANSI escapes and collapse ``\\r`` progress-bar rewrites.

    tqdm and friends redraw the same line via carriage returns; Jupyter
    renders only the final frame, so keeping the text after the last ``\\r``
    of each line reproduces what the notebook displays without the invisible
    intermediate frames.
    """
    from tools.ansi_strip import strip_ansi

    cleaned = strip_ansi(text).replace("\r\n", "\n")
    lines = []
    for line in cleaned.split("\n"):
        frames = [frame for frame in line.split("\r") if frame]
        lines.append(frames[-1] if frames else "")
    return "\n".join(lines)


# Notebook outputs longer than this are truncated per output block so a
# single runaway training log cannot flood the extracted text.
_MAX_OUTPUT_CHARS = 20_000
# ANSI/control cleanup is handed a progressive raw prefix before any display
# transformation. The shared sanitizer is single-pass, so four display windows
# preserve the existing 20k useful-output contract even when progress controls
# consume much of the raw text, while still bounding total per-cell work.
_MAX_NOTEBOOK_OUTPUT_RAW_CHARS = 4 * _MAX_OUTPUT_CHARS


def _bounded_source_text(
    source: Any,
    limit: int,
    *,
    separator: str = "",
) -> tuple[str, bool]:
    """Return at most ``limit`` source chars and whether more were present.

    List-form notebook fields are consumed progressively, so this helper does
    not first join an attacker-sized list merely to take a bounded prefix.
    """
    if isinstance(source, str):
        return source[:limit], len(source) > limit
    if not isinstance(source, list):
        return "", False

    # Read one character beyond the requested prefix to distinguish an exact
    # boundary from omitted input without materializing the whole field.
    remaining = max(0, limit) + 1
    parts: list[str] = []
    have_text = False
    for item in source:
        if not isinstance(item, str):
            continue
        if have_text and separator:
            piece = separator[:remaining]
            parts.append(piece)
            remaining -= len(piece)
            if remaining == 0:
                break
        have_text = True
        piece = item[:remaining]
        parts.append(piece)
        remaining -= len(piece)
        if remaining == 0:
            break
    text = "".join(parts)
    return text[:limit], len(text) > limit


class _NotebookOutputSanitizer:
    """One progressive raw-input budget shared by every output in a cell."""

    def __init__(self) -> None:
        self.remaining = _MAX_NOTEBOOK_OUTPUT_RAW_CHARS
        self.truncated = False

    def clean(self, source: Any, *, separator: str = "") -> str:
        raw, truncated = _bounded_source_text(
            source,
            self.remaining,
            separator=separator,
        )
        self.remaining -= len(raw)
        self.truncated = self.truncated or truncated
        return _clean_stream_text(raw)


def _notebook_output_text(
    output: Any,
    sanitizer: Optional[_NotebookOutputSanitizer] = None,
) -> str:
    """Render one notebook output as compact text.

    Keeps stream text, error tracebacks, and textual results; replaces
    token-heavy payloads (base64 images, HTML, widget state) with short
    sized placeholders. Handles both nbformat v4 output shapes and the
    legacy v3 ones (``pyout``/``pyerr``; data flat on the output dict).
    """
    if not isinstance(output, dict):
        return ""
    sanitizer = sanitizer or _NotebookOutputSanitizer()
    otype = output.get("output_type")

    if otype == "stream":
        body = sanitizer.clean(output.get("text", ""))
        return body if body.strip() else ""

    if otype in {"error", "pyerr"}:
        traceback = output.get("traceback")
        ename = sanitizer.clean(output.get("ename", ""))
        evalue = sanitizer.clean(output.get("evalue", ""))
        tb_text = ""
        if isinstance(traceback, list):
            tb_text = sanitizer.clean(traceback, separator="\n")
        header = f"Error: {ename}: {evalue}".rstrip(": ")
        return f"{header}\n{tb_text}".rstrip()

    if otype in {"execute_result", "display_data", "pyout"}:
        data = output.get("data")
        if not isinstance(data, dict):
            # nbformat v3 stores mime data flat on the output dict.
            data = {}
            if isinstance(output.get("text"), (str, list)):
                data["text/plain"] = output["text"]
            for v3_key, mime in (("png", "image/png"), ("jpeg", "image/jpeg"),
                                 ("svg", "image/svg+xml"), ("html", "text/html")):
                if v3_key in output:
                    data[mime] = output[v3_key]

        if "application/vnd.jupyter.widget-view+json" in data:
            return "[interactive widget — omitted]"

        # Prefer readable text: models consume text/plain (e.g. the pandas
        # twin of an HTML table) far better than markup.
        for mime in ("text/plain", "text/markdown"):
            if mime in data:
                body = sanitizer.clean(data[mime])
                if body.strip():
                    return body

        for mime, value in data.items():
            if isinstance(mime, str) and mime.startswith("image/"):
                size = _base64_bytes(_source_text(value))
                return f"[{mime} output — {_human_size(size)}, omitted]"

        if "text/html" in data:
            html = _source_text(data["text/html"])
            return f"[text/html output — {len(html):,} chars, omitted]"

        mimes = ", ".join(str(m) for m in data) or "unknown"
        return f"[{mimes} output — omitted]"

    return ""


def _notebook_outputs(cell: dict, jq_pointer: str = "", filename: str = "") -> str:
    outputs = cell.get("outputs")
    if not isinstance(outputs, list):
        return ""
    sanitizer = _NotebookOutputSanitizer()
    blocks = []
    for output in outputs:
        text = _notebook_output_text(output, sanitizer)
        if text:
            blocks.append(text)
        if sanitizer.truncated:
            blocks.append("… [raw output omitted before display sanitization]")
            break
    if not blocks:
        return ""
    joined = "\n".join(blocks)
    if len(joined) > _MAX_OUTPUT_CHARS:
        omitted = len(joined) - _MAX_OUTPUT_CHARS
        hint = ""
        if jq_pointer and filename:
            # This is deliberately data-only guidance, not a shell snippet.
            # JSON escaping keeps arbitrary filenames (quotes, newlines,
            # metacharacters, leading dashes) visible without making them
            # executable through copy/paste or prompt-following automation.
            hint = (
                " — full output location: notebook file "
                f"{json.dumps(filename, ensure_ascii=True)}, JSON path "
                f"{json.dumps(jq_pointer, ensure_ascii=True)}"
            )
        joined = joined[:_MAX_OUTPUT_CHARS] + f"\n… [{omitted:,} output chars truncated{hint}]"
    return joined


def _extract_notebook(path: str) -> str:
    try:
        # Read one byte beyond the cap from the same open file descriptor.
        # A separate stat followed by json.load() is racy: the file can grow
        # between those operations and make the parser consume it unbounded.
        with open(path, "rb") as fh:
            raw = fh.read(MAX_NOTEBOOK_BYTES + 1)
        if len(raw) > MAX_NOTEBOOK_BYTES:
            raise ExtractionError(
                "Notebook too large to parse "
                f"(more than {MAX_NOTEBOOK_BYTES:,} bytes); "
                "split the notebook or inspect selected cells as raw JSON"
            )
        nb = json.loads(raw)
    except ExtractionError:
        raise
    except (UnicodeError, json.JSONDecodeError, RecursionError) as exc:
        raise NotebookSyntaxError(f"Not a valid notebook: {exc}") from exc
    except (OSError, ValueError) as exc:
        raise ExtractionError(f"Not a valid notebook: {exc}") from exc
    if not isinstance(nb, dict):
        raise NotebookSemanticError("Notebook root is not an object")

    raw_cells = nb.get("cells")
    legacy_v3 = nb.get("nbformat") == 3 or not isinstance(raw_cells, list)
    if isinstance(raw_cells, list):
        cells = (
            (f".cells[{i}].outputs", cell)
            for i, cell in enumerate(raw_cells)
        )
    else:
        raw_worksheets = nb.get("worksheets")
        worksheets = raw_worksheets if isinstance(raw_worksheets, list) else []
        cells = (
            (f".worksheets[{wi}].cells[{ci}].outputs", cell)
            for wi, ws in enumerate(worksheets)
            if isinstance(ws, dict)
            for worksheet_cells in (ws.get("cells"),)
            if isinstance(worksheet_cells, list)
            for ci, cell in enumerate(worksheet_cells)
        )

    nb_name = os.path.basename(path)
    counts = {"markdown": 0, "code": 0, "raw": 0, "heading": 0}
    labels = {"markdown": "Markdown", "code": "Code", "raw": "Raw"}
    if legacy_v3:
        labels["heading"] = "Heading"
    out = _BoundedText("notebook")
    candidate_cells = 0
    rendered_cells = 0
    for jq_pointer, cell in cells:
        candidate_cells += 1
        if not isinstance(cell, dict):
            continue
        typ = cell.get("cell_type")
        if typ not in labels:
            continue
        counts[typ] += 1
        rendered_cells += 1
        suffix = f" {counts[typ]}" if typ != "raw" else ""
        if not out.append(f"# ── {labels[typ]} cell{suffix} ──\n"):
            break
        source_field = "input" if legacy_v3 and typ == "code" else "source"
        source = _source_text(cell.get(source_field, "")).rstrip("\n")
        if legacy_v3 and typ == "heading":
            level = cell.get("level", 1)
            if isinstance(level, bool) or not isinstance(level, int) or not 1 <= level <= 6:
                level = 1
            source = f"{'#' * level} {source}".rstrip()
        if not out.append(source + "\n\n"):
            break
        if typ == "code":
            rendered = _notebook_outputs(cell, jq_pointer, nb_name)
            if rendered:
                if not out.append(f"# ── Output (cell {counts[typ]}) ──\n"):
                    break
                if not out.append(rendered.rstrip("\n") + "\n\n"):
                    break
    if not candidate_cells:
        raise NotebookSemanticError("Notebook contains no cells")
    if not rendered_cells:
        raise NotebookSemanticError("Notebook contains no readable cells")
    return out.render()


_CENTRAL_DIRECTORY_HEADER = struct.Struct("<4s6H3L5H2L")
_CENTRAL_DIRECTORY_SIGNATURE = b"PK\x01\x02"
_END_DIRECTORY_SIGNATURE = b"PK\x05\x06"


def _preflight_office_zip(raw: BinaryIO, label: str) -> int:
    """Validate the bounded central directory before ZipFile allocates it."""
    end_record_reader = getattr(zipfile, "_EndRecData", None)
    if not callable(end_record_reader):
        raise ExtractionError(
            f"Cannot safely read {label}: this Python runtime lacks the "
            "bounded ZIP preflight API"
        )
    try:
        # This stdlib helper reads only the fixed EOCD/tail window and expands
        # valid ZIP64 metadata without constructing ZipInfo objects.
        end = end_record_reader(raw)
    except (OSError, struct.error, zipfile.BadZipFile) as exc:
        raise ExtractionError(f"Not a valid {label}: {exc}") from exc
    if end is None:
        raise ExtractionError(f"Not a valid {label}: missing ZIP end record")

    signature = end[0]
    disk_number, directory_disk = int(end[1]), int(end[2])
    entries_on_disk, declared_entries = int(end[3]), int(end[4])
    directory_size, directory_offset = int(end[5]), int(end[6])
    directory_end = int(end[9])

    if (
        signature != _END_DIRECTORY_SIGNATURE
        or declared_entries == 0xFFFF
        or directory_size == 0xFFFFFFFF
        or directory_offset == 0xFFFFFFFF
    ):
        raise ExtractionError(
            f"{label} uses ZIP64 metadata, which is not supported within the "
            f"{MAX_DOCUMENT_BYTES:,}-byte document limit; re-save it as a standard Office file"
        )
    if (
        disk_number != 0
        or directory_disk != 0
        or entries_on_disk != declared_entries
    ):
        raise ExtractionError(f"{label} uses an unsupported multi-disk ZIP archive")
    if declared_entries > MAX_OOXML_ZIP_ENTRIES:
        raise ExtractionError(
            f"{label} ZIP declares {declared_entries:,} entries "
            f"(limit is {MAX_OOXML_ZIP_ENTRIES:,})"
        )
    if directory_size > MAX_OOXML_CENTRAL_DIRECTORY_BYTES:
        raise ExtractionError(
            f"{label} ZIP central directory is {directory_size:,} bytes "
            f"(limit is {MAX_OOXML_CENTRAL_DIRECTORY_BYTES:,})"
        )

    # ZipFile computes the same start for ordinary and concatenated archives:
    # EOCD location minus the declared central-directory byte length.
    directory_start = directory_end - directory_size
    if directory_start < 0 or directory_end < directory_start:
        raise ExtractionError(f"Not a valid {label}: invalid central-directory offset")

    try:
        raw.seek(directory_start)
        remaining = directory_size
        actual_entries = 0
        while remaining:
            if remaining < _CENTRAL_DIRECTORY_HEADER.size:
                raise ExtractionError(f"Not a valid {label}: truncated central directory")
            header = raw.read(_CENTRAL_DIRECTORY_HEADER.size)
            if len(header) != _CENTRAL_DIRECTORY_HEADER.size:
                raise ExtractionError(f"Not a valid {label}: truncated central directory")
            fields = _CENTRAL_DIRECTORY_HEADER.unpack(header)
            if fields[0] != _CENTRAL_DIRECTORY_SIGNATURE:
                raise ExtractionError(
                    f"Not a valid {label}: bad central-directory signature"
                )
            compressed_size, uncompressed_size = int(fields[8]), int(fields[9])
            name_size, extra_size, comment_size = map(int, fields[10:13])
            member_disk, local_offset = int(fields[13]), int(fields[16])
            if member_disk != 0:
                raise ExtractionError(
                    f"{label} uses an unsupported multi-disk ZIP archive"
                )
            if 0xFFFFFFFF in (compressed_size, uncompressed_size, local_offset):
                raise ExtractionError(
                    f"{label} contains a ZIP64 member, which is not supported"
                )
            variable_size = name_size + extra_size + comment_size
            record_size = _CENTRAL_DIRECTORY_HEADER.size + variable_size
            if record_size > remaining:
                raise ExtractionError(f"Not a valid {label}: truncated central directory")
            raw.seek(variable_size, os.SEEK_CUR)
            remaining -= record_size
            actual_entries += 1
            if actual_entries > MAX_OOXML_ZIP_ENTRIES:
                raise ExtractionError(
                    f"{label} ZIP contains more than "
                    f"{MAX_OOXML_ZIP_ENTRIES:,} entries"
                )
    except OSError as exc:
        raise ExtractionError(f"Not a valid {label}: {exc}") from exc

    if actual_entries != declared_entries:
        raise ExtractionError(
            f"{label} ZIP declares {declared_entries:,} entries but its "
            f"central directory contains {actual_entries:,}"
        )
    raw.seek(0)
    return actual_entries


def _unsafe_zip_member(name: str) -> bool:
    if not name or "\x00" in name or "\\" in name or name.startswith("/"):
        return True
    trimmed = name[:-1] if name.endswith("/") else name
    if not trimmed:
        return True
    parts = trimmed.split("/")
    if any(part in {"", ".", ".."} for part in parts):
        return True
    return len(parts[0]) >= 2 and parts[0][0].isalpha() and parts[0][1] == ":"


class _OfficeZip:
    def __init__(
        self,
        zf: zipfile.ZipFile,
        label: str,
        expected_entries: int,
        is_relevant: Callable[[str], bool],
    ):
        self._zf = zf
        self.label = label
        infos = zf.infolist()
        if len(infos) != expected_entries:
            raise ExtractionError(
                f"{label} ZIP changed while opening: preflight found "
                f"{expected_entries:,} entries, ZipFile found {len(infos):,}"
            )
        if len(infos) > MAX_OOXML_ZIP_ENTRIES:
            raise ExtractionError(
                f"{label} ZIP contains {len(infos):,} entries "
                f"(limit is {MAX_OOXML_ZIP_ENTRIES:,})"
            )

        self._infos: dict[str, zipfile.ZipInfo] = {}
        relevant_bytes = 0
        for info in infos:
            # ZipInfo preserves the central-directory spelling in
            # orig_filename, but filename may already be truncated at a NUL
            # (and can be platform-normalized by Python).  Validate the
            # original spelling before using the normalized name for lookup
            # and duplicate-collision checks.
            original_name = getattr(info, "orig_filename", info.filename)
            if _unsafe_zip_member(original_name):
                raise ExtractionError(
                    f"Unsafe ZIP member path in {label}: {original_name!r}"
                )
            name = info.filename
            if _unsafe_zip_member(name):
                raise ExtractionError(f"Unsafe ZIP member path in {label}: {name!r}")
            if name in self._infos:
                raise ExtractionError(f"Duplicate ZIP member in {label}: {name!r}")
            self._infos[name] = info
            if info.flag_bits & 0x1:
                raise ExtractionError(
                    f"Encrypted ZIP member in {label} is not supported: {name!r}"
                )
            if info.is_dir() or not is_relevant(name):
                continue
            if info.compress_type not in {zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED}:
                raise ExtractionError(
                    f"Unsupported ZIP compression method for {label} member "
                    f"{name!r}; only stored or deflated OOXML is supported"
                )
            if info.file_size > MAX_OOXML_MEMBER_BYTES:
                raise ExtractionError(
                    f"{label} member {name} is {info.file_size:,} uncompressed "
                    f"bytes (limit is {MAX_OOXML_MEMBER_BYTES:,})"
                )
            if info.file_size >= MIN_ZIP_RATIO_MEMBER_BYTES:
                ratio = (
                    float("inf")
                    if info.compress_size <= 0
                    else info.file_size / info.compress_size
                )
                if ratio > MAX_ZIP_COMPRESSION_RATIO:
                    raise ExtractionError(
                        f"{label} member {name} has compression ratio "
                        f"{ratio:,.1f}:1 (limit is {MAX_ZIP_COMPRESSION_RATIO:,.0f}:1)"
                    )
            relevant_bytes += info.file_size

        if relevant_bytes > MAX_OOXML_RELEVANT_BYTES:
            raise ExtractionError(
                f"{label} relevant XML expands to {relevant_bytes:,} bytes "
                f"(limit is {MAX_OOXML_RELEVANT_BYTES:,})"
            )

    @property
    def names(self) -> set[str]:
        return set(self._infos)

    @contextmanager
    def open_member(self, name: str) -> Iterator[BinaryIO]:
        info = self._infos.get(name)
        if info is None:
            raise ExtractionError(f"Missing {name}")
        try:
            with self._zf.open(info, "r") as stream:
                yield stream
        except ExtractionError:
            raise
        except (OSError, EOFError, RuntimeError, NotImplementedError, zipfile.BadZipFile) as exc:
            raise ExtractionError(f"Cannot read {self.label} member {name}: {exc}") from exc


@contextmanager
def _open_office_zip(
    path: str,
    label: str,
    is_relevant: Callable[[str], bool],
) -> Iterator[_OfficeZip]:
    try:
        with open(path, "rb") as raw:
            expected_entries = _preflight_office_zip(raw, label)
            try:
                with zipfile.ZipFile(raw) as zf:
                    yield _OfficeZip(zf, label, expected_entries, is_relevant)
            except ExtractionError:
                raise
            except (zipfile.BadZipFile, RuntimeError, NotImplementedError) as exc:
                raise ExtractionError(f"Not a valid {label}: {exc}") from exc
    except ExtractionError:
        raise
    except OSError as exc:
        raise ExtractionError(str(exc)) from exc


class _BoundedXmlReader:
    """Small-chunk XML reader that enforces bytes and rejects DTD/entities."""

    def __init__(self, stream: BinaryIO, name: str):
        self._stream = stream
        self._name = name
        self._read = 0
        self._scan_tail = b""

    def read(self, size: int = -1) -> bytes:
        if size == 0:
            return b""
        remaining = MAX_OOXML_MEMBER_BYTES - self._read
        requested = _XML_READ_CHUNK_BYTES if size < 0 else min(size, _XML_READ_CHUNK_BYTES)
        data = self._stream.read(min(max(requested, 1), remaining + 1))
        self._read += len(data)
        if self._read > MAX_OOXML_MEMBER_BYTES:
            raise ExtractionError(
                f"XML member {self._name} exceeded the "
                f"{MAX_OOXML_MEMBER_BYTES:,}-byte streaming limit"
            )
        # OOXML does not permit custom DTDs. Rejecting them also removes XML
        # entity expansion as a second decompression/amplification channel.
        normalized = data.replace(b"\x00", b"").upper()
        scan = self._scan_tail + normalized
        if b"<!DOCTYPE" in scan or b"<!ENTITY" in scan:
            raise ExtractionError(
                f"Unsafe DTD/entity declaration in XML member {self._name}"
            )
        self._scan_tail = scan[-16:]
        return data


class _XmlEventBudget:
    """Cumulative XML parser-work budget shared across one Office file."""

    def __init__(self, label: str):
        self.label = label
        self.events = 0

    def consume(self, member: str) -> None:
        self.events += 1
        if self.events > MAX_OOXML_XML_EVENTS:
            raise _XmlEventBudgetExceeded(
                f"{self.label} XML exceeds the extraction-wide budget of "
                f"{MAX_OOXML_XML_EVENTS:,} events while parsing {member}"
            )


def _xml_events(
    archive: _OfficeZip,
    name: str,
    budget: _XmlEventBudget,
) -> Iterator[tuple[str, ET.Element]]:
    with archive.open_member(name) as stream:
        reader = _BoundedXmlReader(stream, name)
        parents: list[ET.Element] = []
        try:
            for event, node in ET.iterparse(reader, events=("start", "end")):
                budget.consume(name)
                if event == "start":
                    if len(parents) >= MAX_OOXML_XML_DEPTH:
                        raise ExtractionError(
                            f"XML member {name} exceeds the maximum nesting "
                            f"depth of {MAX_OOXML_XML_DEPTH:,}"
                        )
                    parents.append(node)
                    yield event, node
                    continue
                yield event, node
                # Element.clear() releases attributes/text/children but does
                # not unlink the cleared child from its parent. Without this
                # removal a 16 MiB XML part containing hundreds of thousands
                # of empty elements still retains a huge parent child-list.
                if len(parents) >= 2:
                    parents[-2].remove(node)
                node.clear()
                if parents:
                    parents.pop()
        except ExtractionError:
            raise
        except ET.ParseError:
            raise
        except (OSError, EOFError, RuntimeError, zipfile.BadZipFile) as exc:
            raise ExtractionError(f"Cannot parse XML member {name}: {exc}") from exc


def _extract_docx(path: str) -> str:
    out = _BoundedText("DOCX")
    out.append(
        "[Extraction view: main document body only; headers, footers, "
        "footnotes, endnotes, and comments are not included.]\n"
    )
    paragraph_tags = _namespace_tags(_NS_W_FAMILY, "p")
    text_tags = _namespace_tags(_NS_W_FAMILY, "t")
    tab_tags = _namespace_tags(_NS_W_FAMILY, "tab")
    break_tags = (
        _namespace_tags(_NS_W_FAMILY, "br")
        | _namespace_tags(_NS_W_FAMILY, "cr")
    )
    paragraph_depth = 0
    has_text = False
    elements = 0
    xml_budget = _XmlEventBudget("DOCX")
    try:
        with _open_office_zip(
            path, "DOCX", lambda name: name == "word/document.xml"
        ) as archive:
            for event, node in _xml_events(
                archive,
                "word/document.xml",
                xml_budget,
            ):
                if event == "start":
                    if node.tag in paragraph_tags:
                        paragraph_depth += 1
                    continue
                elements += 1
                if elements > MAX_DOCX_XML_ELEMENTS:
                    out.mark_truncated(
                        f"the {MAX_DOCX_XML_ELEMENTS:,}-element DOCX body safety limit"
                    )
                    break
                if paragraph_depth:
                    if node.tag in text_tags:
                        text = node.text or ""
                        has_text = has_text or bool(text.strip())
                        if not out.append(text):
                            break
                    elif node.tag in tab_tags:
                        if not out.append("\t"):
                            break
                    elif node.tag in break_tags:
                        if not out.append("\n"):
                            break
                    elif node.tag in paragraph_tags:
                        paragraph_depth -= 1
                        if not out.append("\n"):
                            break
                node.clear()
    except _XmlEventBudgetExceeded:
        out.mark_truncated(
            f"the {MAX_OOXML_XML_EVENTS:,}-event XML parser safety limit"
        )
    except ET.ParseError as exc:
        raise ExtractionError(f"Malformed XML in word/document.xml: {exc}") from exc
    if not has_text:
        raise ExtractionError("DOCX contains no extractable text")
    return out.render()


def _xlsx_relevant_member(name: str) -> bool:
    return name in {
        "xl/workbook.xml",
        "xl/_rels/workbook.xml.rels",
        "xl/sharedStrings.xml",
    } or (name.startswith("xl/worksheets/") and name.endswith(".xml"))


def _extract_xlsx(path: str) -> str:
    out = _BoundedText("XLSX")
    rendered_sheets = 0
    xml_budget = _XmlEventBudget("XLSX")
    with _open_office_zip(path, "XLSX", _xlsx_relevant_member) as archive:
        shared = _shared_strings(archive, xml_budget)
        sheets = _workbook_sheets(archive, xml_budget)
        rels = _workbook_rels(archive, xml_budget)
        for name, state, rid in sheets:
            if state in {"hidden", "veryHidden"}:
                continue
            if not rid:
                raise ExtractionError(
                    f"Visible XLSX sheet {name!r} is missing a relationship Id"
                )
            if rid not in rels:
                raise ExtractionError(
                    f"Visible XLSX sheet {name!r} references missing workbook "
                    f"relationship {rid!r}"
                )
            part = _sheet_part(rels[rid])
            if not part:
                raise ExtractionError(
                    f"Visible XLSX sheet {name!r} relationship {rid!r} does not "
                    "target a worksheet part"
                )
            if part not in archive.names:
                raise ExtractionError(
                    f"Visible XLSX sheet {name!r} worksheet part is missing: {part}"
                )
            checkpoint = out.checkpoint()
            row_count = 0
            try:
                if not out.append(
                    f"# ── Sheet: {name} ──\n{_XLSX_COVERAGE_NOTE}"
                ):
                    rendered_sheets += 1
                    break
                for row in _sheet_rows(archive, part, shared, xml_budget):
                    row_count += 1
                    for index, value in enumerate(row):
                        if index and not out.append("\t"):
                            break
                        if not out.append(value):
                            break
                    if out.truncated or not out.append("\n"):
                        break
                if not row_count and not out.truncated:
                    out.append("(empty)\n")
                if not out.truncated:
                    out.append("\n")
            except _XmlEventBudgetExceeded:
                # Workbook metadata, relationships, and shared strings are
                # parsed before this content loop and remain fail-closed. A
                # valid worksheet that merely exhausts the bounded parser view
                # can still return the rows already rendered.
                out.mark_truncated(
                    f"the {MAX_OOXML_XML_EVENTS:,}-event XML parser safety limit"
                )
            except ET.ParseError as exc:
                out.rollback(checkpoint)
                raise ExtractionError(
                    f"Malformed XML in {part} for sheet {name!r}: {exc}"
                ) from exc
            rendered_sheets += 1
            if out.truncated:
                break

    if not rendered_sheets:
        raise ExtractionError("XLSX has no visible sheets with content")
    return out.render()


def _shared_strings(
    archive: _OfficeZip,
    xml_budget: _XmlEventBudget,
) -> list[str]:
    path = "xl/sharedStrings.xml"
    if path not in archive.names:
        return []
    string_item_tags = _namespace_tags(_NS_S_FAMILY, "si")
    text_tags = _namespace_tags(_NS_S_FAMILY, "t")
    strings: list[str] = []
    current: Optional[list[str]] = None
    current_chars = 0
    total_chars = 0
    try:
        for event, node in _xml_events(archive, path, xml_budget):
            if event == "start":
                if node.tag in string_item_tags:
                    current = []
                    current_chars = 0
                continue
            if current is not None and node.tag in text_tags:
                text = node.text or ""
                current_chars += len(text)
                if current_chars > MAX_XLSX_CELL_CHARS:
                    raise ExtractionError(
                        f"XLSX shared string exceeds {MAX_XLSX_CELL_CHARS:,} characters"
                    )
                current.append(text)
            elif node.tag in string_item_tags and current is not None:
                if len(strings) >= MAX_XLSX_SHARED_STRINGS:
                    raise ExtractionError(
                        f"XLSX has more than {MAX_XLSX_SHARED_STRINGS:,} shared strings"
                    )
                total_chars += current_chars
                if total_chars > MAX_XLSX_SHARED_STRING_CHARS:
                    raise ExtractionError(
                        "XLSX shared strings exceed the "
                        f"{MAX_XLSX_SHARED_STRING_CHARS:,}-character limit"
                    )
                strings.append("".join(current))
                current = None
            node.clear()
    except ET.ParseError as exc:
        raise ExtractionError(f"Malformed XML in {path}: {exc}") from exc
    return strings


def _workbook_sheets(
    archive: _OfficeZip,
    xml_budget: _XmlEventBudget,
) -> list[tuple[str, str, str]]:
    path = "xl/workbook.xml"
    sheet_tags = _namespace_tags(_NS_S_FAMILY, "sheet")
    relationship_id_attributes = tuple(
        f"{{{namespace}}}id" for namespace in _NS_REL_FAMILY
    )
    sheets: list[tuple[str, str, str]] = []
    try:
        for event, sheet in _xml_events(archive, path, xml_budget):
            if event == "end" and sheet.tag in sheet_tags:
                if len(sheets) >= MAX_XLSX_SHEETS:
                    raise ExtractionError(
                        f"XLSX has more than {MAX_XLSX_SHEETS:,} sheets"
                    )
                sheets.append(
                    (
                        sheet.get("name", "Sheet"),
                        sheet.get("state", "visible"),
                        next(
                            (
                                value
                                for attribute in relationship_id_attributes
                                if (value := sheet.get(attribute)) is not None
                            ),
                            "",
                        ),
                    )
                )
            if event == "end":
                sheet.clear()
    except ET.ParseError as exc:
        raise ExtractionError(f"Malformed XML in {path}: {exc}") from exc
    return sheets


def _workbook_rels(
    archive: _OfficeZip,
    xml_budget: _XmlEventBudget,
) -> dict[str, str]:
    path = "xl/_rels/workbook.xml.rels"
    if path not in archive.names:
        return {}
    rel_tags = _namespace_tags(_NS_PKG_REL_FAMILY, "Relationship")
    rels: dict[str, str] = {}
    relationship_count = 0
    try:
        for event, rel in _xml_events(archive, path, xml_budget):
            if event == "end" and rel.tag in rel_tags:
                relationship_count += 1
                if relationship_count > MAX_XLSX_SHEETS * 4:
                    raise ExtractionError("XLSX has too many workbook relationships")
                rid = rel.get("Id", "")
                if rid:
                    if rid in rels:
                        raise ExtractionError(
                            f"Duplicate XLSX workbook relationship Id: {rid!r}"
                        )
                    rels[rid] = rel.get("Target", "")
            if event == "end":
                rel.clear()
    except ET.ParseError as exc:
        raise ExtractionError(f"Malformed XML in {path}: {exc}") from exc
    return rels


def _sheet_part(target: str) -> str:
    if "\\" in target or "\x00" in target or ".." in target.split("/"):
        raise ExtractionError(f"Unsafe XLSX worksheet relationship target: {target!r}")
    target = target.lstrip("/")
    part = posixpath.normpath(target if target.startswith("xl/") else f"xl/{target}")
    if not part.startswith("xl/worksheets/"):
        return ""
    return part


def _col_index(ref: str) -> int:
    idx = 0
    for ch in ref[:16]:
        if not ch.isalpha():
            break
        idx = idx * 26 + ord(ch.upper()) - ord("A") + 1
    return max(idx - 1, 0)


def _sheet_rows(
    archive: _OfficeZip,
    part: str,
    shared: list[str],
    xml_budget: _XmlEventBudget,
) -> Iterator[list[str]]:
    row_tags = _namespace_tags(_NS_S_FAMILY, "row")
    cell_tags = _namespace_tags(_NS_S_FAMILY, "c")
    value_tags = _namespace_tags(_NS_S_FAMILY, "v")
    text_tags = _namespace_tags(_NS_S_FAMILY, "t")
    cells: Optional[dict[int, str]] = None
    max_col = -1
    rows_seen = 0
    pending_blank_rows = 0
    in_cell = False
    cells_in_row = 0
    cell_ref = ""
    cell_type = ""
    cell_value = ""
    inline_parts: list[str] = []
    for event, node in _xml_events(archive, part, xml_budget):
        if event == "start":
            if node.tag in row_tags:
                if rows_seen >= _MAX_XLSX_ROWS_PER_SHEET:
                    break
                cells = {}
                max_col = -1
                cells_in_row = 0
            elif node.tag in cell_tags and cells is not None:
                cells_in_row += 1
                if cells_in_row > MAX_XLSX_CELLS_PER_ROW:
                    raise ExtractionError(
                        "XLSX row contains more than "
                        f"{MAX_XLSX_CELLS_PER_ROW:,} cells"
                    )
                in_cell = True
                cell_ref = node.get("r", "")
                cell_type = node.get("t", "")
                cell_value = ""
                inline_parts = []
            continue

        # _xml_events unlinks each child after yielding its end event. Capture
        # cell values incrementally so a malicious row cannot make ElementTree
        # retain every completed <c>, while preserving rich inline strings.
        if in_cell and node.tag in value_tags:
            cell_value = node.text or ""
        elif in_cell and node.tag in text_tags and cell_type == "inlineStr":
            inline_parts.append(node.text or "")
        elif node.tag in cell_tags and cells is not None:
            col = _col_index(cell_ref) if cell_ref else max_col + 1
            if col < _MAX_XLSX_COLS:
                cells[col] = _cell_value(
                    cell_type,
                    cell_value,
                    "".join(inline_parts),
                    shared,
                    part=part,
                    cell_ref=cell_ref,
                )
                max_col = max(max_col, col)
            in_cell = False
        elif node.tag in row_tags and cells is not None:
            row = (
                [cells.get(i, "") for i in range(max_col + 1)]
                if max_col >= 0
                else []
            )
            rows_seen += 1
            if any(value.strip() for value in row):
                for _ in range(pending_blank_rows):
                    yield []
                pending_blank_rows = 0
                yield row
            else:
                pending_blank_rows += 1
            cells = None


def _cell_value(
    typ: str,
    value: str,
    inline_text: str,
    shared: list[str],
    *,
    part: str,
    cell_ref: str,
) -> str:
    if typ == "s":
        try:
            shared_index = int(value)
        except ValueError as exc:
            raise ExtractionError(
                f"Invalid XLSX shared-string index {value!r} in {part} "
                f"cell {cell_ref or '(unknown)'}"
            ) from exc
        if shared_index < 0 or shared_index >= len(shared):
            raise ExtractionError(
                f"Invalid XLSX shared-string index {value!r} in {part} "
                f"cell {cell_ref or '(unknown)'}"
            )
        value = shared[shared_index]
    elif typ == "inlineStr":
        value = inline_text
    elif typ == "b":
        normalized = value.strip().lower()
        if normalized in {"1", "true"}:
            value = "TRUE"
        elif normalized in {"0", "false"}:
            value = "FALSE"
        else:
            raise ExtractionError(
                f"Invalid XLSX boolean value {value!r} in {part} "
                f"cell {cell_ref or '(unknown)'}"
            )
    elif typ == "e":
        value = value or "#ERROR"
    if len(value) > MAX_XLSX_CELL_CHARS:
        raise ExtractionError(
            f"XLSX cell text exceeds {MAX_XLSX_CELL_CHARS:,} characters"
        )
    return value
