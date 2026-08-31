"""Unit tests for the truncate-and-store web_extract path (no LLM).

Covers convert_base64_images_to_links, _truncate_with_footer, _store_full_text,
_get_extract_char_limit, and the end-to-end web_extract_tool truncation behavior.
"""
import asyncio
import json
import os
from pathlib import Path
from unittest.mock import patch

import pytest

import tools.web_tools as wt


class TestImageConversion:
    def test_markdown_base64_image_keeps_alt_drops_blob(self):
        blob = "A" * 5000
        text = f"before ![a cat]( data:image/png;base64,{blob}) after"
        out = wt.convert_base64_images_to_links(text)
        assert "[IMAGE: a cat]" in out
        assert "base64" not in out
        assert blob not in out
        assert "before" in out and "after" in out


    def test_bare_and_parenthesised_base64_become_placeholder(self):
        blob = "Z" * 3000
        bare = wt.convert_base64_images_to_links(f"data:image/gif;base64,{blob}")
        assert bare == "[IMAGE]"
        paren = wt.convert_base64_images_to_links(f"(data:image/gif;base64,{blob})")
        assert paren == "[IMAGE]"


class TestTruncation:
    def test_short_content_returned_whole(self):
        content = "# Title\n\nshort body\n"
        out, truncated = wt._truncate_with_footer(content, "https://e.com", 15000)
        assert out == content
        assert truncated is False


    def test_long_content_force_redacts_returned_and_stored_text(
        self, tmp_path, monkeypatch
    ):
        """The long-result model and disk boundaries stay mandatory."""
        monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
        monkeypatch.setattr("agent.redact._REDACT_ENABLED", False)
        head_secret = "sk-" + "H" * 32
        middle_secret = "ghp_" + "M" * 32
        public_url = (
            "https://example.com/docs?code=public-guide&session=summer"
        )
        body = (
            f"head credential {head_secret}\n"
            f"[Public docs]({public_url})\n"
            + "\n".join(f"ordinary row {i}" for i in range(300))
            + f"\nmiddle credential {middle_secret}\n"
            + "\n".join(f"ordinary tail row {i}" for i in range(300))
        )

        out, truncated = wt._truncate_with_footer(
            body, "https://example.com/private-page", 3000
        )

        assert truncated is True
        assert "head credential" in out
        assert head_secret not in out
        assert public_url in out
        path_line = next(
            line for line in out.splitlines() if "Full text saved to:" in line
        )
        stored_path = path_line.split("Full text saved to:", 1)[1].strip()
        stored = Path(stored_path).read_text(encoding="utf-8")
        assert head_secret not in stored
        assert middle_secret not in stored
        assert "middle credential" in stored
        assert public_url in stored


    def test_store_full_text_force_redacts_when_globally_disabled(
        self, tmp_path, monkeypatch
    ):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
        monkeypatch.setattr("agent.redact._REDACT_ENABLED", False)
        secret = "sk-" + "S" * 32
        public_url = "https://example.com/callback?code=public-code&state=keep"

        stored_path = wt._store_full_text(
            "https://example.com/direct-store",
            f"credential {secret}\n{public_url}\n",
        )

        assert stored_path is not None
        stored = Path(stored_path).read_text(encoding="utf-8")
        assert secret not in stored
        assert "credential" in stored
        assert public_url in stored


    def test_store_full_text_same_url_reuses_absolute_path_and_replaces_content(
        self, tmp_path, monkeypatch
    ):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
        url = "https://example.com/changing-page"

        first_path = wt._store_full_text(url, "first page version\n")
        second_path = wt._store_full_text(url, "second page version\n")

        assert first_path is not None
        assert second_path is not None
        assert Path(first_path).is_absolute()
        assert second_path == first_path
        stored = Path(second_path).read_text(encoding="utf-8")
        assert stored == "second page version\n"
        assert "first page version" not in stored


    def test_truncation_stores_full_text_readable(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
        body = "UNIQUE_MIDDLE_MARKER\n" + ("\n".join(f"row {i}" for i in range(5000)))
        out, truncated = wt._truncate_with_footer(body, "https://example.com/doc", 3000)
        assert truncated is True
        # Extract the stored path from the footer and confirm full text is there.
        path_line = next(ln for ln in out.splitlines() if "Full text saved to:" in ln)
        stored_path = path_line.split("Full text saved to:", 1)[1].strip()
        assert os.path.exists(stored_path)
        full = Path(stored_path).read_text(encoding="utf-8")
        assert "UNIQUE_MIDDLE_MARKER" in full
        assert "row 2500" in full  # the omitted-middle row is in the stored file


class TestCharLimitConfig:
    def test_default_when_unset(self):
        with patch("tools.web_tools._load_web_config", return_value={}):
            assert wt._get_extract_char_limit() == wt.DEFAULT_EXTRACT_CHAR_LIMIT


    def test_bad_value_falls_back(self):
        with patch("tools.web_tools._load_web_config", return_value={"extract_char_limit": "nope"}):
            assert wt._get_extract_char_limit() == wt.DEFAULT_EXTRACT_CHAR_LIMIT


class TestEndToEnd:
    def test_web_extract_truncates_large_page_no_llm(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
        big = "\n".join(f"para {i} " + "y" * 80 for i in range(3000))

        class FakeProvider:
            name = "fake"
            display_name = "Fake"

            def supports_extract(self):
                return True

            async def extract(self, urls, **kwargs):
                return [{"url": urls[0], "title": "Big Page", "content": big,
                         "raw_content": big, "metadata": {}}]

        with patch("tools.web_tools._ensure_web_plugins_loaded"), \
             patch("tools.web_tools._get_extract_backend", return_value="fake"), \
             patch("tools.web_tools.async_is_safe_url", new=_AsyncTrue()), \
             patch("agent.web_search_registry.get_provider", return_value=FakeProvider()):
            result = json.loads(asyncio.new_event_loop().run_until_complete(
                wt.web_extract_tool(["https://example.com/big"], char_limit=5000)
            ))

        assert "results" in result
        content = result["results"][0]["content"]
        assert "[TRUNCATED]" in content
        assert "Full text saved to:" in content
        # No LLM was involved: para 0 (head) and the last para (tail) are verbatim.
        assert "para 0 " in content
        assert "para 2999 " in content


    def test_truncated_extract_handoff_stays_inline_with_cache_disabled(
        self, tmp_path, monkeypatch
    ):
        hermes_home = tmp_path / ".hermes"
        monkeypatch.setenv("HERMES_HOME", str(hermes_home))
        large_page = "\n".join(
            f"long row {index} " + "z" * 80 for index in range(5000)
        )

        class FakeProvider:
            name = "fake"
            display_name = "Fake"

            @staticmethod
            def supports_extract():
                return True

            async def extract(self, urls, **_kwargs):
                return [
                    {
                        "url": urls[0],
                        "title": "Large page",
                        "content": large_page,
                        "raw_content": large_page,
                        "metadata": {},
                    }
                ]

        with patch("tools.web_tools._ensure_web_plugins_loaded"), \
             patch("tools.web_tools._get_extract_backend", return_value="fake"), \
             patch("tools.web_tools.async_is_safe_url", new=_AsyncTrue()), \
             patch("agent.web_search_registry.get_provider", return_value=FakeProvider()), \
             patch(
                 "tools.web_result_cache._web_config",
                 return_value={"cache_enabled": False},
             ):
            tool_result = asyncio.run(
                wt.web_extract_tool(
                    ["https://example.com/large-handoff"],
                    char_limit=5000,
                )
            )

        from tools.tool_result_storage import maybe_persist_tool_result

        assert "[TRUNCATED]" in tool_result
        handed_off = maybe_persist_tool_result(
            content=tool_result,
            tool_name="web_extract",
            tool_use_id="web-large-handoff",
        )

        assert handed_off == tool_result
        assert not (hermes_home / "cache" / "spillover").exists()
        web_cache = hermes_home / "cache" / "web"
        assert len(list(web_cache.glob("*.md"))) == 1
        assert not (web_cache / "extract-index.json").exists()
        assert list(web_cache.glob("*.cache.md")) == []


def _make_awaitable(value):
    async def _coro(*a, **k):
        return value
    return _coro()


class _AsyncTrue:
    """Async callable that always returns True (re-awaitable per call)."""
    async def __call__(self, *a, **k):
        return True
