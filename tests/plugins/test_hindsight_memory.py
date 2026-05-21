from plugins.memory.hindsight import HindsightMemoryProvider


def test_same_turn_recall_fills_empty_prefetch(monkeypatch):
    provider = HindsightMemoryProvider()
    provider._same_turn_recall_when_prefetch_empty = True
    provider._same_turn_recall_platforms = []
    provider._auto_recall = True
    provider._memory_mode = "context"
    provider._recall_max_input_chars = 8
    provider._recall_prompt_preamble = "Memory header"

    seen = {}

    def _recall(query: str) -> str:
        seen["query"] = query
        return "- remembered fact"

    monkeypatch.setattr(provider, "_recall_context_text", _recall)

    result = provider.prefetch("0123456789abcdef")

    assert seen["query"] == "01234567"
    assert result == "Memory header\n\n- remembered fact"


def test_same_turn_recall_respects_platform_allowlist(monkeypatch):
    provider = HindsightMemoryProvider()
    provider._same_turn_recall_when_prefetch_empty = True
    provider._same_turn_recall_platforms = ["api_server"]
    provider._platform = "cli"
    provider._auto_recall = True
    provider._memory_mode = "context"

    called = False

    def _recall(query: str) -> str:
        nonlocal called
        called = True
        return "- should not appear"

    monkeypatch.setattr(provider, "_recall_context_text", _recall)

    assert provider.prefetch("hello") == ""
    assert called is False
