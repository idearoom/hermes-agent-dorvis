from types import SimpleNamespace

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

    def _recall(query: str):
        seen["query"] = query
        record = {
            "id": "fact-1",
            "text": "remembered fact",
            "score": None,
            "type": "world",
        }
        return SimpleNamespace(text="- remembered fact", count=1, records=(record,))

    monkeypatch.setattr(provider, "_do_recall", _recall)

    result = provider.prefetch("0123456789abcdef")

    assert seen["query"] == "01234567"
    assert result == "Memory header\n\n- remembered fact"
    # Same-turn recall carries per-memory structure too (AE-194).
    assert provider.consume_prefetch_memories() == [
        {"id": "fact-1", "text": "remembered fact", "score": None, "type": "world"}
    ]


def test_same_turn_recall_respects_platform_allowlist(monkeypatch):
    provider = HindsightMemoryProvider()
    provider._same_turn_recall_when_prefetch_empty = True
    provider._same_turn_recall_platforms = ["api_server"]
    provider._platform = "cli"
    provider._auto_recall = True
    provider._memory_mode = "context"

    called = False

    def _recall(query: str):
        nonlocal called
        called = True
        return SimpleNamespace(text="- should not appear", count=1, records=())

    monkeypatch.setattr(provider, "_do_recall", _recall)

    assert provider.prefetch("hello") == ""
    assert called is False
    assert provider.consume_prefetch_memories() == []
