from agent.turn_finalizer import (
    _bounded_memory_recall_metadata,
    _response_metadata_from_hook_results,
)


def test_terminal_response_metadata_is_bounded_json_and_first_writer_wins():
    results = [
        {"response_metadata": {"dorvis_trace_manifest": {"trace_id": "first"}}},
        {"response_metadata": {"dorvis_trace_manifest": {"trace_id": "late"}}},
        {"response_metadata": {"bad key": "ignored"}},
        {"response_metadata": {"not_json": object()}},
        {"response_metadata": {"oversized": "x" * (64 * 1024)}},
        "not-a-mapping",
    ]

    assert _response_metadata_from_hook_results(results) == {
        "dorvis_trace_manifest": {"trace_id": "first"}
    }


def _recall_payload(memories):
    return {
        "provider": "hindsight",
        "status": "injected",
        "count": len(memories),
        "memories": memories,
        "query_char_count": 12,
        "injected_char_count": 240,
    }


def test_memory_recall_metadata_passes_through_when_memories_were_injected():
    memories = [{"id": "fact-1", "text": "snippet", "score": None, "type": "world"}]

    assert _bounded_memory_recall_metadata(_recall_payload(memories)) == {
        "provider": "hindsight",
        "status": "injected",
        "count": 1,
        "memories": memories,
        "query_char_count": 12,
        "injected_char_count": 240,
    }


def test_memory_recall_metadata_is_omitted_when_nothing_was_injected():
    assert _bounded_memory_recall_metadata(None) is None
    assert _bounded_memory_recall_metadata({}) is None
    assert _bounded_memory_recall_metadata(_recall_payload([])) is None
    assert _bounded_memory_recall_metadata(_recall_payload(["not-a-dict"])) is None


def test_memory_recall_metadata_sheds_memories_to_stay_bounded():
    memories = [
        {"id": f"fact-{i}", "text": "x" * 2000, "score": None, "type": "world"}
        for i in range(100)
    ]

    bounded = _bounded_memory_recall_metadata(_recall_payload(memories))

    assert bounded is not None
    assert bounded["truncated"] is True
    # AE-196: the explicit name the web contract reads.
    assert bounded["memories_truncated"] is True
    assert 0 < len(bounded["memories"]) < len(memories)
    # count still reports what the turn actually injected.
    assert bounded["count"] == 100


def test_memory_recall_metadata_fits_a_realistic_full_recall():
    """AE-196: the UI shows memory text, so the 48 KiB envelope has to carry a
    full 25-record recall of substantial memories unshed — the old 16 KiB cap
    shed most of them. Only the absolute worst case (every one of the 25
    records at the provider's 2000-char ceiling, ~51 KiB encoded) still sheds
    its tail, which is the intended graceful degradation."""
    memories = [
        {
            "id": f"fact-{i}",
            "text": "x" * 1800,
            "text_truncated": True,
            "score": None,
            "type": "world",
        }
        for i in range(25)
    ]

    bounded = _bounded_memory_recall_metadata(_recall_payload(memories))

    assert bounded is not None
    assert bounded["memories"] == memories
    assert "memories_truncated" not in bounded
    assert "truncated" not in bounded


def test_memory_recall_metadata_rejects_unserializable_payloads():
    assert _bounded_memory_recall_metadata(
        _recall_payload([{"id": "fact-1", "text": object()}])
    ) is None
