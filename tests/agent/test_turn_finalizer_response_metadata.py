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
        {"id": f"fact-{i}", "text": "x" * 280, "score": None, "type": "world"}
        for i in range(100)
    ]

    bounded = _bounded_memory_recall_metadata(_recall_payload(memories))

    assert bounded is not None
    assert bounded["truncated"] is True
    assert 0 < len(bounded["memories"]) < len(memories)
    # count still reports what the turn actually injected.
    assert bounded["count"] == 100


def test_memory_recall_metadata_rejects_unserializable_payloads():
    assert _bounded_memory_recall_metadata(
        _recall_payload([{"id": "fact-1", "text": object()}])
    ) is None
