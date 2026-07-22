from agent.turn_finalizer import _response_metadata_from_hook_results


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
