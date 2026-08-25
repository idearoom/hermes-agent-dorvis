from types import SimpleNamespace
import asyncio
import concurrent.futures
import gc
import weakref
from unittest.mock import patch

from agent.runtime_usage import (
    attribute_auxiliary_usage,
    commit_primary_response,
    initialize_agent_usage_attribution,
    mark_primary_usage_missing,
    observe_auxiliary_attempts,
    record_auxiliary_response,
    rollup_delegated_usage,
    snapshot_agent_usage,
    track_auxiliary_dispatch,
    track_auxiliary_dispatch_async,
    track_auxiliary_stream_dispatch,
    track_primary_dispatch,
    validate_primary_usage_components,
)
from agent.usage_pricing import CanonicalUsage, normalize_usage


def _agent(**overrides):
    agent = SimpleNamespace(
        session_prompt_tokens=100,
        session_completion_tokens=20,
        session_total_tokens=120,
        session_api_calls=1,
    )
    initialize_agent_usage_attribution(agent)
    for key, value in overrides.items():
        setattr(agent, key, value)
    return agent


def _response(*, prompt=30, completion=7, total=37):
    return SimpleNamespace(
        id="aux-response-1",
        usage=SimpleNamespace(
            prompt_tokens=prompt,
            completion_tokens=completion,
            total_tokens=total,
        ),
    )


def test_auxiliary_usage_is_attributed_once_per_provider_response():
    agent = _agent()
    response = _response()

    with attribute_auxiliary_usage(agent):
        record_auxiliary_response(response, task="compression")
        record_auxiliary_response(response, task="compression")

    usage = snapshot_agent_usage(agent)
    assert usage["input_tokens"] == 130
    assert usage["output_tokens"] == 27
    assert usage["total_tokens"] == 157
    assert usage["completeness"] == "complete"
    assert usage["breakdown"]["parent"] == {
        "input_tokens": 100,
        "output_tokens": 20,
        "total_tokens": 120,
    }
    assert usage["breakdown"]["auxiliary"] == {
        "input_tokens": 30,
        "output_tokens": 7,
        "total_tokens": 37,
    }


def test_seen_auxiliary_responses_are_strongly_held_for_run_lifetime():
    class Response:
        pass

    agent = _agent()
    response = Response()
    response.usage = SimpleNamespace(
        prompt_tokens=3,
        completion_tokens=1,
        total_tokens=4,
    )
    response_ref = weakref.ref(response)
    response_id = id(response)

    with attribute_auxiliary_usage(agent):
        record_auxiliary_response(response, task="compression")
    del response
    gc.collect()

    assert response_ref() is not None
    assert agent._runtime_usage_seen_aux_response_objects[response_id] is response_ref()


def test_contextvar_attribution_propagates_to_tool_worker_context():
    from tools.thread_context import propagate_context_to_thread

    agent = _agent()
    response = _response(prompt=8, completion=2, total=10)
    with attribute_auxiliary_usage(agent):
        wrapped = propagate_context_to_thread(
            lambda: record_auxiliary_response(response, task="plugin_llm")
        )
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            executor.submit(wrapped).result()

    usage = snapshot_agent_usage(agent)
    assert usage["breakdown"]["auxiliary"]["total_tokens"] == 10


def test_auxiliary_dispatch_emits_one_terminal_observer_generation_per_attempt():
    agent = _agent(
        session_id="session-aux-observer",
        _current_turn_id="turn-aux-observer",
        _request_metadata={"source": "dorvis-web", "chat": {"id": "chat-1"}},
    )
    captured = []

    with patch(
        "hermes_cli.plugins.invoke_hook",
        side_effect=lambda name, **kwargs: captured.append((name, kwargs)),
    ), attribute_auxiliary_usage(agent):
        response = track_auxiliary_dispatch(
            lambda: _response(prompt=13, completion=5, total=18),
            task="compression",
            provider="openai-codex",
            model="gpt-5.6-codex",
            base_url="https://chatgpt.com/backend-api/codex",
            api_mode="codex_responses",
            request={"method": "POST", "body": {"messages": [{"role": "user", "content": "compress"}]}},
        )

    assert response.usage.total_tokens == 18
    assert [name for name, _ in captured] == [
        "pre_auxiliary_api_request",
        "post_auxiliary_api_request",
    ]
    started, completed = (payload for _, payload in captured)
    assert started["api_request_id"] == completed["api_request_id"]
    assert started["api_request_id"].startswith("aux-")
    assert started["purpose"] == "compression"
    assert started["provider"] == "openai-codex"
    assert started["model"] == "gpt-5.6-codex"
    assert started["session_id"] == "session-aux-observer"
    assert started["turn_id"] == "turn-aux-observer"
    assert started["request"]["body"]["messages"][0]["content"] == "compress"
    assert completed["usage"]["input_tokens"] == 13
    assert completed["usage"]["output_tokens"] == 5
    assert completed["usage"]["total_tokens"] == 18
    assert completed["usage"]["cost_status"] == "included"
    assert completed["usage"]["cost_usd"] == 0.0
    assert completed["api_duration"] >= 0
    assert completed["request_metadata"]["source"] == "dorvis-web"


def test_auxiliary_observer_preserves_provider_total_mismatch_evidence():
    agent = _agent(
        session_id="session-aux-mismatch",
        _current_turn_id="turn-aux-mismatch",
        _request_metadata={"source": "dorvis-web", "chat": {"id": "chat-1"}},
    )
    captured = []
    response = SimpleNamespace(
        usage=SimpleNamespace(
            input_tokens=4617,
            output_tokens=1260,
            reasoning_tokens=1005,
            total_tokens=6882,
        )
    )

    with patch(
        "hermes_cli.plugins.invoke_hook",
        side_effect=lambda name, **kwargs: captured.append((name, kwargs)),
    ), attribute_auxiliary_usage(agent):
        track_auxiliary_dispatch(
            lambda: response,
            task="compression",
            provider="openrouter",
            model="test-model",
        )

    completed = captured[-1][1]
    assert completed["usage"]["total_tokens"] == 5877
    assert completed["usage"]["provider_reported_total_tokens"] == 6882
    assert completed["usage"]["usage_completeness"] == "partial"
    assert completed["usage"]["usage_warnings"] == [
        "total_does_not_match_input_output"
    ]


def test_auxiliary_retry_attempts_have_independent_terminal_error_events():
    agent = _agent(
        session_id="session-aux-retry",
        _current_turn_id="turn-aux-retry",
        _request_metadata={"source": "dorvis-headless-worker"},
    )
    captured = []

    def fail():
        raise TimeoutError("provider timed out")

    with patch(
        "hermes_cli.plugins.invoke_hook",
        side_effect=lambda name, **kwargs: captured.append((name, kwargs)),
    ), attribute_auxiliary_usage(agent):
        for _ in range(2):
            try:
                track_auxiliary_dispatch(
                    fail,
                    task="title_generation",
                    provider="openrouter",
                    model="provider/model",
                    base_url="https://openrouter.ai/api/v1",
                    request={"method": "POST", "body": {"messages": []}},
                )
            except TimeoutError:
                pass

    starts = [payload for name, payload in captured if name == "pre_auxiliary_api_request"]
    errors = [payload for name, payload in captured if name == "auxiliary_api_request_error"]
    assert len(starts) == len(errors) == 2
    assert len({payload["api_request_id"] for payload in starts}) == 2
    assert {payload["api_request_id"] for payload in starts} == {
        payload["api_request_id"] for payload in errors
    }
    assert all(payload["error"]["type"] == "TimeoutError" for payload in errors)
    assert all(payload["ended_at"] for payload in errors)


def test_missing_usage_is_never_described_as_a_complete_total():
    agent = _agent()

    with attribute_auxiliary_usage(agent):
        record_auxiliary_response(
            SimpleNamespace(id="aux-without-usage", usage=None),
            task="title_generation",
        )
    mark_primary_usage_missing(agent, reason="provider_response_missing_usage")

    usage = snapshot_agent_usage(agent)
    assert usage["input_tokens"] == 100
    assert usage["total_tokens"] == 120
    assert usage["completeness"] == "partial"
    assert usage["warnings"] == [
        "auxiliary:title_generation:provider_response_missing_usage",
        "primary:provider_response_missing_usage",
    ]


def test_auxiliary_total_without_input_output_is_partial_not_zero_input():
    agent = _agent()
    response = SimpleNamespace(
        usage=SimpleNamespace(total_tokens=41),
    )

    with attribute_auxiliary_usage(agent):
        record_auxiliary_response(response, task="web_extract")

    usage = snapshot_agent_usage(agent)
    assert usage["total_tokens"] == 161
    assert usage["input_tokens"] == 100
    assert usage["output_tokens"] == 20
    assert usage["completeness"] == "partial"
    assert usage["warnings"] == [
        "auxiliary:web_extract:input_output_breakdown_missing_or_invalid"
    ]


def test_malformed_auxiliary_counts_are_not_clamped_into_complete_usage():
    agent = _agent()
    response = SimpleNamespace(
        usage=SimpleNamespace(
            prompt_tokens=-2,
            completion_tokens=1.5,
            total_tokens=9,
        ),
    )

    with attribute_auxiliary_usage(agent):
        record_auxiliary_response(response, task="vision")

    usage = snapshot_agent_usage(agent)
    assert usage["total_tokens"] == 129
    assert usage["breakdown"]["auxiliary"]["input_tokens"] == 0
    assert usage["breakdown"]["auxiliary"]["output_tokens"] == 0
    assert usage["completeness"] == "partial"


def test_delegated_rollup_includes_nested_child_aggregate_once():
    parent = _agent()
    child_snapshot = {
        "input_tokens": 55,
        "output_tokens": 11,
        "total_tokens": 66,
        "completeness": "complete",
        "warnings": [],
        "breakdown": {
            "parent": {"input_tokens": 40, "output_tokens": 8, "total_tokens": 48},
            "auxiliary": {"input_tokens": 5, "output_tokens": 1, "total_tokens": 6},
            "delegated": {"input_tokens": 10, "output_tokens": 2, "total_tokens": 12},
        },
    }

    rollup_delegated_usage(parent, child_snapshot, child_session_id="child-1")
    usage = snapshot_agent_usage(parent)

    assert usage["input_tokens"] == 155
    assert usage["output_tokens"] == 31
    assert usage["total_tokens"] == 186
    assert usage["breakdown"]["delegated"] == {
        "input_tokens": 55,
        "output_tokens": 11,
        "total_tokens": 66,
    }


def test_partial_child_makes_parent_aggregate_partial_with_safe_warning():
    parent = _agent()
    rollup_delegated_usage(
        parent,
        {
            "input_tokens": 9,
            "output_tokens": 2,
            "total_tokens": 11,
            "completeness": "partial",
            "warnings": ["primary:provider_response_missing_usage"],
            "breakdown": {},
        },
        child_session_id="secret-looking-session-id-is-not-exported",
    )

    usage = snapshot_agent_usage(parent)
    assert usage["completeness"] == "partial"
    assert usage["warnings"] == ["delegated:child_usage_partial"]


def test_child_total_without_output_component_is_partial():
    parent = _agent()
    rollup_delegated_usage(
        parent,
        {
            "input_tokens": 9,
            "total_tokens": 11,
            "completeness": "complete",
            "warnings": [],
            "breakdown": {},
        },
    )

    usage = snapshot_agent_usage(parent)
    assert usage["completeness"] == "partial"
    assert usage["warnings"] == [
        "delegated:child_input_output_breakdown_missing_or_invalid",
    ]


def test_child_mismatched_total_uses_exact_component_sum():
    parent = _agent()
    rollup_delegated_usage(
        parent,
        {
            "input_tokens": 9,
            "output_tokens": 2,
            "total_tokens": 99,
            "completeness": "complete",
            "warnings": [],
            "breakdown": {},
        },
    )

    usage = snapshot_agent_usage(parent)
    assert usage["breakdown"]["delegated"] == {
        "input_tokens": 9,
        "output_tokens": 2,
        "total_tokens": 11,
    }
    assert usage["completeness"] == "partial"
    assert usage["warnings"] == [
        "delegated:child_total_does_not_match_input_output"
    ]


def test_no_reported_usage_is_unavailable_instead_of_zero_total():
    agent = _agent(
        session_prompt_tokens=0,
        session_completion_tokens=0,
        session_total_tokens=0,
        session_api_calls=0,
    )

    usage = snapshot_agent_usage(agent)
    assert usage["completeness"] == "unavailable"
    assert usage["warnings"] == ["no_provider_usage_reported"]


def test_primary_usage_missing_one_side_is_partial():
    agent = _agent()
    validate_primary_usage_components(
        agent,
        {"prompt_tokens": 8, "total_tokens": 11},
    )

    usage = snapshot_agent_usage(agent)
    assert usage["completeness"] == "partial"
    assert usage["warnings"] == [
        "primary:input_output_breakdown_missing_or_invalid"
    ]


def test_invalid_primary_response_usage_is_retained_before_success_commit():
    agent = _agent(
        session_prompt_tokens=0,
        session_completion_tokens=0,
        session_total_tokens=0,
        session_api_calls=0,
    )
    invalid = SimpleNamespace(
        choices=[],
        usage=SimpleNamespace(
            prompt_tokens=10,
            completion_tokens=2,
            total_tokens=12,
        ),
    )
    valid = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="ok"))],
        usage=SimpleNamespace(
            prompt_tokens=20,
            completion_tokens=3,
            total_tokens=23,
        ),
    )

    assert track_primary_dispatch(agent, lambda: invalid) is invalid
    assert track_primary_dispatch(agent, lambda: valid) is valid
    commit_primary_response(
        agent,
        valid,
        normalize_usage(valid.usage, api_mode="chat_completions"),
    )

    usage = snapshot_agent_usage(agent)
    assert usage["completeness"] == "complete"
    assert usage["breakdown"]["parent"] == {
        "input_tokens": 30,
        "output_tokens": 5,
        "total_tokens": 35,
    }


def test_primary_dispatch_exception_then_success_is_partial():
    agent = _agent(
        session_prompt_tokens=0,
        session_completion_tokens=0,
        session_total_tokens=0,
        session_api_calls=0,
    )

    def _timeout():
        raise TimeoutError("provider timed out")

    try:
        track_primary_dispatch(agent, _timeout)
    except TimeoutError:
        pass
    valid = _response(prompt=7, completion=2, total=9)
    track_primary_dispatch(agent, lambda: valid)
    commit_primary_response(
        agent,
        valid,
        normalize_usage(valid.usage, api_mode="chat_completions"),
    )

    usage = snapshot_agent_usage(agent)
    assert usage["total_tokens"] == 9
    assert usage["completeness"] == "partial"
    assert usage["warnings"] == [
        "primary:dispatched_attempt_usage_unavailable"
    ]


def test_preexisting_interrupt_is_not_mislabeled_as_dispatched_usage():
    agent = _agent(
        session_prompt_tokens=0,
        session_completion_tokens=0,
        session_total_tokens=0,
        session_api_calls=0,
        _interrupt_requested=True,
    )

    try:
        track_primary_dispatch(
            agent,
            lambda: (_ for _ in ()).throw(InterruptedError("already stopped")),
        )
    except InterruptedError:
        pass

    usage = snapshot_agent_usage(agent)
    assert usage["completeness"] == "unavailable"
    assert usage["warnings"] == ["no_provider_usage_reported"]


def test_primary_response_identity_is_provisional_then_committed_once():
    agent = _agent(
        session_prompt_tokens=0,
        session_completion_tokens=0,
        session_total_tokens=0,
        session_api_calls=0,
    )
    response = _response(prompt=4, completion=1, total=5)
    track_primary_dispatch(agent, lambda: response)
    track_primary_dispatch(agent, lambda: response)
    canonical = normalize_usage(response.usage, api_mode="chat_completions")
    assert commit_primary_response(agent, response, canonical) is True
    assert commit_primary_response(agent, response, canonical) is False

    usage = snapshot_agent_usage(agent)
    assert usage["total_tokens"] == 5
    assert usage["breakdown"]["parent"]["input_tokens"] == 4


def test_anthropic_cached_input_is_included_in_provisional_parent_usage():
    agent = _agent(
        session_prompt_tokens=0,
        session_completion_tokens=0,
        session_total_tokens=0,
        session_api_calls=0,
    )
    response = SimpleNamespace(
        usage=SimpleNamespace(
            input_tokens=5,
            output_tokens=2,
            cache_read_input_tokens=7,
            cache_creation_input_tokens=3,
        )
    )

    track_primary_dispatch(agent, lambda: response)
    usage = snapshot_agent_usage(agent)

    assert usage["completeness"] == "complete"
    assert usage["breakdown"]["parent"] == {
        "input_tokens": 15,
        "output_tokens": 2,
        "total_tokens": 17,
    }


def test_mismatched_reported_total_keeps_exact_component_sum():
    agent = _agent(
        session_prompt_tokens=0,
        session_completion_tokens=0,
        session_total_tokens=0,
        session_api_calls=0,
    )
    response = SimpleNamespace(
        usage=SimpleNamespace(
            input_tokens=5,
            output_tokens=2,
            cache_read_input_tokens=7,
            total_tokens=7,
        )
    )

    with attribute_auxiliary_usage(agent):
        record_auxiliary_response(response, task="compression")
    usage = snapshot_agent_usage(agent)

    assert usage["breakdown"]["auxiliary"] == {
        "input_tokens": 12,
        "output_tokens": 2,
        "total_tokens": 14,
    }
    assert usage["completeness"] == "partial"
    assert usage["warnings"] == [
        "auxiliary:compression:total_does_not_match_input_output"
    ]


def test_mapping_usage_normalizes_without_erasing_provisional_counts():
    agent = _agent(
        session_prompt_tokens=0,
        session_completion_tokens=0,
        session_total_tokens=0,
        session_api_calls=0,
    )
    response = {
        "usage": {
            "prompt_tokens": 9,
            "completion_tokens": 2,
            "total_tokens": 11,
            "prompt_tokens_details": {"cached_tokens": 4},
        }
    }

    track_primary_dispatch(agent, lambda: response)
    canonical = normalize_usage(
        response["usage"],
        api_mode="chat_completions",
    )
    commit_primary_response(agent, response, canonical)
    usage = snapshot_agent_usage(agent)

    assert usage["completeness"] == "complete"
    assert usage["breakdown"]["parent"] == {
        "input_tokens": 9,
        "output_tokens": 2,
        "total_tokens": 11,
    }


def test_auxiliary_fallback_after_dispatched_exception_is_partial():
    agent = _agent()

    def _timeout():
        raise TimeoutError("provider timed out")

    with attribute_auxiliary_usage(agent):
        try:
            track_auxiliary_dispatch(_timeout, task="compression")
        except TimeoutError:
            pass
        track_auxiliary_dispatch(
            lambda: _response(prompt=6, completion=2, total=8),
            task="compression",
        )

    usage = snapshot_agent_usage(agent)
    assert usage["breakdown"]["auxiliary"]["total_tokens"] == 8
    assert usage["completeness"] == "partial"
    assert usage["warnings"] == [
        "auxiliary:compression:dispatched_attempt_usage_unavailable"
    ]


def test_auxiliary_stream_observer_terminalizes_after_last_chunk():
    agent = _agent()
    agent.session_id = "session-stream"
    agent._current_task_id = "task-stream"
    agent._current_turn_id = "turn-stream"
    agent._request_metadata = {"source": "dorvis-web"}
    terminal = _response(prompt=4, completion=2, total=6)
    events = []

    with (
        attribute_auxiliary_usage(agent),
        patch(
            "hermes_cli.plugins.invoke_hook",
            side_effect=lambda name, **kwargs: events.append((name, kwargs)),
        ),
    ):
        stream = track_auxiliary_stream_dispatch(
            lambda: iter([SimpleNamespace(usage=None), terminal]),
            task="moa_aggregator",
            provider="openrouter",
            model="aggregate-model",
            request={"stream": True},
        )
        assert list(stream)[-1] is terminal

    assert [name for name, _ in events] == [
        "pre_auxiliary_api_request",
        "post_auxiliary_api_request",
    ]
    assert events[0][1]["api_request_id"] == events[1][1]["api_request_id"]
    assert events[1][1]["usage"]["input_tokens"] == 4
    assert events[1][1]["usage"]["output_tokens"] == 2
    assert events[1][1]["usage"]["total_tokens"] == 6
    assert events[1][1]["usage"]["cost_status"] == "unknown"
    assert "cost_usd" not in events[1][1]["usage"]


def test_auxiliary_stream_completed_response_is_returned_and_terminalized():
    agent = _agent(
        session_id="session-completed-stream",
        _current_task_id="task-completed-stream",
        _current_turn_id="turn-completed-stream",
        _request_metadata={"source": "dorvis-web"},
    )
    completed = _response(prompt=4, completion=2, total=6)
    completed.choices = [SimpleNamespace(finish_reason="stop")]
    events = []

    with (
        attribute_auxiliary_usage(agent),
        patch(
            "hermes_cli.plugins.invoke_hook",
            side_effect=lambda name, **kwargs: events.append((name, kwargs)),
        ),
    ):
        result = track_auxiliary_stream_dispatch(
            lambda: completed,
            task="compression",
            provider="openrouter",
            model="completed-model",
            request={"stream": True},
            completed_response_predicate=lambda value: hasattr(value, "choices"),
        )

    assert result is completed
    assert [name for name, _ in events] == [
        "pre_auxiliary_api_request",
        "post_auxiliary_api_request",
    ]
    assert events[1][1]["usage"]["total_tokens"] == 6
    usage = snapshot_agent_usage(agent)
    assert usage["breakdown"]["auxiliary"] == {
        "input_tokens": 4,
        "output_tokens": 2,
        "total_tokens": 6,
    }


def test_auxiliary_attempt_observer_sees_hidden_failure_and_response():
    events = []

    def _observe(**event):
        events.append(event)

    response = _response(prompt=2, completion=1, total=3)
    with observe_auxiliary_attempts(_observe):
        try:
            track_auxiliary_dispatch(
                lambda: (_ for _ in ()).throw(TimeoutError()),
                task="moa_reference",
            )
        except TimeoutError:
            pass
        track_auxiliary_dispatch(lambda: response, task="moa_reference")

    assert events == [
        {
            "response": None,
            "reason": "dispatched_attempt_usage_unavailable",
        },
        {"response": response, "reason": None},
    ]


def test_primary_accounting_failure_cannot_replace_provider_success(monkeypatch):
    from agent import runtime_usage

    agent = _agent()
    response = _response()
    monkeypatch.setattr(
        runtime_usage,
        "record_primary_dispatch_response",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            RuntimeError("accounting broke")
        ),
    )

    assert track_primary_dispatch(agent, lambda: response) is response
    usage = snapshot_agent_usage(agent)
    assert usage["completeness"] == "partial"
    assert usage["warnings"] == ["primary:usage_accounting_failed"]


def test_auxiliary_accounting_failure_cannot_replace_sync_or_async_success(
    monkeypatch,
):
    from agent import runtime_usage

    agent = _agent()
    sync_response = _response(prompt=2, completion=1, total=3)
    async_response = _response(prompt=4, completion=1, total=5)
    monkeypatch.setattr(
        runtime_usage,
        "_record_tracked_auxiliary_response",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            RuntimeError("accounting broke")
        ),
    )

    async def _async_response():
        return async_response

    with attribute_auxiliary_usage(agent):
        assert (
            track_auxiliary_dispatch(
                lambda: sync_response,
                task="compression",
            )
            is sync_response
        )
        assert (
            asyncio.run(
                track_auxiliary_dispatch_async(
                    _async_response,
                    task="title_generation",
                )
            )
            is async_response
        )

    usage = snapshot_agent_usage(agent)
    assert usage["completeness"] == "partial"
    assert usage["warnings"] == [
        "auxiliary:compression:usage_accounting_failed",
        "auxiliary:title_generation:usage_accounting_failed",
    ]
