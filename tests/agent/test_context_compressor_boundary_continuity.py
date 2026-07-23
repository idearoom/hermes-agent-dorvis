"""Regression coverage for the assistant handoff at a compression boundary."""

from unittest.mock import MagicMock, patch

from agent.context_compressor import ContextCompressor


def _compressor() -> ContextCompressor:
    with patch(
        "agent.context_compressor.get_model_context_length",
        return_value=100_000,
    ):
        return ContextCompressor(
            model="test/model",
            threshold_percent=0.70,
            protect_first_n=1,
            protect_last_n=3,
            quiet_mode=True,
        )


def _response(content: str):
    response = MagicMock()
    response.choices = [MagicMock()]
    response.choices[0].message.content = content
    return response


def _messages(completed_handoff: str, active_request: str):
    return [
        {"role": "system", "content": "system prompt"},
        {"role": "user", "content": "inspect the supplier rows"},
        {"role": "assistant", "content": "I will inspect them."},
        {"role": "user", "content": "apply the approved margin update"},
        {"role": "assistant", "content": "running the approved update"},
        {"role": "user", "content": "please continue"},
        {"role": "assistant", "content": completed_handoff},
        {"role": "user", "content": active_request},
    ]


def _summary_prompt(mock_call) -> str:
    return mock_call.call_args.kwargs["messages"][0]["content"]


def test_immediately_prior_assistant_handoff_is_visible_to_summarizer():
    compressor = _compressor()
    completed_handoff = (
        "BOUNDARY-COMPLETE-71: the margin update is complete, verified, "
        "and left unpublished."
    )
    active_request = "Now compare pricing for a different customer."

    with patch(
        "agent.context_compressor.call_llm",
        return_value=_response("updated checkpoint"),
    ) as mock_call:
        compressed = compressor.compress(
            _messages(completed_handoff, active_request),
            current_tokens=80_000,
        )

    prompt = _summary_prompt(mock_call)
    assert completed_handoff in prompt
    assert active_request in prompt
    assert any(
        completed_handoff in str(message.get("content", ""))
        for message in compressed
    )


def test_consecutive_compressions_each_include_the_latest_completed_handoff():
    compressor = _compressor()
    first_handoff = (
        "BOUNDARY-COMPLETE-72: the first operation is complete and verified."
    )

    with patch(
        "agent.context_compressor.call_llm",
        return_value=_response("first checkpoint"),
    ) as first_call:
        first_result = compressor.compress(
            _messages(first_handoff, "Prepare the follow-up analysis."),
            current_tokens=80_000,
        )

    assert first_handoff in _summary_prompt(first_call)

    second_handoff = (
        "BOUNDARY-COMPLETE-73: the follow-up analysis is complete and verified."
    )
    resumed = [
        *first_result,
        {"role": "user", "content": "Start the next independent task."},
        {"role": "assistant", "content": second_handoff},
        {"role": "user", "content": "Continue."},
    ]
    with patch(
        "agent.context_compressor.call_llm",
        return_value=_response("second checkpoint"),
    ) as second_call:
        compressor.compress(resumed, current_tokens=80_000)

    assert second_handoff in _summary_prompt(second_call)
