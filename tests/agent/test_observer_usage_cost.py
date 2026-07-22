from types import SimpleNamespace

from run_agent import AIAgent
from agent.runtime_usage import _observer_usage


def _agent(*, provider="openai", model="gpt-5.6-sol", base_url="https://api.openai.com/v1"):
    agent = AIAgent.__new__(AIAgent)
    agent.provider = provider
    agent.api_mode = "chat_completions"
    agent.model = model
    agent.base_url = base_url
    return agent


def test_primary_observer_usage_includes_honest_cost_disposition():
    response = SimpleNamespace(
        usage=SimpleNamespace(
            prompt_tokens=100,
            completion_tokens=20,
            total_tokens=120,
        )
    )

    summary = _agent()._usage_summary_for_api_request_hook(response)

    assert summary["input_tokens"] == 100
    assert summary["output_tokens"] == 20
    assert summary["total_tokens"] == 120
    assert summary["cost_status"] in {"actual", "estimated", "included", "unknown"}
    if summary["cost_status"] in {"actual", "estimated"}:
        assert summary["cost_usd"] > 0
        assert summary["cost_source"] != "none"


def test_primary_observer_usage_keeps_unknown_cost_absent():
    response = SimpleNamespace(
        usage=SimpleNamespace(
            prompt_tokens=1,
            completion_tokens=1,
            total_tokens=2,
        )
    )

    summary = _agent(
        provider="unknown-provider",
        model="unpriced-model",
        base_url="https://provider.invalid/v1",
    )._usage_summary_for_api_request_hook(response)

    assert summary["cost_status"] == "unknown"
    assert "cost_usd" not in summary


def test_auxiliary_observer_usage_keeps_cache_reasoning_and_cost_distinct():
    response = SimpleNamespace(
        usage=SimpleNamespace(
            prompt_tokens=100,
            completion_tokens=20,
            total_tokens=120,
            prompt_tokens_details=SimpleNamespace(cached_tokens=40),
            completion_tokens_details=SimpleNamespace(reasoning_tokens=5),
        )
    )

    summary = _observer_usage(
        response,
        {
            "provider": "openai",
            "api_mode": "chat_completions",
            "model": "gpt-5.6-sol",
            "base_url": "https://api.openai.com/v1",
        },
    )

    assert summary["input_tokens"] == 60
    assert summary["cache_read_tokens"] == 40
    assert summary["output_tokens"] == 20
    assert summary["reasoning_tokens"] == 5
    assert summary["total_tokens"] == 120
    assert summary["cost_status"] in {"actual", "estimated", "included", "unknown"}
