from types import SimpleNamespace

from agent.codex_runtime import _record_codex_app_server_compaction
from agent.conversation_compression import COMPACTION_STATUS, compress_context
from agent.transports.codex_app_server_session import TurnResult


class FakeCodexSession:
    def __init__(self, result):
        self.result = result
        self.calls = 0
        self.closed = False

    def compact_thread(self):
        self.calls += 1
        return self.result

    def close(self):
        self.closed = True


class DummyAgent:
    def __init__(
        self,
        result,
        *,
        auto_compaction="native",
    ):
        self.api_mode = "codex_app_server"
        self.codex_app_server_auto_compaction = auto_compaction
        self.session_id = "hermes-session-1"
        self.platform = "cli"
        self._cached_system_prompt = "cached prompt"
        self._codex_session = FakeCodexSession(result)
        self.context_compressor = SimpleNamespace(
            compression_count=0,
            last_compression_rough_tokens=0,
            last_prompt_tokens=123,
            last_completion_tokens=45,
            awaiting_real_usage_after_compression=False,
        )
        self.statuses = []
        self.warnings = []
        self.events = []
        self.built_prompts = []
        self.session_prompt_tokens = 0
        self.session_completion_tokens = 0
        self.session_total_tokens = 0
        self.session_api_calls = 0
        self.session_input_tokens = 0
        self.session_output_tokens = 0
        self.session_cache_read_tokens = 0
        self.session_cache_write_tokens = 0
        self.session_reasoning_tokens = 0
        self.session_estimated_cost_usd = 0.0
        self._session_db = None
        self.model = "gpt-5-codex"
        self.provider = "openai-codex"
        self.base_url = "https://chatgpt.com/backend-api/codex"
        self.api_key = "test"
        from agent.runtime_usage import initialize_agent_usage_attribution

        initialize_agent_usage_attribution(self)

    def _emit_status(self, message):
        self.statuses.append(message)

    def _emit_warning(self, message):
        self.warnings.append(message)

    def _build_system_prompt(self, system_message):
        self.built_prompts.append(system_message)
        return "built prompt"

    def event_callback(self, name, payload):
        self.events.append((name, payload))


def test_codex_app_server_native_auto_mode_leaves_thread_compaction_to_codex():
    agent = DummyAgent(
        TurnResult(thread_id="thread-1", turn_id="compact-turn-1")
    )
    messages = [{"role": "user", "content": "hi"}]

    returned, prompt = compress_context(
        agent,
        messages,
        "system",
        approx_tokens=100000,
        task_id="test",
    )

    assert returned is messages
    assert prompt == "cached prompt"
    assert agent._codex_session.calls == 0
    assert agent.context_compressor.compression_count == 0
    assert agent.events == []


def test_codex_app_server_manual_compression_routes_to_codex_thread():
    agent = DummyAgent(
        TurnResult(thread_id="thread-1", turn_id="compact-turn-1")
    )
    messages = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
    ]

    returned, prompt = compress_context(
        agent,
        messages,
        "system",
        approx_tokens=100000,
        task_id="test",
        force=True,
    )

    assert returned is messages
    assert prompt == "cached prompt"
    assert agent._codex_session.calls == 1
    assert agent.context_compressor.compression_count == 1
    assert agent.context_compressor.last_compression_rough_tokens == 100000
    # This minimal fake compressor does not implement update_from_response(),
    # so the runtime preserves its existing pending-usage bookkeeping here.
    assert agent.context_compressor.last_prompt_tokens == -1
    assert agent.context_compressor.last_completion_tokens == 0
    assert agent.context_compressor.awaiting_real_usage_after_compression is True
    assert agent.events == [
        (
            "session:compress",
            {
                "platform": "cli",
                "session_id": "hermes-session-1",
                "old_session_id": "",
                "in_place": False,
                "compression_count": 1,
                "runtime": "codex_app_server",
                "thread_id": "thread-1",
                "turn_id": "compact-turn-1",
            },
        )
    ]


def test_codex_app_server_hermes_mode_auto_compression_routes_to_codex_thread():
    agent = DummyAgent(
        TurnResult(thread_id="thread-1", turn_id="compact-turn-1"),
        auto_compaction="hermes",
    )
    messages = [{"role": "user", "content": "hi"}]

    returned, prompt = compress_context(
        agent,
        messages,
        "system",
        approx_tokens=100000,
    )

    assert returned is messages
    assert prompt == "cached prompt"
    assert agent._codex_session.calls == 1
    assert agent.context_compressor.compression_count == 1


def test_codex_app_server_compression_failure_preserves_bookkeeping():
    agent = DummyAgent(TurnResult(error="compact failed"))
    messages = [{"role": "user", "content": "hi"}]

    returned, prompt = compress_context(
        agent,
        messages,
        "system",
        approx_tokens=100000,
        force=True,
    )

    assert returned is messages
    assert prompt == "cached prompt"
    assert agent._codex_session.calls == 1
    assert agent.context_compressor.compression_count == 0
    assert agent.context_compressor.last_prompt_tokens == 123
    assert agent.warnings
    from agent.runtime_usage import snapshot_agent_usage

    usage = snapshot_agent_usage(agent)
    assert usage["completeness"] == "unavailable"
    assert usage["warnings"] == [
        "no_provider_usage_reported",
        "primary:codex_compaction_dispatched_usage_unavailable",
    ]


def test_codex_app_server_success_without_compaction_usage_is_partial():
    agent = DummyAgent(TurnResult(thread_id="thread-1", turn_id="compact-turn-1"))

    compress_context(
        agent,
        [{"role": "user", "content": "hi"}],
        "system",
        approx_tokens=100000,
        force=True,
    )

    from agent.runtime_usage import snapshot_agent_usage

    usage = snapshot_agent_usage(agent)
    assert usage["completeness"] == "partial"
    assert usage["warnings"] == ["primary:codex_turn_missing_usage"]


def test_failed_compaction_preserves_reported_usage_as_captured(monkeypatch):
    agent = DummyAgent(
        TurnResult(
            error="compact failed",
            token_usage_last={
                "inputTokens": 5,
                "cachedInputTokens": 3,
                "outputTokens": 2,
                "reasoningOutputTokens": 1,
                "totalTokens": 10,
            },
        )
    )
    monkeypatch.setattr(
        "agent.usage_pricing.estimate_usage_cost",
        lambda *_args, **_kwargs: SimpleNamespace(
            amount_usd=None,
            status="unknown",
            source="none",
        ),
    )

    compress_context(
        agent,
        [{"role": "user", "content": "hi"}],
        "system",
        approx_tokens=100000,
        force=True,
    )

    from agent.runtime_usage import snapshot_agent_usage

    usage = snapshot_agent_usage(agent)
    assert usage["breakdown"]["parent"] == {
        "input_tokens": 8,
        "output_tokens": 2,
        "total_tokens": 10,
    }
    assert usage["completeness"] == "partial"
    assert usage["warnings"] == [
        "primary:codex_compaction_dispatched_usage_unavailable"
    ]


def test_codex_app_server_native_compaction_notice_emits_status_and_event():
    agent = DummyAgent(
        TurnResult(thread_id="thread-1", turn_id="normal-turn-1")
    )
    turn = TurnResult(
        thread_id="thread-1",
        turn_id="normal-turn-1",
        compacted=True,
    )

    recorded = _record_codex_app_server_compaction(agent, turn)

    assert recorded is True
    assert agent.context_compressor.compression_count == 1
    assert agent.statuses == [COMPACTION_STATUS]
    assert agent.events == [
        (
            "session:compress",
            {
                "platform": "cli",
                "session_id": "hermes-session-1",
                "old_session_id": "",
                "in_place": False,
                "compression_count": 1,
                "runtime": "codex_app_server",
                "thread_id": "thread-1",
                "turn_id": "normal-turn-1",
            },
        )
    ]


def test_codex_native_boundary_clears_stale_hermes_fallback_streak():
    from unittest.mock import patch

    from agent.context_compressor import ContextCompressor

    with patch(
        "agent.context_compressor.get_model_context_length",
        return_value=100_000,
    ):
        compressor = ContextCompressor(model="test-model", quiet_mode=True)
    compressor._fallback_compression_streak = 1
    compressor._last_summary_fallback_used = True

    agent = DummyAgent(
        TurnResult(thread_id="thread-1", turn_id="normal-turn-1")
    )
    agent.context_compressor = compressor
    turn = TurnResult(
        thread_id="thread-1",
        turn_id="normal-turn-1",
        compacted=True,
    )

    assert _record_codex_app_server_compaction(agent, turn) is True
    assert compressor._fallback_compression_streak == 0
    assert compressor._verify_compaction_cleared_threshold is True
