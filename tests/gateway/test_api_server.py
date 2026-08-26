"""
Tests for the OpenAI-compatible API server gateway adapter.

Tests cover:
- Chat Completions endpoint (request parsing, response format)
- Responses API endpoint (request parsing, response format)
- previous_response_id chaining (store/retrieve)
- Auth (valid key, invalid key, no key configured)
- /v1/models endpoint
- /health endpoint
- System prompt extraction
- Error handling (invalid JSON, missing fields)
"""

import asyncio
import base64
import json
import logging
import os
import stat
import threading
import sys
import time
import types
import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.api_server import (
    APIServerAdapter,
    ResponseStore,
    ThreadSafeAsyncQueue,
    _IdempotencyCache,
    _append_request_identity_prompt,
    _derive_chat_session_id,
    _hermes_version,
    _redact_api_error_text,
    _request_agent_overrides,
    _responses_usage_payload,
    _session_usage_snapshot,
    check_api_server_requirements,
    cors_middleware,
    make_response_store,
    security_headers_middleware,
)


# ---------------------------------------------------------------------------
# check_api_server_requirements
# ---------------------------------------------------------------------------


class TestCheckRequirements:

    @patch("gateway.platforms.api_server.AIOHTTP_AVAILABLE", False)
    def test_returns_false_without_aiohttp(self):
        assert check_api_server_requirements() is False


# ---------------------------------------------------------------------------
# _redact_api_error_text — guards every outward error site (envelopes, SSE
# error events, cron-endpoint 500 bodies) that routes raw exception text to
# authenticated HTTP clients. #37733
# ---------------------------------------------------------------------------


class TestRedactApiErrorText:
    def test_masks_secret_value_but_preserves_structure(self):
        secret = "sk-api-server-leak-1234567890"
        out = _redact_api_error_text(Exception(f"auth failed OPENAI_API_KEY={secret}"))
        assert secret not in out
        assert "OPENAI_API_KEY=" in out

    def test_redacts_regardless_of_global_redaction_setting(self):
        # force=True must mask even when global redaction is disabled.
        secret = "sk-forced-redaction-0987654321"
        with patch("agent.redact._REDACT_ENABLED", False):
            out = _redact_api_error_text(Exception(f"boom AWS_SECRET_ACCESS_KEY={secret}"))
        assert secret not in out

    def test_limit_truncates_after_redaction(self):
        assert len(_redact_api_error_text("x" * 500, limit=50)) == 50


# ---------------------------------------------------------------------------
# ResponseStore
# ---------------------------------------------------------------------------


class TestResponseStore:
    def test_configured_postgres_init_failure_is_fatal(self, monkeypatch):
        monkeypatch.setenv(
            "HERMES_RESPONSE_STORE_DSN",
            "postgresql://configured-but-unavailable",
        )

        with (
            patch(
                "gateway.platforms.response_store_pg.PgResponseStore",
                side_effect=RuntimeError("database unavailable"),
            ),
            patch("gateway.platforms.api_server.ResponseStore") as sqlite_store,
            pytest.raises(
                RuntimeError,
                match="HERMES_RESPONSE_STORE_POSTGRES_REQUIRED",
            ),
        ):
            make_response_store()

        sqlite_store.assert_not_called()

    def test_put_and_get(self):
        store = ResponseStore(max_size=10)
        store.put("resp_1", {"output": "hello"})
        assert store.get("resp_1") == {"output": "hello"}

    def test_get_missing_returns_none(self):
        store = ResponseStore(max_size=10)
        assert store.get("resp_missing") is None

    def test_lru_eviction(self):
        store = ResponseStore(max_size=3)
        store.put("resp_1", {"output": "one"})
        store.put("resp_2", {"output": "two"})
        store.put("resp_3", {"output": "three"})
        # Adding a 4th should evict resp_1
        store.put("resp_4", {"output": "four"})
        assert store.get("resp_1") is None
        assert store.get("resp_2") is not None
        assert len(store) == 3

    def test_lru_never_evicts_an_active_owned_response(self):
        store = ResponseStore(max_size=1)
        assert store.claim(
            "resp_active",
            {"response": {"id": "resp_active", "status": "in_progress"}},
            owner_id="owner-a",
            owner_epoch="epoch-a",
        )
        store.put(
            "resp_done_1",
            {"response": {"id": "resp_done_1", "status": "completed"}},
        )
        assert store.get("resp_active") is not None
        assert store.get("resp_done_1") is not None

        store.put(
            "resp_done_2",
            {"response": {"id": "resp_done_2", "status": "completed"}},
        )
        assert store.get("resp_active") is not None
        assert store.get("resp_done_1") is None
        assert store.get("resp_done_2") is not None

    def test_terminal_owned_transitions_participate_in_lru_eviction(self):
        store = ResponseStore(max_size=1)
        for response_id, conversation in (
            ("resp_stream_1", "chat-1"),
            ("resp_stream_2", "chat-2"),
        ):
            assert store.claim(
                response_id,
                {"response": {"id": response_id, "status": "in_progress"}},
                owner_id="owner-a",
                owner_epoch=f"epoch-{response_id}",
                conversation=conversation,
            )
            assert store.transition(
                response_id,
                {"response": {"id": response_id, "status": "completed"}},
                owner_id="owner-a",
                owner_epoch=f"epoch-{response_id}",
                terminal=True,
            )

        assert store.get("resp_stream_1") is None
        assert store.get_conversation("chat-1") is None
        assert store.get("resp_stream_2") is not None
        assert store.get_conversation("chat-2") == "resp_stream_2"


    def test_delete_clears_conversation_mapping(self):
        """Deleting a response also removes conversation mappings that reference it."""
        store = ResponseStore(max_size=10)
        store.put("resp_1", {"output": "hello"})
        store.set_conversation("chat-a", "resp_1")
        assert store.get_conversation("chat-a") == "resp_1"
        store.delete("resp_1")
        assert store.get_conversation("chat-a") is None

    def test_terminal_transition_is_monotonic_across_store_connections(self, tmp_path):
        """A stale owner completion cannot overwrite an accepted cancel."""
        db_path = str(tmp_path / "response-state.db")
        canceller = ResponseStore(max_size=10, db_path=db_path)
        stale_writer = ResponseStore(max_size=10, db_path=db_path)
        owner_id = "gateway-owner-a"
        owner_epoch = "response-epoch-1"
        active = {
            "response": {"id": "resp_race", "status": "in_progress"},
            "conversation_history": [],
        }
        cancelled = {
            "response": {"id": "resp_race", "status": "cancelled"},
            "conversation_history": [],
        }
        completed = {
            "response": {"id": "resp_race", "status": "completed"},
            "conversation_history": [],
        }
        try:
            assert canceller.claim(
                "resp_race",
                active,
                owner_id=owner_id,
                owner_epoch=owner_epoch,
            )
            assert canceller.transition(
                "resp_race",
                cancelled,
                owner_id=owner_id,
                owner_epoch=owner_epoch,
                terminal=True,
            )
            assert not stale_writer.transition(
                "resp_race",
                completed,
                owner_id=owner_id,
                owner_epoch=owner_epoch,
                terminal=True,
            )
            assert stale_writer.get("resp_race")["response"]["status"] == "cancelled"
        finally:
            canceller.close()
            stale_writer.close()

    def test_stale_owned_response_recovers_to_terminal_incomplete(self):
        store = ResponseStore(max_size=10)
        active = {
            "response": {"id": "resp_orphan", "status": "in_progress"},
            "conversation_history": [{"role": "user", "content": "hello"}],
        }
        assert store.claim(
            "resp_orphan",
            active,
            owner_id="dead-owner",
            owner_epoch="dead-epoch",
        )

        assert not store.recover_stale_owned(
            "resp_orphan", stale_before=time.time() - 60
        )
        assert store.recover_stale_owned(
            "resp_orphan", stale_before=time.time() + 1
        )

        recovered = store.get("resp_orphan")
        assert recovered["response"]["status"] == "incomplete"
        assert recovered["response"]["incomplete_details"] == {
            "reason": "owner_lost"
        }
        assert store.get_control("resp_orphan")["terminal"] is True
        assert not store.heartbeat(
            "resp_orphan", owner_id="dead-owner", owner_epoch="dead-epoch"
        )

    def test_owner_heartbeat_prevents_stale_recovery(self):
        store = ResponseStore(max_size=10)
        assert store.claim(
            "resp_live",
            {"response": {"id": "resp_live", "status": "in_progress"}},
            owner_id="live-owner",
            owner_epoch="live-epoch",
        )
        assert store.heartbeat(
            "resp_live", owner_id="live-owner", owner_epoch="live-epoch"
        )
        assert not store.recover_stale_owned(
            "resp_live", stale_before=time.time() - 1
        )
        assert store.get_control("resp_live")["terminal"] is False

    def test_legacy_ownerless_active_response_expires(self):
        store = ResponseStore(max_size=10)
        store.put(
            "resp_legacy",
            {"response": {"id": "resp_legacy", "status": "in_progress"}},
        )
        with store._conn:
            store._conn.execute(
                """UPDATE responses
                   SET owner_heartbeat_at = NULL, accessed_at = 1
                   WHERE response_id = ?""",
                ("resp_legacy",),
            )

        assert store.get_control("resp_legacy") == {
            "owner_id": None,
            "owner_epoch": None,
            "terminal": False,
        }
        assert store.recover_stale_owned(
            "resp_legacy", stale_before=2
        )
        assert store.get("resp_legacy")["response"] == {
            "id": "resp_legacy",
            "status": "incomplete",
            "incomplete_details": {"reason": "owner_lost"},
        }
        assert store.delete_terminal("resp_legacy") == "deleted"

    def test_conversation_mapping_cannot_dangle_after_terminal_delete(self):
        store = ResponseStore(max_size=10)
        store.put(
            "resp_terminal",
            {"response": {"id": "resp_terminal", "status": "completed"}},
        )
        assert store.set_conversation("chat-a", "resp_terminal")
        assert store.delete_terminal("resp_terminal") == "deleted"

        # A stale mapping write racing behind DELETE must not recreate the
        # reference after the response row is gone.
        assert not store.set_conversation("chat-a", "resp_terminal")
        assert store.get_conversation("chat-a") is None


# ---------------------------------------------------------------------------
# _IdempotencyCache
# ---------------------------------------------------------------------------


class TestIdempotencyCache:
    @pytest.mark.asyncio
    async def test_concurrent_same_key_and_fingerprint_runs_once(self):
        cache = _IdempotencyCache()
        gate = asyncio.Event()
        started = asyncio.Event()
        calls = 0

        async def compute():
            nonlocal calls
            calls += 1
            started.set()
            await gate.wait()
            return ("response", {"total_tokens": 1})

        first = asyncio.create_task(cache.get_or_set("idem-key", "fp-1", compute))
        second = asyncio.create_task(cache.get_or_set("idem-key", "fp-1", compute))

        await started.wait()
        assert calls == 1

        gate.set()
        first_result, second_result = await asyncio.gather(first, second)

        assert first_result == second_result == ("response", {"total_tokens": 1})


# ---------------------------------------------------------------------------
# Adapter initialization
# ---------------------------------------------------------------------------


class TestAdapterInit:
    def test_default_config(self):
        config = PlatformConfig(enabled=True)
        adapter = APIServerAdapter(config)
        assert adapter._host == "127.0.0.1"
        assert adapter._port == 8642
        assert adapter._api_key == ""
        assert adapter.platform == Platform.API_SERVER

    def test_custom_config_from_extra(self):
        config = PlatformConfig(
            enabled=True,
            extra={
                "host": "0.0.0.0",
                "port": 9999,
                "key": "sk-test",
                "cors_origins": ["http://localhost:3000"],
            },
        )
        adapter = APIServerAdapter(config)
        assert adapter._host == "0.0.0.0"
        assert adapter._port == 9999
        assert adapter._api_key == "sk-test"
        assert adapter._cors_origins == ("http://localhost:3000",)


    def test_create_agent_forwards_runtime_config(self, monkeypatch):
        captured = {}

        class FakeAgent:
            def __init__(self, **kwargs):
                captured.update(kwargs)

        monkeypatch.setattr("run_agent.AIAgent", FakeAgent)
        monkeypatch.setattr(
            "gateway.run._resolve_runtime_agent_kwargs",
            lambda: {
                "provider": "openai-codex",
                "base_url": "https://example.test/v1",
                "api_mode": "codex_responses",
            },
        )
        monkeypatch.setattr("gateway.run._resolve_gateway_model", lambda: "gpt-5.5")
        monkeypatch.setattr(
            "gateway.run._load_gateway_config",
            lambda: {
                "agent": {"reasoning_effort": "xhigh"},
                "checkpoints": {
                    "enabled": True,
                    "max_snapshots": 7,
                    "max_total_size_mb": 321,
                    "max_file_size_mb": 4,
                },
            },
        )
        monkeypatch.setattr(
            "gateway.run.GatewayRunner._load_reasoning_config",
            staticmethod(lambda model="": {"enabled": True, "effort": "xhigh"}),
        )
        monkeypatch.setattr("gateway.run.GatewayRunner._load_fallback_model", staticmethod(lambda: None))
        monkeypatch.setattr("hermes_cli.tools_config._get_platform_tools", lambda *_: set())

        adapter = APIServerAdapter(PlatformConfig(enabled=True))
        monkeypatch.setattr(adapter, "_ensure_session_db", lambda: None)

        agent = adapter._create_agent(session_id="api-session")

        assert isinstance(agent, FakeAgent)
        assert captured["reasoning_config"] == {"enabled": True, "effort": "xhigh"}
        assert captured["checkpoints_enabled"] is True
        assert captured["checkpoint_max_snapshots"] == 7
        assert captured["checkpoint_max_total_size_mb"] == 321
        assert captured["checkpoint_max_file_size_mb"] == 4


# ---------------------------------------------------------------------------
# Auth checking
# ---------------------------------------------------------------------------


class TestAuth:
    def test_no_key_configured_allows_all(self):
        config = PlatformConfig(enabled=True)
        adapter = APIServerAdapter(config)
        mock_request = MagicMock()
        mock_request.headers = {}
        assert adapter._check_auth(mock_request) is None


    def test_non_ascii_bearer_token_returns_401_not_500(self):
        """A non-ASCII byte in the bearer token must be rejected with 401, not
        crash the handler: hmac.compare_digest raises TypeError on a str with
        non-ASCII characters, and the token is raw client input."""
        config = PlatformConfig(enabled=True, extra={"key": "sk-test123"})
        adapter = APIServerAdapter(config)
        mock_request = MagicMock()
        mock_request.headers = {"Authorization": "Bearer ské-not-the-key"}
        result = adapter._check_auth(mock_request)  # must not raise
        assert result is not None
        assert result.status == 401


# ---------------------------------------------------------------------------
# Concurrency cap (gateway.api_server.max_concurrent_runs) — #7483
# ---------------------------------------------------------------------------


class TestConcurrencyCap:

    def test_resolve_reads_config_value(self):
        cfg = {"gateway": {"api_server": {"max_concurrent_runs": 3}}}
        with patch("hermes_cli.config.load_config", return_value=cfg):
            assert APIServerAdapter._resolve_max_concurrent_runs() == 3


    def test_under_cap_returns_none(self):
        adapter = _make_adapter()
        adapter._max_concurrent_runs = 5
        adapter._inflight_agent_runs = 2
        assert adapter._concurrency_limited_response() is None

    def test_at_cap_returns_429_with_retry_after(self):
        adapter = _make_adapter()
        adapter._max_concurrent_runs = 3
        adapter._inflight_agent_runs = 3
        resp = adapter._concurrency_limited_response()
        assert resp is not None
        assert resp.status == 429
        assert resp.headers.get("Retry-After")


# ---------------------------------------------------------------------------
# Helpers for HTTP tests
# ---------------------------------------------------------------------------


def _make_adapter(api_key: str = "", cors_origins=None) -> APIServerAdapter:
    """Create an adapter with optional API key."""
    extra = {}
    if api_key:
        extra["key"] = api_key
    if cors_origins is not None:
        extra["cors_origins"] = cors_origins
    config = PlatformConfig(enabled=True, extra=extra)
    return APIServerAdapter(config)


def _create_app(adapter: APIServerAdapter) -> web.Application:
    """Create the aiohttp app from the adapter (without starting the full server)."""
    mws = [mw for mw in (cors_middleware, security_headers_middleware) if mw is not None]
    app = web.Application(middlewares=mws)
    app["api_server_adapter"] = adapter
    app.router.add_get("/health", adapter._handle_health)
    app.router.add_get("/health/detailed", adapter._handle_health_detailed)
    app.router.add_get("/v1/health", adapter._handle_health)
    app.router.add_get("/v1/models", adapter._handle_models)
    app.router.add_get("/api/model/options", adapter._handle_model_options)
    app.router.add_get("/v1/capabilities", adapter._handle_capabilities)
    app.router.add_get("/v1/skills", adapter._handle_skills)
    app.router.add_get("/v1/toolsets", adapter._handle_toolsets)
    app.router.add_post("/api/sessions/{session_id}/chat", adapter._handle_session_chat)
    app.router.add_post("/api/sessions/{session_id}/chat/stream", adapter._handle_session_chat_stream)
    app.router.add_post("/v1/chat/completions", adapter._handle_chat_completions)
    app.router.add_post("/v1/responses", adapter._handle_responses)
    app.router.add_get("/v1/responses/{response_id}", adapter._handle_get_response)
    app.router.add_post(
        "/v1/responses/{response_id}/cancel", adapter._handle_cancel_response
    )
    app.router.add_delete("/v1/responses/{response_id}", adapter._handle_delete_response)
    app.router.add_post(
        "/api/platforms/{platform}/events",
        adapter._handle_platform_event_callback,
    )
    return app


class _FakeGoogleChatAdapter:
    def __init__(self, *, verify_ok: bool = True, verify_code: str = ""):
        self.verify_ok = verify_ok
        self.verify_code = verify_code
        self.dispatched = []

    def verify_http_event_request(self, auth_header: str):
        self.auth_header = auth_header
        return self.verify_ok, self.verify_code

    async def dispatch_http_event(self, payload):
        self.dispatched.append(payload)
        return {"ok": True}


def _parse_sse_events(body: str) -> list[dict]:
    events = []
    for line in body.splitlines():
        if not line.startswith("data: "):
            continue
        try:
            events.append(json.loads(line[len("data: "):]))
        except json.JSONDecodeError:
            continue
    return events


@pytest.fixture
def adapter():
    return _make_adapter()


@pytest.fixture
def auth_adapter():
    return _make_adapter(api_key="sk-secret")


def _make_api_agent(final_response: str = "ok") -> MagicMock:
    mock_agent = MagicMock()
    mock_agent.run_conversation.return_value = {
        "final_response": final_response,
        "messages": [],
        "api_calls": 1,
    }
    mock_agent.session_prompt_tokens = 1
    mock_agent.session_completion_tokens = 2
    mock_agent.session_total_tokens = 3
    mock_agent._session_messages = [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": final_response},
    ]
    mock_agent.shutdown_memory_provider = MagicMock()
    mock_agent.close = MagicMock()
    return mock_agent


# ---------------------------------------------------------------------------
# Adapter internals
# ---------------------------------------------------------------------------


class TestAgentExecution:
    @pytest.mark.asyncio
    async def test_chained_response_attempts_use_fresh_non_cumulative_usage(self, adapter):
        first = _make_api_agent("first")
        first.session_prompt_tokens = 10
        first.session_completion_tokens = 2
        first.session_total_tokens = 12
        second = _make_api_agent("repair")
        second.session_prompt_tokens = 3
        second.session_completion_tokens = 1
        second.session_total_tokens = 4

        with patch.object(adapter, "_create_agent", side_effect=[first, second]):
            _, first_usage = await adapter._run_agent(
                user_message="initial",
                conversation_history=[],
                session_id="session-initial",
            )
            _, repair_usage = await adapter._run_agent(
                user_message="repair",
                conversation_history=[
                    {"role": "assistant", "content": "first"},
                ],
                session_id="session-repair",
            )

        assert first_usage["total_tokens"] == 12
        assert repair_usage["total_tokens"] == 4
        assert repair_usage["input_tokens"] == 3

    @pytest.mark.asyncio
    async def test_run_agent_usage_aggregates_auxiliary_and_delegated_tokens(self, adapter):
        from agent.runtime_usage import initialize_agent_usage_attribution

        mock_agent = _make_api_agent()
        initialize_agent_usage_attribution(mock_agent)
        mock_agent.session_api_calls = 1
        mock_agent.session_auxiliary_input_tokens = 5
        mock_agent.session_auxiliary_output_tokens = 1
        mock_agent.session_auxiliary_total_tokens = 6
        mock_agent.session_auxiliary_response_count = 1
        mock_agent.session_delegated_input_tokens = 13
        mock_agent.session_delegated_output_tokens = 4
        mock_agent.session_delegated_total_tokens = 17
        mock_agent.session_delegated_response_count = 1

        with patch.object(adapter, "_create_agent", return_value=mock_agent):
            _, usage = await adapter._run_agent(
                user_message="hello",
                conversation_history=[],
                session_id="session-aggregate",
            )

        assert usage["input_tokens"] == 19
        assert usage["output_tokens"] == 7
        assert usage["total_tokens"] == 26
        assert usage["completeness"] == "complete"
        assert usage["breakdown"] == {
            "parent": {"input_tokens": 1, "output_tokens": 2, "total_tokens": 3},
            "auxiliary": {"input_tokens": 5, "output_tokens": 1, "total_tokens": 6},
            "delegated": {"input_tokens": 13, "output_tokens": 4, "total_tokens": 17},
        }

    @pytest.mark.asyncio
    async def test_run_agent_uses_session_id_as_task_id(self, adapter):
        mock_agent = MagicMock()
        mock_agent.run_conversation.return_value = {"final_response": "ok"}
        mock_agent.session_prompt_tokens = 1
        mock_agent.session_completion_tokens = 2
        mock_agent.session_total_tokens = 3

        model_options = {"reasoning": {"enabled": False}, "fast": False}
        with patch.object(adapter, "_create_agent", return_value=mock_agent) as mock_create_agent:
            result, usage = await adapter._run_agent(
                user_message="hello",
                conversation_history=[],
                session_id="session-123",
                requested_model="MiniMax-M3",
                requested_provider="minimax",
                model_options=model_options,
            )

        # _run_agent annotates result with the effective agent.session_id
        # when it's a real string, so the response-header writer can track
        # compression-triggered session rotations (#16938). The mock agent
        # here doesn't set an explicit session_id string so the guard skips
        # the annotation — header will fall back to the provided session_id.
        assert result["final_response"] == "ok"
        assert usage["input_tokens"] == 1
        assert usage["output_tokens"] == 2
        assert usage["total_tokens"] == 3
        assert usage["scope"] == "run_aggregate"
        assert usage["completeness"] == "complete"
        assert usage["warnings"] == []
        create_kwargs = mock_create_agent.call_args.kwargs
        assert create_kwargs["requested_model"] == "MiniMax-M3"
        assert create_kwargs["requested_provider"] == "minimax"
        assert create_kwargs["model_options"] == model_options
        mock_agent.run_conversation.assert_called_once_with(
            user_message="hello",
            conversation_history=[],
            task_id="session-123",
        )

    @pytest.mark.asyncio
    async def test_run_agent_usage_includes_context_compaction_and_cost(self, adapter):
        class _FakeCompressor:
            last_prompt_tokens = 4096
            context_length = 8192
            compression_count = 2

        class _FakeCost:
            status = "estimated"
            amount_usd = 0.0123

        mock_agent = MagicMock()
        mock_agent.run_conversation.return_value = {"final_response": "ok"}
        mock_agent.session_prompt_tokens = 10
        mock_agent.session_completion_tokens = 5
        mock_agent.session_total_tokens = 15
        mock_agent.session_input_tokens = 11
        mock_agent.session_output_tokens = 5
        mock_agent.session_cache_read_tokens = 3
        mock_agent.session_cache_write_tokens = 2
        mock_agent.context_compressor = _FakeCompressor()
        mock_agent.model = "gpt-5.5"
        mock_agent.provider = "openai"
        mock_agent.base_url = None

        with (
            patch.object(adapter, "_create_agent", return_value=mock_agent),
            patch("agent.usage_pricing.estimate_usage_cost", return_value=_FakeCost()) as mock_cost,
        ):
            result, usage = await adapter._run_agent(
                user_message="hello",
                conversation_history=[],
                session_id="session-123",
            )

        assert result["final_response"] == "ok"
        assert usage["input_tokens"] == 10
        assert usage["output_tokens"] == 5
        assert usage["total_tokens"] == 15
        assert usage["context_used"] == 4096
        assert usage["context_max"] == 8192
        assert usage["context_percent"] == 50
        assert usage["compressions"] == 2
        assert usage["cost_status"] == "estimated"
        assert usage["cost_usd"] == 0.0123
        assert usage["cache_read_tokens"] == 3
        assert usage["cache_write_tokens"] == 2
        assert mock_cost.call_count == 1

    @pytest.mark.asyncio
    async def test_run_agent_shuts_down_memory_provider_after_success(self, adapter):
        mock_agent = _make_api_agent()

        with patch.object(adapter, "_create_agent", return_value=mock_agent):
            result, usage = await adapter._run_agent(
                user_message="hello",
                conversation_history=[],
                session_id="session-123",
            )

        assert result["final_response"] == "ok"
        assert usage["input_tokens"] == 1
        assert usage["output_tokens"] == 2
        assert usage["total_tokens"] == 3
        assert usage["completeness"] == "complete"
        mock_agent.shutdown_memory_provider.assert_called_once_with(mock_agent._session_messages)
        mock_agent.close.assert_not_called()

    @pytest.mark.asyncio
    async def test_run_agent_shuts_down_memory_provider_after_exception(self, adapter):
        mock_agent = _make_api_agent()
        mock_agent.run_conversation.side_effect = RuntimeError("boom")

        with patch.object(adapter, "_create_agent", return_value=mock_agent):
            with pytest.raises(RuntimeError, match="boom"):
                await adapter._run_agent(
                    user_message="hello",
                    conversation_history=[],
                    session_id="session-123",
                )

        mock_agent.shutdown_memory_provider.assert_called_once_with(mock_agent._session_messages)
        mock_agent.close.assert_not_called()

    @pytest.mark.asyncio
    async def test_run_agent_shuts_down_memory_provider_after_cancellation(self, adapter):
        ready = threading.Event()
        interrupted = threading.Event()
        mock_agent = _make_api_agent("cancelled")

        def _interrupt(message=None):
            interrupted.set()

        def _blocking_run(user_message=None, conversation_history=None, task_id=None):
            ready.set()
            interrupted.wait(timeout=3)
            return {"final_response": "cancelled", "messages": [], "api_calls": 1}

        mock_agent.interrupt = MagicMock(side_effect=_interrupt)
        mock_agent.run_conversation.side_effect = _blocking_run
        agent_ref = [None]

        with patch.object(adapter, "_create_agent", return_value=mock_agent):
            task = asyncio.create_task(
                adapter._run_agent(
                    user_message="hello",
                    conversation_history=[],
                    session_id="session-123",
                    agent_ref=agent_ref,
                )
            )
            started = await asyncio.to_thread(ready.wait, 3)
            if not started:
                task.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await task
            assert started
            assert agent_ref[0] is mock_agent

            mock_agent.interrupt("test cancellation")
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task

            for _ in range(20):
                if mock_agent.shutdown_memory_provider.called:
                    break
                await asyncio.sleep(0.05)

        mock_agent.shutdown_memory_provider.assert_called_once_with(mock_agent._session_messages)
        assert agent_ref[0] is None
        mock_agent.close.assert_not_called()

    @pytest.mark.asyncio
    async def test_run_agent_sets_and_clears_process_ownership_markers(self, adapter):
        """#76188 review: this surface runs its own agent lifecycle outside
        TurnRunner, so it needs its own baseline snapshot/clear — verify the
        markers _reap_disconnected_agent_processes() reads are actually
        populated during the turn and cleared once it finishes."""
        mock_agent = MagicMock()
        mock_agent.session_prompt_tokens = 0
        mock_agent.session_completion_tokens = 0
        mock_agent.session_total_tokens = 0
        captured = {}

        def _capture_markers(**_kwargs):
            captured["task_id"] = mock_agent._gateway_turn_process_task_id
            captured["baseline"] = mock_agent._gateway_turn_process_baseline
            return {"final_response": "ok"}

        mock_agent.run_conversation.side_effect = _capture_markers

        with patch.object(adapter, "_create_agent", return_value=mock_agent):
            await adapter._run_agent(
                user_message="hello",
                conversation_history=[],
                session_id="session-456",
                requested_model="MiniMax-M3",
                requested_provider="minimax",
                model_options={"reasoning": {"enabled": False}, "fast": False},
            )

        assert captured["task_id"] == "session-456"
        assert isinstance(captured["baseline"], frozenset)
        # Turn completed normally — markers must be cleared so a disconnect
        # arriving after this point can't reap work this turn left running.
        assert mock_agent._gateway_turn_process_task_id == ""
        assert mock_agent._gateway_turn_process_baseline == frozenset()


class TestDisconnectedAgentReap:
    """#76188 review: SSE disconnect handlers must reap only the background
    processes the disconnected turn created, and must no-op when no turn
    ownership was ever recorded on the agent."""

    def test_reaps_baseline_diff_for_owned_turn(self, monkeypatch):
        from gateway.platforms.api_server import _reap_disconnected_agent_processes
        from tools.process_registry import process_registry

        calls = []
        monkeypatch.setattr(
            process_registry,
            "kill_started_since",
            lambda task_id, baseline, *, source: calls.append(
                (task_id, baseline, source)
            )
            or 1,
        )
        agent = types.SimpleNamespace(
            _gateway_turn_process_task_id="session-abc",
            _gateway_turn_process_baseline=frozenset({"proc-1"}),
        )

        _reap_disconnected_agent_processes(agent)

        deadline = time.time() + 1.0
        while not calls and time.time() < deadline:
            time.sleep(0.01)
        assert calls == [
            ("session-abc", frozenset({"proc-1"}), "api_server_sse_disconnect")
        ]

    def test_noop_when_agent_has_no_ownership_markers(self, monkeypatch):
        from gateway.platforms.api_server import _reap_disconnected_agent_processes
        from tools.process_registry import process_registry

        calls = []
        monkeypatch.setattr(
            process_registry,
            "kill_started_since",
            lambda *a, **k: calls.append(True),
        )
        agent = types.SimpleNamespace(
            _gateway_turn_process_task_id="",
            _gateway_turn_process_baseline=None,
        )

        _reap_disconnected_agent_processes(agent)

        time.sleep(0.1)
        assert calls == []

    def test_stale_epoch_skips_reap_when_newer_run_claimed_task_id(self, monkeypatch):
        """#76188 follow-up: concurrent API runs can share a client-provided
        session_id (same task_id). A disconnecting run whose epoch has been
        superseded must NOT kill the newer run's processes."""
        from gateway.platforms.api_server import (
            _clear_turn_process_ownership,
            _publish_turn_process_ownership,
            _reap_disconnected_agent_processes,
        )
        from tools.process_registry import process_registry

        calls = []
        monkeypatch.setattr(
            process_registry,
            "kill_started_since",
            lambda *a, **k: calls.append(True) or 1,
        )
        monkeypatch.setattr(
            process_registry, "snapshot_running_ids", lambda _tid: frozenset()
        )

        run_a = types.SimpleNamespace()
        run_b = types.SimpleNamespace()
        _publish_turn_process_ownership(run_a, "shared-session")
        # Run B claims the same session_id — supersedes A's epoch.
        _publish_turn_process_ownership(run_b, "shared-session")

        _reap_disconnected_agent_processes(run_a)
        time.sleep(0.2)
        assert calls == [], "stale run A must not reap run B's processes"

        # Run B disconnecting IS current — its reap proceeds.
        _reap_disconnected_agent_processes(run_b)
        deadline = time.time() + 1.0
        while not calls and time.time() < deadline:
            time.sleep(0.01)
        assert calls == [True]
        _clear_turn_process_ownership(run_b)

    def test_reap_proceeds_when_own_clear_pruned_the_epoch_entry(self, monkeypatch):
        """A missing epoch entry (the abandoned run's own finally already
        cleared it) means no newer claimant — the reap must proceed using a
        pre-captured marker snapshot, or the leak survives."""
        from gateway.platforms.api_server import (
            _clear_turn_process_ownership,
            _publish_turn_process_ownership,
            _reap_disconnected_agent_processes,
        )
        from tools.process_registry import process_registry

        calls = []
        monkeypatch.setattr(
            process_registry,
            "kill_started_since",
            lambda *a, **k: calls.append(True) or 1,
        )
        monkeypatch.setattr(
            process_registry, "snapshot_running_ids", lambda _tid: frozenset()
        )

        run = types.SimpleNamespace()
        _publish_turn_process_ownership(run, "solo-session")
        # Simulate the disconnect handler capturing the agent while the
        # worker's finally clears ownership: snapshot markers, then clear.
        stale_view = types.SimpleNamespace(
            _gateway_turn_process_task_id=run._gateway_turn_process_task_id,
            _gateway_turn_process_baseline=run._gateway_turn_process_baseline,
            _gateway_turn_process_epoch=run._gateway_turn_process_epoch,
        )
        _clear_turn_process_ownership(run)

        _reap_disconnected_agent_processes(stale_view)
        deadline = time.time() + 1.0
        while not calls and time.time() < deadline:
            time.sleep(0.01)
        assert calls == [True]

    def test_publish_and_clear_ownership_roundtrip(self, monkeypatch):
        from gateway.platforms.api_server import (
            _TURN_PROCESS_EPOCHS,
            _clear_turn_process_ownership,
            _publish_turn_process_ownership,
        )
        from tools.process_registry import process_registry

        monkeypatch.setattr(
            process_registry,
            "snapshot_running_ids",
            lambda tid: frozenset({f"pre-{tid}"}),
        )

        agent = types.SimpleNamespace()
        _publish_turn_process_ownership(agent, "sess-rt")
        assert agent._gateway_turn_process_task_id == "sess-rt"
        assert agent._gateway_turn_process_baseline == frozenset({"pre-sess-rt"})
        assert isinstance(agent._gateway_turn_process_epoch, int)
        assert "sess-rt" in _TURN_PROCESS_EPOCHS

        _clear_turn_process_ownership(agent)
        assert agent._gateway_turn_process_task_id == ""
        assert agent._gateway_turn_process_baseline == frozenset()
        assert agent._gateway_turn_process_epoch is None
        # Entry pruned — dict stays bounded to in-flight runs.
        assert "sess-rt" not in _TURN_PROCESS_EPOCHS

    @pytest.mark.asyncio
    async def test_stop_run_reaps_owned_processes(self, adapter, monkeypatch):
        """POST /v1/runs/{id}/stop abandons the run — it must reap the
        background processes that run created (#76115 sibling surface)."""
        from gateway.platforms.api_server import _publish_turn_process_ownership
        from tools.process_registry import process_registry

        calls = []
        monkeypatch.setattr(
            process_registry,
            "kill_started_since",
            lambda task_id, baseline, *, source: calls.append(
                (task_id, baseline, source)
            )
            or 1,
        )
        monkeypatch.setattr(
            process_registry, "snapshot_running_ids", lambda _tid: frozenset()
        )

        agent = MagicMock()
        _publish_turn_process_ownership(agent, "run-stop-sess")
        adapter._active_run_agents["run_x"] = agent

        request = MagicMock()
        request.match_info = {"run_id": "run_x"}
        resp = await adapter._handle_stop_run(request)
        assert resp.status == 200

        deadline = time.time() + 1.0
        while not calls and time.time() < deadline:
            time.sleep(0.01)
        assert calls == [("run-stop-sess", frozenset(), "api_server_run_stop")]
        agent.interrupt.assert_called_once()


class TestRunEventCallback:

    @pytest.mark.asyncio
    async def test_subagent_events_redact_secrets_and_carry_child_session(self, adapter):
        """Free-text fields (goal/summary/output_tail/preview) must pass the
        forced secret redaction before hitting the public /v1/runs stream,
        and child_session_id must survive the allowlist so clients can
        correlate the child's session."""
        run_id = "run_subagent_redact"
        loop = asyncio.get_running_loop()
        queue = asyncio.Queue()
        adapter._run_streams[run_id] = queue
        adapter._run_statuses.pop(run_id, None)

        callback = adapter._make_run_event_callback(run_id, loop)
        secret = "sk-proj-abcdef1234567890abcdef1234567890abcdef12"
        callback(
            "subagent.complete",
            preview=f"leaked {secret}",
            goal=f"use key {secret} to fetch data",
            subagent_id="deleg_999",
            child_session_id="child-sess-42",
            status="completed",
            summary=f"exported OPENAI_API_KEY={secret} then ran",
            output_tail=f"env shows {secret}",
        )

        event = await asyncio.wait_for(queue.get(), timeout=1.0)
        assert event["child_session_id"] == "child-sess-42"
        for field in ("preview", "goal", "summary", "output_tail"):
            assert secret not in event[field], field


# ---------------------------------------------------------------------------
# _session_usage_snapshot — prompt-cache split and honest context reporting.
#
# The envelope's aggregate ``input_tokens`` is prompt-inclusive, so consumers
# that want "fresh input" need the cached buckets published alongside it. The
# context trio must never be fabricated: a run-cumulative counter is not a
# context size, and reporting one clamps a consumer's context bar to 100% and
# fakes a "compaction imminent" warning.
# ---------------------------------------------------------------------------


class _UsageSnapshotAgent:
    """Minimal agent-shaped object for _session_usage_snapshot unit tests."""

    def __init__(self, **attrs):
        self.session_prompt_tokens = 100
        self.session_completion_tokens = 20
        self.session_total_tokens = 120
        for name, value in attrs.items():
            setattr(self, name, value)


class _StubCompressor:
    def __init__(self, last_prompt_tokens, context_length=8192, compression_count=2):
        self.last_prompt_tokens = last_prompt_tokens
        self.context_length = context_length
        self.compression_count = compression_count


class TestSessionUsageSnapshotCacheSplit:
    def test_publishes_parent_scope_cache_split(self):
        agent = _UsageSnapshotAgent(
            session_cache_read_tokens=70,
            session_cache_write_tokens=5,
        )

        usage = _session_usage_snapshot(agent)

        assert usage["input_tokens"] == 100
        assert usage["output_tokens"] == 20
        assert usage["total_tokens"] == 120
        assert usage["cache_read_tokens"] == 70
        assert usage["cache_write_tokens"] == 5
        # Existing keys are untouched by the additive split.
        assert usage["scope"] == "run_aggregate"
        assert usage["completeness"] == "complete"
        assert usage["warnings"] == []
        assert usage["breakdown"]["parent"] == {
            "input_tokens": 100,
            "output_tokens": 20,
            "total_tokens": 120,
        }

    def test_cache_split_reports_zero_when_counters_absent(self):
        usage = _session_usage_snapshot(_UsageSnapshotAgent())

        assert usage["cache_read_tokens"] == 0
        assert usage["cache_write_tokens"] == 0

    def test_cache_split_ignores_malformed_counters(self):
        agent = _UsageSnapshotAgent(
            session_cache_read_tokens="lots",
            session_cache_write_tokens=None,
        )

        usage = _session_usage_snapshot(agent)

        assert usage["cache_read_tokens"] == 0
        assert usage["cache_write_tokens"] == 0

    def test_responses_payload_passes_cache_split_through(self):
        payload = _responses_usage_payload(
            {
                "input_tokens": 100,
                "output_tokens": 20,
                "total_tokens": 120,
                "cache_read_tokens": 70,
                "cache_write_tokens": 5,
            }
        )

        assert payload["cache_read_tokens"] == 70
        assert payload["cache_write_tokens"] == 5


class TestSessionUsageSnapshotContext:
    def test_reports_context_trio_from_a_real_reading(self):
        agent = _UsageSnapshotAgent(
            context_compressor=_StubCompressor(4096),
        )

        usage = _session_usage_snapshot(agent)

        assert usage["context_used"] == 4096
        assert usage["context_max"] == 8192
        assert usage["context_percent"] == 50
        assert usage["compressions"] == 2

    def test_omits_context_trio_before_any_provider_reading(self):
        """A 0 reading means "not measured yet", not "empty context"."""
        agent = _UsageSnapshotAgent(
            context_compressor=_StubCompressor(0),
        )

        usage = _session_usage_snapshot(agent)

        assert "context_used" not in usage
        assert "context_max" not in usage
        assert "context_percent" not in usage
        # Compaction reporting is independent of the context reading.
        assert usage["compressions"] == 2

    def test_omits_context_trio_for_post_compaction_sentinel(self):
        """The -1 sentinel parks the reading until real usage arrives."""
        agent = _UsageSnapshotAgent(
            context_compressor=_StubCompressor(-1),
        )

        usage = _session_usage_snapshot(agent)

        assert "context_used" not in usage
        assert "context_max" not in usage
        assert "context_percent" not in usage
        assert usage["compressions"] == 2

    def test_omits_context_trio_when_engine_does_not_track_the_reading(self):
        class _NoReadingEngine:
            context_length = 8192
            compression_count = 0

        agent = _UsageSnapshotAgent(context_compressor=_NoReadingEngine())

        usage = _session_usage_snapshot(agent)

        assert "context_used" not in usage
        assert "context_max" not in usage
        assert "context_percent" not in usage
        assert usage["compressions"] == 0

    def test_never_substitutes_run_cumulative_totals_for_context_used(self):
        """session_total_tokens is tokens processed, not a context size.

        Substituting it pinned the reported context at 100% for any run whose
        cumulative processing exceeded the window, even with a nearly empty
        conversation.
        """
        agent = _UsageSnapshotAgent(
            session_prompt_tokens=900_000,
            session_completion_tokens=100_000,
            session_total_tokens=1_000_000,
            context_compressor=_StubCompressor(0),
        )

        usage = _session_usage_snapshot(agent)

        assert usage["total_tokens"] == 1_000_000
        assert "context_used" not in usage
        assert "context_percent" not in usage

    def test_responses_payload_omits_absent_context_keys(self):
        payload = _responses_usage_payload(
            {
                "input_tokens": 100,
                "output_tokens": 20,
                "total_tokens": 120,
                "compressions": 3,
            }
        )

        assert payload["compressions"] == 3
        assert "context_used" not in payload
        assert "context_max" not in payload
        assert "context_percent" not in payload


# ---------------------------------------------------------------------------
# /health endpoint
# ---------------------------------------------------------------------------


class TestHealthEndpoint:
    def test_store_attachment_emits_exact_non_secret_boot_attestation(
        self, adapter, caplog
    ):
        class _AttestingStore:
            def __init__(self, payload):
                self._payload = payload

            def storage_attestation(self):
                return dict(self._payload)

        adapter._response_store = _AttestingStore({"backend": "postgres"})
        session_store = types.SimpleNamespace(
            _db=_AttestingStore(
                {
                    "backend": "postgres",
                    "schema_version": 26,
                    "surface_marker": (
                        "cd2cb9ee351693e62e9dc8e425885a4a08148551d9577d506f4a11be4a715d5f"
                    ),
                }
            )
        )

        caplog.set_level(logging.WARNING)
        adapter.set_session_store(session_store)

        assert (
            "DORVIS_STORAGE_ATTESTATION "
            '{"response_store":{"backend":"postgres"},'
            '"session_store":{"backend":"postgres","schema_version":26,'
            '"surface_marker":"cd2cb9ee351693e62e9dc8e425885a4a08148551d9577d506f4a11be4a715d5f"}}'
        ) in caplog.messages

    @pytest.mark.asyncio
    async def test_health_reports_typed_storage_attestations(self, adapter):
        class _AttestingStore:
            def __init__(self, payload):
                self._payload = payload

            def storage_attestation(self):
                return dict(self._payload)

        adapter._response_store = _AttestingStore({"backend": "postgres"})
        adapter._session_store = types.SimpleNamespace(
            _db=_AttestingStore(
                {
                    "backend": "postgres",
                    "schema_version": 26,
                    "surface_marker": "marker-v26",
                }
            )
        )

        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.get("/health")
            assert resp.status == 200
            data = await resp.json()

        assert data["response_store"] == {"backend": "postgres"}
        assert data["session_store"] == {
            "backend": "postgres",
            "schema_version": 26,
            "surface_marker": "marker-v26",
        }

    @pytest.mark.asyncio
    async def test_security_headers_present(self, adapter):
        """Responses should include basic security headers."""
        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.get("/health")
            assert resp.status == 200
            assert resp.headers.get("Content-Security-Policy") == "default-src 'none'; frame-ancestors 'none'"
            assert resp.headers.get("Permissions-Policy") == "camera=(), microphone=(), geolocation=()"
            assert resp.headers.get("Strict-Transport-Security") == "max-age=31536000; includeSubDomains"
            assert resp.headers.get("X-Content-Type-Options") == "nosniff"
            assert resp.headers.get("X-Frame-Options") == "DENY"
            assert resp.headers.get("X-XSS-Protection") == "0"
            assert resp.headers.get("Referrer-Policy") == "no-referrer"


    @pytest.mark.asyncio
    async def test_health_reports_version(self, adapter):
        """GET /health must expose a non-empty version so orchestrators (e.g.
        AgentOS) can read the gateway version without scraping. Regression
        guard for the missing-version gap."""
        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.get("/health")
            assert resp.status == 200
            data = await resp.json()
            assert "version" in data
            assert isinstance(data["version"], str)
            assert data["version"] != ""

    @pytest.mark.asyncio
    async def test_health_reports_source_revision(self, adapter):
        """GET /health exposes the runtime source revision for deploy audits."""
        source_revision = {
            "commit": "a" * 40,
            "source": "env:HERMES_SOURCE_COMMIT",
        }

        app = _create_app(adapter)
        with patch("gateway.status.get_source_revision", return_value=source_revision):
            async with TestClient(TestServer(app)) as cli:
                resp = await cli.get("/health")
                assert resp.status == 200
                data = await resp.json()
                assert data["source_revision"] == source_revision

    @pytest.mark.asyncio
    async def test_v1_health_alias_returns_ok(self, adapter):
        """GET /v1/health should return the same response as /health."""
        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.get("/v1/health")
            assert resp.status == 200
            data = await resp.json()
            assert data["status"] == "ok"
            assert data["platform"] == "hermes-agent"
            assert data.get("version")

# ---------------------------------------------------------------------------
# /health/detailed endpoint
# ---------------------------------------------------------------------------


class TestHealthDetailedEndpoint:
    @pytest.mark.asyncio
    async def test_health_detailed_returns_ok(self, adapter):
        """GET /health/detailed returns status, platform, and runtime fields."""
        app = _create_app(adapter)
        with patch("gateway.status.read_runtime_status", return_value={
            "gateway_state": "running",
            "platforms": {"telegram": {"state": "connected"}},
            "active_agents": 2,
            "exit_reason": None,
            "updated_at": "2026-04-14T00:00:00Z",
        }), patch("gateway.run._resolve_gateway_model", return_value="test/model"):
            async with TestClient(TestServer(app)) as cli:
                resp = await cli.get("/health/detailed")
                assert resp.status == 200
                data = await resp.json()
                assert data["status"] == "ok"
                assert data["platform"] == "hermes-agent"
                assert data["gateway_state"] == "running"
                assert data["platforms"] == {"telegram": {"state": "connected"}}
                assert data["active_agents"] == 2
                # Derived busy/drainable: this endpoint is served BY the live
                # gateway, so running + 2 agents ⇒ busy and drainable.
                assert data["gateway_busy"] is True
                assert data["gateway_drainable"] is True
                assert isinstance(data["pid"], int)
                assert "updated_at" in data

    @pytest.mark.asyncio
    async def test_health_detailed_reports_source_revision(self, adapter):
        source_revision = {
            "commit": "b" * 40,
            "source": "git",
        }

        app = _create_app(adapter)
        with patch("gateway.status.read_runtime_status", return_value=None), \
             patch("gateway.status.get_source_revision", return_value=source_revision):
            async with TestClient(TestServer(app)) as cli:
                resp = await cli.get("/health/detailed")
                assert resp.status == 200
                data = await resp.json()
                assert data["source_revision"] == source_revision

    @pytest.mark.asyncio
    async def test_health_detailed_no_runtime_status(self, adapter):
        """When gateway_state.json is missing, fields are None."""
        app = _create_app(adapter)
        with patch("gateway.status.read_runtime_status", return_value=None):
            async with TestClient(TestServer(app)) as cli:
                resp = await cli.get("/health/detailed")
                assert resp.status == 200
                data = await resp.json()
                assert data["status"] == "degraded"
                assert data["readiness"]["checks"]["gateway"]["status"] == "degraded"
                assert data["gateway_state"] is None
                assert data["platforms"] == {}
                # No runtime file ⇒ state None ⇒ not busy, not drainable.
                assert data["gateway_busy"] is False
                assert data["gateway_drainable"] is False

    @pytest.mark.asyncio
    async def test_health_detailed_requires_auth(self, auth_adapter):
        """Detailed health must not leak runtime state without Bearer auth."""
        app = _create_app(auth_adapter)
        with patch("gateway.status.read_runtime_status", return_value=None):
            async with TestClient(TestServer(app)) as cli:
                resp = await cli.get("/health/detailed")
                assert resp.status == 401

    @pytest.mark.asyncio
    async def test_health_detailed_allows_authenticated_request(self, auth_adapter):
        app = _create_app(auth_adapter)
        headers = {"Authorization": f"Bearer {auth_adapter._api_key}"}
        with patch("gateway.status.read_runtime_status", return_value={"gateway_state": "running"}):
            async with TestClient(TestServer(app)) as cli:
                resp = await cli.get("/health/detailed", headers=headers)
                assert resp.status == 200

    @pytest.mark.asyncio
    async def test_health_detailed_reports_runtime_readiness(self, adapter):
        """Detailed health exposes bounded readiness probes without changing /health."""
        app = _create_app(adapter)
        expected = {
            "status": "degraded",
            "checks": {
                "state_db": {"status": "ok"},
                "config": {"status": "degraded", "detail": "invalid config"},
            },
        }
        with patch("gateway.status.read_runtime_status", return_value={"gateway_state": "running"}), \
             patch("gateway.platforms.api_server.collect_runtime_readiness", return_value=expected):
            async with TestClient(TestServer(app)) as cli:
                resp = await cli.get("/health/detailed")
                assert resp.status == 200
                data = await resp.json()
                assert data["status"] == "degraded"
                assert data["readiness"] == expected

    @pytest.mark.asyncio
    async def test_public_health_does_not_run_readiness_probes(self, adapter):
        app = _create_app(adapter)
        with patch("gateway.platforms.api_server.collect_runtime_readiness") as probe:
            async with TestClient(TestServer(app)) as cli:
                resp = await cli.get("/health")
                assert resp.status == 200
                assert (await resp.json())["status"] == "ok"
        probe.assert_not_called()


    def test_readiness_work_counts_include_stopping_runs(self, adapter):
        """Regression: _handle_stop_run() sets status="stopping" and holds it
        there — cooperatively, with no hard timeout — until the agent notices
        the interrupt and the task actually exits. A run in that window is
        still doing real executor-thread work and must count as active,
        the same as "running"; excluding it undercounts active_api_runs for
        the whole (now-unbounded) cooperative-stop duration."""
        adapter._run_statuses = {
            "queued": {"status": "queued"},
            "running": {"status": "running"},
            "approval": {"status": "waiting_for_approval"},
            "stopping": {"status": "stopping"},
            "done": {"status": "completed"},
            "cancelled": {"status": "cancelled"},
        }

        with patch("tools.process_registry.process_registry.completion_queue.qsize", return_value=0), \
             patch("tools.async_delegation.active_count", return_value=0):
            assert adapter._readiness_work_counts() == (4, 0, 0)


# ---------------------------------------------------------------------------
# /v1/models endpoint
# ---------------------------------------------------------------------------


class TestModelsEndpoint:
    @pytest.mark.asyncio
    async def test_models_returns_hermes_agent(self, adapter):
        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.get("/v1/models")
            assert resp.status == 200
            data = await resp.json()
            assert data["object"] == "list"
            assert len(data["data"]) == 1
            assert data["data"][0]["id"] == "hermes-agent"
            assert data["data"][0]["owned_by"] == "hermes"

    @pytest.mark.asyncio
    async def test_models_returns_profile_name(self):
        """When running under a named profile, /v1/models advertises the profile name."""
        with patch("gateway.platforms.api_server.APIServerAdapter._resolve_model_name", return_value="lucas"):
            adapter = _make_adapter()
        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.get("/v1/models")
            assert resp.status == 200
            data = await resp.json()
            assert data["data"][0]["id"] == "lucas"
            assert data["data"][0]["root"] == "lucas"


    def test_resolve_model_name_default_profile(self):
        """Default profile falls back to 'hermes-agent'."""
        with patch("hermes_cli.profiles.get_active_profile_name", return_value="default"):
            assert APIServerAdapter._resolve_model_name("") == "hermes-agent"


    @pytest.mark.asyncio
    async def test_model_options_returns_shared_inventory(self, adapter, monkeypatch):
        """GET /api/model/options builds the shared picker payload off-loop."""
        from hermes_cli import inventory

        ctx = object()
        payload = {
            "providers": [{"slug": "nous", "name": "Nous Portal", "models": ["gpt-5.5"]}],
            "model": "gpt-5.5",
            "provider": "nous",
        }
        seen = {"thread_calls": 0}

        monkeypatch.setattr(inventory, "load_picker_context", lambda: ctx)

        def fake_build_model_options_payload(received_ctx, **kwargs):
            seen["ctx"] = received_ctx
            seen["kwargs"] = kwargs
            return payload

        async def fake_to_thread(func, *args, **kwargs):
            seen["thread_calls"] += 1
            return func(*args, **kwargs)

        monkeypatch.setattr(
            inventory,
            "build_model_options_payload",
            fake_build_model_options_payload,
        )
        monkeypatch.setattr(
            "gateway.platforms.api_server.asyncio.to_thread",
            fake_to_thread,
        )

        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.get("/api/model/options?refresh=true")
            assert resp.status == 200
            data = await resp.json()

        assert data == payload
        assert seen["thread_calls"] == 1
        assert seen["ctx"] is ctx
        assert seen["kwargs"] == {
            "include_unconfigured": True,
            "refresh": True,
        }


# ---------------------------------------------------------------------------
# /v1/capabilities endpoint
# ---------------------------------------------------------------------------


class TestCapabilitiesEndpoint:
    @pytest.mark.asyncio
    async def test_capabilities_advertises_plugin_safe_contract(self, adapter):
        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.get("/v1/capabilities")
            assert resp.status == 200
            data = await resp.json()
            assert data["object"] == "hermes.api_server.capabilities"
            assert data["platform"] == "hermes-agent"
            assert data["model"] == "hermes-agent"
            assert data["auth"]["type"] == "bearer"
            assert data["auth"]["required"] is False
            assert data["runtime"]["mode"] == "server_agent"
            assert data["runtime"]["tool_execution"] == "server"
            assert data["runtime"]["split_runtime"] is False
            assert "API-server host" in data["runtime"]["description"]
            assert data["features"]["chat_completions"] is True
            assert data["features"]["run_status"] is True
            assert data["features"]["run_events_sse"] is True
            assert data["features"]["model_options"] is True
            assert data["features"]["session_continuity_header"] == "X-Hermes-Session-Id"
            assert data["endpoints"]["run_status"]["path"] == "/v1/runs/{run_id}"
            assert data["endpoints"]["model_options"] == {"method": "GET", "path": "/api/model/options"}
            assert data["endpoints"]["skills"] == {"method": "GET", "path": "/v1/skills"}
            assert data["endpoints"]["toolsets"] == {"method": "GET", "path": "/v1/toolsets"}


# ---------------------------------------------------------------------------
# /v1/skills and /v1/toolsets endpoints
# ---------------------------------------------------------------------------


class TestSkillsEndpoint:
    @pytest.mark.asyncio
    async def test_skills_returns_list_envelope(self, adapter):
        fake_skills = [
            {"name": "github", "description": "GitHub workflow skill", "category": "github"},
            {"name": "ascii-art", "description": "ASCII art generation", "category": "creative"},
        ]
        with patch(
            "tools.skills_tool._find_all_skills",
            return_value=list(fake_skills),
        ):
            app = _create_app(adapter)
            async with TestClient(TestServer(app)) as cli:
                resp = await cli.get("/v1/skills")
                assert resp.status == 200
                data = await resp.json()
                assert data["object"] == "list"
                names = sorted(s["name"] for s in data["data"])
                assert names == ["ascii-art", "github"]
                for entry in data["data"]:
                    assert set(entry.keys()) >= {"name", "description", "category"}


class TestToolsetsEndpoint:
    @pytest.mark.asyncio
    async def test_toolsets_returns_resolved_tools(self, adapter):
        fake_toolsets = [
            ("default", "Default Tools", "Core tools"),
            ("web", "Web Tools", "Search and extract"),
        ]
        feature_snapshot = object()
        with patch(
            "hermes_cli.tools_config._get_effective_configurable_toolsets",
            return_value=fake_toolsets,
        ), patch(
            "hermes_cli.tools_config._get_platform_tools",
            return_value={"default"},
        ), patch(
            "hermes_cli.tools_config.get_nous_subscription_features",
            return_value=feature_snapshot,
        ) as resolve_features, patch(
            "hermes_cli.tools_config._toolset_has_keys",
            return_value=True,
        ) as has_keys, patch(
            "toolsets.resolve_toolset",
            side_effect=lambda name: {
                "default": ["terminal", "read_file"],
                "web": ["web_search"],
            }[name],
        ):
            app = _create_app(adapter)
            async with TestClient(TestServer(app)) as cli:
                resp = await cli.get("/v1/toolsets")
                assert resp.status == 200
                data = await resp.json()
                assert data["object"] == "list"
                assert data["platform"] == "api_server"
                by_name = {ts["name"]: ts for ts in data["data"]}
                assert by_name["default"]["enabled"] is True
                assert by_name["default"]["tools"] == ["read_file", "terminal"]
                assert by_name["web"]["enabled"] is False
                assert by_name["web"]["tools"] == ["web_search"]
                assert by_name["default"]["configured"] is True

        resolve_features.assert_called_once()
        assert has_keys.call_count == len(fake_toolsets)
        assert all(
            call.kwargs["features"] is feature_snapshot
            for call in has_keys.call_args_list
        )


# ---------------------------------------------------------------------------
# /v1/chat/completions endpoint
# ---------------------------------------------------------------------------


class TestChatCompletionsEndpoint:
    @pytest.mark.asyncio
    async def test_invalid_json_returns_400(self, adapter):
        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.post(
                "/v1/chat/completions",
                data="not json",
                headers={"Content-Type": "application/json"},
            )
            assert resp.status == 400
            data = await resp.json()
            assert "Invalid JSON" in data["error"]["message"]

    @pytest.mark.asyncio
    async def test_missing_messages_returns_400(self, adapter):
        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.post("/v1/chat/completions", json={"model": "test"})
            assert resp.status == 400
            data = await resp.json()
            assert "messages" in data["error"]["message"]

    @pytest.mark.asyncio
    async def test_session_stream_disconnect_uses_hard_interrupt(self, adapter):
        """Compression state cannot mask an abandoned session stream stop."""

        class _Agent:
            def __init__(self):
                self.hard_interrupt = MagicMock()
                self.interrupt = MagicMock()

        agent = _Agent()
        adapter._active_run_agents["run-disconnected"] = agent
        task = asyncio.create_task(asyncio.sleep(0))
        await task

        await adapter._drain_session_stream_task_on_disconnect(
            "run-disconnected",
            task,
            interrupt_message="SSE client disconnected",
            shield_wait=False,
        )

        agent.hard_interrupt.assert_called_once_with("SSE client disconnected")
        agent.interrupt.assert_not_called()


    @pytest.mark.asyncio
    async def test_non_streaming_shuts_down_memory_provider(self, adapter):
        mock_agent = _make_api_agent("Hello!")
        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_create_agent", return_value=mock_agent):
                resp = await cli.post(
                    "/v1/chat/completions",
                    json={
                        "model": "test",
                        "messages": [{"role": "user", "content": "hi"}],
                    },
                )
                assert resp.status == 200
                data = await resp.json()

        assert data["choices"][0]["message"]["content"] == "Hello!"
        mock_agent.shutdown_memory_provider.assert_called_once_with(mock_agent._session_messages)
        mock_agent.close.assert_not_called()

    @pytest.mark.asyncio
    async def test_non_streaming_preserves_terminal_plugin_metadata(self, adapter):
        mock_result = {
            "final_response": "OK",
            "messages": [],
            "api_calls": 1,
            "response_metadata": {
                "dorvis_trace_manifest": {"trace_id": "1" * 32}
            },
        }
        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_run_agent", new_callable=AsyncMock) as mock_run:
                mock_run.return_value = (
                    mock_result,
                    {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
                )
                resp = await cli.post(
                    "/v1/responses",
                    json={"model": "hermes-agent", "input": "Hello"},
                )
            data = await resp.json()
            assert data["metadata"] == mock_result["response_metadata"]
            stored = await cli.get(f"/v1/responses/{data['id']}")
            assert (await stored.json())["metadata"] == mock_result["response_metadata"]

    @pytest.mark.asyncio
    async def test_stream_true_returns_sse(self, adapter):
        """stream=true returns SSE format with the full response."""
        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            async def _mock_run_agent(**kwargs):
                # Simulate streaming: invoke stream_delta_callback with tokens.
                cb = kwargs.get("stream_delta_callback")
                if cb:
                    cb("Hello!")
                    cb(None)
                return (
                    {"final_response": "Hello!", "messages": [], "api_calls": 1},
                    {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
                )

            with patch.object(adapter, "_run_agent", side_effect=_mock_run_agent):
                resp = await cli.post(
                    "/v1/chat/completions",
                    json={
                        "model": "test",
                        "messages": [{"role": "user", "content": "hi"}],
                        "stream": True,
                    },
                )
                assert resp.status == 200
                assert "text/event-stream" in resp.headers.get("Content-Type", "")
                assert resp.headers.get("X-Accel-Buffering") == "no"
                body = await resp.text()
                assert "data: " in body
                assert "[DONE]" in body
                assert "Hello!" in body

    @pytest.mark.asyncio
    async def test_chat_completions_stream_passes_request_model_provider_options(self, adapter):
        app = _create_app(adapter)
        model_options = {"reasoning": {"enabled": False}, "reasoning_effort": "none", "fast": False}

        async def _mock_run_agent(**kwargs):
            cb = kwargs.get("stream_delta_callback")
            if cb:
                cb("ok")
            return (
                {"final_response": "ok", "messages": [], "api_calls": 1},
                {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
            )

        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_run_agent", side_effect=_mock_run_agent) as mock_run:
                resp = await cli.post(
                    "/v1/chat/completions",
                    json={
                        "model": "MiniMax-M3",
                        "provider": "minimax",
                        "model_options": model_options,
                        "messages": [{"role": "user", "content": "hi"}],
                        "stream": True,
                    },
                )
                assert resp.status == 200
                body = await resp.text()

        assert "data: " in body
        kwargs = mock_run.call_args.kwargs
        assert kwargs["requested_model"] == "MiniMax-M3"
        assert kwargs["requested_provider"] == "minimax"
        assert kwargs["model_options"] == model_options


    @pytest.mark.asyncio
    async def test_session_chat_stream_passes_request_model_provider_options(self, adapter):
        app = _create_app(adapter)
        model_options = {"reasoning_effort": "medium", "service_tier": "priority"}
        async with TestClient(TestServer(app)) as cli:
            with (
                patch.object(adapter, "_get_existing_session_or_404", return_value=({"id": "s1"}, None)),
                patch.object(adapter, "_conversation_history_for_session", return_value=[]),
                patch.object(adapter, "_run_agent", new_callable=AsyncMock) as mock_run,
            ):
                mock_run.return_value = (
                    {"final_response": "ok", "messages": [], "api_calls": 1},
                    {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
                )
                resp = await cli.post(
                    "/api/sessions/s1/chat/stream",
                    json={
                        "message": "hi",
                        "model": "MiniMax-M3",
                        "provider": "minimax",
                        "model_options": model_options,
                    },
                )
                assert resp.status == 200
                body = await resp.text()

        assert "event: run.completed" in body
        kwargs = mock_run.call_args.kwargs
        assert kwargs["requested_model"] == "MiniMax-M3"
        assert kwargs["requested_provider"] == "minimax"
        assert kwargs["model_options"] == model_options


    @pytest.mark.asyncio
    async def test_stream_task_done_callback_enqueues_eos_for_chat_completions(self, adapter):
        """Regression guard for #24451: completion callback must signal SSE EOS."""
        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            class _FakeTask:
                def __init__(self):
                    self.callbacks = []

                def add_done_callback(self, cb):
                    self.callbacks.append(cb)

            fake_task = _FakeTask()

            def _fake_ensure_future(coro):
                # We short-circuit task scheduling in this unit test.
                coro.close()
                return fake_task

            with (
                patch.object(
                    adapter,
                    "_run_agent",
                    new=AsyncMock(
                        return_value=(
                            {"final_response": "ok", "messages": [], "api_calls": 1},
                            {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
                        )
                    ),
                ),
                patch("gateway.platforms.api_server.asyncio.ensure_future", side_effect=_fake_ensure_future),
                patch.object(adapter, "_write_sse_chat_completion", new_callable=AsyncMock) as mock_write_sse,
            ):
                mock_write_sse.return_value = web.Response(status=200, text="ok")
                resp = await cli.post(
                    "/v1/chat/completions",
                    json={
                        "model": "test",
                        "messages": [{"role": "user", "content": "hi"}],
                        "stream": True,
                    },
                )
                assert resp.status == 200

            assert len(fake_task.callbacks) == 1
            stream_q = mock_write_sse.call_args.args[4]
            assert stream_q.empty()
            fake_task.callbacks[0](fake_task)
            assert stream_q.get_nowait() is None


    @pytest.mark.asyncio
    async def test_stream_includes_tool_progress(self, adapter):
        """tool_start_callback fires → progress appears as custom SSE event, not in delta.content."""
        import asyncio

        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            async def _mock_run_agent(**kwargs):
                cb = kwargs.get("stream_delta_callback")
                ts_cb = kwargs.get("tool_start_callback")
                # Simulate the structured tool start the gateway now consumes.
                if ts_cb:
                    ts_cb("call_terminal_1", "terminal", {"command": "ls -la"})
                if cb:
                    await asyncio.sleep(0.05)
                    cb("Here are the files.")
                return (
                    {"final_response": "Here are the files.", "messages": [], "api_calls": 1},
                    {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
                )

            with patch.object(adapter, "_run_agent", side_effect=_mock_run_agent):
                resp = await cli.post(
                    "/v1/chat/completions",
                    json={
                        "model": "test",
                        "messages": [{"role": "user", "content": "list files"}],
                        "stream": True,
                    },
                )
                assert resp.status == 200
                body = await resp.text()
                assert "[DONE]" in body
                # Tool progress must appear as a custom SSE event, not in
                # delta.content — prevents model from learning to imitate
                # markers instead of calling tools (#6972).
                assert "event: hermes.tool.progress" in body
                assert '"tool": "terminal"' in body
                # ``label`` is now derived by ``build_tool_preview`` from the
                # tool args rather than passed by the caller, so we assert
                # only that *some* label exists rather than a literal value.
                assert '"label":' in body
                # The progress marker must NOT appear inside any
                # chat.completion.chunk delta.content field.
                import json as _json
                for line in body.splitlines():
                    if line.startswith("data: ") and line.strip() != "data: [DONE]":
                        try:
                            chunk = _json.loads(line[len("data: "):])
                        except _json.JSONDecodeError:
                            continue
                        if chunk.get("object") == "chat.completion.chunk":
                            for choice in chunk.get("choices", []):
                                content = choice.get("delta", {}).get("content", "")
                                # Tool emoji markers must never leak into content
                                assert "ls -la" not in content or content == "Here are the files."
                # Final content must also be present
                assert "Here are the files." in body


    @pytest.mark.asyncio
    async def test_stream_emits_tool_lifecycle_with_call_id(self, adapter):
        """Regression for #16588.

        ``/v1/chat/completions`` streaming previously emitted only a
        ``tool.started``-style ``hermes.tool.progress`` event; clients
        rendering tool lifecycle UI had no way to mark a tool as finished
        because no matching ``status: completed`` event was emitted, and
        no ``toolCallId`` was carried for correlation.

        The fix adds ``tool_start_callback`` / ``tool_complete_callback``
        to the chat completions agent invocation and writes both halves
        of the lifecycle pair on the same ``event: hermes.tool.progress``
        SSE line, with stable ``toolCallId`` and ``status``.
        """
        import asyncio
        import json as _json

        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            async def _mock_run_agent(**kwargs):
                cb = kwargs.get("stream_delta_callback")
                ts_cb = kwargs.get("tool_start_callback")
                tc_cb = kwargs.get("tool_complete_callback")
                # The structured callbacks own the chat-completions SSE
                # channel now; ``tool_progress_callback`` is intentionally
                # not wired so each tool start emits exactly one event.
                if ts_cb:
                    ts_cb("call_terminal_1", "terminal", {"command": "ls -la"})
                if tc_cb:
                    tc_cb("call_terminal_1", "terminal", {"command": "ls -la"}, "ok")
                if cb:
                    await asyncio.sleep(0.05)
                    cb("done.")
                return (
                    {"final_response": "done.", "messages": [], "api_calls": 1},
                    {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
                )

            with patch.object(adapter, "_run_agent", side_effect=_mock_run_agent):
                resp = await cli.post(
                    "/v1/chat/completions",
                    json={
                        "model": "test",
                        "messages": [{"role": "user", "content": "list"}],
                        "stream": True,
                    },
                )
                assert resp.status == 200
                body = await resp.text()

            # Walk the SSE body and collect *(status, toolCallId)* pairs
            # per event so the assertions verify per-event correlation —
            # an event missing ``toolCallId`` would not pass even if a
            # different event happens to carry the right id.
            pairs: list[tuple[str | None, str | None]] = []
            lines = body.splitlines()
            for i, line in enumerate(lines):
                if line.strip() != "event: hermes.tool.progress":
                    continue
                for follow in lines[i + 1: i + 4]:
                    if follow.startswith("data: "):
                        try:
                            payload = _json.loads(follow[len("data: "):])
                        except _json.JSONDecodeError:
                            break
                        pairs.append((payload.get("status"), payload.get("toolCallId")))
                        break

            # Each tool start must emit exactly one event (no duplicate
            # legacy + new emit), and each lifecycle pair must carry the
            # same toolCallId on every event — not just somewhere in the
            # aggregate.
            assert len(pairs) == 2, f"expected 2 events (running+completed), got {pairs}"
            assert pairs[0] == ("running", "call_terminal_1"), pairs
            assert pairs[1] == ("completed", "call_terminal_1"), pairs

    @pytest.mark.asyncio
    async def test_stream_tool_lifecycle_skips_internal_and_orphan_completes(self, adapter):
        """Internal tools (``_thinking``-style) and ``completed`` events
        without a prior matching ``running`` must produce no lifecycle
        events on the wire — otherwise clients would see orphaned
        ``status: completed`` updates they cannot correlate."""
        import asyncio

        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            async def _mock_run_agent(**kwargs):
                cb = kwargs.get("stream_delta_callback")
                ts_cb = kwargs.get("tool_start_callback")
                tc_cb = kwargs.get("tool_complete_callback")
                # Internal tool — must be filtered.
                if ts_cb:
                    ts_cb("call_internal_1", "_thinking", {})
                if tc_cb:
                    tc_cb("call_internal_1", "_thinking", {}, "")
                # Completion without start — orphan, must be dropped.
                if tc_cb:
                    tc_cb("call_orphan_1", "web_search", {}, "ok")
                if cb:
                    await asyncio.sleep(0.05)
                    cb("ok.")
                return (
                    {"final_response": "ok.", "messages": [], "api_calls": 1},
                    {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
                )

            with patch.object(adapter, "_run_agent", side_effect=_mock_run_agent):
                resp = await cli.post(
                    "/v1/chat/completions",
                    json={
                        "model": "test",
                        "messages": [{"role": "user", "content": "ok"}],
                        "stream": True,
                    },
                )
                assert resp.status == 200
                body = await resp.text()

            # Neither the internal call_id nor the orphan call_id should
            # surface as a lifecycle payload on the wire.
            assert "call_internal_1" not in body
            assert "call_orphan_1" not in body
            assert '"status": "running"' not in body
            assert '"status": "completed"' not in body


# ---------------------------------------------------------------------------
# _derive_chat_session_id unit tests
# ---------------------------------------------------------------------------


class TestDeriveChatSessionId:
    def test_deterministic(self):
        """Same inputs always produce the same session ID."""
        a = _derive_chat_session_id("sys", "hello")
        b = _derive_chat_session_id("sys", "hello")
        assert a == b


    def test_different_system_prompt(self):
        a = _derive_chat_session_id("You are a pirate.", "Hello")
        b = _derive_chat_session_id("You are a robot.", "Hello")
        assert a != b


# ---------------------------------------------------------------------------
# /v1/responses endpoint
# ---------------------------------------------------------------------------


class TestResponsesEndpoint:
    @pytest.mark.parametrize(
        "content",
        [
            "[CONTEXT COMPACTION — REFERENCE ONLY]\n## State Ledger",
            (
                "[PRIOR CONTEXT — for reference only; not a new message]\n"
                "completed handoff\n"
                "[END OF PRIOR CONTEXT — COMPACTION SUMMARY BELOW]\n"
                "## State Ledger"
            ),
        ],
    )
    def test_compaction_summary_markers_are_authoritative(self, content):
        assert APIServerAdapter._messages_include_compaction_summary(
            [{"role": "assistant", "content": content}]
        )

    @pytest.mark.asyncio
    async def test_missing_input_returns_400(self, adapter):
        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.post("/v1/responses", json={"model": "test"})
            assert resp.status == 400
            data = await resp.json()
            assert "input" in data["error"]["message"]


    @pytest.mark.asyncio
    async def test_successful_response_with_string_input(self, adapter):
        """String input is wrapped in a user message."""
        mock_result = {
            "final_response": "Paris is the capital of France.",
            "messages": [],
            "api_calls": 1,
        }

        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_run_agent", new_callable=AsyncMock) as mock_run:
                mock_run.return_value = (mock_result, {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0})
                resp = await cli.post(
                    "/v1/responses",
                    json={
                        "model": "hermes-agent",
                        "input": "What is the capital of France?",
                    },
                )

            assert resp.status == 200
            data = await resp.json()
            assert data["object"] == "response"
            assert data["id"].startswith("resp_")
            assert data["status"] == "completed"
            assert len(data["output"]) == 1
            assert data["output"][0]["type"] == "message"
            assert data["output"][0]["content"][0]["type"] == "output_text"
            assert data["output"][0]["content"][0]["text"] == "Paris is the capital of France."

    @pytest.mark.asyncio
    async def test_response_body_metadata_reaches_agent_without_mutating_input(self, adapter):
        """Responses metadata is first-class request state, not prompt text."""
        mock_result = {"final_response": "ok", "messages": [], "api_calls": 1}
        metadata = {
            "source": "dorvis-web",
            "environment": "staging",
            "caller": {
                "email": "user@example.com",
                "name": "User Name",
                "uid": "user-123",
            },
            "chat": {"id": "chat-456", "type": "web"},
        }

        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_run_agent", new_callable=AsyncMock) as mock_run:
                mock_run.return_value = (mock_result, {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0})
                resp = await cli.post(
                    "/v1/responses",
                    json={"model": "hermes-agent", "input": "hello cleanly", "metadata": metadata},
                )

            assert resp.status == 200
            call_kwargs = mock_run.call_args.kwargs
            assert call_kwargs["user_message"] == "hello cleanly"
            assert "[caller:" not in call_kwargs["user_message"]
            assert call_kwargs["request_metadata"]["caller"]["email"] == "user@example.com"
            assert call_kwargs["request_metadata"]["chat"]["id"] == "chat-456"

    @pytest.mark.asyncio
    async def test_response_metadata_adds_identity_to_ephemeral_prompt(self, adapter):
        """Identity metadata becomes model-visible context without touching user text."""
        metadata = {
            "source": "dorvis-web",
            "environment": "staging",
            "caller": {
                "email": "dalton@example.com",
                "name": "Dalton Orvis",
                "uid": "user-123",
            },
            "chat": {"id": "chat-456", "type": "web"},
        }

        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_run_agent", new_callable=AsyncMock) as mock_run:
                mock_run.return_value = (
                    {"final_response": "ok", "messages": [], "api_calls": 1},
                    {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
                )
                resp = await cli.post(
                    "/v1/responses",
                    json={
                        "model": "hermes-agent",
                        "input": "hello cleanly",
                        "instructions": "Talk like a pirate.",
                        "metadata": metadata,
                    },
                )

            assert resp.status == 200
            call_kwargs = mock_run.call_args.kwargs
            assert call_kwargs["user_message"] == "hello cleanly"
            assert "[caller:" not in call_kwargs["user_message"]
            assert call_kwargs["ephemeral_system_prompt"] == (
                "Talk like a pirate.\n\n"
                "Current signed-in web user: Dalton Orvis <dalton@example.com>."
            )

    @pytest.mark.asyncio
    async def test_response_header_metadata_used_when_body_metadata_absent(self, adapter):
        metadata = {"caller": {"email": "header@example.com", "uid": "u1"}, "chat": {"id": "c1"}}
        encoded = base64.urlsafe_b64encode(json.dumps(metadata).encode()).decode().rstrip("=")

        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_run_agent", new_callable=AsyncMock) as mock_run:
                mock_run.return_value = (
                    {"final_response": "ok", "messages": [], "api_calls": 1},
                    {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
                )
                resp = await cli.post(
                    "/v1/responses",
                    json={"input": "hello"},
                    headers={"X-Hermes-Metadata": encoded},
                )

            assert resp.status == 200
            assert mock_run.call_args.kwargs["request_metadata"]["caller"]["email"] == "header@example.com"

    @pytest.mark.asyncio
    async def test_response_body_metadata_wins_over_header_metadata(self, adapter):
        header_metadata = {"caller": {"email": "header@example.com", "uid": "u1"}}
        encoded = base64.urlsafe_b64encode(json.dumps(header_metadata).encode()).decode().rstrip("=")

        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_run_agent", new_callable=AsyncMock) as mock_run:
                mock_run.return_value = (
                    {"final_response": "ok", "messages": [], "api_calls": 1},
                    {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
                )
                resp = await cli.post(
                    "/v1/responses",
                    json={"input": "hello", "metadata": {"caller": {"email": "body@example.com", "uid": "u2"}}},
                    headers={"X-Hermes-Metadata": encoded},
                )

            assert resp.status == 200
            assert mock_run.call_args.kwargs["request_metadata"]["caller"]["email"] == "body@example.com"

    @pytest.mark.asyncio
    async def test_response_metadata_sanitizer_drops_deeply_nested_values(self, adapter):
        deeply_nested = "leaf"
        for _ in range(32):
            deeply_nested = {"next": deeply_nested}

        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_run_agent", new_callable=AsyncMock) as mock_run:
                mock_run.return_value = (
                    {"final_response": "ok", "messages": [], "api_calls": 1},
                    {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
                )
                resp = await cli.post(
                    "/v1/responses",
                    json={
                        "input": "hello",
                        "metadata": {
                            "caller": {"email": "depth@example.com", "uid": "u1"},
                            "deep": deeply_nested,
                        },
                    },
                )

            assert resp.status == 200
            request_metadata = mock_run.call_args.kwargs["request_metadata"]
            assert request_metadata["caller"]["email"] == "depth@example.com"
            assert request_metadata["deep"]["next"]["next"]["next"]
            assert "leaf" not in json.dumps(request_metadata)

    @pytest.mark.asyncio
    async def test_response_idempotency_fingerprint_includes_metadata(self, adapter):
        app = _create_app(adapter)
        key = f"metadata-idem-{uuid.uuid4()}"

        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_run_agent", new_callable=AsyncMock) as mock_run:
                mock_run.return_value = (
                    {"final_response": "ok", "messages": [], "api_calls": 1},
                    {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
                )
                for email in ("one@example.com", "two@example.com"):
                    resp = await cli.post(
                        "/v1/responses",
                        json={
                            "input": "same prompt",
                            "metadata": {"caller": {"email": email, "uid": email}},
                        },
                        headers={"Idempotency-Key": key},
                    )
                    assert resp.status == 200

            assert mock_run.await_count == 2
            assert [
                call.kwargs["request_metadata"]["caller"]["email"]
                for call in mock_run.await_args_list
            ] == ["one@example.com", "two@example.com"]

    @pytest.mark.asyncio
    async def test_successful_response_with_array_input(self, adapter):
        """Array input with role/content objects."""
        mock_result = {"final_response": "Done", "messages": [], "api_calls": 1}

        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_run_agent", new_callable=AsyncMock) as mock_run:
                mock_run.return_value = (mock_result, {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0})
                resp = await cli.post(
                    "/v1/responses",
                    json={
                        "model": "hermes-agent",
                        "input": [
                            {"role": "user", "content": "Hello"},
                            {"role": "user", "content": "What is 2+2?"},
                        ],
                    },
                )

            assert resp.status == 200
            call_kwargs = mock_run.call_args.kwargs
            # Last message is user_message, rest are history
            assert call_kwargs["user_message"] == "What is 2+2?"
            assert len(call_kwargs["conversation_history"]) == 1

    @pytest.mark.asyncio
    async def test_instructions_as_ephemeral_prompt(self, adapter):
        """The instructions field maps to ephemeral_system_prompt."""
        mock_result = {"final_response": "Ahoy!", "messages": [], "api_calls": 1}

        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_run_agent", new_callable=AsyncMock) as mock_run:
                mock_run.return_value = (mock_result, {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0})
                resp = await cli.post(
                    "/v1/responses",
                    json={
                        "model": "hermes-agent",
                        "input": "Hello",
                        "instructions": "Talk like a pirate.",
                    },
                )

            assert resp.status == 200
            call_kwargs = mock_run.call_args.kwargs
            assert call_kwargs["ephemeral_system_prompt"] == "Talk like a pirate."

    def test_request_identity_prompt_appends_to_instructions(self):
        prompt = _append_request_identity_prompt(
            "Talk like a pirate.",
            {
                "caller": {
                    "email": "user@example.com",
                    "name": "User Name",
                    "uid": "user-123",
                },
                "chat": {"id": "chat-456", "type": "web"},
            },
        )

        assert prompt == "Talk like a pirate.\n\nCurrent signed-in web user: User Name <user@example.com>."

    def test_request_identity_prompt_omitted_without_identity(self):
        assert _append_request_identity_prompt("Talk like a pirate.", {}) == "Talk like a pirate."

    @pytest.mark.asyncio
    async def test_previous_response_id_chaining(self, adapter):
        """Test that responses can be chained via previous_response_id."""
        mock_result_1 = {
            "final_response": "2",
            "messages": [{"role": "assistant", "content": "2"}],
            "api_calls": 1,
        }

        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            # First request
            with patch.object(adapter, "_run_agent", new_callable=AsyncMock) as mock_run:
                mock_run.return_value = (mock_result_1, {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0})
                resp1 = await cli.post(
                    "/v1/responses",
                    json={"model": "hermes-agent", "input": "What is 1+1?"},
                )

            assert resp1.status == 200
            data1 = await resp1.json()
            response_id = data1["id"]

            # Second request chaining from the first
            mock_result_2 = {
                "final_response": "3",
                "messages": [{"role": "assistant", "content": "3"}],
                "api_calls": 1,
            }

            with patch.object(adapter, "_run_agent", new_callable=AsyncMock) as mock_run:
                mock_run.return_value = (mock_result_2, {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0})
                resp2 = await cli.post(
                    "/v1/responses",
                    json={
                        "model": "hermes-agent",
                        "input": "Now add 1 more",
                        "previous_response_id": response_id,
                    },
                )

            assert resp2.status == 200
            # The conversation_history should contain the full history from the first response
            call_kwargs = mock_run.call_args.kwargs
            assert len(call_kwargs["conversation_history"]) > 0
            assert call_kwargs["user_message"] == "Now add 1 more"

    @pytest.mark.asyncio
    async def test_previous_response_id_stores_full_agent_transcript_once(self, adapter):
        """Chained Responses storage must not append result["messages"] twice."""
        first_history = [
            {"role": "user", "content": "What is 1+1?"},
            {"role": "assistant", "content": "2"},
        ]

        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_run_agent", new_callable=AsyncMock) as mock_run:
                mock_run.return_value = (
                    {
                        "final_response": "2",
                        "messages": list(first_history),
                        "api_calls": 1,
                    },
                    {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
                )
                resp1 = await cli.post(
                    "/v1/responses",
                    json={"model": "hermes-agent", "input": "What is 1+1?"},
                )

            assert resp1.status == 200
            resp1_data = await resp1.json()
            stored_first = adapter._response_store.get(resp1_data["id"])
            assert stored_first["conversation_history"] == first_history

            second_history = first_history + [
                {"role": "user", "content": "Now add 1 more"},
                {"role": "assistant", "content": "3"},
            ]
            with patch.object(adapter, "_run_agent", new_callable=AsyncMock) as mock_run:
                mock_run.return_value = (
                    {
                        "final_response": "3",
                        "messages": list(second_history),
                        "api_calls": 1,
                    },
                    {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
                )
                resp2 = await cli.post(
                    "/v1/responses",
                    json={
                        "model": "hermes-agent",
                        "input": "Now add 1 more",
                        "previous_response_id": resp1_data["id"],
                    },
                )

            assert resp2.status == 200
            resp2_data = await resp2.json()
            stored_second = adapter._response_store.get(resp2_data["id"])
            stored_history = stored_second["conversation_history"]
            assert stored_history == second_history
            assert stored_history.count(first_history[0]) == 1
            assert stored_history.count(
                {"role": "user", "content": "Now add 1 more"}
            ) == 1

    @pytest.mark.asyncio
    async def test_previous_response_id_stores_compressed_transcript_directly(self, adapter):
        """After compression, stored history is the compressed transcript, not prior + compressed."""
        prior_history = [
            {"role": "user", "content": "What is 1+1?"},
            {"role": "assistant", "content": "2"},
        ] * 10  # 20 messages — enough to simulate a long conversation
        adapter._response_store.put(
            "resp_prev",
            {
                "response": {"id": "resp_prev", "status": "completed"},
                "conversation_history": list(prior_history),
                "session_id": "api-test-session",
            },
        )

        compressed_history = [
            # Compressed transcript starts with summary, NOT with prior[0]
            {"role": "user", "content": "[Compressed summary of earlier conversation]"},
            {"role": "user", "content": "Now add 1 more"},
            {"role": "assistant", "content": "3"},
        ]

        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_run_agent", new_callable=AsyncMock) as mock_run:
                mock_run.return_value = (
                    {
                        "final_response": "3",
                        "messages": list(compressed_history),
                        "_compressed": True,
                        "api_calls": 1,
                    },
                    {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
                )
                resp = await cli.post(
                    "/v1/responses",
                    json={
                        "model": "hermes-agent",
                        "input": "Now add 1 more",
                        "previous_response_id": "resp_prev",
                    },
                )
                assert resp.status == 200
                data = await resp.json()

        stored = adapter._response_store.get(data["id"])
        stored_history = stored["conversation_history"]
        # Must NOT contain the original prior_history messages
        for msg in prior_history:
            assert msg not in stored_history, (
                f"Prior history message leaked into stored compressed transcript: {msg}"
            )
        # Must contain the compressed transcript
        assert stored_history == compressed_history


    @pytest.mark.asyncio
    async def test_previous_response_id_ignores_private_persistence_metadata(self, adapter):
        """Runtime-only message markers must not defeat transcript prefix detection."""
        user = {"role": "user", "content": "What is 1+1?"}
        runtime_user = {**user, "_db_persisted": True}
        runtime_assistant = {
            "role": "assistant",
            "content": "2",
            "_db_persisted": True,
        }

        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_run_agent", new_callable=AsyncMock) as mock_run:
                mock_run.return_value = (
                    {
                        "final_response": "2",
                        "messages": [runtime_user, runtime_assistant],
                        "api_calls": 1,
                    },
                    {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
                )
                resp = await cli.post(
                    "/v1/responses",
                    json={"model": "hermes-agent", "input": user["content"]},
                )

            assert resp.status == 200
            response = await resp.json()
            stored = adapter._response_store.get(response["id"])
            assert stored["conversation_history"] == [
                runtime_user,
                runtime_assistant,
            ]

            second_user = {
                "role": "user",
                "content": "Now add 1 more",
                "_db_persisted": True,
            }
            second_assistant = {
                "role": "assistant",
                "content": "3",
                "_db_persisted": True,
            }
            with patch.object(adapter, "_run_agent", new_callable=AsyncMock) as mock_run:
                mock_run.return_value = (
                    {
                        "final_response": "3",
                        "messages": [
                            runtime_user,
                            runtime_assistant,
                            second_user,
                            second_assistant,
                        ],
                        "api_calls": 1,
                    },
                    {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
                )
                chained = await cli.post(
                    "/v1/responses",
                    json={
                        "model": "hermes-agent",
                        "input": second_user["content"],
                        "previous_response_id": response["id"],
                    },
                )

            assert chained.status == 200
            chained_response = await chained.json()
            chained_history = adapter._response_store.get(
                chained_response["id"]
            )["conversation_history"]
            assert [
                message["content"]
                for message in chained_history
                if message["role"] == "user"
            ] == [user["content"], second_user["content"]]

    # ------------------------------------------------------------------
    # Staging history doubling (AE-204 family / staging 2026-08-11)
    #
    # On staging, Hindsight prefetch stamps an ``api_content`` sidecar on
    # the current turn's user row (agent/turn_context.py). The sidecar is
    # not underscore-prefixed, so ``_response_prefix_projection`` kept it
    # and the expected-prefix match failed on the very first turn — the
    # fallthrough then stored ``prior + user + result["messages"]``, i.e.
    # the current user row twice, adjacent. Every later turn,
    # ``repair_message_sequence`` merged those adjacent user rows in the
    # agent's transcript, so BOTH prefix checks failed and the fallthrough
    # re-embedded the whole prior history: h(n+1) = 2*h(n) + 2.
    # ------------------------------------------------------------------

    @staticmethod
    def _simulate_staging_agent_turn(conversation_history, user_message, reply):
        """Mimic run_conversation's transcript shape on staging.

        Two real perturbations are reproduced:
        - ``repair_message_sequence`` pass 2 (agent/agent_runtime_helpers.py)
          merges consecutive user rows with a blank-line separator and drops
          their stale ``api_content`` sidecar.
        - Hindsight memory prefetch stamps the ``api_content`` sidecar and
          the ``_db_persisted`` marker on this turn's user row
          (agent/turn_context.py).
        """
        messages = []
        for msg in conversation_history:
            if (
                messages
                and isinstance(msg, dict)
                and msg.get("role") == "user"
                and isinstance(messages[-1], dict)
                and messages[-1].get("role") == "user"
                and isinstance(messages[-1].get("content"), str)
                and isinstance(msg.get("content"), str)
            ):
                merged = dict(messages[-1])
                merged["content"] = merged["content"] + "\n\n" + msg["content"]
                merged.pop("api_content", None)
                messages[-1] = merged
                continue
            messages.append(msg)
        messages.append({
            "role": "user",
            "content": user_message,
            "api_content": (
                "<recalled_memories>staging recall block</recalled_memories>"
                "\n\n" + str(user_message)
            ),
            "_db_persisted": True,
        })
        messages.append({
            "role": "assistant",
            "content": reply,
            "_db_persisted": True,
        })
        return messages

    @pytest.mark.asyncio
    async def test_memory_sidecar_chained_turns_store_each_message_once(self, adapter):
        """Reproduces the staging 2n+2 doubling across chained turns.

        Before the fix, stored history lengths grew 3 -> 8 -> 18 (each turn
        re-embedding the whole prior history); the staging Langfuse evidence
        continued 18 -> 35 -> 67. After the fix each real message is stored
        exactly once per turn: 2 -> 4 -> 6.
        """
        turns = [
            ("staging turn one", "answer one"),
            ("staging turn two", "answer two"),
            ("staging turn three", "answer three"),
        ]

        def _make_mock(reply):
            async def _mock_run_agent(**kwargs):
                return (
                    {
                        "final_response": reply,
                        "messages": self._simulate_staging_agent_turn(
                            kwargs["conversation_history"],
                            kwargs["user_message"],
                            reply,
                        ),
                        "api_calls": 1,
                    },
                    {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
                )
            return _mock_run_agent

        app = _create_app(adapter)
        prev_id = None
        stored_histories = []
        async with TestClient(TestServer(app)) as cli:
            for user_text, reply in turns:
                payload = {"model": "hermes-agent", "input": user_text}
                if prev_id:
                    payload["previous_response_id"] = prev_id
                with patch.object(
                    adapter, "_run_agent", side_effect=_make_mock(reply)
                ):
                    resp = await cli.post("/v1/responses", json=payload)
                assert resp.status == 200
                data = await resp.json()
                prev_id = data["id"]
                stored_histories.append(
                    adapter._response_store.get(prev_id)["conversation_history"]
                )

        for turn_number, history in enumerate(stored_histories, start=1):
            for user_text, reply in turns[:turn_number]:
                user_rows = [
                    m for m in history
                    if m.get("role") == "user" and user_text in str(m.get("content"))
                ]
                assert len(user_rows) == 1, (
                    f"after turn {turn_number}, user text {user_text!r} is stored "
                    f"{len(user_rows)} times (history len {len(history)})"
                )
                assistant_rows = [
                    m for m in history
                    if m.get("role") == "assistant" and reply in str(m.get("content"))
                ]
                assert len(assistant_rows) == 1, (
                    f"after turn {turn_number}, reply {reply!r} is stored "
                    f"{len(assistant_rows)} times"
                )
        assert [len(h) for h in stored_histories] == [2, 4, 6]

    @pytest.mark.asyncio
    async def test_memory_sidecar_on_first_turn_stores_user_once(self, adapter):
        """The api_content sidecar must not defeat first-turn prefix detection."""
        user_text = "hello with recall"
        agent_messages = [
            {
                "role": "user",
                "content": user_text,
                "api_content": "<recalled_memories>x</recalled_memories>\n\n" + user_text,
                "_db_persisted": True,
            },
            {"role": "assistant", "content": "hi", "_db_persisted": True},
        ]

        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_run_agent", new_callable=AsyncMock) as mock_run:
                mock_run.return_value = (
                    {
                        "final_response": "hi",
                        "messages": list(agent_messages),
                        "api_calls": 1,
                    },
                    {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
                )
                resp = await cli.post(
                    "/v1/responses",
                    json={"model": "hermes-agent", "input": user_text},
                )

            assert resp.status == 200
            data = await resp.json()
            stored = adapter._response_store.get(data["id"])["conversation_history"]
            assert stored == agent_messages
            assert [m["role"] for m in stored] == ["user", "assistant"]

    @pytest.mark.asyncio
    async def test_perturbed_prior_history_never_reembeds_prior(self, adapter):
        """Repair-merged prior rows must not trigger prior re-embedding.

        When alternation repair rewrote the prior rows inside the agent's
        transcript (so neither prefix check can match), the transcript is
        still authoritative — the stored history must be adopted from it,
        never concatenated after the prior history again.
        """
        prior_history = [
            {"role": "user", "content": "first question"},
            {"role": "user", "content": "second question"},
            {"role": "assistant", "content": "combined answer"},
        ]
        adapter._response_store.put(
            "resp_prev",
            {
                "response": {"id": "resp_prev", "status": "completed"},
                "conversation_history": list(prior_history),
                "session_id": "api-test-session",
            },
        )
        # repair merged the adjacent prior user rows, then the turn ran.
        agent_messages = [
            {"role": "user", "content": "first question\n\nsecond question"},
            {"role": "assistant", "content": "combined answer"},
            {"role": "user", "content": "third question", "_db_persisted": True},
            {"role": "assistant", "content": "third answer", "_db_persisted": True},
        ]

        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_run_agent", new_callable=AsyncMock) as mock_run:
                mock_run.return_value = (
                    {
                        "final_response": "third answer",
                        "messages": list(agent_messages),
                        "api_calls": 1,
                    },
                    {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
                )
                resp = await cli.post(
                    "/v1/responses",
                    json={
                        "model": "hermes-agent",
                        "input": "third question",
                        "previous_response_id": "resp_prev",
                    },
                )

            assert resp.status == 200
            data = await resp.json()
            stored = adapter._response_store.get(data["id"])["conversation_history"]
            assert stored == agent_messages
            for needle in ("first question", "second question", "third question"):
                rows = [
                    m for m in stored
                    if m.get("role") == "user" and needle in str(m.get("content"))
                ]
                assert len(rows) == 1, f"{needle!r} stored {len(rows)} times"

    @pytest.mark.asyncio
    async def test_tool_bearing_perturbed_turn_keeps_tool_rows_once(self, adapter):
        """Perturbed transcripts with tool calls keep every tool row exactly once."""
        prior_history = [
            {"role": "user", "content": "look something up"},
            {"role": "user", "content": "please"},
            {"role": "assistant", "content": "done"},
        ]
        adapter._response_store.put(
            "resp_prev",
            {
                "response": {"id": "resp_prev", "status": "completed"},
                "conversation_history": list(prior_history),
                "session_id": "api-test-session",
            },
        )
        agent_messages = [
            {"role": "user", "content": "look something up\n\nplease"},
            {"role": "assistant", "content": "done"},
            {
                "role": "user",
                "content": "read the file",
                "api_content": "<recalled_memories>y</recalled_memories>\n\nread the file",
            },
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "function": {"name": "read_file", "arguments": '{"path":"a.txt"}'},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call_1", "content": '{"content":"data"}'},
            {"role": "assistant", "content": "file says data"},
        ]

        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_run_agent", new_callable=AsyncMock) as mock_run:
                mock_run.return_value = (
                    {
                        "final_response": "file says data",
                        "messages": list(agent_messages),
                        "api_calls": 1,
                    },
                    {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
                )
                resp = await cli.post(
                    "/v1/responses",
                    json={
                        "model": "hermes-agent",
                        "input": "read the file",
                        "previous_response_id": "resp_prev",
                    },
                )

            assert resp.status == 200
            data = await resp.json()
            stored = adapter._response_store.get(data["id"])["conversation_history"]
            assert stored == agent_messages
            tool_rows = [m for m in stored if m.get("role") == "tool"]
            assert len(tool_rows) == 1
            # Output items replay only the current turn's tool artifacts:
            # one function_call and one function_call_output for call_1.
            output_types = [item["type"] for item in data["output"]]
            assert output_types.count("function_call") == 1
            assert output_types.count("function_call_output") == 1
            assert "call_1" in json.dumps(data["output"])

    @pytest.mark.asyncio
    async def test_suffix_only_result_messages_still_concatenate(self, adapter):
        """Assistant/tool-only suffix results keep the concatenation semantics."""
        prior_history = [
            {"role": "user", "content": "What is 1+1?"},
            {"role": "assistant", "content": "2"},
        ]
        adapter._response_store.put(
            "resp_prev",
            {
                "response": {"id": "resp_prev", "status": "completed"},
                "conversation_history": list(prior_history),
                "session_id": "api-test-session",
            },
        )
        suffix_messages = [
            {
                "role": "assistant",
                "tool_calls": [
                    {"id": "call_add", "function": {"name": "add", "arguments": '{"a":2,"b":1}'}}
                ],
            },
            {"role": "tool", "tool_call_id": "call_add", "content": "3"},
            {"role": "assistant", "content": "3"},
        ]

        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_run_agent", new_callable=AsyncMock) as mock_run:
                mock_run.return_value = (
                    {
                        "final_response": "3",
                        "messages": list(suffix_messages),
                        "api_calls": 1,
                    },
                    {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
                )
                resp = await cli.post(
                    "/v1/responses",
                    json={
                        "model": "hermes-agent",
                        "input": "Now add 1 more",
                        "previous_response_id": "resp_prev",
                    },
                )

            assert resp.status == 200
            data = await resp.json()
            stored = adapter._response_store.get(data["id"])["conversation_history"]
            assert stored == prior_history + [
                {"role": "user", "content": "Now add 1 more"}
            ] + suffix_messages

    @pytest.mark.asyncio
    async def test_suffix_starting_at_current_user_row_prepends_prior_once(self, adapter):
        """A suffix that already includes this turn's user row must not duplicate it."""
        prior_history = [
            {"role": "user", "content": "What is 1+1?"},
            {"role": "assistant", "content": "2"},
        ]
        adapter._response_store.put(
            "resp_prev",
            {
                "response": {"id": "resp_prev", "status": "completed"},
                "conversation_history": list(prior_history),
                "session_id": "api-test-session",
            },
        )
        suffix_messages = [
            {"role": "user", "content": "Now add 1 more"},
            {"role": "assistant", "content": "3"},
        ]

        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_run_agent", new_callable=AsyncMock) as mock_run:
                mock_run.return_value = (
                    {
                        "final_response": "3",
                        "messages": list(suffix_messages),
                        "api_calls": 1,
                    },
                    {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
                )
                resp = await cli.post(
                    "/v1/responses",
                    json={
                        "model": "hermes-agent",
                        "input": "Now add 1 more",
                        "previous_response_id": "resp_prev",
                    },
                )

            assert resp.status == 200
            data = await resp.json()
            stored = adapter._response_store.get(data["id"])["conversation_history"]
            assert stored == prior_history + suffix_messages
            user_rows = [
                m for m in stored
                if m.get("role") == "user" and m.get("content") == "Now add 1 more"
            ]
            assert len(user_rows) == 1

    @pytest.mark.asyncio
    async def test_nudge_bearing_suffix_preserves_prior_history(self, adapter):
        """A suffix with a mid-turn user-role nudge must not drop prior turns.

        conversation_loop.py appends user-role rows mid-turn (recovery and
        verification nudges, continue markers), so a mocked/older host could
        return a suffix whose LAST user row is a nudge rather than the turn
        anchor. The current-user anchor must still find this turn's user row
        at index 0 and prepend the prior history exactly once.
        """
        prior_history = [
            {"role": "user", "content": "What is 1+1?"},
            {"role": "assistant", "content": "2"},
        ]
        adapter._response_store.put(
            "resp_prev",
            {
                "response": {"id": "resp_prev", "status": "completed"},
                "conversation_history": list(prior_history),
                "session_id": "api-test-session",
            },
        )
        suffix_messages = [
            {"role": "user", "content": "Now add 1 more"},
            {"role": "assistant", "content": ""},
            {
                "role": "user",
                "content": "[System: Continue now. Execute the required tool calls.]",
            },
            {"role": "assistant", "content": "3"},
        ]

        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_run_agent", new_callable=AsyncMock) as mock_run:
                mock_run.return_value = (
                    {
                        "final_response": "3",
                        "messages": list(suffix_messages),
                        "api_calls": 1,
                    },
                    {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
                )
                resp = await cli.post(
                    "/v1/responses",
                    json={
                        "model": "hermes-agent",
                        "input": "Now add 1 more",
                        "previous_response_id": "resp_prev",
                    },
                )

            assert resp.status == 200
            data = await resp.json()
            stored = adapter._response_store.get(data["id"])["conversation_history"]
            assert stored == prior_history + suffix_messages
            assert stored[0] == prior_history[0], "prior history must be preserved"
            current_rows = [
                m for m in stored
                if m.get("role") == "user" and m.get("content") == "Now add 1 more"
            ]
            assert len(current_rows) == 1

    @pytest.mark.asyncio
    async def test_nudge_only_suffix_without_current_user_keeps_prior_and_user(self, adapter):
        """A nudge-bearing suffix lacking this turn's user row keeps the
        legacy ``prior + current_user + suffix`` shape — adopt-verbatim must
        never fire without structural proof that the result embeds prior
        history, or the prior turns would be silently dropped."""
        prior_history = [
            {"role": "user", "content": "What is 1+1?"},
            {"role": "assistant", "content": "2"},
        ]
        adapter._response_store.put(
            "resp_prev",
            {
                "response": {"id": "resp_prev", "status": "completed"},
                "conversation_history": list(prior_history),
                "session_id": "api-test-session",
            },
        )
        suffix_messages = [
            {"role": "assistant", "content": ""},
            {
                "role": "user",
                "content": "[System: Continue now. Execute the required tool calls.]",
            },
            {"role": "assistant", "content": "3"},
        ]

        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_run_agent", new_callable=AsyncMock) as mock_run:
                mock_run.return_value = (
                    {
                        "final_response": "3",
                        "messages": list(suffix_messages),
                        "api_calls": 1,
                    },
                    {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
                )
                resp = await cli.post(
                    "/v1/responses",
                    json={
                        "model": "hermes-agent",
                        "input": "Now add 1 more",
                        "previous_response_id": "resp_prev",
                    },
                )

            assert resp.status == 200
            data = await resp.json()
            stored = adapter._response_store.get(data["id"])["conversation_history"]
            assert stored == prior_history + [
                {"role": "user", "content": "Now add 1 more"}
            ] + suffix_messages
            assert stored[0] == prior_history[0], "prior history must be preserved"

    @pytest.mark.asyncio
    async def test_turn_after_incomplete_snapshot_stores_each_message_once(self, adapter):
        """Chaining off an incomplete snapshot (trailing user row) must not double.

        An aborted stream persists ``prior + user`` with no assistant reply.
        On the next turn the agent's alternation repair merges that trailing
        user row with the new user message, so no exact current-user row
        exists in the transcript — the store must still keep each real
        message exactly once.
        """
        incomplete_history = [
            {"role": "user", "content": "first message"},
        ]
        adapter._response_store.put(
            "resp_prev",
            {
                "response": {"id": "resp_prev", "status": "incomplete"},
                "conversation_history": list(incomplete_history),
                "session_id": "api-test-session",
            },
        )
        # repair merged the orphaned user row into this turn's user message.
        agent_messages = [
            {"role": "user", "content": "first message\n\nsecond message"},
            {"role": "assistant", "content": "answer", "_db_persisted": True},
        ]

        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_run_agent", new_callable=AsyncMock) as mock_run:
                mock_run.return_value = (
                    {
                        "final_response": "answer",
                        "messages": list(agent_messages),
                        "api_calls": 1,
                    },
                    {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
                )
                resp = await cli.post(
                    "/v1/responses",
                    json={
                        "model": "hermes-agent",
                        "input": "second message",
                        "previous_response_id": "resp_prev",
                    },
                )

            assert resp.status == 200
            data = await resp.json()
            stored = adapter._response_store.get(data["id"])["conversation_history"]
            assert stored == agent_messages
            first_rows = [
                m for m in stored
                if m.get("role") == "user" and "first message" in str(m.get("content"))
            ]
            assert len(first_rows) == 1

    @pytest.mark.asyncio
    async def test_streamed_memory_sidecar_chained_turns_store_each_message_once(self, adapter):
        """The streaming persistence path shares the doubling fix."""
        turns = [
            ("stream turn one", "reply one"),
            ("stream turn two", "reply two"),
        ]

        def _make_mock(reply):
            async def _mock_run_agent(**kwargs):
                cb = kwargs.get("stream_delta_callback")
                if cb:
                    cb(reply)
                return (
                    {
                        "final_response": reply,
                        "messages": self._simulate_staging_agent_turn(
                            kwargs["conversation_history"],
                            kwargs["user_message"],
                            reply,
                        ),
                        "api_calls": 1,
                    },
                    {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
                )
            return _mock_run_agent

        app = _create_app(adapter)
        prev_id = None
        stored_histories = []
        async with TestClient(TestServer(app)) as cli:
            for user_text, reply in turns:
                payload = {
                    "model": "hermes-agent",
                    "input": user_text,
                    "stream": True,
                }
                if prev_id:
                    payload["previous_response_id"] = prev_id
                with patch.object(
                    adapter, "_run_agent", side_effect=_make_mock(reply)
                ):
                    resp = await cli.post("/v1/responses", json=payload)
                    body = await resp.text()
                assert resp.status == 200
                prev_id = None
                for event in _parse_sse_events(body):
                    if event.get("type") == "response.completed":
                        prev_id = event["response"]["id"]
                        break
                assert prev_id
                stored_histories.append(
                    adapter._response_store.get(prev_id)["conversation_history"]
                )

        for turn_number, history in enumerate(stored_histories, start=1):
            for user_text, _reply in turns[:turn_number]:
                rows = [
                    m for m in history
                    if m.get("role") == "user" and user_text in str(m.get("content"))
                ]
                assert len(rows) == 1, (
                    f"after streamed turn {turn_number}, {user_text!r} stored "
                    f"{len(rows)} times"
                )
        assert [len(h) for h in stored_histories] == [2, 4]

    @pytest.mark.asyncio
    async def test_previous_response_id_stores_compacted_transcript_as_authoritative(self, adapter):
        """After compression, previous_response_id must resume compacted history."""
        prior_history = [
            {"role": "user", "content": "old large turn"},
            {"role": "assistant", "content": "old answer"},
        ]
        adapter._response_store.put(
            "resp_prev",
            {
                "response": {"id": "resp_prev", "status": "completed"},
                "conversation_history": list(prior_history),
                "session_id": "root-session",
            },
        )
        compacted_history = [
            {"role": "system", "content": "system prompt"},
            {
                "role": "assistant",
                "content": "[CONTEXT COMPACTION - REFERENCE ONLY]\n## State Ledger\nactive_task: Continue",
            },
            {"role": "user", "content": "Now add 1 more"},
            {"role": "assistant", "content": "3"},
        ]

        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_run_agent", new_callable=AsyncMock) as mock_run:
                mock_run.return_value = (
                    {
                        "final_response": "3",
                        "messages": list(compacted_history),
                        "session_id": "compressed-session",
                        "api_calls": 1,
                    },
                    {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
                )
                resp = await cli.post(
                    "/v1/responses",
                    json={
                        "model": "hermes-agent",
                        "input": "Now add 1 more",
                        "previous_response_id": "resp_prev",
                    },
                )
                assert resp.status == 200
                data = await resp.json()

        stored = adapter._response_store.get(data["id"])
        assert stored["conversation_history"] == compacted_history
        assert stored["session_id"] == "compressed-session"
        assert prior_history[0] not in stored["conversation_history"]

    @pytest.mark.asyncio
    async def test_previous_response_id_outputs_only_current_turn_items(self, adapter):
        """Response output must not replay previous tool artifacts."""
        prior_history = [
            {"role": "user", "content": "Read old file"},
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "call_old",
                        "function": {
                            "name": "read_file",
                            "arguments": '{"path":"old.txt"}',
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call_old",
                "content": '{"content":"old"}',
            },
            {"role": "assistant", "content": "old"},
        ]
        adapter._response_store.put(
            "resp_prev",
            {
                "response": {"id": "resp_prev", "status": "completed"},
                "conversation_history": list(prior_history),
                "session_id": "api-test-session",
            },
        )
        full_agent_transcript = prior_history + [
            {"role": "user", "content": "Read new file"},
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "call_new",
                        "function": {
                            "name": "read_file",
                            "arguments": '{"path":"new.txt"}',
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call_new",
                "content": '{"content":"new"}',
            },
            {"role": "assistant", "content": "new"},
        ]

        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_run_agent", new_callable=AsyncMock) as mock_run:
                mock_run.return_value = (
                    {
                        "final_response": "new",
                        "messages": list(full_agent_transcript),
                        "api_calls": 1,
                    },
                    {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
                )
                resp = await cli.post(
                    "/v1/responses",
                    json={
                        "model": "hermes-agent",
                        "input": "Read new file",
                        "previous_response_id": "resp_prev",
                    },
                )
                assert resp.status == 200
                data = await resp.json()

        output_json = json.dumps(data["output"])
        assert "call_new" in output_json
        assert "call_old" not in output_json
        assert "old.txt" not in output_json


    @pytest.mark.asyncio
    async def test_non_streaming_shuts_down_memory_provider(self, adapter):
        mock_agent = _make_api_agent("Hello!")
        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_create_agent", return_value=mock_agent):
                resp = await cli.post(
                    "/v1/responses",
                    json={"model": "hermes-agent", "input": "Hello"},
                )
                assert resp.status == 200
                data = await resp.json()

        assert data["output"][-1]["content"][0]["text"] == "Hello!"
        mock_agent.shutdown_memory_provider.assert_called_once_with(mock_agent._session_messages)
        mock_agent.close.assert_not_called()

    @pytest.mark.asyncio
    async def test_invalid_previous_response_id_returns_404(self, adapter):
        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.post(
                "/v1/responses",
                json={
                    "model": "hermes-agent",
                    "input": "follow up",
                    "previous_response_id": "resp_nonexistent",
                },
            )
            assert resp.status == 404


    @pytest.mark.asyncio
    async def test_store_string_false_does_not_store(self, adapter):
        """Quoted false must preserve ephemeral store=false semantics."""
        mock_result = {"final_response": "OK", "messages": [], "api_calls": 1}

        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_run_agent", new_callable=AsyncMock) as mock_run:
                mock_run.return_value = (
                    mock_result,
                    {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
                )
                resp = await cli.post(
                    "/v1/responses",
                    json={
                        "model": "hermes-agent",
                        "input": "Hello",
                        "store": "false",
                    },
                )

            assert resp.status == 200
            data = await resp.json()
            assert adapter._response_store.get(data["id"]) is None

    @pytest.mark.asyncio
    async def test_instructions_inherited_from_previous(self, adapter):
        """If no instructions provided, carry forward from previous response."""
        mock_result = {"final_response": "Ahoy!", "messages": [], "api_calls": 1}

        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            # First request with instructions
            with patch.object(adapter, "_run_agent", new_callable=AsyncMock) as mock_run:
                mock_run.return_value = (mock_result, {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0})
                resp1 = await cli.post(
                    "/v1/responses",
                    json={
                        "model": "hermes-agent",
                        "input": "Hello",
                        "instructions": "Be a pirate",
                    },
                )

            data1 = await resp1.json()
            resp_id = data1["id"]

            # Second request without instructions
            with patch.object(adapter, "_run_agent", new_callable=AsyncMock) as mock_run:
                mock_run.return_value = (mock_result, {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0})
                resp2 = await cli.post(
                    "/v1/responses",
                    json={
                        "model": "hermes-agent",
                        "input": "Tell me more",
                        "previous_response_id": resp_id,
                    },
                )

            assert resp2.status == 200
            call_kwargs = mock_run.call_args.kwargs
            assert call_kwargs["ephemeral_system_prompt"] == "Be a pirate"


    @pytest.mark.asyncio
    async def test_result_error_fallback_is_redacted(self, adapter):
        raw_secret = "sk-responses-leak-1234567890"
        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_run_agent", new_callable=AsyncMock) as mock_run:
                mock_run.return_value = (
                    {
                        "final_response": "",
                        "error": f"provider auth failed OPENAI_API_KEY={raw_secret}",
                        "messages": [],
                        "api_calls": 1,
                    },
                    {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
                )
                resp = await cli.post(
                    "/v1/responses",
                    json={"model": "hermes-agent", "input": "Hello"},
                )

            assert resp.status == 200
            data = await resp.json()
            body = json.dumps(data)
            assert raw_secret not in body
            assert "OPENAI_API_KEY=" in body
            assert data["output"][0]["content"][0]["text"] != f"provider auth failed OPENAI_API_KEY={raw_secret}"


class TestResponsesStreaming:

    @pytest.mark.asyncio
    async def test_stream_true_returns_responses_sse(self, adapter):
        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            async def _mock_run_agent(**kwargs):
                cb = kwargs.get("stream_delta_callback")
                if cb:
                    cb("Hello")
                    cb(" world")
                return (
                    {"final_response": "Hello world", "messages": [], "api_calls": 1},
                    {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
                )

            with patch.object(adapter, "_run_agent", side_effect=_mock_run_agent):
                resp = await cli.post(
                    "/v1/responses",
                    json={"model": "hermes-agent", "input": "hi", "stream": True},
                )
                assert resp.status == 200
                assert "text/event-stream" in resp.headers.get("Content-Type", "")
                body = await resp.text()
                assert "event: response.created" in body
                assert "event: response.output_text.delta" in body
                assert "event: response.output_text.done" in body
                assert "event: response.completed" in body
                assert '"sequence_number":' in body
                assert '"logprobs": []' in body
                assert "Hello" in body
                assert " world" in body

    @pytest.mark.asyncio
    async def test_transient_owner_heartbeat_error_does_not_kill_stream(
        self, adapter
    ):
        app = _create_app(adapter)
        original_heartbeat = adapter._response_store.heartbeat
        heartbeat_calls = 0

        def flaky_heartbeat(*args, **kwargs):
            nonlocal heartbeat_calls
            heartbeat_calls += 1
            if heartbeat_calls == 1:
                raise OSError("transient database disconnect")
            return original_heartbeat(*args, **kwargs)

        async with TestClient(TestServer(app)) as cli:
            async def _mock_run_agent(**kwargs):
                callback = kwargs.get("stream_delta_callback")
                for part in ("still", " running", " safely"):
                    await asyncio.sleep(0.02)
                    callback(part)
                return (
                    {
                        "final_response": "still running safely",
                        "messages": [],
                        "api_calls": 1,
                    },
                    {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
                )

            with (
                patch.object(adapter, "_run_agent", side_effect=_mock_run_agent),
                patch.object(
                    adapter._response_store,
                    "heartbeat",
                    side_effect=flaky_heartbeat,
                ),
                patch(
                    "gateway.platforms.api_server._response_owner_heartbeat_seconds",
                    return_value=0.01,
                ),
                patch(
                    "gateway.platforms.api_server._response_owner_stale_seconds",
                    return_value=1.0,
                ),
            ):
                resp = await cli.post(
                    "/v1/responses",
                    json={"model": "hermes-agent", "input": "hi", "stream": True},
                )
                body = await resp.text()

        events = _parse_sse_events(body)
        assert any(event.get("type") == "response.completed" for event in events)
        assert not any(event.get("type") == "response.failed" for event in events)
        assert heartbeat_calls >= 2

    @pytest.mark.asyncio
    async def test_stream_terminal_event_preserves_plugin_metadata(self, adapter):
        metadata = {"dorvis_trace_manifest": {"trace_id": "1" * 32}}
        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            async def _mock_run_agent(**_kwargs):
                return (
                    {
                        "final_response": "OK",
                        "messages": [],
                        "api_calls": 1,
                        "response_metadata": metadata,
                    },
                    {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
                )

            with patch.object(adapter, "_run_agent", side_effect=_mock_run_agent):
                resp = await cli.post(
                    "/v1/responses",
                    json={"model": "hermes-agent", "input": "hi", "stream": True},
                )
                body = await resp.text()

            completed = next(
                event for event in _parse_sse_events(body)
                if event.get("type") == "response.completed"
            )
            assert completed["response"]["metadata"] == metadata

    @pytest.mark.asyncio
    async def test_stream_completed_preserves_responses_usage_context(self, adapter):
        app = _create_app(adapter)
        usage = {
            "input_tokens": 10,
            "output_tokens": 5,
            "total_tokens": 15,
            "context_used": 4096,
            "context_max": 8192,
            "context_percent": 50,
            "compressions": 2,
            "cost_usd": 0.0123,
        }

        async with TestClient(TestServer(app)) as cli:
            async def _mock_run_agent(**kwargs):
                cb = kwargs.get("stream_delta_callback")
                if cb:
                    cb("Usage visible")
                return (
                    {"final_response": "Usage visible", "messages": [], "api_calls": 1},
                    dict(usage),
                )

            with patch.object(adapter, "_run_agent", side_effect=_mock_run_agent):
                resp = await cli.post(
                    "/v1/responses",
                    json={"model": "hermes-agent", "input": "hi", "stream": True},
                )
                assert resp.status == 200
                body = await resp.text()

            completed = next(
                event for event in _parse_sse_events(body)
                if event.get("type") == "response.completed"
            )
            assert completed["response"]["usage"] == usage

            response_id = completed["response"]["id"]
            get_resp = await cli.get(f"/v1/responses/{response_id}")
            assert get_resp.status == 200
            stored = await get_resp.json()
            assert stored["usage"] == usage

    @pytest.mark.asyncio
    async def test_stream_emits_compression_lifecycle_events(self, adapter):
        """Compaction hooks surface as response.context_compression.* SSE events."""
        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            async def _mock_run_agent(**kwargs):
                compression_cb = kwargs.get("compression_event_callback")
                delta_cb = kwargs.get("stream_delta_callback")
                assert compression_cb is not None
                compression_cb("context_compression_started", {
                    "session_id": "sess-1",
                    "pre_message_count": 42,
                    "pre_tokens": 180000,
                    "model": "gpt-5.6-sol",
                    "focus_topic": None,
                })
                compression_cb("context_compression_completed", {
                    "session_id": "sess-1",
                    "pre_message_count": 42,
                    "post_message_count": 11,
                    "pre_tokens": 180000,
                    "post_tokens": 60000,
                    "quality_gate_passed": True,
                })
                if delta_cb:
                    delta_cb("Continuing after compaction")
                return (
                    {"final_response": "Continuing after compaction", "messages": [], "api_calls": 1},
                    {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
                )

            with patch.object(adapter, "_run_agent", side_effect=_mock_run_agent):
                resp = await cli.post(
                    "/v1/responses",
                    json={"model": "hermes-agent", "input": "hi", "stream": True},
                )
                assert resp.status == 200
                body = await resp.text()

            assert "event: response.context_compression.started" in body
            assert "event: response.context_compression.completed" in body
            events = _parse_sse_events(body)
            started = next(
                e for e in events
                if e.get("type") == "response.context_compression.started"
            )
            assert started["pre_message_count"] == 42
            assert started["pre_tokens"] == 180000
            assert "sequence_number" in started
            # Free-text hook fields never reach the client stream.
            assert "model" not in started
            assert "focus_topic" not in started
            completed = next(
                e for e in events
                if e.get("type") == "response.context_compression.completed"
            )
            assert completed["post_message_count"] == 11
            assert completed["post_tokens"] == 60000
            assert completed["quality_gate_passed"] is True

    @pytest.mark.asyncio
    async def test_stream_compression_event_whitelist_and_unknown_phase(self, adapter):
        """Aborted events drop raw error text; unknown phases are not emitted."""
        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            async def _mock_run_agent(**kwargs):
                compression_cb = kwargs.get("compression_event_callback")
                compression_cb("context_compression_aborted", {
                    "session_id": "sess-1",
                    "pre_message_count": 40,
                    "abort_reason": "provider exploded: secret-detail",
                    "quality_gate_passed": False,
                })
                compression_cb("context_compression_bogus", {"pre_message_count": 1})
                return (
                    {"final_response": "ok", "messages": [], "api_calls": 1},
                    {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
                )

            with patch.object(adapter, "_run_agent", side_effect=_mock_run_agent):
                resp = await cli.post(
                    "/v1/responses",
                    json={"model": "hermes-agent", "input": "hi", "stream": True},
                )
                assert resp.status == 200
                body = await resp.text()

            assert "event: response.context_compression.aborted" in body
            assert "secret-detail" not in body
            assert "context_compression.bogus" not in body
            assert "context_compression_bogus" not in body
            aborted = next(
                e for e in _parse_sse_events(body)
                if e.get("type") == "response.context_compression.aborted"
            )
            assert aborted["pre_message_count"] == 40
            assert aborted["quality_gate_passed"] is False
            assert "abort_reason" not in aborted

    @pytest.mark.asyncio
    async def test_stream_failed_preserves_usage_and_structured_error(self, adapter):
        app = _create_app(adapter)
        usage = {
            "input_tokens": 25,
            "output_tokens": 0,
            "total_tokens": 25,
            "context_used": 7000,
            "context_max": 8000,
            "context_percent": 88,
            "compressions": 1,
            "cost_usd": 0.02,
        }

        async with TestClient(TestServer(app)) as cli:
            async def _mock_run_agent(**kwargs):
                return (
                    {
                        "final_response": "",
                        "messages": [],
                        "api_calls": 1,
                        "error": "OpenAI quota exceeded for this account",
                    },
                    dict(usage),
                )

            with patch.object(adapter, "_run_agent", side_effect=_mock_run_agent):
                resp = await cli.post(
                    "/v1/responses",
                    json={"model": "hermes-agent", "input": "hi", "stream": True},
                )
                assert resp.status == 200
                body = await resp.text()

        failed = next(
            event for event in _parse_sse_events(body)
            if event.get("type") == "response.failed"
        )
        assert failed["error"]["type"] == "quota_exceeded"
        assert failed["response"]["error"]["type"] == "quota_exceeded"
        assert failed["response"]["usage"] == usage

    @pytest.mark.asyncio
    async def test_stream_agent_exception_emits_failed_event(self, adapter):
        """An agent exception remains a terminal Responses API failure."""
        app = _create_app(adapter)

        async with TestClient(TestServer(app)) as cli:
            with patch.object(
                adapter,
                "_run_agent",
                side_effect=RuntimeError("provider unavailable"),
            ):
                resp = await cli.post(
                    "/v1/responses",
                    json={"model": "hermes-agent", "input": "hi", "stream": True},
                )
                assert resp.status == 200
                body = await resp.text()

        failed = next(
            event for event in _parse_sse_events(body)
            if event.get("type") == "response.failed"
        )
        assert failed["response"]["status"] == "failed"
        assert failed["error"]["type"] == "provider_error"
        assert "provider unavailable" in failed["error"]["message"]
        assert "UnboundLocalError" not in body

    @pytest.mark.asyncio
    async def test_stream_string_false_returns_json_response(self, adapter):
        """Quoted false must not route Responses API requests into SSE mode."""
        mock_result = {
            "final_response": "Paris is the capital of France.",
            "messages": [],
            "api_calls": 1,
        }

        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_run_agent", new_callable=AsyncMock) as mock_run:
                mock_run.return_value = (
                    mock_result,
                    {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
                )
                resp = await cli.post(
                    "/v1/responses",
                    json={
                        "model": "hermes-agent",
                        "input": "What is the capital of France?",
                        "stream": "false",
                    },
                )

            assert resp.status == 200
            assert "text/event-stream" not in resp.headers.get("Content-Type", "")
            data = await resp.json()
            assert data["object"] == "response"
            assert data["output"][0]["content"][0]["text"] == mock_result["final_response"]

    @pytest.mark.asyncio
    async def test_stream_task_done_callback_enqueues_eos_for_responses(self, adapter):
        """Regression guard for #24451 on /v1/responses streaming path."""
        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            class _FakeTask:
                def __init__(self):
                    self.callbacks = []

                def add_done_callback(self, cb):
                    self.callbacks.append(cb)

            fake_task = _FakeTask()

            def _fake_ensure_future(coro):
                # We short-circuit task scheduling in this unit test.
                coro.close()
                return fake_task

            with (
                patch.object(
                    adapter,
                    "_run_agent",
                    new=AsyncMock(
                        return_value=(
                            {"final_response": "ok", "messages": [], "api_calls": 1},
                            {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
                        )
                    ),
                ),
                patch("gateway.platforms.api_server.asyncio.ensure_future", side_effect=_fake_ensure_future),
                patch.object(adapter, "_write_sse_responses", new_callable=AsyncMock) as mock_write_sse,
            ):
                mock_write_sse.return_value = web.Response(status=200, text="ok")
                resp = await cli.post(
                    "/v1/responses",
                    json={"model": "hermes-agent", "input": "hi", "stream": True},
                )
                assert resp.status == 200

            assert len(fake_task.callbacks) == 1
            stream_q = mock_write_sse.call_args.kwargs["stream_q"]
            assert stream_q.empty()
            fake_task.callbacks[0](fake_task)
            assert stream_q.get_nowait() is None


    @pytest.mark.asyncio
    async def test_streamed_previous_response_id_stores_compacted_session(self, adapter):
        prior_history = [
            {"role": "user", "content": "old large turn"},
            {"role": "assistant", "content": "old answer"},
        ]
        adapter._response_store.put(
            "resp_prev",
            {
                "response": {"id": "resp_prev", "status": "completed"},
                "conversation_history": list(prior_history),
                "session_id": "root-session",
            },
        )
        compacted_history = [
            {"role": "system", "content": "system prompt"},
            {
                "role": "assistant",
                "content": "[CONTEXT COMPACTION - REFERENCE ONLY]\n## State Ledger\nactive_task: Continue",
            },
            {"role": "user", "content": "Now add 1 more"},
            {"role": "assistant", "content": "3"},
        ]

        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            async def _mock_run_agent(**kwargs):
                cb = kwargs.get("stream_delta_callback")
                if cb:
                    cb("3")
                return (
                    {
                        "final_response": "3",
                        "messages": list(compacted_history),
                        "session_id": "compressed-session",
                        "api_calls": 1,
                    },
                    {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
                )

            with patch.object(adapter, "_run_agent", side_effect=_mock_run_agent):
                resp = await cli.post(
                    "/v1/responses",
                    json={
                        "model": "hermes-agent",
                        "input": "Now add 1 more",
                        "previous_response_id": "resp_prev",
                        "stream": True,
                    },
                )
                body = await resp.text()

        assert resp.status == 200
        response_id = None
        for line in body.splitlines():
            if line.startswith("data: "):
                try:
                    payload = json.loads(line[len("data: "):])
                except json.JSONDecodeError:
                    continue
                if payload.get("type") == "response.completed":
                    response_id = payload["response"]["id"]
                    break

        assert response_id
        stored = adapter._response_store.get(response_id)
        assert stored["conversation_history"] == compacted_history
        assert stored["session_id"] == "compressed-session"
        assert prior_history[0] not in stored["conversation_history"]

    @pytest.mark.asyncio
    async def test_stream_cancelled_persists_incomplete_snapshot(self, adapter):
        """Server-side asyncio.CancelledError (shutdown, request timeout) must
        still leave an ``incomplete`` snapshot in ResponseStore so
        GET /v1/responses/{id} and previous_response_id chaining keep
        working.  Regression for PR #15171 follow-up.

        Calls _write_sse_responses directly so the test can await the
        handler to completion (TestClient disconnection races the server
        handler, which makes end-to-end assertion on the final stored
        snapshot flaky).
        """
        # Build a minimal fake request + stream queue the writer understands.
        fake_request = MagicMock()
        fake_request.headers = {}

        written_payloads: list = []

        class _FakeStreamResponse:
            async def prepare(self, req):
                pass

            async def write(self, payload):
                written_payloads.append(payload)

        # Patch web.StreamResponse for the duration of the writer call.
        import gateway.platforms.api_server as api_mod

        # The SSE writers consume an asyncio queue (ThreadSafeAsyncQueue),
        # not a plain queue.Queue — a stdlib queue would block the drain
        # loop's ``await stream_q.get()`` forever.
        stream_q = api_mod.ThreadSafeAsyncQueue()

        async def _agent_coro():
            # Feed one partial delta into the stream queue...
            stream_q.put_nowait("partial output")
            # ...then give the drain loop a moment to pick it up before
            # raising CancelledError to simulate a server-side cancel.
            await asyncio.sleep(0.01)
            raise asyncio.CancelledError()

        agent_task = asyncio.ensure_future(_agent_coro())
        response_id = f"resp_{uuid.uuid4().hex[:28]}"

        with patch.object(api_mod.web, "StreamResponse", return_value=_FakeStreamResponse()):
            with pytest.raises(asyncio.CancelledError):
                await adapter._write_sse_responses(
                    request=fake_request,
                    response_id=response_id,
                    model="hermes-agent",
                    created_at=int(time.time()),
                    stream_q=stream_q,
                    agent_task=agent_task,
                    agent_ref=[None],
                    conversation_history=[],
                    user_message="will be cancelled",
                    instructions=None,
                    conversation=None,
                    store=True,
                    session_id=None,
                )

        # The in_progress snapshot was persisted on response.created,
        # and the CancelledError handler must have updated it to
        # ``incomplete`` with the partial text it saw.
        stored = adapter._response_store.get(response_id)
        assert stored is not None, "snapshot must be retrievable after cancellation"
        assert stored["response"]["status"] == "incomplete"
        # Partial text captured before cancel should be preserved.
        output_text = "".join(
            part.get("text", "")
            for item in stored["response"].get("output", [])
            if item.get("type") == "message"
            for part in item.get("content", [])
        )
        assert "partial output" in output_text

    @pytest.mark.asyncio
    async def test_stream_client_disconnect_persists_incomplete_snapshot(self, adapter):
        """Client disconnect (ConnectionResetError) during streaming must
        persist an ``incomplete`` snapshot in ResponseStore.  Regression
        for PR #15171."""
        fake_request = MagicMock()
        fake_request.headers = {}

        write_call_count = {"n": 0}

        class _DisconnectingStreamResponse:
            async def prepare(self, req):
                pass

            async def write(self, payload):
                # First two writes succeed (prepare + response.created).
                # On the third write (a text delta), the "client"
                # disconnects — simulate with ConnectionResetError.
                write_call_count["n"] += 1
                if write_call_count["n"] >= 3:
                    raise ConnectionResetError("simulated client disconnect")

        import gateway.platforms.api_server as api_mod

        # asyncio queue to match the writers' consumer (see the note in
        # test_stream_cancelled_persists_incomplete_snapshot).
        stream_q = api_mod.ThreadSafeAsyncQueue()
        stream_q.put_nowait("some streamed text")
        stream_q.put_nowait(None)  # EOS sentinel

        async def _agent_coro():
            await asyncio.sleep(0.01)
            return ({"final_response": "", "messages": [], "api_calls": 0},
                    {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0})

        agent_task = asyncio.ensure_future(_agent_coro())
        response_id = f"resp_{uuid.uuid4().hex[:28]}"

        with patch.object(api_mod.web, "StreamResponse", return_value=_DisconnectingStreamResponse()):
            await adapter._write_sse_responses(
                request=fake_request,
                response_id=response_id,
                model="hermes-agent",
                created_at=int(time.time()),
                stream_q=stream_q,
                agent_task=agent_task,
                agent_ref=[None],
                conversation_history=[],
                user_message="will disconnect",
                instructions=None,
                conversation=None,
                store=True,
                session_id=None,
            )

        stored = adapter._response_store.get(response_id)
        assert stored is not None, "snapshot must survive client disconnect"
        assert stored["response"]["status"] == "incomplete"


# ---------------------------------------------------------------------------
# Auth on endpoints
# ---------------------------------------------------------------------------


class TestEndpointAuth:
    @pytest.mark.asyncio
    async def test_chat_completions_requires_auth(self, auth_adapter):
        app = _create_app(auth_adapter)
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.post(
                "/v1/chat/completions",
                json={"model": "test", "messages": [{"role": "user", "content": "hi"}]},
            )
            assert resp.status == 401


# ---------------------------------------------------------------------------
# Config integration
# ---------------------------------------------------------------------------


class TestConfigIntegration:
    def test_platform_enum_has_api_server(self):
        assert Platform.API_SERVER.value == "api_server"


    def test_env_override_cors_origins(self, monkeypatch):
        monkeypatch.setenv("API_SERVER_ENABLED", "true")
        monkeypatch.setenv("API_SERVER_KEY", "opensslrandhex32strongkey")
        monkeypatch.setenv(
            "API_SERVER_CORS_ORIGINS",
            "http://localhost:3000, http://127.0.0.1:3000",
        )
        from gateway.config import load_gateway_config
        config = load_gateway_config()
        assert config.platforms[Platform.API_SERVER].extra.get("cors_origins") == [
            "http://localhost:3000",
            "http://127.0.0.1:3000",
        ]

    def test_api_server_in_connected_platforms(self):
        config = GatewayConfig()
        config.platforms[Platform.API_SERVER] = PlatformConfig(
            enabled=True, extra={"key": "opensslrandhex32strongkey"}
        )
        connected = config.get_connected_platforms()
        assert Platform.API_SERVER in connected


# ---------------------------------------------------------------------------
# Multiple system messages
# ---------------------------------------------------------------------------


class TestMultipleSystemMessages:
    @pytest.mark.asyncio
    async def test_multiple_system_messages_concatenated(self, adapter):
        mock_result = {"final_response": "OK", "messages": [], "api_calls": 1}

        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_run_agent", new_callable=AsyncMock) as mock_run:
                mock_run.return_value = (mock_result, {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0})
                resp = await cli.post(
                    "/v1/chat/completions",
                    json={
                        "model": "hermes-agent",
                        "messages": [
                            {"role": "system", "content": "You are helpful."},
                            {"role": "system", "content": "Be concise."},
                            {"role": "user", "content": "Hello"},
                        ],
                    },
                )

            assert resp.status == 200
            call_kwargs = mock_run.call_args.kwargs
            prompt = call_kwargs["ephemeral_system_prompt"]
            assert "You are helpful." in prompt
            assert "Be concise." in prompt


# ---------------------------------------------------------------------------
# send() method (not used but required by base)
# ---------------------------------------------------------------------------


class TestSendMethod:
    @pytest.mark.asyncio
    async def test_send_returns_not_supported(self):
        config = PlatformConfig(enabled=True)
        adapter = APIServerAdapter(config)
        result = await adapter.send("chat1", "hello")
        assert result.success is False
        assert "HTTP request/response" in result.error


class TestPlatformEventCallbackEndpoint:

    @pytest.mark.asyncio
    async def test_rejects_invalid_google_chat_auth(self, adapter):
        app = _create_app(adapter)
        app["platform_event_adapters"] = {
            "google_chat": _FakeGoogleChatAdapter(
                verify_ok=False,
                verify_code="invalid_google_bearer",
            )
        }

        async with TestClient(TestServer(app)) as cli:
            resp = await cli.post(
                "/api/platforms/google_chat/events",
                headers={"Authorization": "Bearer bad"},
                json={"type": "MESSAGE"},
            )
            body = await resp.json()

        assert resp.status == 401
        assert body["error"]["code"] == "invalid_google_bearer"


# ---------------------------------------------------------------------------
# GET /v1/responses/{response_id}
# ---------------------------------------------------------------------------


class TestGetResponse:
    @pytest.mark.asyncio
    async def test_get_stored_response(self, adapter):
        """GET returns a previously stored response."""
        mock_result = {"final_response": "Hello!", "messages": [], "api_calls": 1}

        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            # Create a response first
            with patch.object(adapter, "_run_agent", new_callable=AsyncMock) as mock_run:
                mock_run.return_value = (mock_result, {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15})
                resp = await cli.post(
                    "/v1/responses",
                    json={"model": "hermes-agent", "input": "Hi"},
                )

            assert resp.status == 200
            data = await resp.json()
            response_id = data["id"]

            # Now GET it
            resp2 = await cli.get(f"/v1/responses/{response_id}")
            assert resp2.status == 200
            data2 = await resp2.json()
            assert data2["id"] == response_id
            assert data2["object"] == "response"
            assert data2["status"] == "completed"

    @pytest.mark.asyncio
    async def test_get_recovers_expired_owner_as_terminal_incomplete(self, adapter):
        """GET never leaves a dead owner's response permanently in progress."""
        response_id = "resp_expired_owner"
        assert adapter._response_store.claim(
            response_id,
            {
                "response": {
                    "id": response_id,
                    "object": "response",
                    "status": "in_progress",
                },
                "conversation_history": [],
            },
            owner_id="dead-owner",
            owner_epoch="dead-epoch",
        )
        with adapter._response_store._conn:
            adapter._response_store._conn.execute(
                "UPDATE responses SET owner_heartbeat_at = 1 WHERE response_id = ?",
                (response_id,),
            )

        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.get(f"/v1/responses/{response_id}")
            body = await resp.json()

        assert resp.status == 200
        assert body["status"] == "incomplete"
        assert body["incomplete_details"] == {"reason": "owner_lost"}
        assert adapter._response_store.get_control(response_id)["terminal"] is True


# ---------------------------------------------------------------------------
# POST /v1/responses/{response_id}/cancel
# ---------------------------------------------------------------------------


class _DualInterruptAgent:
    """Agent double that proves explicit stops prefer the hard-stop ABI."""

    def __init__(self):
        self.hard_interrupt = MagicMock()
        self.interrupt = MagicMock()
        self.session_prompt_tokens = 0
        self.session_completion_tokens = 0
        self.session_total_tokens = 0


class TestCancelResponse:
    """Terminate an abandoned streaming response without destroying it.

    A headless client that loses its SSE stream needs two things the pinned
    gateway could not give it: the runtime to stop executing tools on its
    behalf, and the stored envelope to survive so recovery and forensics can
    still read what the run did.  DELETE only ever offered the second half
    inverted — it removes exactly the record the client came back for.
    """

    @staticmethod
    def _start_stream(adapter, *, agent_ref, stream_q, agent_task, response_id):
        """Drive _write_sse_responses as a task, as _handle_responses does.

        Returns the writer task, the StreamResponse patcher, and the list of
        SSE payloads written so far — tests wait on an actual emitted event
        rather than on a sleep, since the writer batches text deltas.
        """
        import gateway.platforms.api_server as api_mod

        written: list[str] = []

        class _FakeStreamResponse:
            async def prepare(self, req):
                pass

            async def write(self, payload):
                written.append(payload.decode() if isinstance(payload, bytes) else str(payload))

        fake_request = MagicMock()
        fake_request.headers = {}
        agent_task.add_done_callback(lambda _fut: stream_q.put_nowait(None))
        patcher = patch.object(
            api_mod.web, "StreamResponse", return_value=_FakeStreamResponse()
        )
        patcher.start()
        writer = asyncio.ensure_future(
            adapter._write_sse_responses(
                request=fake_request,
                response_id=response_id,
                model="hermes-agent",
                created_at=int(time.time()),
                stream_q=stream_q,
                agent_task=agent_task,
                agent_ref=agent_ref,
                conversation_history=[],
                user_message="long running turn",
                instructions=None,
                conversation=None,
                store=True,
                session_id=None,
            )
        )
        return writer, patcher, written

    @staticmethod
    async def _await_event(written: list, event_type: str) -> None:
        for _ in range(200):
            await asyncio.sleep(0.01)
            if any(event_type in payload for payload in written):
                return
        raise AssertionError(f"SSE event {event_type} never arrived")

    @pytest.mark.asyncio
    async def test_cancel_live_response_stops_agent_and_preserves_envelope(self, adapter):
        """The acceptance case: a live turn is stopped, its record survives."""
        import gateway.platforms.api_server as api_mod

        agent = _DualInterruptAgent()
        agent_ref = [agent, None]
        stream_q = api_mod.ThreadSafeAsyncQueue()

        async def _agent_coro():
            stream_q.put_nowait("partial answer")
            await asyncio.sleep(30)
            return ({}, {})

        agent_task = asyncio.ensure_future(_agent_coro())
        response_id = f"resp_{uuid.uuid4().hex[:28]}"
        writer, patcher, written = self._start_stream(
            adapter,
            agent_ref=agent_ref,
            stream_q=stream_q,
            agent_task=agent_task,
            response_id=response_id,
        )
        try:
            app = _create_app(adapter)
            async with TestClient(TestServer(app)) as cli:
                await self._await_event(written, "response.output_text.delta")
                assert adapter._response_store.get(response_id)["response"]["status"] == (
                    "in_progress"
                )

                resp = await cli.post(f"/v1/responses/{response_id}/cancel")
                assert resp.status == 200
                body = await resp.json()
                assert body["id"] == response_id
                assert body["object"] == "response"
                assert body["status"] == "cancelled"
                assert adapter._response_store.get_control(response_id)["terminal"] is True

                # The agent loop was revoked, not merely detached from the stream.
                agent.hard_interrupt.assert_called_once_with(
                    "Cancelled via /v1/responses/{id}/cancel"
                )
                agent.interrupt.assert_not_called()

                with pytest.raises(asyncio.CancelledError):
                    await writer

                # GET still serves the envelope — this is what DELETE destroys.
                stored_resp = await cli.get(f"/v1/responses/{response_id}")
                assert stored_resp.status == 200
                stored_body = await stored_resp.json()
                assert stored_body["status"] == "cancelled"
                output_text = "".join(
                    part.get("text", "")
                    for item in stored_body.get("output", [])
                    if item.get("type") == "message"
                    for part in item.get("content", [])
                )
                assert "partial answer" in output_text
        finally:
            patcher.stop()
            adapter._inflight_responses.pop(response_id, None)

    @pytest.mark.asyncio
    async def test_cancel_does_not_report_success_when_interrupt_routing_fails(
        self, adapter
    ):
        agent = MagicMock()
        agent.interrupt.side_effect = RuntimeError("interrupt channel unavailable")
        agent_ref = [agent, None]
        stream_q = ThreadSafeAsyncQueue()
        release_owner = asyncio.Event()

        async def _agent_coro():
            await release_owner.wait()
            return (
                {"final_response": "still running", "messages": [], "api_calls": 1},
                {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
            )

        agent_task = asyncio.ensure_future(_agent_coro())
        response_id = f"resp_{uuid.uuid4().hex[:28]}"
        writer, patcher, written = self._start_stream(
            adapter,
            agent_ref=agent_ref,
            stream_q=stream_q,
            agent_task=agent_task,
            response_id=response_id,
        )
        try:
            await self._await_event(written, "response.created")
            app = _create_app(adapter)
            async with TestClient(TestServer(app)) as cli:
                response = await cli.post(f"/v1/responses/{response_id}/cancel")
                assert response.status == 503
                assert (await response.json())["error"]["code"] == (
                    "response_interrupt_unavailable"
                )

            assert adapter._response_store.get(response_id)["response"]["status"] == (
                "in_progress"
            )
            assert not agent_task.cancelled()
        finally:
            release_owner.set()
            await writer
            patcher.stop()
            adapter._inflight_responses.pop(response_id, None)

    @pytest.mark.asyncio
    async def test_accepted_cancel_suppresses_stale_owner_completion(self, adapter):
        """The durable CAS governs both storage and the terminal SSE event."""
        agent = MagicMock()
        agent_ref = [agent, None]
        stream_q = ThreadSafeAsyncQueue()
        release_owner = asyncio.Event()

        async def _agent_coro():
            await release_owner.wait()
            return (
                {"final_response": "late completion", "messages": [], "api_calls": 1},
                {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
            )

        agent_task = asyncio.ensure_future(_agent_coro())
        response_id = f"resp_{uuid.uuid4().hex[:28]}"
        writer, patcher, written = self._start_stream(
            adapter,
            agent_ref=agent_ref,
            stream_q=stream_q,
            agent_task=agent_task,
            response_id=response_id,
        )
        try:
            await self._await_event(written, "response.created")
            # Model an executor-backed run that cannot be cancelled by the
            # asyncio wrapper: the interrupt succeeds, then the owner still
            # returns a late result and attempts its terminal write.
            adapter._inflight_responses[response_id]["task"] = MagicMock(
                done=MagicMock(return_value=True)
            )
            app = _create_app(adapter)
            async with TestClient(TestServer(app)) as cli:
                response = await cli.post(f"/v1/responses/{response_id}/cancel")
                assert response.status == 200

            release_owner.set()
            await writer
            assert adapter._response_store.get(response_id)["response"]["status"] == (
                "cancelled"
            )
            assert not any("response.completed" in payload for payload in written)
            assert any("response.cancelled" in payload for payload in written)
        finally:
            release_owner.set()
            if not writer.done():
                await writer
            patcher.stop()
            adapter._inflight_responses.pop(response_id, None)

    @pytest.mark.asyncio
    async def test_cancel_during_created_delivery_is_durably_terminal(self, adapter):
        """The owner claim must exist before response.created becomes visible."""
        import gateway.platforms.api_server as api_mod

        created_write_started = asyncio.Event()
        release_created_write = asyncio.Event()

        class _BlockedCreatedStream:
            async def prepare(self, _request):
                return None

            async def write(self, payload):
                text = payload.decode() if isinstance(payload, bytes) else str(payload)
                if "response.created" in text:
                    created_write_started.set()
                    await release_created_write.wait()

        agent = MagicMock()
        agent_ref = [agent, None]
        stream_q = ThreadSafeAsyncQueue()

        async def _agent_coro():
            await asyncio.sleep(30)
            return ({}, {})

        agent_task = asyncio.ensure_future(_agent_coro())
        agent_task.add_done_callback(lambda _fut: stream_q.put_nowait(None))
        response_id = f"resp_{uuid.uuid4().hex[:28]}"
        fake_request = MagicMock()
        fake_request.headers = {}

        with patch.object(
            api_mod.web, "StreamResponse", return_value=_BlockedCreatedStream()
        ):
            writer = asyncio.create_task(
                adapter._write_sse_responses(
                    request=fake_request,
                    response_id=response_id,
                    model="hermes-agent",
                    created_at=int(time.time()),
                    stream_q=stream_q,
                    agent_task=agent_task,
                    agent_ref=agent_ref,
                    conversation_history=[],
                    user_message="cancel while created is flushing",
                    instructions=None,
                    conversation=None,
                    store=True,
                    session_id="session-created-race",
                )
            )
            try:
                await created_write_started.wait()
                app = _create_app(adapter)
                async with TestClient(TestServer(app)) as cli:
                    response = await cli.post(f"/v1/responses/{response_id}/cancel")
                    assert response.status == 200

                assert adapter._response_store.get(response_id)["response"]["status"] == (
                    "cancelled"
                )
                assert adapter._response_store.get_control(response_id)["terminal"] is True
            finally:
                release_created_write.set()
                with pytest.raises(asyncio.CancelledError):
                    await writer
                adapter._inflight_responses.pop(response_id, None)

    @pytest.mark.asyncio
    async def test_cancelled_envelope_carries_usage_accrued_before_the_cancel(
        self, adapter
    ):
        """An abandoned run's spend has to survive its cancellation.

        The writer's ``usage`` is only filled in when the agent task returns,
        so a cancel that beats the agent to the end published zeros — an
        abandoned headless run that had already paid for completed turns
        looked free to everything reading the envelope.
        """
        import gateway.platforms.api_server as api_mod

        agent = _make_api_agent()
        agent.session_prompt_tokens = 120
        agent.session_completion_tokens = 45
        agent.session_total_tokens = 165
        agent.session_api_calls = 2
        agent_ref = [agent, None]
        stream_q = api_mod.ThreadSafeAsyncQueue()

        async def _agent_coro():
            stream_q.put_nowait("partial answer")
            await asyncio.sleep(30)
            return ({}, {})

        agent_task = asyncio.ensure_future(_agent_coro())
        response_id = f"resp_{uuid.uuid4().hex[:28]}"
        writer, patcher, written = self._start_stream(
            adapter,
            agent_ref=agent_ref,
            stream_q=stream_q,
            agent_task=agent_task,
            response_id=response_id,
        )
        try:
            app = _create_app(adapter)
            async with TestClient(TestServer(app)) as cli:
                await self._await_event(written, "response.output_text.delta")

                resp = await cli.post(f"/v1/responses/{response_id}/cancel")
                assert resp.status == 200
                usage = (await resp.json())["usage"]
                assert usage["input_tokens"] == 120
                assert usage["output_tokens"] == 45
                assert usage["total_tokens"] == 165
                # Turn-boundary counters are all that exist: tokens spent
                # inside the request in flight at the interrupt are reported
                # by nobody, so this is an honest undercount, not a total.
                assert usage["completeness"] == "partial"
                assert "run_interrupted_before_completion" in usage["warnings"]

                with pytest.raises(asyncio.CancelledError):
                    await writer

                # The stored envelope is what run-cost accounting reads back.
                stored_body = await (await cli.get(f"/v1/responses/{response_id}")).json()
                assert stored_body["usage"]["total_tokens"] == 165
        finally:
            patcher.stop()
            adapter._inflight_responses.pop(response_id, None)

    def test_interrupted_usage_snapshot_skips_the_pricing_lookup(self, adapter):
        """Cost resolution can block on a provider request.

        Both interrupted-usage callers read the agent from the event loop
        while its own thread is still running, so that snapshot must not reach
        the pricing path — no envelope field is worth stalling the gateway
        for. The ordinary end-of-run snapshot, taken on the executor thread,
        still prices.
        """
        import gateway.platforms.api_server as api_mod

        agent = _make_api_agent()
        agent.model = "claude-opus-4"
        agent.provider = "anthropic"
        agent.base_url = ""

        with patch("agent.usage_pricing.estimate_usage_cost") as priced:
            snapshot = api_mod._interrupted_usage_snapshot(agent)
        priced.assert_not_called()
        assert snapshot["total_tokens"] == 3
        assert "cost_usd" not in snapshot
        assert "cost_status" not in snapshot

        with patch("agent.usage_pricing.estimate_usage_cost") as priced:
            api_mod._session_usage_snapshot(agent)
        priced.assert_called_once()

    @pytest.mark.asyncio
    async def test_incomplete_envelope_carries_usage_accrued_before_the_unwind(
        self, adapter
    ):
        """The disconnect unwind had the identical zero-usage gap.

        ``incomplete`` is the record a client that lost its stream comes back
        to, so it has to account for the run the same way ``cancelled`` does.
        """
        import gateway.platforms.api_server as api_mod

        class _DisconnectingStreamResponse:
            async def prepare(self, req):
                pass

            async def write(self, payload):
                raise ConnectionResetError("client went away")

        agent = _make_api_agent()
        agent.session_prompt_tokens = 90
        agent.session_completion_tokens = 30
        agent.session_total_tokens = 120
        agent.session_api_calls = 1
        agent_ref = [agent, None]
        stream_q = ThreadSafeAsyncQueue()

        async def _agent_coro():
            await asyncio.sleep(30)
            return ({}, {})

        agent_task = asyncio.ensure_future(_agent_coro())
        agent_task.add_done_callback(lambda _fut: stream_q.put_nowait(None))
        response_id = f"resp_{uuid.uuid4().hex[:28]}"
        fake_request = MagicMock()
        fake_request.headers = {}

        with patch.object(
            api_mod.web, "StreamResponse", return_value=_DisconnectingStreamResponse()
        ):
            await adapter._write_sse_responses(
                request=fake_request,
                response_id=response_id,
                model="hermes-agent",
                created_at=int(time.time()),
                stream_q=stream_q,
                agent_task=agent_task,
                agent_ref=agent_ref,
                conversation_history=[],
                user_message="long running turn",
                instructions=None,
                conversation=None,
                store=True,
                session_id="session-disconnect-usage",
            )

        stored = adapter._response_store.get(response_id)["response"]
        assert stored["status"] == "incomplete"
        assert stored["usage"]["input_tokens"] == 90
        assert stored["usage"]["output_tokens"] == 30
        assert stored["usage"]["total_tokens"] == 120
        assert stored["usage"]["completeness"] == "partial"
        assert "run_interrupted_before_completion" in stored["usage"]["warnings"]

    @pytest.mark.asyncio
    async def test_cancel_before_the_first_provider_response_reports_unavailable(
        self, adapter
    ):
        """An unknown spend must not be published as a measured zero.

        Cancelling during the very first provider request leaves nothing
        committed, and a bare ``{0, 0, 0}`` carries no way to tell that apart
        from a run that genuinely cost nothing — consumers read it as a
        reading. The aggregate's own ``unavailable`` says which it is.
        """
        agent = _make_api_agent()
        agent.session_prompt_tokens = 0
        agent.session_completion_tokens = 0
        agent.session_total_tokens = 0
        agent.session_api_calls = 0
        agent_ref = [agent, None]
        stream_q = ThreadSafeAsyncQueue()

        async def _agent_coro():
            await asyncio.sleep(30)
            return ({}, {})

        agent_task = asyncio.ensure_future(_agent_coro())
        response_id = f"resp_{uuid.uuid4().hex[:28]}"
        writer, patcher, written = self._start_stream(
            adapter,
            agent_ref=agent_ref,
            stream_q=stream_q,
            agent_task=agent_task,
            response_id=response_id,
        )
        try:
            app = _create_app(adapter)
            async with TestClient(TestServer(app)) as cli:
                await self._await_event(written, "response.created")

                resp = await cli.post(f"/v1/responses/{response_id}/cancel")
                usage = (await resp.json())["usage"]
                assert usage["total_tokens"] == 0
                assert usage["completeness"] == "unavailable"
                assert "no_provider_usage_reported" in usage["warnings"]
                assert "run_interrupted_before_completion" in usage["warnings"]

                with pytest.raises(asyncio.CancelledError):
                    await writer
        finally:
            patcher.stop()
            adapter._inflight_responses.pop(response_id, None)

    @pytest.mark.asyncio
    async def test_unwind_after_the_agent_finished_publishes_its_own_snapshot(
        self, adapter
    ):
        """The agent can beat the writer to the end and still be unwound.

        ``_run_agent`` drops its ``agent_ref`` on the executor thread before
        the task resolves, so a disconnect during the final drain finds no
        live agent to read while the finished task already holds a complete
        snapshot. Falling back to zeros there discards a figure the gateway
        genuinely has.
        """
        import gateway.platforms.api_server as api_mod

        class _DropOnDeltaStreamResponse:
            async def prepare(self, req):
                pass

            async def write(self, payload):
                text = payload.decode() if isinstance(payload, bytes) else str(payload)
                if "output_text.delta" in text:
                    raise ConnectionResetError("client went away")

        agent = _make_api_agent()
        agent_ref = [agent, None]
        stream_q = ThreadSafeAsyncQueue()

        async def _agent_coro():
            stream_q.put_nowait("partial answer")
            # Mirrors _run_agent's executor finally: the ref is cleared before
            # the future resolves.
            agent_ref[0] = None
            return (
                {"final_response": "partial answer", "messages": [], "api_calls": 1},
                {
                    "input_tokens": 90,
                    "output_tokens": 30,
                    "total_tokens": 120,
                    "scope": "run_aggregate",
                    "completeness": "complete",
                    "warnings": [],
                },
            )

        agent_task = asyncio.ensure_future(_agent_coro())
        agent_task.add_done_callback(lambda _fut: stream_q.put_nowait(None))
        response_id = f"resp_{uuid.uuid4().hex[:28]}"
        fake_request = MagicMock()
        fake_request.headers = {}

        with patch.object(
            api_mod.web,
            "StreamResponse",
            return_value=_DropOnDeltaStreamResponse(),
        ):
            await adapter._write_sse_responses(
                request=fake_request,
                response_id=response_id,
                model="hermes-agent",
                created_at=int(time.time()),
                stream_q=stream_q,
                agent_task=agent_task,
                agent_ref=agent_ref,
                conversation_history=[],
                user_message="long running turn",
                instructions=None,
                conversation=None,
                store=True,
                session_id="session-late-disconnect",
            )

        stored = adapter._response_store.get(response_id)["response"]
        assert stored["status"] == "incomplete"
        assert stored["usage"]["total_tokens"] == 120
        # The run itself finished; only its delivery was abandoned, so the
        # agent's own completeness stands rather than being downgraded.
        assert stored["usage"]["completeness"] == "complete"
        assert "run_interrupted_before_completion" not in stored["usage"]["warnings"]

    @pytest.mark.asyncio
    async def test_terminal_unstored_stream_reports_already_terminal(self, adapter):
        """``store=false`` leaves no envelope to read terminality from.

        The stored record is this route's usual witness, so a turn that had
        already finished without one was answered 200 — reporting containment
        of a run there was nothing left to contain. The in-flight entry knows
        the turn is over even when nothing was written down.
        """
        import gateway.platforms.api_server as api_mod

        release = asyncio.Event()
        written: list[str] = []

        class _ParkedStreamResponse:
            """Holds the stream open on its terminal event.

            The entry is unregistered the moment the writer returns, so the
            window this asserts on only exists while the terminal write is
            still in flight.
            """

            async def prepare(self, req):
                pass

            async def write(self, payload):
                text = payload.decode() if isinstance(payload, bytes) else str(payload)
                written.append(text)
                if "response.completed" in text:
                    await release.wait()

        agent = MagicMock()
        agent_ref = [agent, None]
        stream_q = ThreadSafeAsyncQueue()

        async def _agent_coro():
            return (
                {"final_response": "done", "messages": [], "api_calls": 1},
                {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
            )

        agent_task = asyncio.ensure_future(_agent_coro())
        agent_task.add_done_callback(lambda _fut: stream_q.put_nowait(None))
        response_id = f"resp_{uuid.uuid4().hex[:28]}"
        fake_request = MagicMock()
        fake_request.headers = {}

        with patch.object(
            api_mod.web, "StreamResponse", return_value=_ParkedStreamResponse()
        ):
            writer = asyncio.ensure_future(
                adapter._write_sse_responses(
                    request=fake_request,
                    response_id=response_id,
                    model="hermes-agent",
                    created_at=int(time.time()),
                    stream_q=stream_q,
                    agent_task=agent_task,
                    agent_ref=agent_ref,
                    conversation_history=[],
                    user_message="short turn",
                    instructions=None,
                    conversation=None,
                    store=False,
                    session_id=None,
                )
            )
            try:
                app = _create_app(adapter)
                async with TestClient(TestServer(app)) as cli:
                    await self._await_event(written, "response.completed")
                    assert adapter._response_store.get(response_id) is None

                    resp = await cli.post(f"/v1/responses/{response_id}/cancel")
                    assert resp.status == 409
                    assert (await resp.json())["error"]["code"] == (
                        "response_already_terminal"
                    )
                    agent.interrupt.assert_not_called()
            finally:
                release.set()
                await writer
                adapter._inflight_responses.pop(response_id, None)

    @pytest.mark.asyncio
    async def test_repeat_cancel_reports_already_terminal(self, adapter):
        """Idempotency: the second cancel is answered 409, not 200."""
        agent = MagicMock()
        agent_ref = [agent, None]
        stream_q = ThreadSafeAsyncQueue()

        async def _agent_coro():
            await asyncio.sleep(30)
            return ({}, {})

        agent_task = asyncio.ensure_future(_agent_coro())
        response_id = f"resp_{uuid.uuid4().hex[:28]}"
        writer, patcher, written = self._start_stream(
            adapter,
            agent_ref=agent_ref,
            stream_q=stream_q,
            agent_task=agent_task,
            response_id=response_id,
        )
        try:
            app = _create_app(adapter)
            async with TestClient(TestServer(app)) as cli:
                await self._await_event(written, "response.created")

                assert (await cli.post(f"/v1/responses/{response_id}/cancel")).status == 200
                repeat = await cli.post(f"/v1/responses/{response_id}/cancel")
                assert repeat.status == 409
                assert (await repeat.json())["error"]["code"] == "response_already_terminal"

                with pytest.raises(asyncio.CancelledError):
                    await writer
        finally:
            patcher.stop()
            adapter._inflight_responses.pop(response_id, None)

    @pytest.mark.asyncio
    async def test_cancel_completed_response_returns_409(self, adapter):
        """A finished response cannot be cancelled, and says so distinguishably."""
        mock_result = {"final_response": "Hello!", "messages": [], "api_calls": 1}

        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_run_agent", new_callable=AsyncMock) as mock_run:
                mock_run.return_value = (
                    mock_result,
                    {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
                )
                created = await cli.post(
                    "/v1/responses", json={"model": "hermes-agent", "input": "Hi"}
                )
            response_id = (await created.json())["id"]

            resp = await cli.post(f"/v1/responses/{response_id}/cancel")
            assert resp.status == 409
            body = await resp.json()
            assert body["error"]["code"] == "response_already_terminal"

            # 409 must not have mutated the record.
            stored = await cli.get(f"/v1/responses/{response_id}")
            assert (await stored.json())["status"] == "completed"

    @pytest.mark.asyncio
    async def test_cancel_incomplete_response_returns_409(self, adapter):
        """A disconnect-detected turn is already contained; report that."""
        response_id = "resp_incomplete"
        adapter._response_store.put(
            response_id,
            {
                "response": {
                    "id": response_id,
                    "object": "response",
                    "status": "incomplete",
                },
                "conversation_history": [],
                "instructions": None,
                "session_id": None,
            },
        )

        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.post(f"/v1/responses/{response_id}/cancel")
            assert resp.status == 409

    @pytest.mark.asyncio
    async def test_cancel_unknown_id_returns_openai_error_body(self, adapter):
        """404 must carry an OpenAI error body.

        The agent-platform worker distinguishes "the gateway knows this route
        and has never seen that id" from "this build has no cancel route" by
        the presence of that body; a bare framework 404 reads as unsupported.
        """
        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.post("/v1/responses/resp_nonexistent/cancel")
            assert resp.status == 404
            body = await resp.json()
            assert isinstance(body.get("error"), dict)
            assert body["error"]["code"] == "response_not_found"

    @pytest.mark.asyncio
    async def test_sibling_adapter_cannot_report_cancellation_it_cannot_route(
        self, adapter
    ):
        """A shared row is not proof that this process contained its owner.

        During a blue/green overlap, either ECS task can receive the cancel
        request while only the task that started the response holds the live
        agent handle.  The sibling must leave the row nonterminal and return a
        retryable non-success response instead of claiming cancellation.
        """
        agent = MagicMock()
        agent_ref = [agent, None]
        stream_q = ThreadSafeAsyncQueue()
        release_owner = asyncio.Event()

        async def _agent_coro():
            await release_owner.wait()
            return (
                {"final_response": "owner completed", "messages": [], "api_calls": 1},
                {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
            )

        agent_task = asyncio.ensure_future(_agent_coro())
        response_id = f"resp_{uuid.uuid4().hex[:28]}"
        writer, patcher, written = self._start_stream(
            adapter,
            agent_ref=agent_ref,
            stream_q=stream_q,
            agent_task=agent_task,
            response_id=response_id,
        )
        sibling = _make_adapter()
        sibling._response_store.close()
        sibling._response_store = adapter._response_store
        try:
            await self._await_event(written, "response.created")
            app = _create_app(sibling)
            async with TestClient(TestServer(app)) as cli:
                delete_response = await cli.delete(f"/v1/responses/{response_id}")
                assert delete_response.status == 409
                assert (await delete_response.json())["error"]["code"] == (
                    "response_not_terminal"
                )

                response = await cli.post(f"/v1/responses/{response_id}/cancel")
                assert response.status == 503
                assert (await response.json())["error"]["code"] == (
                    "response_owner_unavailable"
                )
                assert response.headers["Retry-After"]

            assert adapter._response_store.get(response_id)["response"]["status"] == (
                "in_progress"
            )
            agent.interrupt.assert_not_called()

            release_owner.set()
            await writer
            assert adapter._response_store.get(response_id)["response"]["status"] == (
                "completed"
            )
        finally:
            release_owner.set()
            if not writer.done():
                await writer
            patcher.stop()
            adapter._inflight_responses.pop(response_id, None)

    @pytest.mark.asyncio
    async def test_cancel_orphaned_in_progress_record_reports_owner_unavailable(
        self, adapter
    ):
        """A durable row alone cannot prove that its executor was contained."""
        response_id = "resp_orphan"
        adapter._response_store.put(
            response_id,
            {
                "response": {
                    "id": response_id,
                    "object": "response",
                    "status": "in_progress",
                    "output": [],
                },
                "conversation_history": [{"role": "user", "content": "hi"}],
                "instructions": None,
                "session_id": "sess-orphan",
            },
        )

        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.post(f"/v1/responses/{response_id}/cancel")
            assert resp.status == 503
            assert (await resp.json())["error"]["code"] == (
                "response_owner_unavailable"
            )

        stored = adapter._response_store.get(response_id)
        assert stored["response"]["status"] == "in_progress"
        assert stored["conversation_history"] == [{"role": "user", "content": "hi"}]
        assert stored["session_id"] == "sess-orphan"

    @pytest.mark.asyncio
    async def test_cancel_requires_auth(self, auth_adapter):
        app = _create_app(auth_adapter)
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.post("/v1/responses/resp_any/cancel")
            assert resp.status == 401

    def test_cancel_route_is_registered(self, adapter):
        assert (
            "POST",
            "/v1/responses/{response_id}/cancel",
            adapter._handle_cancel_response,
        ) in adapter._http_route_table()

    def test_cancel_parks_interrupt_reason_before_the_agent_exists(self, adapter):
        """The construction window must not swallow a cancel."""
        agent_ref = [None, None]
        entry = {"agent_ref": agent_ref, "task": None, "cancelled": False}

        assert adapter._interrupt_inflight_response(entry, "resp_early") is True
        assert agent_ref[1]

    @pytest.mark.asyncio
    async def test_parked_interrupt_fires_when_the_agent_appears(self, adapter):
        """_run_agent honors the parked reason before running a single tool."""
        mock_agent = _make_api_agent()
        mock_agent.interrupt.side_effect = RuntimeError("interrupt hook failed")
        agent_ref = [None, "Cancelled via /v1/responses/{id}/cancel"]

        with patch.object(adapter, "_create_agent", return_value=mock_agent):
            await adapter._run_agent(
                user_message="hello",
                conversation_history=[],
                session_id="session-parked-interrupt",
                agent_ref=agent_ref,
            )

        mock_agent.interrupt.assert_called_once_with(
            "Cancelled via /v1/responses/{id}/cancel"
        )
        mock_agent.run_conversation.assert_not_called()

    @pytest.mark.asyncio
    async def test_single_slot_agent_ref_is_untouched(self, adapter):
        """Every other _run_agent caller passes [None]; leave them alone."""
        mock_agent = _make_api_agent()
        agent_ref = [None]

        with patch.object(adapter, "_create_agent", return_value=mock_agent):
            await adapter._run_agent(
                user_message="hello",
                conversation_history=[],
                session_id="session-single-slot",
                agent_ref=agent_ref,
            )

        mock_agent.interrupt.assert_not_called()

    @pytest.mark.asyncio
    async def test_cancel_halts_tool_execution_on_the_executor_thread(self, adapter):
        """The stop must reach the executor thread, not just the stream.

        ``_run_agent`` runs the agent on a thread pool, so cancelling its
        asyncio wrapper detaches the stream and leaves the tool loop running —
        the exact orphan this route exists to end.  Drive the real executor
        path with an agent that keeps calling tools until something interrupts
        it, and assert the loop stops advancing once the HTTP cancel lands.
        """
        interrupted = threading.Event()
        started = threading.Event()
        tool_calls: list[int] = []

        class _LoopingAgent:
            """Calls a tool every tick until interrupted, like a real loop."""

            session_prompt_tokens = 0
            session_completion_tokens = 0
            session_total_tokens = 0
            session_id = "session-cancel-tools"
            _session_messages: list = []

            def interrupt(self, reason=None):
                interrupted.set()

            def run_conversation(self, **kwargs):
                started.set()
                deadline = time.monotonic() + 5
                while not interrupted.is_set() and time.monotonic() < deadline:
                    tool_calls.append(len(tool_calls))
                    time.sleep(0.01)
                return {"final_response": "", "messages": [], "api_calls": len(tool_calls)}

            def shutdown_memory_provider(self):
                pass

            def close(self):
                pass

        agent_ref = [None, None]
        stream_q = ThreadSafeAsyncQueue()
        response_id = f"resp_{uuid.uuid4().hex[:28]}"

        with patch.object(adapter, "_create_agent", return_value=_LoopingAgent()):
            agent_task = asyncio.ensure_future(
                adapter._run_agent(
                    user_message="run tools until stopped",
                    conversation_history=[],
                    session_id="session-cancel-tools",
                    agent_ref=agent_ref,
                )
            )
            writer, patcher, written = self._start_stream(
                adapter,
                agent_ref=agent_ref,
                stream_q=stream_q,
                agent_task=agent_task,
                response_id=response_id,
            )
            try:
                app = _create_app(adapter)
                async with TestClient(TestServer(app)) as cli:
                    for _ in range(200):
                        await asyncio.sleep(0.01)
                        if started.is_set() and tool_calls:
                            break
                    assert tool_calls, "agent never reached its tool loop"

                    resp = await cli.post(f"/v1/responses/{response_id}/cancel")
                    assert resp.status == 200
                    assert (await resp.json())["status"] == "cancelled"

                    assert interrupted.wait(timeout=2), "interrupt never reached the thread"
                    calls_at_cancel = len(tool_calls)
                    # Long enough for ~25 more ticks had the loop survived.
                    await asyncio.sleep(0.25)
                    assert len(tool_calls) <= calls_at_cancel + 1

                    with pytest.raises(asyncio.CancelledError):
                        await writer
            finally:
                interrupted.set()
                patcher.stop()
                adapter._inflight_responses.pop(response_id, None)

    @pytest.mark.asyncio
    async def test_writer_crash_interrupts_agent_before_recording_incomplete(self, adapter):
        """``incomplete`` is this route's already-terminal answer, so it must
        never outlive a running agent.

        The writer's generic-exception unwind (a failed SSE write that is not
        an ``OSError``, a serialization bug) abandons the turn without the
        client having disconnected — nothing else will stop the executor
        thread afterwards.  Left uninterrupted, the record reads ``incomplete``
        while tools keep running and a later cancel answers 409 for a run
        nothing contained.
        """
        import gateway.platforms.api_server as api_mod

        class _ExplodingStreamResponse:
            async def prepare(self, req):
                pass

            async def write(self, payload):
                raise RuntimeError("transport is closing")

        agent = _DualInterruptAgent()
        agent_ref = [agent, None]
        stream_q = ThreadSafeAsyncQueue()

        async def _agent_coro():
            await asyncio.sleep(30)
            return ({}, {})

        agent_task = asyncio.ensure_future(_agent_coro())
        agent_task.add_done_callback(lambda _fut: stream_q.put_nowait(None))
        response_id = f"resp_{uuid.uuid4().hex[:28]}"
        fake_request = MagicMock()
        fake_request.headers = {}

        with patch.object(
            api_mod.web, "StreamResponse", return_value=_ExplodingStreamResponse()
        ):
            await adapter._write_sse_responses(
                request=fake_request,
                response_id=response_id,
                model="hermes-agent",
                created_at=int(time.time()),
                stream_q=stream_q,
                agent_task=agent_task,
                agent_ref=agent_ref,
                conversation_history=[],
                user_message="long running turn",
                instructions=None,
                conversation=None,
                store=True,
                session_id="session-writer-crash",
            )

        assert adapter._response_store.get(response_id)["response"]["status"] == "incomplete"
        agent.hard_interrupt.assert_called_once_with("SSE writer failed mid-stream")
        agent.interrupt.assert_not_called()
        with pytest.raises(asyncio.CancelledError):
            await agent_task
        assert response_id not in adapter._inflight_responses

        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.post(f"/v1/responses/{response_id}/cancel")
            assert resp.status == 409

    @pytest.mark.asyncio
    async def test_prepare_failure_hard_stops_the_agent_task(self, adapter):
        """Transport setup can fail after the executor has already started."""
        import gateway.platforms.api_server as api_mod

        class _PrepareFailure:
            async def prepare(self, req):
                raise RuntimeError("could not prepare stream")

            async def write(self, payload):
                raise AssertionError("an unprepared stream must not be written")

        agent = _DualInterruptAgent()
        agent_ref = [agent, None]
        stream_q = ThreadSafeAsyncQueue()

        async def _agent_coro():
            await asyncio.sleep(30)

        agent_task = asyncio.create_task(_agent_coro())
        fake_request = MagicMock()
        fake_request.headers = {}
        try:
            with (
                patch.object(
                    api_mod.web, "StreamResponse", return_value=_PrepareFailure()
                ),
                pytest.raises(RuntimeError, match="could not prepare stream"),
            ):
                await adapter._write_sse_responses(
                    request=fake_request,
                    response_id=f"resp_{uuid.uuid4().hex[:28]}",
                    model="hermes-agent",
                    created_at=int(time.time()),
                    stream_q=stream_q,
                    agent_task=agent_task,
                    agent_ref=agent_ref,
                    conversation_history=[],
                    user_message="long running turn",
                    instructions=None,
                    conversation=None,
                    store=True,
                    session_id="session-prepare-failure",
                )

            agent.hard_interrupt.assert_called_once_with(
                "SSE response setup failed"
            )
            agent.interrupt.assert_not_called()
            with pytest.raises(asyncio.CancelledError):
                await agent_task
        finally:
            if not agent_task.done():
                agent_task.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await agent_task

    @pytest.mark.asyncio
    async def test_store_failure_cannot_skip_agent_containment(self, adapter):
        """Durable-store outages are best-effort only after hard containment."""
        import gateway.platforms.api_server as api_mod

        class _AlwaysFailingStore:
            def claim(self, *args, **kwargs):
                raise RuntimeError("response store unavailable")

            def transition(self, *args, **kwargs):
                raise RuntimeError("response store unavailable")

        class _PreparedStream:
            async def prepare(self, req):
                pass

            async def write(self, payload):
                pass

        agent = _DualInterruptAgent()
        agent_ref = [agent, None]
        stream_q = ThreadSafeAsyncQueue()

        async def _agent_coro():
            await asyncio.sleep(30)

        agent_task = asyncio.create_task(_agent_coro())
        fake_request = MagicMock()
        fake_request.headers = {}
        adapter._response_store = _AlwaysFailingStore()
        try:
            with patch.object(
                api_mod.web, "StreamResponse", return_value=_PreparedStream()
            ):
                await adapter._write_sse_responses(
                    request=fake_request,
                    response_id=f"resp_{uuid.uuid4().hex[:28]}",
                    model="hermes-agent",
                    created_at=int(time.time()),
                    stream_q=stream_q,
                    agent_task=agent_task,
                    agent_ref=agent_ref,
                    conversation_history=[],
                    user_message="long running turn",
                    instructions=None,
                    conversation=None,
                    store=True,
                    session_id="session-store-failure",
                )

            agent.hard_interrupt.assert_called_once_with(
                "SSE writer failed mid-stream"
            )
            agent.interrupt.assert_not_called()
            with pytest.raises(asyncio.CancelledError):
                await agent_task
        finally:
            if not agent_task.done():
                agent_task.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await agent_task


# ---------------------------------------------------------------------------
# DELETE /v1/responses/{response_id}
# ---------------------------------------------------------------------------


class TestDeleteResponse:
    @pytest.mark.asyncio
    async def test_delete_active_response_requires_cancellation_first(self, adapter):
        response_id = "resp_active_delete"
        assert adapter._response_store.claim(
            response_id,
            {
                "response": {
                    "id": response_id,
                    "object": "response",
                    "status": "in_progress",
                },
                "conversation_history": [],
            },
            owner_id="owner-a",
            owner_epoch="epoch-a",
        )

        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            response = await cli.delete(f"/v1/responses/{response_id}")
            assert response.status == 409
            assert (await response.json())["error"]["code"] == "response_not_terminal"

        assert adapter._response_store.get(response_id) is not None

    @pytest.mark.asyncio
    async def test_delete_stored_response(self, adapter):
        """DELETE removes a stored response and returns confirmation."""
        mock_result = {"final_response": "Hello!", "messages": [], "api_calls": 1}

        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_run_agent", new_callable=AsyncMock) as mock_run:
                mock_run.return_value = (mock_result, {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0})
                resp = await cli.post(
                    "/v1/responses",
                    json={
                        "model": "hermes-agent",
                        "input": "Hi",
                        "conversation": "delete-test",
                    },
                )

            data = await resp.json()
            response_id = data["id"]
            assert adapter._response_store.get_conversation("delete-test") == response_id

            # Delete it
            resp2 = await cli.delete(f"/v1/responses/{response_id}")
            assert resp2.status == 200
            data2 = await resp2.json()
            assert data2["id"] == response_id
            assert data2["object"] == "response"
            assert data2["deleted"] is True

            # Verify it's gone
            resp3 = await cli.get(f"/v1/responses/{response_id}")
            assert resp3.status == 404
            assert adapter._response_store.get_conversation("delete-test") is None


# ---------------------------------------------------------------------------
# Tool calls in output
# ---------------------------------------------------------------------------


class TestToolCallsInOutput:
    @pytest.mark.asyncio
    async def test_tool_calls_in_output(self, adapter):
        """When agent returns tool calls, they appear as function_call items."""
        mock_result = {
            "final_response": "The result is 42.",
            "messages": [
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_abc123",
                            "function": {
                                "name": "calculator",
                                "arguments": '{"expression": "6*7"}',
                            },
                        }
                    ],
                },
                {
                    "role": "tool",
                    "tool_call_id": "call_abc123",
                    "content": "42",
                },
                {
                    "role": "assistant",
                    "content": "The result is 42.",
                },
            ],
            "api_calls": 2,
        }

        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_run_agent", new_callable=AsyncMock) as mock_run:
                mock_run.return_value = (mock_result, {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0})
                resp = await cli.post(
                    "/v1/responses",
                    json={"model": "hermes-agent", "input": "What is 6*7?"},
                )

            assert resp.status == 200
            data = await resp.json()
            output = data["output"]

            # Should have: function_call, function_call_output, message
            assert len(output) == 3
            assert output[0]["type"] == "function_call"
            assert output[0]["name"] == "calculator"
            assert output[0]["arguments"] == '{"expression": "6*7"}'
            assert output[0]["call_id"] == "call_abc123"
            # Replayed server-executed calls must be marked completed so
            # OpenAI clients don't treat them as pending calls to execute.
            assert output[0]["status"] == "completed"
            assert output[0]["id"].startswith("fc_")
            assert output[1]["type"] == "function_call_output"
            assert output[1]["call_id"] == "call_abc123"
            assert output[1]["output"] == "42"
            assert output[1]["status"] == "completed"
            assert output[1]["id"].startswith("fco_")
            assert output[2]["type"] == "message"
            assert output[2]["content"][0]["text"] == "The result is 42."


# ---------------------------------------------------------------------------
# Usage / token counting
# ---------------------------------------------------------------------------


class TestUsageCounting:
    @pytest.mark.asyncio
    async def test_responses_usage(self, adapter):
        """Responses API returns real token counts."""
        mock_result = {"final_response": "Done", "messages": [], "api_calls": 1}
        usage = {"input_tokens": 100, "output_tokens": 50, "total_tokens": 150}

        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_run_agent", new_callable=AsyncMock) as mock_run:
                mock_run.return_value = (mock_result, usage)
                resp = await cli.post(
                    "/v1/responses",
                    json={"model": "hermes-agent", "input": "Hi"},
                )

            assert resp.status == 200
            data = await resp.json()
            assert data["usage"]["input_tokens"] == 100
            assert data["usage"]["output_tokens"] == 50
            assert data["usage"]["total_tokens"] == 150


# ---------------------------------------------------------------------------
# Truncation
# ---------------------------------------------------------------------------


class TestTruncation:


    @pytest.mark.asyncio
    async def test_truncation_auto_preserves_non_leading_compaction_summary(self, adapter):
        """A summary sitting after a retained system head must survive too.

        The gateway /compress path can force a user-leading layout that
        leaves the compaction summary after a kept system message, so the
        preservation predicate must not assume the summary is at index 0.
        """
        mock_result = {"final_response": "OK", "messages": [], "api_calls": 1}

        system_head = {"role": "system", "content": "You are a helpful agent."}
        summary = {
            "role": "user",
            "content": "[CONTEXT COMPACTION — REFERENCE ONLY]\nEarlier work.",
            "_compressed_summary": True,
        }
        long_history = [system_head, summary] + [
            {"role": "user", "content": f"msg {i}"}
            for i in range(148)
        ]
        adapter._response_store.put("resp_summary_mid", {
            "response": {"id": "resp_summary_mid", "object": "response"},
            "conversation_history": long_history,
            "instructions": None,
        })

        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_run_agent", new_callable=AsyncMock) as mock_run:
                mock_run.return_value = (mock_result, {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0})
                resp = await cli.post(
                    "/v1/responses",
                    json={
                        "model": "hermes-agent",
                        "input": "follow up",
                        "previous_response_id": "resp_summary_mid",
                        "truncation": "auto",
                    },
                )

        assert resp.status == 200
        history = mock_run.call_args.kwargs["conversation_history"]
        assert len(history) == 100
        assert history[0] == summary
        assert history[1]["content"] == "msg 49"
        assert history[-1]["content"] == "msg 147"


# ---------------------------------------------------------------------------
# Response-side truncation / failure handling (issue #22496)
# ---------------------------------------------------------------------------


class TestChatCompletionsAgentIncomplete:
    """When the agent run yields a partial / failed result, the API server
    must NOT pretend it succeeded. Either signal truncation via
    finish_reason='length' (with the partial text), or 502 with an OpenAI
    error envelope (no usable text). Issue #22496."""


    @pytest.mark.asyncio
    async def test_hard_failure_redacts_secret_like_error_text(self, adapter):
        raw_secret = "sk-api-server-leak-1234567890"
        mock_result = {
            "final_response": "",
            "completed": False,
            "partial": False,
            "failed": True,
            "error": f"provider auth failed OPENAI_API_KEY={raw_secret}",
            "messages": [],
            "api_calls": 1,
        }
        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_run_agent", new_callable=AsyncMock) as mock_run:
                mock_run.return_value = (mock_result, {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0})
                resp = await cli.post(
                    "/v1/chat/completions",
                    json={"model": "hermes-agent", "messages": [{"role": "user", "content": "hello"}]},
                )

            assert resp.status == 502
            data = await resp.json()
            body = json.dumps(data)
            assert raw_secret not in body
            assert raw_secret not in resp.headers.get("X-Hermes-Error", "")
            assert "OPENAI_API_KEY=" in body
            assert data["error"]["hermes"]["failed"] is True


# ---------------------------------------------------------------------------
# CORS
# ---------------------------------------------------------------------------


class TestCORS:
    def test_origin_allowed_for_non_browser_client(self, adapter):
        assert adapter._origin_allowed("") is True


    def test_origin_allowed_for_allowlist_match(self):
        adapter = _make_adapter(cors_origins=["http://localhost:3000"])
        assert adapter._origin_allowed("http://localhost:3000") is True


    @pytest.mark.asyncio
    async def test_browser_origin_rejected_by_default(self, adapter):
        """Browser-originated requests are rejected unless explicitly allowed."""
        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.get("/health", headers={"Origin": "http://evil.example"})
            assert resp.status == 403
            assert resp.headers.get("Access-Control-Allow-Origin") is None


    @pytest.mark.asyncio
    async def test_cors_allows_idempotency_key_header(self):
        adapter = _make_adapter(cors_origins=["http://localhost:3000"])
        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.options(
                "/v1/chat/completions",
                headers={
                    "Origin": "http://localhost:3000",
                    "Access-Control-Request-Method": "POST",
                    "Access-Control-Request-Headers": "Idempotency-Key",
                },
            )
            assert resp.status == 200
            assert "Idempotency-Key" in resp.headers.get("Access-Control-Allow-Headers", "")


    @pytest.mark.asyncio
    async def test_cors_options_preflight_allowed_for_configured_origin(self):
        """Configured origins can complete browser preflight."""
        adapter = _make_adapter(cors_origins=["http://localhost:3000"])
        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.options(
                "/v1/chat/completions",
                headers={
                    "Origin": "http://localhost:3000",
                    "Access-Control-Request-Method": "POST",
                    "Access-Control-Request-Headers": "Authorization, Content-Type",
                },
            )
            assert resp.status == 200
            assert resp.headers.get("Access-Control-Allow-Origin") == "http://localhost:3000"
            assert "Authorization" in resp.headers.get("Access-Control-Allow-Headers", "")


# ---------------------------------------------------------------------------
# Conversation parameter
# ---------------------------------------------------------------------------


class TestConversationParameter:


    @pytest.mark.asyncio
    async def test_separate_conversations_are_isolated(self, adapter):
        """Different conversation names have independent histories."""
        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_run_agent", new_callable=AsyncMock) as mock_run:
                mock_run.return_value = (
                    {"final_response": "Response A", "messages": [], "api_calls": 1},
                    {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
                )
                # Conversation A
                await cli.post("/v1/responses", json={"input": "conv-a msg", "conversation": "conv-a"})
                # Conversation B
                mock_run.return_value = (
                    {"final_response": "Response B", "messages": [], "api_calls": 1},
                    {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
                )
                await cli.post("/v1/responses", json={"input": "conv-b msg", "conversation": "conv-b"})

                # They should have different response IDs in the mapping
                assert adapter._response_store.get_conversation("conv-a") != adapter._response_store.get_conversation("conv-b")


    @pytest.mark.asyncio
    async def test_conversation_reuse_after_eviction_no_404(self, adapter):
        """After eviction clears a conversation mapping, reusing that name starts fresh (no 404)."""
        adapter._response_store = ResponseStore(max_size=1)
        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_run_agent", new_callable=AsyncMock) as mock_run:
                mock_run.return_value = (
                    {"final_response": "First", "messages": [], "api_calls": 1},
                    {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
                )
                # Create conversation -> resp stored
                resp1 = await cli.post("/v1/responses", json={
                    "input": "hello",
                    "conversation": "my-chat",
                })
                assert resp1.status == 200

                # Evict by adding another response
                mock_run.return_value = (
                    {"final_response": "Other", "messages": [], "api_calls": 1},
                    {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
                )
                await cli.post("/v1/responses", json={"input": "other"})

                # Conversation mapping should have been cleaned by eviction
                assert adapter._response_store.get_conversation("my-chat") is None

                # Reuse conversation name — should start fresh, not 404
                mock_run.return_value = (
                    {"final_response": "Restarted", "messages": [], "api_calls": 1},
                    {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
                )
                resp3 = await cli.post("/v1/responses", json={
                    "input": "hello again",
                    "conversation": "my-chat",
                })
                assert resp3.status == 200


# ---------------------------------------------------------------------------
# X-Hermes-Session-Id header (session continuity)
# ---------------------------------------------------------------------------


class TestSessionIdHeader:


    @pytest.mark.asyncio
    async def test_traversal_session_id_header_rejected(self, auth_adapter):
        """Security (#5958): a path-traversal X-Hermes-Session-Id must be
        rejected with 400 so it can't reach the filesystem artifact paths
        (session snapshot / request dump) and escape the sessions dir."""
        app = _create_app(auth_adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch.object(auth_adapter, "_run_agent", new_callable=AsyncMock) as mock_run:
                for bad in ("../../../../etc/pwned", "/abs/path", "..\\win"):
                    resp = await cli.post(
                        "/v1/chat/completions",
                        headers={"X-Hermes-Session-Id": bad, "Authorization": "Bearer sk-secret"},
                        json={"model": "hermes-agent", "messages": [{"role": "user", "content": "hi"}]},
                    )
                    assert resp.status == 400, f"{bad!r} should be rejected"
                # The agent is never invoked for a rejected ID.
                assert mock_run.call_count == 0

    @pytest.mark.asyncio
    async def test_provided_session_id_loads_history_from_db(self, auth_adapter):
        """When X-Hermes-Session-Id is provided, history comes from SessionDB not request body."""
        mock_result = {"final_response": "OK", "messages": [], "api_calls": 1}
        db_history = [
            {"role": "user", "content": "stored message 1"},
            {"role": "assistant", "content": "stored reply 1"},
        ]
        mock_db = MagicMock()
        mock_db.get_messages_as_conversation.return_value = db_history
        auth_adapter._session_db = mock_db
        app = _create_app(auth_adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch.object(auth_adapter, "_run_agent", new_callable=AsyncMock) as mock_run:
                mock_run.return_value = (mock_result, {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0})

                resp = await cli.post(
                    "/v1/chat/completions",
                    headers={"X-Hermes-Session-Id": "existing-session", "Authorization": "Bearer sk-secret"},
                    # Request body has different history — should be ignored
                    json={
                        "model": "hermes-agent",
                        "messages": [
                            {"role": "user", "content": "old msg from client"},
                            {"role": "assistant", "content": "old reply from client"},
                            {"role": "user", "content": "new question"},
                        ],
                    },
                )

            assert resp.status == 200
            call_kwargs = mock_run.call_args.kwargs
            # History must come from DB, not from the request body
            assert call_kwargs["conversation_history"] == db_history
            assert call_kwargs["user_message"] == "new question"


# ---------------------------------------------------------------------------
# X-Hermes-Session-Key header (long-term memory scoping)
# ---------------------------------------------------------------------------


class TestSessionKeyHeader:
    """The session key is a stable per-channel identifier that scopes
    long-term memory (e.g. Honcho) independently of the transcript-scoped
    session_id.  A third-party Web UI passes one stable key per assistant
    channel and rotates session_id on /new, matching the native
    gateway's session_key / session_id split.
    """


    @pytest.mark.asyncio
    async def test_session_key_threads_into_create_agent(self, auth_adapter):
        """End-to-end: verify AIAgent(gateway_session_key=...) receives the key via _create_agent."""
        captured_kwargs = {}

        def _fake_create_agent(**kwargs):
            captured_kwargs.update(kwargs)
            mock_agent = MagicMock()
            mock_agent.run_conversation.return_value = {"final_response": "ok", "messages": []}
            mock_agent.session_prompt_tokens = 0
            mock_agent.session_completion_tokens = 0
            mock_agent.session_total_tokens = 0
            return mock_agent

        app = _create_app(auth_adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch.object(auth_adapter, "_create_agent", side_effect=_fake_create_agent):
                resp = await cli.post(
                    "/v1/chat/completions",
                    headers={
                        "X-Hermes-Session-Key": "agent:main:webui:dm:user-7",
                        "Authorization": "Bearer sk-secret",
                    },
                    json={"model": "hermes-agent", "messages": [{"role": "user", "content": "hi"}]},
                )
            assert resp.status == 200
            # _create_agent must be called with gateway_session_key threaded through
            assert captured_kwargs.get("gateway_session_key") == "agent:main:webui:dm:user-7"

    @pytest.mark.asyncio
    async def test_responses_endpoint_accepts_session_key(self, auth_adapter):
        """Responses API honors the same X-Hermes-Session-Key contract."""
        mock_result = {"final_response": "ok", "messages": [], "api_calls": 1}
        app = _create_app(auth_adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch.object(auth_adapter, "_run_agent", new_callable=AsyncMock) as mock_run:
                mock_run.return_value = (mock_result, {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0})
                resp = await cli.post(
                    "/v1/responses",
                    headers={
                        "X-Hermes-Session-Key": "webui:chan-1",
                        "Authorization": "Bearer sk-secret",
                    },
                    json={"model": "hermes-agent", "input": "hello", "store": False},
                )
            assert resp.status == 200
            assert resp.headers.get("X-Hermes-Session-Key") == "webui:chan-1"
            call_kwargs = mock_run.call_args.kwargs
            assert call_kwargs["gateway_session_key"] == "webui:chan-1"

    @pytest.mark.asyncio
    async def test_capabilities_advertises_session_key_header(self, adapter):
        """GET /v1/capabilities should advertise the new header so clients can feature-detect."""
        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.get("/v1/capabilities")
            assert resp.status == 200
            data = await resp.json()
            assert data["features"]["session_key_header"] == "X-Hermes-Session-Key"


# ---------------------------------------------------------------------------
# Per-client model routing (model_routes)
# ---------------------------------------------------------------------------


def _make_routing_adapter(routes) -> APIServerAdapter:
    """Create an adapter with model_routes configured."""
    config = PlatformConfig(enabled=True, extra={"model_routes": routes})
    return APIServerAdapter(config)


def _patch_create_agent_runtime(monkeypatch, captured: dict, fake_agent_cls):
    """Stub out every external dependency of _create_agent."""
    monkeypatch.setattr("run_agent.AIAgent", fake_agent_cls)
    monkeypatch.setattr(
        "gateway.run._resolve_runtime_agent_kwargs",
        lambda: {
            "provider": "openrouter",
            "api_key": "sk-global",
            "base_url": "https://openrouter.ai/api/v1",
            "api_mode": "chat_completions",
        },
    )
    monkeypatch.setattr("gateway.run._resolve_gateway_model", lambda: "global/model")
    monkeypatch.setattr("gateway.run._load_gateway_config", lambda: {})
    monkeypatch.setattr(
        "gateway.run.GatewayRunner._load_reasoning_config", staticmethod(lambda model="": {})
    )
    monkeypatch.setattr(
        "gateway.run.GatewayRunner._load_fallback_model", staticmethod(lambda: None)
    )
    monkeypatch.setattr("gateway.run._current_max_iterations", lambda: 90)
    monkeypatch.setattr("hermes_cli.tools_config._get_platform_tools", lambda *_: set())


class TestModelRoutesParsing:
    def test_valid_routes_are_parsed(self):
        routes = {"minimax-m2": {"model": "minimax/minimax-m1", "provider": "openrouter"}}
        adapter = _make_routing_adapter(routes)
        assert adapter._model_routes == routes


    def test_route_without_model_is_dropped(self):
        adapter = _make_routing_adapter({"bad": {"provider": "openrouter"}})
        assert adapter._model_routes == {}


class TestModelRoutesModelsEndpoint:

    @pytest.mark.asyncio
    async def test_models_endpoint_route_alias_fields_and_no_secrets(self):
        routes = {"my-alias": {"model": "openai/gpt-5", "api_key": "sk-route-secret"}}
        adapter = _make_routing_adapter(routes)
        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.get("/v1/models")
            data = await resp.json()
            alias_entry = next(m for m in data["data"] if m["id"] == "my-alias")
            assert alias_entry["root"] == "openai/gpt-5"
            assert alias_entry["parent"] == adapter._model_name
            # per-route api_key must never leak through the discovery endpoint
            assert "sk-route-secret" not in json.dumps(data)


class TestModelRoutesHandlers:
    @pytest.mark.asyncio
    async def test_chat_completions_passes_route_to_run_agent(self):
        routes = {"minimax-m2": {"model": "minimax/minimax-m1", "provider": "openrouter"}}
        adapter = _make_routing_adapter(routes)
        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_run_agent", new_callable=AsyncMock) as mock_run:
                mock_run.return_value = (
                    {"final_response": "hi", "messages": [], "api_calls": 1},
                    {"input_tokens": 5, "output_tokens": 5, "total_tokens": 10},
                )
                resp = await cli.post("/v1/chat/completions", json={
                    "model": "minimax-m2",
                    "messages": [{"role": "user", "content": "hello"}],
                })
                assert resp.status == 200
                kwargs = mock_run.call_args.kwargs
                assert kwargs.get("route") == {
                    "model": "minimax/minimax-m1", "provider": "openrouter",
                }


class TestModelRoutesAgentCreation:

    def test_route_provider_resolves_provider_credentials(self, monkeypatch):
        captured = {}

        class FakeAgent:
            def __init__(self, **kwargs):
                captured.update(kwargs)

        _patch_create_agent_runtime(monkeypatch, captured, FakeAgent)
        monkeypatch.setattr(
            "gateway.run._resolve_runtime_agent_kwargs_for_provider",
            lambda provider: {
                "provider": provider,
                "api_key": f"sk-{provider}",
                "base_url": f"https://{provider}.example/v1",
                "api_mode": "chat_completions",
            },
        )
        adapter = _make_routing_adapter(
            {"alias": {"model": "other/model", "provider": "otherprov"}}
        )
        monkeypatch.setattr(adapter, "_ensure_session_db", lambda: None)
        monkeypatch.setattr(adapter, "_session_model_override_for", lambda *_: None)

        adapter._create_agent(session_id="s1", route=adapter._resolve_route("alias"))

        assert captured["model"] == "other/model"
        assert captured["provider"] == "otherprov"
        assert captured["api_key"] == "sk-otherprov"


    def test_session_model_override_beats_route(self, monkeypatch):
        """A user-issued /model on the session must win over static route config."""
        captured = {}

        class FakeAgent:
            def __init__(self, **kwargs):
                captured.update(kwargs)

        _patch_create_agent_runtime(monkeypatch, captured, FakeAgent)
        adapter = _make_routing_adapter({"alias": {"model": "route/model", "api_key": "sk-route"}})
        monkeypatch.setattr(adapter, "_ensure_session_db", lambda: None)
        monkeypatch.setattr(
            adapter,
            "_session_model_override_for",
            lambda key: {
                "model": "session/override-model",
                "provider": "sessionprov",
                "api_key": "sk-session",
                "base_url": "https://session.example/v1",
                "api_mode": "responses",
                "credential_pool": "pool-session",
            },
        )

        adapter._create_agent(session_id="s1", route=adapter._resolve_route("alias"))

        assert captured["model"] == "session/override-model"
        assert captured["provider"] == "sessionprov"
        assert captured["api_key"] == "sk-session"


class TestStoredSessionModelFilter:
    """A session row that persisted the advertised virtual model must read as
    "no stored model" — replaying "hermes-agent" upstream 400s. Found live
    (Aug 2026): the first cross-gateway `hermes peer dm` against a fresh
    api_server failed every turn with "hermes-agent is not a valid model ID".
    """

    def test_virtual_model_is_filtered(self):
        adapter = _make_routing_adapter({})
        assert adapter._stored_session_model({"model": adapter._model_name}) is None

    def test_real_model_passes_through(self):
        adapter = _make_routing_adapter({})
        assert adapter._stored_session_model({"model": "google/gemini-3.7-flash"}) == "google/gemini-3.7-flash"

    def test_missing_or_bad_shapes(self):
        adapter = _make_routing_adapter({})
        assert adapter._stored_session_model({}) is None
        assert adapter._stored_session_model(None) is None


# ---------------------------------------------------------------------------
# Event-loop offloading for synchronous SessionDB calls (P1)
# ---------------------------------------------------------------------------


class TestSessionDbOffEventLoop:
    """Regression: synchronous SessionDB calls in the OpenAI-compatible API
    server must run OFF the aiohttp event loop. A blocking SQLite read/write on
    the loop freezes every in-flight request under load (same class of bug as
    gateway build_channel_directory, #60794 / #60810), so each call is wrapped
    in asyncio.to_thread.
    """

    @pytest.mark.asyncio
    async def test_get_existing_session_or_404_offloads(self, auth_adapter):
        import threading

        captured = {}

        class FakeDB:
            def get_session(self, session_id):
                captured["thread"] = threading.current_thread()
                return {"id": session_id, "source": "api_server"}

        auth_adapter._session_db = FakeDB()
        session, err = await auth_adapter._get_existing_session_or_404("sess-x")
        assert err is None
        assert session["id"] == "sess-x"
        # The blocking DB call must NOT execute on the event-loop thread.
        assert captured["thread"] is not None
        assert captured["thread"] != threading.current_thread()

    @pytest.mark.asyncio
    async def test_create_session_without_model_does_not_persist_virtual_alias(self, auth_adapter):
        """A session created with no ``model`` field must not persist the
        virtual model alias (self._model_name, e.g. "hermes-agent") as if it
        were a real provider model id.

        Regression: _handle_create_session previously did
        ``model = body.get("model") or self._model_name``, so an omitted
        model fell back to the virtual alias and that string got stored on
        the session row. _handle_session_chat later reads it back as a raw
        session_model override (since it's not a model_routes alias) and
        sends it to the provider literally — Bedrock/OpenAI then reject
        "hermes-agent" as an invalid model identifier on every turn.
        """
        app = _create_app(auth_adapter)
        app.router.add_post("/api/sessions", auth_adapter._handle_create_session)

        async with TestClient(TestServer(app)) as cli:
            resp = await cli.post(
                "/api/sessions",
                json={},
                headers={"Authorization": "Bearer sk-secret"},
            )
            assert resp.status == 201
            data = await resp.json()
            assert data["session"]["model"] != auth_adapter._model_name
            assert data["session"]["model"] is None

    @pytest.mark.asyncio
    async def test_create_session_with_explicit_virtual_alias_does_not_persist_it(self, auth_adapter):
        """Sending ``model: "hermes-agent"`` explicitly (the virtual alias
        itself, e.g. a client that just echoes /v1/models' advertised id)
        must be treated the same as omitting model entirely."""
        app = _create_app(auth_adapter)
        app.router.add_post("/api/sessions", auth_adapter._handle_create_session)

        async with TestClient(TestServer(app)) as cli:
            resp = await cli.post(
                "/api/sessions",
                json={"model": auth_adapter._model_name},
                headers={"Authorization": "Bearer sk-secret"},
            )
            assert resp.status == 201
            data = await resp.json()
            assert data["session"]["model"] is None

    @pytest.mark.asyncio
    async def test_create_session_with_real_model_persists_it(self, auth_adapter):
        """Regression guard: a genuine model id must still be stored as before."""
        app = _create_app(auth_adapter)
        app.router.add_post("/api/sessions", auth_adapter._handle_create_session)

        async with TestClient(TestServer(app)) as cli:
            resp = await cli.post(
                "/api/sessions",
                json={"model": "openai/gpt-5"},
                headers={"Authorization": "Bearer sk-secret"},
            )
            assert resp.status == 201
            data = await resp.json()
            assert data["session"]["model"] == "openai/gpt-5"

    @pytest.mark.asyncio
    async def test_create_session_with_provider_prefixed_virtual_alias_does_not_persist_it(self, auth_adapter):
        """A provider-prefixed echo of the virtual alias (e.g. a client that
        threads /v1/models' advertised id through a provider:: prefix) must
        also be treated as "no model", not stored as a raw override.

        Regression: _handle_create_session used to re-derive its own `model`
        straight from the raw request body, bypassing the provider-prefix
        split that _session_runtime_request_from_body performs — so
        "openrouter::hermes-agent" never matched self._model_name and leaked
        through as a literal session override.
        """
        app = _create_app(auth_adapter)
        app.router.add_post("/api/sessions", auth_adapter._handle_create_session)

        async with TestClient(TestServer(app)) as cli:
            resp = await cli.post(
                "/api/sessions",
                json={"model": f"openrouter::{auth_adapter._model_name}"},
                headers={"Authorization": "Bearer sk-secret"},
            )
            assert resp.status == 201
            data = await resp.json()
            assert data["session"]["model"] is None


# ---------------------------------------------------------------------------
# _api_key_passes_startup_guard — fail-closed on an unverifiable key
# ---------------------------------------------------------------------------

class TestApiKeyStartupGuardFailsClosed:
    """The guard is the only thing between a guessable key and an endpoint the
    code itself describes as ``terminal-capable agent work`` where "a guessable
    key is remote code execution".

    So "the strength check could not be run" must never resolve to "start
    anyway" — the same posture ``tools/credential_files.py`` takes when its
    deny-list cannot be consulted.
    """

    class _Stub:
        name = "api_server"
        _host = "0.0.0.0"

        def __init__(self, key):
            self._api_key = key

    @staticmethod
    def _guard(key):
        return APIServerAdapter._api_key_passes_startup_guard(
            TestApiKeyStartupGuardFailsClosed._Stub(key)
        )

    @staticmethod
    def _blocking_auth_import():
        real_import = __import__

        def _blocked(name, *args, **kwargs):
            if name == "hermes_cli.auth":
                raise ImportError("simulated: hermes_cli.auth unavailable")
            return real_import(name, *args, **kwargs)

        return patch("builtins.__import__", _blocked)

    def test_weak_key_refused_when_check_is_unavailable(self):
        """The bug: an unimportable auth module silently dropped the check and
        the server started on a 4-character key."""
        with self._blocking_auth_import():
            assert self._guard("test") is False

    def test_strong_key_also_refused_when_check_is_unavailable(self):
        """Fail-closed: we cannot verify the key, so we do not expose the
        endpoint — the log tells the operator to repair the install."""
        with self._blocking_auth_import():
            assert self._guard("a" * 40) is False


class TestKeyRejectionSetsNonRetryableFatalError:
    """Each startup-guard rejection must set a non-retryable fatal error so
    the reconnect watcher drops the platform from the retry queue instead of
    looping indefinitely.

    Previously connect() returned bare ``False``, which gateway.run treated
    as retryable — re-queueing every backoff interval forever and
    re-instantiating the adapter (with its ResponseStore sqlite connection)
    each retry (#38803: ~501 leaked connections / 1002 fds over 2.5 days,
    ending in EMFILE for the whole gateway). Mirrors the port-conflict
    precedent (test_port_conflict_sets_non_retryable_fatal_error, #65665).
    """

    @staticmethod
    def _make_adapter(key, monkeypatch):
        monkeypatch.delenv("API_SERVER_KEY", raising=False)
        return APIServerAdapter(
            PlatformConfig(
                enabled=True,
                extra={"host": "127.0.0.1", "port": 0, "key": key},
            )
        )

    @staticmethod
    async def _assert_key_rejection_is_fatal(adapter):
        try:
            assert await adapter.connect() is False
            assert adapter.has_fatal_error is True
            assert adapter.fatal_error_retryable is False
            assert adapter.fatal_error_code == "api_server_key_invalid"
            assert "API_SERVER_KEY" in (adapter.fatal_error_message or "")
        finally:
            await adapter.disconnect()

    @pytest.mark.asyncio
    async def test_missing_key_sets_non_retryable_fatal_error(self, monkeypatch):
        adapter = self._make_adapter("", monkeypatch)
        await self._assert_key_rejection_is_fatal(adapter)


# ---------------------------------------------------------------------------
# Bare-model opt-in gate (direct_model_requests) for _request_agent_overrides
# ---------------------------------------------------------------------------


class TestDirectModelRequestsGate:
    """Bare ``model`` (no ``provider``) is opt-in on OpenAI-compatible
    endpoints so generic clients hardcoding "gpt-4o" keep falling back to
    the gateway default (idea credit: PR #22825 by @mssteuer)."""

    def test_bare_model_dropped_when_disallowed(self):
        overrides = _request_agent_overrides(
            {"model": "openai/gpt-5"}, allow_bare_model=False
        )
        assert "requested_model" not in overrides


    def test_adapter_flag_opt_in(self):
        adapter = APIServerAdapter(
            PlatformConfig(enabled=True, extra={"direct_model_requests": True})
        )
        assert adapter._direct_model_requests is True


    @pytest.mark.asyncio
    async def test_chat_completions_bare_model_honored_when_enabled(self):
        adapter = APIServerAdapter(
            PlatformConfig(enabled=True, extra={"direct_model_requests": True})
        )
        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_run_agent", new_callable=AsyncMock) as mock_run:
                mock_run.return_value = (
                    {"final_response": "ok", "messages": [], "api_calls": 1},
                    {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
                )
                resp = await cli.post(
                    "/v1/chat/completions",
                    json={
                        "model": "openai/gpt-5",
                        "messages": [{"role": "user", "content": "hi"}],
                    },
                )
        assert resp.status == 200
        assert mock_run.call_args.kwargs.get("requested_model") == "openai/gpt-5"


class TestRouteWithoutModelKeepsDefault:
    """A model_routes alias whose route has no ``model`` key must keep the
    global default model — the alias string itself is never a model name."""

    def test_alias_never_leaks_as_model(self, monkeypatch):
        captured = {}

        class FakeAgent:
            def __init__(self, **kwargs):
                captured.update(kwargs)

        _patch_create_agent_runtime(monkeypatch, captured, FakeAgent)
        adapter = _make_routing_adapter(
            {"alias": {"model": "", "api_key": "sk-route"}}
        )
        # _parse_model_routes drops routes without model; simulate a
        # credentials-only route surviving via direct dict (defensive path).
        adapter._model_routes = {"alias": {"api_key": "sk-route"}}
        monkeypatch.setattr(adapter, "_ensure_session_db", lambda: None)
        monkeypatch.setattr(adapter, "_session_model_override_for", lambda *_: None)

        adapter._create_agent(
            session_id="s1",
            route=adapter._resolve_route("alias"),
            requested_model="alias",
        )

        assert captured["model"] == "global/model"
        assert captured["api_key"] == "sk-route"


# ---------------------------------------------------------------------------
# Empty-model recovery + provider-auth error typing in _create_agent
# (salvaged from PR #57947 by @FvanW)
# ---------------------------------------------------------------------------


class TestCreateAgentModelRecovery:
    def test_create_agent_defaults_to_provider_catalog_model_when_empty(self, monkeypatch):
        """api_server.py had no equivalent of run.py's provider-catalog
        default when model resolves empty but a provider did resolve (e.g.
        `hermes auth add openai-codex` without `hermes model`) —
        AIAgent(model="") 400s every call."""
        captured = {}

        class FakeAgent:
            def __init__(self, **kwargs):
                captured.update(kwargs)

        _patch_create_agent_runtime(monkeypatch, captured, FakeAgent)
        monkeypatch.setattr(
            "gateway.run._resolve_runtime_agent_kwargs",
            lambda: {"provider": "openai-codex", "base_url": "https://example.test/v1",
                     "api_mode": "codex_responses"},
        )
        monkeypatch.setattr("gateway.run._resolve_gateway_model", lambda: "")
        monkeypatch.setattr(
            "hermes_cli.models.get_default_model_for_provider",
            lambda provider: "gpt-5.5-codex" if provider == "openai-codex" else None,
        )

        adapter = APIServerAdapter(PlatformConfig(enabled=True))
        monkeypatch.setattr(adapter, "_ensure_session_db", lambda: None)

        agent = adapter._create_agent(session_id="api-session")

        assert isinstance(agent, FakeAgent)
        assert captured["model"] == "gpt-5.5-codex"

    def test_create_agent_recovers_last_known_good_model_when_empty(self, monkeypatch):
        """Last-known-good recovery (#35314): a transient config-cache miss
        producing an empty model would build AIAgent(model="") and fail every
        call until manual retry, instead of reusing the model that just
        worked."""
        captured = []

        class FakeAgent:
            def __init__(self, **kwargs):
                captured.append(dict(kwargs))

        _patch_create_agent_runtime(monkeypatch, {}, FakeAgent)
        monkeypatch.setattr("run_agent.AIAgent", FakeAgent)

        adapter = APIServerAdapter(PlatformConfig(enabled=True))
        monkeypatch.setattr(adapter, "_ensure_session_db", lambda: None)

        # Turn 1: model resolves fine — populates the last-known-good cache
        # (keyed on gateway_session_key).
        monkeypatch.setattr("gateway.run._resolve_gateway_model", lambda: "minimax/minimax-m3")
        adapter._create_agent(session_id="api-session", gateway_session_key="stable-chan-1")
        assert captured[0]["model"] == "minimax/minimax-m3"
        assert adapter._last_resolved_model["stable-chan-1"] == "minimax/minimax-m3"

        # Turn 2: transient empty resolution, no provider catalog default —
        # must recover the model from turn 1, not build model="".
        monkeypatch.setattr("gateway.run._resolve_gateway_model", lambda: "")
        monkeypatch.setattr(
            "gateway.run._resolve_runtime_agent_kwargs",
            lambda: {"provider": None, "base_url": None, "api_mode": None},
        )
        adapter._create_agent(session_id="another-session", gateway_session_key="stable-chan-1")
        assert captured[1]["model"] == "minimax/minimax-m3"

    # ── Recovery-net alias guards (PR for #79101) ──────────────────────

    def test_create_agent_does_not_cache_virtual_alias(self, monkeypatch):
        """Write-side guard: the advertised virtual model (``hermes-agent``)
        must never enter ``_last_resolved_model``, even when a prior turn
        (or the session-row bug) dispatched it."""
        captured = []

        class FakeAgent:
            def __init__(self, **kwargs):
                captured.append(dict(kwargs))

        _patch_create_agent_runtime(monkeypatch, {}, FakeAgent)

        adapter = APIServerAdapter(PlatformConfig(enabled=True))
        monkeypatch.setattr(adapter, "_ensure_session_db", lambda: None)

        virtual = adapter._model_name
        # Make _resolve_gateway_model return the virtual alias — the
        # condition the session-row bug can produce after a prior turn.
        monkeypatch.setattr(
            "gateway.run._resolve_gateway_model", lambda: virtual,
        )

        adapter._create_agent(session_id="s1", gateway_session_key="ch")
        assert captured[0]["model"] == virtual
        # Cache must reject the alias.
        assert adapter._last_resolved_model.get("ch") != virtual
        assert adapter._last_resolved_model.get("*") != virtual

    def test_create_agent_rejects_virtual_alias_from_cache(self, monkeypatch):
        """Read-side gate: an empty-model dispatch with the alias in
        ``_last_resolved_model`` must NOT recover it — the recovery net
        must never serve the advertised virtual model."""
        captured = []

        class FakeAgent:
            def __init__(self, **kwargs):
                captured.append(dict(kwargs))

        _patch_create_agent_runtime(monkeypatch, {}, FakeAgent)

        adapter = APIServerAdapter(PlatformConfig(enabled=True))
        monkeypatch.setattr(adapter, "_ensure_session_db", lambda: None)

        # Seed the cache with the alias (simulate a prior poisoned turn).
        adapter._last_resolved_model["ch"] = adapter._model_name
        adapter._last_resolved_model["*"] = adapter._model_name

        # Trigger an empty resolution.
        monkeypatch.setattr("gateway.run._resolve_gateway_model", lambda: "")
        monkeypatch.setattr(
            "gateway.run._resolve_runtime_agent_kwargs",
            lambda: {"provider": None, "base_url": None, "api_mode": None},
        )
        adapter._create_agent(session_id="s1", gateway_session_key="ch")

        # The alias must not be dispatched.
        assert captured[0]["model"] != adapter._model_name

    def test_create_agent_recovery_still_works_for_legitimate_model(
        self, monkeypatch,
    ):
        """Non-regression: a real dispatched model still enters the cache
        and recovers on a subsequent empty-resolution turn — the alias
        guard must not break legitimate recovery."""
        captured = []

        class FakeAgent:
            def __init__(self, **kwargs):
                captured.append(dict(kwargs))

        _patch_create_agent_runtime(monkeypatch, {}, FakeAgent)

        adapter = APIServerAdapter(PlatformConfig(enabled=True))
        monkeypatch.setattr(adapter, "_ensure_session_db", lambda: None)

        # Turn 1: legitimate model — must enter the cache.
        monkeypatch.setattr(
            "gateway.run._resolve_gateway_model",
            lambda: "anthropic/claude-opus-4.6",
        )
        adapter._create_agent(session_id="s1", gateway_session_key="ch")
        assert captured[0]["model"] == "anthropic/claude-opus-4.6"
        assert adapter._last_resolved_model["ch"] == "anthropic/claude-opus-4.6"

        # Turn 2: empty resolution — must recover the legitimate model.
        monkeypatch.setattr("gateway.run._resolve_gateway_model", lambda: "")
        monkeypatch.setattr(
            "gateway.run._resolve_runtime_agent_kwargs",
            lambda: {"provider": None, "base_url": None, "api_mode": None},
        )
        adapter._create_agent(session_id="s2", gateway_session_key="ch")
        assert captured[1]["model"] == "anthropic/claude-opus-4.6"
