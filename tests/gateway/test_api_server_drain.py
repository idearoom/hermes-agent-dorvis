"""API server drain-mode integration tests (AE-117, parent-repo ADR 0177).

Covers:
- 503 {"error": {"code": "gateway_draining"}} refusal on every run-launch
  endpoint while draining
- established streams, run polling, and read-only endpoints keep working
- active-run counting across the /v1/runs and _run_agent paths
- drain-cap force-terminate emits clean terminal events on run streams
- readiness surface reports {draining, active_runs} and flips not-ready
- /admin/drain trigger (one-way, authenticated)
- orphan sweeper never force-stops still-active runs mid-drain
"""

import asyncio
from contextlib import contextmanager
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
from unittest.mock import MagicMock, patch

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from gateway.config import PlatformConfig
from gateway.drain_mode import reset_drain_mode_for_tests
from gateway.platforms.api_server import (
    APIServerAdapter,
    cors_middleware,
    security_headers_middleware,
)


@pytest.fixture(autouse=True)
def _fresh_drain_mode():
    """Isolate the process-global drain latch per test.

    Adapters bind the global DrainMode at construction, so the reset must
    happen before the adapter fixture runs (fixture ordering: autouse
    function-scoped fixtures run before the test requests `adapter`).
    """
    reset_drain_mode_for_tests()
    yield
    reset_drain_mode_for_tests()


def _make_adapter(api_key: str = "") -> APIServerAdapter:
    extra = {}
    if api_key:
        extra["key"] = api_key
    return APIServerAdapter(PlatformConfig(enabled=True, extra=extra))


def _create_app(adapter: APIServerAdapter) -> web.Application:
    """App with every drain-relevant route registered."""
    mws = [mw for mw in (cors_middleware, security_headers_middleware) if mw is not None]
    app = web.Application(middlewares=mws)
    app["api_server_adapter"] = adapter
    app.router.add_get("/health", adapter._handle_health)
    app.router.add_get("/ready", adapter._handle_ready)
    app.router.add_get("/v1/models", adapter._handle_models)
    app.router.add_post("/admin/drain", adapter._handle_admin_drain)
    app.router.add_get("/admin/drain", adapter._handle_admin_drain_status)
    app.router.add_post("/v1/chat/completions", adapter._handle_chat_completions)
    app.router.add_post("/v1/responses", adapter._handle_responses)
    app.router.add_post("/api/sessions/{session_id}/chat", adapter._handle_session_chat)
    app.router.add_post(
        "/api/sessions/{session_id}/chat/stream", adapter._handle_session_chat_stream
    )
    app.router.add_post("/v1/runs", adapter._handle_runs)
    app.router.add_get("/v1/runs/{run_id}", adapter._handle_get_run)
    app.router.add_get("/v1/runs/{run_id}/events", adapter._handle_run_events)
    app.router.add_post("/v1/runs/{run_id}/stop", adapter._handle_stop_run)
    return app


_REPO_ROOT = Path(__file__).resolve().parents[2]


def _set_local_api_health(adapter: APIServerAdapter, *, connected: bool) -> None:
    adapter._running = connected
    adapter._runner = object()
    adapter._site = object()


async def _get_readiness(adapter: APIServerAdapter) -> tuple[int, dict]:
    app = _create_app(adapter)
    async with TestClient(TestServer(app)) as cli:
        resp = await cli.get("/ready")
        return resp.status, await resp.json()


def _write_peer_runtime_status(
    *, gateway_state: str = "stopped", platform_state: str = "disconnected"
) -> None:
    script = (
        "from gateway.status import write_runtime_status; "
        f"write_runtime_status(gateway_state={gateway_state!r}, "
        f"platform='api_server', platform_state={platform_state!r})"
    )
    subprocess.run(
        [sys.executable, "-c", script],
        cwd=_REPO_ROOT,
        env=os.environ.copy(),
        check=True,
        capture_output=True,
        text=True,
    )


@contextmanager
def _owned_runtime_lock():
    from gateway.status import (
        acquire_gateway_runtime_lock,
        release_gateway_runtime_lock,
    )

    assert acquire_gateway_runtime_lock() is True
    try:
        yield
    finally:
        release_gateway_runtime_lock()


@contextmanager
def _peer_owned_runtime_lock():
    from gateway.status import release_gateway_runtime_lock

    release_gateway_runtime_lock()
    script = """
from gateway.status import (
    acquire_gateway_runtime_lock,
    release_gateway_runtime_lock,
    write_runtime_status,
)
write_runtime_status(
    gateway_state="running",
    platform="api_server",
    platform_state="connected",
)
if not acquire_gateway_runtime_lock():
    raise SystemExit("peer could not acquire runtime lock")
print("locked", flush=True)
input()
release_gateway_runtime_lock()
"""
    process = subprocess.Popen(
        [sys.executable, "-c", script],
        cwd=_REPO_ROOT,
        env=os.environ.copy(),
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        assert process.stdout is not None
        assert process.stdout.readline().strip() == "locked"
        yield
    finally:
        stdout, stderr = process.communicate("release\n", timeout=5)
        assert process.returncode == 0, (stdout, stderr)


def _make_slow_agent():
    """Mock agent blocking in run_conversation until interrupt() is called."""
    ready = threading.Event()
    interrupted = threading.Event()
    mock_agent = MagicMock()

    def _do_interrupt(message=None):
        interrupted.set()

    mock_agent.interrupt = MagicMock(side_effect=_do_interrupt)

    def _slow_run(user_message=None, conversation_history=None, task_id=None):
        ready.set()
        interrupted.wait(timeout=10)
        return {"final_response": "interrupted"}

    mock_agent.run_conversation.side_effect = _slow_run
    mock_agent.session_prompt_tokens = 0
    mock_agent.session_completion_tokens = 0
    mock_agent.session_total_tokens = 0
    mock_agent._session_messages = []
    mock_agent.shutdown_memory_provider = MagicMock()
    mock_agent.close = MagicMock()
    return mock_agent, ready, interrupted


@pytest.fixture
def adapter(_fresh_drain_mode):
    return _make_adapter()


@pytest.fixture
def auth_adapter(_fresh_drain_mode):
    return _make_adapter(api_key="sk-secret")


async def _assert_drain_refusal(resp) -> None:
    assert resp.status == 503
    assert resp.headers.get("Retry-After")
    data = await resp.json()
    assert data["error"]["code"] == "gateway_draining"
    assert data["error"]["type"] == "unavailable_error"


# ---------------------------------------------------------------------------
# Refusal contract on launch endpoints
# ---------------------------------------------------------------------------


class TestDrainRefusal:
    @pytest.mark.asyncio
    async def test_v1_runs_refused_while_draining(self, adapter):
        adapter._drain_mode.begin("test")
        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.post("/v1/runs", json={"input": "hello"})
            await _assert_drain_refusal(resp)
        assert adapter._run_streams == {}
        assert adapter._run_statuses == {}

    @pytest.mark.asyncio
    async def test_v1_responses_refused_while_draining(self, adapter):
        adapter._drain_mode.begin("test")
        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.post("/v1/responses", json={"input": "hello"})
            await _assert_drain_refusal(resp)

    @pytest.mark.asyncio
    async def test_v1_chat_completions_refused_while_draining(self, adapter):
        adapter._drain_mode.begin("test")
        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.post(
                "/v1/chat/completions",
                json={"messages": [{"role": "user", "content": "hi"}]},
            )
            await _assert_drain_refusal(resp)

    @pytest.mark.asyncio
    async def test_session_chat_refused_while_draining(self, adapter):
        adapter._drain_mode.begin("test")
        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.post("/api/sessions/sess_x/chat", json={"message": "hi"})
            await _assert_drain_refusal(resp)

    @pytest.mark.asyncio
    async def test_session_chat_stream_refused_while_draining(self, adapter):
        adapter._drain_mode.begin("test")
        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.post(
                "/api/sessions/sess_x/chat/stream", json={"message": "hi"}
            )
            await _assert_drain_refusal(resp)

    @pytest.mark.asyncio
    async def test_not_draining_accepts_runs(self, adapter):
        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_create_agent") as mock_create:
                mock_agent = MagicMock()
                mock_agent.run_conversation.return_value = {"final_response": "ok"}
                mock_agent.session_prompt_tokens = 0
                mock_agent.session_completion_tokens = 0
                mock_agent.session_total_tokens = 0
                mock_create.return_value = mock_agent
                resp = await cli.post("/v1/runs", json={"input": "hello"})
                assert resp.status == 202

    @pytest.mark.asyncio
    async def test_auth_still_precedes_drain_refusal(self, auth_adapter):
        """Unauthenticated callers see 401, not the drain state."""
        auth_adapter._drain_mode.begin("test")
        app = _create_app(auth_adapter)
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.post("/v1/runs", json={"input": "hello"})
            assert resp.status == 401


# ---------------------------------------------------------------------------
# In-flight work and read paths keep working during drain
# ---------------------------------------------------------------------------


class TestDrainKeepsInFlightWorking:
    @pytest.mark.asyncio
    async def test_established_run_survives_drain_and_streams_to_completion(self, adapter):
        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_create_agent") as mock_create:
                mock_agent, ready, interrupted = _make_slow_agent()
                mock_create.return_value = mock_agent

                resp = await cli.post("/v1/runs", json={"input": "hello"})
                assert resp.status == 202
                run_id = (await resp.json())["run_id"]
                assert await asyncio.to_thread(ready.wait, 5)

                # Drain begins mid-run.
                adapter._drain_mode.begin("test")

                # New launches are refused...
                refused = await cli.post("/v1/runs", json={"input": "more"})
                await _assert_drain_refusal(refused)

                # ...but polling the in-flight run still works...
                status_resp = await cli.get(f"/v1/runs/{run_id}")
                assert status_resp.status == 200

                # ...read-only endpoints still work...
                assert (await cli.get("/health")).status == 200
                assert (await cli.get("/v1/models")).status == 200

                # ...and the established stream still serves. Let the run
                # finish naturally and observe its clean completion.
                interrupted.set()
                events_resp = await cli.get(f"/v1/runs/{run_id}/events")
                assert events_resp.status == 200
                body = await events_resp.text()
                assert "run.completed" in body
        assert mock_agent.interrupt.call_count == 0  # drain never interrupted it

    @pytest.mark.asyncio
    async def test_active_run_count_tracks_runs_and_inflight_paths(self, adapter):
        assert adapter._drain_active_run_count() == 0
        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_create_agent") as mock_create:
                mock_agent, ready, interrupted = _make_slow_agent()
                mock_create.return_value = mock_agent
                resp = await cli.post("/v1/runs", json={"input": "hello"})
                run_id = (await resp.json())["run_id"]
                assert await asyncio.to_thread(ready.wait, 5)
                assert adapter._drain_active_run_count() == 1
                assert adapter._drain_mode.active_runs() == 1  # registered source

                interrupted.set()
                for _ in range(100):
                    if adapter._drain_active_run_count() == 0:
                        break
                    await asyncio.sleep(0.05)
                assert adapter._drain_active_run_count() == 0
                # Terminal status retained for polling even at zero active.
                status = await cli.get(f"/v1/runs/{run_id}")
                assert status.status == 200

    @pytest.mark.asyncio
    async def test_run_agent_path_counts_and_registers_inflight_agent(self, adapter):
        with patch.object(adapter, "_create_agent") as mock_create:
            mock_agent, ready, interrupted = _make_slow_agent()
            mock_create.return_value = mock_agent

            run_task = asyncio.create_task(
                adapter._run_agent(user_message="hi", conversation_history=[])
            )
            assert await asyncio.to_thread(ready.wait, 5)
            assert adapter._inflight_agent_runs == 1
            assert len(adapter._inflight_run_agents) == 1
            assert adapter._drain_active_run_count() == 1

            # Drain-cap force-terminate reaches this agent too.
            terminated = adapter._drain_force_terminate("drain cap reached")
            assert terminated == 1
            mock_agent.interrupt.assert_called_once_with("drain cap reached")

            await asyncio.wait_for(run_task, timeout=5.0)
            assert adapter._inflight_agent_runs == 0
            assert adapter._inflight_run_agents == {}
            assert adapter._drain_active_run_count() == 0


# ---------------------------------------------------------------------------
# Drain-cap force-termination emits clean terminal events
# ---------------------------------------------------------------------------


class TestDrainForceTerminate:
    @pytest.mark.asyncio
    async def test_force_terminate_run_emits_terminal_event_and_closes_stream(self, adapter):
        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_create_agent") as mock_create:
                mock_agent, ready, interrupted = _make_slow_agent()
                mock_create.return_value = mock_agent

                resp = await cli.post("/v1/runs", json={"input": "hello"})
                run_id = (await resp.json())["run_id"]
                assert await asyncio.to_thread(ready.wait, 5)

                adapter._drain_mode.begin("test")
                terminated = adapter._drain_force_terminate(
                    "Gateway drain cap reached — run terminated before shutdown"
                )
                assert terminated >= 1
                assert mock_agent.interrupt.called

                events_resp = await cli.get(f"/v1/runs/{run_id}/events")
                assert events_resp.status == 200
                body = await events_resp.text()
                # A definite terminal event, then the stream-close sentinel —
                # clients see a clean end, never a hang.
                assert (
                    "run.cancelled" in body
                    or "run.failed" in body
                    or "run.completed" in body
                )
                assert "stream closed" in body

                for _ in range(100):
                    if adapter._drain_active_run_count() == 0:
                        break
                    await asyncio.sleep(0.05)
                assert adapter._drain_active_run_count() == 0

    @pytest.mark.asyncio
    async def test_force_terminate_with_no_runs_is_noop(self, adapter):
        assert adapter._drain_force_terminate("cap") == 0


# ---------------------------------------------------------------------------
# Readiness surface
# ---------------------------------------------------------------------------


class TestDrainReadiness:
    @pytest.mark.asyncio
    async def test_static_owner_stays_ready_when_shared_status_belongs_to_other_task(
        self, auth_adapter, monkeypatch
    ):
        """A blue/green peer's EFS status must not evict this healthy task."""
        monkeypatch.setenv("HERMES_RUN_GATEWAY", "1")
        monkeypatch.setenv("HERMES_GATEWAY_OWNER", "main-hermes")
        _set_local_api_health(auth_adapter, connected=True)
        _write_peer_runtime_status()

        with _owned_runtime_lock():
            status, data = await _get_readiness(auth_adapter)

        assert status == 200
        assert data["status"] == "ready"
        assert data["checks"]["runtime_status_current_pid"] is False
        assert data["checks"]["gateway_state_running"] is False
        assert data["checks"]["api_server_platform_connected"] is False
        assert data["checks"]["api_server_local_connected"] is True
        assert data["advisories"]

    @pytest.mark.asyncio
    async def test_generic_gateway_keeps_strict_runtime_status_check(
        self, auth_adapter, monkeypatch
    ):
        monkeypatch.delenv("HERMES_RUN_GATEWAY", raising=False)
        monkeypatch.delenv("HERMES_GATEWAY_OWNER", raising=False)
        _set_local_api_health(auth_adapter, connected=True)
        _write_peer_runtime_status()

        with _owned_runtime_lock():
            status, data = await _get_readiness(auth_adapter)

        assert status == 503
        assert data["status"] == "not_ready"
        assert any("runtime status belongs to pid" in reason for reason in data["reasons"])

    @pytest.mark.asyncio
    async def test_static_owner_still_requires_local_api_connection(
        self, auth_adapter, monkeypatch
    ):
        monkeypatch.setenv("HERMES_RUN_GATEWAY", "1")
        monkeypatch.setenv("HERMES_GATEWAY_OWNER", "main-hermes")
        _set_local_api_health(auth_adapter, connected=False)
        _write_peer_runtime_status()

        with _owned_runtime_lock():
            status, data = await _get_readiness(auth_adapter)

        assert status == 503
        assert data["checks"]["api_server_local_connected"] is False
        assert "local api server adapter is not connected" in data["reasons"]

    @pytest.mark.asyncio
    async def test_static_owner_requires_its_own_runtime_lock(
        self, auth_adapter, monkeypatch
    ):
        monkeypatch.setenv("HERMES_RUN_GATEWAY", "1")
        monkeypatch.setenv("HERMES_GATEWAY_OWNER", "main-hermes")
        _set_local_api_health(auth_adapter, connected=True)

        with _peer_owned_runtime_lock():
            status, data = await _get_readiness(auth_adapter)

        assert status == 503
        assert data["checks"]["runtime_lock_owned_by_current_process"] is False
        assert "gateway runtime lock is owned by another process" in data["reasons"]

    def test_payload_reports_not_draining_by_default(self, adapter):
        payload, _status = adapter._readiness_payload()
        assert payload["draining"] is False
        assert payload["active_runs"] == 0
        assert payload["checks"]["gateway_not_draining"] is True

    def test_configured_postgres_backends_require_positive_attestation(
        self, adapter, monkeypatch
    ):
        monkeypatch.setenv("HERMES_STATE_STORE_DSN", "postgresql://configured")
        monkeypatch.setenv("HERMES_RESPONSE_STORE_DSN", "postgresql://configured")

        payload, status = adapter._readiness_payload()

        assert status == 503
        assert payload["checks"]["session_store_postgres"] is False
        assert payload["checks"]["response_store_postgres"] is False
        assert payload["session_store"]["backend"] != "postgres"
        assert payload["response_store"]["backend"] != "postgres"

    def test_payload_flips_when_draining(self, adapter):
        adapter._drain_mode.begin("test")
        payload, status = adapter._readiness_payload()
        assert status == 503
        assert payload["status"] == "not_ready"
        assert payload["draining"] is True
        assert payload["checks"]["gateway_not_draining"] is False
        assert any("drain" in r for r in payload["reasons"])

    @pytest.mark.asyncio
    async def test_ready_endpoint_served_while_draining(self, adapter):
        adapter._drain_mode.begin("test")
        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.get("/ready")
            assert resp.status == 503
            data = await resp.json()
            assert data["draining"] is True
            assert isinstance(data["active_runs"], int)

    @pytest.mark.asyncio
    async def test_active_runs_reported_in_readiness(self, adapter):
        with patch.object(adapter, "_create_agent") as mock_create:
            mock_agent, ready, interrupted = _make_slow_agent()
            mock_create.return_value = mock_agent
            run_task = asyncio.create_task(
                adapter._run_agent(user_message="hi", conversation_history=[])
            )
            assert await asyncio.to_thread(ready.wait, 5)
            payload, _ = adapter._readiness_payload()
            assert payload["active_runs"] == 1
            interrupted.set()
            await asyncio.wait_for(run_task, timeout=5.0)
            payload, _ = adapter._readiness_payload()
            assert payload["active_runs"] == 0


# ---------------------------------------------------------------------------
# /admin/drain trigger
# ---------------------------------------------------------------------------


class TestAdminDrainEndpoint:
    @pytest.mark.asyncio
    async def test_requires_auth(self, auth_adapter):
        app = _create_app(auth_adapter)
        async with TestClient(TestServer(app)) as cli:
            assert (await cli.post("/admin/drain", json={})).status == 401
            assert (await cli.get("/admin/drain")).status == 401
        assert auth_adapter._drain_mode.draining is False

    @pytest.mark.asyncio
    async def test_post_engages_drain_one_way(self, auth_adapter):
        app = _create_app(auth_adapter)
        headers = {"Authorization": "Bearer sk-secret"}
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.post(
                "/admin/drain", json={"reason": "deploy-blue-green"}, headers=headers
            )
            assert resp.status == 202
            data = await resp.json()
            assert data["object"] == "hermes.gateway.drain"
            assert data["draining"] is True
            assert data["already_draining"] is False
            assert data["drain_reason"] == "admin:deploy-blue-green"
            assert isinstance(data["active_runs"], int)
            assert data["drain_cap_seconds"] > 0

            # Idempotent second call reports already_draining.
            resp2 = await cli.post("/admin/drain", json={}, headers=headers)
            assert resp2.status == 200
            data2 = await resp2.json()
            assert data2["already_draining"] is True
            assert data2["drain_reason"] == "admin:deploy-blue-green"

            status = await cli.get("/admin/drain", headers=headers)
            assert status.status == 200
            sdata = await status.json()
            assert sdata["draining"] is True
        assert auth_adapter._drain_mode.draining is True

    @pytest.mark.asyncio
    async def test_get_status_does_not_engage(self, auth_adapter):
        app = _create_app(auth_adapter)
        headers = {"Authorization": "Bearer sk-secret"}
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.get("/admin/drain", headers=headers)
            assert resp.status == 200
            data = await resp.json()
            assert data["draining"] is False
        assert auth_adapter._drain_mode.draining is False


# ---------------------------------------------------------------------------
# Orphan sweeper interplay
# ---------------------------------------------------------------------------


class TestOrphanSweeperDuringDrain:
    async def _one_sweep(self, adapter):
        async def _one_sweep_then_cancel(delay):
            if getattr(_one_sweep_then_cancel, "called", False):
                raise asyncio.CancelledError()
            _one_sweep_then_cancel.called = True

        with patch("gateway.platforms.api_server.asyncio.sleep", new=_one_sweep_then_cancel):
            with pytest.raises(asyncio.CancelledError):
                await adapter._sweep_orphaned_runs()

    @pytest.mark.asyncio
    async def test_draining_established_stream_is_not_swept(self, adapter):
        adapter._drain_mode.begin("test")
        run_id = "run_active_drain"
        mock_agent = MagicMock()
        adapter._run_streams[run_id] = asyncio.Queue()
        adapter._run_streams_created[run_id] = time.time() - adapter._RUN_STREAM_TTL - 1
        adapter._run_stream_subscribers.add(run_id)
        adapter._active_run_agents[run_id] = mock_agent

        async def _never():
            await asyncio.Event().wait()

        pending = asyncio.create_task(_never())
        adapter._active_run_tasks[run_id] = pending
        try:
            await self._one_sweep(adapter)
            # An established subscriber survives the transport TTL while the
            # draining executor-backed run continues.
            assert run_id in adapter._run_streams
            assert run_id in adapter._active_run_tasks
            mock_agent.interrupt.assert_not_called()
            assert not pending.cancelled()
        finally:
            pending.cancel()

    @pytest.mark.asyncio
    async def test_draining_completed_stream_still_ages_out(self, adapter):
        adapter._drain_mode.begin("test")
        run_id = "run_done_drain"
        adapter._run_streams[run_id] = asyncio.Queue()
        adapter._run_streams_created[run_id] = time.time() - adapter._RUN_STREAM_TTL - 1
        # No active task/agent: the run already finished, only the
        # unconsumed stream remains — that may still be swept mid-drain.
        await self._one_sweep(adapter)
        assert run_id not in adapter._run_streams

    @pytest.mark.asyncio
    async def test_not_draining_keeps_upstream_sweep_behavior(self, adapter):
        run_id = "run_upstream"
        mock_agent = MagicMock()
        adapter._run_streams[run_id] = asyncio.Queue()
        adapter._run_streams_created[run_id] = time.time() - adapter._RUN_STREAM_TTL - 1
        adapter._active_run_agents[run_id] = mock_agent
        await self._one_sweep(adapter)
        assert run_id not in adapter._run_streams
        # Current upstream treats the TTL as a transport-buffer bound, not a
        # run lifetime: it never interrupts work merely because no SSE client
        # subscribed before the buffer expired.
        mock_agent.interrupt.assert_not_called()
