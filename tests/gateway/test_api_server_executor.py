"""AE-288: the API server's default executor follows max_concurrent_runs.

Agent turns run on the loop's default executor. asyncio sizes that pool from
the CPU count (8 threads on the 4-vCPU Fargate task), so the admission cap
could admit more turns than could execute. These tests pin the sizing rule,
the install-once behavior across reconnects, and that a full cap of blocking
turns runs in parallel with to_thread headroom left over.
"""

import asyncio
import concurrent.futures
import logging
import secrets
import threading
from unittest.mock import MagicMock, patch

import pytest

from gateway.config import PlatformConfig
from gateway.platforms import api_server_executor
from gateway.platforms.api_server import APIServerAdapter
from gateway.platforms.api_server_executor import (
    EXECUTOR_HEADROOM,
    THREAD_NAME_PREFIX,
    UNCAPPED_EXECUTOR_WORKERS,
    api_server_executor_workers,
    install_api_server_default_executor,
)

# Safety net so a regression fails instead of parking threads forever.
_WAIT_TIMEOUT = 10.0
# The parent repo's target cap: 16 web chat + 5 headless triage.
_TARGET_CAP = 21
# The stock asyncio default executor measured on the 4-vCPU Hermes task.
_FARGATE_STOCK_WORKERS = 8


def _default_executor(loop):
    return getattr(loop, "_default_executor", None)


def _make_adapter(cap: int) -> APIServerAdapter:
    with patch.object(
        APIServerAdapter, "_resolve_max_concurrent_runs", return_value=cap
    ):
        adapter = APIServerAdapter(
            PlatformConfig(
                enabled=True,
                extra={
                    "host": "127.0.0.1",
                    "port": 0,
                    "key": secrets.token_hex(32),
                },
            )
        )
    assert adapter._max_concurrent_runs == cap
    return adapter


def _make_agent(arrived, release):
    agent = MagicMock()

    def _run_conversation(**_kwargs):
        arrived()
        assert release.wait(timeout=_WAIT_TIMEOUT), "turn was never released"
        return {"final_response": "ok", "messages": [], "api_calls": 1}

    agent.run_conversation.side_effect = _run_conversation
    agent.session_prompt_tokens = 1
    agent.session_completion_tokens = 1
    agent.session_total_tokens = 2
    agent._session_messages = []
    return agent


async def _run_blocking_turns(adapter, turns, *, fill_timeout=_WAIT_TIMEOUT):
    """Start *turns* blocking _run_agent calls; return (peak, release, tasks)."""
    lock = threading.Lock()
    state = {"live": 0, "peak": 0}
    release = threading.Event()

    def arrived():
        with lock:
            state["live"] += 1
            state["peak"] = max(state["peak"], state["live"])

    agents = [_make_agent(arrived, release) for _ in range(turns)]
    # Replace the factory on this throwaway adapter for good, not inside a
    # context manager: turns still queued for a thread when this helper
    # returns must never reach the real AIAgent (and a live model).
    adapter._create_agent = MagicMock(side_effect=agents)
    tasks = [
        asyncio.create_task(
            adapter._run_agent(
                user_message=f"turn {index}",
                conversation_history=[],
                session_id=f"ae288-{index}",
            )
        )
        for index in range(turns)
    ]
    # Let the executor pick up everything it can.
    deadline = asyncio.get_running_loop().time() + fill_timeout
    while state["live"] < turns and asyncio.get_running_loop().time() < deadline:
        await asyncio.sleep(0.02)
    # A saturated pool stays saturated; settle before sampling the peak.
    await asyncio.sleep(0.2)
    return state, release, tasks


class TestExecutorSizing:
    @pytest.mark.parametrize(
        ("cap", "workers"),
        [(1, 32), (10, 32), (16, 32), (_TARGET_CAP, 37), (100, 116)],
    )
    def test_workers_follow_cap_with_headroom(self, cap, workers):
        assert api_server_executor_workers(cap) == workers

    @pytest.mark.parametrize("cap", [0, -1])
    def test_disabled_cap_uses_bounded_pool(self, cap):
        assert api_server_executor_workers(cap) == UNCAPPED_EXECUTOR_WORKERS


class TestInstall:
    @pytest.mark.asyncio
    async def test_replaces_stock_default_and_is_idempotent(self, caplog):
        loop = asyncio.get_running_loop()
        # Force asyncio to create its lazy stock executor first, as gateway
        # startup work may do before the adapter connects.
        await loop.run_in_executor(None, lambda: None)
        stock = _default_executor(loop)
        assert stock is not None

        with caplog.at_level(logging.INFO, logger=api_server_executor.__name__):
            first = install_api_server_default_executor(loop, _TARGET_CAP)
            second = install_api_server_default_executor(loop, _TARGET_CAP)
        try:
            assert first is second
            assert _default_executor(loop) is first
            assert first._max_workers == 37
            assert first._thread_name_prefix == THREAD_NAME_PREFIX
            assert stock._shutdown, "asyncio's replaced pool must be released"
            logs = [r for r in caplog.records if "default executor" in r.getMessage()]
            assert len(logs) == 1
            assert "37 workers" in logs[0].getMessage()
            names = await asyncio.to_thread(lambda: threading.current_thread().name)
            assert names.startswith(THREAD_NAME_PREFIX)
        finally:
            first.shutdown(wait=True)

    @pytest.mark.asyncio
    async def test_larger_cap_grows_and_smaller_cap_keeps(self):
        loop = asyncio.get_running_loop()
        small = install_api_server_default_executor(loop, 10)
        large = install_api_server_default_executor(loop, 40)
        kept = install_api_server_default_executor(loop, 5)
        try:
            assert large is not small
            assert small._shutdown
            assert kept is large
            assert large._max_workers == 56
        finally:
            large.shutdown(wait=True)

    @pytest.mark.asyncio
    async def test_foreign_executor_is_replaced_but_left_to_its_owner(self):
        loop = asyncio.get_running_loop()
        foreign = concurrent.futures.ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="someone-else"
        )
        loop.set_default_executor(foreign)
        ours = install_api_server_default_executor(loop, _TARGET_CAP)
        try:
            assert _default_executor(loop) is ours
            assert not foreign._shutdown
        finally:
            ours.shutdown(wait=True)
            foreign.shutdown(wait=True)

    @pytest.mark.asyncio
    async def test_reinstalls_when_someone_else_replaced_ours(self):
        loop = asyncio.get_running_loop()
        ours = install_api_server_default_executor(loop, _TARGET_CAP)
        foreign = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        loop.set_default_executor(foreign)
        again = install_api_server_default_executor(loop, _TARGET_CAP)
        try:
            assert again is not ours
            assert _default_executor(loop) is again
        finally:
            for executor in (ours, again, foreign):
                executor.shutdown(wait=True)


class TestConnectInstallsExecutor:
    @pytest.mark.asyncio
    async def test_connect_sizes_default_executor_from_cap(self):
        loop = asyncio.get_running_loop()
        adapter = _make_adapter(_TARGET_CAP)
        assert await adapter.connect() is True
        try:
            installed = _default_executor(loop)
            assert installed._max_workers == _TARGET_CAP + EXECUTOR_HEADROOM
            assert installed._thread_name_prefix == THREAD_NAME_PREFIX
        finally:
            await adapter.disconnect()

        # disconnect() leaves the loop's executor alone; a reconnect (a new
        # adapter on the same loop) reuses it instead of rebuilding.
        assert not installed._shutdown
        again = _make_adapter(_TARGET_CAP)
        assert await again.connect() is True
        try:
            assert _default_executor(loop) is installed
        finally:
            await again.disconnect()
            installed.shutdown(wait=True)

    @pytest.mark.asyncio
    async def test_rejected_key_does_not_touch_executor(self, monkeypatch):
        loop = asyncio.get_running_loop()
        before = _default_executor(loop)
        monkeypatch.delenv("API_SERVER_KEY", raising=False)
        adapter = APIServerAdapter(
            PlatformConfig(enabled=True, extra={"host": "127.0.0.1", "port": 0})
        )
        assert await adapter.connect() is False
        assert _default_executor(loop) is before


class TestTurnsRunInParallel:
    @pytest.mark.asyncio
    async def test_full_cap_of_blocking_turns_runs_concurrently(self):
        """Every admitted turn executes at once, with to_thread room left."""
        adapter = _make_adapter(_TARGET_CAP)
        assert await adapter.connect() is True
        installed = _default_executor(asyncio.get_running_loop())
        state = release = tasks = None
        try:
            state, release, tasks = await _run_blocking_turns(adapter, _TARGET_CAP)
            assert state["peak"] == _TARGET_CAP

            # Auth verification and session-DB opens use to_thread; they must
            # not queue behind a full cap of turns.
            probe = await asyncio.wait_for(
                asyncio.to_thread(lambda: "auth-ok"), timeout=_WAIT_TIMEOUT
            )
            assert probe == "auth-ok"
        finally:
            if release is not None:
                release.set()
            if tasks:
                results = await asyncio.gather(*tasks, return_exceptions=True)
                errors = [r for r in results if isinstance(r, BaseException)]
                assert not errors, errors
            await adapter.disconnect()
            installed.shutdown(wait=True)

    @pytest.mark.asyncio
    async def test_control_stock_fargate_pool_caps_turns_at_eight(self):
        """Without the install, 21 admitted turns run 8 at a time."""
        loop = asyncio.get_running_loop()
        stock = concurrent.futures.ThreadPoolExecutor(
            max_workers=_FARGATE_STOCK_WORKERS, thread_name_prefix="asyncio"
        )
        loop.set_default_executor(stock)
        adapter = _make_adapter(_TARGET_CAP)
        state = release = tasks = None
        try:
            state, release, tasks = await _run_blocking_turns(
                adapter, _TARGET_CAP, fill_timeout=1.0
            )
            assert state["peak"] == _FARGATE_STOCK_WORKERS
        finally:
            if release is not None:
                release.set()
            if tasks:
                results = await asyncio.gather(*tasks, return_exceptions=True)
                errors = [r for r in results if isinstance(r, BaseException)]
                assert not errors, errors
            stock.shutdown(wait=True)
