"""Tests for gateway/drain_mode.py (AE-117, parent-repo ADR 0177).

Covers:
- DrainMode one-way latch, source registry, active-run counting
- sigterm_begins_drain env gating + escalation on second SIGTERM
- drain_coordinator_loop: self-exit at zero (settle), drain-cap
  force-termination, relay drain-marker write
- EcsTaskProtection: set/renew/release ordering, non-ECS no-op,
  failure tolerance (never raises)
"""

import asyncio
import time

import pytest

from gateway import drain_mode
from gateway.drain_mode import (
    DRAIN_REFUSAL_ERROR_CODE,
    DrainMode,
    EcsTaskProtection,
    drain_coordinator_loop,
    get_drain_mode,
    reset_drain_mode_for_tests,
    sigterm_begins_drain,
    task_protection_loop,
)


@pytest.fixture(autouse=True)
def _fresh_drain_mode():
    reset_drain_mode_for_tests()
    yield
    reset_drain_mode_for_tests()


# ---------------------------------------------------------------------------
# DrainMode latch + registry
# ---------------------------------------------------------------------------


class TestDrainModeLatch:
    def test_not_draining_by_default(self):
        drain = DrainMode()
        assert drain.draining is False
        assert drain.reason is None
        assert drain.started_at is None
        assert drain.elapsed_seconds() is None

    def test_begin_engages_once_and_is_one_way(self):
        drain = DrainMode()
        assert drain.begin("sigterm") is True
        assert drain.draining is True
        assert drain.reason == "sigterm"
        assert drain.started_at is not None
        # Second begin is ignored and does not overwrite the reason.
        assert drain.begin("admin:other") is False
        assert drain.reason == "sigterm"
        assert drain.draining is True

    def test_active_runs_sums_sources_and_tolerates_failures(self):
        drain = DrainMode()
        drain.register_source("a", lambda: 2)
        drain.register_source("b", lambda: 3)

        def _broken():
            raise RuntimeError("boom")

        drain.register_source("broken", _broken)
        assert drain.active_runs() == 5

    def test_register_source_replaces_by_name(self):
        drain = DrainMode()
        drain.register_source("api_server", lambda: 7)
        drain.register_source("api_server", lambda: 1)
        assert drain.active_runs() == 1
        drain.unregister_source("api_server")
        assert drain.active_runs() == 0

    def test_force_terminate_all_invokes_registered_terminators(self):
        drain = DrainMode()
        calls = []
        drain.register_source(
            "a", lambda: 1, lambda reason: calls.append(("a", reason)) or 2
        )
        drain.register_source("no-term", lambda: 0, None)

        def _broken(reason):
            raise RuntimeError("boom")

        drain.register_source("broken", lambda: 0, _broken)
        total = drain.force_terminate_all("cap reached")
        assert total == 2
        assert calls == [("a", "cap reached")]

    def test_snapshot_shape(self):
        drain = DrainMode()
        drain.register_source("a", lambda: 4)
        snap = drain.snapshot()
        assert snap["draining"] is False
        assert snap["active_runs"] == 4
        assert snap["drain_reason"] is None
        assert snap["drain_started_at"] is None
        assert snap["drain_cap_seconds"] == drain_mode.DEFAULT_DRAIN_CAP_SECONDS
        drain.begin("test")
        snap = drain.snapshot()
        assert snap["draining"] is True
        assert snap["drain_reason"] == "test"
        assert snap["drain_started_at"] is not None


# ---------------------------------------------------------------------------
# SIGTERM hook
# ---------------------------------------------------------------------------


class TestSigtermBeginsDrain:
    def test_disabled_by_default(self, monkeypatch):
        monkeypatch.delenv("HERMES_DRAIN_ON_SIGTERM", raising=False)
        assert sigterm_begins_drain() is False
        assert get_drain_mode().draining is False

    def test_enabled_first_sigterm_begins_drain(self, monkeypatch):
        monkeypatch.setenv("HERMES_DRAIN_ON_SIGTERM", "1")
        assert sigterm_begins_drain() is True
        drain = get_drain_mode()
        assert drain.draining is True
        assert drain.reason == "sigterm"

    def test_second_sigterm_escalates_to_immediate_shutdown(self, monkeypatch):
        monkeypatch.setenv("HERMES_DRAIN_ON_SIGTERM", "true")
        assert sigterm_begins_drain() is True
        # Already draining: the hook declines, so the caller proceeds with
        # the normal immediate stop path.
        assert sigterm_begins_drain() is False
        assert get_drain_mode().draining is True

    def test_falsy_env_values_disable(self, monkeypatch):
        for value in ("0", "false", "no", ""):
            reset_drain_mode_for_tests()
            monkeypatch.setenv("HERMES_DRAIN_ON_SIGTERM", value)
            assert sigterm_begins_drain() is False


# ---------------------------------------------------------------------------
# Env parsing
# ---------------------------------------------------------------------------


class TestEnvParsing:
    def test_drain_cap_default_and_override(self, monkeypatch):
        monkeypatch.delenv("HERMES_DRAIN_CAP_SECONDS", raising=False)
        assert drain_mode.drain_cap_seconds() == 3600.0
        monkeypatch.setenv("HERMES_DRAIN_CAP_SECONDS", "120")
        assert drain_mode.drain_cap_seconds() == 120.0

    def test_invalid_values_fall_back_to_default(self, monkeypatch):
        for bad in ("abc", "-5", "0"):
            monkeypatch.setenv("HERMES_DRAIN_CAP_SECONDS", bad)
            assert drain_mode.drain_cap_seconds() == 3600.0
        monkeypatch.setenv("HERMES_DRAIN_SETTLE_SECONDS", "nope")
        assert drain_mode.drain_settle_seconds() == 10.0


# ---------------------------------------------------------------------------
# Drain coordinator
# ---------------------------------------------------------------------------


class TestDrainCoordinatorLoop:
    @pytest.mark.asyncio
    async def test_self_exit_at_zero_after_settle(self):
        drain = DrainMode()
        counts = [2, 2, 1, 0]

        def _count():
            return counts.pop(0) if counts else 0

        drain.register_source("fake", _count)
        shutdown_calls = []

        task = asyncio.create_task(
            drain_coordinator_loop(
                drain,
                shutdown_cb=lambda: shutdown_calls.append(time.monotonic()),
                poll_interval=0.01,
                cap_seconds=60.0,
                settle_seconds=0.05,
                force_grace_seconds=0.05,
                write_marker=False,
            )
        )
        await asyncio.sleep(0.05)
        assert not shutdown_calls  # not draining yet — coordinator idles
        drain.begin("test")
        await asyncio.wait_for(task, timeout=5.0)
        assert len(shutdown_calls) == 1

    @pytest.mark.asyncio
    async def test_settle_window_resets_if_run_reappears(self):
        drain = DrainMode()
        # Dip to zero once, come back with one run, then finish for real.
        counts = [1, 0, 1, 1, 0, 0, 0, 0]

        def _count():
            return counts.pop(0) if counts else 0

        drain.register_source("fake", _count)
        shutdown_calls = []
        drain.begin("test")
        await asyncio.wait_for(
            drain_coordinator_loop(
                drain,
                shutdown_cb=lambda: shutdown_calls.append(1),
                poll_interval=0.01,
                cap_seconds=60.0,
                settle_seconds=0.03,
                force_grace_seconds=0.05,
                write_marker=False,
            ),
            timeout=5.0,
        )
        assert shutdown_calls == [1]

    @pytest.mark.asyncio
    async def test_cap_forces_termination_then_shutdown(self):
        drain = DrainMode()
        state = {"active": 1, "terminated": None}

        def _terminate(reason):
            state["terminated"] = reason
            state["active"] = 0
            return 1

        drain.register_source("fake", lambda: state["active"], _terminate)
        shutdown_calls = []
        drain.begin("test")
        await asyncio.wait_for(
            drain_coordinator_loop(
                drain,
                shutdown_cb=lambda: shutdown_calls.append(1),
                poll_interval=0.01,
                cap_seconds=0.05,
                settle_seconds=0.02,
                force_grace_seconds=1.0,
                write_marker=False,
            ),
            timeout=5.0,
        )
        assert state["terminated"] is not None
        assert "drain cap" in state["terminated"].lower()
        assert shutdown_calls == [1]

    @pytest.mark.asyncio
    async def test_cap_shutdown_even_if_runs_never_flush(self):
        drain = DrainMode()
        # Terminator is called but the count never reaches zero: the
        # force-grace window bounds the wait and shutdown still happens.
        terminate_calls = []
        drain.register_source(
            "stuck", lambda: 1, lambda reason: terminate_calls.append(reason) or 1
        )
        shutdown_calls = []
        drain.begin("test")
        await asyncio.wait_for(
            drain_coordinator_loop(
                drain,
                shutdown_cb=lambda: shutdown_calls.append(1),
                poll_interval=0.01,
                cap_seconds=0.03,
                settle_seconds=0.02,
                force_grace_seconds=0.05,
                write_marker=False,
            ),
            timeout=5.0,
        )
        assert len(terminate_calls) == 1
        assert shutdown_calls == [1]

    @pytest.mark.asyncio
    async def test_awaitable_shutdown_cb_is_awaited(self):
        drain = DrainMode()
        drain.register_source("fake", lambda: 0)
        done = asyncio.Event()

        async def _shutdown():
            done.set()

        drain.begin("test")
        await asyncio.wait_for(
            drain_coordinator_loop(
                drain,
                shutdown_cb=_shutdown,
                poll_interval=0.01,
                cap_seconds=60.0,
                settle_seconds=0.01,
                write_marker=False,
            ),
            timeout=5.0,
        )
        assert done.is_set()

    @pytest.mark.asyncio
    async def test_writes_relay_drain_marker(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        from gateway.drain_control import drain_request_path, read_drain_request

        drain = DrainMode()
        drain.register_source("fake", lambda: 0)
        drain.begin("sigterm")
        await asyncio.wait_for(
            drain_coordinator_loop(
                drain,
                shutdown_cb=lambda: None,
                poll_interval=0.01,
                cap_seconds=60.0,
                settle_seconds=0.01,
                write_marker=True,
            ),
            timeout=5.0,
        )
        assert drain_request_path().exists()
        body = read_drain_request()
        assert body["principal"] == "drain-mode:sigterm"
        assert body["suppress_notification"] is True

    @pytest.mark.asyncio
    async def test_releases_task_protection_at_end(self):
        drain = DrainMode()
        drain.register_source("fake", lambda: 0)
        calls = []
        protection = EcsTaskProtection(
            agent_uri="http://ecs-agent.local",
            http_call=lambda url, payload: calls.append(payload),
        )
        # Simulate protection held from steady-state activity.
        assert protection.set_protection_sync(True) is True
        drain.begin("test")
        await asyncio.wait_for(
            drain_coordinator_loop(
                drain,
                shutdown_cb=lambda: None,
                protection=protection,
                poll_interval=0.01,
                cap_seconds=60.0,
                settle_seconds=0.01,
                write_marker=False,
            ),
            timeout=5.0,
        )
        assert calls[-1] == {"ProtectionEnabled": False}
        assert protection.protected is False


# ---------------------------------------------------------------------------
# ECS task protection
# ---------------------------------------------------------------------------


class TestEcsTaskProtection:
    def test_disabled_outside_ecs_is_clean_noop(self, monkeypatch, caplog):
        monkeypatch.delenv("ECS_AGENT_URI", raising=False)
        protection = EcsTaskProtection()
        assert protection.enabled is False
        with caplog.at_level("INFO"):
            assert protection.set_protection_sync(True) is True
            assert protection.set_protection_sync(False) is True
        assert protection.protected is False
        noop_lines = [r for r in caplog.records if "no-op" in r.getMessage()]
        assert len(noop_lines) == 1  # logged once, not per call

    def test_set_and_release_payloads(self):
        calls = []
        protection = EcsTaskProtection(
            agent_uri="http://169.254.170.2/api/abc123",
            expires_minutes=15,
            http_call=lambda url, payload: calls.append((url, payload)),
        )
        assert protection.enabled is True
        assert protection.set_protection_sync(True) is True
        assert protection.protected is True
        assert protection.set_protection_sync(False) is True
        assert protection.protected is False
        assert calls == [
            (
                "http://169.254.170.2/api/abc123/task-protection/v1/state",
                {"ProtectionEnabled": True, "ExpiresInMinutes": 15},
            ),
            (
                "http://169.254.170.2/api/abc123/task-protection/v1/state",
                {"ProtectionEnabled": False},
            ),
        ]

    def test_failure_is_loud_but_never_raises(self, caplog):
        def _boom(url, payload):
            raise OSError("connection refused")

        protection = EcsTaskProtection(
            agent_uri="http://ecs-agent.local", http_call=_boom
        )
        with caplog.at_level("ERROR"):
            assert protection.set_protection_sync(True) is False
        assert protection.protected is False
        assert any(
            "task-protection PUT failed" in r.getMessage() for r in caplog.records
        )

    @pytest.mark.asyncio
    async def test_protection_loop_set_renew_release_ordering(self):
        drain = DrainMode()
        state = {"active": 1}
        drain.register_source("fake", lambda: state["active"])
        calls = []
        protection = EcsTaskProtection(
            agent_uri="http://ecs-agent.local",
            expires_minutes=15,
            http_call=lambda url, payload: calls.append(dict(payload)),
        )

        # Iteration 1: active -> SET.
        await task_protection_loop(
            drain, protection, check_interval=0.01, renew_seconds=300.0,
            max_iterations=1,
        )
        assert calls == [{"ProtectionEnabled": True, "ExpiresInMinutes": 15}]

        # Iteration 2: still active, renew window not reached -> no call.
        await task_protection_loop(
            drain, protection, check_interval=0.01, renew_seconds=300.0,
            max_iterations=1,
        )
        assert len(calls) == 1

        # Iteration 3: still active, renew window elapsed -> RENEW.
        protection._last_set_monotonic -= 301.0
        await task_protection_loop(
            drain, protection, check_interval=0.01, renew_seconds=300.0,
            max_iterations=1,
        )
        assert calls[-1] == {"ProtectionEnabled": True, "ExpiresInMinutes": 15}
        assert len(calls) == 2

        # Iteration 4: zero active runs -> RELEASE.
        state["active"] = 0
        await task_protection_loop(
            drain, protection, check_interval=0.01, renew_seconds=300.0,
            max_iterations=1,
        )
        assert calls[-1] == {"ProtectionEnabled": False}
        assert protection.protected is False

        # Iteration 5: zero and already released -> no further calls.
        await task_protection_loop(
            drain, protection, check_interval=0.01, renew_seconds=300.0,
            max_iterations=1,
        )
        assert len(calls) == 3

    @pytest.mark.asyncio
    async def test_protection_loop_noops_outside_ecs(self, monkeypatch):
        monkeypatch.delenv("ECS_AGENT_URI", raising=False)
        drain = DrainMode()
        drain.register_source("fake", lambda: 5)
        protection = EcsTaskProtection()
        # Returns immediately instead of looping forever.
        await asyncio.wait_for(
            task_protection_loop(drain, protection, check_interval=0.01),
            timeout=1.0,
        )

    @pytest.mark.asyncio
    async def test_protection_loop_survives_put_failures(self):
        drain = DrainMode()
        drain.register_source("fake", lambda: 1)

        def _boom(url, payload):
            raise OSError("connection refused")

        protection = EcsTaskProtection(
            agent_uri="http://ecs-agent.local", http_call=_boom
        )
        await task_protection_loop(
            drain, protection, check_interval=0.01, renew_seconds=300.0,
            max_iterations=3,
        )
        assert protection.protected is False  # failed, but loop kept running


# ---------------------------------------------------------------------------
# Global accessor
# ---------------------------------------------------------------------------


class TestGlobalAccessor:
    def test_get_drain_mode_is_singleton(self):
        assert get_drain_mode() is get_drain_mode()

    def test_reset_replaces_instance(self):
        first = get_drain_mode()
        first.begin("test")
        second = reset_drain_mode_for_tests()
        assert second is not first
        assert get_drain_mode() is second
        assert second.draining is False

    def test_refusal_contract_constants(self):
        assert DRAIN_REFUSAL_ERROR_CODE == "gateway_draining"
        assert drain_mode.DRAIN_REFUSAL_STATUS == 503
