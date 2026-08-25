"""Gateway drain mode: refuse new runs, finish in-flight, exit at zero (AE-117).

Drain-based blue/green deploys (parent-repo ADR 0177) need the OUTGOING
gateway task to stop accepting new work, keep executing its in-flight runs and
serving their established SSE streams, and exit on its own when the last run
finishes. Hermes runs execute purely in gateway process memory (see
``gateway/platforms/api_server.py`` ``_run_streams`` / ``_active_run_agents``
and ``_inflight_agent_runs``), so an invisible deploy requires the process
holding a run to live until the run finishes.

This module is deliberately additive (rebase durability): it owns all drain
state and the ECS coupling, and the rest of the gateway attaches through three
narrow hooks:

* ``gateway.run``'s SIGTERM handler calls :func:`sigterm_begins_drain` —
  when ``HERMES_DRAIN_ON_SIGTERM`` is truthy, the first unplanned SIGTERM
  begins a drain instead of the immediate stop path (ECS sends SIGTERM at
  task stop). A second SIGTERM while draining escalates to the normal
  immediate shutdown. Planned stops (``hermes gateway stop``, ``--replace``
  takeover, Ctrl+C) are untouched.
* ``GatewayRunner.start()`` calls :func:`start_drain_mode_tasks` to launch
  the coordinator (:func:`drain_coordinator_loop`) and the ECS task
  scale-in protection manager (:func:`task_protection_loop`).
* ``APIServerAdapter`` registers itself as an active-run *source* (count +
  force-terminate callbacks), refuses new launches with a 503
  ``{"error": {"code": "gateway_draining"}}`` while draining, and reports
  ``{draining, active_runs}`` on the readiness surface.

Relationship to upstream drain machinery
----------------------------------------

Upstream already carries two adjacent mechanisms, both reused rather than
duplicated:

* ``gateway/drain_control.py`` — the dashboard's *cancellable* relay-side
  quiesce driven by a ``.drain_request.json`` marker in HERMES_HOME. When a
  drain-mode drain begins we WRITE that marker (stamped with this
  instantiation's epoch, ``suppress_notification=True``) so the runner's
  existing ``_drain_control_watcher`` flips ``gateway_state -> draining``
  and refuses new relay turns for free. Drain mode itself is one-way: the
  marker is never removed by this process, and a marker that survives onto
  a NEW task (HERMES_HOME on shared/durable storage) is ignored there via
  the epoch check (NS-570 semantics).
* The shutdown drain in ``GatewayRunner.stop()`` (``_drain_active_agents``
  + ``HERMES_RESTART_DRAIN_TIMEOUT``) — drain mode ends by calling
  ``runner.stop()``, so the final teardown reuses the upstream graceful
  shutdown machinery unchanged.

Background work during a drain: new API and relay agent-run launches are
refused, while work admitted before the drain remains counted through its
whole execution. Context/session compression only starts inside a running
turn, so an in-flight turn may still compress as part of finishing (intended).
Cron jobs retain upstream scheduling behavior but are registered as runner
work from pre-dispatch through completion; the coordinator therefore waits
for them and its hard-cap path fences a forced run as interrupted.

Environment contract (all optional):

* ``HERMES_DRAIN_ON_SIGTERM`` — truthy: first unplanned SIGTERM begins a
  drain instead of stopping. Default: off (upstream behaviour).
* ``HERMES_DRAIN_CAP_SECONDS`` — hard cap on the drain window (default
  3600, matching the ALB idle timeout per ADR 0177). At the cap the
  remaining runs are interrupted so their streams receive clean terminal
  events, then the gateway exits.
* ``HERMES_DRAIN_SETTLE_SECONDS`` — quiet period required at zero active
  runs before self-exit (default 10) so established SSE consumers can
  finish reading already-queued terminal events.
* ``HERMES_DRAIN_FORCE_GRACE_SECONDS`` — bounded wait after the cap
  force-terminate for terminal events to flush (default 30).
* ``ECS_AGENT_URI`` — provided by the ECS agent inside a task; enables the
  task scale-in protection manager. Absent (local/dev/tests) the manager
  logs once and no-ops.
* ``HERMES_TASK_PROTECTION_EXPIRES_MINUTES`` — ECS protection expiry per
  PUT (default 15).
* ``HERMES_TASK_PROTECTION_RENEW_SECONDS`` — renew cadence while runs are
  active (default 300).
* ``HERMES_TASK_PROTECTION_HTTP_TIMEOUT_SECONDS`` — bounded ECS-agent request
  timeout (default 3; clamped to 0.1–10 seconds).
* ``HERMES_TASK_PROTECTION_FAILURE_BACKOFF_SECONDS`` — fail-closed admission
  backoff after a protection request fails (default 2).
"""
from __future__ import annotations

import asyncio
import concurrent.futures
import inspect
import json
import logging
import os
import threading
import time
import urllib.error
import urllib.request
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

_TASK_PROTECTION_EXECUTOR = concurrent.futures.ThreadPoolExecutor(
    max_workers=1,
    thread_name_prefix="hermes-task-protection",
)

# Programmatically detectable refusal contract: HTTP 503 with an OpenAI-style
# error envelope carrying this code. Clients (Hermes web, the agent-platform
# worker) match on ``error.code == "gateway_draining"`` and retry against a
# healthy task.
DRAIN_REFUSAL_ERROR_CODE = "gateway_draining"
DRAIN_REFUSAL_STATUS = 503

DEFAULT_DRAIN_CAP_SECONDS = 3600.0
DEFAULT_DRAIN_SETTLE_SECONDS = 10.0
DEFAULT_FORCE_GRACE_SECONDS = 30.0
DEFAULT_PROTECTION_EXPIRES_MINUTES = 15
DEFAULT_PROTECTION_RENEW_SECONDS = 300.0
DEFAULT_PROTECTION_CHECK_SECONDS = 15.0
DEFAULT_TASK_PROTECTION_HTTP_TIMEOUT_SECONDS = 3.0
DEFAULT_TASK_PROTECTION_FAILURE_BACKOFF_SECONDS = 2.0


def _positive_float_env(name: str, default: float) -> float:
    raw = os.getenv(name, "")
    if not str(raw).strip():
        return default
    try:
        value = float(str(raw).strip())
    except (TypeError, ValueError):
        logger.warning("drain-mode: invalid %s=%r, using default %.0f", name, raw, default)
        return default
    if value <= 0:
        logger.warning("drain-mode: non-positive %s=%r, using default %.0f", name, raw, default)
        return default
    return value


def drain_on_sigterm_enabled() -> bool:
    """True when SIGTERM should begin a drain instead of an immediate stop."""
    from utils import env_bool

    return env_bool("HERMES_DRAIN_ON_SIGTERM", default=False)


def drain_cap_seconds() -> float:
    return _positive_float_env("HERMES_DRAIN_CAP_SECONDS", DEFAULT_DRAIN_CAP_SECONDS)


def drain_settle_seconds() -> float:
    return _positive_float_env("HERMES_DRAIN_SETTLE_SECONDS", DEFAULT_DRAIN_SETTLE_SECONDS)


def drain_force_grace_seconds() -> float:
    return _positive_float_env("HERMES_DRAIN_FORCE_GRACE_SECONDS", DEFAULT_FORCE_GRACE_SECONDS)


class DrainMode:
    """Process-wide one-way drain latch plus active-run accounting registry.

    The API adapter and gateway runner (relay sessions plus cron, without
    overlap) each register a count callable and optional force-terminate
    callable keyed by name. Re-registering the same name replaces previous
    callbacks, so adapter reconnect loops that construct fresh instances never
    leave stale counts behind.

    Thread-safety: ``begin`` may be called from the asyncio signal handler,
    an HTTP handler, or tests; registration happens at adapter construction.
    A plain lock keeps the registry and the latch consistent. Count callables
    themselves must be cheap and non-blocking (they read in-memory dicts).
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._draining = False
        self._reason: Optional[str] = None
        self._started_wall: Optional[float] = None
        self._started_mono: Optional[float] = None
        self._sources: Dict[str, Tuple[Callable[[], int], Optional[Callable[[str], int]]]] = {}
        self._last_logged_active: Optional[int] = None

    # -- latch ---------------------------------------------------------

    @property
    def draining(self) -> bool:
        return self._draining

    @property
    def reason(self) -> Optional[str]:
        return self._reason

    @property
    def started_at(self) -> Optional[float]:
        """Wall-clock (time.time) drain start, or None."""
        return self._started_wall

    def begin(self, reason: str) -> bool:
        """Engage drain mode. One-way: once draining, never un-drain.

        Returns True on the engaging call, False when already draining.
        """
        with self._lock:
            if self._draining:
                logger.info(
                    "drain-mode: begin(%s) ignored — already draining (reason=%s, %.1fs ago)",
                    reason, self._reason, self.elapsed_seconds() or 0.0,
                )
                return False
            self._draining = True
            self._reason = reason
            self._started_wall = time.time()
            self._started_mono = time.monotonic()
        logger.warning(
            "drain-mode: ENGAGED (reason=%s, active_runs=%d) — refusing new runs, "
            "finishing in-flight work, will exit at zero",
            reason, self.active_runs(),
        )
        return True

    def elapsed_seconds(self) -> Optional[float]:
        started = self._started_mono
        if started is None:
            return None
        return time.monotonic() - started

    # -- active-run accounting ------------------------------------------

    def register_source(
        self,
        name: str,
        count_fn: Callable[[], int],
        force_terminate_fn: Optional[Callable[[str], int]] = None,
    ) -> None:
        with self._lock:
            self._sources[name] = (count_fn, force_terminate_fn)

    def unregister_source(self, name: str) -> None:
        with self._lock:
            self._sources.pop(name, None)

    def active_runs(self) -> int:
        """Sum registered sources, conservatively counting a failed source."""
        with self._lock:
            sources = list(self._sources.items())
        total = 0
        for name, (count_fn, _term) in sources:
            try:
                total += max(0, int(count_fn()))
            except Exception:
                # A counter failure is not evidence of idleness. Counting one
                # keeps task protection and the drain wait fail-closed until a
                # later poll succeeds (or the existing hard cap intervenes).
                total += 1
                logger.error("drain-mode: active-run source %s failed", name, exc_info=True)
        return total

    def force_terminate_all(self, reason: str) -> int:
        """Interrupt every remaining run so streams see clean terminal events."""
        with self._lock:
            sources = list(self._sources.items())
        terminated = 0
        for name, (_count, term_fn) in sources:
            if term_fn is None:
                continue
            try:
                terminated += int(term_fn(reason) or 0)
            except Exception:
                logger.error(
                    "drain-mode: force-terminate source %s failed", name, exc_info=True,
                )
        return terminated

    def log_active_transition(self, active: int) -> None:
        """Log drain progress whenever the active-run count changes."""
        if active != self._last_logged_active:
            logger.info(
                "drain-mode: draining — active_runs=%d (elapsed %.1fs)",
                active, self.elapsed_seconds() or 0.0,
            )
            self._last_logged_active = active

    def snapshot(self) -> Dict[str, Any]:
        """Readiness/health payload fragment for this drain state."""
        return {
            "draining": self._draining,
            "active_runs": self.active_runs(),
            "drain_reason": self._reason,
            "drain_started_at": self._started_wall,
            "drain_cap_seconds": drain_cap_seconds(),
        }


_drain_mode: Optional[DrainMode] = None
_drain_mode_lock = threading.Lock()
_task_protection: Optional["EcsTaskProtection"] = None
_task_protection_lock = threading.Lock()
_task_protection_wakeup_lock = threading.Lock()
_task_protection_wakeup: Optional[Tuple[Any, "asyncio.Event"]] = None


def get_drain_mode() -> DrainMode:
    """Process-global drain state (created lazily)."""
    global _drain_mode
    with _drain_mode_lock:
        if _drain_mode is None:
            _drain_mode = DrainMode()
        return _drain_mode


def reset_drain_mode_for_tests() -> DrainMode:
    """Replace global drain/protection state. Test-only."""
    global _drain_mode, _task_protection, _task_protection_wakeup
    with _drain_mode_lock:
        _drain_mode = DrainMode()
    with _task_protection_lock:
        _task_protection = None
    with _task_protection_wakeup_lock:
        _task_protection_wakeup = None
    return _drain_mode


def sigterm_begins_drain() -> bool:
    """SIGTERM hook for ``gateway.run``'s shutdown signal handler.

    Returns True when the signal was absorbed as a drain trigger (the handler
    must return without stopping). Returns False when SIGTERM should keep its
    normal immediate-shutdown meaning: the feature is disabled, or a drain is
    already in progress (second SIGTERM escalates).
    """
    if not drain_on_sigterm_enabled():
        return False
    drain = get_drain_mode()
    if drain.draining:
        logger.warning(
            "drain-mode: SIGTERM received while already draining "
            "(active_runs=%d, elapsed %.1fs) — escalating to immediate shutdown",
            drain.active_runs(), drain.elapsed_seconds() or 0.0,
        )
        return False
    drain.begin("sigterm")
    return True


# ---------------------------------------------------------------------------
# ECS task scale-in protection
# ---------------------------------------------------------------------------


class EcsTaskProtection:
    """Task scale-in protection client for the in-task ECS agent endpoint.

    Calls ``PUT $ECS_AGENT_URI/task-protection/v1/state`` with
    ``{"ProtectionEnabled": bool, "ExpiresInMinutes": N}``. Outside ECS
    (``ECS_AGENT_URI`` unset) every call is a clean no-op, logged once.
    Failures never raise past this class — losing protection must not crash
    the gateway; it is logged loudly instead.

    ``http_call`` is injectable for tests: ``(url, payload_dict) -> None``,
    raising on failure.
    """

    def __init__(
        self,
        agent_uri: Optional[str] = None,
        expires_minutes: Optional[int] = None,
        http_call: Optional[Callable[[str, Dict[str, Any]], None]] = None,
    ) -> None:
        self._agent_uri = (
            agent_uri if agent_uri is not None else os.getenv("ECS_AGENT_URI", "")
        ).strip().rstrip("/")
        if expires_minutes is None:
            expires_minutes = int(
                _positive_float_env(
                    "HERMES_TASK_PROTECTION_EXPIRES_MINUTES",
                    DEFAULT_PROTECTION_EXPIRES_MINUTES,
                )
            )
        self._expires_minutes = max(1, int(expires_minutes))
        self._http_timeout_seconds = min(
            10.0,
            max(
                0.1,
                _positive_float_env(
                    "HERMES_TASK_PROTECTION_HTTP_TIMEOUT_SECONDS",
                    DEFAULT_TASK_PROTECTION_HTTP_TIMEOUT_SECONDS,
                ),
            ),
        )
        self._failure_backoff_seconds = max(
            0.0,
            _positive_float_env(
                "HERMES_TASK_PROTECTION_FAILURE_BACKOFF_SECONDS",
                DEFAULT_TASK_PROTECTION_FAILURE_BACKOFF_SECONDS,
            ),
        )
        self._http_call = http_call or self._default_http_call
        self._protected = False
        self._last_set_monotonic = 0.0
        self._last_protect_failure_monotonic = 0.0
        # Incremented under _operation_lock for every admitted work item,
        # including admissions that reuse a fresh SET acknowledgement. An idle
        # snapshot taken on the asyncio owner thread carries this fence into
        # the worker that performs the blocking ECS PUT.
        self._admission_epoch = 0
        self._noop_logged = False
        # Admission, renewal, and release all share this client. Serialize the
        # state transition plus its ECS acknowledgement so concurrent 0 -> 1
        # admissions single-flight their PUT and a stale idle reconciliation
        # cannot release protection after newly admitted work appears.
        self._operation_lock = threading.Lock()

    @property
    def enabled(self) -> bool:
        return bool(self._agent_uri)

    @property
    def protected(self) -> bool:
        """Last state successfully acknowledged by the ECS agent."""
        return self._protected

    @property
    def last_set_monotonic(self) -> float:
        return self._last_set_monotonic

    @property
    def admission_epoch(self) -> int:
        """Monotonic fence for work admitted since an idle snapshot."""
        return self._admission_epoch

    @property
    def safe_refresh_seconds(self) -> float:
        """Renew by half the ECS expiry even if polling is misconfigured."""
        return max(1.0, float(self._expires_minutes) * 30.0)

    @property
    def endpoint(self) -> str:
        return f"{self._agent_uri}/task-protection/v1/state"

    def _default_http_call(self, url: str, payload: Dict[str, Any]) -> None:
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=data,
            method="PUT",
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(  # nosec B310 - link-local ECS agent URI from the task metadata contract
            req, timeout=self._http_timeout_seconds
        ) as resp:
            resp.read()

    def _log_noop_once(self) -> None:
        if not self._noop_logged:
            logger.info(
                "drain-mode: ECS task scale-in protection disabled — "
                "ECS_AGENT_URI is not set (not running on ECS); protection "
                "calls are no-ops",
            )
            self._noop_logged = True

    def _set_protection_locked(self, protect: bool) -> bool:
        """Perform one blocking PUT while ``_operation_lock`` is held."""
        now = time.monotonic()
        if (
            protect
            and self._last_protect_failure_monotonic > 0
            and now - self._last_protect_failure_monotonic
            < self._failure_backoff_seconds
        ):
            logger.debug(
                "drain-mode: task-protection admission remains in %.1fs "
                "failure backoff; refusing without another blocking PUT",
                self._failure_backoff_seconds,
            )
            return False
        payload: Dict[str, Any] = {"ProtectionEnabled": bool(protect)}
        if protect:
            payload["ExpiresInMinutes"] = self._expires_minutes
        try:
            self._http_call(self.endpoint, payload)
        except Exception as exc:
            # A transport failure is ambiguous: ECS may have applied the PUT
            # and lost only the response.  In particular, retaining a prior
            # ``True`` after an ambiguous release lets the next admission
            # reuse a stale acknowledgement even though ECS may now be
            # unprotected.  Collapse unknown to the fail-closed local state so
            # every subsequent admission must obtain a fresh SET response.
            self._protected = False
            self._last_set_monotonic = 0.0
            if protect:
                self._last_protect_failure_monotonic = time.monotonic()
            # Loud but non-fatal: without protection ECS may scale this task
            # in mid-run, but crashing the gateway here would kill the runs
            # immediately and for certain.
            logger.error(
                "drain-mode: ECS task-protection PUT failed "
                "(ProtectionEnabled=%s, endpoint=%s): %s",
                protect, self.endpoint, exc,
            )
            return False
        self._last_protect_failure_monotonic = 0.0
        self._protected = bool(protect)
        self._last_set_monotonic = time.monotonic()
        logger.info(
            "drain-mode: ECS task scale-in protection %s (expires_in_minutes=%s)",
            "SET" if protect else "RELEASED",
            self._expires_minutes if protect else None,
        )
        return True

    def set_protection_sync(self, protect: bool) -> bool:
        """Blocking PUT. Returns True on success (or non-ECS no-op)."""
        with self._operation_lock:
            if not self.enabled:
                self._log_noop_once()
                return True
            return self._set_protection_locked(protect)

    def ensure_protected_sync(self) -> bool:
        """Acknowledge protection once, coalescing concurrent admissions."""
        with self._operation_lock:
            if not self.enabled:
                self._log_noop_once()
                return True
            self._admission_epoch += 1
            fresh = (
                time.monotonic() - self._last_set_monotonic
                < self.safe_refresh_seconds
            )
            if self._protected and fresh:
                return True
            return self._set_protection_locked(True)

    def reconcile_sync(
        self,
        active_count: int,
        *,
        renew_seconds: float,
        observed_admission_epoch: Optional[int] = None,
    ) -> bool:
        """Reconcile an owner-thread work snapshot with ECS protection.

        The potentially blocking PUT stays off the event loop, but callbacks
        that inspect asyncio tasks/dicts do not. ``observed_admission_epoch``
        closes the handoff: if an admission acquired this transition lock
        after the idle snapshot began, a stale worker must not release the
        acknowledgement that admission relied on.
        """
        with self._operation_lock:
            if not self.enabled:
                self._log_noop_once()
                return True
            active = max(0, int(active_count))
            if active > 0:
                effective_renew_seconds = min(
                    renew_seconds,
                    self.safe_refresh_seconds,
                )
                stale = (
                    time.monotonic() - self._last_set_monotonic
                    >= effective_renew_seconds
                )
                if not self._protected or stale:
                    return self._set_protection_locked(True)
                return True
            if (
                observed_admission_epoch is not None
                and observed_admission_epoch != self._admission_epoch
            ):
                return True
            if self._protected:
                return self._set_protection_locked(False)
            return True

    async def set_protection(self, protect: bool) -> bool:
        if not self.enabled:
            self._log_noop_once()
            return True
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            _TASK_PROTECTION_EXECUTOR,
            self.set_protection_sync,
            protect,
        )

    async def ensure_protected(self) -> bool:
        if not self.enabled:
            self._log_noop_once()
            return True
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            _TASK_PROTECTION_EXECUTOR,
            self.ensure_protected_sync,
        )

    async def reconcile(
        self,
        active_count: Callable[[], int],
        *,
        renew_seconds: float,
    ) -> bool:
        if not self.enabled:
            self._log_noop_once()
            return True
        # Snapshot the epoch first. Every admission publishes its work count
        # before waiting on ensure_protected_sync(), so any admission that
        # races after this point is either visible in ``active`` or advances
        # the epoch before the worker can release protection.
        observed_admission_epoch = self.admission_epoch
        try:
            active = max(0, int(active_count()))
        except Exception:
            logger.error(
                "drain-mode: task-protection active-work snapshot failed",
                exc_info=True,
            )
            active = 1
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            _TASK_PROTECTION_EXECUTOR,
            lambda: self.reconcile_sync(
                active,
                renew_seconds=renew_seconds,
                observed_admission_epoch=observed_admission_epoch,
            ),
        )


def get_task_protection() -> EcsTaskProtection:
    """Process-global client shared by admission, renewal, and release."""
    global _task_protection
    with _task_protection_lock:
        if _task_protection is None:
            _task_protection = EcsTaskProtection()
        return _task_protection


class TaskProtectionUnavailableError(RuntimeError):
    """New work was rolled back because ECS protection was not acknowledged."""


def ensure_task_protection_for_admission_sync() -> bool:
    """Synchronous admission gate for thread-pool work such as cron jobs."""
    # This shared seam is used by API, relay/startup, and cron. Re-check the
    # one-way latch here (not only at each transport) so no internal/manual
    # source can repopulate work during the coordinator's final zero settle.
    # Every caller publishes its pending/running count before reaching this
    # gate, so an admission already in progress when drain begins remains
    # visible until it either receives protection or rolls itself back.
    if get_drain_mode().draining:
        return False
    return get_task_protection().ensure_protected_sync()


def require_task_protection_for_admission_sync() -> None:
    """Raise a stable error when synchronous work cannot be protected."""
    if not ensure_task_protection_for_admission_sync():
        raise TaskProtectionUnavailableError(
            "ECS task protection was not acknowledged; work was not started"
        )


async def ensure_task_protection_for_admission() -> bool:
    """Protect this ECS task before newly admitted agent work may execute."""
    if get_drain_mode().draining:
        return False
    return await get_task_protection().ensure_protected()


def notify_task_protection_work_changed() -> None:
    """Wake protection reconciliation after an admission/release transition."""
    with _task_protection_wakeup_lock:
        target = _task_protection_wakeup
    if target is None:
        return
    loop, wakeup = target
    try:
        loop.call_soon_threadsafe(wakeup.set)
    except RuntimeError:
        # The loop is already closed during interpreter/gateway teardown. The
        # expiry on ECS protection remains the final fail-safe.
        return


async def task_protection_loop(
    drain: DrainMode,
    protection: EcsTaskProtection,
    *,
    check_interval: Optional[float] = None,
    renew_seconds: Optional[float] = None,
    max_iterations: Optional[int] = None,
) -> None:
    """Keep ECS scale-in protection aligned with active-run existence.

    Protection is held whenever active runs exist — steady state AND during a
    drain — renewed every ``renew_seconds`` (default 300) with a
    ``HERMES_TASK_PROTECTION_EXPIRES_MINUTES`` (default 15) expiry so a hung
    gateway can never hold protection forever. It is released promptly when
    the active-run count reaches zero. Outside ECS the loop logs once and
    exits. ``max_iterations`` is a test hook.
    """
    if not protection.enabled:
        protection._log_noop_once()
        return
    if check_interval is None:
        check_interval = _positive_float_env(
            "HERMES_TASK_PROTECTION_CHECK_SECONDS", DEFAULT_PROTECTION_CHECK_SECONDS
        )
    if renew_seconds is None:
        renew_seconds = _positive_float_env(
            "HERMES_TASK_PROTECTION_RENEW_SECONDS", DEFAULT_PROTECTION_RENEW_SECONDS
        )
    loop = asyncio.get_running_loop()
    wakeup = asyncio.Event()
    global _task_protection_wakeup
    with _task_protection_wakeup_lock:
        _task_protection_wakeup = (loop, wakeup)
    iterations = 0
    try:
        while True:
            # Clearing before reconciliation is race-safe: a state change that
            # happened before this point is observed by active_runs(); one
            # during/after reconciliation sets the event for an immediate
            # second pass.
            wakeup.clear()
            try:
                await protection.reconcile(
                    drain.active_runs,
                    renew_seconds=renew_seconds,
                )
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.error(
                    "drain-mode: task-protection loop iteration failed",
                    exc_info=True,
                )
            iterations += 1
            if max_iterations is not None and iterations >= max_iterations:
                return
            try:
                await asyncio.wait_for(wakeup.wait(), timeout=check_interval)
            except asyncio.TimeoutError:
                pass
    finally:
        with _task_protection_wakeup_lock:
            if _task_protection_wakeup == (loop, wakeup):
                _task_protection_wakeup = None


# ---------------------------------------------------------------------------
# Drain coordinator
# ---------------------------------------------------------------------------


async def drain_coordinator_loop(
    drain: DrainMode,
    *,
    shutdown_cb: Callable[[], Any],
    protection: Optional[EcsTaskProtection] = None,
    poll_interval: float = 1.0,
    cap_seconds: Optional[float] = None,
    settle_seconds: Optional[float] = None,
    force_grace_seconds: Optional[float] = None,
    write_marker: bool = True,
) -> None:
    """Own the drain lifecycle: wait for engage, watch to zero, exit.

    Once ``drain.begin()`` fires (SIGTERM hook, POST /admin/drain, or tests):

    1. Best-effort write the upstream ``.drain_request.json`` marker so the
       runner's existing drain-control watcher quiesces the relay side and
       flips ``gateway_state -> draining`` (readiness then reports 503).
    2. Poll the active-run count, logging every transition.
    3. Zero active runs sustained for ``settle_seconds`` -> release task
       protection and invoke ``shutdown_cb`` (normally ``runner.stop()``,
       i.e. the upstream graceful-shutdown machinery).
    4. ``cap_seconds`` exceeded -> force-terminate remaining runs (each
       stream receives its normal clean terminal event via the interrupt
       path), wait up to ``force_grace_seconds`` for them to flush, then
       shut down anyway.
    """
    while not drain.draining:
        await asyncio.sleep(poll_interval)

    cap = cap_seconds if cap_seconds is not None else drain_cap_seconds()
    settle = settle_seconds if settle_seconds is not None else drain_settle_seconds()
    force_grace = (
        force_grace_seconds
        if force_grace_seconds is not None
        else drain_force_grace_seconds()
    )

    if write_marker:
        try:
            from gateway.drain_control import write_drain_request

            write_drain_request(
                principal=f"drain-mode:{drain.reason or 'unknown'}",
                suppress_notification=True,
            )
        except Exception:
            logger.warning(
                "drain-mode: failed to write relay drain marker (relay-side "
                "turns may still be accepted)", exc_info=True,
            )

    logger.warning(
        "drain-mode: coordinator active (reason=%s, cap=%.0fs, settle=%.0fs)",
        drain.reason, cap, settle,
    )

    forced = False
    while True:
        active = drain.active_runs()
        drain.log_active_transition(active)

        if active <= 0:
            # Settle window: hold at zero so established SSE consumers can
            # finish reading already-queued terminal events before the HTTP
            # server goes away. New runs cannot appear (launches are refused)
            # but re-check anyway — fail toward staying alive.
            settle_deadline = time.monotonic() + settle
            still_zero = True
            while time.monotonic() < settle_deadline:
                await asyncio.sleep(min(poll_interval, settle))
                if drain.active_runs() > 0:
                    still_zero = False
                    break
            if still_zero:
                break
            continue

        if not forced and (drain.elapsed_seconds() or 0.0) >= cap:
            forced = True
            logger.error(
                "drain-mode: drain cap %.0fs exceeded with %d active run(s) — "
                "force-terminating remaining runs so streams end cleanly",
                cap, active,
            )
            terminated = drain.force_terminate_all(
                "Gateway drain cap reached — run terminated before shutdown"
            )
            logger.warning(
                "drain-mode: force-terminate interrupted %d run(s); waiting up "
                "to %.0fs for terminal events to flush", terminated, force_grace,
            )
            grace_deadline = time.monotonic() + force_grace
            while time.monotonic() < grace_deadline and drain.active_runs() > 0:
                await asyncio.sleep(poll_interval)
            break

        await asyncio.sleep(poll_interval)

    if protection is not None and protection.enabled:
        try:
            await protection.set_protection(False)
        except Exception:
            logger.error(
                "drain-mode: failed to release task protection at drain end",
                exc_info=True,
            )

    logger.warning(
        "drain-mode: drain complete (reason=%s, elapsed=%.1fs, active_runs=%d, "
        "forced=%s) — shutting down",
        drain.reason, drain.elapsed_seconds() or 0.0, drain.active_runs(), forced,
    )
    result = shutdown_cb()
    if inspect.isawaitable(result):
        await result


_module_retained_tasks: set = set()


def start_drain_mode_tasks(runner: Any) -> List["asyncio.Task"]:
    """Launch the drain coordinator + ECS protection manager for a runner.

    Called once from ``GatewayRunner.start()``. Tasks are retained in the
    runner's ``_background_tasks`` set (so they are cancelled by the normal
    shutdown path — the coordinator schedules ``runner.stop()`` as its own
    task rather than awaiting it, so cancelling the coordinator during stop
    cannot wedge the stop).
    """
    drain = get_drain_mode()
    protection = get_task_protection()
    # APIServerAdapter owns its own non-overlapping source. Relay turns and
    # cron jobs live on GatewayRunner and were previously invisible here,
    # which let a relay-only task look idle to both ECS protection and the
    # drain coordinator for its entire lifetime.
    drain.register_source(
        "gateway_runner",
        lambda: max(0, int(runner._running_agent_count()))
        + max(0, int(runner._active_cron_job_count())),
        getattr(runner, "_drain_mode_force_terminate", None),
    )

    def _shutdown() -> None:
        stop_task = asyncio.get_running_loop().create_task(runner.stop())
        # Retain a strong reference so the stop task can't be GC'd mid-flight
        # (it is deliberately NOT placed in runner._background_tasks: stop()
        # cancels that set, and while _stop_impl itself is shielded via
        # _stop_task, keeping the wrapper out of the set avoids the churn).
        _module_retained_tasks.add(stop_task)
        stop_task.add_done_callback(_module_retained_tasks.discard)

    tasks = [
        asyncio.create_task(
            drain_coordinator_loop(drain, shutdown_cb=_shutdown, protection=protection),
            name="drain-mode-coordinator",
        ),
        asyncio.create_task(
            task_protection_loop(drain, protection),
            name="drain-mode-task-protection",
        ),
    ]
    retained = getattr(runner, "_background_tasks", None)
    if retained is not None:
        for task in tasks:
            try:
                retained.add(task)
                task.add_done_callback(retained.discard)
            except (TypeError, AttributeError):
                pass
    logger.info(
        "drain-mode: coordinator started (sigterm_drain=%s, cap=%.0fs, "
        "task_protection=%s)",
        drain_on_sigterm_enabled(), drain_cap_seconds(),
        "enabled" if protection.enabled else "disabled (no ECS_AGENT_URI)",
    )
    return tasks
