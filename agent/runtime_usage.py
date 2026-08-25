"""Run-scoped token attribution for primary, auxiliary, and delegated calls.

Hermes historically exposed only the primary ``AIAgent`` counters in gateway
response envelopes.  Auxiliary ``call_llm`` work and delegated sessions are
real model calls, but live outside those counters.  This module keeps the
three buckets separate and produces one aggregate with an explicit
completeness contract.

The accounting context is a ``ContextVar`` because one gateway process
executes many agents concurrently. ``AIAgent.run_conversation`` binds the
correct agent, Hermes's existing context-copy wrapper carries that binding
into concurrent tool workers, and each ``delegate_task`` child binds itself in
its own execution thread. Provider response objects are identity-deduplicated so a
validation/recovery path cannot count the same response twice; distinct retry
responses remain distinct spend and are counted once each when they expose
usage.
"""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime, timezone
import logging
import threading
import time
from typing import Any, Callable, Dict, Iterator, Mapping, Optional, TypeVar
import uuid


_ACTIVE_AGENT: ContextVar[Any] = ContextVar(
    "hermes_runtime_usage_active_agent",
    default=None,
)
_AUXILIARY_ATTEMPT_OBSERVER: ContextVar[Optional[Callable[..., None]]] = ContextVar(
    "hermes_runtime_usage_auxiliary_attempt_observer",
    default=None,
)
_PRIMARY_ACCOUNTED_AUXILIARY_TASKS = frozenset({"moa_reference", "moa_aggregator"})
_VALID_COMPLETENESS = frozenset({"complete", "partial", "unavailable"})
_T = TypeVar("_T")
logger = logging.getLogger(__name__)


def initialize_agent_usage_attribution(agent: Any) -> None:
    """Initialize additive attribution counters on an AIAgent-like object."""

    agent.session_auxiliary_input_tokens = 0
    agent.session_auxiliary_output_tokens = 0
    agent.session_auxiliary_total_tokens = 0
    agent.session_auxiliary_response_count = 0
    agent.session_delegated_input_tokens = 0
    agent.session_delegated_output_tokens = 0
    agent.session_delegated_total_tokens = 0
    agent.session_delegated_response_count = 0
    # Provider responses that returned from the primary dispatch seam but have
    # not reached the normal successful-response accounting block. These are
    # most commonly malformed/terminal responses followed by a retry. Keeping
    # them separate lets the final parent bucket include known retry spend,
    # while ``commit_primary_response`` atomically removes the final valid
    # response before adding its canonical counters (no double count).
    agent.session_primary_retry_input_tokens = 0
    agent.session_primary_retry_output_tokens = 0
    agent.session_primary_retry_total_tokens = 0
    agent.session_primary_retry_response_count = 0
    agent.session_primary_usage_missing_count = 0
    agent._runtime_usage_warnings = set()
    # Keep strong references for the run lifetime. A bare set of id(response)
    # is unsafe because CPython may reuse an id after GC for a distinct retry
    # response, silently undercounting it.
    agent._runtime_usage_seen_aux_response_objects = {}
    agent._runtime_usage_pending_primary_response_objects = {}
    agent._runtime_usage_committed_primary_response_objects = {}
    agent._runtime_usage_lock = threading.Lock()
    agent._runtime_usage_initialized = True


@contextmanager
def attribute_auxiliary_usage(agent: Any) -> Iterator[None]:
    """Attribute central ``call_llm`` responses on this thread to *agent*."""

    token = _ACTIVE_AGENT.set(agent)
    try:
        yield
    finally:
        _ACTIVE_AGENT.reset(token)


@contextmanager
def observe_auxiliary_attempts(observer: Callable[..., None]) -> Iterator[None]:
    """Observe each provider dispatch inside one auxiliary operation.

    MoA reference calls run in their own raw executor threads and are folded
    into the primary bucket by ``moa_loop`` rather than the central auxiliary
    bucket. The observer gives that loop the exact response objects (including
    invalid responses later retried by ``call_llm``) plus conservative
    dispatch-uncertainty signals without leaking an agent ContextVar across
    reference threads.
    """

    token = _AUXILIARY_ATTEMPT_OBSERVER.set(observer)
    try:
        yield
    finally:
        _AUXILIARY_ATTEMPT_OBSERVER.reset(token)


def track_primary_dispatch(agent: Any, dispatch: Callable[[], _T]) -> _T:
    """Execute one already-preflighted primary provider dispatch.

    This wrapper belongs immediately around the function that begins request
    execution, not request construction. A thrown/abandoned attempt may have
    reached the provider without returning terminal usage, so it makes the run
    partial. A returned response is retained and provisionally counted until
    the normal accounting path commits it.
    """

    interrupted_before_dispatch = bool(
        getattr(agent, "_interrupt_requested", False)
    )
    try:
        response = dispatch()
    except BaseException as exc:
        # The interruptible transports reject an already-cancelled turn before
        # opening a request. Do not fabricate provider uncertainty in that one
        # provably pre-dispatch case; interrupts raised after execution starts
        # remain partial.
        if not (isinstance(exc, InterruptedError) and interrupted_before_dispatch):
            _best_effort_mark_primary_dispatch_uncertain(
                agent,
                reason="dispatched_attempt_usage_unavailable",
            )
        raise
    try:
        record_primary_dispatch_response(agent, response)
    except Exception:
        logger.debug("Primary response usage accounting failed", exc_info=True)
        _best_effort_mark_primary_dispatch_uncertain(
            agent,
            reason="usage_accounting_failed",
        )
    return response


def track_auxiliary_dispatch(
    dispatch: Callable[[], _T],
    *,
    task: Optional[str] = None,
    provider: Optional[str] = None,
    model: Optional[str] = None,
    base_url: Optional[str] = None,
    api_mode: Optional[str] = None,
    request: Any = None,
) -> _T:
    """Execute and attribute one synchronous auxiliary provider attempt."""

    observer = _begin_auxiliary_observer_attempt(
        task=task,
        provider=provider,
        model=model,
        base_url=base_url,
        api_mode=api_mode,
        request=request,
    )
    try:
        response = dispatch()
    except BaseException as exc:
        _end_auxiliary_observer_attempt(observer, error=exc)
        _notify_auxiliary_attempt_observer(
            response=None,
            reason="dispatched_attempt_usage_unavailable",
        )
        _best_effort_mark_auxiliary_dispatch_uncertain(
            task=task,
            reason="dispatched_attempt_usage_unavailable",
        )
        raise
    _notify_auxiliary_attempt_observer(response=response, reason=None)
    try:
        _record_tracked_auxiliary_response(response, task=task)
    except Exception:
        logger.debug("Auxiliary response usage accounting failed", exc_info=True)
        _best_effort_mark_auxiliary_dispatch_uncertain(
            task=task,
            reason="usage_accounting_failed",
        )
    _end_auxiliary_observer_attempt(observer, response=response)
    return response


class _ObservedAuxiliaryStream:
    """Transparent iterator proxy that terminalizes telemetry on consumption."""

    def __init__(self, stream: Any, observer: Optional[Dict[str, Any]], *, task: Optional[str]):
        self._stream = stream
        self._iterator = iter(stream)
        self._observer = observer
        self._task = task
        self._last_chunk = None
        self._terminal = False

    def __iter__(self):
        return self

    def __next__(self):
        try:
            chunk = next(self._iterator)
        except StopIteration:
            self._finish(response=self._last_chunk)
            raise
        except BaseException as exc:
            self._finish(error=exc)
            raise
        self._last_chunk = chunk
        return chunk

    def _finish(self, *, response: Any = None, error: Optional[BaseException] = None) -> None:
        if self._terminal:
            return
        self._terminal = True
        _end_auxiliary_observer_attempt(self._observer, response=response, error=error)
        if error is None and response is not None:
            try:
                _record_tracked_auxiliary_response(response, task=self._task)
            except Exception:
                logger.debug("Auxiliary stream usage accounting failed", exc_info=True)

    def close(self):
        close = getattr(self._stream, "close", None)
        try:
            if callable(close):
                return close()
        finally:
            if not self._terminal:
                self._finish(error=RuntimeError("auxiliary stream closed before terminal chunk"))

    def __enter__(self):
        enter = getattr(self._stream, "__enter__", None)
        if callable(enter):
            enter()
        return self

    def __exit__(self, exc_type, exc, traceback):
        if exc is not None:
            self._finish(error=exc)
        self.close()
        return False

    def __getattr__(self, name: str) -> Any:
        return getattr(self._stream, name)


def track_auxiliary_stream_dispatch(
    dispatch: Callable[[], Any],
    *,
    task: Optional[str] = None,
    provider: Optional[str] = None,
    model: Optional[str] = None,
    base_url: Optional[str] = None,
    api_mode: Optional[str] = None,
    request: Any = None,
    completed_response_predicate: Optional[Callable[[Any], bool]] = None,
) -> Any:
    """Dispatch a stream and keep its observer open until consume/close/error.

    Some OpenAI-compatible adapters ignore ``stream=True`` and return an
    already-completed response.  Callers that support that wire behavior may
    provide ``completed_response_predicate``; a matching response is accounted
    and observed immediately, then returned unchanged instead of being wrapped
    as an iterator.
    """
    observer = _begin_auxiliary_observer_attempt(
        task=task,
        provider=provider,
        model=model,
        base_url=base_url,
        api_mode=api_mode,
        request=request,
    )
    try:
        stream = dispatch()
    except BaseException as exc:
        _end_auxiliary_observer_attempt(observer, error=exc)
        raise
    if (
        completed_response_predicate is not None
        and completed_response_predicate(stream)
    ):
        _notify_auxiliary_attempt_observer(response=stream, reason=None)
        try:
            _record_tracked_auxiliary_response(stream, task=task)
        except Exception:
            logger.debug(
                "Auxiliary completed-stream usage accounting failed",
                exc_info=True,
            )
            _best_effort_mark_auxiliary_dispatch_uncertain(
                task=task,
                reason="usage_accounting_failed",
            )
        _end_auxiliary_observer_attempt(observer, response=stream)
        return stream
    return _ObservedAuxiliaryStream(stream, observer, task=task)


async def track_auxiliary_dispatch_async(
    dispatch: Callable[[], Any],
    *,
    task: Optional[str] = None,
    provider: Optional[str] = None,
    model: Optional[str] = None,
    base_url: Optional[str] = None,
    api_mode: Optional[str] = None,
    request: Any = None,
) -> Any:
    """Execute and attribute one asynchronous auxiliary provider attempt."""

    observer = _begin_auxiliary_observer_attempt(
        task=task,
        provider=provider,
        model=model,
        base_url=base_url,
        api_mode=api_mode,
        request=request,
    )
    try:
        response = await dispatch()
    except BaseException as exc:
        _end_auxiliary_observer_attempt(observer, error=exc)
        _notify_auxiliary_attempt_observer(
            response=None,
            reason="dispatched_attempt_usage_unavailable",
        )
        _best_effort_mark_auxiliary_dispatch_uncertain(
            task=task,
            reason="dispatched_attempt_usage_unavailable",
        )
        raise
    _notify_auxiliary_attempt_observer(response=response, reason=None)
    try:
        _record_tracked_auxiliary_response(response, task=task)
    except Exception:
        logger.debug("Async auxiliary response usage accounting failed", exc_info=True)
        _best_effort_mark_auxiliary_dispatch_uncertain(
            task=task,
            reason="usage_accounting_failed",
        )
    _end_auxiliary_observer_attempt(observer, response=response)
    return response


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _observer_jsonable(value: Any, *, depth: int = 0) -> Any:
    """Best-effort JSON projection for observer payloads.

    The downstream adapter owns policy-level capture and bounds. This narrow
    runtime projection only prevents provider SDK objects from escaping into
    hooks as opaque or unserialisable values.
    """

    if depth >= 8:
        return "<max-depth>"
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Mapping):
        return {
            str(key): _observer_jsonable(item, depth=depth + 1)
            for key, item in list(value.items())[:200]
        }
    if isinstance(value, (list, tuple, set)):
        return [_observer_jsonable(item, depth=depth + 1) for item in list(value)[:200]]
    for method_name in ("model_dump", "to_dict", "dict"):
        method = getattr(value, method_name, None)
        if callable(method):
            try:
                return _observer_jsonable(method(), depth=depth + 1)
            except Exception:
                pass
    try:
        public = {
            key: item
            for key, item in vars(value).items()
            if not str(key).startswith("_")
        }
    except Exception:
        public = None
    if public:
        return _observer_jsonable(public, depth=depth + 1)
    return str(value)[:20_000]


def current_observer_context() -> Dict[str, Any]:
    """Return invocation-scoped observer correlation for the active agent."""

    agent = _ACTIVE_AGENT.get()
    if agent is None:
        return {}
    return {
        "session_id": str(getattr(agent, "session_id", "") or ""),
        "task_id": str(getattr(agent, "_current_task_id", "") or ""),
        "turn_id": str(getattr(agent, "_current_turn_id", "") or ""),
        "request_metadata": dict(getattr(agent, "_request_metadata", None) or {}),
    }


def _invoke_observer_hook(name: str, **payload: Any) -> None:
    if not payload.get("session_id"):
        return
    try:
        from hermes_cli.plugins import invoke_hook

        invoke_hook(name, **payload)
    except Exception:
        # Telemetry cannot alter provider execution, retry, or accounting.
        logger.debug("Auxiliary observer hook %s failed", name, exc_info=True)


def _begin_auxiliary_observer_attempt(
    *,
    task: Optional[str],
    provider: Optional[str],
    model: Optional[str],
    base_url: Optional[str],
    api_mode: Optional[str],
    request: Any,
) -> Optional[Dict[str, Any]]:
    context = current_observer_context()
    if not context.get("session_id"):
        return None
    started_at = _utc_now()
    state = {
        **context,
        "api_request_id": f"aux-{uuid.uuid4().hex}",
        "purpose": _safe_task(task),
        "provider": str(provider or "unknown"),
        "model": str(model or "unknown"),
        "base_url": str(base_url or ""),
        "api_mode": str(api_mode or "chat_completions"),
        "started_at": started_at,
        "request": _observer_jsonable(request),
        "monotonic_started": time.perf_counter(),
    }
    _invoke_observer_hook(
        "pre_auxiliary_api_request",
        **{key: value for key, value in state.items() if key != "monotonic_started"},
    )
    return state


def _observer_usage(
    response: Any,
    state: Optional[Mapping[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    raw_usage = _get(response, "usage")
    if raw_usage is None:
        return None
    state = state or {}
    try:
        from agent.usage_pricing import CanonicalUsage, estimate_usage_cost, normalize_usage

        canonical = normalize_usage(
            raw_usage,
            provider=str(state.get("provider") or ""),
            api_mode=str(state.get("api_mode") or "chat_completions"),
        )
        parsed_input, parsed_output, parsed_total, warning_reasons = (
            _usage_components(raw_usage)
        )
        provider_reported_total = _token_value(
            raw_usage,
            "total_tokens",
            "totalTokens",
        )
        if canonical.total_tokens == 0 and (parsed_total or 0) > 0:
            canonical = CanonicalUsage(
                input_tokens=parsed_input or 0,
                output_tokens=parsed_output or 0,
            )
        usage: Dict[str, Any] = {
            "input_tokens": canonical.input_tokens,
            "output_tokens": canonical.output_tokens,
            "total_tokens": canonical.total_tokens,
            "cache_read_tokens": canonical.cache_read_tokens,
            "cache_write_tokens": canonical.cache_write_tokens,
            "reasoning_tokens": canonical.reasoning_tokens,
            "usage_completeness": (
                "partial" if warning_reasons else "complete"
            ),
        }
        if warning_reasons:
            usage["usage_warnings"] = list(warning_reasons)
        if (
            provider_reported_total is not None
            and provider_reported_total != canonical.total_tokens
        ):
            usage["provider_reported_total_tokens"] = provider_reported_total
        cost = estimate_usage_cost(
            str(state.get("model") or ""),
            canonical,
            provider=str(state.get("provider") or ""),
            base_url=str(state.get("base_url") or ""),
            api_key="",
        )
        usage["cost_status"] = cost.status
        usage["cost_source"] = cost.source
        if cost.amount_usd is not None:
            usage["cost_usd"] = float(cost.amount_usd)
        return usage
    except Exception:
        input_tokens, output_tokens, total_tokens, warning_reasons = (
            _usage_components(raw_usage)
        )
        usage = {
            key: value
            for key, value in {
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": total_tokens,
            }.items()
            if value is not None
        }
        if usage:
            usage["cost_status"] = "unknown"
            usage["usage_completeness"] = (
                "partial" if warning_reasons else "complete"
            )
            if warning_reasons:
                usage["usage_warnings"] = list(warning_reasons)
        return usage or None


def _end_auxiliary_observer_attempt(
    state: Optional[Dict[str, Any]],
    *,
    response: Any = None,
    error: Optional[BaseException] = None,
) -> None:
    if not state:
        return
    ended_at = _utc_now()
    duration = max(0.0, time.perf_counter() - state["monotonic_started"])
    common = {
        key: value
        for key, value in state.items()
        if key not in {"monotonic_started", "request"}
    }
    common.update({"ended_at": ended_at, "api_duration": duration})
    if error is not None:
        _invoke_observer_hook(
            "auxiliary_api_request_error",
            **common,
            error={"type": type(error).__name__, "message": str(error)},
        )
        return
    _invoke_observer_hook(
        "post_auxiliary_api_request",
        **common,
        usage=_observer_usage(response, state),
        response=_observer_jsonable(response),
        response_model=str(_get(response, "model") or state.get("model") or ""),
    )


def record_auxiliary_response(response: Any, *, task: Optional[str] = None) -> None:
    """Record one provider response when it exposes deterministic usage.

    MoA references/aggregation are excluded here because the primary
    conversation loop already folds those responses into its canonical usage
    counters.  Recording them again would double count the same provider call.
    """

    agent = _ACTIVE_AGENT.get()
    if agent is None or (task or "") in _PRIMARY_ACCOUNTED_AUXILIARY_TASKS:
        return
    _ensure_initialized(agent)

    response_key = id(response)
    lock = agent._runtime_usage_lock
    with lock:
        seen_response = agent._runtime_usage_seen_aux_response_objects.get(response_key)
        if seen_response is response:
            return
        # A collision should be impossible while the prior object remains
        # strongly held, but fail open for accounting if a non-CPython runtime
        # ever presents one: replace and count the distinct object.
        agent._runtime_usage_seen_aux_response_objects[response_key] = response

        usage = _get(response, "usage")
        input_tokens, output_tokens, total_tokens, warning_reasons = (
            _usage_components(usage)
        )
        for warning_reason in warning_reasons:
            agent._runtime_usage_warnings.add(
                f"auxiliary:{_safe_task(task)}:{warning_reason}"
            )
        if input_tokens is None and output_tokens is None and total_tokens is None:
            return

        agent.session_auxiliary_input_tokens += input_tokens or 0
        agent.session_auxiliary_output_tokens += output_tokens or 0
        agent.session_auxiliary_total_tokens += total_tokens or 0
        agent.session_auxiliary_response_count += 1


def record_primary_dispatch_response(agent: Any, response: Any) -> None:
    """Provisionally retain/count one distinct returned primary response.

    Response-shape validation happens after the dispatch seam. Invalid or
    terminal response objects may therefore be retried before the historical
    primary accounting block runs. Exact usage from those responses is still
    billable and belongs in the parent aggregate. The eventual valid response
    is removed by identity in ``commit_primary_response`` before its canonical
    counters are added.
    """

    _ensure_initialized(agent)
    response_key = id(response)
    with agent._runtime_usage_lock:
        pending = agent._runtime_usage_pending_primary_response_objects.get(
            response_key
        )
        if pending is not None and pending.get("response") is response:
            return
        committed = agent._runtime_usage_committed_primary_response_objects.get(
            response_key
        )
        if committed is response:
            return

        usage = _get(response, "usage")
        components = _usage_components(usage)
        entry = {
            "response": response,
            "input_tokens": components[0],
            "output_tokens": components[1],
            "total_tokens": components[2],
        }
        agent._runtime_usage_pending_primary_response_objects[response_key] = entry

        input_tokens, output_tokens, total_tokens, warning_reasons = components
        if input_tokens is None and output_tokens is None and total_tokens is None:
            agent.session_primary_usage_missing_count += 1
            agent._runtime_usage_warnings.add(
                "primary:provider_response_missing_usage"
            )
            return

        agent.session_primary_retry_input_tokens += input_tokens or 0
        agent.session_primary_retry_output_tokens += output_tokens or 0
        agent.session_primary_retry_total_tokens += (
            total_tokens
            if total_tokens is not None
            else (input_tokens or 0) + (output_tokens or 0)
        )
        agent.session_primary_retry_response_count += 1
        for warning_reason in warning_reasons:
            agent._runtime_usage_warnings.add(f"primary:{warning_reason}")


def commit_primary_response(agent: Any, response: Any, canonical_usage: Any) -> bool:
    """Move one valid primary response from provisional to canonical totals.

    Returns ``False`` when the same response object was already committed.
    Identity retention makes retry/re-validation idempotent without treating
    two distinct response objects with the same provider id as one call.
    """

    _ensure_initialized(agent)
    response_key = id(response)
    prompt_tokens = _nonnegative_int(
        _get(canonical_usage, "prompt_tokens")
    )
    output_tokens = _nonnegative_int(
        _get(canonical_usage, "output_tokens")
    )
    total_tokens = _nonnegative_int(
        _get(canonical_usage, "total_tokens")
    )
    input_tokens = _nonnegative_int(
        _get(canonical_usage, "input_tokens")
    )
    cache_read_tokens = _nonnegative_int(
        _get(canonical_usage, "cache_read_tokens")
    )
    cache_write_tokens = _nonnegative_int(
        _get(canonical_usage, "cache_write_tokens")
    )
    reasoning_tokens = _nonnegative_int(
        _get(canonical_usage, "reasoning_tokens")
    )
    with agent._runtime_usage_lock:
        committed = agent._runtime_usage_committed_primary_response_objects.get(
            response_key
        )
        if committed is response:
            return False

        if (
            prompt_tokens is None
            or output_tokens is None
            or total_tokens is None
            or total_tokens != prompt_tokens + output_tokens
            or (
                input_tokens is not None
                and cache_read_tokens is not None
                and cache_write_tokens is not None
                and input_tokens + cache_read_tokens + cache_write_tokens
                != prompt_tokens
            )
        ):
            # Leave any provisionally parsed raw usage in the retry bucket. A
            # malformed canonical adapter result must not erase known provider
            # counts before making the aggregate partial.
            agent._runtime_usage_warnings.add(
                "primary:canonical_usage_missing_or_invalid"
            )
            return False

        pending = agent._runtime_usage_pending_primary_response_objects.get(
            response_key
        )
        if pending is not None and pending.get("response") is response:
            pending_input = _nonnegative_int(pending.get("input_tokens"))
            pending_output = _nonnegative_int(pending.get("output_tokens"))
            pending_total = _nonnegative_int(pending.get("total_tokens"))
            if (
                pending_input is not None
                or pending_output is not None
                or pending_total is not None
            ):
                agent.session_primary_retry_input_tokens = max(
                    0,
                    agent.session_primary_retry_input_tokens - (pending_input or 0),
                )
                agent.session_primary_retry_output_tokens = max(
                    0,
                    agent.session_primary_retry_output_tokens - (pending_output or 0),
                )
                agent.session_primary_retry_total_tokens = max(
                    0,
                    agent.session_primary_retry_total_tokens
                    - (
                        pending_total
                        if pending_total is not None
                        else (pending_input or 0) + (pending_output or 0)
                    ),
                )
                agent.session_primary_retry_response_count = max(
                    0,
                    agent.session_primary_retry_response_count - 1,
                )
            del agent._runtime_usage_pending_primary_response_objects[response_key]

        agent.session_prompt_tokens = (
            _counter(agent, "session_prompt_tokens") + prompt_tokens
        )
        agent.session_completion_tokens = (
            _counter(agent, "session_completion_tokens") + output_tokens
        )
        agent.session_total_tokens = (
            _counter(agent, "session_total_tokens") + total_tokens
        )
        agent.session_api_calls = _counter(agent, "session_api_calls") + 1
        agent.session_input_tokens = (
            _counter(agent, "session_input_tokens") + (input_tokens or 0)
        )
        agent.session_output_tokens = (
            _counter(agent, "session_output_tokens") + output_tokens
        )
        agent.session_cache_read_tokens = (
            _counter(agent, "session_cache_read_tokens")
            + (cache_read_tokens or 0)
        )
        agent.session_cache_write_tokens = (
            _counter(agent, "session_cache_write_tokens")
            + (cache_write_tokens or 0)
        )
        agent.session_reasoning_tokens = (
            _counter(agent, "session_reasoning_tokens")
            + (reasoning_tokens or 0)
        )
        agent._runtime_usage_committed_primary_response_objects[response_key] = response
        return True


def mark_primary_dispatch_uncertain(agent: Any, *, reason: str) -> None:
    """Mark a provider attempt that began but returned no terminal usage."""

    _ensure_initialized(agent)
    with agent._runtime_usage_lock:
        agent.session_primary_usage_missing_count += 1
        agent._runtime_usage_warnings.add(f"primary:{_safe_reason(reason)}")


def mark_moa_reference_usage_incomplete(
    agent: Any,
    warning_reasons: Any,
) -> None:
    """Carry safe MoA advisor-attempt uncertainty into the run aggregate."""

    _ensure_initialized(agent)
    if isinstance(warning_reasons, str):
        warning_reasons = [warning_reasons]
    with agent._runtime_usage_lock:
        for reason in warning_reasons or []:
            agent._runtime_usage_warnings.add(
                f"primary:moa_reference:{_safe_reason(str(reason))}"
            )


def mark_auxiliary_usage_accounting_failed(*, task: Optional[str] = None) -> None:
    """Best-effort warning for bookkeeping that failed after provider success."""

    _best_effort_mark_auxiliary_dispatch_uncertain(
        task=task,
        reason="usage_accounting_failed",
    )


def finalize_moa_pending_accounting(agent: Any) -> None:
    """Consume any MoA advisor rollup before a conversation can exit.

    The normal response path consumes advisor usage alongside aggregator
    usage. Truncation, terminal aggregator failure, interruption, and other
    early exits bypass that block. This finalizer is deliberately idempotent:
    the MoA facade's consume methods clear their pending values, so calling it
    after the normal path is a no-op and calling it after an early exit folds
    the otherwise-lost advisor tokens, cost, warnings, and trace exactly once.
    Instrumentation is best-effort and must never replace the run result.
    """

    if str(getattr(agent, "provider", "") or "").lower() != "moa":
        return
    client = getattr(agent, "client", None)
    if client is None:
        return

    try:
        if hasattr(client, "consume_reference_usage"):
            try:
                ref_usage, ref_cost, warning_reasons = (
                    client.consume_reference_usage()
                )
            except Exception:
                logger.debug("MoA final advisor usage consume failed", exc_info=True)
                mark_moa_reference_usage_incomplete(
                    agent,
                    ("usage_rollup_failed",),
                )
            else:
                if warning_reasons:
                    mark_moa_reference_usage_incomplete(agent, warning_reasons)
                _commit_pending_moa_reference_usage(agent, ref_usage, ref_cost)
    except Exception:
        logger.debug("MoA final advisor accounting failed", exc_info=True)
        try:
            mark_moa_reference_usage_incomplete(
                agent,
                ("usage_rollup_failed",),
            )
        except Exception:
            pass
    finally:
        if hasattr(client, "consume_and_save_trace"):
            try:
                streamed_text = (
                    getattr(agent, "_current_streamed_assistant_text", "") or ""
                )
                client.consume_and_save_trace(
                    getattr(agent, "session_id", None),
                    aggregator_output_fallback=streamed_text or None,
                )
            except Exception:
                logger.debug("MoA final trace flush failed", exc_info=True)


def _commit_pending_moa_reference_usage(
    agent: Any,
    usage: Any,
    cost: Any,
) -> None:
    """Add one already-consumed advisor rollup to the acting parent bucket."""

    _ensure_initialized(agent)
    prompt_tokens = _nonnegative_int(_get(usage, "prompt_tokens"))
    output_tokens = _nonnegative_int(_get(usage, "output_tokens"))
    total_tokens = _nonnegative_int(_get(usage, "total_tokens"))
    input_tokens = _nonnegative_int(_get(usage, "input_tokens"))
    cache_read_tokens = _nonnegative_int(_get(usage, "cache_read_tokens"))
    cache_write_tokens = _nonnegative_int(_get(usage, "cache_write_tokens"))
    reasoning_tokens = _nonnegative_int(_get(usage, "reasoning_tokens"))
    has_usage = any(
        value is not None and value > 0
        for value in (prompt_tokens, output_tokens, total_tokens)
    )
    valid_usage = (
        prompt_tokens is not None
        and output_tokens is not None
        and total_tokens is not None
        and total_tokens == prompt_tokens + output_tokens
    )
    if has_usage and not valid_usage:
        mark_moa_reference_usage_incomplete(
            agent,
            ("usage_rollup_missing_or_invalid",),
        )
    elif valid_usage and has_usage:
        with agent._runtime_usage_lock:
            agent.session_prompt_tokens = (
                _counter(agent, "session_prompt_tokens") + prompt_tokens
            )
            agent.session_completion_tokens = (
                _counter(agent, "session_completion_tokens") + output_tokens
            )
            agent.session_total_tokens = (
                _counter(agent, "session_total_tokens") + total_tokens
            )
            agent.session_input_tokens = (
                _counter(agent, "session_input_tokens") + (input_tokens or 0)
            )
            agent.session_output_tokens = (
                _counter(agent, "session_output_tokens") + output_tokens
            )
            agent.session_cache_read_tokens = (
                _counter(agent, "session_cache_read_tokens")
                + (cache_read_tokens or 0)
            )
            agent.session_cache_write_tokens = (
                _counter(agent, "session_cache_write_tokens")
                + (cache_write_tokens or 0)
            )
            agent.session_reasoning_tokens = (
                _counter(agent, "session_reasoning_tokens")
                + (reasoning_tokens or 0)
            )

    parsed_cost: Optional[float] = None
    if cost is not None:
        try:
            parsed_cost = float(cost)
            agent.session_estimated_cost_usd = float(
                getattr(agent, "session_estimated_cost_usd", 0.0) or 0.0
            ) + parsed_cost
            if getattr(agent, "session_cost_status", "unknown") == "unknown":
                agent.session_cost_status = "estimated"
                agent.session_cost_source = "moa_reference_rollup"
        except (TypeError, ValueError):
            mark_moa_reference_usage_incomplete(
                agent,
                ("cost_rollup_invalid",),
            )

    if valid_usage and has_usage and getattr(agent, "_session_db", None):
        session_id = getattr(agent, "session_id", None)
        if session_id:
            try:
                if not getattr(agent, "_session_db_created", False):
                    agent._ensure_db_session()
                agent._session_db.update_token_counts(
                    session_id,
                    input_tokens=input_tokens or 0,
                    output_tokens=output_tokens,
                    cache_read_tokens=cache_read_tokens or 0,
                    cache_write_tokens=cache_write_tokens or 0,
                    reasoning_tokens=reasoning_tokens or 0,
                    estimated_cost_usd=parsed_cost,
                    cost_status=getattr(agent, "session_cost_status", "unknown"),
                    cost_source=getattr(agent, "session_cost_source", "none"),
                    billing_provider="moa",
                    billing_base_url=getattr(agent, "base_url", None),
                    model=getattr(agent, "model", None),
                    api_call_count=0,
                )
            except Exception:
                logger.debug("MoA final advisor persistence failed", exc_info=True)


def usage_has_exact_input_output(usage: Any) -> bool:
    """Return whether usage has safe, reconciling input/output components."""

    input_tokens, output_tokens, total_tokens, warnings = _usage_components(usage)
    return (
        input_tokens is not None
        and output_tokens is not None
        and total_tokens is not None
        and not warnings
        and total_tokens == input_tokens + output_tokens
    )


def mark_primary_usage_missing(agent: Any, *, reason: str) -> None:
    """Mark a primary provider response whose token usage was unavailable."""

    _ensure_initialized(agent)
    with agent._runtime_usage_lock:
        agent.session_primary_usage_missing_count += 1
        agent._runtime_usage_warnings.add(f"primary:{_safe_reason(reason)}")


def validate_primary_usage_components(agent: Any, usage: Any) -> None:
    """Fail closed when a primary response lacks exact input/output buckets."""

    _ensure_initialized(agent)
    input_tokens, output_tokens, total_tokens, warning_reasons = _usage_components(
        usage
    )
    with agent._runtime_usage_lock:
        for warning_reason in warning_reasons:
            agent._runtime_usage_warnings.add(f"primary:{warning_reason}")


def rollup_delegated_usage(
    parent_agent: Any,
    child_usage: Mapping[str, Any],
    *,
    child_session_id: Optional[str] = None,
) -> None:
    """Fold a child's already-aggregated snapshot into its direct parent.

    A child snapshot already includes its own nested descendants.  Summing
    only the three top-level values here makes nested delegation additive once
    per tree edge without walking or recounting the child's breakdown.
    ``child_session_id`` is intentionally accepted only for call-site clarity;
    it is never copied into warnings or response metadata.
    """

    del child_session_id
    _ensure_initialized(parent_agent)
    parsed_input_tokens = _nonnegative_int(child_usage.get("input_tokens"))
    parsed_output_tokens = _nonnegative_int(child_usage.get("output_tokens"))
    input_tokens = parsed_input_tokens or 0
    output_tokens = parsed_output_tokens or 0
    reported_total_tokens = _nonnegative_int(child_usage.get("total_tokens"))
    total_tokens = (
        input_tokens + output_tokens
        if parsed_input_tokens is not None and parsed_output_tokens is not None
        else reported_total_tokens
        if reported_total_tokens is not None
        else input_tokens + output_tokens
    )
    completeness = child_usage.get("completeness")

    with parent_agent._runtime_usage_lock:
        parent_agent.session_delegated_input_tokens += input_tokens
        parent_agent.session_delegated_output_tokens += output_tokens
        parent_agent.session_delegated_total_tokens += total_tokens
        if completeness != "unavailable":
            parent_agent.session_delegated_response_count += 1
        if parsed_input_tokens is None or parsed_output_tokens is None:
            parent_agent._runtime_usage_warnings.add(
                "delegated:child_input_output_breakdown_missing_or_invalid"
            )
        if (
            reported_total_tokens is not None
            and parsed_input_tokens is not None
            and parsed_output_tokens is not None
            and reported_total_tokens != input_tokens + output_tokens
        ):
            parent_agent._runtime_usage_warnings.add(
                "delegated:child_total_does_not_match_input_output"
            )
        if completeness != "complete":
            parent_agent._runtime_usage_warnings.add(
                "delegated:child_usage_unavailable"
                if completeness == "unavailable"
                else "delegated:child_usage_partial"
            )


def mark_delegated_usage_uncertain(parent_agent: Any, *, reason: str) -> None:
    """Mark a child timeout/interruption where final provider usage may be lost."""

    _ensure_initialized(parent_agent)
    with parent_agent._runtime_usage_lock:
        parent_agent._runtime_usage_warnings.add(
            f"delegated:{_safe_reason(reason)}"
        )


def snapshot_agent_usage(agent: Any) -> Dict[str, Any]:
    """Return aggregate usage, breakdown, and a durable completeness status."""

    _ensure_initialized(agent)
    with agent._runtime_usage_lock:
        parent = {
            "input_tokens": (
                _counter(agent, "session_prompt_tokens")
                + _counter(agent, "session_primary_retry_input_tokens")
            ),
            "output_tokens": (
                _counter(agent, "session_completion_tokens")
                + _counter(agent, "session_primary_retry_output_tokens")
            ),
            "total_tokens": (
                _counter(agent, "session_total_tokens")
                + _counter(agent, "session_primary_retry_total_tokens")
            ),
        }
        auxiliary = {
            "input_tokens": _counter(agent, "session_auxiliary_input_tokens"),
            "output_tokens": _counter(agent, "session_auxiliary_output_tokens"),
            "total_tokens": _counter(agent, "session_auxiliary_total_tokens"),
        }
        delegated = {
            "input_tokens": _counter(agent, "session_delegated_input_tokens"),
            "output_tokens": _counter(agent, "session_delegated_output_tokens"),
            "total_tokens": _counter(agent, "session_delegated_total_tokens"),
        }
        warnings = sorted(str(value) for value in agent._runtime_usage_warnings)
        known_response_count = (
            _counter(agent, "session_api_calls")
            + _counter(agent, "session_primary_retry_response_count")
            + _counter(agent, "session_auxiliary_response_count")
            + _counter(agent, "session_delegated_response_count")
        )

    aggregate = {
        key: parent[key] + auxiliary[key] + delegated[key]
        for key in ("input_tokens", "output_tokens", "total_tokens")
    }
    if known_response_count == 0 and not any(aggregate.values()):
        completeness = "unavailable"
        warnings = sorted(set(warnings) | {"no_provider_usage_reported"})
    elif warnings:
        completeness = "partial"
    else:
        completeness = "complete"

    return {
        **aggregate,
        "completeness": completeness,
        "warnings": warnings,
        "breakdown": {
            "parent": parent,
            "auxiliary": auxiliary,
            "delegated": delegated,
        },
    }


def _ensure_initialized(agent: Any) -> None:
    if getattr(agent, "_runtime_usage_initialized", None) is not True:
        initialize_agent_usage_attribution(agent)


def _get(value: Any, key: str) -> Any:
    if isinstance(value, Mapping):
        return value.get(key)
    return getattr(value, key, None)


def _usage_components(
    usage: Any,
) -> tuple[Optional[int], Optional[int], Optional[int], tuple[str, ...]]:
    """Parse raw provider usage without inventing or clamping token counts."""

    if usage is None:
        return None, None, None, ("provider_response_missing_usage",)

    prompt_tokens = _token_value(usage, "prompt_tokens")
    snake_input_tokens = _token_value(usage, "input_tokens")
    camel_input_tokens = _token_value(usage, "inputTokens")
    warnings: list[str] = []
    # Codex app-server mirrors the Responses API contract: camelCase
    # inputTokens is the whole prompt total, while cachedInputTokens (and newer
    # cacheWriteInputTokens) are subsets. They must never be added to the
    # provider total a second time.
    uses_camel_input = prompt_tokens is None and snake_input_tokens is None
    camel_cache_read_tokens = (
        _token_value(usage, "cachedInputTokens") or 0
        if uses_camel_input
        else 0
    )
    camel_cache_write_tokens = (
        _token_value(usage, "cacheWriteInputTokens") or 0
        if uses_camel_input
        else 0
    )
    if (
        uses_camel_input
        and camel_input_tokens is not None
        and camel_cache_read_tokens + camel_cache_write_tokens > camel_input_tokens
    ):
        warnings.append("cache_tokens_exceed_input_tokens")
    base_input_tokens = (
        prompt_tokens
        if prompt_tokens is not None
        else snake_input_tokens
        if snake_input_tokens is not None
        else camel_input_tokens
    )
    anthropic_cache_read_tokens = _token_value(
        usage,
        "cache_read_input_tokens",
    )
    anthropic_cache_write_tokens = _token_value(
        usage,
        "cache_creation_input_tokens",
    )
    anthropic_cache_tokens = (
        (anthropic_cache_read_tokens or 0)
        + (anthropic_cache_write_tokens or 0)
        if prompt_tokens is None
        and snake_input_tokens is not None
        and (
            anthropic_cache_read_tokens is not None
            or anthropic_cache_write_tokens is not None
        )
        else 0
    )
    input_tokens = (
        base_input_tokens + anthropic_cache_tokens
        if base_input_tokens is not None
        else None
    )
    output_tokens = _token_value(
        usage,
        "completion_tokens",
        "output_tokens",
        "outputTokens",
    )
    reported_total = _token_value(usage, "total_tokens", "totalTokens")
    if input_tokens is None or output_tokens is None:
        warnings.append("input_output_breakdown_missing_or_invalid")
    expected_total = (
        input_tokens + output_tokens
        if input_tokens is not None and output_tokens is not None
        else None
    )
    if (
        reported_total is not None
        and expected_total is not None
        and reported_total != expected_total
    ):
        warnings.append("total_does_not_match_input_output")
    # Exact input/output components are the strongest attributable evidence and
    # keep every emitted bucket internally reconciling. Some Anthropic proxies
    # report a ``total_tokens`` value that excludes cache-read/write input; keep
    # the mismatch warning (so status is partial) but use the known component
    # sum. A reported total is retained only when components are unavailable.
    total_tokens = expected_total if expected_total is not None else reported_total
    if input_tokens is None and output_tokens is None and total_tokens is None:
        return None, None, None, ("provider_response_missing_usage",)
    return input_tokens, output_tokens, total_tokens, tuple(warnings)


def _best_effort_mark_primary_dispatch_uncertain(
    agent: Any,
    *,
    reason: str,
) -> None:
    try:
        mark_primary_dispatch_uncertain(agent, reason=reason)
    except Exception:
        logger.debug("Primary usage uncertainty marker failed", exc_info=True)


def _best_effort_mark_auxiliary_dispatch_uncertain(
    *,
    task: Optional[str],
    reason: str,
) -> None:
    try:
        _mark_auxiliary_dispatch_uncertain(task=task, reason=reason)
    except Exception:
        logger.debug("Auxiliary usage uncertainty marker failed", exc_info=True)


def _mark_auxiliary_dispatch_uncertain(
    *,
    task: Optional[str],
    reason: str,
) -> None:
    normalized_task = task or ""
    # Reference workers carry their uncertainty through the per-reference
    # observer so the MoA rollup can add one safe warning without double
    # reporting it through the parent ContextVar.
    if normalized_task == "moa_reference":
        return
    agent = _ACTIVE_AGENT.get()
    if agent is None:
        return
    _ensure_initialized(agent)
    with agent._runtime_usage_lock:
        if normalized_task == "moa_aggregator":
            agent.session_primary_usage_missing_count += 1
            agent._runtime_usage_warnings.add(
                f"primary:moa_aggregator:{_safe_reason(reason)}"
            )
        else:
            agent._runtime_usage_warnings.add(
                f"auxiliary:{_safe_task(task)}:{_safe_reason(reason)}"
            )


def _record_tracked_auxiliary_response(
    response: Any,
    *,
    task: Optional[str],
) -> None:
    # The MoA aggregator is the acting primary model. Its final response is
    # committed by the normal conversation accounting path, while any distinct
    # invalid response retried inside call_llm remains as parent retry spend.
    if (task or "") == "moa_aggregator":
        agent = _ACTIVE_AGENT.get()
        if agent is not None:
            record_primary_dispatch_response(agent, response)
        return
    record_auxiliary_response(response, task=task)


def _notify_auxiliary_attempt_observer(
    *,
    response: Any,
    reason: Optional[str],
) -> None:
    observer = _AUXILIARY_ATTEMPT_OBSERVER.get()
    if observer is None:
        return
    try:
        observer(response=response, reason=reason)
    except Exception:
        # Usage instrumentation must never alter provider retry behavior.
        return


def _token_value(value: Any, *keys: str) -> Optional[int]:
    for key in keys:
        parsed = _nonnegative_int(_get(value, key))
        if parsed is not None:
            return parsed
    return None


def _nonnegative_int(value: Any) -> Optional[int]:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value if value >= 0 else None
    if isinstance(value, float) and value.is_integer() and value >= 0:
        return int(value)
    return None


def _counter(agent: Any, name: str) -> int:
    return _nonnegative_int(getattr(agent, name, 0)) or 0


def _safe_task(task: Optional[str]) -> str:
    normalized = "".join(
        char if char.isalnum() or char in {"_", "-"} else "_"
        for char in (task or "unclassified")[:80]
    )
    return normalized or "unclassified"


def _safe_reason(reason: str) -> str:
    normalized = "".join(
        char if char.isalnum() or char in {"_", "-"} else "_"
        for char in str(reason)[:80]
    )
    return normalized or "usage_unavailable"


def valid_usage_completeness(value: Any) -> str:
    """Normalize an externally supplied completeness value fail-closed."""

    return value if value in _VALID_COMPLETENESS else "partial"
