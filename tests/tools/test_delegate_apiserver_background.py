"""delegate_task(background=true) delivery contracts by channel capability.

A raw API-server session id provides Hermes history continuity, but it is not
a live return channel.  Stateless API requests therefore keep delegated work
inside the current turn so their result is materialized by the request's
consumer.  Push-capable sessions still detach and retain their captured origin
metadata.
"""

import json
import threading
import time
from unittest.mock import MagicMock

import pytest

from gateway.session_context import set_session_vars
from tools.process_registry import process_registry


@pytest.fixture(autouse=True)
def _clean_queue_and_context(monkeypatch):
    monkeypatch.delenv("HERMES_SESSION_ID", raising=False)
    while not process_registry.completion_queue.empty():
        try:
            process_registry.completion_queue.get_nowait()
        except Exception:
            break
    yield
    # Restore ContextVars to the pristine "never set" sentinel rather than
    # clear_session_vars()'s explicit-"" state, which would mask env vars for
    # unrelated tests running later in the same worker.
    import gateway.session_context as sc

    for var in sc._VAR_MAP.values():
        var.set(sc._UNSET)
    sc._SESSION_ASYNC_DELIVERY.set(sc._UNSET)
    # set_current_session_id (invoked by the clobber-reproducing fake child
    # build) writes os.environ directly — scrub it so it can't leak into
    # other test modules.
    import os

    os.environ.pop("HERMES_SESSION_ID", None)
    while not process_registry.completion_queue.empty():
        try:
            process_registry.completion_queue.get_nowait()
        except Exception:
            break


def _drain_one(timeout=5.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if not process_registry.completion_queue.empty():
            return process_registry.completion_queue.get_nowait()
        time.sleep(0.02)
    return None


def _fake_parent():
    parent = MagicMock()
    parent._delegate_depth = 0
    parent.session_id = "sess"
    parent._interrupt_requested = False
    parent._active_children = []
    parent._active_children_lock = None
    return parent


def _patch_delegate(monkeypatch, *, child_runner=None, observed_attached=None):
    import tools.delegate_tool as dt

    fake_child = MagicMock()
    fake_child._delegate_role = "leaf"
    fake_child._subagent_id = "s1"

    def fast_child(task_index, goal, child=None, parent_agent=None, **kw):
        if observed_attached is not None:
            observed_attached.append(child in parent_agent._active_children)
        try:
            if child_runner is not None:
                return child_runner(
                    task_index=task_index,
                    goal=goal,
                    child=child,
                    parent_agent=parent_agent,
                )
            return {
                "task_index": 0, "status": "completed", "summary": f"done: {goal}",
                "api_calls": 1, "duration_seconds": 0.1, "model": "m",
                "exit_reason": "completed",
            }
        finally:
            try:
                parent_agent._active_children.remove(child)
            except ValueError:
                pass

    creds = {
        "model": "m", "provider": None, "base_url": None, "api_key": None,
        "api_mode": None, "command": None, "args": None,
    }
    def clobbering_build_child(**kw):
        # Reproduce what the real _build_child_agent -> AIAgent -> agent_init
        # path does: it synchronizes the child's internal session id into the
        # HERMES_SESSION_ID ContextVar + os.environ, clobbering the spawner's
        # id ~milliseconds before delegate_tool dispatches the batch.
        from gateway.session_context import set_current_session_id

        set_current_session_id("20260715_child1")
        kw["parent_agent"]._active_children.append(fake_child)
        return fake_child

    monkeypatch.setattr(dt, "_build_child_agent", clobbering_build_child)
    monkeypatch.setattr(dt, "_run_single_child", fast_child)
    monkeypatch.setattr(dt, "_resolve_delegation_credentials", lambda *a, **k: creds)
    return dt


def test_apiserver_session_with_id_stays_synchronous(monkeypatch):
    """A raw session id cannot override the API server's stateless contract."""
    observed_attached = []
    dt = _patch_delegate(monkeypatch, observed_attached=observed_attached)
    monkeypatch.setenv("HERMES_SESSION_ID", "raw-sid-7")
    set_session_vars(
        platform="api_server",
        chat_id="raw-sid-7",
        session_key="raw-sid-7",
        session_id="raw-sid-7",
        async_delivery=False,
    )

    out = dt.delegate_task(
        goal="bg on api_server", context="ctx",
        background=True, parent_agent=_fake_parent(),
    )
    parsed = json.loads(out)
    assert parsed.get("status") != "dispatched", parsed
    assert "SYNCHRONOUSLY" in parsed.get("note", "")
    assert parsed["results"][0]["summary"] == "done: bg on api_server"
    assert observed_attached == [True]
    assert process_registry.completion_queue.empty()


def test_true_capability_dispatches_and_keeps_origin_session_id(monkeypatch):
    """A true capability remains detached with captured origin metadata."""
    dt = _patch_delegate(monkeypatch)
    monkeypatch.setenv("HERMES_SESSION_ID", "raw-sid-7")
    set_session_vars(
        platform="api_server",
        chat_id="raw-sid-7",
        session_key="raw-sid-7",
        session_id="raw-sid-7",
        async_delivery=True,
    )

    out = dt.delegate_task(
        goal="bg on routable channel", context="ctx",
        background=True, parent_agent=_fake_parent(),
    )
    parsed = json.loads(out)
    assert parsed["status"] == "dispatched", parsed
    assert parsed["mode"] == "background"

    evt = _drain_one()
    assert evt is not None
    assert evt["type"] == "async_delegation"
    # Preserve the spawner's id rather than the child-internal id written while
    # constructing the child. Other async consumers rely on this attribution.
    assert evt["origin_session_id"] == "raw-sid-7"


# ---------------------------------------------------------------------------
# _current_origin_session_id — the clobber-proof origin capture helper
# ---------------------------------------------------------------------------


def test_apiserver_session_without_id_stays_synchronous(monkeypatch):
    """No session id to wake → keep the sync fallback (a detached result
    would never re-enter any conversation)."""
    dt = _patch_delegate(monkeypatch)
    set_session_vars(
        platform="api_server",
        chat_id="",
        session_key="",
        session_id="",
        async_delivery=False,
    )

    out = dt.delegate_task(
        goal="one-shot", context="ctx",
        background=True, parent_agent=_fake_parent(),
    )
    parsed = json.loads(out)
    assert parsed.get("status") != "dispatched", parsed
    assert "SYNCHRONOUSLY" in parsed.get("note", "")
    assert process_registry.completion_queue.empty()


def test_apiserver_sync_fallback_propagates_parent_interrupt(monkeypatch):
    """Stop owns the attached child and leaves no late completion queued."""
    child_started = threading.Event()

    def interrupted_child(*, task_index, child, parent_agent, **kw):
        child_started.set()
        deadline = time.monotonic() + 5
        while not parent_agent._interrupt_requested and time.monotonic() < deadline:
            time.sleep(0.01)
        return {
            "task_index": task_index,
            "status": "interrupted",
            "summary": None,
            "error": "Parent agent interrupted",
            "api_calls": 0,
            "duration_seconds": 0.1,
            "model": "m",
            "exit_reason": "interrupted",
        }

    observed_attached = []
    dt = _patch_delegate(
        monkeypatch,
        child_runner=interrupted_child,
        observed_attached=observed_attached,
    )
    parent = _fake_parent()
    result = {}

    def run_delegate():
        set_session_vars(
            platform="api_server",
            chat_id="raw-sid-stop",
            session_key="raw-sid-stop",
            session_id="raw-sid-stop",
            async_delivery=False,
        )
        result["out"] = dt.delegate_task(
            goal="long child", context="ctx", background=True, parent_agent=parent
        )

    worker = threading.Thread(
        target=run_delegate,
        daemon=True,
    )
    worker.start()
    assert child_started.wait(timeout=2)
    assert parent._active_children
    parent._interrupt_requested = True
    worker.join(timeout=5)

    assert not worker.is_alive()
    parsed = json.loads(result["out"])
    assert parsed["results"][0]["status"] == "interrupted"
    assert observed_attached == [True]
    assert process_registry.completion_queue.empty()
