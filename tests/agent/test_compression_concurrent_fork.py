"""Regression: prevent transcript fork when two paths compress the same session_id.

Damien's incident (Discord, 2026-05-28): a long Hermes session in a Discord
gateway hit the compression threshold at the end of a turn.  The parent agent
finished delivering the response and ``conversation_loop.py`` fired
``_spawn_background_review(...)`` — which builds a forked ``AIAgent`` that
inherits ``agent.session_id`` (see ``agent/background_review.py``::
``review_agent.session_id = agent.session_id``).  Roughly two seconds later
a synthetic ``Background process proc_… completed`` event arrived and
started a fresh turn on the same parent ``session_id`` (still cached in the
gateway's ``SessionEntry``).  Both paths hit preflight compression on the
same parent transcript and called ``_compress_context`` concurrently.  Each
ended the parent and created its own CHILD session in ``state.db``, both
parented to the same old id.  The gateway's ``SessionEntry`` only caught one
rotation; the other child became an orphan that silently accumulated writes.

Repro shape on Damien's machine:

  parent 20260527_234659_e65f0e  ended_at=set  end_reason='compression'
  child  20260528_113619_fc80e1  parent=20260527_234659_e65f0e  (in SessionEntry)
  child  <orphan>                parent=20260527_234659_e65f0e  (silent writes)

This regression simulates the two concurrent ``compress_context`` calls
against a shared ``state.db`` and asserts that the per-session compression
lock added in this PR prevents the orphan child.  Without the lock the
fixture deterministically produces 2 children; with the lock, exactly 1.
"""

from __future__ import annotations

import copy
import inspect
import json
import logging
import os
import sqlite3
import threading
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from agent.context_compressor import SUMMARY_PREFIX
from hermes_state import SessionDB


def _build_agent_with_db(
    db: SessionDB,
    session_id: str,
    *,
    stub_compressor: bool = True,
):
    """Build an AIAgent that's wired to ``db`` and pinned to ``session_id``."""
    with patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"}):
        from run_agent import AIAgent

        agent = AIAgent(
            api_key="test-key",
            base_url="https://openrouter.ai/api/v1",
            model="test/model",
            quiet_mode=True,
            session_db=db,
            session_id=session_id,
            skip_context_files=True,
            skip_memory=True,
        )

    # Stub the compressor so it returns deterministic output and DOESN'T make
    # an LLM call.  Sleep inside compress() so the two threads' rotations
    # actually overlap — without that the OS could happen to serialize them
    # and hide the bug.
    if not stub_compressor:
        return agent

    compressor = MagicMock()

    def _compress_with_overlap(*_a, **_kw):
        time.sleep(0.15)
        return [
            {"role": "user", "content": "[CONTEXT COMPACTION] summary"},
            {"role": "user", "content": "tail"},
        ]

    compressor.compress.side_effect = _compress_with_overlap
    compressor.compression_count = 1
    compressor.last_prompt_tokens = 0
    compressor.last_completion_tokens = 0
    compressor._last_summary_error = None
    compressor._last_compress_aborted = False
    compressor._last_aux_model_failure_model = None
    compressor._last_aux_model_failure_error = None
    agent.context_compressor = compressor
    # The compressor is a stub — the one-time compression-model feasibility
    # probe would resolve a REAL auxiliary provider (credential pools, live
    # token exchanges over the network on some dev machines). That makes the
    # first _compress_context call in a test nondeterministically slow (>2s)
    # and flakes event-based timing assertions. Mark it done: these tests
    # exercise locking/fencing/rotation, never aux-model feasibility.
    agent._compression_feasibility_checked = True
    # These tests cover the ROTATION fallback path (forking, child sessions,
    # lock contention) — pin in_place=False so they keep exercising it
    # regardless of the global default (which flipped to True in #38763).
    agent.compression_in_place = False
    # AE-204: the losing path now waits for the holder before giving up. Keep
    # the default wait out of the suite's runtime; tests that exercise the
    # wait itself set their own budget.
    agent._compression_lock_wait_seconds = 0.25
    return agent


def _count_children(db: SessionDB, parent_sid: str) -> int:
    """Count rows in state.db whose parent_session_id == parent_sid."""
    rows = db._conn.execute(
        "SELECT id FROM sessions WHERE parent_session_id = ?",
        (parent_sid,),
    ).fetchall()
    return len(rows)


def _live_child_id(db: SessionDB, parent_sid: str) -> str | None:
    """The single child id of ``parent_sid``, or None when there is none.

    Fails loudly on more than one child: callers use this to prove the agents
    converged on the winner's session, so a multi-child state is a fork and
    must not be silently reduced to 'the first row'.
    """
    rows = db._conn.execute(
        "SELECT id FROM sessions WHERE parent_session_id = ?",
        (parent_sid,),
    ).fetchall()
    assert len(rows) <= 1, f"expected at most one child of {parent_sid}, got {rows!r}"
    return rows[0][0] if rows else None


def _wait_for_touch(touch_calls: list[str], value: str, timeout: float = 1.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if value in touch_calls:
            return
        time.sleep(0.01)
    pytest.fail(f"Timed out waiting for touch activity {value!r}; calls={touch_calls!r}")


def test_compression_activity_heartbeat_touches_agent_during_long_compress(tmp_path: Path) -> None:
    """Long compression must refresh agent activity so gateway watchdogs do not fire."""
    db = SessionDB(db_path=tmp_path / "state.db")
    session_id = "HEARTBEAT_TEST"
    db.create_session(session_id, source="test")

    agent = _build_agent_with_db(db, session_id)
    agent._compression_activity_heartbeat_interval = 0.1
    touch_calls: list[str] = []
    agent._touch_activity = lambda desc, **_kw: touch_calls.append(desc)

    def _slow_compress(*_a, **_kw):
        _wait_for_touch(touch_calls, "context compression in progress")
        return [
            {"role": "user", "content": "[CONTEXT COMPACTION] summary"},
            {"role": "user", "content": "tail"},
        ]

    agent.context_compressor.compress.side_effect = _slow_compress
    messages = [{"role": "user", "content": f"m{i}"} for i in range(20)]

    agent._compress_context(messages, "sys", approx_tokens=120_000)

    assert touch_calls[0] == "context compression started"
    assert "context compression in progress" in touch_calls
    assert touch_calls[-1] == "context compression completed"
    assert db.get_compression_lock_holder(session_id) is None


def test_compression_activity_heartbeat_stops_on_compress_exception(tmp_path: Path) -> None:
    """Exception paths must stop the heartbeat and release the compression lock."""
    db = SessionDB(db_path=tmp_path / "state.db")
    session_id = "HEARTBEAT_FAIL_TEST"
    db.create_session(session_id, source="test")

    agent = _build_agent_with_db(db, session_id)
    agent._compression_activity_heartbeat_interval = 0.1
    touch_calls: list[str] = []
    agent._touch_activity = lambda desc, **_kw: touch_calls.append(desc)

    def _failing_compress(*_a, **_kw):
        _wait_for_touch(touch_calls, "context compression in progress")
        raise RuntimeError("compress boom")

    agent.context_compressor.compress.side_effect = _failing_compress
    messages = [{"role": "user", "content": f"m{i}"} for i in range(20)]

    with pytest.raises(RuntimeError, match="compress boom"):
        agent._compress_context(messages, "sys", approx_tokens=120_000)

    assert touch_calls[0] == "context compression started"
    assert "context compression in progress" in touch_calls
    assert touch_calls[-1] == "context compression failed"
    assert db.get_compression_lock_holder(session_id) is None


def test_compression_activity_heartbeat_ignores_touch_errors(tmp_path: Path) -> None:
    """Activity touch failures must not affect compression success semantics."""
    db = SessionDB(db_path=tmp_path / "state.db")
    session_id = "HEARTBEAT_TOUCH_ERROR_TEST"
    db.create_session(session_id, source="test")

    agent = _build_agent_with_db(db, session_id)
    agent._compression_activity_heartbeat_interval = 0.1
    agent._touch_activity = lambda _desc, **_kw: (_ for _ in ()).throw(RuntimeError("touch boom"))
    messages = [{"role": "user", "content": f"m{i}"} for i in range(20)]

    compressed, _sp = agent._compress_context(messages, "sys", approx_tokens=120_000)

    assert compressed[0]["content"] == "[CONTEXT COMPACTION] summary"
    assert db.get_compression_lock_holder(session_id) is None


def test_compression_activity_heartbeat_strict_signature_fallback_releases_lock(tmp_path: Path) -> None:
    """Strict compressor signatures still compress while heartbeat cleanup runs.

    Main inspects the engine signature up front (_supported_compression_kwargs)
    instead of catching TypeError, so a strict-signature engine is invoked
    exactly once with only the kwargs it accepts. The heartbeat (with a
    non-numeric configured interval falling back to the default) must still
    wrap the call and stop cleanly.
    """
    db = SessionDB(db_path=tmp_path / "state.db")
    session_id = "HEARTBEAT_TYPEERROR_TEST"
    db.create_session(session_id, source="test")

    agent = _build_agent_with_db(db, session_id)
    agent._compression_activity_heartbeat_interval = "not-a-number"
    touch_calls: list[str] = []
    agent._touch_activity = lambda desc, **_kw: touch_calls.append(desc)
    messages = [{"role": "user", "content": f"m{i}"} for i in range(20)]

    strict_calls: list[int | None] = []

    def _strict_compress(messages, current_tokens=None):
        strict_calls.append(current_tokens)
        return [
            {"role": "user", "content": "[CONTEXT COMPACTION] strict summary"},
            {"role": "user", "content": "tail"},
        ]

    agent.context_compressor.compress = _strict_compress

    compressed, _sp = agent._compress_context(messages, "sys", approx_tokens=120_000)

    assert compressed[0]["content"] == "[CONTEXT COMPACTION] strict summary"
    assert touch_calls[0] == "context compression started"
    assert touch_calls[-1] == "context compression completed"
    assert db.get_compression_lock_holder(session_id) is None
    assert strict_calls == [120_000]


def test_compression_activity_heartbeat_nonfinite_interval_falls_back(tmp_path: Path) -> None:
    """Non-finite heartbeat intervals must not reach Event.wait()."""
    from agent.conversation_compression import _CompressionActivityHeartbeat

    db = SessionDB(db_path=tmp_path / "state.db")
    session_id = "HEARTBEAT_NONFINITE_INTERVAL_TEST"
    db.create_session(session_id, source="test")

    agent = _build_agent_with_db(db, session_id)
    touch_calls: list[str] = []
    touch_provenances: list = []

    def _capture(desc, *, provenance=None, force_persist=False):
        touch_calls.append(desc)
        touch_provenances.append(provenance)

    agent._touch_activity = _capture

    heartbeat = _CompressionActivityHeartbeat(agent, interval_seconds=float("inf"))

    assert heartbeat._interval_seconds == 60.0
    heartbeat.start()
    heartbeat.stop()
    assert touch_calls == ["context compression started", "context compression completed"]
    from agent.session_activity import ActivityProvenance

    assert touch_provenances == [
        ActivityProvenance.AGENT_COMPRESSION,
        ActivityProvenance.AGENT_COMPRESSION,
    ]


def test_compression_heartbeat_stop_persists_completed_over_in_progress(
    tmp_path: Path,
) -> None:
    """/compress is outside run_conversation, so turn-end clear never runs.

    Heartbeat progress stamps persist to SessionDB; completion is often
    rate-limited out. stop() must force-persist the terminal label so idle
    sessions show "context compression completed", not a stale "in progress".
    """
    from agent.conversation_compression import _CompressionActivityHeartbeat
    from agent.session_activity import ActivityProvenance

    db = SessionDB(db_path=tmp_path / "state.db")
    session_id = "HEARTBEAT_PERSIST_COMPLETED_TEST"
    db.create_session(session_id, source="test")

    agent = _build_agent_with_db(db, session_id)
    # Long interval: only start/stop touch; we inject the progress stamp.
    hb = _CompressionActivityHeartbeat(agent, interval_seconds=3600.0)
    hb.start()

    agent._session_activity_last_persist_mono = 0.0
    agent._touch_activity(
        "context compression in progress",
        provenance=ActivityProvenance.AGENT_COMPRESSION,
    )
    row = db.get_session(session_id)
    assert row["last_activity_description"] == "context compression in progress"
    assert row["last_activity_provenance"] == ActivityProvenance.AGENT_COMPRESSION.value

    # Mimic the common case: completion falls inside the 60s persist window.
    agent._session_activity_last_persist_mono = time.monotonic()
    hb.stop("context compression completed")

    row = db.get_session(session_id)
    assert row["last_activity_description"] == "context compression completed"
    assert row["last_activity_provenance"] == ActivityProvenance.AGENT_COMPRESSION.value
    assert agent._last_activity_desc == "context compression completed"
    assert agent._last_activity_provenance is ActivityProvenance.AGENT_COMPRESSION


def test_compression_heartbeat_does_not_clobber_timeout_provenance() -> None:
    """Detached heartbeat/stop must not overwrite a host timeout stamp."""
    from types import SimpleNamespace

    from agent.conversation_compression import _CompressionActivityHeartbeat
    from agent.session_activity import ActivityProvenance

    agent = SimpleNamespace(
        _last_activity_provenance=ActivityProvenance.AGENT_COMPRESSION_TIMEOUT,
        _last_activity_desc="context compression timed out",
        touches=[],
    )

    def _touch(desc, *, provenance=None, force_persist=False):
        agent.touches.append((desc, provenance))
        agent._last_activity_provenance = provenance
        agent._last_activity_desc = desc

    agent._touch_activity = _touch

    hb = _CompressionActivityHeartbeat(agent, interval_seconds=60.0)
    hb._touch("context compression in progress")
    hb.stop("context compression completed")

    assert agent.touches == []
    assert agent._last_activity_provenance is ActivityProvenance.AGENT_COMPRESSION_TIMEOUT
    assert agent._last_activity_desc == "context compression timed out"


def test_compression_heartbeat_does_not_clobber_cooldown_provenance() -> None:
    """Cooldown/abort stamps must also survive a late heartbeat stop."""
    from types import SimpleNamespace

    from agent.conversation_compression import _CompressionActivityHeartbeat
    from agent.session_activity import ActivityProvenance

    agent = SimpleNamespace(
        _last_activity_provenance=ActivityProvenance.AGENT_COMPRESSION_COOLDOWN,
        _last_activity_desc="compression blocked (cooldown: 30s remaining)",
        touches=[],
    )

    def _touch(desc, *, provenance=None, force_persist=False):
        agent.touches.append((desc, provenance))
        agent._last_activity_provenance = provenance
        agent._last_activity_desc = desc

    agent._touch_activity = _touch

    hb = _CompressionActivityHeartbeat(agent, interval_seconds=60.0)
    hb._touch("context compression in progress")
    hb.stop("context compression failed")

    assert agent.touches == []
    assert agent._last_activity_provenance is ActivityProvenance.AGENT_COMPRESSION_COOLDOWN


def test_compression_heartbeat_start_republishes_after_terminal_provenance() -> None:
    """A new compression episode may overwrite a prior timeout/cooldown stamp."""
    from types import SimpleNamespace

    from agent.conversation_compression import _CompressionActivityHeartbeat
    from agent.session_activity import ActivityProvenance

    agent = SimpleNamespace(
        _last_activity_provenance=ActivityProvenance.AGENT_COMPRESSION_TIMEOUT,
        _last_activity_desc="context compression timed out",
        touches=[],
    )

    def _touch(desc, *, provenance=None, force_persist=False):
        agent.touches.append((desc, provenance))
        agent._last_activity_provenance = provenance
        agent._last_activity_desc = desc

    agent._touch_activity = _touch

    hb = _CompressionActivityHeartbeat(agent, interval_seconds=60.0)
    hb.start()
    hb.stop()

    assert agent.touches[0] == (
        "context compression started",
        ActivityProvenance.AGENT_COMPRESSION,
    )
    assert agent.touches[-1] == (
        "context compression completed",
        ActivityProvenance.AGENT_COMPRESSION,
    )
    assert agent._last_activity_provenance is ActivityProvenance.AGENT_COMPRESSION


def test_compression_heartbeat_does_not_rearm_after_unknown_provenance() -> None:
    """After a terminal stamp, UNKNOWN must not re-arm a detached heartbeat."""
    from types import SimpleNamespace

    from agent.conversation_compression import _CompressionActivityHeartbeat
    from agent.session_activity import ActivityProvenance

    agent = SimpleNamespace(
        _last_activity_provenance=ActivityProvenance.AGENT_COMPRESSION_TIMEOUT,
        _last_activity_desc="context compression timed out",
        touches=[],
    )

    def _touch(desc, *, provenance=None, force_persist=False):
        agent.touches.append((desc, provenance))
        agent._last_activity_provenance = provenance
        agent._last_activity_desc = desc

    agent._touch_activity = _touch

    hb = _CompressionActivityHeartbeat(agent, interval_seconds=60.0)
    # First tick observes TIMEOUT and latches silent.
    hb._touch("context compression in progress")
    assert hb._suppressed is True
    # Turn continues / ends and clears labels to UNKNOWN — must stay silent.
    agent._last_activity_provenance = ActivityProvenance.UNKNOWN
    agent._last_activity_desc = "calling model"
    hb._touch("context compression in progress")
    hb.stop("context compression completed")

    assert agent.touches == []
    assert agent._last_activity_provenance is ActivityProvenance.UNKNOWN
    assert agent._last_activity_desc == "calling model"


def test_compression_heartbeat_stops_when_commit_fence_cancelled() -> None:
    """Host fence cancel must silence detached heartbeat refresh and late stop."""
    from types import SimpleNamespace

    from agent.conversation_compression import (
        CompressionCommitFence,
        _CompressionActivityHeartbeat,
    )
    from agent.session_activity import ActivityProvenance

    agent = SimpleNamespace(
        _last_activity_provenance=ActivityProvenance.AGENT_COMPRESSION,
        _last_activity_desc="context compression started",
        touches=[],
    )

    def _touch(desc, *, provenance=None, force_persist=False):
        agent.touches.append((desc, provenance))
        agent._last_activity_provenance = provenance
        agent._last_activity_desc = desc

    agent._touch_activity = _touch

    fence = CompressionCommitFence()
    assert fence.cancel_before_commit() is True

    hb = _CompressionActivityHeartbeat(
        agent, interval_seconds=60.0, commit_fence=fence
    )
    hb._touch("context compression in progress")
    hb.stop("context compression completed")

    assert agent.touches == []
    assert hb._suppressed is True
    assert agent._last_activity_provenance is ActivityProvenance.AGENT_COMPRESSION
    assert agent._last_activity_desc == "context compression started"


def test_concurrent_compression_does_not_fork_session(tmp_path: Path) -> None:
    """Two AIAgents that share a session_id MUST NOT both rotate it.

    Without the per-session compression lock this fixture deterministically
    produces 2 child sessions (transcript fork). With the lock at most one
    path rotates: normally exactly 1 canonical child, or — under heavy DB
    write contention that makes the winner's child create_session exhaust its
    retries — 0, because _compress_context safely rolls back to the parent
    instead of orphaning a child. The forbidden outcome is 2+ (the fork).
    """
    db = SessionDB(db_path=tmp_path / "state.db")

    parent_sid = "PARENT_TEST_SESSION"
    db.create_session(parent_sid, source="discord")

    # Two agents on the same session_id, both wired to the same db —
    # mirrors the parent-turn agent + the background-review fork right
    # after a turn ends.
    agent_a = _build_agent_with_db(db, parent_sid)
    agent_b = _build_agent_with_db(db, parent_sid)
    messages = [{"role": "user", "content": f"m{i}"} for i in range(20)]

    def run(agent):
        try:
            agent._compress_context(messages, "sys", approx_tokens=120_000)
        except Exception:
            # Surface to the test if either raises — should not happen.
            raise

    t_a = threading.Thread(target=run, args=(agent_a,), name="main_turn")
    t_b = threading.Thread(target=run, args=(agent_b,), name="review_fork")
    t_a.start()
    t_b.start()
    t_a.join(timeout=10)
    t_b.join(timeout=10)

    # The invariant Damien's incident is about: the parent must NEVER end up
    # with two (or more) children — that is the transcript fork. The lock
    # guarantees only one path rotates.
    #
    # Zero children is also a valid, non-forking outcome: under heavy DB write
    # contention the winner's child ``create_session`` can exhaust its retry
    # budget, and ``_compress_context`` deliberately rolls the live id back to
    # the (still-indexed) parent rather than stranding an orphan child — see
    # the create-failure rollback in agent/conversation_compression.py. That
    # safe rollback leaves 0 children and is correct. So the contract is
    # ``children <= 1``; only ``>= 2`` is the bug. Asserting an exact ``== 1``
    # made this test flaky under the concurrent CI load that triggers the
    # contention rollback (#54465 churn surfaced it).
    n_children = _count_children(db, parent_sid)
    assert n_children <= 1, (
        f"Compression lock failed: parent session has {n_children} children in "
        "state.db (transcript fork). This is Damien's incident shape — see the "
        "test docstring. Two or more children means the lock did not serialize "
        "the concurrent rotations."
    )

    # Every agent that moved off the parent must have landed on the SAME id.
    # Counting movers is the wrong contract: the loser can legitimately end up
    # on the child too, without rotating anything itself — it takes the lock
    # after the winner released it, sees the parent was already rotated, and
    # _adopt_live_compression_child() points it at the winner's single child
    # (the "compression recovery: stale session=... adopted live child=..."
    # path). That convergence is the fix working, not a fork; the fork is two
    # DIFFERENT live ids. Asserting ``movers <= 1`` failed on that healthy
    # outcome under concurrent load.
    moved = {a.session_id for a in (agent_a, agent_b) if a.session_id != parent_sid}
    assert len(moved) <= 1, (
        f"Expected at most one post-compression session id, got {sorted(moved)}. "
        "Two distinct ids means the lock didn't serialize them (transcript fork)."
    )
    assert len(moved) == n_children, (
        f"Inconsistent state: agents live on {sorted(moved)} but {n_children} "
        "child session(s) exist — rotation and child creation diverged."
    )
    if moved:
        child = _live_child_id(db, parent_sid)
        assert moved == {child}, (
            f"Agents live on {sorted(moved)} but the parent's only child is "
            f"{child} — an agent is writing to a session outside the lineage."
        )

    # The lock must be released after both paths finished, regardless of
    # whether the winner committed a child or rolled back.
    assert db.get_compression_lock_holder(parent_sid) is None, (
        "Compression lock leaked: still held after both paths completed."
    )


def test_skipped_compression_returns_messages_unchanged(tmp_path: Path) -> None:
    """A holder that never finishes must not stall or mutate the loser."""
    db = SessionDB(db_path=tmp_path / "state.db")
    parent_sid = "LOSER_TEST"
    db.create_session(parent_sid, source="discord")

    held = db.try_acquire_compression_lock(parent_sid, "external_holder")
    assert held is True

    agent = _build_agent_with_db(db, parent_sid)
    messages = [
        {"role": "user", "content": "m1"},
        {"role": "user", "content": "m2"},
    ]

    compressed, _sp = agent._compress_context(
        messages, "sys", approx_tokens=120_000
    )

    assert compressed is messages or compressed == messages
    assert agent.session_id == parent_sid
    agent.context_compressor.compress.assert_not_called()


def test_durable_message_committed_before_lease_is_adopted(
    tmp_path: Path,
) -> None:
    """A durable row absent from the caller snapshot must still be compressed.

    Previously this path aborted and returned the stale snapshot unchanged,
    which permanently wedged busy sessions: every compress attempt saw the
    DB ahead of the in-memory list, logged "changed before lease
    acquisition", and never called the compressor. Adopting the durable
    transcript keeps the late-committed turn and lets compression proceed.
    """
    db = SessionDB(db_path=tmp_path / "state.db")
    parent_sid = "PRE_LEASE_DURABLE_RACE"
    db.create_session(parent_sid, source="webui")
    db.append_message(parent_sid, "user", "old durable")

    # Frontend takes its snapshot, then another producer commits before this
    # compressor acquires the lease.
    stale_snapshot = [{"role": "user", "content": "old durable"}]
    db.append_message(parent_sid, "assistant", "late committed before lease")
    agent = _build_agent_with_db(db, parent_sid)

    returned, _system_prompt = agent._compress_context(
        stale_snapshot, "sys", approx_tokens=120_000
    )

    agent.context_compressor.compress.assert_called_once()
    compressed_arg = agent.context_compressor.compress.call_args.args[0]
    assert [m["content"] for m in compressed_arg] == [
        "old durable",
        "late committed before lease",
    ]
    # Must not echo the stale snapshot — compression proceeded on the
    # adopted durable transcript (rotation publishes a child session).
    assert returned is not stale_snapshot
    assert returned[0]["content"] == "[CONTEXT COMPACTION] summary"
    assert agent.session_id != parent_sid
    child_id = _live_child_id(db, parent_sid)
    assert child_id is not None
    assert child_id == agent.session_id





def test_fence_cancelled_compression_leaves_lock_reacquirable(tmp_path: Path) -> None:
    """A fence-cancelled attempt must not poison the per-session lock.

    Lock-release verification for the hygiene-timeout path: after the gateway
    times out and cancels a hygiene compression at the commit fence, the very
    next attempt on the same session (e.g. the user running ``/compress``)
    must acquire the compression lock and commit normally. A leaked lock here
    would silently block every future compaction for the session until TTL
    expiry.
    """
    from agent.conversation_compression import CompressionCommitFence

    db = SessionDB(db_path=tmp_path / "state.db")
    session_id = "HYGIENE_LOCK_REACQUIRE"
    db.create_session(session_id, source="telegram")

    agent = _build_agent_with_db(db, session_id)
    agent.compression_in_place = True
    agent._cached_system_prompt = "sys"
    summary_started = threading.Event()
    release_summary = threading.Event()

    def _slow_summary(*_args, **_kwargs):
        summary_started.set()
        assert release_summary.wait(timeout=5)
        agent.context_compressor._proactive_prune_rearm_tokens = 0
        return [
            {"role": "user", "content": "[CONTEXT COMPACTION] summary"},
            {"role": "user", "content": "tail"},
        ]

    agent.context_compressor.compress.side_effect = _slow_summary
    agent.context_compressor._proactive_prune_rearm_tokens = 120_000
    messages = [{"role": "user", "content": f"m{i}"} for i in range(20)]
    fence = CompressionCommitFence()
    result = {}

    def _run_compression() -> None:
        result["value"] = agent._compress_context(
            messages,
            "sys",
            approx_tokens=120_000,
            commit_fence=fence,
        )

    worker = threading.Thread(target=_run_compression, name="fenced-hygiene")
    worker.start()
    assert summary_started.wait(timeout=2)
    assert fence.cancel_before_commit() is True
    release_summary.set()
    worker.join(timeout=5)
    assert not worker.is_alive()

    # Cancelled attempt: no mutation, and — the invariant under test — the
    # per-session compression lock is fully released.
    assert result["value"][0] is messages
    assert agent.context_compressor._proactive_prune_rearm_tokens == 120_000
    assert db.get_compression_lock_holder(session_id) is None

    # The NEXT attempt (no fence — a manual /compress retry) must be able to
    # acquire the lock and commit an in-place compaction normally.
    agent.context_compressor.compress.side_effect = lambda *_a, **_kw: [
        {"role": "user", "content": "[CONTEXT COMPACTION] retry summary"},
        {"role": "user", "content": "tail"},
    ]
    retried, _sp = agent._compress_context(messages, "sys", approx_tokens=120_000)

    assert retried is not messages
    assert len(retried) < len(messages)
    assert agent.session_id == session_id  # in-place: same session id
    assert agent._last_compaction_in_place is True
    assert db.get_compression_lock_holder(session_id) is None


def test_commit_fence_waits_for_an_active_commit() -> None:
    """A timeout that loses the fence race cannot overlap the live turn."""
    from agent.conversation_compression import CompressionCommitFence

    fence = CompressionCommitFence()
    assert fence.begin_commit() is True
    assert fence.try_cancel_before_commit() is None
    cancel_started = threading.Event()
    cancel_finished = threading.Event()
    result = {}

    def _cancel() -> None:
        cancel_started.set()
        result["cancelled"] = fence.cancel_before_commit()
        cancel_finished.set()

    waiter = threading.Thread(target=_cancel, name="hygiene-timeout-fence")
    waiter.start()
    try:
        assert cancel_started.wait(timeout=2)
        assert not cancel_finished.is_set()
    finally:
        fence.finish_commit()
    waiter.join(timeout=2)

    assert not waiter.is_alive()
    assert result["cancelled"] is False


def test_delayed_contender_adopts_unique_rotated_child(tmp_path: Path) -> None:
    """A stale agent must continue on the winner's compacted child transcript."""
    db = SessionDB(db_path=tmp_path / "state.db")
    parent_sid = "STALE_PARENT"
    child_sid = "CANONICAL_CHILD"
    db.create_session(parent_sid, source="webui")
    db.end_session(parent_sid, "compression")
    db.create_session(child_sid, source="webui", parent_session_id=parent_sid)
    compacted = [
        {"role": "user", "content": "[CONTEXT COMPACTION] summary"},
        {"role": "assistant", "content": "compacted tail"},
    ]
    db.replace_messages(child_sid, compacted)

    agent = _build_agent_with_db(db, parent_sid)
    stale_messages = [
        {"role": "user", "content": "stale"},
        {"role": "assistant", "content": "x" * 1000},
    ]
    recovered, _system_prompt = agent._compress_context(
        stale_messages, "sys", approx_tokens=120_000
    )

    assert agent.session_id == child_sid
    assert [(m["role"], m["content"]) for m in recovered] == [
        ("user", "[CONTEXT COMPACTION] summary"),
        ("assistant", "compacted tail"),
    ]
    assert agent._session_db_created is True
    assert agent._flushed_db_message_session_id == child_sid
    assert agent._last_flushed_db_idx == len(recovered)
    agent.context_compressor.compress.assert_not_called()
    lifecycle_args, lifecycle_kwargs = agent.context_compressor.on_session_start.call_args
    assert lifecycle_args == (child_sid,)
    assert lifecycle_kwargs["boundary_reason"] == "compression"
    assert lifecycle_kwargs["old_session_id"] == parent_sid
    assert lifecycle_kwargs["session_db"] is db








def _no_consecutive_user_roles(messages: list) -> bool:
    roles = [m.get("role") for m in messages if isinstance(m, dict)]
    return all(
        not (roles[i] == roles[i + 1] == "user") for i in range(len(roles) - 1)
    )


def test_restored_anchor_never_creates_consecutive_user_roles() -> None:
    """Anchor restoration must preserve strict role alternation (#55677).

    The original insertion helper could land the human anchor directly next
    to user-role scaffolding (index-0 insert before a leading synthetic user
    turn, or a bare scaffolding-only transcript), producing user/user
    adjacency that strict chat templates reject.
    """
    from agent.conversation_compression import _insert_real_user_anchor

    anchor = {"role": "user", "content": "REAL HUMAN ASK"}

    # Leading synthetic user turn before the assistant summary.
    compressed = [
        {
            "role": "user",
            "content": "[System: Your previous response was truncated ...]",
            "_empty_recovery_synthetic": True,
        },
        {"role": "assistant", "content": "summary"},
        {
            "role": "user",
            "content": "[Your active task list was preserved across context compression]",
            "_todo_snapshot_synthetic": True,
        },
    ]
    _insert_real_user_anchor(compressed, dict(anchor))
    assert _no_consecutive_user_roles(compressed)
    assert any(m.get("content", "").startswith("REAL HUMAN ASK") for m in compressed)

    # Scaffolding-only transcript: the anchor is merged, not inserted
    # adjacent, and the merged turn leads with the human ask.
    compressed = [
        {
            "role": "user",
            "content": "[Your active task list was preserved across context compression]",
            "_todo_snapshot_synthetic": True,
        },
    ]
    _insert_real_user_anchor(compressed, dict(anchor))
    assert _no_consecutive_user_roles(compressed)
    assert len(compressed) == 1
    assert compressed[0]["content"].startswith("REAL HUMAN ASK")
    assert not compressed[0].get("_todo_snapshot_synthetic")




def test_compression_persists_child_handoff_immediately(tmp_path: Path) -> None:
    db = SessionDB(db_path=tmp_path / "state.db")
    parent_sid = "HEADLESS_PREFLIGHT_PARENT"
    db.create_session(parent_sid, source="cli")

    agent = _build_agent_with_db(db, parent_sid)
    messages = [{"role": "user", "content": f"m{i}"} for i in range(20)]

    compressed, _sp = agent._compress_context(messages, "sys", approx_tokens=120_000)
    child_sid = agent.session_id

    assert child_sid != parent_sid
    assert db.get_session(parent_sid)["end_reason"] == "compression"
    assert len(db.get_messages(child_sid)) == len(compressed)

    agent._flush_messages_to_session_db(compressed, None)
    assert len(db.get_messages(child_sid)) == len(compressed)




def test_rotation_publish_failure_restores_proactive_prune_runway(
    tmp_path: Path,
) -> None:
    db = SessionDB(db_path=tmp_path / "state.db")
    parent_sid = "PRUNE_RUNWAY_ROLLBACK_PARENT"
    db.create_session(
        parent_sid,
        source="cli",
        model_config={"keep": "value", "_proactive_prune_rearm_tokens": 120_000},
    )
    messages = [{"role": "user", "content": f"m{i}"} for i in range(20)]
    db.append_messages_batch(parent_sid, messages)
    for message in messages:
        message["_db_persisted"] = True
    agent = _build_agent_with_db(db, parent_sid)
    agent.context_compressor._proactive_prune_rearm_tokens = 120_000

    def _compress(*_args, **_kwargs):
        agent.context_compressor._proactive_prune_rearm_tokens = 0
        return [
            {"role": "user", "content": "[CONTEXT COMPACTION] summary"},
            {"role": "user", "content": "tail"},
        ]

    agent.context_compressor.compress.side_effect = _compress
    durable_before = db.get_messages_as_conversation(parent_sid)
    with patch.object(
        db,
        "publish_compression_child",
        side_effect=RuntimeError("publish failed"),
    ):
        returned, _sp = agent._compress_context(
            messages, "sys", approx_tokens=120_000,
        )

    assert returned is messages
    assert agent.session_id == parent_sid
    assert agent.context_compressor._proactive_prune_rearm_tokens == 120_000
    assert db.get_messages_as_conversation(parent_sid) == durable_before
    assert json.loads(db.get_session(parent_sid)["model_config"]) == {
        "keep": "value",
        "_proactive_prune_rearm_tokens": 120_000,
    }


def test_full_in_place_compression_atomically_clears_durable_prune_runway(
    tmp_path: Path,
) -> None:
    db = SessionDB(db_path=tmp_path / "state.db")
    session_id = "IN_PLACE_CLEARS_PRUNE_RUNWAY"
    db.create_session(
        session_id,
        source="cli",
        model_config={"keep": "value", "_proactive_prune_rearm_tokens": 120_000},
    )
    messages = [{"role": "user", "content": f"m{i}"} for i in range(20)]
    db.append_messages_batch(session_id, messages)
    agent = _build_agent_with_db(db, session_id)
    agent.compression_in_place = True
    agent.context_compressor._proactive_prune_rearm_tokens = 120_000

    compressed, _sp = agent._compress_context(
        messages, "sys", approx_tokens=120_000,
    )

    assert agent.session_id == session_id
    assert [message["content"] for message in db.get_messages_as_conversation(session_id)] == [
        message["content"] for message in compressed
    ]
    assert json.loads(db.get_session(session_id)["model_config"]) == {"keep": "value"}


def test_rotation_child_starts_without_durable_prune_runway(tmp_path: Path) -> None:
    db = SessionDB(db_path=tmp_path / "state.db")
    parent_sid = "ROTATION_CLEARS_PRUNE_RUNWAY"
    db.create_session(
        parent_sid,
        source="cli",
        model_config={"keep": "parent", "_proactive_prune_rearm_tokens": 120_000},
    )
    messages = [{"role": "user", "content": f"m{i}"} for i in range(20)]
    db.append_messages_batch(parent_sid, messages)
    agent = _build_agent_with_db(db, parent_sid)

    agent._compress_context(messages, "sys", approx_tokens=120_000)

    assert agent.session_id != parent_sid
    child_config = json.loads(db.get_session(agent.session_id)["model_config"])
    assert "_proactive_prune_rearm_tokens" not in child_config
    assert json.loads(db.get_session(parent_sid)["model_config"])[
        "_proactive_prune_rearm_tokens"
    ] == 120_000


@pytest.mark.parametrize("in_place", [False, True])
def test_equal_copy_compression_result_does_not_rewrite_session(
    tmp_path: Path,
    in_place: bool,
) -> None:
    db = SessionDB(db_path=tmp_path / "state.db")
    parent_sid = f"EQUAL_COPY_NOOP_{in_place}"
    db.create_session(parent_sid, source="cli")

    agent = _build_agent_with_db(db, parent_sid)
    setattr(agent, "compression_in_place", in_place)
    messages = [{"role": "user", "content": f"m{i}"} for i in range(20)]
    compressor = getattr(agent, "context_compressor")
    compressor.compress.side_effect = lambda incoming, **_kw: list(incoming)

    with patch.object(
        db,
        "archive_and_compact",
        wraps=db.archive_and_compact,
    ) as archive_and_compact:
        returned, _sp = agent._compress_context(
            messages,
            "sys",
            approx_tokens=120_000,
        )

    assert returned is messages
    assert getattr(agent, "session_id") == parent_sid
    assert _count_children(db, parent_sid) == 0
    parent = db.get_session(parent_sid)
    assert parent is not None
    assert parent["end_reason"] is None
    assert db.get_compression_lock_holder(parent_sid) is None
    archive_and_compact.assert_not_called()




def test_post_compress_exception_stops_lock_refresher(tmp_path: Path, monkeypatch) -> None:
    """A warning-path exception after compress() returns must still release the lock."""
    real_try_acquire = SessionDB.try_acquire_compression_lock

    def _short_ttl(self, session_id: str, holder: str, ttl_seconds: float = 300.0) -> bool:
        return real_try_acquire(self, session_id, holder, ttl_seconds=0.15)

    monkeypatch.setattr(SessionDB, "try_acquire_compression_lock", _short_ttl)

    db = SessionDB(db_path=tmp_path / "state.db")
    parent_sid = "REFRESH_EXCEPTION_TEST"
    db.create_session(parent_sid, source="discord")

    agent = _build_agent_with_db(db, parent_sid)
    agent._compression_lock_ttl_seconds = 0.15
    agent._compression_lock_refresh_interval = 0.05
    agent.context_compressor._last_summary_error = "summary failed"
    agent._emit_warning = lambda *_a, **_k: (_ for _ in ()).throw(RuntimeError("warn boom"))

    messages = [{"role": "user", "content": f"m{i}"} for i in range(20)]

    with pytest.raises(RuntimeError, match="warn boom"):
        agent._compress_context(messages, "sys", approx_tokens=120_000)

    time.sleep(0.25)
    assert db.try_acquire_compression_lock(parent_sid, "probe", ttl_seconds=1.0) is True








def test_signature_introspection_exception_releases_lock_and_refresher(
    tmp_path: Path, monkeypatch
) -> None:
    """Capability inspection failures must not leak the acquired lock lease."""
    from agent.conversation_compression import (
        _CompressionLockLeaseRefresher as RealLeaseRefresher,
    )

    refreshers = []

    class RecordingLeaseRefresher(RealLeaseRefresher):
        def start(self):
            refreshers.append(self)
            return super().start()

    monkeypatch.setattr(
        "agent.conversation_compression._CompressionLockLeaseRefresher",
        RecordingLeaseRefresher,
    )

    db = SessionDB(db_path=tmp_path / "state.db")
    parent_sid = "SIGNATURE_EXCEPTION_TEST"
    db.create_session(parent_sid, source="discord")

    agent = _build_agent_with_db(db, parent_sid)
    agent._compression_lock_refresh_interval = 0.1

    class SignatureBomb:
        calls = 0

        @property
        def __signature__(self):
            raise RuntimeError("signature boom")

        def __call__(self, *_args, **_kwargs):
            self.calls += 1
            raise AssertionError("engine must not run after signature failure")

    bomb = SignatureBomb()
    agent.context_compressor.compress = bomb
    messages = [{"role": "user", "content": f"m{i}"} for i in range(20)]

    with pytest.raises(RuntimeError, match="signature boom"):
        agent._compress_context(messages, "sys", approx_tokens=120_000)

    assert bomb.calls == 0
    assert db.get_compression_lock_holder(parent_sid) is None
    assert len(refreshers) == 1
    assert not refreshers[0]._thread.is_alive()








def _make_legacy_session_db_class() -> type:
    """Model the class retained in ``sys.modules`` before the lock API existed.

    During the real version-skew incident, a re-imported compression module
    imports the same still-loaded ``hermes_state`` module, whose ``SessionDB``
    class is old. The test replaces that module attribute with this lockless
    class and forwards all persistence operations to a current real database.
    """
    source_path = inspect.getfile(SessionDB)
    namespace = {"__name__": "hermes_state"}
    source = '''
class SessionDB:
    def __init__(self, real_db):
        self._real = real_db

    def __getattribute__(self, name):
        if name in {"_real", "__class__"}:
            return object.__getattribute__(self, name)
        return getattr(object.__getattribute__(self, "_real"), name)
'''
    exec(compile(source, source_path, "exec"), namespace)
    return namespace["SessionDB"]


class _NominalSessionDBImpostor:
    """A proxy that spoofs names but lacks the real SessionDB source contract."""

    def __init__(self, real_db: SessionDB) -> None:
        self._real = real_db

    def create_session(self, *args, **kwargs):
        return self._real.create_session(*args, **kwargs)

    def __getattr__(self, name):
        if name == "try_acquire_compression_lock":
            raise AttributeError(name)
        return getattr(self._real, name)


_NominalSessionDBImpostor.__module__ = "hermes_state"
_NominalSessionDBImpostor.__name__ = "SessionDB"


class _BrokenLockLookupDB:
    """A present lock API whose instance lookup fails unexpectedly."""

    def __init__(self, real_db: SessionDB, error: Exception) -> None:
        self._real = real_db
        self._error = error

    def try_acquire_compression_lock(self, *_args, **_kwargs):
        raise AssertionError("the broken lookup must not resolve to a callable")

    def __getattribute__(self, name):
        if name == "try_acquire_compression_lock":
            raise object.__getattribute__(self, "_error")
        if name in {"_real", "_error", "__class__"}:
            return object.__getattribute__(self, name)
        return getattr(object.__getattribute__(self, "_real"), name)


class _NonCallableLockAPI:
    """A present lock API descriptor that resolves to a non-callable value."""

    def __init__(self, real_db: SessionDB) -> None:
        self._real = real_db

    try_acquire_compression_lock = None

    def __getattr__(self, name):
        return getattr(self._real, name)










@pytest.mark.parametrize(
    "error",
    [
        RuntimeError("simulated lock-table corruption"),
        AttributeError("simulated internal lock attribute error"),
        TypeError("simulated internal lock type error"),
    ],
)
def test_real_lock_api_internal_errors_fail_closed_skips_compression(
    tmp_path: Path, monkeypatch, error: Exception
) -> None:
    """Errors after a real lock API resolves must preserve session lineage.

    ``AttributeError`` only means version skew while resolving the method. This
    test injects failures beneath the real ``SessionDB.try_acquire...`` body,
    proving that an internal AttributeError or TypeError cannot take the
    structural-absence compatibility path.
    """
    db = SessionDB(db_path=tmp_path / "state.db")
    parent_sid = "ERRORING_LOCK_TEST"
    db.create_session(parent_sid, source="discord")

    def _fail_lock_write(_fn):
        raise error

    monkeypatch.setattr(db, "_execute_write", _fail_lock_write)
    agent = _build_agent_with_db(db, parent_sid)
    messages = [{"role": "user", "content": f"m{i}"} for i in range(20)]

    compressed, _sp = agent._compress_context(messages, "sys", approx_tokens=120_000)

    # Skipped: messages returned verbatim, no rotation, compressor never ran.
    assert compressed is messages or compressed == messages
    assert agent.session_id == parent_sid
    assert _count_children(db, parent_sid) == 0
    agent.context_compressor.compress.assert_not_called()




def test_review_fork_disables_compression_to_prevent_stale_parent_fork(tmp_path: Path) -> None:
    """The background-review fork must set ``compression_enabled = False``
    so it can never compress the parent it shares a session_id with
    (issue #38727).

    The per-session compression lock only serialises a SAME-WINDOW concurrent
    race. It does NOT stop a stale parent from being compressed again in a
    LATER turn: if ``review_agent`` had won the race, its new child session is
    never adopted by the gateway (the fork is single-lifecycle and dies right
    after one ``run_conversation``), so the foreground path would start the
    next turn from the stale parent and compress it AGAIN — leaving the same
    parent with two sibling children.

    The fix makes the review fork never trigger compression at all. Both
    compression trigger sites in ``agent/conversation_loop.py`` gate on
    ``agent.compression_enabled`` BEFORE calling ``_compress_context``:
      • preflight (``if agent.compression_enabled and len(messages) > ...``)
      • mid-loop  (``if agent.compression_enabled and _compressor.should_compress(...)``)
    so a fork with the flag cleared never reaches the rotation path.

    This test pins the contract at the source: ``_run_review_in_thread``
    must set ``review_agent.compression_enabled = False`` on the fork it
    builds. It calls the real worker synchronously with
    ``AIAgent.run_conversation`` patched (so no LLM call happens) and
    captures the constructed review agent to assert the flag.
    """
    import agent.background_review as br

    captured = {}

    def _fake_run_conversation(self, *_a, **_k):
        captured["compression_enabled"] = self.compression_enabled
        captured["session_id"] = self.session_id
        return {"final_response": "", "messages": []}

    parent_sid = "REVIEW_FORK_FLAG_TEST"

    db = SessionDB(db_path=tmp_path / "state.db")
    db.create_session(parent_sid, source="discord")
    parent = _build_agent_with_db(db, parent_sid)

    # The worker does a local ``from run_agent import AIAgent``; patching
    # the class method covers that import path.
    from run_agent import AIAgent

    with patch.object(AIAgent, "run_conversation", _fake_run_conversation):
        br._run_review_in_thread(
            parent,
            [{"role": "user", "content": "hi"}],
            "review this conversation",
        )

    assert captured, (
        "_run_review_in_thread never reached run_conversation — the spawn path "
        "changed; update this test to capture the review AIAgent."
    )
    assert captured["session_id"] == parent_sid, (
        "Review fork should inherit the parent's session_id (shared id is the "
        "whole reason compression must be disabled)."
    )
    assert captured["compression_enabled"] is False, (
        "FIX REGRESSION: background-review fork did NOT disable compression. "
        "It shares the parent's session_id, so an enabled fork can rotate the "
        "parent into an orphan child (issue #38727). The trigger gates in "
        "conversation_loop.py only short-circuit when compression_enabled is "
        "False — this flag MUST be cleared on the review fork."
    )
    db.close()


# ── Lease-refresher bounded-failure tolerance (salvage follow-up, #54465) ────
# A single falsy refresh (transient DB blip) must NOT permanently kill the
# lease — only a *persistent* failure (genuine lost-ownership) should stop the
# refresher after a bounded number of consecutive failures. Without this, one
# escaped lock-contention error silently reintroduces the TTL-expiry wedge the
# PR set out to fix.


class _FlakyRefreshDB:
    """A db whose refresh_compression_lock returns a scripted sequence."""

    def __init__(self, results):
        self._results = list(results)
        self.calls = 0

    def refresh_compression_lock(self, session_id, holder, ttl_seconds=300.0):
        self.calls += 1
        if self._results:
            return self._results.pop(0)
        return True  # steady-state success after the scripted prefix


def _no_sleep(refresher) -> None:
    """Make the refresher loop iterate without real wall-clock sleeps.

    ``_stop.wait(interval)`` returns False (keep looping) instantly instead of
    blocking for the (clamped) interval, so count-based tests stay fast and
    deterministic — the loop's termination is driven by the failure cap / the
    scripted db, not by timing.
    """
    refresher._stop.wait = lambda _interval: False  # type: ignore[assignment]








def test_lease_refresher_failure_window_is_bounded_by_ttl() -> None:
    """Persistent failure stops within one lease's worth of time, not forever.

    The contract (not a magic count): the give-up window
    ``cap * refresh_interval`` must be <= the TTL, so a stuck refresher can
    never hold the lock past its TTL. We assert that relationship directly
    rather than freezing a literal cap (behavior contract over snapshot).
    """
    from agent.conversation_compression import _CompressionLockLeaseRefresher

    ttl, interval = 10.0, 2.0  # cap should be int(10/2) = 5
    db = _FlakyRefreshDB([False] * 50)  # never recovers (lost ownership)
    refresher = _CompressionLockLeaseRefresher(
        db, "sess", "holder", ttl_seconds=ttl, refresh_interval_seconds=interval
    )
    _no_sleep(refresher)
    refresher._run()

    cap = refresher._max_consecutive_failures
    assert cap == int(ttl / interval), "cap must derive from ttl/interval"
    # Stops at the cap — not on the first failure, not forever.
    assert db.calls == cap
    # The invariant that makes the cap honest: total tolerance <= one TTL.
    assert cap * interval <= ttl, (
        f"give-up window {cap * interval}s must not exceed the lease TTL {ttl}s"
    )


def test_hard_interrupt_aborts_compression_and_unblocks_session_writes(tmp_path: Path) -> None:
    """Ctrl+C must abort an interrupt-protected summary without leaving the
    session write-blocked behind its compression lease."""
    from agent import auxiliary_client as aux

    db = SessionDB(db_path=tmp_path / "state.db")
    session_id = "HARD_INTERRUPT_COMPRESSION_TEST"
    db.create_session(session_id, source="cli")

    agent = _build_agent_with_db(db, session_id)
    messages = [{"role": "user", "content": f"m{i}"} for i in range(20)]
    original_messages = copy.deepcopy(messages)

    def _cancelled_compress(current, *_args, **_kwargs):
        agent._hard_interrupt_requested.set()
        assert aux._aux_interrupt_cancel_requested() is True
        # F3 isolation: the engine only ever sees the pooled snapshot, so an
        # in-place mutation lands on ``current`` (the worker's copy), never on
        # the caller's live list. The rollback contract is that the RETURNED
        # transcript equals the pre-compression one.
        current[0]["content"] = "must be rolled back"
        raise aux.AuxiliaryExplicitCancellation()

    agent.context_compressor.compress.side_effect = _cancelled_compress

    compressed, _prompt = agent._compress_context(
        messages, "sys", approx_tokens=120_000
    )

    assert compressed == original_messages
    assert messages == original_messages
    assert db.get_compression_lock_holder(session_id) is None
    db.append_message(session_id, "assistant", "writes recovered")


def test_late_hard_interrupt_restores_full_compressor_attempt_state_and_retry(
    tmp_path: Path,
) -> None:
    """A stop after provider success but before compress() returns is a true no-op."""
    db = SessionDB(db_path=tmp_path / "state.db")
    session_id = "LATE_HARD_INTERRUPT_STATE_TEST"
    db.create_session(session_id, source="cli")
    agent = _build_agent_with_db(db, session_id)
    agent.compression_in_place = True
    agent._cached_system_prompt = "sys"
    messages = [{"role": "user", "content": f"m{i}"} for i in range(20)]
    provider_returned = threading.Event()
    allow_compress_return = threading.Event()
    shared_telemetry = {"shared": [1, 2, 3]}
    state_fields = {
        "_previous_summary": "old-summary",
        "_summary_has_user_turn": False,
        "compression_count": 4,
        "_last_compression_savings_pct": 37.5,
        "_ineffective_compression_count": 1,
        "_anti_thrash_recovery_deadline": 123.0,
        "_fallback_compression_streak": 1,
        "_verify_compaction_cleared_threshold": False,
        "_last_compression_made_progress": False,
        "_summary_failure_cooldown_until": 456.0,
        "_cooldown_persist_failed": True,
        "_last_summary_error": "old-error",
        "_consecutive_timeout_failures": 2,
        "_last_summary_dropped_count": 3,
        "_last_summary_fallback_used": True,
        "_last_compress_aborted": False,
        "_last_summary_auth_failure": True,
        "_last_summary_network_failure": True,
        "_last_aux_model_failure_error": "old-aux-error",
        "_last_aux_model_failure_model": "old-aux-model",
        "_summary_model_fallen_back": True,
        "summary_model": "old-summary-model",
        "_last_compression_telemetry": shared_telemetry,
        "_active_compression_telemetry": shared_telemetry,
        "_compression_telemetry_seed": {"seed": [3]},
    }
    for name, value in state_fields.items():
        setattr(agent.context_compressor, name, copy.deepcopy(value))
    restored_shared_telemetry = copy.deepcopy(shared_telemetry)
    agent.context_compressor._last_compression_telemetry = restored_shared_telemetry
    agent.context_compressor._active_compression_telemetry = restored_shared_telemetry

    def _provider_succeeded_then_waits(*_args, **_kwargs):
        for name in state_fields:
            setattr(agent.context_compressor, name, f"mutated-{name}")
        provider_returned.set()
        assert allow_compress_return.wait(timeout=5)
        return [
            {"role": "user", "content": "[CONTEXT COMPACTION] cancelled summary"},
            {"role": "user", "content": "tail"},
        ]

    agent.context_compressor.compress.side_effect = _provider_succeeded_then_waits
    result: dict[str, tuple] = {}
    worker = threading.Thread(
        target=lambda: result.setdefault(
            "value", agent._compress_context(messages, "sys", approx_tokens=120_000)
        ),
        daemon=True,
    )
    worker.start()
    assert provider_returned.wait(timeout=2)
    agent.hard_interrupt("cancel after provider return")
    allow_compress_return.set()
    worker.join(timeout=5)

    assert not worker.is_alive()
    assert result["value"][0] is messages
    assert {
        name: copy.deepcopy(getattr(agent.context_compressor, name))
        for name in state_fields
    } == state_fields
    assert (
        agent.context_compressor._active_compression_telemetry
        is agent.context_compressor._last_compression_telemetry
    )
    assert db.get_compression_lock_holder(session_id) is None

    agent.clear_interrupt()
    agent.context_compressor.compress.side_effect = lambda *_a, **_kw: [
        {"role": "user", "content": "[CONTEXT COMPACTION] retry summary"},
        {"role": "user", "content": "tail"},
    ]
    retried, _prompt = agent._compress_context(
        messages, "sys", approx_tokens=120_000
    )
    assert retried is not messages
    assert retried[0]["content"] == "[CONTEXT COMPACTION] retry summary"


def test_force_cancel_restores_newer_durable_cooldown_captured_under_lease(
    tmp_path: Path,
) -> None:
    """A stale forced attempt rolls back to the lease-protected durable row."""
    from agent.auxiliary_client import AuxiliaryExplicitCancellation
    from agent.context_compressor import ContextCompressor

    db = SessionDB(db_path=tmp_path / "state.db")
    session_id = "FORCE_CANCEL_DURABLE_COOLDOWN"
    db.create_session(session_id, source="cli")

    # B binds first and therefore has no local cooldown. A then persists a
    # newer cooldown for the same durable session before B acquires its lease.
    stale_agent = _build_agent_with_db(
        db, session_id, stub_compressor=False
    )
    writer_agent = _build_agent_with_db(
        db, session_id, stub_compressor=False
    )
    stale = stale_agent.context_compressor
    writer = writer_agent.context_compressor
    assert isinstance(stale, ContextCompressor)
    assert isinstance(writer, ContextCompressor)
    assert stale._summary_failure_cooldown_until == 0.0

    writer._record_compression_failure_cooldown(120.0, "newer durable failure")
    durable_before = tuple(
        db._conn.execute(
            "SELECT compression_failure_cooldown_until, compression_failure_error "
            "FROM sessions WHERE id = ?",
            (session_id,),
        ).fetchone()
    )
    assert durable_before[0] is not None

    stale_seed = {"seed": ["truly-pre-attempt"]}
    stale._compression_telemetry_seed = copy.deepcopy(stale_seed)
    stale._previous_summary = "pre-attempt-summary"
    stale_agent._compression_feasibility_checked = True
    stale_agent.compression_in_place = True
    messages = [{"role": "user", "content": f"m{i}"} for i in range(20)]

    real_clear = ContextCompressor._clear_compression_failure_cooldown

    def _clear_then_hard_cancel() -> None:
        real_clear(stale)
        stale_agent._hard_interrupt_requested.set()
        raise AuxiliaryExplicitCancellation()

    # Exercise the built-in force=True mutation point deterministically: force
    # clears the durable cooldown, then the frozen host cancellation unwinds it.
    stale._clear_compression_failure_cooldown = _clear_then_hard_cancel

    compressed, _prompt = stale_agent._compress_context(
        messages,
        "sys",
        approx_tokens=120_000,
        force=True,
    )

    assert compressed is messages
    durable_after = tuple(
        db._conn.execute(
            "SELECT compression_failure_cooldown_until, compression_failure_error "
            "FROM sessions WHERE id = ?",
            (session_id,),
        ).fetchone()
    )
    assert durable_after == durable_before
    assert stale._summary_failure_cooldown_until > time.monotonic()
    assert stale._last_summary_error == "newer durable failure"
    assert stale._cooldown_persist_failed is False
    assert stale._compression_telemetry_seed == stale_seed
    assert stale._previous_summary == "pre-attempt-summary"
    assert db.get_compression_lock_holder(session_id) is None

    # A future compressor refresh must still observe the exact row rather than
    # the cancelled force attempt having permanently cleared it.
    future_agent = _build_agent_with_db(
        db, session_id, stub_compressor=False
    )
    future = future_agent.context_compressor.get_active_compression_failure_cooldown(
        refresh=True
    )
    assert future is not None
    assert future["error"] == "newer durable failure"


def test_unrelated_interrupted_error_propagates_and_releases_compression_lease(
    tmp_path: Path,
) -> None:
    """A plugin/OS InterruptedError is a failure, not an explicit transaction abort."""
    db = SessionDB(db_path=tmp_path / "state.db")
    session_id = "UNRELATED_INTERRUPT_COMPRESSION_TEST"
    db.create_session(session_id, source="cli")

    agent = _build_agent_with_db(db, session_id)
    messages = [{"role": "user", "content": f"m{i}"} for i in range(20)]

    def _provider_interrupted(*_args, **_kwargs):
        messages[0]["content"] = "must be rolled back"
        raise InterruptedError("provider syscall interrupted")

    agent.context_compressor.compress.side_effect = _provider_interrupted

    with pytest.raises(InterruptedError, match="provider syscall interrupted"):
        agent._compress_context(messages, "sys", approx_tokens=120_000)

    assert db.get_compression_lock_holder(session_id) is None
    db.append_message(session_id, "assistant", "writes recovered")


def test_redirect_interrupt_remains_protected_during_compression(tmp_path: Path) -> None:
    """Redirects use interrupt_requested=True/message=None; only the atomic
    hard-cancel event may override summary protection."""
    from agent import auxiliary_client as aux

    db = SessionDB(db_path=tmp_path / "state.db")
    session_id = "REDIRECT_COMPRESSION_TEST"
    db.create_session(session_id, source="cli")
    agent = _build_agent_with_db(db, session_id)
    agent._interrupt_requested = True
    agent._interrupt_message = None
    agent._pending_redirect = "new correction"
    messages = [{"role": "user", "content": f"m{i}"} for i in range(20)]

    def _protected_noop(current, **_kwargs):
        assert aux._aux_interrupt_cancel_requested() is False
        return copy.deepcopy(current)

    agent.context_compressor.compress.side_effect = _protected_noop

    compressed, _prompt = agent._compress_context(
        messages, "sys", approx_tokens=120_000
    )

    assert compressed == messages
    assert db.get_compression_lock_holder(session_id) is None


def test_hard_cancel_between_compress_return_and_commit_begin_wins_atomically(
    tmp_path: Path,
) -> None:
    """The hard-stop admission and commit admission share one fence lock."""
    db = SessionDB(db_path=tmp_path / "state.db")
    session_id = "HARD_CANCEL_COMMIT_RACE"
    db.create_session(session_id, source="tui")
    agent = _build_agent_with_db(db, session_id)
    agent.compression_in_place = True
    agent._cached_system_prompt = "sys"
    messages = [{"role": "user", "content": f"m{i}"} for i in range(20)]
    before_commit = threading.Event()
    allow_commit_check = threading.Event()

    class _CommitBarrierList(list):
        def __eq__(self, other):
            before_commit.set()
            assert allow_commit_check.wait(timeout=5)
            return super().__eq__(other)

    agent.context_compressor.compress.side_effect = lambda *_a, **_kw: _CommitBarrierList(
        [
            {"role": "user", "content": "[CONTEXT COMPACTION] summary"},
            {"role": "user", "content": "tail"},
        ]
    )
    archive_spy = MagicMock(wraps=db.archive_and_compact)
    db.archive_and_compact = archive_spy
    result: dict[str, tuple] = {}
    worker = threading.Thread(
        target=lambda: result.setdefault(
            "value", agent._compress_context(messages, "sys", approx_tokens=120_000)
        ),
        daemon=True,
    )
    worker.start()
    assert before_commit.wait(timeout=2)

    agent.hard_interrupt("cancel before commit admission")
    allow_commit_check.set()
    worker.join(timeout=5)

    assert not worker.is_alive()
    assert result["value"][0] is messages
    archive_spy.assert_not_called()
    assert db.get_compression_lock_holder(session_id) is None


def test_hard_stop_waits_for_commit_already_admitted(tmp_path: Path) -> None:
    """A surfaced stop never races an untracked post-return transcript commit."""
    db = SessionDB(db_path=tmp_path / "state.db")
    session_id = "HARD_CANCEL_AFTER_COMMIT_ADMISSION"
    db.create_session(session_id, source="tui")
    agent = _build_agent_with_db(db, session_id)
    agent.compression_in_place = True
    agent._cached_system_prompt = "sys"
    messages = [{"role": "user", "content": f"m{i}"} for i in range(20)]
    commit_started = threading.Event()
    allow_commit = threading.Event()
    stop_returned = threading.Event()
    original_archive = db.archive_and_compact

    def _blocked_archive(*args, **kwargs):
        commit_started.set()
        assert allow_commit.wait(timeout=5)
        return original_archive(*args, **kwargs)

    db.archive_and_compact = _blocked_archive
    agent.context_compressor.compress.side_effect = lambda *_a, **_kw: [
        {"role": "user", "content": "[CONTEXT COMPACTION] summary"},
        {"role": "user", "content": "tail"},
    ]
    compression_result: dict[str, tuple] = {}
    compression = threading.Thread(
        target=lambda: compression_result.setdefault(
            "value", agent._compress_context(messages, "sys", approx_tokens=120_000)
        ),
        daemon=True,
    )
    compression.start()
    assert commit_started.wait(timeout=2)

    stop = threading.Thread(
        target=lambda: (
            agent.hard_interrupt("stop after commit admission"),
            stop_returned.set(),
        ),
        daemon=True,
    )
    stop.start()
    assert not stop_returned.wait(timeout=0.1)
    allow_commit.set()
    compression.join(timeout=5)
    stop.join(timeout=5)

    assert not compression.is_alive()
    assert not stop.is_alive()
    assert stop_returned.is_set()
    assert compression_result["value"][0][0]["content"] == (
        "[CONTEXT COMPACTION] summary"
    )
    assert agent._hard_interrupt_requested.is_set()
    assert db.get_compression_lock_holder(session_id) is None


@pytest.mark.parametrize("deadline_offset", [-10.0, 0.05, None])
def test_force_cancel_restores_exact_expired_or_expiring_cooldown_row(
    tmp_path: Path,
    deadline_offset: float | None,
) -> None:
    """Cancellation preserves raw cooldown columns even after their deadline."""
    from agent.auxiliary_client import AuxiliaryExplicitCancellation
    from agent.context_compressor import ContextCompressor

    db = SessionDB(db_path=tmp_path / "state.db")
    session_id = f"RAW_COOLDOWN_{deadline_offset}"
    db.create_session(session_id, source="cli")
    deadline = time.time() + deadline_offset if deadline_offset is not None else None
    db.restore_compression_failure_cooldown_row(
        session_id,
        {
            "session_exists": True,
            "cooldown_until": deadline,
            "error": "expired-but-exact",
        },
    )
    before = db.get_compression_failure_cooldown_row(session_id)

    agent = _build_agent_with_db(db, session_id, stub_compressor=False)
    compressor = agent.context_compressor
    assert isinstance(compressor, ContextCompressor)
    # A stale local persistence-failure marker must not suppress restoration
    # once the raw durable row was captured authoritatively under the lease.
    compressor._cooldown_persist_failed = True
    agent._compression_feasibility_checked = True
    agent.compression_in_place = True
    messages = [{"role": "user", "content": f"m{i}"} for i in range(20)]
    real_clear = ContextCompressor._clear_compression_failure_cooldown

    def _mutate_then_cancel() -> None:
        real_clear(compressor)
        if deadline_offset is not None and deadline_offset > 0:
            assert deadline is not None
            while time.time() <= deadline:
                time.sleep(0.005)
        agent._hard_interrupt_requested.set()
        raise AuxiliaryExplicitCancellation()

    compressor._clear_compression_failure_cooldown = _mutate_then_cancel

    compressed, _prompt = agent._compress_context(
        messages,
        "sys",
        approx_tokens=120_000,
        force=True,
    )

    assert compressed is messages
    assert db.get_compression_failure_cooldown_row(session_id) == before
    assert db.get_compression_lock_holder(session_id) is None


def test_cooldown_rollback_failure_surfaces_and_releases_lease(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failed compensating write cannot masquerade as a mutation-free cancel."""
    from agent.auxiliary_client import AuxiliaryExplicitCancellation
    from agent.context_compressor import ContextCompressor

    db = SessionDB(db_path=tmp_path / "state.db")
    session_id = "COOLDOWN_ROLLBACK_WRITE_FAILURE"
    db.create_session(session_id, source="cli")
    db.record_compression_failure_cooldown(
        session_id,
        time.time() + 120.0,
        "must-restore",
    )
    agent = _build_agent_with_db(db, session_id, stub_compressor=False)
    compressor = agent.context_compressor
    assert isinstance(compressor, ContextCompressor)
    agent._compression_feasibility_checked = True
    agent.compression_in_place = True
    messages = [{"role": "user", "content": f"m{i}"} for i in range(20)]
    real_clear = ContextCompressor._clear_compression_failure_cooldown

    def _mutate_then_cancel() -> None:
        real_clear(compressor)
        agent._hard_interrupt_requested.set()
        raise AuxiliaryExplicitCancellation()

    compressor._clear_compression_failure_cooldown = _mutate_then_cancel

    def _rollback_write_fails(_self, _session_id, _snapshot) -> None:
        raise sqlite3.OperationalError("forced rollback write failure")

    monkeypatch.setattr(
        SessionDB,
        "restore_compression_failure_cooldown_row",
        _rollback_write_fails,
    )

    with pytest.raises(sqlite3.OperationalError, match="forced rollback write failure"):
        agent._compress_context(
            messages,
            "sys",
            approx_tokens=120_000,
            force=True,
        )

    assert db.get_compression_lock_holder(session_id) is None


def test_exact_cooldown_restore_api_propagates_sqlite_write_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db = SessionDB(db_path=tmp_path / "state.db")
    session_id = "RAW_COOLDOWN_WRITE_FAILURE"
    db.create_session(session_id, source="test")

    def _write_fails(_callback) -> None:
        raise sqlite3.OperationalError("forced low-level write failure")

    monkeypatch.setattr(db, "_execute_write", _write_fails)

    with pytest.raises(sqlite3.OperationalError, match="forced low-level write failure"):
        db.restore_compression_failure_cooldown_row(
            session_id,
            {
                "session_exists": True,
                "cooldown_until": time.time() + 10.0,
                "error": "must propagate",
            },
        )
# ── AE-204: the losing path must converge on the winner's compacted state ───
#
# Before AE-204 the loser returned its (uncompacted) message list verbatim.
# The session was not forked, but two divergent views of one conversation
# stayed alive: the winner's compacted transcript in the session store and the
# loser's full pre-compaction copy in memory. Downstream transcript merges
# (Responses-store history rebuild, gateway history_offset re-baselining, the
# identity-diffed session flush) then concatenated the conversation with
# itself — production chat ca5c63dd: 158 msgs → 39 compacted, loser returns
# 158 unchanged, next compaction sees 326 msgs / 1.94x tokens.

# A REAL compaction summary. Adoption demands positive proof that the holder
# actually compacted, and the proof is the compressor's own summary prefix —
# so the fixture has to carry the genuine article, not a look-alike.
COMPACTION_SUMMARY = f"{SUMMARY_PREFIX}\n## State Ledger"


def _compacted_payload() -> list:
    """Fresh copies — compress_context mutates/annotates what it returns."""
    return [
        {"role": "user", "content": COMPACTION_SUMMARY},
        {"role": "assistant", "content": "carrying on from the summary"},
    ]


def _build_in_place_agent(db: SessionDB, session_id: str):
    agent = _build_agent_with_db(db, session_id)
    agent.compression_in_place = True
    return agent


def _signal_on_contention(db: SessionDB, session_id: str) -> threading.Event:
    """Return an Event set the moment an acquire of ``session_id``'s lock fails.

    Wall-clock timers race agent construction (plugin discovery alone can take
    seconds), so any test that needs "the loser has provably lost" must key off
    the *failed acquire* itself rather than a sleep.
    """
    contended = threading.Event()
    original = db.try_acquire_compression_lock

    def _instrumented(sid: str, new_holder: str, ttl_seconds: float = 300.0) -> bool:
        acquired = original(sid, new_holder, ttl_seconds=ttl_seconds)
        if not acquired and sid == session_id:
            contended.set()
        return acquired

    setattr(db, "try_acquire_compression_lock", _instrumented)
    return contended


def _release_lock_when_contended(
    db: SessionDB, session_id: str, holder: str
) -> threading.Thread:
    """Release ``holder``'s lock the moment a compression path is blocked by it.

    Same determinism argument as :func:`_signal_on_contention`: releasing on a
    timer would let the losing path acquire the lock outright and compress for
    real.
    """
    contended = _signal_on_contention(db, session_id)

    def _release() -> None:
        if contended.wait(timeout=20):
            db.release_compression_lock(session_id, holder)

    thread = threading.Thread(target=_release, name="lock-releaser", daemon=True)
    thread.start()
    return thread


def _seed_history(db: SessionDB, session_id: str, count: int) -> list:
    history = [
        {
            "role": "user" if index % 2 == 0 else "assistant",
            "content": f"turn {index} " + ("x" * 200),
        }
        for index in range(count)
    ]
    db.replace_messages(session_id, history)
    return history


def test_race_loser_adopts_winner_compacted_state(tmp_path: Path, caplog) -> None:
    """Two real threads race one session: the loser must adopt, not double.

    The winner compacts in place; the loser (mid-turn, holding the same
    pre-compaction history plus its own live user turn) must come back with
    the winner's compacted transcript plus its own turn — never the
    pre-compression transcript, which is what downstream merges concatenate.
    """
    db = SessionDB(db_path=tmp_path / "state.db")
    sid = "ADOPT_RACE"
    db.create_session(sid, source="api")
    history = _seed_history(db, sid, 20)

    winner = _build_in_place_agent(db, sid)
    loser = _build_in_place_agent(db, sid)
    loser._compression_lock_wait_seconds = 15.0

    winner_holds_lock = threading.Event()
    # Set when a try_acquire for this session RETURNS FALSE, i.e. the loser has
    # provably lost the race. Gating the winner on that (rather than on a sleep
    # past "the loser thread started") is what makes this deterministic: a slow
    # box could otherwise run the loser's acquire after the winner released,
    # letting it win the lock and compress for real.
    loser_lost_the_race = _signal_on_contention(db, sid)

    def _winner_compress(*_a, **_kw):
        winner_holds_lock.set()
        # Hold the lock until the loser's acquire has actually failed.
        assert loser_lost_the_race.wait(timeout=30)
        return _compacted_payload()

    winner.context_compressor.compress.side_effect = _winner_compress

    winner_messages = [dict(msg) for msg in history]
    # The loser is mid-turn: history + this turn's live user message.
    loser_messages = [dict(msg) for msg in history]
    loser_messages.append({"role": "user", "content": "what about the invoice?"})
    loser._persist_user_message_idx = len(history)
    pre_race_len = len(loser_messages)

    winner_result: list = []
    loser_result: list = []

    def _run_winner():
        compressed, _sp = winner._compress_context(
            winner_messages, "sys", approx_tokens=120_000
        )
        winner_result.append(compressed)

    def _run_loser():
        assert winner_holds_lock.wait(timeout=30)
        compressed, _sp = loser._compress_context(
            loser_messages, "sys", approx_tokens=120_000
        )
        loser_result.append(compressed)

    threads = [
        threading.Thread(target=_run_winner, name="winner"),
        threading.Thread(target=_run_loser, name="loser"),
    ]
    with caplog.at_level(logging.INFO, logger="agent.conversation_compression"):
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=30)
    assert not any(thread.is_alive() for thread in threads)

    assert winner_result and loser_result
    won = winner_result[0]
    lost = loser_result[0]

    # 1. No doubling: the loser's transcript is bounded by the winner's
    #    compacted shape plus its own live turn — not 2x the pre-race list.
    assert len(lost) <= len(won) + 1
    assert len(lost) < pre_race_len
    from agent.model_metadata import estimate_messages_tokens_rough

    assert estimate_messages_tokens_rough(lost) < estimate_messages_tokens_rough(
        loser_messages
    )

    # 2. The loser's result DERIVES from the winner's compacted state (it
    #    carries the compaction summary) rather than the pre-race transcript.
    assert any(
        COMPACTION_SUMMARY in str(msg.get("content", "")) for msg in lost
    )
    assert not any(str(msg.get("content", "")).startswith("turn 0 ") for msg in lost)

    # 3. The loser's own live turn survived the adoption exactly once.
    assert [msg for msg in lost if msg.get("content") == "what about the invoice?"] == [
        {"role": "user", "content": "what about the invoice?"}
    ]

    # 4. It adopted rather than re-compressed, and never forked the session.
    loser.context_compressor.compress.assert_not_called()
    assert loser.session_id == sid
    assert _count_children(db, sid) == 0
    assert db.get_session(sid)["end_reason"] is None
    assert db.get_compression_lock_holder(sid) is None
    # The adopted transcript is the session's live transcript, so downstream
    # (gateway history_offset, flush baseline) must re-baseline like in-place.
    assert loser._last_compaction_in_place is True
    # 5. Token accounting is parked like a completed compaction, so the stale
    #    pre-compaction prompt count cannot re-trigger compression against the
    #    transcript we just adopted.
    assert loser.context_compressor.last_prompt_tokens == -1
    assert loser.context_compressor.awaiting_real_usage_after_compression is True
    adopted_telemetry = [
        json.loads(record.getMessage().split(": ", 1)[1])
        for record in caplog.records
        if "context compression attempt telemetry:" in record.getMessage()
        and '"split_status":"in_place_adopted"' in record.getMessage()
    ]
    assert len(adopted_telemetry) == 1
    assert adopted_telemetry[0]["commit_status"] == "committed"
    assert adopted_telemetry[0].get("failure_class") is None


def test_race_loser_adopts_rotated_child_and_preserves_live_tail(
    tmp_path: Path,
    caplog,
) -> None:
    """A lock loser follows a rotating winner without dropping its live turn.

    Upstream's canonical child-tip recovery owns the session/context-engine
    rebind. AE-204 convergence must layer onto that recovery rather than
    treating rotation as an unadoptable in-place result or resetting the
    child flush cursor.
    """
    db = SessionDB(db_path=tmp_path / "state.db")
    parent_sid = "ADOPT_ROTATED_RACE"
    child_sid = "ADOPT_ROTATED_CHILD"
    db.create_session(parent_sid, source="api")
    history = _seed_history(db, parent_sid, 20)
    compacted = _compacted_payload()
    holder = "rotating_winner"
    assert db.try_acquire_compression_lock(parent_sid, holder) is True
    loser_contended = _signal_on_contention(db, parent_sid)

    def _publish_rotated_child() -> None:
        assert loser_contended.wait(timeout=20)
        db.end_session(parent_sid, "compression")
        db.create_session(
            child_sid,
            source="api",
            parent_session_id=parent_sid,
        )
        db.replace_messages(child_sid, compacted)
        db.release_compression_lock(parent_sid, holder)

    publisher = threading.Thread(
        target=_publish_rotated_child,
        name="rotating-winner",
        daemon=True,
    )
    publisher.start()

    loser = _build_agent_with_db(db, parent_sid)
    loser._compression_lock_wait_seconds = 10.0
    messages = [dict(message) for message in history]
    messages.append({"role": "user", "content": "preserve this live turn"})
    loser._persist_user_message_idx = len(history)

    with caplog.at_level(logging.INFO, logger="agent.conversation_compression"):
        adopted, _system_prompt = loser._compress_context(
            messages,
            "sys",
            approx_tokens=120_000,
        )
    publisher.join(timeout=5)
    assert not publisher.is_alive()

    assert loser.session_id == child_sid
    assert [message["content"] for message in adopted] == [
        COMPACTION_SUMMARY,
        "carrying on from the summary",
        "preserve this live turn",
    ]
    assert loser._persist_user_message_idx == len(compacted)
    assert loser._last_compression_attempt_in_place is False
    assert loser._last_compaction_in_place is False
    assert loser._compression_skipped_due_to_lock == holder
    assert loser._flushed_db_message_session_id == child_sid
    assert loser._last_flushed_db_idx == len(compacted)
    loser.context_compressor.compress.assert_not_called()
    loser.context_compressor.on_session_start.assert_called_once()

    # The recovered child rows are seeded as already durable; only the loser's
    # live tail is appended when the turn next flushes.
    loser._flush_messages_to_session_db(adopted, None)
    assert [row["content"] for row in db.get_messages(child_sid)] == [
        COMPACTION_SUMMARY,
        "carrying on from the summary",
        "preserve this live turn",
    ]
    assert _count_children(db, parent_sid) == 1
    assert db.get_compression_lock_holder(parent_sid) is None

    telemetry_records = [
        json.loads(record.getMessage().split(": ", 1)[1])
        for record in caplog.records
        if "context compression attempt telemetry:" in record.getMessage()
    ]
    assert len(telemetry_records) == 1
    assert telemetry_records[0]["commit_status"] == "committed"
    assert telemetry_records[0]["split_status"] == "rotated_adopted"
    assert telemetry_records[0].get("failure_class") is None


def test_lock_wait_cancellation_prevents_late_adoption(
    tmp_path: Path,
    caplog,
) -> None:
    """A host timeout revokes convergence before it can mutate live state."""
    from agent.conversation_compression import CompressionCommitFence

    db = SessionDB(db_path=tmp_path / "state.db")
    session_id = "ADOPT_WAIT_CANCELLED"
    db.create_session(session_id, source="api")
    history = _seed_history(db, session_id, 12)
    holder = "blocked_winner"
    assert db.try_acquire_compression_lock(session_id, holder) is True
    contended = _signal_on_contention(db, session_id)

    agent = _build_in_place_agent(db, session_id)
    agent._compression_lock_wait_seconds = 10.0
    agent.status_callback = MagicMock()
    fence = CompressionCommitFence()
    result: list[tuple[list, str]] = []

    def _run_loser() -> None:
        result.append(
            agent._compress_context(
                history,
                "sys",
                approx_tokens=120_000,
                commit_fence=fence,
            )
        )

    with caplog.at_level(logging.INFO, logger="agent.conversation_compression"):
        loser = threading.Thread(target=_run_loser, name="cancelled-lock-loser")
        loser.start()
        assert contended.wait(timeout=5)
        assert fence.cancel_before_commit() is True
        loser.join(timeout=3)

    assert not loser.is_alive()
    assert result and result[0][0] is history
    assert agent.session_id == session_id
    assert agent._compression_skipped_due_to_lock == holder
    agent.context_compressor.compress.assert_not_called()
    assert db.get_compression_lock_holder(session_id) == holder
    assert any(
        '"failure_class":"lock_contended"' in record.getMessage()
        for record in caplog.records
    )
    agent.status_callback.assert_any_call(
        "compacted",
        "✓ Context compaction complete — continuing turn...",
    )
    db.release_compression_lock(session_id, holder)


def test_adopted_state_is_not_reappended_by_the_next_flush(tmp_path: Path) -> None:
    """Adopted rows are already durable — flushing must not duplicate them.

    The winner releases synchronously after the loser's failed acquire, before
    ``compress_context`` can perform its post-failure holder lookup. This pins
    the real race where the pre-acquire observation is the only confirmed
    holder evidence; treating the later ``None`` as an unconfirmed backend
    failure would skip safe adoption and return the stale full transcript.
    """
    db = SessionDB(db_path=tmp_path / "state.db")
    sid = "ADOPT_FLUSH"
    db.create_session(sid, source="api")
    history = _seed_history(db, sid, 12)

    compacted = _compacted_payload()
    db.archive_and_compact(sid, compacted)
    held = db.try_acquire_compression_lock(sid, "external_holder")
    assert held is True

    agent = _build_in_place_agent(db, sid)
    agent._compression_lock_wait_seconds = 10.0
    messages = [dict(msg) for msg in history]
    messages.append({"role": "user", "content": "live turn"})
    agent._persist_user_message_idx = len(history)

    real_acquire = db.try_acquire_compression_lock
    released_before_failed_acquire_returned = threading.Event()

    def _acquire_after_winner_finishes(
        session_id: str,
        holder: str,
        ttl_seconds: float = 300.0,
    ) -> bool:
        acquired = real_acquire(session_id, holder, ttl_seconds=ttl_seconds)
        if not acquired and session_id == sid:
            db.release_compression_lock(sid, "external_holder")
            assert db.get_compression_lock_holder(sid) is None
            released_before_failed_acquire_returned.set()
        return acquired

    db.try_acquire_compression_lock = _acquire_after_winner_finishes
    adopted, _sp = agent._compress_context(messages, "sys", approx_tokens=120_000)

    assert released_before_failed_acquire_returned.is_set()
    assert len(adopted) == len(compacted) + 1
    active_before = len(db.get_messages(sid))
    agent._flush_messages_to_session_db(adopted, None)
    active_after = len(db.get_messages(sid))
    # Only the un-persisted live turn was written; the adopted compacted rows
    # were recognised as durable (this is the in-place doubling hazard that
    # conversation_history_after_compression documents).
    assert active_before == len(compacted)
    assert active_after == len(compacted) + 1
    summaries = [
        row for row in db.get_messages(sid)
        if COMPACTION_SUMMARY in str(row.get("content", ""))
    ]
    assert len(summaries) == 1


def test_adoption_does_not_duplicate_an_already_flushed_tail(tmp_path: Path) -> None:
    """A turn that flushed part of its tail before losing must not re-add it.

    Multi-iteration turns flush as they go. When the winner compacts after
    such a flush, those rows survive in the compacted live set, so appending
    the in-memory tail wholesale would duplicate exactly that overlap.
    """
    db = SessionDB(db_path=tmp_path / "state.db")
    sid = "ADOPT_OVERLAP"
    db.create_session(sid, source="api")
    history = _seed_history(db, sid, 12)

    compacted = _compacted_payload()
    db.archive_and_compact(sid, compacted)
    # The loser's first tail message was already durable when the winner
    # compacted, so it survives as an active row after the compaction.
    db.append_message(sid, "user", "live turn")
    held = db.try_acquire_compression_lock(sid, "external_holder")
    assert held is True

    agent = _build_in_place_agent(db, sid)
    agent._compression_lock_wait_seconds = 10.0
    messages = [dict(msg) for msg in history]
    messages.append({"role": "user", "content": "live turn"})
    messages.append({"role": "assistant", "content": "partial answer"})
    agent._persist_user_message_idx = len(history)

    releaser = _release_lock_when_contended(db, sid, "external_holder")
    adopted, _sp = agent._compress_context(messages, "sys", approx_tokens=120_000)
    releaser.join(timeout=5)

    live_turns = [msg for msg in adopted if msg.get("content") == "live turn"]
    assert len(live_turns) == 1, "already-flushed tail message was duplicated"
    assert adopted[-1] == {"role": "assistant", "content": "partial answer"}
    assert len(adopted) == len(compacted) + 2


def test_holder_timeout_falls_back_with_its_own_log_tag(
    tmp_path: Path, caplog
) -> None:
    """A crashed/hung holder must not deadlock the loser — bounded fallback."""
    db = SessionDB(db_path=tmp_path / "state.db")
    sid = "ADOPT_TIMEOUT"
    db.create_session(sid, source="api")
    history = _seed_history(db, sid, 12)

    # A holder that never releases and never compacts (crashed mid-run).
    assert db.try_acquire_compression_lock(sid, "crashed_holder") is True

    agent = _build_in_place_agent(db, sid)
    agent._compression_lock_wait_seconds = 0.4
    messages = [dict(msg) for msg in history]
    messages.append({"role": "user", "content": "live turn"})
    agent._persist_user_message_idx = len(history)

    started = time.monotonic()
    with caplog.at_level(logging.WARNING, logger="agent.conversation_compression"):
        returned, _sp = agent._compress_context(
            messages, "sys", approx_tokens=120_000
        )
    elapsed = time.monotonic() - started

    assert returned == messages
    assert 0.3 <= elapsed < 10.0, "wait was not bounded by the configured budget"
    assert not getattr(agent, "_last_compaction_in_place", False)
    assert agent.session_id == sid
    assert _count_children(db, sid) == 0
    tags = [record.getMessage() for record in caplog.records]
    assert any("compression skipped: holder timeout" in tag for tag in tags)
    assert not any("no compacted state to adopt" in tag for tag in tags)


def test_holder_release_without_compaction_keeps_the_skip_tag(
    tmp_path: Path, caplog
) -> None:
    """Holder finished without compacting: skip, but under the distinct tag."""
    db = SessionDB(db_path=tmp_path / "state.db")
    sid = "ADOPT_NOTHING"
    db.create_session(sid, source="api")
    history = _seed_history(db, sid, 12)

    assert db.try_acquire_compression_lock(sid, "aborting_holder") is True

    agent = _build_in_place_agent(db, sid)
    agent._compression_lock_wait_seconds = 10.0
    messages = [dict(msg) for msg in history]

    releaser = _release_lock_when_contended(db, sid, "aborting_holder")
    with caplog.at_level(logging.WARNING, logger="agent.conversation_compression"):
        returned, _sp = agent._compress_context(
            messages, "sys", approx_tokens=120_000
        )
    releaser.join(timeout=5)

    assert returned is messages
    assert not getattr(agent, "_last_compaction_in_place", False)
    tags = [record.getMessage() for record in caplog.records]
    assert any("another path is compressing" in tag for tag in tags)
    assert any("no compacted state to adopt" in tag for tag in tags)
    assert not any("holder timeout" in tag for tag in tags)


def test_ended_parent_without_live_child_is_not_adopted(tmp_path: Path) -> None:
    """An ended parent without a canonical live continuation fails closed."""
    from agent.conversation_compression import _adopt_concurrent_compaction

    db = SessionDB(db_path=tmp_path / "state.db")
    sid = "ADOPT_ROTATED"
    db.create_session(sid, source="api")
    history = _seed_history(db, sid, 12)
    db.archive_and_compact(sid, _compacted_payload())
    db.end_session(sid, "compression")

    agent = _build_in_place_agent(db, sid)
    messages = [dict(msg) for msg in history]
    agent._persist_user_message_idx = len(history)

    assert _adopt_concurrent_compaction(agent, messages) is None


def test_adoption_declines_when_the_turn_anchor_is_untrustworthy(
    tmp_path: Path,
) -> None:
    """Never split the list at an anchor that no longer names a user turn."""
    from agent.conversation_compression import _adopt_concurrent_compaction

    db = SessionDB(db_path=tmp_path / "state.db")
    sid = "ADOPT_BAD_ANCHOR"
    db.create_session(sid, source="api")
    history = _seed_history(db, sid, 12)
    db.archive_and_compact(sid, _compacted_payload())

    agent = _build_in_place_agent(db, sid)
    messages = [dict(msg) for msg in history]
    messages.append({"role": "user", "content": "live turn"})

    # Anchor points at an assistant row (stale after an earlier rewrite).
    agent._persist_user_message_idx = 1
    assert _adopt_concurrent_compaction(agent, messages) is None
    # Out-of-range anchors are refused too.
    agent._persist_user_message_idx = len(messages) + 5
    assert _adopt_concurrent_compaction(agent, messages) is None
    # A sound anchor still adopts.
    agent._persist_user_message_idx = len(history)
    adopted = _adopt_concurrent_compaction(agent, messages)
    assert adopted is not None and len(adopted) == 3


def test_uncompacted_session_is_never_adopted(tmp_path: Path) -> None:
    """No compaction happened → nothing to adopt (holder still mid-summary)."""
    from agent.conversation_compression import _adopt_concurrent_compaction

    db = SessionDB(db_path=tmp_path / "state.db")
    sid = "ADOPT_NO_COMPACTION"
    db.create_session(sid, source="api")
    history = _seed_history(db, sid, 12)

    agent = _build_in_place_agent(db, sid)
    messages = [dict(msg) for msg in history]
    agent._persist_user_message_idx = len(history)

    assert _adopt_concurrent_compaction(agent, messages) is None


def test_shorter_live_set_without_a_summary_is_not_a_compaction(
    tmp_path: Path,
) -> None:
    """Size alone is not proof: a shorter active set can mean no compaction.

    The durable active set drops below this path's in-memory history for
    reasons that have nothing to do with compaction — a best-effort flush
    failure is swallowed, ``rewind_to_message`` soft-archives rows out of the
    active set, alternation repair merges rows. Without positive proof that a
    summary was written, adoption would launder an UNCOMPACTED transcript as a
    compaction and hand downstream exactly the list that gets re-concatenated.
    """
    from agent.conversation_compression import _adopt_concurrent_compaction

    db = SessionDB(db_path=tmp_path / "state.db")
    sid = "ADOPT_NO_SUMMARY"
    db.create_session(sid, source="api")
    history = _seed_history(db, sid, 12)
    # Two rows never reached the store, so the active set is strictly shorter
    # than the history this path holds — with nothing ever compacted.
    db.replace_messages(sid, history[:10])

    agent = _build_in_place_agent(db, sid)
    messages = [dict(msg) for msg in history]
    messages.append({"role": "user", "content": "live turn"})
    agent._persist_user_message_idx = len(history)

    # Every size guard passes (10 < 12, adopted 11 < 13) — only the summary
    # proof stands between this and a bogus adoption.
    assert _adopt_concurrent_compaction(agent, messages) is None

    # The same shape WITH a real summary in the stored set is adoptable.
    db.replace_messages(sid, _compacted_payload())
    adopted = _adopt_concurrent_compaction(agent, messages)
    assert adopted is not None
    assert len(adopted) == len(_compacted_payload()) + 1


def test_aborted_holder_is_not_mistaken_for_a_winner(
    tmp_path: Path, caplog
) -> None:
    """Holder aborts having written nothing → skip, never a fake adoption.

    An aux-model failure mid-summary aborts the holder's compression and it
    releases the lock having written no summary. If the loser adopted on the
    size guards alone it would log ``compression adopted``, claim an in-place
    compaction, and park token accounting — all for a transcript that was
    never compacted.
    """
    db = SessionDB(db_path=tmp_path / "state.db")
    sid = "ADOPT_ABORTED_HOLDER"
    db.create_session(sid, source="api")
    history = _seed_history(db, sid, 12)
    # Pre-existing deficit (swallowed flush failure), no compaction anywhere.
    db.replace_messages(sid, history[:10])

    assert db.try_acquire_compression_lock(sid, "aborting_holder") is True

    agent = _build_in_place_agent(db, sid)
    agent._compression_lock_wait_seconds = 10.0
    agent.context_compressor.last_prompt_tokens = 90_000
    agent.context_compressor.awaiting_real_usage_after_compression = False
    messages = [dict(msg) for msg in history]
    messages.append({"role": "user", "content": "live turn"})
    agent._persist_user_message_idx = len(history)

    releaser = _release_lock_when_contended(db, sid, "aborting_holder")
    with caplog.at_level(logging.INFO, logger="agent.conversation_compression"):
        returned, _sp = agent._compress_context(
            messages, "sys", approx_tokens=120_000
        )
    releaser.join(timeout=5)

    assert returned is messages
    agent.context_compressor.compress.assert_not_called()
    assert not getattr(agent, "_last_compaction_in_place", False)
    # Token accounting stays live: nothing was compacted, so the next cycle
    # must keep grading the real (uncompacted) transcript.
    assert agent.context_compressor.last_prompt_tokens == 90_000
    assert agent.context_compressor.awaiting_real_usage_after_compression is False
    assert agent.session_id == sid
    assert _count_children(db, sid) == 0
    tags = [record.getMessage() for record in caplog.records]
    assert any("no compacted state to adopt" in tag for tag in tags)
    assert not any("compression adopted" in tag for tag in tags)


def test_compression_lock_wait_budget_precedence_and_clamp(monkeypatch) -> None:
    """Attribute → env → default, always clamped to the lease TTL."""
    from agent.conversation_compression import (
        COMPRESSION_LOCK_WAIT_ENV,
        COMPRESSION_LOCK_WAIT_SECONDS_DEFAULT,
        _compression_lock_wait_budget,
    )

    class _Agent:
        _compression_lock_wait_seconds = None

    agent = _Agent()
    monkeypatch.delenv(COMPRESSION_LOCK_WAIT_ENV, raising=False)
    assert _compression_lock_wait_budget(agent, 300.0) == (
        COMPRESSION_LOCK_WAIT_SECONDS_DEFAULT
    )
    # The lease is the hard ceiling: waiting past it can only stall behind a
    # holder whose row is already reclaimable.
    assert _compression_lock_wait_budget(agent, 5.0) == 5.0

    monkeypatch.setenv(COMPRESSION_LOCK_WAIT_ENV, "12.5")
    assert _compression_lock_wait_budget(agent, 300.0) == 12.5
    monkeypatch.setenv(COMPRESSION_LOCK_WAIT_ENV, "not-a-number")
    assert _compression_lock_wait_budget(agent, 300.0) == (
        COMPRESSION_LOCK_WAIT_SECONDS_DEFAULT
    )

    agent._compression_lock_wait_seconds = 3.0
    assert _compression_lock_wait_budget(agent, 300.0) == 3.0
    agent._compression_lock_wait_seconds = -1
    assert _compression_lock_wait_budget(agent, 300.0) == 0.0


def test_holder_wait_aborts_on_interrupt(tmp_path: Path) -> None:
    """An interrupted turn must stop waiting immediately."""
    from agent.conversation_compression import _wait_for_compression_holder

    db = SessionDB(db_path=tmp_path / "state.db")
    sid = "ADOPT_INTERRUPT"
    db.create_session(sid, source="api")
    assert db.try_acquire_compression_lock(sid, "stuck_holder") is True

    agent = _build_in_place_agent(db, sid)
    agent._interrupt_requested = True

    started = time.monotonic()
    released, waited = _wait_for_compression_holder(
        agent, db, sid, "stuck_holder", 30.0
    )
    assert released is False
    assert waited < 1.0
    assert time.monotonic() - started < 1.0


def test_manual_forced_compression_never_adopts(tmp_path: Path, caplog) -> None:
    """``/compress here N`` hands us a HEAD SLICE — adoption would corrupt it.

    Manual compression passes ``force=True`` and, in partial mode, only the
    head of the transcript (the caller re-appends the verbatim tail). The
    session-wide compacted transcript is not a valid substitute for that
    slice, so forced callers keep the historical immediate skip: no wait, no
    adoption, original log tag.
    """
    db = SessionDB(db_path=tmp_path / "state.db")
    sid = "ADOPT_FORCED"
    db.create_session(sid, source="api")
    history = _seed_history(db, sid, 20)
    db.archive_and_compact(sid, _compacted_payload())
    assert db.try_acquire_compression_lock(sid, "external_holder") is True

    agent = _build_in_place_agent(db, sid)
    # Long budget on purpose: a forced call must not spend any of it.
    agent._compression_lock_wait_seconds = 30.0
    head = [dict(msg) for msg in history[:12]]

    started = time.monotonic()
    with caplog.at_level(logging.WARNING, logger="agent.conversation_compression"):
        returned, _sp = agent._compress_context(
            head, "sys", approx_tokens=120_000, force=True
        )
    elapsed = time.monotonic() - started

    assert returned is head
    assert elapsed < 5.0, "forced compression waited on the holder"
    assert not getattr(agent, "_last_compaction_in_place", False)
    tags = [record.getMessage() for record in caplog.records]
    assert any(
        "compression skipped: another path is compressing" in tag
        and "no compacted state to adopt" not in tag
        for tag in tags
    )
    assert not any("holder timeout" in tag for tag in tags)
