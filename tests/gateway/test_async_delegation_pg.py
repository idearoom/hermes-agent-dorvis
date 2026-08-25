"""Durable async-delegation store on Postgres (IdeaRoom AE-183).

``tools/async_delegation.py`` used to open its own raw sqlite3 connection to
``$HERMES_HOME/state.db``, which on AWS is an EFS file that two Fargate tasks
share during a blue/green drain — a second writer on the file that the
2026-07-27 incident corrupted. It now persists through the same SessionDB
dispatch as the session store.

Requires a reachable Postgres and is skipped otherwise, same contract as
``test_session_store_pg.py``: set ``HERMES_STATE_TEST_DSN`` (or the D6a
``HERMES_D6_TEST_DSN``) to a throwaway database — each test drops and
recreates the ``hermes_state`` schema.
"""

import json
import os
import queue
import time

import pytest

_DSN = (
    os.environ.get("HERMES_STATE_TEST_DSN", "").strip()
    or os.environ.get("HERMES_D6_TEST_DSN", "").strip()
)

psycopg = pytest.importorskip("psycopg")
pytest.importorskip("psycopg_pool")

if not _DSN:
    pytest.skip(
        "HERMES_STATE_TEST_DSN / HERMES_D6_TEST_DSN not set",
        allow_module_level=True,
    )

import gateway.status as status_mod
from hermes_state_pg import _SCHEMA
from tools import async_delegation as ad

_FOREIGN = "some-other-fargate-task"


def _drop_schema():
    with psycopg.connect(_DSN, autocommit=True) as conn:
        conn.execute(f"DROP SCHEMA IF EXISTS {_SCHEMA} CASCADE")


def _query(sql, params=None):
    with psycopg.connect(_DSN, autocommit=True) as conn:
        conn.execute(f"SET search_path TO {_SCHEMA}, public")
        return conn.execute(sql, params).fetchall()


@pytest.fixture()
def pg_store(tmp_path, monkeypatch):
    """Point the delegation store at Postgres for the duration of one test."""
    _drop_schema()
    # A real HERMES_HOME so "did anything touch state.db?" is answerable.
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv(ad._STORE_DSN_ENV, _DSN)
    ad._reset_for_tests()
    yield tmp_path
    ad._reset_for_tests()


def _dispatch_row(delegation_id, **overrides):
    record = {
        "delegation_id": delegation_id,
        "session_key": "owner",
        "origin_ui_session_id": "",
        "parent_session_id": None,
        "dispatched_at": 1.0,
    }
    record.update(overrides)
    ad._persist_dispatch(record)


def _stored(delegation_id):
    """``get_durable_delegation`` narrowed to a present row."""
    row = ad.get_durable_delegation(delegation_id)
    assert row is not None, f"{delegation_id} is not in the durable store"
    return row


def _force_owner(delegation_id, *, instance, pid, updated_at):
    with psycopg.connect(_DSN, autocommit=True) as conn:
        conn.execute(f"SET search_path TO {_SCHEMA}, public")
        conn.execute(
            "UPDATE async_delegations SET owner_instance=%s, owner_pid=%s, "
            "owner_started_at=NULL, updated_at=%s WHERE delegation_id=%s",
            (instance, pid, updated_at, delegation_id),
        )


# ── Dispatch ───────────────────────────────────────────────────────────────


def test_dispatch_writes_to_postgres_and_never_opens_state_db(pg_store):
    _dispatch_row("deleg_pg")

    rows = _query(
        "SELECT origin_session, state, delivery_state, owner_pid, owner_instance "
        "FROM async_delegations WHERE delegation_id=%s", ("deleg_pg",)
    )
    assert rows == [("owner", "running", "pending", os.getpid(), ad._OWNER_INSTANCE)]
    # The point of the change: no second writer on the shared EFS file.
    assert not (pg_store / "state.db").exists()


def test_dispatch_without_dsn_stays_on_the_local_sqlite_file(tmp_path, monkeypatch):
    _drop_schema()
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.delenv(ad._STORE_DSN_ENV, raising=False)
    ad._reset_for_tests()
    try:
        _dispatch_row("deleg_local")
        assert _stored("deleg_local")["state"] == "running"
        assert (tmp_path / "state.db").exists()
    finally:
        ad._reset_for_tests()

    with psycopg.connect(_DSN, autocommit=True) as conn:
        exists = conn.execute(
            "SELECT to_regclass(%s)", (f"{_SCHEMA}.async_delegations",)
        ).fetchone()[0]
    assert exists is None


def test_redispatching_an_id_resets_the_terminal_columns(pg_store):
    """The Postgres path must reproduce SQLite's INSERT OR REPLACE semantics."""
    _dispatch_row("deleg_reuse")
    ad._persist_completion(
        {"delegation_id": "deleg_reuse", "status": "completed", "completed_at": 2.0},
        {"status": "completed", "summary": "first"},
    )
    _dispatch_row("deleg_reuse", dispatched_at=9.0)

    row = _stored("deleg_reuse")
    assert row["state"] == "running"
    assert row["completed_at"] is None
    assert row["result"] is None
    assert row["dispatched_at"] == 9.0


# ── Persist / complete / claim / deliver round-trip ────────────────────────


def test_completion_round_trip_and_exclusive_claim(pg_store):
    _dispatch_row("deleg_claim")
    ad._persist_completion(
        {
            "type": "async_delegation", "delegation_id": "deleg_claim",
            "status": "completed", "completed_at": time.time(), "summary": "done",
        },
        {"status": "completed", "summary": "done"},
    )

    durable = _stored("deleg_claim")
    assert durable["state"] == "completed"
    assert durable["result"]["summary"] == "done"
    assert durable["delivery_state"] == "pending"
    assert durable["delivery_attempts"] == 0

    restored = queue.Queue()
    assert ad.restore_undelivered_completions(restored) == 1
    assert restored.get_nowait()["restored"] is True

    # The CAS guards are what keep two gateway tasks from double-delivering.
    assert ad.claim_completion_delivery("deleg_claim", "consumer-a")
    assert not ad.claim_completion_delivery("deleg_claim", "consumer-b")
    assert ad.release_completion_delivery("deleg_claim", "consumer-a")
    assert ad.claim_completion_delivery("deleg_claim", "consumer-b")
    assert ad.complete_completion_delivery("deleg_claim", "consumer-b")
    assert not ad.claim_completion_delivery("deleg_claim", "consumer-c")

    assert _stored("deleg_claim")["delivery_state"] == "delivered"
    assert ad.restore_undelivered_completions(queue.Queue()) == 0


def test_mark_delivered_is_idempotent_and_reports_rowcount(pg_store):
    _dispatch_row("deleg_ack")
    ad._persist_completion(
        {"delegation_id": "deleg_ack", "status": "completed", "completed_at": 2.0},
        {"status": "completed", "summary": "s"},
    )
    assert ad.mark_completion_delivered("deleg_ack") is True
    assert ad.mark_completion_delivered("deleg_ack") is False
    assert ad.mark_completion_delivered("deleg_missing") is False


def test_delivery_attempt_and_delete(pg_store):
    _dispatch_row("deleg_attempt")
    ad._note_delivery_attempt("deleg_attempt")
    ad._note_delivery_attempt("deleg_attempt")
    assert _stored("deleg_attempt")["delivery_attempts"] == 2

    ad._delete_durable_delegation("deleg_attempt")
    assert ad.get_durable_delegation("deleg_attempt") is None


# ── Owner identity: the cross-task false-abandonment hazard ────────────────


def test_recover_reclaims_a_dead_pid_owned_by_this_instance(pg_store, monkeypatch):
    monkeypatch.setattr(status_mod, "_pid_exists", lambda pid: False)
    _dispatch_row("deleg_mine")
    _force_owner(
        "deleg_mine", instance=ad._OWNER_INSTANCE, pid=99999999, updated_at=1.0
    )

    assert ad.recover_abandoned_delegations() == 1
    assert _stored("deleg_mine")["state"] == "unknown"


def test_recover_never_touches_a_fresh_row_owned_by_another_instance(
    pg_store, monkeypatch
):
    """The blue/green hazard: the incoming task must not bury live work.

    ``_pid_exists`` is forced False the way the OTHER task's pids genuinely
    look from inside this PID namespace.
    """
    import time

    monkeypatch.setattr(status_mod, "_pid_exists", lambda pid: False)
    _dispatch_row("deleg_theirs")
    _force_owner(
        "deleg_theirs", instance=_FOREIGN, pid=99999999, updated_at=time.time()
    )

    assert ad.recover_abandoned_delegations() == 0
    assert _stored("deleg_theirs")["state"] == "running"
    assert ad.restore_undelivered_completions(queue.Queue()) == 0


def test_recover_reclaims_another_instance_only_past_the_staleness_ttl(
    pg_store, monkeypatch
):
    import time

    monkeypatch.setattr(status_mod, "_pid_exists", lambda pid: True)
    _dispatch_row("deleg_stale")
    _force_owner(
        "deleg_stale",
        instance=_FOREIGN,
        pid=os.getpid(),
        updated_at=time.time() - ad._FOREIGN_OWNER_STALE_SECONDS - 60,
    )

    # A live pid in THIS namespace must not save a foreign row either — the
    # identity mismatch means that pid number describes a different process.
    assert ad.recover_abandoned_delegations() == 1
    durable = _stored("deleg_stale")
    assert durable["state"] == "unknown"
    assert durable["delivery_state"] == "pending"


def test_recover_treats_a_null_identity_as_same_host_legacy(pg_store, monkeypatch):
    """Rows written before this column keep the pre-AE-183 pid semantics."""
    import time

    monkeypatch.setattr(status_mod, "_pid_exists", lambda pid: False)
    _dispatch_row("deleg_legacy")
    _force_owner("deleg_legacy", instance=None, pid=99999999, updated_at=time.time())

    assert ad.recover_abandoned_delegations() == 1
    assert _stored("deleg_legacy")["state"] == "unknown"


def test_recovered_event_payload_survives_the_round_trip(pg_store, monkeypatch):
    monkeypatch.setattr(status_mod, "_pid_exists", lambda pid: False)
    ad._persist_dispatch(
        {
            "delegation_id": "deleg_payload",
            "session_key": "owner-session",
            "origin_ui_session_id": "ui-1",
            "parent_session_id": "parent-1",
            "dispatched_at": 1.0,
            "goal": "research the thing",
            "toolsets": ["web"],
            "role": "leaf",
            "model": "m",
            "is_batch": True,
        }
    )
    _force_owner(
        "deleg_payload", instance=ad._OWNER_INSTANCE, pid=99999999, updated_at=1.0
    )
    assert ad.recover_abandoned_delegations() == 1

    restored = queue.Queue()
    assert ad.restore_undelivered_completions(restored) == 1
    evt = restored.get_nowait()
    assert evt["status"] == "unknown"
    assert evt["goal"] == "research the thing"
    assert evt["toolsets"] == ["web"]
    assert evt["is_batch"] is True
    assert evt["parent_session_id"] == "parent-1"
    assert evt["origin_ui_session_id"] == "ui-1"


# ── Retention ──────────────────────────────────────────────────────────────


def test_prune_drops_delivered_before_undelivered(pg_store, monkeypatch):
    monkeypatch.setattr(ad, "_MAX_RETAINED_COMPLETED", 2)
    for index, delivery_state in enumerate(("pending", "delivered", "pending")):
        delegation_id = f"deleg_{index}"
        _dispatch_row(delegation_id, dispatched_at=float(index + 1))
        ad._persist_completion(
            {
                "delegation_id": delegation_id,
                "status": "completed",
                "completed_at": float(index + 1),
            },
            {"status": "completed", "summary": delegation_id},
        )
        if delivery_state == "delivered":
            ad.mark_completion_delivered(delegation_id)

    ad._prune_durable_records()

    assert ad.get_durable_delegation("deleg_0") is not None
    assert ad.get_durable_delegation("deleg_1") is None
    assert ad.get_durable_delegation("deleg_2") is not None


def test_prune_expires_delivered_rows_past_the_retention_window(pg_store):
    _dispatch_row("deleg_old")
    ad._persist_completion(
        {"delegation_id": "deleg_old", "status": "completed", "completed_at": 1.0},
        {"status": "completed", "summary": "old"},
    )
    ad.mark_completion_delivered("deleg_old")
    with psycopg.connect(_DSN, autocommit=True) as conn:
        conn.execute(f"SET search_path TO {_SCHEMA}, public")
        conn.execute("UPDATE async_delegations SET updated_at=0")

    ad._prune_durable_records()
    assert ad.get_durable_delegation("deleg_old") is None


def test_pending_overflow_is_bounded(pg_store, monkeypatch):
    monkeypatch.setattr(ad, "_MAX_RETAINED_COMPLETED", 100)
    monkeypatch.setattr(ad, "_MAX_DURABLE_PENDING", 1)
    for index in range(3):
        delegation_id = f"deleg_pending_{index}"
        _dispatch_row(delegation_id, dispatched_at=float(index + 1))
        ad._persist_completion(
            {
                "delegation_id": delegation_id,
                "status": "completed",
                "completed_at": float(index + 1),
            },
            {"status": "completed", "summary": delegation_id},
        )

    ad._prune_durable_records()
    remaining = _query(
        "SELECT COUNT(*) FROM async_delegations WHERE delivery_state='pending'"
    )
    assert remaining[0][0] == 1


# ── End-to-end through the public dispatch API ─────────────────────────────


def test_background_dispatch_completes_through_postgres(pg_store):
    import time

    from tools.process_registry import process_registry

    while not process_registry.completion_queue.empty():
        process_registry.completion_queue.get_nowait()

    handle = ad.dispatch_async_delegation(
        goal="durable on pg", context="ctx", toolsets=["terminal"], role="leaf",
        model="m", session_key="owner", parent_session_id="parent",
        runner=lambda: {"status": "completed", "summary": "survived"},
    )
    assert handle["status"] == "dispatched"

    deadline = time.monotonic() + 10
    while ad.active_count() and time.monotonic() < deadline:
        time.sleep(0.02)

    durable = _stored(handle["delegation_id"])
    assert durable["state"] == "completed"
    assert durable["result"]["summary"] == "survived"
    stored = _query(
        "SELECT event_json FROM async_delegations WHERE delegation_id=%s",
        (handle["delegation_id"],),
    )
    assert json.loads(stored[0][0])["summary"] == "survived"
    assert not (pg_store / "state.db").exists()
