"""Postgres-backed SessionDB integration tests (IdeaRoom D6b / AE-115).

Requires a reachable Postgres and is skipped otherwise, mirroring
``test_response_store_pg.py``: set ``HERMES_STATE_TEST_DSN`` (or reuse the
D6a ``HERMES_D6_TEST_DSN``) to something like::

    postgresql://postgres:test@localhost:54330/postgres

Each test drops and recreates the ``hermes_state`` schema, so point it at a
throwaway database.
"""

import json
import os
import subprocess
import sys
import textwrap
import threading
import time
from pathlib import Path

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

import hermes_state
from hermes_state import AsyncSessionDB, SessionDB
import hermes_state_pg
from hermes_state_pg import _SCHEMA, EXPECTED_SCHEMA_VERSION, PgSessionDB

_REPO_ROOT = Path(__file__).resolve().parents[2]
_V22_SCHEMA_FIXTURE = _REPO_ROOT / "tests" / "fixtures" / "hermes_state_pg_v22.sql"
_V22_SURFACE_MARKER = "ffb802aede5aab2e95d1eb46188864c11b4b8e290c538ada64c06b9a14747654"
_V22_STATEMENT_DELIMITER = "-- hermes-v22-statement\n"


def _drop_schema():
    with psycopg.connect(_DSN, autocommit=True) as conn:
        conn.execute(f"DROP SCHEMA IF EXISTS {_SCHEMA} CASCADE")


@pytest.fixture()
def pg_db():
    _drop_schema()
    db = PgSessionDB(dsn=_DSN)
    yield db
    db.close()


@pytest.fixture()
def sqlite_db(tmp_path):
    db = SessionDB(tmp_path / "state.db")
    yield db
    db.close()


# ── CRUD / append / load round-trip ────────────────────────────────────────


def test_full_crud_round_trip(pg_db):
    sid = pg_db.create_session(
        "sess-1", "webui", user_id="u1", model="m1",
        model_config={"temperature": 0.2}, system_prompt="sys",
    )
    assert sid == "sess-1"

    mid1 = pg_db.append_message(sid, "user", "hello world")
    mid2 = pg_db.append_message(
        sid,
        "assistant",
        "hi there",
        tool_calls=[{"id": "t1", "function": {"name": "f"}}],
        token_count=7,
        finish_reason="stop",
        reasoning="thinking...",
    )
    assert isinstance(mid1, int) and mid2 > mid1

    # Multimodal content round-trips through the JSON envelope.
    pg_db.append_message(
        sid, "user", [{"type": "text", "text": "part"}, {"type": "image_url"}]
    )

    msgs = pg_db.get_messages(sid)
    assert [m["role"] for m in msgs] == ["user", "assistant", "user"]
    assert msgs[0]["content"] == "hello world"
    assert msgs[1]["tool_calls"][0]["id"] == "t1"
    assert isinstance(msgs[2]["content"], list)

    sess = pg_db.get_session(sid)
    assert sess["message_count"] == 3
    assert sess["tool_call_count"] == 1
    assert json.loads(sess["model_config"])["temperature"] == 0.2

    conv = pg_db.get_messages_as_conversation(sid)
    assert any(m["role"] == "assistant" for m in conv)
    assert [m["role"] for m in pg_db.get_messages(sid, offset=1)] == [
        "assistant",
        "user",
    ]

    pg_db.update_token_counts(sid, input_tokens=100, output_tokens=50)
    assert pg_db.get_session(sid)["input_tokens"] == 100

    assert pg_db.delete_session(sid) is True
    assert pg_db.get_session(sid) is None
    assert pg_db.get_messages(sid) == []


def test_second_current_schema_boot_is_catalog_neutral(pg_db):
    with psycopg.connect(_DSN, autocommit=True) as conn:
        before = _catalog_signature(conn)

    second = PgSessionDB(dsn=_DSN)
    second.close()

    with psycopg.connect(_DSN, autocommit=True) as conn:
        assert _catalog_signature(conn) == before


def test_fresh_schema_uses_current_surface_and_summary_column(pg_db):
    assert pg_db.storage_attestation() == {
        "backend": "postgres",
        "schema_version": EXPECTED_SCHEMA_VERSION,
        "surface_marker": hermes_state_pg.EXPECTED_SCHEMA_SURFACE_SHA256,
    }
    with psycopg.connect(_DSN, autocommit=True) as conn:
        column = conn.execute(
            "SELECT data_type, is_nullable, column_default "
            "FROM information_schema.columns "
            "WHERE table_schema = %s AND table_name = 'messages' "
            "AND column_name = '_compressed_summary'",
            (_SCHEMA,),
        ).fetchone()
        assert column == ("bigint", "NO", "0")


def test_replace_messages_and_counters(pg_db):
    sid = pg_db.create_session("sess-r", "webui")
    for i in range(4):
        pg_db.append_message(sid, "user", f"old {i}")
    pg_db.replace_messages(sid, [{"role": "user", "content": "new only"}])
    msgs = pg_db.get_messages(sid)
    assert len(msgs) == 1 and msgs[0]["content"] == "new only"
    assert pg_db.get_session(sid)["message_count"] == 1


# ── Gateway peer metadata + routing index ──────────────────────────────────


def test_gateway_peer_and_routing_round_trip(pg_db):
    sid = pg_db.create_session("sess-gw", "webui", user_id="u1")
    pg_db.record_gateway_session_peer(
        sid,
        source="webui",
        user_id="u1",
        session_key="webui:u1:c1",
        chat_id="c1",
        chat_type="dm",
        thread_id=None,
        display_name="Dalton",
        origin_json='{"chat_id": "c1"}',
    )
    pg_db.set_expiry_finalized(sid, True)
    sess = pg_db.get_session(sid)
    assert sess["display_name"] == "Dalton"
    assert sess["expiry_finalized"] == 1

    rows = pg_db.list_gateway_sessions(platform="webui")
    assert [r["id"] for r in rows] == [sid]
    assert (
        pg_db.find_session_by_origin(platform="webui", chat_id="c1", user_id="u1")
        == sid
    )

    pg_db.save_gateway_routing_entry("k1", '{"v": 1}', scope="dirA")
    pg_db.save_gateway_routing_entry("k2", '{"v": 2}', scope="dirA")
    pg_db.save_gateway_routing_entry("k1", '{"v": 9}', scope="dirB")
    assert pg_db.load_gateway_routing_entries(scope="dirA") == {
        "k1": '{"v": 1}',
        "k2": '{"v": 2}',
    }

    pg_db.replace_gateway_routing_entries({"k3": '{"v": 3}'}, scope="dirA")
    assert pg_db.load_gateway_routing_entries(scope="dirA") == {"k3": '{"v": 3}'}
    # Other scope untouched by the replace.
    assert pg_db.load_gateway_routing_entries(scope="dirB") == {"k1": '{"v": 9}'}

    pg_db.delete_gateway_routing_entries(["k3"], scope="dirA")
    assert pg_db.load_gateway_routing_entries(scope="dirA") == {}


# ── archive_and_compact atomicity ──────────────────────────────────────────


def test_archive_and_compact_soft_archives_and_inserts(pg_db):
    sid = pg_db.create_session("sess-c", "webui")
    for i in range(5):
        pg_db.append_message(sid, "user", f"pre-compaction {i}")
    n = pg_db.archive_and_compact(
        sid, [{"role": "user", "content": "the summary"}]
    )
    assert n == 1
    live = pg_db.get_messages(sid)
    assert [m["content"] for m in live] == ["the summary"]
    everything = pg_db.get_messages(sid, include_inactive=True)
    assert len(everything) == 6
    archived = [m for m in everything if m["active"] == 0]
    assert len(archived) == 5
    assert all(m["compacted"] == 1 for m in archived)
    assert pg_db.has_archived_messages(sid) is True
    assert pg_db.get_session(sid)["message_count"] == 1
    # Archived (compacted=1) rows stay searchable by default.
    hits = pg_db.search_messages("pre-compaction")
    assert len(hits) == 5


def test_archive_and_compact_rolls_back_atomically(pg_db):
    sid = pg_db.create_session("sess-atom", "webui")
    for i in range(3):
        pg_db.append_message(sid, "user", f"keep me {i}")
    # Second message is unstorable (token_count must be numeric) — the whole
    # transaction (soft-archive + inserts) must roll back.
    bad = [
        {"role": "user", "content": "summary"},
        {"role": "user", "content": "boom", "token_count": {"not": "an int"}},
    ]
    with pytest.raises(Exception):
        pg_db.archive_and_compact(sid, bad)
    live = pg_db.get_messages(sid)
    assert [m["content"] for m in live] == ["keep me 0", "keep me 1", "keep me 2"]
    assert all(m["active"] == 1 and m["compacted"] == 0 for m in live)
    assert pg_db.get_session(sid)["message_count"] == 3


# ── FTS parity (tsvector vs FTS5) ──────────────────────────────────────────

_CORPUS = [
    ("user", "we should deploy the new gateway build today"),
    ("assistant", "deployment finished without errors"),
    ("user", "docker compose keeps restarting the browserless container"),
    ("user", "let's talk about kubernetes ingress and docker networking"),
    ("assistant", "the chat-send helper drops messages on reconnect"),
    ("user", "大别山项目的进展如何"),
    ("assistant", "大别山项目 进展顺利, 明天继续"),
    ("user", "totally unrelated message about gardening"),
]

_PARITY_QUERIES = [
    "docker",                # exact word
    "deploy*",               # prefix
    "docker kubernetes",     # multi-term implicit AND
    "docker OR gardening",   # boolean OR
    '"docker compose"',      # phrase
    "chat-send",             # hyphenated term (sanitizer quotes it)
    "大别山项目",              # non-ASCII (CJK trigram/ILIKE path)
]


def _index_by_content(db, sid):
    return {m["content"]: m["id"] for m in db.get_messages(sid)}


@pytest.mark.parametrize("query", _PARITY_QUERIES)
def test_fts_parity_with_sqlite(pg_db, sqlite_db, query):
    if not sqlite_db._fts_enabled:
        pytest.skip("SQLite build lacks FTS5")
    for db in (pg_db, sqlite_db):
        sid = db.create_session("sess-fts", "webui")
        for role, content in _CORPUS:
            db.append_message(sid, role, content)

    def matched_contents(db):
        by_content = _index_by_content(db, "sess-fts")
        by_id = {v: k for k, v in by_content.items()}
        return {by_id[m["id"]] for m in db.search_messages(query, limit=50)}

    pg_hits = matched_contents(pg_db)
    sq_hits = matched_contents(sqlite_db)
    assert pg_hits == sq_hits, (
        f"FTS parity broken for {query!r}: pg={pg_hits} sqlite={sq_hits}"
    )
    assert pg_hits, f"parity query {query!r} matched nothing on either backend"


def test_search_filters_and_sort(pg_db):
    sid = pg_db.create_session("sess-f", "webui")
    other = pg_db.create_session("sess-o", "cli")
    pg_db.append_message(sid, "user", "alpha needle one")
    pg_db.append_message(sid, "assistant", "alpha needle two")
    pg_db.append_message(other, "user", "alpha needle three")

    assert len(pg_db.search_messages("needle")) == 3
    assert len(pg_db.search_messages("needle", source_filter=["webui"])) == 2
    assert len(pg_db.search_messages("needle", exclude_sources=["webui"])) == 1
    assert len(pg_db.search_messages("needle", role_filter=["assistant"])) == 1

    newest = pg_db.search_messages("needle", sort="newest")
    assert [m["snippet"] for m in newest][0].count("needle") >= 1
    ts = [m["timestamp"] for m in newest]
    assert ts == sorted(ts, reverse=True)

    hit = pg_db.search_messages("needle", role_filter=["assistant"])[0]
    # prev + match (+ next when one exists; the assistant hit is last in its
    # session, matching the SQLite implementation's context shape).
    assert "context" in hit and len(hit["context"]) == 2
    assert "content" not in hit


# ── Compression locks: cross-connection correctness ────────────────────────


def test_compression_lock_contention_two_connections(pg_db):
    """Two pools (= two processes' worth of connections) against one DB."""
    db2 = PgSessionDB(dsn=_DSN)
    try:
        sid = pg_db.create_session("sess-lock", "webui")

        assert pg_db.try_acquire_compression_lock(sid, "holder-A", 5.0) is True
        # Graceful conflict, not a PK-violation exception:
        assert db2.try_acquire_compression_lock(sid, "holder-B", 5.0) is False
        assert db2.get_compression_lock_holder(sid) == "holder-A"

        # Refresh only works for the owner.
        assert pg_db.refresh_compression_lock(sid, "holder-A", 5.0) is True
        assert db2.refresh_compression_lock(sid, "holder-B", 5.0) is False

        # Release is owner-scoped and idempotent.
        db2.release_compression_lock(sid, "holder-B")  # no-op, not ours
        assert pg_db.get_compression_lock_holder(sid) == "holder-A"
        pg_db.release_compression_lock(sid, "holder-A")
        assert pg_db.get_compression_lock_holder(sid) is None

        # TTL expiry is honored: an expired lock is reclaimed transparently.
        assert pg_db.try_acquire_compression_lock(sid, "holder-A", 0.05) is True
        time.sleep(0.15)
        assert db2.try_acquire_compression_lock(sid, "holder-B", 5.0) is True
        assert pg_db.get_compression_lock_holder(sid) == "holder-B"
        db2.release_compression_lock(sid, "holder-B")
    finally:
        db2.close()


def test_compression_lock_thread_race_single_winner(pg_db):
    db2 = PgSessionDB(dsn=_DSN)
    try:
        sid = pg_db.create_session("sess-race", "webui")
        wins = []
        barrier = threading.Barrier(8)

        def worker(db, holder):
            barrier.wait()
            if db.try_acquire_compression_lock(sid, holder, 10.0):
                wins.append(holder)

        threads = [
            threading.Thread(
                target=worker,
                args=(pg_db if i % 2 == 0 else db2, f"h{i}"),
            )
            for i in range(8)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert len(wins) == 1
        assert pg_db.get_compression_lock_holder(sid) == wins[0]
    finally:
        db2.close()


# ── Schema-version assertion ───────────────────────────────────────────────


def test_boot_fails_on_upstream_schema_version_drift(monkeypatch):
    _drop_schema()
    monkeypatch.setattr(
        hermes_state, "SCHEMA_VERSION", EXPECTED_SCHEMA_VERSION + 1
    )
    with pytest.raises(RuntimeError, match="rebase-drift"):
        PgSessionDB(dsn=_DSN)


def test_boot_fails_on_persisted_version_mismatch(pg_db):
    """A database initialized by a different build refuses this build."""
    with psycopg.connect(_DSN, autocommit=True) as conn:
        conn.execute(f"UPDATE {_SCHEMA}.schema_version SET version = 18")
    with pytest.raises(RuntimeError, match="expects "):
        PgSessionDB(dsn=_DSN)


def test_boot_fails_on_persisted_surface_mismatch(pg_db):
    with psycopg.connect(_DSN, autocommit=True) as conn:
        conn.execute(
            f"UPDATE {_SCHEMA}.state_meta SET value = 'deadbeef' "
            "WHERE key = 'pg_backend_schema_surface_sha256'"
        )
    with pytest.raises(RuntimeError, match="Rebase drift"):
        PgSessionDB(dsn=_DSN)


def test_boot_rejects_same_named_wrong_unique_index(pg_db):
    with psycopg.connect(_DSN, autocommit=True) as conn:
        conn.execute(f"DROP INDEX {_SCHEMA}.idx_sessions_title_unique")
        conn.execute(
            f"CREATE INDEX idx_sessions_title_unique "
            f"ON {_SCHEMA}.sessions(started_at)"
        )

    with pytest.raises(RuntimeError, match="idx_sessions_title_unique"):
        PgSessionDB(dsn=_DSN)


# ── drained v22 → v26 explicit migration ────────────────────────────────────


def _create_frozen_v22_store(conn):
    """Create the exact predecessor catalog without consulting v26 DDL."""
    fixture = _V22_SCHEMA_FIXTURE.read_text(encoding="utf-8")
    statements = [
        statement.strip()
        for statement in fixture.split(_V22_STATEMENT_DELIMITER)[1:]
        if statement.strip()
    ]
    assert statements, "frozen v22 schema fixture contains no statements"
    for statement in statements:
        conn.execute(statement)
    conn.execute(
        f"INSERT INTO {_SCHEMA}.schema_version (version) VALUES (22)"
    )
    conn.execute(
        f"INSERT INTO {_SCHEMA}.state_meta (key, value) VALUES (%s, %s)",
        ("pg_backend_schema_surface_sha256", _V22_SURFACE_MARKER),
    )


def _catalog_signature(conn):
    """Stable catalog snapshot proving a rejected boot performed no DDL."""
    columns = conn.execute(
        "SELECT table_name, column_name, data_type, is_nullable, column_default "
        "FROM information_schema.columns WHERE table_schema = %s "
        "ORDER BY table_name, ordinal_position",
        (_SCHEMA,),
    ).fetchall()
    indexes = conn.execute(
        "SELECT indexname, indexdef FROM pg_indexes WHERE schemaname = %s "
        "ORDER BY indexname",
        (_SCHEMA,),
    ).fetchall()
    constraints = conn.execute(
        "SELECT conrelid::regclass::text, conname, pg_get_constraintdef(oid) "
        "FROM pg_constraint WHERE connamespace = %s::regnamespace "
        "ORDER BY conrelid::regclass::text, conname",
        (_SCHEMA,),
    ).fetchall()
    return (tuple(columns), tuple(indexes), tuple(constraints))


def _run_v26_migration(*extra_args):
    return subprocess.run(
        [
            sys.executable,
            str(_REPO_ROOT / "scripts" / "migrate_pg_schema_v22_to_v26.py"),
            "--dsn",
            _DSN,
            *extra_args,
        ],
        cwd=_REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )


def _run_v26_surface_migration(*extra_args):
    return subprocess.run(
        [
            sys.executable,
            str(_REPO_ROOT / "scripts" / "migrate_pg_schema_v26_surface.py"),
            "--dsn",
            _DSN,
            *extra_args,
        ],
        cwd=_REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )


def _downgrade_v26_summary_surface(conn):
    """Restore the exact pre-summary v26 source accepted by AE-240."""
    conn.execute(
        f"ALTER TABLE {_SCHEMA}.messages DROP COLUMN _compressed_summary"
    )
    conn.execute(
        f"UPDATE {_SCHEMA}.state_meta SET value = %s WHERE key = %s",
        (
            hermes_state_pg._V26_PRE_SUMMARY_SCHEMA_SURFACE_SHA256,
            hermes_state_pg._META_SURFACE_KEY,
        ),
    )


def test_v26_surface_migration_preserves_rows_and_is_idempotent():
    _drop_schema()
    current = PgSessionDB(dsn=_DSN)
    try:
        current.create_session("surface-bridge", source="webui")
        current.append_message("surface-bridge", "user", "before expansion")
    finally:
        current.close()

    with psycopg.connect(_DSN, autocommit=True) as conn:
        _downgrade_v26_summary_surface(conn)
        before = _catalog_signature(conn)

    with pytest.raises(RuntimeError, match="surface migration required"):
        PgSessionDB(dsn=_DSN)
    with psycopg.connect(_DSN, autocommit=True) as conn:
        assert _catalog_signature(conn) == before

    dry_run = _run_v26_surface_migration()
    assert dry_run.returncode == 0, dry_run.stderr
    assert "migration_required=true" in dry_run.stdout
    with psycopg.connect(_DSN, autocommit=True) as conn:
        assert _catalog_signature(conn) == before

    applied = _run_v26_surface_migration("--apply")
    assert applied.returncode == 0, applied.stderr
    assert "Migration applied" in applied.stdout

    target = PgSessionDB(dsn=_DSN)
    try:
        assert target.storage_attestation()["surface_marker"] == (
            hermes_state_pg.EXPECTED_SCHEMA_SURFACE_SHA256
        )
        assert target.get_messages("surface-bridge")[0]["content"] == (
            "before expansion"
        )
        target.append_message(
            "surface-bridge",
            "assistant",
            "after expansion",
            _compressed_summary=True,
        )
    finally:
        target.close()

    with psycopg.connect(_DSN, autocommit=True) as conn:
        marker = conn.execute(
            f"SELECT value FROM {_SCHEMA}.state_meta WHERE key = %s",
            (hermes_state_pg._META_SURFACE_KEY,),
        ).fetchone()[0]
        assert marker == hermes_state_pg.EXPECTED_SCHEMA_SURFACE_SHA256
        summary_markers = conn.execute(
            f"SELECT _compressed_summary FROM {_SCHEMA}.messages "
            "WHERE session_id = %s ORDER BY id",
            ("surface-bridge",),
        ).fetchall()
        assert summary_markers == [(0,), (1,)]

    replay = _run_v26_surface_migration("--apply")
    assert replay.returncode == 0, replay.stderr
    assert "already current" in replay.stdout


def test_v26_surface_migration_lock_timeout_rolls_back(monkeypatch):
    _drop_schema()
    bridge = PgSessionDB(dsn=_DSN)
    try:
        bridge.create_session("surface-lock", source="webui")
        bridge.append_message("surface-lock", "user", "preserve me")
    finally:
        bridge.close()
    with psycopg.connect(_DSN, autocommit=True) as conn:
        _downgrade_v26_summary_surface(conn)

    monkeypatch.setattr(
        hermes_state_pg, "_V26_MIGRATION_ADVISORY_TIMEOUT", "1000ms"
    )
    monkeypatch.setattr(
        hermes_state_pg, "_V26_MIGRATION_LOCK_TIMEOUT", "200ms"
    )
    monkeypatch.setattr(
        hermes_state_pg, "_V26_MIGRATION_STATEMENT_TIMEOUT", "1000ms"
    )

    blocker = psycopg.connect(_DSN, autocommit=False)
    try:
        blocker.execute(
            f"SELECT id FROM {_SCHEMA}.messages WHERE session_id = %s",
            ("surface-lock",),
        ).fetchall()
        started = time.monotonic()
        with pytest.raises(psycopg.errors.LockNotAvailable):
            hermes_state_pg.migrate_v26_surface(_DSN)
        assert time.monotonic() - started < 2.0
    finally:
        blocker.rollback()
        blocker.close()

    with psycopg.connect(_DSN, autocommit=True) as conn:
        marker = conn.execute(
            f"SELECT value FROM {_SCHEMA}.state_meta WHERE key = %s",
            (hermes_state_pg._META_SURFACE_KEY,),
        ).fetchone()[0]
        assert marker == hermes_state_pg._V26_PRE_SUMMARY_SCHEMA_SURFACE_SHA256
        assert conn.execute(
            "SELECT 1 FROM information_schema.columns "
            "WHERE table_schema = %s AND table_name = 'messages' "
            "AND column_name = '_compressed_summary'",
            (_SCHEMA,),
        ).fetchone() is None

    result = hermes_state_pg.migrate_v26_surface(_DSN)
    assert result["migration_applied"] is True


def test_v26_surface_migration_advisory_lock_is_bounded(monkeypatch):
    _drop_schema()
    bridge = PgSessionDB(dsn=_DSN)
    bridge.close()
    with psycopg.connect(_DSN, autocommit=True) as conn:
        _downgrade_v26_summary_surface(conn)
    monkeypatch.setattr(
        hermes_state_pg, "_V26_MIGRATION_ADVISORY_TIMEOUT", "200ms"
    )

    blocker = psycopg.connect(_DSN, autocommit=True)
    try:
        blocker.execute(
            "SELECT pg_advisory_lock(%s)",
            (PgSessionDB._BOOTSTRAP_LOCK_KEY,),
        )
        started = time.monotonic()
        with pytest.raises(psycopg.errors.LockNotAvailable):
            hermes_state_pg.migrate_v26_surface(_DSN)
        assert time.monotonic() - started < 2.0
    finally:
        blocker.execute(
            "SELECT pg_advisory_unlock(%s)",
            (PgSessionDB._BOOTSTRAP_LOCK_KEY,),
        )
        blocker.close()

    with psycopg.connect(_DSN, autocommit=True) as conn:
        assert conn.execute(
            f"SELECT value FROM {_SCHEMA}.state_meta WHERE key = %s",
            (hermes_state_pg._META_SURFACE_KEY,),
        ).fetchone()[0] == hermes_state_pg._V26_PRE_SUMMARY_SCHEMA_SURFACE_SHA256
        assert conn.execute(
            "SELECT 1 FROM information_schema.columns "
            "WHERE table_schema = %s AND table_name = 'messages' "
            "AND column_name = '_compressed_summary'",
            (_SCHEMA,),
        ).fetchone() is None


def test_v26_surface_migration_statement_timeout_rolls_back_ddl(monkeypatch):
    _drop_schema()
    bridge = PgSessionDB(dsn=_DSN)
    try:
        bridge.create_session("surface-rollback", source="webui")
        bridge.append_message("surface-rollback", "user", "preserve me")
    finally:
        bridge.close()
    with psycopg.connect(_DSN, autocommit=True) as conn:
        _downgrade_v26_summary_surface(conn)

    monkeypatch.setattr(
        hermes_state_pg, "_V26_MIGRATION_ADVISORY_TIMEOUT", "1000ms"
    )
    monkeypatch.setattr(
        hermes_state_pg, "_V26_MIGRATION_LOCK_TIMEOUT", "1000ms"
    )
    monkeypatch.setattr(
        hermes_state_pg, "_V26_MIGRATION_STATEMENT_TIMEOUT", "200ms"
    )

    blocker = psycopg.connect(_DSN, autocommit=False)
    try:
        blocker.execute(
            f"SELECT value FROM {_SCHEMA}.state_meta WHERE key = %s FOR UPDATE",
            (hermes_state_pg._META_SURFACE_KEY,),
        ).fetchone()
        started = time.monotonic()
        with pytest.raises(psycopg.errors.QueryCanceled):
            hermes_state_pg.migrate_v26_surface(_DSN)
        assert time.monotonic() - started < 2.0
    finally:
        blocker.rollback()
        blocker.close()

    with psycopg.connect(_DSN, autocommit=True) as conn:
        assert conn.execute(
            f"SELECT value FROM {_SCHEMA}.state_meta WHERE key = %s",
            (hermes_state_pg._META_SURFACE_KEY,),
        ).fetchone()[0] == hermes_state_pg._V26_PRE_SUMMARY_SCHEMA_SURFACE_SHA256
        assert conn.execute(
            "SELECT 1 FROM information_schema.columns "
            "WHERE table_schema = %s AND table_name = 'messages' "
            "AND column_name = '_compressed_summary'",
            (_SCHEMA,),
        ).fetchone() is None

        assert conn.execute(
            f"SELECT content FROM {_SCHEMA}.messages WHERE session_id = %s",
            ("surface-rollback",),
        ).fetchone()[0] == "preserve me"

    with pytest.raises(RuntimeError, match="surface migration required"):
        PgSessionDB(dsn=_DSN)


def test_migration_mode_refuses_absent_schema_without_initializing_it():
    _drop_schema()
    with pytest.raises(RuntimeError, match="requires an existing hermes_state"):
        PgSessionDB(dsn=_DSN, allow_schema_migration=True)
    with psycopg.connect(_DSN, autocommit=True) as conn:
        assert conn.execute(
            "SELECT to_regnamespace(%s)", (_SCHEMA,)
        ).fetchone()[0] is None


def test_v22_optional_trigram_index_is_accepted_by_live_migration():
    _drop_schema()
    with psycopg.connect(_DSN, autocommit=True) as conn:
        _create_frozen_v22_store(conn)
        for statement in hermes_state_pg.PG_TRGM_SQL:
            conn.execute(statement)

    dry_run = _run_v26_migration()
    assert dry_run.returncode == 0, dry_run.stderr

    applied = _run_v26_migration("--apply")
    assert applied.returncode == 0, applied.stderr


def test_v22_requires_explicit_migration_and_preserves_rows():
    old = "sess-v22-old"
    new = "sess-v22-new"
    _drop_schema()
    with psycopg.connect(_DSN, autocommit=True) as conn:
        _create_frozen_v22_store(conn)
        conn.execute(
            f"INSERT INTO {_SCHEMA}.sessions "
            "(id, source, model, system_prompt, started_at, title, "
            "message_count, input_tokens, output_tokens) "
            "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)",
            (
                old,
                "webui",
                "m1",
                "shared legacy prompt",
                10.0,
                "duplicate",
                1,
                10,
                5,
            ),
        )
        conn.execute(
            f"INSERT INTO {_SCHEMA}.sessions "
            "(id, source, model, system_prompt, started_at, title) "
            "VALUES (%s, %s, %s, %s, %s, %s)",
            (new, "webui", "m1", "shared legacy prompt", 20.0, "duplicate"),
        )
        conn.execute(
            f"INSERT INTO {_SCHEMA}.messages "
            "(session_id, role, content, timestamp) VALUES (%s, %s, %s, %s)",
            (old, "user", "written before the migration", 11.0),
        )
        before = _catalog_signature(conn)

    # Ordinary runtime boot is observation-only against v22.
    with pytest.raises(RuntimeError, match="schema migration required"):
        PgSessionDB(dsn=_DSN)
    with psycopg.connect(_DSN, autocommit=True) as conn:
        assert _catalog_signature(conn) == before

    # Default CLI mode performs a real read-only source preflight.
    dry_run = _run_v26_migration()
    assert dry_run.returncode == 0, dry_run.stderr
    assert "Dry run passed without database writes" in dry_run.stdout
    with psycopg.connect(_DSN, autocommit=True) as conn:
        assert _catalog_signature(conn) == before

    applied = _run_v26_migration("--apply")
    assert applied.returncode == 0, applied.stderr
    assert "Migration complete: backend=postgres schema_version=26" in applied.stdout

    db = PgSessionDB(dsn=_DSN)
    try:
        assert db.get_messages(old)[0]["content"] == "written before the migration"
        db.append_message(
            old,
            "assistant",
            "written after the migration",
            _compressed_summary=True,
            display_kind="internal_notification",
            display_metadata={"source": "migration-test"},
        )
        db.record_auxiliary_usage(
            old, "vision", model="gemini", input_tokens=7, output_tokens=3
        )
        assert db.storage_attestation() == {
            "backend": "postgres",
            "schema_version": EXPECTED_SCHEMA_VERSION,
            "surface_marker": hermes_state_pg.EXPECTED_SCHEMA_SURFACE_SHA256,
        }
    finally:
        db.close()

    with psycopg.connect(_DSN, autocommit=True) as conn:
        assert conn.execute(
            f"SELECT version FROM {_SCHEMA}.schema_version"
        ).fetchone()[0] == EXPECTED_SCHEMA_VERSION
        assert conn.execute(
            f"SELECT value FROM {_SCHEMA}.state_meta WHERE key = %s",
            (hermes_state_pg._META_SURFACE_KEY,),
        ).fetchone()[0] == hermes_state_pg.EXPECTED_SCHEMA_SURFACE_SHA256
        prompt_rows = conn.execute(
            f"SELECT prompt FROM {_SCHEMA}.system_prompts ORDER BY hash"
        ).fetchall()
        assert prompt_rows == [("shared legacy prompt",)]
        session_rows = conn.execute(
            f"SELECT id, system_prompt, system_prompt_hash, title "
            f"FROM {_SCHEMA}.sessions WHERE id IN (%s, %s) ORDER BY id",
            (old, new),
        ).fetchall()
        by_id = {row[0]: row[1:] for row in session_rows}
        assert by_id[old][0] is None
        assert by_id[new][0] is None
        assert by_id[old][1] == by_id[new][1]
        assert by_id[old][2] is None
        assert by_id[new][2] == "duplicate"
        display = conn.execute(
            f"SELECT display_kind, display_metadata, _compressed_summary "
            f"FROM {_SCHEMA}.messages "
            "WHERE session_id = %s ORDER BY id DESC LIMIT 1",
            (old,),
        ).fetchone()
        assert display == (
            "internal_notification",
            '{"source": "migration-test"}',
            1,
        )

    # Migration mode is deliberately one-shot, not an idempotent ensure-current
    # operation: replaying it against v26 must refuse a wrong/reused target.
    replay = _run_v26_migration("--apply")
    assert replay.returncode != 0
    assert "requires schema_version 22 exactly" in replay.stderr


def test_migration_preflight_rejects_partial_v22_without_widening():
    _drop_schema()
    with psycopg.connect(_DSN, autocommit=True) as conn:
        _create_frozen_v22_store(conn)
        conn.execute(
            f"ALTER TABLE {_SCHEMA}.sessions DROP COLUMN compression_fallback_streak"
        )
        before = _catalog_signature(conn)

    dry_run = _run_v26_migration()
    assert dry_run.returncode != 0
    assert "v22 migration source catalog verification failed" in dry_run.stderr
    applied = _run_v26_migration("--apply")
    assert applied.returncode != 0
    assert "v22 migration source catalog verification failed" in applied.stderr

    with psycopg.connect(_DSN, autocommit=True) as conn:
        assert _catalog_signature(conn) == before
        assert conn.execute(
            f"SELECT version FROM {_SCHEMA}.schema_version"
        ).fetchone()[0] == 22


def test_migration_preflight_rejects_same_named_wrong_v22_index():
    _drop_schema()
    with psycopg.connect(_DSN, autocommit=True) as conn:
        _create_frozen_v22_store(conn)
        conn.execute(f"DROP INDEX {_SCHEMA}.idx_sessions_started")
        conn.execute(
            f"CREATE INDEX idx_sessions_started ON {_SCHEMA}.sessions(source)"
        )
        before = _catalog_signature(conn)

    dry_run = _run_v26_migration()
    assert dry_run.returncode != 0
    assert "idx_sessions_started" in dry_run.stderr
    applied = _run_v26_migration("--apply")
    assert applied.returncode != 0
    assert "idx_sessions_started" in applied.stderr

    with psycopg.connect(_DSN, autocommit=True) as conn:
        assert _catalog_signature(conn) == before
        assert conn.execute(
            f"SELECT version FROM {_SCHEMA}.schema_version"
        ).fetchone()[0] == 22


def test_per_model_usage_accumulates_across_calls(pg_db):
    """Covers the DO UPDATE SET statement override (bare self-references)."""
    sid = pg_db.create_session("sess-usage", "webui", model="m1")
    for _ in range(3):
        pg_db.record_auxiliary_usage(
            sid, "compression", model="m-aux", input_tokens=100, output_tokens=20
        )
    with psycopg.connect(_DSN, autocommit=True) as conn:
        row = conn.execute(
            f"SELECT api_call_count, input_tokens, output_tokens FROM "
            f"{_SCHEMA}.session_model_usage "
            "WHERE session_id = %s AND task = 'compression'",
            (sid,),
        ).fetchone()
    assert row == (3, 300, 60)


def test_api_content_and_effect_disposition_round_trip(pg_db):
    sid = pg_db.create_session("sess-cols", "webui")
    pg_db.append_message(
        sid, "tool", "result", tool_name="t", effect_disposition="none",
        api_content="exact-bytes-sent",
    )
    msg = pg_db.get_messages(sid)[0]
    assert msg.get("api_content") == "exact-bytes-sent"
    assert msg.get("effect_disposition") == "none"


def test_set_latest_user_api_content_backfills_on_pg(pg_db):
    """The in-place compaction api_content backfill must work on Postgres.

    Regression guard: ``set_latest_user_api_content`` uses SQLite's
    null-safe ``content IS ?``; without the translator's
    ``IS NOT DISTINCT FROM`` rewrite this raised a Postgres syntax error
    on every call ("in-place compaction api_content backfill failed").
    """
    sid = pg_db.create_session("sess-backfill", "webui")
    pg_db.append_message(sid, "user", "old turn")
    pg_db.append_message(sid, "assistant", "reply")
    pg_db.append_message(sid, "user", "turn text")

    assert (
        pg_db.set_latest_user_api_content(sid, "turn text", "turn text\n\nCTX")
        == 1
    )
    msgs = pg_db.get_messages(sid)
    assert msgs[-1]["api_content"] == "turn text\n\nCTX"
    assert msgs[0].get("api_content") is None  # only the newest user row

    # The defensive content guard still holds: mismatched content → 0 rows.
    assert pg_db.set_latest_user_api_content(sid, "other text", "nope") == 0
    assert pg_db.get_messages(sid)[-1]["api_content"] == "turn text\n\nCTX"


# ── Env-var dispatch, end to end ───────────────────────────────────────────


def test_sessiondb_env_dispatch_end_to_end(monkeypatch):
    _drop_schema()
    monkeypatch.setenv("HERMES_STATE_STORE_DSN", _DSN)
    db = SessionDB()
    try:
        assert type(db) is PgSessionDB
        sid = db.create_session("sess-env", "webui")
        db.append_message(sid, "user", "dispatched")
        assert db.get_messages(sid)[0]["content"] == "dispatched"
    finally:
        db.close()


def test_async_facade_over_pg(pg_db):
    import asyncio

    facade = AsyncSessionDB(pg_db)

    async def flow():
        await facade.create_session("sess-async", "webui")
        await facade.append_message("sess-async", "user", "via facade")
        return await facade.get_messages("sess-async")

    msgs = asyncio.run(flow())
    assert msgs[0]["content"] == "via facade"


# ── Migration script ───────────────────────────────────────────────────────


def test_migration_script_round_trip(tmp_path):
    _drop_schema()
    src = tmp_path / "state.db"
    sdb = SessionDB(src)
    parent = sdb.create_session("mig-parent", "webui", user_id="u1")
    child = sdb.create_session(
        "mig-child", "webui", parent_session_id=parent
    )
    for i in range(6):
        sdb.append_message(parent, "user", f"parent msg {i}")
    sdb.append_message(child, "assistant", "child msg")
    # Multimodal content: SQLite stores it with the "\x00json:" sentinel;
    # the migration must rewrite it to the Pg backend's "\x01json:" form.
    sdb.append_message(
        child, "user", [{"type": "text", "text": "multimodal part"}]
    )
    # Soft-archive so active/compacted flags must survive the copy.
    sdb.archive_and_compact(parent, [{"role": "user", "content": "summary"}])
    sdb.save_gateway_routing_entry("k1", '{"v": 1}', scope="s")
    sdb.set_meta("some_key", "some_value")
    sdb.try_acquire_compression_lock(parent, "mig-holder", 3600.0)
    src_active = len(sdb.get_messages(parent))
    src_all = len(sdb.get_messages(parent, include_inactive=True))
    sdb.close()

    script = _REPO_ROOT / "scripts" / "migrate_state_to_postgres.py"

    # Dry run writes nothing.
    out = subprocess.run(
        [sys.executable, str(script), "--sqlite", str(src), "--dsn", _DSN],
        capture_output=True, text=True, cwd=_REPO_ROOT,
    )
    assert out.returncode == 0, out.stderr
    assert "Dry run" in out.stdout

    out = subprocess.run(
        [sys.executable, str(script), "--sqlite", str(src), "--dsn", _DSN,
         "--apply"],
        capture_output=True, text=True, cwd=_REPO_ROOT,
    )
    assert out.returncode == 0, out.stderr + out.stdout
    assert "Migration complete." in out.stdout
    assert "MISMATCH" not in out.stdout

    pdb = PgSessionDB(dsn=_DSN)
    try:
        assert len(pdb.get_messages(parent)) == src_active
        assert len(pdb.get_messages(parent, include_inactive=True)) == src_all
        assert pdb.get_session(child)["parent_session_id"] == parent
        child_msgs = pdb.get_messages(child)
        assert child_msgs[1]["content"] == [
            {"type": "text", "text": "multimodal part"}
        ]
        assert pdb.load_gateway_routing_entries(scope="s") == {"k1": '{"v": 1}'}
        assert pdb.get_meta("some_key") == "some_value"
        assert pdb.get_compression_lock_holder(parent) == "mig-holder"
        # Identity sequence resumes past migrated ids.
        new_id = pdb.append_message(child, "user", "post-migration")
        max_migrated = max(
            m["id"] for m in pdb.get_messages(parent, include_inactive=True)
        )
        assert new_id > max_migrated
    finally:
        pdb.close()

    # Second --apply without --allow-nonempty refuses; with it, idempotent.
    out = subprocess.run(
        [sys.executable, str(script), "--sqlite", str(src), "--dsn", _DSN,
         "--apply"],
        capture_output=True, text=True, cwd=_REPO_ROOT,
    )
    assert out.returncode == 3
    out = subprocess.run(
        [sys.executable, str(script), "--sqlite", str(src), "--dsn", _DSN,
         "--apply", "--allow-nonempty"],
        capture_output=True, text=True, cwd=_REPO_ROOT,
    )
    assert out.returncode == 0, out.stderr + out.stdout
    pdb = PgSessionDB(dsn=_DSN)
    try:
        assert len(pdb.get_messages(parent, include_inactive=True)) == src_all
    finally:
        pdb.close()


# ── Two-process concurrent smoke ───────────────────────────────────────────

_WORKER_SRC = textwrap.dedent(
    """
    import json, sys, time
    sys.path.insert(0, sys.argv[4])
    from hermes_state_pg import PgSessionDB

    dsn, name, shared_sid = sys.argv[1], sys.argv[2], sys.argv[3]
    db = PgSessionDB(dsn=dsn)
    own_sid = db.create_session(f"proc-{name}", "webui", user_id=name)
    acquired = 0
    for i in range(40):
        db.append_message(own_sid, "user", f"{name} write {i}")
        if db.try_acquire_compression_lock(shared_sid, f"holder-{name}", 0.5):
            acquired += 1
            time.sleep(0.005)
            db.release_compression_lock(shared_sid, f"holder-{name}")
    msgs = db.get_messages(own_sid)
    ok = len(msgs) == 40 and all(
        m["content"] == f"{name} write {i}" for i, m in enumerate(msgs)
    )
    print(json.dumps({"name": name, "ok": ok, "acquired": acquired}))
    db.close()
    """
)


_COLD_BOOT_WORKER_SRC = textwrap.dedent(
    """
    import json, sys, time
    sys.path.insert(0, sys.argv[3])
    from hermes_state_pg import PgSessionDB

    dsn, start_at = sys.argv[1], float(sys.argv[2])
    # Line up all workers on the same instant so their first-boot DDL
    # genuinely overlaps.
    while time.time() < start_at:
        time.sleep(0.001)
    db = PgSessionDB(dsn=dsn)
    db.ensure_session("cold-boot-probe", "webui")
    print(json.dumps({"ok": True}))
    db.close()
    """
)


def test_concurrent_cold_boot_bootstrap(tmp_path):
    """N processes constructing PgSessionDB against an EMPTY database at the
    same instant must all boot (ADR 0177 blue/green overlap, service scaling
    from zero). Guards the advisory-lock serialization in _init_pg_schema:
    Postgres CREATE TABLE IF NOT EXISTS races in pg_type when the tables
    genuinely don't exist yet, and the schema_version seed is
    check-then-insert."""
    _drop_schema()
    worker = tmp_path / "cold_boot_worker.py"
    worker.write_text(_COLD_BOOT_WORKER_SRC, encoding="utf-8")
    start_at = time.time() + 1.5
    procs = [
        subprocess.Popen(
            [sys.executable, str(worker), _DSN, str(start_at), str(_REPO_ROOT)],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
            cwd=tmp_path,
        )
        for _ in range(4)
    ]
    for p in procs:
        out, err = p.communicate(timeout=120)
        assert p.returncode == 0, err
        assert json.loads(out.strip().splitlines()[-1])["ok"], out

    with psycopg.connect(_DSN) as conn:
        versions = conn.execute(
            f"SELECT version FROM {_SCHEMA}.schema_version"
        ).fetchall()
        assert versions == [(EXPECTED_SCHEMA_VERSION,)], versions
        locks = conn.execute(
            f"SELECT * FROM {_SCHEMA}.compression_locks"
        ).fetchall()
        assert locks == [], locks


def test_two_process_concurrent_smoke(pg_db, tmp_path):
    """Two real OS processes, one database: interleaved session writes on
    distinct sessions plus lock contention on a shared session."""
    shared = pg_db.create_session("proc-shared", "webui")
    worker = tmp_path / "worker.py"
    worker.write_text(_WORKER_SRC, encoding="utf-8")
    procs = [
        subprocess.Popen(
            [sys.executable, str(worker), _DSN, name, shared, str(_REPO_ROOT)],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
            cwd=tmp_path,
        )
        for name in ("alpha", "beta")
    ]
    results = []
    for p in procs:
        out, err = p.communicate(timeout=120)
        assert p.returncode == 0, err
        results.append(json.loads(out.strip().splitlines()[-1]))

    assert all(r["ok"] for r in results), results
    # Both processes made progress through the shared lock over 40 rounds.
    assert all(r["acquired"] > 0 for r in results), results
    # No cross-contamination between the two writers' sessions.
    assert len(pg_db.get_messages("proc-alpha")) == 40
    assert len(pg_db.get_messages("proc-beta")) == 40
    assert pg_db.get_compression_lock_holder(shared) is None
