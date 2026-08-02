"""Unit tests for hermes_state_pg that need no live Postgres.

Covers the SQLite→Postgres statement translator, the FTS5→tsquery
converter, the rebase-drift schema guard, the row shim, and the
``SessionDB.__new__`` backend dispatch. The Pg-backed integration suite
lives in ``test_session_store_pg.py`` (guarded by HERMES_STATE_TEST_DSN).
"""

import sqlite3

import pytest

import hermes_state
import hermes_state_pg
from hermes_state import SessionDB
from hermes_state_pg import (
    EXPECTED_SCHEMA_SURFACE_SHA256,
    EXPECTED_SCHEMA_VERSION,
    PgSessionDB,
    _Row,
    _translate_sql,
    assert_schema_compat,
    schema_surface_hash,
)


# ── Statement translator ────────────────────────────────────────────────────


def test_qmark_placeholders_become_pyformat():
    assert (
        _translate_sql("SELECT * FROM sessions WHERE id = ?", with_params=True)
        == "SELECT * FROM sessions WHERE id = %s"
    )


def test_question_mark_inside_literal_is_preserved():
    sql = "SELECT '?' AS q, id FROM sessions WHERE id = ?"
    out = _translate_sql(sql, with_params=True)
    assert out == "SELECT '?' AS q, id FROM sessions WHERE id = %s"


def test_percent_escaping_only_with_params():
    sql = "SELECT id FROM sessions WHERE title LIKE 'a%' AND id = ?"
    out = _translate_sql(sql, with_params=True)
    assert "'a%%'" in out and out.endswith("%s")
    # Without params psycopg does no format pass, so % must survive as-is.
    sql2 = "SELECT id FROM sessions WHERE title LIKE 'a%'"
    assert "'a%'" in _translate_sql(sql2, with_params=False)


def test_like_becomes_ilike_outside_literals():
    sql = "SELECT 1 WHERE title LIKE ? ESCAPE '\\' AND note = 'I LIKE cats'"
    out = _translate_sql(sql, with_params=True)
    assert "title ILIKE %s" in out
    assert "'I LIKE cats'" in out  # literal untouched


def test_insert_or_ignore_rewrites_to_on_conflict():
    sql = "INSERT OR IGNORE INTO compression_locks (a, b) VALUES (?, ?)"
    out = _translate_sql(sql, with_params=True)
    assert out.startswith("INSERT INTO compression_locks")
    assert out.endswith(" ON CONFLICT DO NOTHING")


def test_json_extract_rewrites_to_jsonb_arrow():
    sql = (
        "SELECT 1 FROM sessions WHERE "
        "json_extract(COALESCE(model_config, '{}'), '$._delegate_from') IS NULL"
    )
    out = _translate_sql(sql, with_params=False)
    assert "(COALESCE(model_config, '{}')::jsonb ->> '_delegate_from')" in out


def test_hex_blob_literals_become_escape_strings():
    sql = "SELECT REPLACE(REPLACE(content, X'0A', ' '), X'0D', ' ') FROM messages"
    out = _translate_sql(sql, with_params=False)
    assert "E'\\x0A'" in out and "E'\\x0D'" in out


# ── FTS5 → tsquery ─────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "fts5, expected",
    [
        ("hello", "'hello'"),
        ("hello world", "'hello' & 'world'"),
        ("deploy*", "'deploy':*"),
        ("docker OR kubernetes", "'docker' | 'kubernetes'"),
        ("python NOT java", "'python' & ! 'java'"),
        ('"exact phrase"', "('exact' <-> 'phrase')"),
        ('"chat-send"', "('chat' <-> 'send')"),
    ],
)
def test_fts5_to_tsquery(fts5, expected):
    assert PgSessionDB._fts5_to_tsquery(fts5) == expected


# ── Rebase-drift guard ─────────────────────────────────────────────────────


def test_schema_compat_passes_on_current_tree():
    assert hermes_state.SCHEMA_VERSION == EXPECTED_SCHEMA_VERSION
    assert schema_surface_hash() == EXPECTED_SCHEMA_SURFACE_SHA256
    assert_schema_compat()  # must not raise


def test_schema_compat_fails_loudly_on_version_bump(monkeypatch):
    monkeypatch.setattr(hermes_state, "SCHEMA_VERSION", EXPECTED_SCHEMA_VERSION + 1)
    with pytest.raises(RuntimeError, match="rebase-drift"):
        assert_schema_compat()


def test_schema_compat_fails_loudly_on_ddl_edit(monkeypatch):
    monkeypatch.setattr(
        hermes_state,
        "SCHEMA_SQL",
        hermes_state.SCHEMA_SQL + "\n-- upstream added a table\n",
    )
    with pytest.raises(RuntimeError, match="rebase-drift"):
        assert_schema_compat()


def test_pg_construction_refuses_to_boot_on_drift(monkeypatch):
    """The guard fires before any connection is attempted."""
    monkeypatch.setattr(hermes_state, "SCHEMA_VERSION", EXPECTED_SCHEMA_VERSION + 1)
    with pytest.raises(RuntimeError, match="rebase-drift"):
        PgSessionDB(dsn="postgresql://invalid.invalid/nope")


class _MarkerResult:
    def __init__(self, row):
        self._row = row

    def fetchone(self):
        return self._row


class _MarkerConn:
    """Records statements and answers the two persisted-marker SELECTs."""

    def __init__(self, version, surface):
        self.executed = []
        self._version = version
        self._surface = surface

    def execute(self, sql, params=None):
        self.executed.append((sql, params))
        if "SELECT version" in sql:
            return _MarkerResult(
                None if self._version is None else (self._version,)
            )
        if "SELECT value" in sql:
            return _MarkerResult(
                None if self._surface is None else (self._surface,)
            )
        return _MarkerResult(None)

    def ran(self, prefix):
        return [
            (sql, params)
            for sql, params in self.executed
            if sql.strip().startswith(prefix)
        ]


def test_persisted_markers_advance_from_the_audited_v19_predecessor():
    """AE-182: the v19→v22 delta is additive, so the store expands in place.

    ``_init_pg_schema`` applies PG_SCHEMA_SQL + PG_EXPAND_SQL before the marker
    check, so by the time this runs the database already has the v22 shape.
    The markers are then advanced rather than held back: no build between this
    one and the v19 predecessor can use the Postgres store at all (they trip
    ``assert_schema_compat`` and fall back to local SQLite), so a held-back
    marker would protect nothing and would mask genuine future drift.
    """
    previous_hash = hermes_state_pg._V19_SURFACE_SHA256_PRE_ACTIVE_NULL_INDEX
    conn = _MarkerConn(version=19, surface=previous_hash)

    PgSessionDB._assert_persisted_schema_markers(
        PgSessionDB.__new__(PgSessionDB), conn
    )

    # v20 analytics backfill ran, then the version marker advanced.
    assert conn.ran("INSERT INTO hermes_state.session_model_usage")
    assert conn.ran("UPDATE hermes_state.schema_version SET version") == [
        (
            "UPDATE hermes_state.schema_version SET version = %s",
            (EXPECTED_SCHEMA_VERSION,),
        )
    ]
    assert conn.ran("UPDATE hermes_state.state_meta SET value") == [
        (
            "UPDATE hermes_state.state_meta SET value = %s WHERE key = %s",
            (EXPECTED_SCHEMA_SURFACE_SHA256, "pg_backend_schema_surface_sha256"),
        )
    ]


def test_both_audited_v19_surface_markers_are_accepted():
    for previous_hash in (
        hermes_state_pg._V19_SURFACE_SHA256_PRE_ACTIVE_NULL_INDEX,
        hermes_state_pg._V19_SURFACE_SHA256,
    ):
        conn = _MarkerConn(version=19, surface=previous_hash)
        PgSessionDB._assert_persisted_schema_markers(
            PgSessionDB.__new__(PgSessionDB), conn
        )
        assert conn.ran("UPDATE hermes_state.state_meta SET value")


def test_fresh_store_persists_the_current_schema_markers():
    conn = _MarkerConn(version=None, surface=None)

    PgSessionDB._assert_persisted_schema_markers(
        PgSessionDB.__new__(PgSessionDB), conn
    )

    assert (
        "INSERT INTO hermes_state.schema_version (version) VALUES (%s)",
        (EXPECTED_SCHEMA_VERSION,),
    ) in conn.executed
    assert any(
        sql.startswith("INSERT INTO hermes_state.state_meta")
        and params
        == ("pg_backend_schema_surface_sha256", EXPECTED_SCHEMA_SURFACE_SHA256)
        for sql, params in conn.executed
    )
    # A fresh store is already at the terminal shape — no data migration.
    assert not conn.ran("INSERT INTO hermes_state.session_model_usage")


def test_unknown_persisted_schema_version_fails_closed():
    conn = _MarkerConn(version=EXPECTED_SCHEMA_VERSION + 1, surface=None)
    with pytest.raises(RuntimeError, match="schema_version"):
        PgSessionDB._assert_persisted_schema_markers(
            PgSessionDB.__new__(PgSessionDB), conn
        )


def test_unknown_persisted_schema_surface_fails_closed():
    conn = _MarkerConn(version=EXPECTED_SCHEMA_VERSION, surface="deadbeef")
    with pytest.raises(RuntimeError, match="different"):
        PgSessionDB._assert_persisted_schema_markers(
            PgSessionDB.__new__(PgSessionDB), conn
        )


def test_v20_backfill_failure_does_not_block_boot(monkeypatch):
    """Analytics backfill is best-effort — falling back to SQLite is worse."""

    class _AngryConn(_MarkerConn):
        def execute(self, sql, params=None):
            if "session_model_usage" in sql:
                self.executed.append((sql, params))
                raise RuntimeError("relation is being rewritten")
            return super().execute(sql, params)

    conn = _AngryConn(
        version=19, surface=hermes_state_pg._V19_SURFACE_SHA256
    )
    PgSessionDB._assert_persisted_schema_markers(
        PgSessionDB.__new__(PgSessionDB), conn
    )
    assert conn.ran("UPDATE hermes_state.schema_version SET version")


# ── v19 → v22 schema mirror ────────────────────────────────────────────────


@pytest.mark.parametrize(
    "fragment",
    [
        # v20/v22 per-model, per-task usage attribution.
        "hermes_state.session_model_usage",
        "PRIMARY KEY (session_id, model, billing_provider, billing_base_url,\n"
        "                     billing_mode, task)",
        # v21 durable async delegations (surface parity).
        "hermes_state.async_delegations",
    ],
)
def test_pg_schema_mirrors_the_v22_tables(fragment):
    assert any(fragment in stmt for stmt in hermes_state_pg.PG_SCHEMA_SQL)


@pytest.mark.parametrize(
    "column",
    [
        "compression_fallback_streak",
        "profile_name",
        "effect_disposition",
        "api_content",
    ],
)
def test_v22_columns_are_both_created_and_expanded(column):
    """Fresh databases get the column from CREATE TABLE; existing ones from
    the additive ALTER. Missing either one silently drops writes."""
    assert any(column in stmt for stmt in hermes_state_pg.PG_SCHEMA_SQL)
    assert any(
        column in stmt and "ADD COLUMN IF NOT EXISTS" in stmt
        for stmt in hermes_state_pg.PG_EXPAND_SQL
    )


def test_statement_override_keys_still_exist_upstream():
    """A rebase that edits an overridden statement must fail HERE, not in prod.

    ``STATEMENT_OVERRIDES`` is keyed by the exact upstream statement text. If
    upstream edits one, the key silently stops matching and the untranslated
    SQLite form reaches Postgres — where it raises at write time, on the
    session-store hot path. Anchoring the keys to hermes_state.py's source
    turns that into a unit-test failure during the rebase.
    """
    import inspect

    source = hermes_state_pg._normalize_statement(inspect.getsource(hermes_state))
    assert hermes_state_pg.STATEMENT_OVERRIDES
    for key in hermes_state_pg.STATEMENT_OVERRIDES:
        assert key in source, (
            "an overridden statement no longer matches hermes_state.py — "
            "re-audit hermes_state_pg.STATEMENT_OVERRIDES"
        )


def test_statement_override_is_applied_by_the_translator():
    sqlite_sql = hermes_state_pg._USAGE_UPSERT_SQLITE
    out = _translate_sql(sqlite_sql, with_params=True)
    # Self-references are table-qualified; excluded.* references are not.
    assert "api_call_count = session_model_usage.api_call_count + excluded." in out
    assert (
        "cost_status = COALESCE(excluded.cost_status, "
        "session_model_usage.cost_status)" in out
    )
    assert "?" not in out


def test_expand_migration_is_additive_only():
    """ADR 0177 coexistence: a draining old task must keep working."""
    for stmt in hermes_state_pg.PG_EXPAND_SQL:
        assert "ADD COLUMN IF NOT EXISTS" in stmt
        for destructive in (" DROP ", " RENAME ", " ALTER COLUMN "):
            assert destructive not in f" {stmt} "


# ── Row shim ───────────────────────────────────────────────────────────────


def test_row_supports_index_name_and_dict():
    row = _Row(["id", "title"], ("s1", "hello"))
    assert row[0] == "s1"
    assert row["title"] == "hello"
    assert dict(row) == {"id": "s1", "title": "hello"}
    assert list(row) == ["s1", "hello"]
    assert len(row) == 2
    assert not isinstance(row, sqlite3.Row)


# ── Backend dispatch (SessionDB.__new__) ───────────────────────────────────


def test_sessiondb_without_env_is_sqlite(tmp_path, monkeypatch):
    monkeypatch.delenv("HERMES_STATE_STORE_DSN", raising=False)
    db = SessionDB(tmp_path / "state.db")
    try:
        assert type(db) is SessionDB
        assert (tmp_path / "state.db").exists()
    finally:
        db.close()


def test_sessiondb_with_env_dispatches_to_pg(monkeypatch):
    class _FakePg(SessionDB):
        def __init__(self, db_path=None, read_only=False, **kwargs):
            self.init_args = (db_path, read_only)

    monkeypatch.setenv("HERMES_STATE_STORE_DSN", "postgresql://x/y")
    monkeypatch.setattr(hermes_state_pg, "PgSessionDB", _FakePg)
    db = SessionDB()
    assert type(db) is _FakePg
    assert db.init_args == (None, False)


def test_explicit_db_path_never_redirects(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_STATE_STORE_DSN", "postgresql://x/y")
    db = SessionDB(tmp_path / "explicit.db")
    try:
        assert type(db) is SessionDB
    finally:
        db.close()


# ── Per-profile routing that must NOT defeat dispatch (AE-182) ─────────────


def test_open_session_db_for_default_home_dispatches_to_pg(monkeypatch):
    """The regression that sent production chat writes to the EFS file.

    ``SessionDB(db_path=home / "state.db")`` for the DEFAULT home bypasses the
    HERMES_STATE_STORE_DSN dispatch entirely; the helper must not.
    """

    class _FakePg(SessionDB):
        def __init__(self, db_path=None, read_only=False, **kwargs):
            self.init_args = (db_path, read_only)

    monkeypatch.setenv("HERMES_STATE_STORE_DSN", "postgresql://x/y")
    monkeypatch.setattr(hermes_state_pg, "PgSessionDB", _FakePg)

    db = hermes_state.open_session_db_for_home(hermes_state.DEFAULT_DB_PATH.parent)
    assert type(db) is _FakePg
    assert db.init_args == (None, False)


def test_open_session_db_for_other_profile_home_stays_on_its_file(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("HERMES_STATE_STORE_DSN", "postgresql://x/y")
    db = hermes_state.open_session_db_for_home(tmp_path)
    try:
        assert type(db) is SessionDB
        assert (tmp_path / "state.db").exists()
    finally:
        db.close()


def test_api_server_opens_the_default_profile_through_dispatch(monkeypatch):
    """Guards the exact call site upstream 7aa21e336 regressed."""
    from gateway.platforms.api_server import APIServerAdapter

    class _FakePg(SessionDB):
        def __init__(self, db_path=None, read_only=False, **kwargs):
            self.init_args = (db_path, read_only)

    monkeypatch.setenv("HERMES_STATE_STORE_DSN", "postgresql://x/y")
    monkeypatch.setattr(hermes_state_pg, "PgSessionDB", _FakePg)

    server = APIServerAdapter.__new__(APIServerAdapter)
    db = APIServerAdapter._open_and_cache_session_db(
        server, hermes_state.DEFAULT_DB_PATH.parent
    )
    assert type(db) is _FakePg
