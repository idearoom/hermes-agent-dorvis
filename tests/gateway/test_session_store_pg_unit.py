"""Unit tests for hermes_state_pg that need no live Postgres.

Covers the SQLite→Postgres statement translator, the FTS5→tsquery
converter, the rebase-drift schema guard, the row shim, and the
``SessionDB.__new__`` backend dispatch. The Pg-backed integration suite
lives in ``test_session_store_pg.py`` (guarded by HERMES_STATE_TEST_DSN).
"""

import inspect
import sqlite3

import pytest

import hermes_state
import hermes_state_pg
from scripts import migrate_pg_schema_v22_to_v26 as migration_script
from hermes_state import SessionDB
from hermes_state_pg import (
    EXPECTED_SCHEMA_SURFACE_SHA256,
    EXPECTED_SCHEMA_VERSION,
    PgSessionDB,
    _Row,
    _TxnConn,
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


def test_sqlite_null_safe_parameter_comparisons_translate_to_postgres():
    sql = (
        "UPDATE sessions SET title = ? "
        "WHERE id = ? AND title IS ? AND title_source IS NOT ?"
    )
    out = _translate_sql(sql, with_params=True)
    assert "title IS NOT DISTINCT FROM %s" in out
    assert "title_source IS DISTINCT FROM %s" in out
    assert " IS %s" not in out
    assert " IS NOT %s" not in out


def test_v26_create_session_reset_marker_json_translates_to_jsonb():
    sql = """UPDATE sessions SET model_config = CASE
        WHEN excluded.model_config IS NOT NULL
             AND json_type(sessions.model_config, '$._reset_from') IS NOT NULL
             AND json_remove(sessions.model_config, '$._reset_from') = '{}'
        THEN json_set(
            excluded.model_config,
            '$._reset_from',
            json_extract(sessions.model_config, '$._reset_from')
        )
        ELSE sessions.model_config
    END"""

    out = _translate_sql(sql, with_params=False)

    assert "json_type(" not in out
    assert "json_remove(" not in out
    assert "json_set(" not in out
    assert "jsonb_set(" in out
    assert "sessions.model_config::jsonb -> '_reset_from'" in out


def test_v26_reopen_reset_marker_json_translates_to_jsonb():
    sql = (
        "UPDATE sessions AS child SET model_config = json_set("
        "COALESCE(child.model_config, '{}'), '$._reset_from', "
        "child.parent_session_id) WHERE child.parent_session_id = ?"
    )

    out = _translate_sql(sql, with_params=True)

    assert "json_set(" not in out
    assert "jsonb_set(" in out
    assert "to_jsonb(child.parent_session_id)" in out
    assert out.endswith("child.parent_session_id = %s")


def test_transaction_shim_emulates_sqlite_changes_for_v26_lineage_writes(
    monkeypatch,
):
    class _WriteResult:
        rowcount = 7

    raw = object()
    monkeypatch.setattr(
        hermes_state_pg,
        "_run_statement",
        lambda conn, sql, params: _WriteResult(),
    )
    conn = _TxnConn(raw)

    conn.execute("UPDATE sessions SET archived = 1")
    result = conn.execute("SELECT changes()")

    assert result.fetchone()[0] == 7


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
    assert EXPECTED_SCHEMA_VERSION == 26
    assert (
        EXPECTED_SCHEMA_SURFACE_SHA256
        == "cd2cb9ee351693e62e9dc8e425885a4a08148551d9577d506f4a11be4a715d5f"
    )
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
        if "COUNT(*), MIN(version)" in sql:
            if self._version is None:
                return _MarkerResult((0, None, None))
            return _MarkerResult((1, self._version, self._version))
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


def test_current_persisted_markers_are_accepted_without_mutation():
    conn = _MarkerConn(
        version=EXPECTED_SCHEMA_VERSION,
        surface=EXPECTED_SCHEMA_SURFACE_SHA256,
    )

    PgSessionDB._assert_persisted_schema_markers(
        PgSessionDB.__new__(PgSessionDB), conn
    )

    assert not conn.ran("INSERT")
    assert not conn.ran("UPDATE")


def test_v22_markers_require_the_explicit_drain_migration():
    conn = _MarkerConn(
        version=22,
        surface=hermes_state_pg._V22_SCHEMA_SURFACE_SHA256,
    )

    with pytest.raises(RuntimeError, match="migration required"):
        PgSessionDB._assert_persisted_schema_markers(
            PgSessionDB.__new__(PgSessionDB), conn
        )

    assert not conn.ran("INSERT")
    assert not conn.ran("UPDATE")
    assert not conn.ran("ALTER")


def test_missing_persisted_markers_fail_closed():
    conn = _MarkerConn(version=None, surface=None)

    with pytest.raises(RuntimeError, match="missing schema_version"):
        PgSessionDB._assert_persisted_schema_markers(
            PgSessionDB.__new__(PgSessionDB), conn
        )

    assert not conn.ran("INSERT")


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


class _RowsResult:
    def __init__(self, rows=()):
        self._rows = list(rows)
        self._offset = 0

    def fetchone(self):
        if self._offset >= len(self._rows):
            return None
        row = self._rows[self._offset]
        self._offset += 1
        return row

    def fetchall(self):
        rows = self._rows[self._offset :]
        self._offset = len(self._rows)
        return rows

    def fetchmany(self, size=1):
        rows = self._rows[self._offset : self._offset + size]
        self._offset += len(rows)
        return rows


class _Context:
    def __init__(self, value):
        self.value = value

    def __enter__(self):
        return self.value

    def __exit__(self, exc_type, exc, traceback):
        return False


class _Pool:
    def __init__(self, conn):
        self.conn = conn

    def connection(self):
        return _Context(self.conn)


class _AbsentSchemaConn:
    def __init__(self):
        self.executed = []

    def execute(self, sql, params=None):
        self.executed.append((sql, params))
        if "to_regnamespace" in sql:
            return _RowsResult([(None,)])
        if "COUNT(*), MIN(version)" in sql:
            return _RowsResult([(0, None, None)])
        return _RowsResult()

    def transaction(self):
        return _Context(self)


def test_explicit_migration_mode_refuses_an_absent_schema_before_ddl():
    conn = _AbsentSchemaConn()
    db = PgSessionDB.__new__(PgSessionDB)
    db._pool = _Pool(conn)
    db._allow_schema_migration = True
    db._storage_attestation = None

    with pytest.raises(RuntimeError, match="existing.*v22"):
        db._init_pg_schema()

    assert not any(
        sql.lstrip().startswith(("CREATE", "ALTER", "INSERT", "UPDATE", "DELETE"))
        for sql, _ in conn.executed
    )


def test_explicit_migration_mode_refuses_an_already_v26_store():
    conn = _MarkerConn(
        version=EXPECTED_SCHEMA_VERSION,
        surface=EXPECTED_SCHEMA_SURFACE_SHA256,
    )

    with pytest.raises(RuntimeError, match="requires schema_version 22"):
        PgSessionDB._assert_v22_migration_precondition(conn)

    assert not conn.ran("INSERT")
    assert not conn.ran("UPDATE")
    assert not conn.ran("ALTER")


class _CatalogConn:
    def __init__(
        self,
        *,
        omit_column=None,
        version=22,
        surface=None,
        prompt_rows=(),
        catalog_version=22,
    ):
        self.executed = []
        self.version = version
        self.surface = surface or hermes_state_pg._V22_SCHEMA_SURFACE_SHA256
        self.omit_column = omit_column
        self.prompt_rows = prompt_rows
        self.catalog_version = catalog_version

    def execute(self, sql, params=None):
        self.executed.append((sql, params))
        if "to_regnamespace" in sql:
            return _RowsResult([("hermes_state",)])
        if "to_regclass" in sql:
            return _RowsResult([("hermes_state.schema_version",)])
        if "information_schema.columns" in sql:
            columns = (
                hermes_state_pg._V22_REQUIRED_COLUMNS
                if self.catalog_version == 22
                else hermes_state_pg._REQUIRED_COLUMNS
            )
            rows = [
                (table, column)
                for table, table_columns in columns.items()
                for column in table_columns
                if (table, column) != self.omit_column
            ]
            return _RowsResult(rows)
        if "FROM pg_indexes" in sql:
            indexes = (
                hermes_state_pg._V22_REQUIRED_INDEXES
                if self.catalog_version == 22
                else hermes_state_pg._REQUIRED_INDEXES
            )
            return _RowsResult(
                (index,) for index in indexes
            )
        if "constraint_type = 'PRIMARY KEY'" in sql:
            primary_keys = (
                hermes_state_pg._V22_REQUIRED_PRIMARY_KEYS
                if self.catalog_version == 22
                else hermes_state_pg._REQUIRED_PRIMARY_KEYS
            )
            return _RowsResult(
                (table, column, ordinal)
                for table, columns in primary_keys.items()
                for ordinal, column in enumerate(columns, start=1)
            )
        if "constraint_type = 'FOREIGN KEY'" in sql:
            foreign_keys = (
                hermes_state_pg._V22_REQUIRED_FOREIGN_KEYS
                if self.catalog_version == 22
                else hermes_state_pg._REQUIRED_FOREIGN_KEYS
            )
            return _RowsResult(foreign_keys)
        if "SELECT id, system_prompt" in sql:
            return _RowsResult(self.prompt_rows)
        if "COUNT(*), MIN(version)" in sql:
            return _RowsResult([(1, self.version, self.version)])
        if "SELECT version" in sql:
            return _RowsResult([(self.version,)])
        if "SELECT value" in sql:
            return _RowsResult([(self.surface,)])
        if sql.strip().startswith("UPDATE hermes_state.schema_version"):
            self.version = int(params[0])
        if sql.strip().startswith("UPDATE hermes_state.state_meta"):
            self.surface = str(params[0])
        if sql.lstrip().startswith(("CREATE", "ALTER")):
            self.catalog_version = EXPECTED_SCHEMA_VERSION
        return _RowsResult()

    def transaction(self):
        return _Context(self)


def test_catalog_verification_fails_when_a_v26_column_is_missing():
    conn = _CatalogConn(
        omit_column=("messages", "display_metadata"), catalog_version=26
    )

    with pytest.raises(RuntimeError, match="display_metadata"):
        PgSessionDB._verify_catalog(conn)


def test_ordinary_boot_refuses_v22_without_running_migration_ddl():
    conn = _CatalogConn()
    db = PgSessionDB.__new__(PgSessionDB)
    db._pool = _Pool(conn)
    db._allow_schema_migration = False
    db._storage_attestation = None

    with pytest.raises(RuntimeError, match="migration required"):
        db._init_pg_schema()

    assert not any(
        sql.lstrip().startswith(("CREATE", "ALTER", "INSERT", "UPDATE", "DELETE"))
        for sql, _ in conn.executed
    )
    assert db._storage_attestation is None


def test_explicit_v22_preflight_accepts_only_the_exact_catalog():
    conn = _CatalogConn()

    PgSessionDB._assert_v22_migration_precondition(conn)

    assert not any(
        sql.lstrip().startswith(("CREATE", "ALTER", "INSERT", "UPDATE", "DELETE"))
        for sql, _ in conn.executed
    )


def test_explicit_v22_preflight_rejects_a_partly_widened_catalog():
    conn = _CatalogConn(catalog_version=26)

    with pytest.raises(RuntimeError, match="unexpected"):
        PgSessionDB._assert_v22_migration_precondition(conn)

    assert not any(
        sql.lstrip().startswith(("CREATE", "ALTER", "INSERT", "UPDATE", "DELETE"))
        for sql, _ in conn.executed
    )


def test_migration_script_dry_run_observes_v22_without_exposing_dsn(
    monkeypatch, capsys
):
    observed = []

    def _inspect(dsn):
        observed.append(dsn)
        return {
            "backend": "postgres",
            "schema_version": 22,
            "surface_marker": hermes_state_pg._V22_SCHEMA_SURFACE_SHA256,
        }

    monkeypatch.setattr(
        migration_script, "inspect_v22_migration_precondition", _inspect
    )
    secret_dsn = "postgresql://operator:do-not-print@example.invalid/db"

    assert migration_script.main(["--dsn", secret_dsn]) == 0

    output = capsys.readouterr()
    assert observed == [secret_dsn]
    assert "schema_version=22" in output.out
    assert hermes_state_pg._V22_SCHEMA_SURFACE_SHA256 in output.out
    assert secret_dsn not in output.out
    assert secret_dsn not in output.err


def test_explicit_v22_migration_repairs_titles_before_unique_index():
    conn = _CatalogConn()
    db = PgSessionDB.__new__(PgSessionDB)

    db._migrate_v22_to_v26(conn)

    statements = [sql for sql, _ in conn.executed]
    dedupe_at = statements.index(hermes_state_pg.PG_V26_DEDUPLICATE_TITLES_SQL)
    title_index_at = next(
        i for i, sql in enumerate(statements) if "idx_sessions_title_unique" in sql
    )
    assert dedupe_at < title_index_at
    assert "ORDER BY started_at DESC, id DESC" in (
        " ".join(hermes_state_pg.PG_V26_DEDUPLICATE_TITLES_SQL.split())
    )
    assert conn.version == EXPECTED_SCHEMA_VERSION
    assert conn.surface == EXPECTED_SCHEMA_SURFACE_SHA256


def test_explicit_v22_migration_content_addresses_legacy_prompts():
    conn = _CatalogConn(prompt_rows=(("session-1", "system prompt"),))
    db = PgSessionDB.__new__(PgSessionDB)

    db._migrate_v22_to_v26(conn)

    prompt_insert = next(
        params
        for sql, params in conn.executed
        if sql.startswith("INSERT INTO hermes_state.system_prompts")
    )
    assert prompt_insert == (
        "e16202309c92180728dd7fd1c59f16004a6d5ee245538c28d2a9a22edf2dd2ab",
        "system prompt",
    )
    assert any(
        "SET system_prompt_hash = %s, system_prompt = NULL" in sql
        and params == (prompt_insert[0], "session-1")
        for sql, params in conn.executed
    )


def test_explicit_v22_migration_rejects_an_unknown_surface_before_ddl():
    conn = _CatalogConn(surface="unknown")
    db = PgSessionDB.__new__(PgSessionDB)

    with pytest.raises(RuntimeError, match="Refusing v22→v26 migration"):
        db._migrate_v22_to_v26(conn)

    assert not any(
        sql.lstrip().startswith(("CREATE", "ALTER", "UPDATE"))
        for sql, _ in conn.executed
    )


# ── v26 schema mirror and backend seams ────────────────────────────────────


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
def test_pg_schema_mirrors_the_v26_tables(fragment):
    assert any(fragment in stmt for stmt in hermes_state_pg.PG_SCHEMA_SQL)


@pytest.mark.parametrize(
    "fragment",
    [
        "hermes_state.system_prompts",
        "hermes_state.gateway_hygiene_state",
        "hermes_state.session_turn_leases",
        "idx_session_turn_leases_expires",
        "idx_sessions_system_prompt_hash",
        "idx_sessions_title_unique",
        "idx_messages_session_id",
        "idx_messages_assistant_calls_by_session",
    ],
)
def test_pg_schema_contains_every_v26_table_and_index(fragment):
    assert any(fragment in stmt for stmt in hermes_state_pg.PG_SCHEMA_SQL)


@pytest.mark.parametrize(
    "column",
    [
        "compression_fallback_streak",
        "profile_name",
        "effect_disposition",
        "api_content",
        "system_prompt_hash",
        "git_metadata_generation",
        "title_source",
        "last_activity_at",
        "last_activity_description",
        "last_activity_provenance",
        "compression_ineffective_count",
        "pinned",
        "hidden",
        "last_read_at",
        "display_kind",
        "display_metadata",
    ],
)
def test_v26_columns_are_both_created_and_expanded(column):
    """Fresh databases get the column from CREATE TABLE; existing ones from
    the additive ALTER. Missing either one silently drops writes."""
    assert any(column in stmt for stmt in hermes_state_pg.PG_SCHEMA_SQL)
    assert any(
        column in stmt and "ADD COLUMN IF NOT EXISTS" in stmt
        for stmt in hermes_state_pg.PG_EXPAND_SQL
    )


def test_pg_backend_overrides_new_transaction_and_search_seams():
    signature = inspect.signature(PgSessionDB._execute_write)
    assert "patience_s" in signature.parameters
    assert "_read_ctx" in PgSessionDB.__dict__
    assert "_search_messages_impl" in PgSessionDB.__dict__
    assert "search_messages" not in PgSessionDB.__dict__
    assert "_message_column_names" in PgSessionDB.__dict__


def test_pg_never_reclaims_a_foreign_holder_from_local_pid_liveness():
    db = PgSessionDB.__new__(PgSessionDB)
    assert db._lock_holder_process_is_dead("12345:foreign-task") is False


def test_pg_attestation_is_exact_and_does_not_expose_the_dsn():
    db = PgSessionDB.__new__(PgSessionDB)
    db._storage_attestation = {
        "backend": "postgres",
        "schema_version": EXPECTED_SCHEMA_VERSION,
        "surface_marker": EXPECTED_SCHEMA_SURFACE_SHA256,
    }

    assert db.storage_attestation() == {
        "backend": "postgres",
        "schema_version": 26,
        "surface_marker": EXPECTED_SCHEMA_SURFACE_SHA256,
    }


def test_pg_close_drains_token_writer_before_closing_pool(monkeypatch):
    events = []

    class _ClosePool:
        def close(self):
            events.append("pool-close")

    hook = object()
    db = PgSessionDB.__new__(PgSessionDB)
    db._closed = False
    db._conn = object()
    db._pool = _ClosePool()
    db._token_atexit_hook = hook
    db._stop_token_writer = lambda: events.append("token-drain")
    monkeypatch.setattr(
        hermes_state.atexit,
        "unregister",
        lambda registered: events.append(("unregister", registered)),
    )

    db.close()

    assert events == ["token-drain", ("unregister", hook), "pool-close"]
    assert db._conn is None
    assert db._closed is True


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


def test_explicit_column_expansion_contains_no_destructive_ddl():
    """The drained transaction widens columns without dropping old data."""
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
