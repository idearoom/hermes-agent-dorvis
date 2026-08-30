"""Unit tests for hermes_state_pg that need no live Postgres.

Covers the SQLite→Postgres statement translator, the FTS5→tsquery
converter, the rebase-drift schema guard, the row shim, and the
``SessionDB.__new__`` backend dispatch. The Pg-backed integration suite
lives in ``test_session_store_pg.py`` (guarded by HERMES_STATE_TEST_DSN).
"""

import hashlib
import inspect
from pathlib import Path
import sqlite3
import threading

import pytest

import hermes_state
import hermes_state_pg
from scripts import migrate_pg_schema_v22_to_v26 as migration_script
from scripts import migrate_pg_schema_v26_surface as surface_migration_script
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

_REPO_ROOT = Path(__file__).resolve().parents[2]
_V22_SCHEMA_FIXTURE = _REPO_ROOT / "tests" / "fixtures" / "hermes_state_pg_v22.sql"
_V22_SCHEMA_FIXTURE_SHA256 = (
    "f6ab4089f35bfc5d022cc967339d6cd3d831c5591d7f669d53eb46ee17289894"
)


def test_frozen_v22_schema_fixture_is_pinned():
    fixture = _V22_SCHEMA_FIXTURE.read_bytes()

    assert hashlib.sha256(fixture).hexdigest() == _V22_SCHEMA_FIXTURE_SHA256
    assert fixture.count(b"-- hermes-v22-statement\n") == 29
    assert b"2c3c480648fc0455c2ef6948f5d5451e38f83c12" in fixture


# ── Statement translator ────────────────────────────────────────────────────


def test_qmark_placeholders_become_pyformat():
    assert (
        _translate_sql("SELECT * FROM sessions WHERE id = ?", with_params=True)
        == "SELECT * FROM sessions WHERE id = %s"
    )


def test_offset_only_limit_sentinel_becomes_postgres_unbounded_limit():
    assert _translate_sql(
        "SELECT * FROM messages LIMIT ? OFFSET ?",
        with_params=True,
    ) == "SELECT * FROM messages LIMIT NULLIF(%s, -1) OFFSET %s"


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
# ── Null-safe IS ? / IS NOT ? → IS [NOT] DISTINCT FROM %s ──────────────────


def test_is_placeholder_becomes_is_not_distinct_from():
    out = _translate_sql(
        "SELECT 1 FROM messages WHERE content IS ?", with_params=True
    )
    assert out == "SELECT 1 FROM messages WHERE content IS NOT DISTINCT FROM %s"


def test_multiple_is_placeholders_all_rewritten():
    out = _translate_sql(
        "DELETE FROM t WHERE type IS ? AND name IS ? AND rowid <> ?",
        with_params=True,
    )
    assert out == (
        "DELETE FROM t WHERE type IS NOT DISTINCT FROM %s "
        "AND name IS NOT DISTINCT FROM %s AND rowid <> %s"
    )


def test_is_not_placeholder_becomes_is_distinct_from():
    out = _translate_sql(
        "SELECT 1 FROM messages WHERE content IS NOT ?", with_params=True
    )
    assert out == "SELECT 1 FROM messages WHERE content IS DISTINCT FROM %s"


def test_is_null_and_is_not_null_untouched():
    sql = "SELECT 1 FROM t WHERE a IS NULL AND b IS NOT NULL AND c IS ?"
    out = _translate_sql(sql, with_params=True)
    assert out == (
        "SELECT 1 FROM t WHERE a IS NULL AND b IS NOT NULL "
        "AND c IS NOT DISTINCT FROM %s"
    )


def test_is_placeholder_inside_literal_untouched():
    sql = "SELECT 'x IS ?' AS note FROM t WHERE content IS ?"
    out = _translate_sql(sql, with_params=True)
    assert out == (
        "SELECT 'x IS ?' AS note FROM t WHERE content IS NOT DISTINCT FROM %s"
    )


def test_is_placeholder_composes_with_like_and_qmark_rewrites():
    sql = (
        "SELECT 1 FROM t WHERE title LIKE ? AND content IS ? "
        "AND note = 'a%' AND id = ?"
    )
    out = _translate_sql(sql, with_params=True)
    assert out == (
        "SELECT 1 FROM t WHERE title ILIKE %s "
        "AND content IS NOT DISTINCT FROM %s "
        "AND note = 'a%%' AND id = %s"
    )


def test_is_placeholder_untouched_without_params():
    # Mirrors the ``?`` behavior: no params → psycopg does no format pass,
    # and a bare ``?`` would not be a placeholder anyway.
    sql = "SELECT 1 FROM t WHERE content IS ?"
    assert _translate_sql(sql, with_params=False) == sql


def test_identifier_ending_in_is_not_mangled():
    out = _translate_sql(
        "SELECT 1 FROM t WHERE analysis = ? AND thesis IS ?", with_params=True
    )
    assert out == (
        "SELECT 1 FROM t WHERE analysis = %s AND thesis IS NOT DISTINCT FROM %s"
    )


def test_api_content_backfill_statement_translates_to_valid_pg():
    # The exact statement ``SessionDB.set_latest_user_api_content`` executes
    # (hermes_state.py). Before the IS-rewrite this translated to
    # ``... AND content IS %s`` — a Postgres syntax error — which made the
    # in-place compaction api_content backfill fail on every PG turn
    # ("in-place compaction api_content backfill failed", agent/turn_context.py).
    sqlite_sql = (
        "UPDATE messages SET api_content = ? WHERE id = ("
        "SELECT id FROM messages "
        "WHERE session_id = ? AND role = 'user' AND active = 1 "
        "ORDER BY id DESC LIMIT 1"
        ") AND content IS ?"
    )
    out = _translate_sql(sqlite_sql, with_params=True)
    assert out == (
        "UPDATE messages SET api_content = %s WHERE id = ("
        "SELECT id FROM messages "
        "WHERE session_id = %s AND role = 'user' AND active = 1 "
        "ORDER BY id DESC LIMIT 1"
        ") AND content IS NOT DISTINCT FROM %s"
    )
    assert "IS %s" not in out


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


def test_target_runtime_rejects_the_pre_summary_surface_marker():
    conn = _MarkerConn(
        version=EXPECTED_SCHEMA_VERSION,
        surface=hermes_state_pg._V26_PRE_SUMMARY_SCHEMA_SURFACE_SHA256,
    )

    with pytest.raises(RuntimeError, match="surface migration required"):
        PgSessionDB._assert_persisted_schema_markers(
            PgSessionDB.__new__(PgSessionDB), conn
        )

    assert not conn.ran("INSERT")
    assert not conn.ran("UPDATE")
    assert not conn.ran("ALTER")


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
        wrong_index=None,
        wrong_column=None,
        extra_constraint=None,
        wrong_constraint=None,
        include_optional_index=False,
        extra_index=None,
        pg18_not_null_constraints=False,
        pre_summary_surface=False,
    ):
        self.executed = []
        self.version = version
        self.surface = surface or hermes_state_pg._V22_SCHEMA_SURFACE_SHA256
        self.omit_column = omit_column
        self.prompt_rows = prompt_rows
        self.catalog_version = catalog_version
        self.wrong_index = wrong_index
        self.wrong_column = wrong_column
        self.extra_constraint = extra_constraint
        self.wrong_constraint = wrong_constraint
        self.include_optional_index = include_optional_index
        self.extra_index = extra_index
        self.pg18_not_null_constraints = pg18_not_null_constraints
        self.pre_summary_surface = pre_summary_surface

    def execute(self, sql, params=None):
        self.executed.append((sql, params))
        if "to_regnamespace" in sql:
            return _RowsResult([("hermes_state",)])
        if "to_regclass" in sql:
            return _RowsResult([("hermes_state.schema_version",)])
        if "information_schema.columns" in sql:
            columns = (
                hermes_state_pg._V22_REQUIRED_COLUMN_SPECS
                if self.catalog_version == 22
                else (
                    hermes_state_pg._V26_PRE_SUMMARY_REQUIRED_COLUMN_SPECS
                    if self.pre_summary_surface
                    else hermes_state_pg._REQUIRED_COLUMN_SPECS
                )
            )
            rows = []
            for (table, column), expected in columns.items():
                if (table, column) == self.omit_column:
                    continue
                values = {
                    "data_type": expected.data_type,
                    "nullable": expected.nullable,
                    "default": expected.default,
                    "identity_generation": expected.identity_generation,
                }
                if self.wrong_column and self.wrong_column[:2] == (table, column):
                    values[self.wrong_column[2]] = self.wrong_column[3]
                actual = hermes_state_pg._ColumnSpec(**values)
                rows.append(
                    (
                        table,
                        column,
                        actual.data_type,
                        "YES" if actual.nullable else "NO",
                        actual.default,
                        "YES" if actual.identity_generation is not None else "NO",
                        actual.identity_generation,
                    )
                )
            return _RowsResult(rows)
        if "FROM pg_index AS index_catalog" in sql:
            indexes = (
                dict(hermes_state_pg._V22_REQUIRED_INDEX_SPECS)
                if self.catalog_version == 22
                else dict(hermes_state_pg._REQUIRED_INDEX_SPECS)
            )
            if self.include_optional_index:
                indexes.update(hermes_state_pg._OPTIONAL_INDEX_SPECS)
            if self.extra_index is not None:
                indexes[self.extra_index[0]] = self.extra_index[1]
            rows = []
            for name, expected in indexes.items():
                actual = expected
                if name == self.wrong_index:
                    actual = hermes_state_pg._index_spec(
                        "sessions",
                        "started_at",
                        opclasses=("float8_ops",),
                    )
                rows.append(
                    (
                        actual.table,
                        name,
                        actual.unique,
                        True,
                        True,
                        True,
                        actual.access_method,
                        len(actual.keys),
                        len(actual.keys),
                        actual.keys,
                        actual.options,
                        actual.opclasses,
                        actual.predicate,
                        actual.expression,
                    )
                )
            return _RowsResult(rows)
        if "FROM pg_constraint AS constraint_catalog" in sql:
            constraints = (
                hermes_state_pg._V22_REQUIRED_CONSTRAINT_SPECS
                if self.catalog_version == 22
                else hermes_state_pg._REQUIRED_CONSTRAINT_SPECS
            )
            rows = []
            for (table, name), expected in constraints.items():
                values = {
                    field: getattr(expected, field)
                    for field in expected.__dataclass_fields__
                }
                if self.wrong_constraint and self.wrong_constraint[0] == (
                    table,
                    name,
                ):
                    values[self.wrong_constraint[1]] = self.wrong_constraint[2]
                spec = hermes_state_pg._ConstraintSpec(**values)
                rows.append(
                    (
                        table,
                        name,
                        spec.kind,
                        spec.deferrable,
                        spec.initially_deferred,
                        spec.validated,
                        spec.columns,
                        spec.referenced_schema,
                        spec.referenced_table,
                        spec.referenced_columns,
                        spec.update_action or " ",
                        spec.delete_action or " ",
                        spec.match_type or " ",
                    )
                )
            if self.extra_constraint is not None:
                rows.append(self.extra_constraint)
            if (
                self.pg18_not_null_constraints
                and "constraint_catalog.contype in ('p', 'f', 'u', 'c', 'x')"
                not in " ".join(sql.lower().split())
            ):
                rows.append(
                    (
                        "sessions",
                        "sessions_id_not_null",
                        "n",
                        False,
                        False,
                        True,
                        ("id",),
                        None,
                        None,
                        (),
                        " ",
                        " ",
                        " ",
                    )
                )
            return _RowsResult(rows)
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
            if len(params) == 3 and self.surface != str(params[2]):
                return _RowsResult()
            self.surface = str(params[0])
            return _RowsResult([(self.surface,)])
        if sql == hermes_state_pg.PG_V26_SUMMARY_SURFACE_SQL:
            self.pre_summary_surface = False
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


def test_target_catalog_requires_the_compressed_summary_column():
    conn = _CatalogConn(
        version=EXPECTED_SCHEMA_VERSION,
        surface=hermes_state_pg._V26_PRE_SUMMARY_SCHEMA_SURFACE_SHA256,
        catalog_version=EXPECTED_SCHEMA_VERSION,
        pre_summary_surface=True,
    )

    with pytest.raises(RuntimeError, match="_compressed_summary"):
        PgSessionDB._verify_catalog(conn)


def test_current_marker_without_compressed_summary_column_fails_closed():
    conn = _CatalogConn(
        version=EXPECTED_SCHEMA_VERSION,
        surface=EXPECTED_SCHEMA_SURFACE_SHA256,
        catalog_version=EXPECTED_SCHEMA_VERSION,
        pre_summary_surface=True,
    )

    with pytest.raises(RuntimeError, match="_compressed_summary"):
        PgSessionDB._verify_catalog(conn)


def test_catalog_verification_rejects_same_named_wrong_unique_index():
    conn = _CatalogConn(
        catalog_version=26,
        wrong_index="idx_sessions_title_unique",
    )

    with pytest.raises(RuntimeError, match="idx_sessions_title_unique"):
        PgSessionDB._verify_catalog(conn)


def test_catalog_verification_rejects_wrong_column_type():
    conn = _CatalogConn(
        catalog_version=26,
        wrong_column=("sessions", "started_at", "data_type", "text"),
    )

    with pytest.raises(RuntimeError, match="sessions.started_at"):
        PgSessionDB._verify_catalog(conn)


def test_catalog_verification_rejects_wrong_column_nullability():
    conn = _CatalogConn(
        catalog_version=26,
        wrong_column=("sessions", "source", "nullable", True),
    )

    with pytest.raises(RuntimeError, match="sessions.source"):
        PgSessionDB._verify_catalog(conn)


def test_catalog_verification_rejects_wrong_column_default():
    conn = _CatalogConn(
        catalog_version=26,
        wrong_column=("sessions", "archived", "default", "1"),
    )

    with pytest.raises(RuntimeError, match="sessions.archived"):
        PgSessionDB._verify_catalog(conn)


def test_catalog_verification_rejects_missing_identity_semantics():
    conn = _CatalogConn(
        catalog_version=26,
        wrong_column=(
            "messages",
            "id",
            "identity_generation",
            None,
        ),
    )

    with pytest.raises(RuntimeError, match="messages.id"):
        PgSessionDB._verify_catalog(conn)


def test_catalog_verification_rejects_unexpected_constraint():
    conn = _CatalogConn(
        catalog_version=26,
        extra_constraint=(
            "sessions",
            "sessions_source_check",
            "c",
            False,
            False,
            True,
            ("source",),
            None,
            None,
            (),
            " ",
            " ",
            " ",
        ),
    )

    with pytest.raises(RuntimeError, match="sessions_source_check"):
        PgSessionDB._verify_catalog(conn)


def test_catalog_verification_ignores_postgres18_not_null_catalog_rows():
    PgSessionDB._verify_catalog(
        _CatalogConn(catalog_version=26, pg18_not_null_constraints=True)
    )


def test_catalog_verification_rejects_wrong_foreign_key_delete_action():
    conn = _CatalogConn(
        catalog_version=26,
        wrong_constraint=(
            (
                "session_model_usage",
                "session_model_usage_session_id_fkey",
            ),
            "delete_action",
            "a",
        ),
    )

    with pytest.raises(RuntimeError, match="session_model_usage_session_id_fkey"):
        PgSessionDB._verify_catalog(conn)


def test_catalog_verification_allows_only_the_optional_trigram_index():
    PgSessionDB._verify_catalog(
        _CatalogConn(catalog_version=26, include_optional_index=True)
    )

    conn = _CatalogConn(
        catalog_version=26,
        extra_index=(
            "idx_unreviewed_sessions_source",
            hermes_state_pg._index_spec("sessions", "source"),
        ),
    )
    with pytest.raises(RuntimeError, match="idx_unreviewed_sessions_source"):
        PgSessionDB._verify_catalog(conn)


def test_catalog_verification_rejects_wrong_optional_trigram_index():
    conn = _CatalogConn(
        catalog_version=26,
        include_optional_index=True,
        wrong_index="idx_messages_search_trgm",
    )

    with pytest.raises(RuntimeError, match="idx_messages_search_trgm"):
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


def test_ordinary_v26_boot_is_observation_only():
    conn = _CatalogConn(
        version=EXPECTED_SCHEMA_VERSION,
        surface=EXPECTED_SCHEMA_SURFACE_SHA256,
        catalog_version=EXPECTED_SCHEMA_VERSION,
        include_optional_index=True,
    )
    db = PgSessionDB.__new__(PgSessionDB)
    db._pool = _Pool(conn)
    db._allow_schema_migration = False
    db._storage_attestation = None

    db._init_pg_schema()

    assert not any(
        sql.lstrip().startswith(("CREATE", "ALTER", "INSERT", "UPDATE", "DELETE"))
        for sql, _ in conn.executed
    )
    assert db._storage_attestation == {
        "backend": "postgres",
        "schema_version": EXPECTED_SCHEMA_VERSION,
        "surface_marker": EXPECTED_SCHEMA_SURFACE_SHA256,
    }


def test_target_boot_rejects_pre_summary_surface_without_ddl():
    conn = _CatalogConn(
        version=EXPECTED_SCHEMA_VERSION,
        surface=hermes_state_pg._V26_PRE_SUMMARY_SCHEMA_SURFACE_SHA256,
        catalog_version=EXPECTED_SCHEMA_VERSION,
        include_optional_index=True,
        pre_summary_surface=True,
    )
    db = PgSessionDB.__new__(PgSessionDB)
    db._pool = _Pool(conn)
    db._allow_schema_migration = False
    db._storage_attestation = None

    with pytest.raises(RuntimeError, match="surface migration required"):
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


def test_explicit_v22_preflight_accepts_the_optional_trigram_index():
    conn = _CatalogConn(include_optional_index=True)

    PgSessionDB._assert_v22_migration_precondition(conn)

    assert not any(
        sql.lstrip().startswith(("CREATE", "ALTER", "INSERT", "UPDATE", "DELETE"))
        for sql, _ in conn.executed
    )


@pytest.mark.parametrize(
    ("catalog_drift", "match"),
    [
        (
            {"wrong_column": ("sessions", "source", "nullable", True)},
            "sessions.source",
        ),
        (
            {"wrong_index": "idx_sessions_started"},
            "idx_sessions_started",
        ),
        (
            {
                "extra_constraint": (
                    "sessions",
                    "sessions_source_check",
                    "c",
                    False,
                    False,
                    True,
                    ("source",),
                    None,
                    None,
                    (),
                    " ",
                    " ",
                    " ",
                )
            },
            "sessions_source_check",
        ),
    ],
)
def test_explicit_v22_preflight_rejects_semantic_catalog_drift(
    catalog_drift,
    match,
):
    conn = _CatalogConn(**catalog_drift)

    with pytest.raises(RuntimeError, match=match):
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


def test_v26_surface_migration_is_exact_and_idempotent_at_destination():
    conn = _CatalogConn(
        version=EXPECTED_SCHEMA_VERSION,
        surface=hermes_state_pg._V26_PRE_SUMMARY_SCHEMA_SURFACE_SHA256,
        catalog_version=EXPECTED_SCHEMA_VERSION,
        pre_summary_surface=True,
    )

    result = hermes_state_pg._migrate_v26_surface_on_connection(conn)

    assert result["migration_applied"] is True
    assert result["surface_marker"] == EXPECTED_SCHEMA_SURFACE_SHA256
    assert conn.executed[:2] == [
        (
            "SELECT set_config('lock_timeout', %s, true)",
            (hermes_state_pg._V26_MIGRATION_LOCK_TIMEOUT,),
        ),
        (
            "SELECT set_config('statement_timeout', %s, true)",
            (hermes_state_pg._V26_MIGRATION_STATEMENT_TIMEOUT,),
        ),
    ]
    statements = [sql for sql, _ in conn.executed]
    alter_at = statements.index(hermes_state_pg.PG_V26_SUMMARY_SURFACE_SQL)
    marker_at = next(
        i
        for i, sql in enumerate(statements)
        if sql.startswith("UPDATE hermes_state.state_meta")
    )
    assert alter_at < marker_at
    assert conn.executed[marker_at][1] == (
        EXPECTED_SCHEMA_SURFACE_SHA256,
        hermes_state_pg._META_SURFACE_KEY,
        hermes_state_pg._V26_PRE_SUMMARY_SCHEMA_SURFACE_SHA256,
    )

    conn.executed.clear()
    second = hermes_state_pg._migrate_v26_surface_on_connection(conn)
    assert second["migration_required"] is False
    assert not any(
        sql.lstrip().startswith(("ALTER", "UPDATE"))
        for sql, _ in conn.executed
    )


def test_v26_surface_migration_refuses_marker_catalog_disagreement():
    conn = _CatalogConn(
        version=EXPECTED_SCHEMA_VERSION,
        surface=EXPECTED_SCHEMA_SURFACE_SHA256,
        catalog_version=EXPECTED_SCHEMA_VERSION,
        pre_summary_surface=True,
    )

    with pytest.raises(RuntimeError, match="_compressed_summary"):
        hermes_state_pg._migrate_v26_surface_on_connection(conn)

    assert not any(
        sql.lstrip().startswith(("ALTER", "UPDATE"))
        for sql, _ in conn.executed
    )


def test_v26_surface_migration_refuses_old_marker_with_new_catalog():
    conn = _CatalogConn(
        version=EXPECTED_SCHEMA_VERSION,
        surface=hermes_state_pg._V26_PRE_SUMMARY_SCHEMA_SURFACE_SHA256,
        catalog_version=EXPECTED_SCHEMA_VERSION,
    )

    with pytest.raises(RuntimeError, match="unexpected columns"):
        hermes_state_pg._migrate_v26_surface_on_connection(conn)

    assert not any(
        sql.lstrip().startswith(("ALTER", "UPDATE"))
        for sql, _ in conn.executed
    )


@pytest.mark.parametrize("apply", [False, True])
def test_v26_surface_migration_script_never_prints_dsn(
    monkeypatch, capsys, apply
):
    observed = []

    def _evidence(dsn):
        observed.append(dsn)
        return {
            "backend": "postgres",
            "schema_version": EXPECTED_SCHEMA_VERSION,
            "surface_marker": EXPECTED_SCHEMA_SURFACE_SHA256,
            "target_surface_marker": EXPECTED_SCHEMA_SURFACE_SHA256,
            "migration_required": False,
        }

    monkeypatch.setattr(
        surface_migration_script,
        "migrate_v26_surface" if apply else "inspect_v26_surface_migration_precondition",
        _evidence,
    )
    secret_dsn = (
        "postgresql://operator:do-not-print@example.invalid:6543/db"
        "?application_name=also-do-not-print"
    )
    argv = ["--dsn", secret_dsn]
    if apply:
        argv.append("--apply")

    assert surface_migration_script.main(argv) == 0

    output = capsys.readouterr()
    assert observed == [secret_dsn]
    assert 'dsn_identity={"database":"db","host":"example.invalid","port":"6543"}' in output.out
    assert secret_dsn not in output.out
    assert secret_dsn not in output.err
    assert "operator" not in output.out
    assert "do-not-print" not in output.out
    assert "application_name" not in output.out


@pytest.mark.parametrize("apply", [False, True])
def test_v26_surface_migration_script_redacts_dsn_on_failure(
    monkeypatch, capsys, apply
):
    def _fail(dsn):
        raise RuntimeError(
            f"Postgres v26 surface migration could not connect to {dsn}"
        )

    monkeypatch.setattr(
        surface_migration_script,
        "migrate_v26_surface"
        if apply
        else "inspect_v26_surface_migration_precondition",
        _fail,
    )
    secret_dsn = "postgresql://operator:do-not-print@example.invalid/db"
    argv = ["--dsn", secret_dsn]
    if apply:
        argv.append("--apply")

    assert surface_migration_script.main(argv) == 1

    output = capsys.readouterr()
    assert secret_dsn not in output.out
    assert secret_dsn not in output.err
    assert "do-not-print" not in output.err
    assert "RuntimeError" in output.err
    assert "connection details suppressed" in output.err


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
        "_compressed_summary",
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
    server._session_db_cache_lock = threading.Lock()
    server._session_db_cache_closed = False
    server._session_dbs = {}
    db = APIServerAdapter._open_and_cache_session_db(
        server, hermes_state.DEFAULT_DB_PATH.parent
    )
    assert type(db) is _FakePg
