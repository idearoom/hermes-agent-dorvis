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
