"""Postgres-backed session store (IdeaRoom D6b / AE-115).

``PgSessionDB`` is a drop-in replacement for the SQLite ``SessionDB`` in
``hermes_state.py``. It exists so the Hermes gateway can run drain-based
blue/green deploys (ADR 0177 in idearoom-agents): ``state.db`` on EFS was the
last single-writer constraint, because SQLite requires "one gateway task,
period" while Postgres MVCC allows two gateway tasks to share session state
during a drain window.

Selection: any ``SessionDB()`` constructed WITHOUT an explicit ``db_path``
returns a ``PgSessionDB`` when ``HERMES_STATE_STORE_DSN`` is set (see
``SessionDB.__new__``). With the env var unset, the SQLite path is untouched —
this module is never imported. ``psycopg`` is imported lazily for the same
reason, mirroring ``gateway/platforms/response_store_pg.py`` (the D6a
precedent).

Design: instead of re-implementing SessionDB's ~130 methods, ``PgSessionDB``
subclasses ``SessionDB`` and swaps the connection layer. Inherited method
bodies run their (almost entirely standard) SQL through a small
SQLite→Postgres statement translator (`?`→`%s`, ``INSERT OR IGNORE``→
``ON CONFLICT DO NOTHING``, ``json_extract``→``jsonb ->>``, ``LIKE``→``ILIKE``
to keep SQLite's ASCII case-insensitive matching, ``X'0A'`` hex literals).
Only the genuinely dialect-divergent surfaces are overridden: construction,
schema init, the write-transaction executor, FTS search (tsvector + ILIKE
instead of FTS5/trigram), and SQLite maintenance (WAL checkpoints, VACUUM,
FTS optimize, corruption repair) which Postgres obsoletes.

Tables live in a dedicated ``hermes_state`` schema (the D6a response store
uses ``hermes_gw``) so gateway storage never collides with the web app's
drizzle-managed tables in the shared RDS database.

Rebase safety: ``EXPECTED_SCHEMA_VERSION`` and
``EXPECTED_SCHEMA_SURFACE_SHA256`` pin the upstream ``hermes_state`` schema this adapter was written against.
Boot fails loudly if an upstream rebase moved the schema without this file
being re-audited — see ``assert_schema_compat``.
"""

from __future__ import annotations

import hashlib
import logging
import os
import random
import re
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, TypeVar

import hermes_state
from hermes_state import DEFAULT_DB_PATH, SessionDB

logger = logging.getLogger(__name__)

T = TypeVar("T")

ENV_VAR = "HERMES_STATE_STORE_DSN"

_SCHEMA = "hermes_state"

# ── Rebase-drift guard ──────────────────────────────────────────────────────
# The SCHEMA_VERSION and the sha256 of the upstream SQLite DDL that this
# Postgres adapter was audited against. If an upstream rebase bumps the
# version or edits SCHEMA_SQL/DEFERRED_INDEX_SQL, PgSessionDB must refuse to
# boot until a human re-audits PG_SCHEMA_SQL (and ships any expand/contract
# migration per ADR 0177's coexistence rule), then updates these constants.
EXPECTED_SCHEMA_VERSION = 19
EXPECTED_SCHEMA_SURFACE_SHA256 = (
    "df28fdc5fd8be0e48373abed404a6cd33ccf88f2fa11962ac29d53d75ced15a0"
)

# state_meta keys used by the Pg backend's persisted markers.
_META_SURFACE_KEY = "pg_backend_schema_surface_sha256"


def schema_surface_hash() -> str:
    """sha256 over the upstream SQLite DDL this adapter mirrors."""
    surface = hermes_state.SCHEMA_SQL + hermes_state.DEFERRED_INDEX_SQL
    return hashlib.sha256(surface.encode("utf-8")).hexdigest()


def assert_schema_compat() -> None:
    """Fail loudly when hermes_state.py's schema moved under this adapter.

    This is the guard that makes future upstream rebases safe: the SQLite
    reconciler self-heals schema drift, but the Postgres DDL in this file is
    hand-mirrored and would silently diverge.
    """
    problems = []
    if hermes_state.SCHEMA_VERSION != EXPECTED_SCHEMA_VERSION:
        problems.append(
            f"hermes_state.SCHEMA_VERSION is {hermes_state.SCHEMA_VERSION}, "
            f"but hermes_state_pg was written against {EXPECTED_SCHEMA_VERSION}"
        )
    actual_hash = schema_surface_hash()
    if actual_hash != EXPECTED_SCHEMA_SURFACE_SHA256:
        problems.append(
            "hermes_state.SCHEMA_SQL/DEFERRED_INDEX_SQL content hash is "
            f"{actual_hash}, expected {EXPECTED_SCHEMA_SURFACE_SHA256}"
        )
    if problems:
        raise RuntimeError(
            "PgSessionDB schema-compat check failed — this is the "
            "rebase-drift guard. An upstream hermes_state.py rebase changed "
            "the session schema and hermes_state_pg.py has not been "
            "re-audited against it. Refusing to boot the Postgres session "
            "store rather than corrupting shared state. Fix: mirror the "
            "schema change into PG_SCHEMA_SQL (expand/contract per ADR 0177), "
            "ship any needed data migration, then update "
            "EXPECTED_SCHEMA_VERSION / EXPECTED_SCHEMA_SURFACE_SHA256. "
            "Details: " + "; ".join(problems)
        )


def _normalize_dsn(dsn: str) -> str:
    """node-pg's ``sslmode=no-verify`` → libpq's ``require`` (D6a parity)."""
    return dsn.replace("sslmode=no-verify", "sslmode=require")


# ── Postgres DDL (mirrors hermes_state.SCHEMA_SQL @ v19) ───────────────────
# Type mapping: TEXT→TEXT, REAL→DOUBLE PRECISION, INTEGER→BIGINT,
# AUTOINCREMENT→IDENTITY (BY DEFAULT, so the migration script can insert
# explicit ids). FKs are DEFERRABLE so the one-time migration can bulk-copy
# in one transaction. The telegram tables (created lazily by
# apply_telegram_topic_migration on SQLite) are created eagerly here at their
# terminal v2 shape.
_SEARCH_TEXT_SQL = (
    "left(COALESCE({a}content, '') || ' ' || COALESCE({a}tool_name, '') "
    "|| ' ' || COALESCE({a}tool_calls, ''), 500000)"
)

PG_SCHEMA_SQL: List[str] = [
    f"CREATE SCHEMA IF NOT EXISTS {_SCHEMA}",
    f"""CREATE TABLE IF NOT EXISTS {_SCHEMA}.schema_version (
        version BIGINT NOT NULL
    )""",
    f"""CREATE TABLE IF NOT EXISTS {_SCHEMA}.sessions (
        id TEXT PRIMARY KEY,
        source TEXT NOT NULL,
        user_id TEXT,
        session_key TEXT,
        chat_id TEXT,
        chat_type TEXT,
        thread_id TEXT,
        display_name TEXT,
        origin_json TEXT,
        expiry_finalized BIGINT DEFAULT 0,
        model TEXT,
        model_config TEXT,
        system_prompt TEXT,
        parent_session_id TEXT,
        started_at DOUBLE PRECISION NOT NULL,
        ended_at DOUBLE PRECISION,
        end_reason TEXT,
        message_count BIGINT DEFAULT 0,
        tool_call_count BIGINT DEFAULT 0,
        input_tokens BIGINT DEFAULT 0,
        output_tokens BIGINT DEFAULT 0,
        cache_read_tokens BIGINT DEFAULT 0,
        cache_write_tokens BIGINT DEFAULT 0,
        reasoning_tokens BIGINT DEFAULT 0,
        cwd TEXT,
        git_branch TEXT,
        git_repo_root TEXT,
        billing_provider TEXT,
        billing_base_url TEXT,
        billing_mode TEXT,
        estimated_cost_usd DOUBLE PRECISION,
        actual_cost_usd DOUBLE PRECISION,
        cost_status TEXT,
        cost_source TEXT,
        pricing_version TEXT,
        title TEXT,
        api_call_count BIGINT DEFAULT 0,
        handoff_state TEXT,
        handoff_platform TEXT,
        handoff_error TEXT,
        compression_failure_cooldown_until DOUBLE PRECISION,
        compression_failure_error TEXT,
        rewind_count BIGINT NOT NULL DEFAULT 0,
        archived BIGINT NOT NULL DEFAULT 0,
        FOREIGN KEY (parent_session_id) REFERENCES {_SCHEMA}.sessions(id)
            DEFERRABLE INITIALLY IMMEDIATE
    )""",
    f"""CREATE TABLE IF NOT EXISTS {_SCHEMA}.messages (
        id BIGINT GENERATED BY DEFAULT AS IDENTITY PRIMARY KEY,
        session_id TEXT NOT NULL REFERENCES {_SCHEMA}.sessions(id)
            DEFERRABLE INITIALLY IMMEDIATE,
        role TEXT NOT NULL,
        content TEXT,
        tool_call_id TEXT,
        tool_calls TEXT,
        tool_name TEXT,
        timestamp DOUBLE PRECISION NOT NULL,
        token_count BIGINT,
        finish_reason TEXT,
        reasoning TEXT,
        reasoning_content TEXT,
        reasoning_details TEXT,
        codex_reasoning_items TEXT,
        codex_message_items TEXT,
        platform_message_id TEXT,
        observed BIGINT DEFAULT 0,
        active BIGINT NOT NULL DEFAULT 1,
        compacted BIGINT NOT NULL DEFAULT 0
    )""",
    f"""CREATE TABLE IF NOT EXISTS {_SCHEMA}.state_meta (
        key TEXT PRIMARY KEY,
        value TEXT
    )""",
    f"""CREATE TABLE IF NOT EXISTS {_SCHEMA}.gateway_routing (
        scope TEXT NOT NULL DEFAULT '',
        session_key TEXT NOT NULL,
        entry_json TEXT NOT NULL,
        updated_at DOUBLE PRECISION NOT NULL,
        PRIMARY KEY (scope, session_key)
    )""",
    f"""CREATE TABLE IF NOT EXISTS {_SCHEMA}.compression_locks (
        session_id TEXT PRIMARY KEY,
        holder TEXT NOT NULL,
        acquired_at DOUBLE PRECISION NOT NULL,
        expires_at DOUBLE PRECISION NOT NULL
    )""",
    # Telegram topic-mode tables (terminal v2 shape: ON DELETE CASCADE).
    f"""CREATE TABLE IF NOT EXISTS {_SCHEMA}.telegram_dm_topic_mode (
        chat_id TEXT PRIMARY KEY,
        user_id TEXT NOT NULL,
        enabled BIGINT NOT NULL DEFAULT 1,
        activated_at DOUBLE PRECISION NOT NULL,
        updated_at DOUBLE PRECISION NOT NULL,
        has_topics_enabled BIGINT,
        allows_users_to_create_topics BIGINT,
        capability_checked_at DOUBLE PRECISION,
        intro_message_id TEXT,
        pinned_message_id TEXT
    )""",
    f"""CREATE TABLE IF NOT EXISTS {_SCHEMA}.telegram_dm_topic_bindings (
        chat_id TEXT NOT NULL,
        thread_id TEXT NOT NULL,
        user_id TEXT NOT NULL,
        session_key TEXT NOT NULL,
        session_id TEXT NOT NULL REFERENCES {_SCHEMA}.sessions(id)
            ON DELETE CASCADE DEFERRABLE INITIALLY IMMEDIATE,
        managed_mode TEXT NOT NULL DEFAULT 'auto',
        linked_at DOUBLE PRECISION NOT NULL,
        updated_at DOUBLE PRECISION NOT NULL,
        PRIMARY KEY (chat_id, thread_id)
    )""",
    f"CREATE UNIQUE INDEX IF NOT EXISTS idx_telegram_dm_topic_bindings_session "
    f"ON {_SCHEMA}.telegram_dm_topic_bindings(session_id)",
    f"CREATE INDEX IF NOT EXISTS idx_telegram_dm_topic_bindings_user "
    f"ON {_SCHEMA}.telegram_dm_topic_bindings(user_id, chat_id)",
    # Index parity with SCHEMA_SQL + DEFERRED_INDEX_SQL.
    f"CREATE INDEX IF NOT EXISTS idx_sessions_source ON {_SCHEMA}.sessions(source)",
    f"CREATE INDEX IF NOT EXISTS idx_sessions_source_id ON {_SCHEMA}.sessions(source, id)",
    f"CREATE INDEX IF NOT EXISTS idx_sessions_parent ON {_SCHEMA}.sessions(parent_session_id)",
    f"CREATE INDEX IF NOT EXISTS idx_sessions_started ON {_SCHEMA}.sessions(started_at DESC)",
    f"CREATE INDEX IF NOT EXISTS idx_messages_session ON {_SCHEMA}.messages(session_id, timestamp)",
    f"CREATE INDEX IF NOT EXISTS idx_compression_locks_expires ON {_SCHEMA}.compression_locks(expires_at)",
    f"CREATE INDEX IF NOT EXISTS idx_messages_session_active "
    f"ON {_SCHEMA}.messages(session_id, active, timestamp)",
    f"CREATE INDEX IF NOT EXISTS idx_sessions_session_key "
    f"ON {_SCHEMA}.sessions(session_key, started_at DESC)",
    f"CREATE INDEX IF NOT EXISTS idx_sessions_gateway_peer "
    f"ON {_SCHEMA}.sessions(source, user_id, chat_id, chat_type, thread_id, started_at DESC)",
    f"CREATE INDEX IF NOT EXISTS idx_sessions_handoff_state "
    f"ON {_SCHEMA}.sessions(handoff_state, started_at)",
    f"CREATE INDEX IF NOT EXISTS idx_messages_platform_msg_id "
    f"ON {_SCHEMA}.messages(session_id, platform_message_id) "
    f"WHERE platform_message_id IS NOT NULL",
    # FTS replacement: expression GIN index over the same concatenated text
    # the FTS5 triggers indexed. left(..., 500000) keeps pathological giant
    # messages under the 1MB tsvector hard limit so message INSERTs can
    # never fail on index maintenance.
    f"CREATE INDEX IF NOT EXISTS idx_messages_search_tsv ON {_SCHEMA}.messages "
    f"USING GIN (to_tsvector('simple', {_SEARCH_TEXT_SQL.format(a='')}))",
]

# Best-effort statements: run after PG_SCHEMA_SQL, failures downgrade
# gracefully (pg_trgm may be unavailable / unprivileged on some hosts; the
# ILIKE CJK path still works without the index, just unaccelerated).
PG_TRGM_SQL: List[str] = [
    "CREATE EXTENSION IF NOT EXISTS pg_trgm",
    f"CREATE INDEX IF NOT EXISTS idx_messages_search_trgm ON {_SCHEMA}.messages "
    f"USING GIN ({_SEARCH_TEXT_SQL.format(a='')} gin_trgm_ops)",
]


# ── SQLite → Postgres statement translation ────────────────────────────────

_HEX_LITERAL_RE = re.compile(r"\bX'([0-9A-Fa-f]+)'")
_JSON_EXTRACT_COALESCE_RE = re.compile(
    r"json_extract\(\s*COALESCE\(\s*([A-Za-z_][\w.]*)\s*,\s*'\{\}'\s*\)\s*,"
    r"\s*'\$\.([A-Za-z_]\w*)'\s*\)"
)
_JSON_EXTRACT_PLAIN_RE = re.compile(
    r"json_extract\(\s*([A-Za-z_][\w.]*)\s*,\s*'\$\.([A-Za-z_]\w*)'\s*\)"
)
_INSERT_OR_IGNORE_RE = re.compile(r"^(\s*)INSERT\s+OR\s+IGNORE\s+INTO\b", re.IGNORECASE)
_LIKE_RE = re.compile(r"\bLIKE\b")
_INSERT_MESSAGES_RE = re.compile(r"^\s*INSERT\s+INTO\s+messages\s*\(", re.IGNORECASE)


def _split_literals(sql: str) -> List[tuple]:
    """Split *sql* into (is_literal, text) segments on single-quoted strings."""
    parts: List[tuple] = []
    i, n, start = 0, len(sql), 0
    while i < n:
        if sql[i] != "'":
            i += 1
            continue
        if i > start:
            parts.append((False, sql[start:i]))
        j = i + 1
        while j < n:
            if sql[j] == "'":
                if j + 1 < n and sql[j + 1] == "'":
                    j += 2
                    continue
                break
            j += 1
        end = j if j < n else n - 1
        parts.append((True, sql[i : end + 1]))
        i = end + 1
        start = i
    if start < n:
        parts.append((False, sql[start:]))
    return parts


def _translate_sql(sql: str, *, with_params: bool) -> str:
    """Rewrite a SessionDB SQLite statement into its Postgres equivalent.

    Literal-aware: ``?``→``%s``, ``LIKE``→``ILIKE`` (SQLite LIKE is
    ASCII-case-insensitive) and ``%``→``%%`` escaping only apply outside
    single-quoted string literals; ``json_extract``/hex-literal/
    ``INSERT OR IGNORE`` rewrites run on the raw statement first because
    their patterns intentionally span literals.
    """
    ignore = _INSERT_OR_IGNORE_RE.match(sql)
    if ignore:
        sql = _INSERT_OR_IGNORE_RE.sub(r"\1INSERT INTO", sql, count=1)
    sql = _HEX_LITERAL_RE.sub(
        lambda m: "E'"
        + "".join(
            f"\\x{m.group(1)[k:k + 2]}" for k in range(0, len(m.group(1)), 2)
        )
        + "'",
        sql,
    )
    sql = _JSON_EXTRACT_COALESCE_RE.sub(
        r"(COALESCE(\1, '{}')::jsonb ->> '\2')", sql
    )
    sql = _JSON_EXTRACT_PLAIN_RE.sub(r"(COALESCE(\1, '{}')::jsonb ->> '\2')", sql)

    out: List[str] = []
    for is_literal, seg in _split_literals(sql):
        if is_literal:
            out.append(seg.replace("%", "%%") if with_params else seg)
            continue
        seg = _LIKE_RE.sub("ILIKE", seg)
        if with_params:
            seg = seg.replace("%", "%%").replace("?", "%s")
        out.append(seg)
    sql = "".join(out)
    if ignore:
        sql += " ON CONFLICT DO NOTHING"
    return sql


_TRANSLATE_CACHE: Dict[tuple, str] = {}
_TRANSLATE_CACHE_MAX = 512


def _translate_cached(sql: str, *, with_params: bool) -> str:
    key = (sql, with_params)
    hit = _TRANSLATE_CACHE.get(key)
    if hit is None:
        hit = _translate_sql(sql, with_params=with_params)
        if len(_TRANSLATE_CACHE) >= _TRANSLATE_CACHE_MAX:
            _TRANSLATE_CACHE.clear()
        _TRANSLATE_CACHE[key] = hit
    return hit


def _adapt_param(value: Any) -> Any:
    """Bind-parameter fixups for SQLite-shaped call sites.

    - bool → int: the mirrored columns are BIGINT (SQLite convention), and
      client-side binding would otherwise render ``TRUE``.
    - NUL stripping: Postgres TEXT cannot contain ``\\x00`` (SQLite can).
    - Path → str.
    """
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, str) and "\x00" in value:
        return value.replace("\x00", "")
    if isinstance(value, Path):
        return str(value)
    return value


def _adapt_params(params: Any) -> Any:
    if params is None:
        return None
    return [_adapt_param(p) for p in params]


class _Row:
    """sqlite3.Row-alike: index + name access, keys(), dict(row) support."""

    __slots__ = ("_names", "_values")

    def __init__(self, names: List[str], values: tuple) -> None:
        self._names = names
        self._values = values

    def __getitem__(self, key):
        if isinstance(key, (int, slice)):
            return self._values[key]
        try:
            return self._values[self._names.index(key)]
        except ValueError:
            raise IndexError(f"No item with key {key!r}") from None

    def keys(self) -> List[str]:
        return list(self._names)

    def __iter__(self):
        return iter(self._values)

    def __len__(self) -> int:
        return len(self._values)

    def __repr__(self) -> str:
        return f"_Row({dict(zip(self._names, self._values))!r})"


class _Result:
    """Eagerly-fetched cursor result (pooled connections return immediately)."""

    __slots__ = ("_rows", "_idx", "rowcount", "lastrowid")

    def __init__(self, rows: List[_Row], rowcount: int, lastrowid: Optional[int]) -> None:
        self._rows = rows
        self._idx = 0
        self.rowcount = rowcount
        self.lastrowid = lastrowid

    def fetchone(self) -> Optional[_Row]:
        if self._idx >= len(self._rows):
            return None
        row = self._rows[self._idx]
        self._idx += 1
        return row

    def fetchall(self) -> List[_Row]:
        rows = self._rows[self._idx :]
        self._idx = len(self._rows)
        return rows

    def fetchmany(self, size: int = 1) -> List[_Row]:
        rows = self._rows[self._idx : self._idx + size]
        self._idx += len(rows)
        return rows

    def __iter__(self):
        while True:
            row = self.fetchone()
            if row is None:
                return
            yield row

    def close(self) -> None:  # cursor-interface parity
        pass


def _translate_exception(exc: Exception) -> Exception:
    """Map psycopg errors onto the sqlite3 hierarchy SessionDB callers catch.

    Inherited SessionDB methods (and their gateway callers) wrap degradation
    paths in ``except sqlite3.Error`` — e.g. the compression-lock fail-open
    branches. Translating preserves those semantics; the original psycopg
    error rides along as ``__cause__``.
    """
    import psycopg

    if isinstance(exc, psycopg.errors.IntegrityError):
        new: Exception = sqlite3.IntegrityError(str(exc))
    else:
        new = sqlite3.OperationalError(str(exc))
    new.__cause__ = exc
    return new


def _is_retryable_pg_error(exc: BaseException) -> bool:
    import psycopg

    cause = exc.__cause__ if isinstance(exc, sqlite3.Error) else exc
    return isinstance(
        cause,
        (psycopg.errors.DeadlockDetected, psycopg.errors.SerializationFailure),
    )


def _run_statement(conn, sql: str, params) -> _Result:
    """Translate + execute one statement on a live psycopg connection."""
    import psycopg

    params = _adapt_params(params)
    has_params = bool(params)
    wants_lastrowid = bool(_INSERT_MESSAGES_RE.match(sql))
    pg_sql = _translate_cached(sql, with_params=has_params)
    if wants_lastrowid and " RETURNING " not in pg_sql.upper():
        pg_sql += " RETURNING id"
    try:
        cur = conn.execute(pg_sql, params if has_params else None)
        names: List[str] = (
            [d.name for d in cur.description] if cur.description else []
        )
        rows = (
            [_Row(names, tuple(r)) for r in cur.fetchall()] if names else []
        )
        lastrowid = rows[0][0] if (wants_lastrowid and rows) else None
        return _Result(rows, cur.rowcount, lastrowid)
    except psycopg.Error as exc:
        raise _translate_exception(exc) from exc


class _TxnConn:
    """The ``conn`` handed to ``_execute_write`` bodies: one bound connection
    inside an open transaction. Matches the sqlite3.Connection surface the
    SessionDB write closures use (execute / executemany / cursor)."""

    def __init__(self, raw) -> None:
        self._raw = raw

    def execute(self, sql: str, params=None) -> _Result:
        return _run_statement(self._raw, sql, params)

    def executemany(self, sql: str, seq_of_params) -> None:
        import psycopg

        pg_sql = _translate_cached(sql, with_params=True)
        try:
            with self._raw.cursor() as cur:
                cur.executemany(
                    pg_sql, [_adapt_params(p) for p in seq_of_params]
                )
        except psycopg.Error as exc:
            raise _translate_exception(exc) from exc

    def cursor(self):
        return self

    def executescript(self, script: str) -> None:
        raise sqlite3.OperationalError(
            "executescript is not supported on the Postgres session store; "
            "override the calling method in PgSessionDB"
        )


class _PoolConn:
    """The read-path ``self._conn``: checks a pooled connection out per
    statement (autocommit), fetches eagerly, and returns it. Thread-safe
    without SessionDB's ``self._lock`` (which inherited read paths still
    take — harmlessly, it is reentrant here)."""

    def __init__(self, pool) -> None:
        self._pool = pool

    def execute(self, sql: str, params=None) -> _Result:
        import psycopg

        try:
            with self._pool.connection() as conn:
                return _run_statement(conn, sql, params)
        except psycopg.Error as exc:
            raise _translate_exception(exc) from exc

    def executemany(self, sql: str, seq_of_params) -> None:
        import psycopg

        try:
            with self._pool.connection() as conn:
                _TxnConn(conn).executemany(sql, seq_of_params)
        except psycopg.Error as exc:
            raise _translate_exception(exc) from exc

    def cursor(self):
        return self

    def executescript(self, script: str) -> None:
        raise sqlite3.OperationalError(
            "executescript is not supported on the Postgres session store"
        )

    def close(self) -> None:
        pass


class PgSessionDB(SessionDB):
    """Postgres-backed SessionDB (see module docstring).

    Constructed either directly (``PgSessionDB(dsn=...)``, used by tests and
    the migration script) or through ``SessionDB()``'s env-var dispatch,
    which passes the inherited ``(db_path, read_only)`` signature and lets
    ``__init__`` read the DSN from ``HERMES_STATE_STORE_DSN``.
    """

    # SQLite's multimodal-content sentinel is "\x00json:" — but Postgres
    # TEXT cannot store NUL bytes (the param adapter strips them). Use a
    # \x01 sentinel instead; _encode_content/_decode_content are classmethods
    # and pick this up automatically. scripts/migrate_state_to_postgres.py
    # rewrites the legacy prefix during the one-time copy.
    _CONTENT_JSON_PREFIX = "\x01json:"

    def __init__(
        self,
        db_path: Path = None,
        read_only: bool = False,
        *,
        dsn: Optional[str] = None,
        min_pool: int = 1,
        max_pool: int = 8,
    ) -> None:
        # No super().__init__() — the SQLite constructor is all file/WAL
        # machinery. Replicate the instance attributes inherited methods use.
        dsn = (dsn or os.environ.get(ENV_VAR, "")).strip()
        if not dsn:
            raise ValueError(
                f"PgSessionDB requires a DSN (argument or ${ENV_VAR})"
            )
        assert_schema_compat()

        # Vestigial on Postgres; kept because a few local-CLI callers read
        # ``db.db_path`` for display. No SQLite I/O ever happens through it.
        self.db_path = db_path or DEFAULT_DB_PATH
        self.read_only = read_only
        # Reentrant: inherited read paths take ``with self._lock`` around
        # what is now a per-statement pooled checkout.
        self._lock = threading.RLock()
        self._write_count = 0
        # tsvector search is always available; the ILIKE CJK path replaces
        # the trigram FTS5 table (works with or without pg_trgm).
        self._fts_enabled = True
        self._trigram_available = True
        self._fts_unavailable_warned = False
        self._trigram_unavailable_warned = False
        self._closed = False

        # Lazy imports — only the Postgres path needs psycopg installed.
        from psycopg import ClientCursor
        from psycopg_pool import ConnectionPool

        def _configure(conn) -> None:
            # ClientCursor = client-side binding. SessionDB SQL leans on
            # SQLite's dynamic typing (e.g. ``CASE WHEN ? IS NULL``); with
            # server-side binding Postgres cannot infer those parameter
            # types, with literals it can.
            conn.cursor_factory = ClientCursor
            # Unqualified table names in inherited SQL resolve to our schema.
            conn.execute(f"SET search_path TO {_SCHEMA}, public")

        # open=True fails fast on a bad DSN at construction, mirroring
        # PgResponseStore; callers keep their existing "SessionDB()
        # raised → degrade" behavior.
        self._pool = ConnectionPool(
            conninfo=_normalize_dsn(dsn),
            min_size=min_pool,
            max_size=max_pool,
            kwargs={"autocommit": True},
            configure=_configure,
            open=True,
            name="hermes-session-store",
        )
        try:
            self._init_pg_schema()
        except Exception:
            self._pool.close()
            raise
        self._conn = _PoolConn(self._pool)

    # ── Schema ──────────────────────────────────────────────────────────

    # Serializes concurrent cold-boot bootstrap (see _init_pg_schema).
    # Stable 63-bit advisory-lock key derived from the schema name.
    _BOOTSTRAP_LOCK_KEY = int.from_bytes(
        hashlib.sha256(f"hermes_state_pg:{_SCHEMA}".encode()).digest()[:8],
        "big",
        signed=True,
    )

    def _init_pg_schema(self) -> None:
        # Two gateway tasks can construct PgSessionDB concurrently against
        # the same database (the blue/green overlap window of ADR 0177, or a
        # service scaling from zero). Postgres CREATE TABLE IF NOT EXISTS is
        # not concurrency-safe when the table genuinely doesn't exist yet
        # (both creators race in pg_type: "duplicate key value violates
        # unique constraint pg_type_typname_nsp_index"), and the
        # schema_version seed below is check-then-insert. Serialize the whole
        # bootstrap behind a session advisory lock; after first boot the lock
        # is held only for the duration of no-op IF NOT EXISTS statements.
        with self._pool.connection() as conn:
            conn.execute(
                "SELECT pg_advisory_lock(%s)", (self._BOOTSTRAP_LOCK_KEY,)
            )
            try:
                for stmt in PG_SCHEMA_SQL:
                    conn.execute(stmt)
                for stmt in PG_TRGM_SQL:
                    try:
                        conn.execute(stmt)
                    except Exception as exc:
                        logger.info(
                            "pg_trgm acceleration unavailable for the session "
                            "store (CJK search falls back to unindexed ILIKE): %s",
                            exc,
                        )
                        break
                self._assert_persisted_schema_markers(conn)
            finally:
                try:
                    conn.execute(
                        "SELECT pg_advisory_unlock(%s)", (self._BOOTSTRAP_LOCK_KEY,)
                    )
                except Exception:
                    # Don't mask the bootstrap error; a dead connection
                    # releases its session advisory locks server-side anyway.
                    logger.warning(
                        "failed to release session-store bootstrap advisory lock",
                        exc_info=True,
                    )

    def _assert_persisted_schema_markers(self, conn) -> None:
        """Record/verify the schema version + surface hash inside Postgres.

        Guards the other half of the drift matrix: this *database* was
        initialized by a build pinned to a specific upstream schema. A build
        pinned to a different one must not write into it silently — schema
        moves need a deliberate expand/contract migration (ADR 0177).
        """
        row = conn.execute(
            f"SELECT version FROM {_SCHEMA}.schema_version LIMIT 1"
        ).fetchone()
        if row is None:
            conn.execute(
                f"INSERT INTO {_SCHEMA}.schema_version (version) VALUES (%s)",
                (EXPECTED_SCHEMA_VERSION,),
            )
        elif int(row[0]) != EXPECTED_SCHEMA_VERSION:
            raise RuntimeError(
                f"Postgres session store schema_version is {row[0]}, but "
                f"this build expects {EXPECTED_SCHEMA_VERSION}. A different "
                "hermes build initialized this database — ship an explicit "
                "migration (ADR 0177 expand/contract) before pointing this "
                "build at it."
            )
        row = conn.execute(
            f"SELECT value FROM {_SCHEMA}.state_meta WHERE key = %s",
            (_META_SURFACE_KEY,),
        ).fetchone()
        if row is None:
            conn.execute(
                f"INSERT INTO {_SCHEMA}.state_meta (key, value) VALUES (%s, %s) "
                "ON CONFLICT (key) DO NOTHING",
                (_META_SURFACE_KEY, EXPECTED_SCHEMA_SURFACE_SHA256),
            )
        elif row[0] != EXPECTED_SCHEMA_SURFACE_SHA256:
            raise RuntimeError(
                "Postgres session store was initialized against a different "
                f"upstream schema surface (db={row[0]}, "
                f"build={EXPECTED_SCHEMA_SURFACE_SHA256}). Rebase drift — "
                "re-audit hermes_state_pg.py and migrate deliberately."
            )

    def _init_schema(self) -> None:  # pragma: no cover - safety net
        self._init_pg_schema()

    # ── Write executor ──────────────────────────────────────────────────

    def _execute_write(self, fn: Callable[[Any], T]) -> T:
        """One transaction per write closure, with deadlock retry.

        Replaces SQLite's BEGIN IMMEDIATE + busy-retry: Postgres MVCC makes
        writer convoys structurally impossible; only deadlocks/serialization
        failures are worth retrying.
        """
        last_err: Optional[Exception] = None
        for attempt in range(self._WRITE_MAX_RETRIES):
            try:
                with self._pool.connection() as raw:
                    with raw.transaction():
                        result = fn(_TxnConn(raw))
                self._write_count += 1
                return result
            except sqlite3.Error as exc:
                if _is_retryable_pg_error(exc) and attempt < self._WRITE_MAX_RETRIES - 1:
                    last_err = exc
                    time.sleep(
                        random.uniform(
                            self._WRITE_RETRY_MIN_S, self._WRITE_RETRY_MAX_S
                        )
                    )
                    continue
                raise
            except Exception as exc:
                import psycopg

                if isinstance(exc, psycopg.Error):
                    raise _translate_exception(exc) from exc
                raise
        raise last_err or sqlite3.OperationalError(
            "postgres write failed after max retries"
        )

    # ── Lifecycle / SQLite maintenance obsoleted by Postgres ────────────

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self._pool.close()
        except Exception:
            pass

    def _try_wal_checkpoint(self) -> None:
        pass

    def _try_optimize_fts(self) -> None:
        pass

    def optimize_fts(self) -> int:
        return 0

    def _fts_table_exists(self, name: str) -> bool:
        return False

    def vacuum(self) -> int:
        # Postgres autovacuum owns this; an explicit blocking VACUUM from
        # the gateway would be wasted I/O on shared RDS.
        return 0

    def apply_telegram_topic_migration(self) -> None:
        # Tables are created eagerly at the terminal v2 shape in
        # PG_SCHEMA_SQL (Postgres FKs support ON DELETE CASCADE natively,
        # no rebuild dance needed). Just record the version marker.
        self.set_meta("telegram_dm_topic_schema_version", "2")

    # ── Search (tsvector + ILIKE replaces FTS5 + trigram) ───────────────

    @staticmethod
    def _fts5_to_tsquery(sanitized: str) -> str:
        """Convert a sanitized FTS5 query into a ``to_tsquery`` expression.

        Mirrors FTS5 semantics: implicit AND, OR / NOT operators, quoted
        phrases (``<->`` adjacency), trailing ``*`` prefix matches.
        """

        def _lex(word: str, prefix: bool = False) -> str:
            quoted = "'" + word.replace("'", "''") + "'"
            return quoted + (":*" if prefix else "")

        tokens = re.findall(r'"[^"]*"|\S+', sanitized)
        parts: List[str] = []
        pending_op: Optional[str] = None
        for tok in tokens:
            upper = tok.upper()
            if upper == "AND":
                continue  # implicit
            if upper == "OR":
                pending_op = "|"
                continue
            if upper == "NOT":
                pending_op = "&!"
                continue
            if tok.startswith('"') and tok.endswith('"') and len(tok) >= 2:
                words = re.findall(r"\w+", tok[1:-1])
                if not words:
                    pending_op = None
                    continue
                piece = " <-> ".join(_lex(w) for w in words)
                if len(words) > 1:
                    piece = f"({piece})"
            else:
                prefix = tok.endswith("*")
                words = re.findall(r"\w+", tok.rstrip("*"))
                if not words:
                    pending_op = None
                    continue
                if len(words) == 1:
                    piece = _lex(words[0], prefix)
                else:
                    piece = "(" + " <-> ".join(_lex(w) for w in words) + ")"
            if parts:
                op = pending_op or "&"
                if op == "&!":
                    parts.append("& !")
                else:
                    parts.append(op)
            elif pending_op == "&!":
                parts.append("!")
            parts.append(piece)
            pending_op = None
        return " ".join(parts)

    def search_messages(
        self,
        query: str,
        source_filter: List[str] = None,
        exclude_sources: List[str] = None,
        role_filter: List[str] = None,
        limit: int = 20,
        offset: int = 0,
        sort: str = None,
        include_inactive: bool = False,
    ) -> List[Dict[str, Any]]:
        """Postgres port of :meth:`SessionDB.search_messages`.

        Same contract and result shape; tsvector('simple') replaces the FTS5
        unicode61 table and per-token ILIKE replaces the trigram/LIKE CJK
        paths (ILIKE substring match has no 3-char minimum, so short-CJK
        queries take the same path as long ones).
        """
        if not query or not query.strip():
            return []
        sanitized = self._sanitize_fts5_query(query)
        if not sanitized:
            return []

        if isinstance(sort, str):
            sort_norm = sort.strip().lower()
            if sort_norm not in ("newest", "oldest"):
                sort_norm = None
        else:
            sort_norm = None

        search_text = _SEARCH_TEXT_SQL.format(a="m.")

        shared_where: List[str] = []
        shared_params: List[Any] = []
        if not include_inactive:
            shared_where.append("(m.active = 1 OR m.compacted = 1)")
        if source_filter is not None:
            shared_where.append(
                f"s.source IN ({','.join('?' for _ in source_filter)})"
            )
            shared_params.extend(source_filter)
        if exclude_sources is not None:
            shared_where.append(
                f"s.source NOT IN ({','.join('?' for _ in exclude_sources)})"
            )
            shared_params.extend(exclude_sources)
        if role_filter:
            shared_where.append(
                f"m.role IN ({','.join('?' for _ in role_filter)})"
            )
            shared_params.extend(role_filter)

        if self._contains_cjk(sanitized):
            raw_query = sanitized.strip('"').strip()
            non_op_tokens = [
                t
                for t in raw_query.split()
                if t.upper() not in {"AND", "OR", "NOT"}
            ] or [raw_query]
            token_clauses: List[str] = []
            params: List[Any] = [non_op_tokens[0]]  # snippet anchor
            for tok in non_op_tokens:
                esc = (
                    tok.replace("\\", "\\\\")
                    .replace("%", "\\%")
                    .replace("_", "\\_")
                )
                token_clauses.append(
                    "(m.content LIKE ? ESCAPE '\\' OR m.tool_name LIKE ? "
                    "ESCAPE '\\' OR m.tool_calls LIKE ? ESCAPE '\\')"
                )
                params += [f"%{esc}%"] * 3
            where = [f"({' OR '.join(token_clauses)})"] + shared_where
            params += shared_params
            if sort_norm == "oldest":
                order_by = "ORDER BY m.timestamp ASC, m.id ASC"
            else:
                order_by = "ORDER BY m.timestamp DESC, m.id DESC"
            sql = f"""
                SELECT m.id, m.session_id, m.role,
                       substr(m.content,
                              GREATEST(1, STRPOS(m.content, ?) - 40),
                              120) AS snippet,
                       m.timestamp, m.tool_name,
                       s.source, s.model, s.started_at AS session_started
                FROM messages m
                JOIN sessions s ON s.id = m.session_id
                WHERE {' AND '.join(where)}
                {order_by}
                LIMIT ? OFFSET ?
            """
            params += [limit, offset]
            try:
                matches = [dict(r) for r in self._conn.execute(sql, params).fetchall()]
            except sqlite3.OperationalError:
                return []
        else:
            tsquery = self._fts5_to_tsquery(sanitized)
            if not tsquery:
                return []
            where = [
                f"to_tsvector('simple', {search_text}) @@ to_tsquery('simple', ?)"
            ] + shared_where
            if sort_norm == "newest":
                order_by = "ORDER BY q.timestamp DESC, q._score DESC"
            elif sort_norm == "oldest":
                order_by = "ORDER BY q.timestamp ASC, q._score DESC"
            else:
                order_by = "ORDER BY q._score DESC"
            # Inner query filters/orders/limits; ts_headline runs only on the
            # returned page.
            sql = f"""
                SELECT q.id, q.session_id, q.role,
                       ts_headline('simple', q._search_text,
                                   to_tsquery('simple', ?),
                                   'StartSel=>>>, StopSel=<<<, MaxWords=40, '
                                   'MinWords=10, MaxFragments=2, '
                                   'FragmentDelimiter=" ... "') AS snippet,
                       q.timestamp, q.tool_name,
                       q.source, q.model, q.session_started
                FROM (
                    SELECT m.id, m.session_id, m.role, m.timestamp,
                           m.tool_name, s.source, s.model,
                           s.started_at AS session_started,
                           {search_text} AS _search_text,
                           ts_rank_cd(to_tsvector('simple', {search_text}),
                                      to_tsquery('simple', ?)) AS _score
                    FROM messages m
                    JOIN sessions s ON s.id = m.session_id
                    WHERE {' AND '.join(where)}
                ) q
                {order_by}
                LIMIT ? OFFSET ?
            """
            params = [tsquery, tsquery, tsquery] + shared_params + [limit, offset]
            try:
                matches = [dict(r) for r in self._conn.execute(sql, params).fetchall()]
            except sqlite3.OperationalError:
                # tsquery syntax error despite sanitization — parity with the
                # FTS5 path's empty-result behavior.
                return []
            for m in matches:
                m.pop("_score", None)

        # Surrounding context (1 message before/after), then strip content —
        # same shape as the SQLite implementation.
        for match in matches:
            try:
                rows = self._conn.execute(
                    """WITH target AS (
                           SELECT session_id, timestamp, id
                           FROM messages
                           WHERE id = ?
                       )
                       SELECT role, content FROM (
                           SELECT m.id, m.timestamp, m.role, m.content
                           FROM messages m
                           JOIN target t ON t.session_id = m.session_id
                           WHERE (m.timestamp < t.timestamp)
                              OR (m.timestamp = t.timestamp AND m.id < t.id)
                           ORDER BY m.timestamp DESC, m.id DESC
                           LIMIT 1
                       ) prev_msg
                       UNION ALL
                       SELECT role, content FROM messages WHERE id = ?
                       UNION ALL
                       SELECT role, content FROM (
                           SELECT m.id, m.timestamp, m.role, m.content
                           FROM messages m
                           JOIN target t ON t.session_id = m.session_id
                           WHERE (m.timestamp > t.timestamp)
                              OR (m.timestamp = t.timestamp AND m.id > t.id)
                           ORDER BY m.timestamp ASC, m.id ASC
                           LIMIT 1
                       ) next_msg""",
                    (match["id"], match["id"]),
                ).fetchall()
                context_msgs = []
                for r in rows:
                    decoded = self._decode_content(r["content"])
                    if isinstance(decoded, list):
                        text_parts = [
                            p.get("text", "")
                            for p in decoded
                            if isinstance(p, dict) and p.get("type") == "text"
                        ]
                        text = " ".join(t for t in text_parts if t).strip()
                        preview = text or "[multimodal content]"
                    elif isinstance(decoded, str):
                        preview = decoded
                    else:
                        preview = ""
                    context_msgs.append(
                        {"role": r["role"], "content": preview[:200]}
                    )
                match["context"] = context_msgs
            except Exception:
                match["context"] = []

        for match in matches:
            match.pop("content", None)
            match.pop("_search_text", None)
        return matches
