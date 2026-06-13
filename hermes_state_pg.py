"""Postgres-backed session/state store (IdeaRoom D6b / AE-61) — FOUNDATION.

A Postgres backend for the SessionDB state store (``hermes_state.py``), the
second half of D6 after the response-store (D6a). On AWS the agent home is EFS
(NFS), where SQLite is ~1000x slower than local disk (measured 0.2 vs 201 txn/s,
8 concurrent writers); ``state.db``'s many per-turn writes serialize on the NFS
write lock and stall chat concurrency. Postgres MVCC + a connection pool removes
that, and also retires SessionDB's app-level BEGIN-IMMEDIATE + jitter-retry
machinery and the WAL/checkpoint logic.

STATUS: foundation. This implements the **per-turn agent-loop hot path** — the
methods the AWS gateway exercises every turn — with parity to the SQLite store:
sessions CRUD + token counts, append/get messages, state_meta, compression
locks (advisory via the same DELETE-expired + INSERT-or-ignore protocol), and
Postgres-native full-text search (tsvector + pg_trgm, validated to match the
FTS5 content/trigram behavior). The long-tail SessionDB surface (titles,
list_sessions_rich, telegram topic-mode, handoff, export/prune, rewind/restore,
get_messages_around/anchored, cron-run listing) is NOT yet ported — see
``_DEFERRED`` below. This class is intentionally **not wired into the
SessionDB dispatch yet**: it is dormant until D6b is completed and an env switch
(HERMES_STATE_DSN) is added in hermes_state.py + infra. So importing/using it has
zero effect on the SQLite path. See docs (idearoom-agents):
docs/hermes/d6-postgres-stores.md, docs/platform/ae-61-efs-sqlite-diagnostic-2026-06-13.md.

FTS port: the SQLite FTS5 ``messages_fts`` (content) + ``messages_fts_trigram``
become a generated ``tsvector`` column with a GIN index (full-text) plus a
pg_trgm GIN index on content (substring/CJK). Validated against Postgres.
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_SCHEMA = "hermes_gw"
# SessionDB uses "\x00json:" as the structured-content sentinel, but Postgres
# `text` columns cannot store a NUL byte (\x00) at all. Use a NUL-free sentinel
# (RS control char, which never appears in normal prose) for the Postgres store.
# Decode also accepts the legacy SQLite prefix defensively, though the Postgres
# store is created fresh (no \x00 content can have been written to it).
_CONTENT_JSON_PREFIX = "\x1ejson:"
_LEGACY_SQLITE_PREFIX = "\x00json:"

# SessionDB methods NOT yet ported here (dormant class; full port is the rest of
# D6b). Listed so a completeness check / the dispatch wiring can gate on it.
_DEFERRED = (
    "set_session_title", "get_session_title", "set_session_archived",
    "get_session_by_title", "resolve_session_by_title", "get_next_title_in_lineage",
    "list_sessions_rich", "list_cron_job_runs", "get_session_rich_row",
    "get_messages_around", "get_anchored_view", "get_messages_as_conversation",
    "resolve_resume_session_id", "rewind_to_message", "restore_rewound",
    "list_recent_user_messages", "get_compression_tip", "export_session",
    "export_all", "delete_session", "delete_sessions", "count_empty_sessions",
    "delete_empty_sessions", "prune_sessions", "search_sessions",
    "search_sessions_by_id", "prune_empty_ghost_sessions",
    "finalize_orphaned_compression_sessions", "ensure_session",
    "apply_telegram_topic_migration", "enable_telegram_topic_mode",
    "disable_telegram_topic_mode", "bind_telegram_topic", "request_handoff",
    "claim_handoff", "complete_handoff", "fail_handoff", "optimize_fts",
    "vacuum", "maybe_auto_prune_and_vacuum",
)


def _normalize_dsn(dsn: str) -> str:
    """node-pg ``sslmode=no-verify`` -> libpq ``sslmode=require`` (see D6a)."""
    return dsn.replace("sslmode=no-verify", "sslmode=require")


def _strip_nul(s: str) -> str:
    """Postgres text cannot store NUL bytes; SQLite can. Drop them on write."""
    return s.replace("\x00", "") if "\x00" in s else s


def _encode_content(content: Any) -> Any:
    """Serialize structured message content (parity with SessionDB._encode_content).

    Strips NUL bytes from plain strings (Postgres text constraint) and uses a
    NUL-free sentinel for structured (list/dict) content.
    """
    if content is None or isinstance(content, (int, float)):
        return content
    if isinstance(content, str):
        return _strip_nul(content)
    if isinstance(content, bytes):
        return _strip_nul(content.decode("utf-8", "replace"))
    try:
        return _CONTENT_JSON_PREFIX + _strip_nul(json.dumps(content))
    except (TypeError, ValueError):
        return _strip_nul(str(content))


def _decode_content(content: Any) -> Any:
    """Reverse _encode_content; also accepts the legacy SQLite \\x00 sentinel."""
    if isinstance(content, str):
        for prefix in (_CONTENT_JSON_PREFIX, _LEGACY_SQLITE_PREFIX):
            if content.startswith(prefix):
                try:
                    return json.loads(content[len(prefix):])
                except (json.JSONDecodeError, TypeError):
                    return content
    return content


# Columns the SQLite store accepts on create_session/_insert_session_row.
_SESSION_INSERT_COLS = (
    "model", "model_config", "system_prompt", "user_id",
    "parent_session_id", "cwd",
)


class PgSessionDB:
    """Postgres SessionDB backend — per-turn hot path (D6b foundation)."""

    def __init__(self, dsn: str, *, min_pool: int = 1, max_pool: int = 8) -> None:
        from psycopg.rows import dict_row
        from psycopg_pool import ConnectionPool

        self._dict_row = dict_row
        self._pool = ConnectionPool(
            conninfo=_normalize_dsn(dsn),
            min_size=min_pool,
            max_size=max_pool,
            kwargs={"autocommit": True, "row_factory": dict_row},
            open=True,
            name="hermes-session-store",
        )
        self._closed = False
        self._init_schema()

    # ── schema ──────────────────────────────────────────────────────────────
    def _init_schema(self) -> None:
        with self._pool.connection() as conn:
            conn.execute(f"CREATE SCHEMA IF NOT EXISTS {_SCHEMA}")
            conn.execute("CREATE EXTENSION IF NOT EXISTS pg_trgm")
            conn.execute(
                f"""CREATE TABLE IF NOT EXISTS {_SCHEMA}.sessions (
                    id TEXT PRIMARY KEY,
                    source TEXT NOT NULL,
                    user_id TEXT,
                    model TEXT,
                    model_config TEXT,
                    system_prompt TEXT,
                    parent_session_id TEXT,
                    started_at DOUBLE PRECISION NOT NULL,
                    ended_at DOUBLE PRECISION,
                    end_reason TEXT,
                    message_count INTEGER DEFAULT 0,
                    tool_call_count INTEGER DEFAULT 0,
                    input_tokens INTEGER DEFAULT 0,
                    output_tokens INTEGER DEFAULT 0,
                    cache_read_tokens INTEGER DEFAULT 0,
                    cache_write_tokens INTEGER DEFAULT 0,
                    reasoning_tokens INTEGER DEFAULT 0,
                    cwd TEXT,
                    billing_provider TEXT,
                    billing_base_url TEXT,
                    billing_mode TEXT,
                    estimated_cost_usd DOUBLE PRECISION,
                    actual_cost_usd DOUBLE PRECISION,
                    cost_status TEXT,
                    cost_source TEXT,
                    pricing_version TEXT,
                    title TEXT,
                    api_call_count INTEGER DEFAULT 0,
                    handoff_state TEXT,
                    handoff_platform TEXT,
                    handoff_error TEXT,
                    rewind_count INTEGER NOT NULL DEFAULT 0,
                    archived INTEGER NOT NULL DEFAULT 0
                )"""
            )
            conn.execute(
                f"""CREATE TABLE IF NOT EXISTS {_SCHEMA}.messages (
                    id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
                    session_id TEXT NOT NULL REFERENCES {_SCHEMA}.sessions(id),
                    role TEXT NOT NULL,
                    content TEXT,
                    tool_call_id TEXT,
                    tool_calls TEXT,
                    tool_name TEXT,
                    timestamp DOUBLE PRECISION NOT NULL,
                    token_count INTEGER,
                    finish_reason TEXT,
                    reasoning TEXT,
                    reasoning_content TEXT,
                    reasoning_details TEXT,
                    codex_reasoning_items TEXT,
                    codex_message_items TEXT,
                    platform_message_id TEXT,
                    observed INTEGER DEFAULT 0,
                    active INTEGER NOT NULL DEFAULT 1,
                    fts tsvector GENERATED ALWAYS AS (
                        to_tsvector('simple',
                            coalesce(content,'') || ' ' ||
                            coalesce(tool_name,'') || ' ' ||
                            coalesce(tool_calls,''))
                    ) STORED
                )"""
            )
            conn.execute(
                f"CREATE TABLE IF NOT EXISTS {_SCHEMA}.state_meta "
                f"(key TEXT PRIMARY KEY, value TEXT)"
            )
            conn.execute(
                f"""CREATE TABLE IF NOT EXISTS {_SCHEMA}.compression_locks (
                    session_id TEXT PRIMARY KEY,
                    holder TEXT NOT NULL,
                    acquired_at DOUBLE PRECISION NOT NULL,
                    expires_at DOUBLE PRECISION NOT NULL
                )"""
            )
            for ddl in (
                f"CREATE INDEX IF NOT EXISTS idx_sessions_source ON {_SCHEMA}.sessions(source)",
                f"CREATE INDEX IF NOT EXISTS idx_sessions_started ON {_SCHEMA}.sessions(started_at DESC)",
                f"CREATE INDEX IF NOT EXISTS idx_messages_session ON {_SCHEMA}.messages(session_id, id)",
                f"CREATE INDEX IF NOT EXISTS idx_messages_session_active ON {_SCHEMA}.messages(session_id, active, id)",
                f"CREATE INDEX IF NOT EXISTS idx_messages_fts ON {_SCHEMA}.messages USING GIN (fts)",
                f"CREATE INDEX IF NOT EXISTS idx_messages_trgm ON {_SCHEMA}.messages USING GIN (content gin_trgm_ops)",
                f"CREATE INDEX IF NOT EXISTS idx_compression_locks_expires ON {_SCHEMA}.compression_locks(expires_at)",
            ):
                conn.execute(ddl)

    # ── sessions ────────────────────────────────────────────────────────────
    def _insert_session_row(self, session_id: str, source: str, **kwargs) -> None:
        cols = {k: kwargs.get(k) for k in _SESSION_INSERT_COLS}
        if cols["model_config"] is not None and not isinstance(cols["model_config"], str):
            cols["model_config"] = json.dumps(cols["model_config"])
        with self._pool.connection() as conn:
            conn.execute(
                f"""INSERT INTO {_SCHEMA}.sessions
                    (id, source, user_id, model, model_config, system_prompt,
                     parent_session_id, cwd, started_at)
                    VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)
                    ON CONFLICT (id) DO NOTHING""",
                (session_id, source, cols["user_id"], cols["model"],
                 cols["model_config"], cols["system_prompt"],
                 cols["parent_session_id"], cols["cwd"], time.time()),
            )

    def create_session(self, session_id: str, source: str, **kwargs) -> str:
        self._insert_session_row(session_id, source, **kwargs)
        return session_id

    def end_session(self, session_id: str, end_reason: str) -> None:
        with self._pool.connection() as conn:
            conn.execute(
                f"UPDATE {_SCHEMA}.sessions SET ended_at=%s, end_reason=%s "
                f"WHERE id=%s AND ended_at IS NULL",
                (time.time(), end_reason, session_id),
            )

    def reopen_session(self, session_id: str) -> None:
        with self._pool.connection() as conn:
            conn.execute(
                f"UPDATE {_SCHEMA}.sessions SET ended_at=NULL, end_reason=NULL WHERE id=%s",
                (session_id,),
            )

    def update_session_cwd(self, session_id: str, cwd: str) -> None:
        if not session_id or not cwd:
            return
        with self._pool.connection() as conn:
            conn.execute(f"UPDATE {_SCHEMA}.sessions SET cwd=%s WHERE id=%s", (cwd, session_id))

    def update_session_meta(self, session_id: str, model_config_json: str,
                            model: Optional[str] = None) -> None:
        with self._pool.connection() as conn:
            conn.execute(
                f"UPDATE {_SCHEMA}.sessions SET model_config=%s, model=COALESCE(%s, model) WHERE id=%s",
                (model_config_json, model, session_id),
            )

    def update_system_prompt(self, session_id: str, system_prompt: str) -> None:
        with self._pool.connection() as conn:
            conn.execute(
                f"UPDATE {_SCHEMA}.sessions SET system_prompt=%s WHERE id=%s",
                (system_prompt, session_id),
            )

    def update_session_model(self, session_id: str, model: str) -> None:
        with self._pool.connection() as conn:
            conn.execute(
                f"UPDATE {_SCHEMA}.sessions SET model=%s WHERE id=%s", (model, session_id))

    def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        with self._pool.connection() as conn:
            return conn.execute(
                f"SELECT * FROM {_SCHEMA}.sessions WHERE id=%s", (session_id,)
            ).fetchone()

    def update_token_counts(self, session_id: str, input_tokens: int = 0,
                            output_tokens: int = 0, model: str = None,
                            cache_read_tokens: int = 0, cache_write_tokens: int = 0,
                            reasoning_tokens: int = 0,
                            estimated_cost_usd: Optional[float] = None,
                            actual_cost_usd: Optional[float] = None,
                            cost_status: Optional[str] = None,
                            cost_source: Optional[str] = None,
                            pricing_version: Optional[str] = None,
                            billing_provider: Optional[str] = None,
                            billing_base_url: Optional[str] = None,
                            billing_mode: Optional[str] = None,
                            api_call_count: int = 0, absolute: bool = False) -> None:
        self._insert_session_row(session_id, "unknown", model=model)
        with self._pool.connection() as conn:
            if absolute:
                conn.execute(
                    f"""UPDATE {_SCHEMA}.sessions SET
                        input_tokens=%s, output_tokens=%s, cache_read_tokens=%s,
                        cache_write_tokens=%s, reasoning_tokens=%s,
                        estimated_cost_usd=COALESCE(%s::double precision,0),
                        actual_cost_usd=CASE WHEN %s::double precision IS NULL
                            THEN actual_cost_usd ELSE %s::double precision END,
                        cost_status=COALESCE(%s::text,cost_status),
                        cost_source=COALESCE(%s::text,cost_source),
                        pricing_version=COALESCE(%s::text,pricing_version),
                        billing_provider=COALESCE(billing_provider,%s::text),
                        billing_base_url=COALESCE(billing_base_url,%s::text),
                        billing_mode=COALESCE(billing_mode,%s::text),
                        model=COALESCE(model,%s::text), api_call_count=%s
                        WHERE id=%s""",
                    (input_tokens, output_tokens, cache_read_tokens, cache_write_tokens,
                     reasoning_tokens, estimated_cost_usd, actual_cost_usd, actual_cost_usd,
                     cost_status, cost_source, pricing_version, billing_provider,
                     billing_base_url, billing_mode, model, api_call_count, session_id),
                )
            else:
                conn.execute(
                    f"""UPDATE {_SCHEMA}.sessions SET
                        input_tokens=input_tokens+%s, output_tokens=output_tokens+%s,
                        cache_read_tokens=cache_read_tokens+%s,
                        cache_write_tokens=cache_write_tokens+%s,
                        reasoning_tokens=reasoning_tokens+%s,
                        estimated_cost_usd=COALESCE(estimated_cost_usd,0)+COALESCE(%s::double precision,0),
                        actual_cost_usd=CASE WHEN %s::double precision IS NULL THEN actual_cost_usd
                            ELSE COALESCE(actual_cost_usd,0)+%s::double precision END,
                        cost_status=COALESCE(%s::text,cost_status),
                        cost_source=COALESCE(%s::text,cost_source),
                        pricing_version=COALESCE(%s::text,pricing_version),
                        billing_provider=COALESCE(billing_provider,%s::text),
                        billing_base_url=COALESCE(billing_base_url,%s::text),
                        billing_mode=COALESCE(billing_mode,%s::text),
                        model=COALESCE(model,%s::text), api_call_count=api_call_count+%s
                        WHERE id=%s""",
                    (input_tokens, output_tokens, cache_read_tokens, cache_write_tokens,
                     reasoning_tokens, estimated_cost_usd, actual_cost_usd, actual_cost_usd,
                     cost_status, cost_source, pricing_version, billing_provider,
                     billing_base_url, billing_mode, model, api_call_count, session_id),
                )

    # ── messages ────────────────────────────────────────────────────────────
    def append_message(self, session_id: str, role: str, content: str = None,
                       tool_name: str = None, tool_calls: Any = None,
                       tool_call_id: str = None, token_count: int = None,
                       finish_reason: str = None, reasoning: str = None,
                       reasoning_content: str = None, reasoning_details: Any = None,
                       codex_reasoning_items: Any = None, codex_message_items: Any = None,
                       platform_message_id: str = None, observed: bool = False) -> int:
        reasoning_details_json = json.dumps(reasoning_details) if reasoning_details else None
        codex_items_json = json.dumps(codex_reasoning_items) if codex_reasoning_items else None
        codex_message_items_json = json.dumps(codex_message_items) if codex_message_items else None
        tool_calls_json = json.dumps(tool_calls) if tool_calls else None
        stored_content = _encode_content(content)
        num_tool_calls = (len(tool_calls) if isinstance(tool_calls, list) else 1) if tool_calls is not None else 0

        with self._pool.connection() as conn, conn.transaction():
            row = conn.execute(
                f"""INSERT INTO {_SCHEMA}.messages
                    (session_id, role, content, tool_call_id, tool_calls, tool_name,
                     timestamp, token_count, finish_reason, reasoning, reasoning_content,
                     reasoning_details, codex_reasoning_items, codex_message_items,
                     platform_message_id, observed)
                    VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                    RETURNING id""",
                (session_id, role, stored_content, tool_call_id, tool_calls_json,
                 tool_name, time.time(), token_count, finish_reason, reasoning,
                 reasoning_content, reasoning_details_json, codex_items_json,
                 codex_message_items_json, platform_message_id, 1 if observed else 0),
            ).fetchone()
            msg_id = row["id"]
            if num_tool_calls > 0:
                conn.execute(
                    f"UPDATE {_SCHEMA}.sessions SET message_count=message_count+1, "
                    f"tool_call_count=tool_call_count+%s WHERE id=%s",
                    (num_tool_calls, session_id),
                )
            else:
                conn.execute(
                    f"UPDATE {_SCHEMA}.sessions SET message_count=message_count+1 WHERE id=%s",
                    (session_id,),
                )
            return msg_id

    def get_messages(self, session_id: str, include_inactive: bool = False) -> List[Dict[str, Any]]:
        active_clause = "" if include_inactive else " AND active = 1"
        with self._pool.connection() as conn:
            rows = conn.execute(
                f"SELECT * FROM {_SCHEMA}.messages WHERE session_id=%s{active_clause} ORDER BY id",
                (session_id,),
            ).fetchall()
        for msg in rows:
            msg.pop("fts", None)  # internal column, not part of the SQLite surface
            if "content" in msg:
                msg["content"] = _decode_content(msg["content"])
            if msg.get("tool_calls"):
                try:
                    msg["tool_calls"] = json.loads(msg["tool_calls"])
                except (json.JSONDecodeError, TypeError):
                    msg["tool_calls"] = []
        return rows

    def replace_messages(self, session_id: str, messages: List[Dict[str, Any]]) -> None:
        with self._pool.connection() as conn, conn.transaction():
            conn.execute(f"DELETE FROM {_SCHEMA}.messages WHERE session_id=%s", (session_id,))
            for m in messages:
                conn.execute(
                    f"""INSERT INTO {_SCHEMA}.messages
                        (session_id, role, content, tool_call_id, tool_calls, tool_name,
                         timestamp, token_count, finish_reason, active)
                        VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)""",
                    (session_id, m.get("role"), _encode_content(m.get("content")),
                     m.get("tool_call_id"),
                     json.dumps(m["tool_calls"]) if m.get("tool_calls") else None,
                     m.get("tool_name"), m.get("timestamp") or time.time(),
                     m.get("token_count"), m.get("finish_reason"),
                     1 if m.get("active", 1) else 0),
                )

    def clear_messages(self, session_id: str) -> None:
        with self._pool.connection() as conn:
            conn.execute(f"DELETE FROM {_SCHEMA}.messages WHERE session_id=%s", (session_id,))

    def message_count(self, session_id: str = None) -> int:
        with self._pool.connection() as conn:
            if session_id:
                row = conn.execute(
                    f"SELECT COUNT(*) AS c FROM {_SCHEMA}.messages WHERE session_id=%s AND active=1",
                    (session_id,),
                ).fetchone()
            else:
                row = conn.execute(
                    f"SELECT COUNT(*) AS c FROM {_SCHEMA}.messages WHERE active=1"
                ).fetchone()
            return row["c"] if row else 0

    def session_count(self) -> int:
        with self._pool.connection() as conn:
            row = conn.execute(f"SELECT COUNT(*) AS c FROM {_SCHEMA}.sessions").fetchone()
            return row["c"] if row else 0

    # ── full-text search (FTS5 -> tsvector + pg_trgm) ─────────────────────────
    def search_messages(self, query: str, source_filter: List[str] = None,
                        exclude_sources: List[str] = None, role_filter: List[str] = None,
                        limit: int = 20, offset: int = 0, sort: str = None,
                        include_inactive: bool = False) -> List[Dict[str, Any]]:
        if not query or not query.strip():
            return []
        where = ["(m.fts @@ websearch_to_tsquery('simple', %s) OR m.content ILIKE %s)"]
        params: list = [query, f"%{query}%"]
        if not include_inactive:
            where.append("m.active = 1")
        if source_filter:
            where.append("s.source = ANY(%s)"); params.append(list(source_filter))
        if exclude_sources:
            where.append("NOT (s.source = ANY(%s))"); params.append(list(exclude_sources))
        if role_filter:
            where.append("m.role = ANY(%s)"); params.append(list(role_filter))
        order = ("m.timestamp DESC" if sort == "newest"
                 else "m.timestamp ASC" if sort == "oldest"
                 else "ts_rank(m.fts, websearch_to_tsquery('simple', %s)) DESC")
        if sort not in ("newest", "oldest"):
            params.append(query)
        params.extend([limit, offset])
        sql = (f"SELECT m.*, s.source FROM {_SCHEMA}.messages m "
               f"JOIN {_SCHEMA}.sessions s ON s.id = m.session_id "
               f"WHERE {' AND '.join(where)} ORDER BY {order} LIMIT %s OFFSET %s")
        with self._pool.connection() as conn:
            rows = conn.execute(sql, params).fetchall()
        for msg in rows:
            msg.pop("fts", None)
            if "content" in msg:
                msg["content"] = _decode_content(msg["content"])
        return rows

    # ── state_meta ────────────────────────────────────────────────────────────
    def get_meta(self, key: str) -> Optional[str]:
        with self._pool.connection() as conn:
            row = conn.execute(
                f"SELECT value FROM {_SCHEMA}.state_meta WHERE key=%s", (key,)
            ).fetchone()
            return row["value"] if row else None

    def set_meta(self, key: str, value: str) -> None:
        with self._pool.connection() as conn:
            conn.execute(
                f"INSERT INTO {_SCHEMA}.state_meta (key, value) VALUES (%s,%s) "
                f"ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value",
                (key, value),
            )

    # ── compression locks (same protocol as SQLite: reclaim-expired + insert) ─
    def try_acquire_compression_lock(self, session_id: str, holder: str,
                                     ttl_seconds: float = 300.0) -> bool:
        if not session_id:
            return False
        now = time.time()
        expires_at = now + ttl_seconds
        try:
            with self._pool.connection() as conn, conn.transaction():
                conn.execute(
                    f"DELETE FROM {_SCHEMA}.compression_locks "
                    f"WHERE session_id=%s AND expires_at < %s",
                    (session_id, now),
                )
                conn.execute(
                    f"INSERT INTO {_SCHEMA}.compression_locks "
                    f"(session_id, holder, acquired_at, expires_at) VALUES (%s,%s,%s,%s) "
                    f"ON CONFLICT (session_id) DO NOTHING",
                    (session_id, holder, now, expires_at),
                )
                row = conn.execute(
                    f"SELECT holder FROM {_SCHEMA}.compression_locks WHERE session_id=%s",
                    (session_id,),
                ).fetchone()
                return row is not None and row["holder"] == holder
        except Exception as exc:  # fail-open like the SQLite store
            logger.warning("try_acquire_compression_lock(%s) failed: %s", session_id, exc)
            return False

    def release_compression_lock(self, session_id: str, holder: str) -> None:
        if not session_id:
            return
        try:
            with self._pool.connection() as conn:
                conn.execute(
                    f"DELETE FROM {_SCHEMA}.compression_locks WHERE session_id=%s AND holder=%s",
                    (session_id, holder),
                )
        except Exception as exc:
            logger.warning("release_compression_lock(%s) failed: %s", session_id, exc)

    def get_compression_lock_holder(self, session_id: str) -> Optional[str]:
        if not session_id:
            return None
        with self._pool.connection() as conn:
            row = conn.execute(
                f"SELECT holder FROM {_SCHEMA}.compression_locks "
                f"WHERE session_id=%s AND expires_at >= %s",
                (session_id, time.time()),
            ).fetchone()
            return row["holder"] if row else None

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self._pool.close()
        except Exception:
            pass
