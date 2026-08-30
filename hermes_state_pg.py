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
A small ``STATEMENT_OVERRIDES`` table replaces whole statements that differ by
*scoping* rather than syntax and so cannot be translated by pattern (today:
the ``session_model_usage`` upsert, whose bare ``DO UPDATE SET`` self-
references are ambiguous on Postgres).

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
from collections import deque
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Collection, Dict, List, Optional, TypeVar

import hermes_state
from hermes_state import SessionDB

logger = logging.getLogger(__name__)

T = TypeVar("T")

ENV_VAR = "HERMES_STATE_STORE_DSN"

_SCHEMA = "hermes_state"

# ── Rebase-drift guard ──────────────────────────────────────────────────────
# The SCHEMA_VERSION and the sha256 of the upstream SQLite DDL that this
# Postgres adapter was audited against. If an upstream rebase bumps the
# version or edits SCHEMA_SQL/DEFERRED_INDEX_SQL, PgSessionDB must refuse to
# boot until a human re-audits PG_SCHEMA_SQL (and ships an explicit safe
# migration/cutover when needed), then updates these constants.
EXPECTED_SCHEMA_VERSION = 26
EXPECTED_SCHEMA_SURFACE_SHA256 = (
    "cd2cb9ee351693e62e9dc8e425885a4a08148551d9577d506f4a11be4a715d5f"
)
# v2026.8.27 intentionally keeps SCHEMA_VERSION at 26 while adding one
# persisted marker column.  The compatibility bridge accepts that one exact
# forward surface so the current image remains a valid rollback target after
# the additive migration.  It does not use or write the marker itself.
FORWARD_SCHEMA_SURFACE_SHA256 = (
    "4fcc7bb46a26f9d3ad47322dcdd24ae972fb349b66ca715fcdcb2ae16ba39ca6"
)
_V26_MIGRATION_ADVISORY_TIMEOUT = "30000ms"
_V26_MIGRATION_LOCK_TIMEOUT = "5000ms"
_V26_MIGRATION_STATEMENT_TIMEOUT = "30000ms"
_V22_SCHEMA_SURFACE_SHA256 = (
    "ffb802aede5aab2e95d1eb46188864c11b4b8e290c538ada64c06b9a14747654"
)

# ── Audited predecessor markers (AE-182) ───────────────────────────────────
# v22 is the only audited migration predecessor. This release adds a durable
# cross-task turn lease, so the transition is deliberately drain-only: runtime
# boot observes and refuses v22; the explicit migration command owns all DDL.
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


# ── Postgres DDL (mirrors hermes_state.SCHEMA_SQL @ v26) ───────────────────
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
    f"""CREATE TABLE IF NOT EXISTS {_SCHEMA}.system_prompts (
        hash TEXT PRIMARY KEY,
        prompt TEXT NOT NULL
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
        system_prompt_hash TEXT,
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
        git_metadata_generation BIGINT NOT NULL DEFAULT 0,
        billing_provider TEXT,
        billing_base_url TEXT,
        billing_mode TEXT,
        estimated_cost_usd DOUBLE PRECISION,
        actual_cost_usd DOUBLE PRECISION,
        cost_status TEXT,
        cost_source TEXT,
        pricing_version TEXT,
        title TEXT,
        title_source TEXT,
        last_activity_at DOUBLE PRECISION,
        last_activity_description TEXT,
        last_activity_provenance TEXT,
        api_call_count BIGINT DEFAULT 0,
        handoff_state TEXT,
        handoff_platform TEXT,
        handoff_error TEXT,
        compression_failure_cooldown_until DOUBLE PRECISION,
        compression_failure_error TEXT,
        compression_fallback_streak BIGINT NOT NULL DEFAULT 0,
        compression_ineffective_count BIGINT NOT NULL DEFAULT 0,
        profile_name TEXT,
        rewind_count BIGINT NOT NULL DEFAULT 0,
        archived BIGINT NOT NULL DEFAULT 0,
        pinned BIGINT NOT NULL DEFAULT 0,
        hidden BIGINT NOT NULL DEFAULT 0,
        last_read_at DOUBLE PRECISION,
        FOREIGN KEY (parent_session_id) REFERENCES {_SCHEMA}.sessions(id)
            DEFERRABLE INITIALLY IMMEDIATE,
        FOREIGN KEY (system_prompt_hash) REFERENCES {_SCHEMA}.system_prompts(hash)
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
        effect_disposition TEXT,
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
        compacted BIGINT NOT NULL DEFAULT 0,
        api_content TEXT,
        display_kind TEXT,
        display_metadata TEXT
    )""",
    # v20/v22 (upstream cb7f6bbb2 + eb6aa0360): per-model, per-task usage
    # attribution. Created at the terminal v22 shape — ``task`` is part of the
    # PRIMARY KEY, which is what ``_record_model_usage``'s
    # ``ON CONFLICT (session_id, model, billing_provider, billing_base_url,
    # billing_mode, task) DO UPDATE`` arbitrates on. SQLite needed a
    # rename/rebuild dance for that PK change; Postgres never had a v20/v21
    # shape to rebuild (the store was cut over at v19, before this table
    # existed), so the terminal shape is created directly.
    f"""CREATE TABLE IF NOT EXISTS {_SCHEMA}.session_model_usage (
        session_id TEXT NOT NULL REFERENCES {_SCHEMA}.sessions(id)
            ON DELETE CASCADE DEFERRABLE INITIALLY IMMEDIATE,
        model TEXT NOT NULL,
        billing_provider TEXT NOT NULL DEFAULT '',
        billing_base_url TEXT NOT NULL DEFAULT '',
        billing_mode TEXT NOT NULL DEFAULT '',
        task TEXT NOT NULL DEFAULT '',
        api_call_count BIGINT NOT NULL DEFAULT 0,
        input_tokens BIGINT NOT NULL DEFAULT 0,
        output_tokens BIGINT NOT NULL DEFAULT 0,
        cache_read_tokens BIGINT NOT NULL DEFAULT 0,
        cache_write_tokens BIGINT NOT NULL DEFAULT 0,
        reasoning_tokens BIGINT NOT NULL DEFAULT 0,
        estimated_cost_usd DOUBLE PRECISION NOT NULL DEFAULT 0,
        actual_cost_usd DOUBLE PRECISION NOT NULL DEFAULT 0,
        cost_status TEXT,
        cost_source TEXT,
        first_seen DOUBLE PRECISION,
        last_seen DOUBLE PRECISION,
        PRIMARY KEY (session_id, model, billing_provider, billing_base_url,
                     billing_mode, task)
    )""",
    # v21 (upstream d0e9a42ce). LIVE on Postgres since AE-183:
    # ``tools/async_delegation.py`` now persists through the same SessionDB
    # dispatch as everything else instead of opening its own sqlite3
    # connection to the EFS ``state.db``, which made it a second raw writer on
    # a file two Fargate tasks share during a blue/green drain.
    #
    # ``owner_instance`` is IdeaRoom-owned and deliberately absent from
    # upstream's ``hermes_state.SCHEMA_SQL`` — adding it there would move the
    # hashed schema surface and put every future rebase through the
    # re-audit guard for a column upstream does not have. The sqlite side
    # reconciles it in ``async_delegation._connect()``; here it is created
    # eagerly below and expanded onto existing databases by PG_EXPAND_SQL.
    # It scopes the recovery pass's pid-liveness test to the owner's own PID
    # namespace: without it, the incoming task resolves the outgoing task's
    # pids against ITS namespace and buries live delegations as ``unknown``.
    f"""CREATE TABLE IF NOT EXISTS {_SCHEMA}.async_delegations (
        delegation_id TEXT PRIMARY KEY,
        origin_session TEXT NOT NULL,
        origin_ui_session_id TEXT NOT NULL DEFAULT '',
        parent_session_id TEXT,
        state TEXT NOT NULL,
        dispatched_at DOUBLE PRECISION NOT NULL,
        completed_at DOUBLE PRECISION,
        updated_at DOUBLE PRECISION NOT NULL,
        event_json TEXT,
        result_json TEXT,
        delivery_state TEXT NOT NULL DEFAULT 'pending',
        delivery_attempts BIGINT NOT NULL DEFAULT 0,
        delivered_at DOUBLE PRECISION,
        owner_pid BIGINT,
        owner_started_at BIGINT,
        task_json TEXT,
        delivery_claim TEXT,
        delivery_claimed_at DOUBLE PRECISION,
        owner_instance TEXT,
        origin_session_id TEXT NOT NULL DEFAULT ''
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
    f"""CREATE TABLE IF NOT EXISTS {_SCHEMA}.gateway_hygiene_state (
        session_key TEXT PRIMARY KEY,
        failure_streak BIGINT NOT NULL DEFAULT 0
    )""",
    f"""CREATE TABLE IF NOT EXISTS {_SCHEMA}.compression_locks (
        session_id TEXT PRIMARY KEY,
        holder TEXT NOT NULL,
        acquired_at DOUBLE PRECISION NOT NULL,
        expires_at DOUBLE PRECISION NOT NULL
    )""",
    f"""CREATE TABLE IF NOT EXISTS {_SCHEMA}.session_turn_leases (
        conversation_id TEXT PRIMARY KEY,
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
    f"CREATE INDEX IF NOT EXISTS idx_messages_session_id ON {_SCHEMA}.messages(session_id, id)",
    f"CREATE INDEX IF NOT EXISTS idx_messages_assistant_calls_by_session "
    f"ON {_SCHEMA}.messages(session_id) "
    f"WHERE role = 'assistant' AND tool_calls IS NOT NULL",
    f"CREATE INDEX IF NOT EXISTS idx_compression_locks_expires ON {_SCHEMA}.compression_locks(expires_at)",
    f"CREATE INDEX IF NOT EXISTS idx_session_turn_leases_expires "
    f"ON {_SCHEMA}.session_turn_leases(expires_at)",
    f"CREATE INDEX IF NOT EXISTS idx_session_model_usage_session "
    f"ON {_SCHEMA}.session_model_usage(session_id)",
    f"CREATE INDEX IF NOT EXISTS idx_session_model_usage_model "
    f"ON {_SCHEMA}.session_model_usage(model)",
    f"CREATE INDEX IF NOT EXISTS idx_async_delegations_delivery "
    f"ON {_SCHEMA}.async_delegations(delivery_state, completed_at)",
    f"CREATE INDEX IF NOT EXISTS idx_messages_session_active "
    f"ON {_SCHEMA}.messages(session_id, active, timestamp)",
    # Mirrors upstream's legacy-SQLite repair index. Postgres declares active
    # NOT NULL, so this remains empty; retaining it keeps the audited schema
    # surface aligned without requiring a data migration.
    f"CREATE INDEX IF NOT EXISTS idx_messages_active_null "
    f"ON {_SCHEMA}.messages(active) WHERE active IS NULL",
    f"CREATE INDEX IF NOT EXISTS idx_sessions_session_key "
    f"ON {_SCHEMA}.sessions(session_key, started_at DESC)",
    f"CREATE INDEX IF NOT EXISTS idx_sessions_gateway_peer "
    f"ON {_SCHEMA}.sessions(source, user_id, chat_id, chat_type, thread_id, started_at DESC)",
    f"CREATE INDEX IF NOT EXISTS idx_sessions_handoff_state "
    f"ON {_SCHEMA}.sessions(handoff_state, started_at)",
    f"CREATE INDEX IF NOT EXISTS idx_sessions_system_prompt_hash "
    f"ON {_SCHEMA}.sessions(system_prompt_hash)",
    f"CREATE UNIQUE INDEX IF NOT EXISTS idx_sessions_title_unique "
    f"ON {_SCHEMA}.sessions(title) WHERE title IS NOT NULL",
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

# ── Explicit drain-only column expansion ──────────────────────────────────
# ``CREATE TABLE IF NOT EXISTS`` never adds columns to a table that already
# exists, so the drain-only v22→v26 migration applies these idempotent ALTERs
# explicitly. Runtime boot never executes them: v22 tasks cannot honor v26
# turn leases and therefore must not coexist with a widened store.
PG_EXPAND_SQL: List[str] = [
    f"ALTER TABLE {_SCHEMA}.sessions ADD COLUMN IF NOT EXISTS "
    "compression_fallback_streak BIGINT NOT NULL DEFAULT 0",
    f"ALTER TABLE {_SCHEMA}.sessions ADD COLUMN IF NOT EXISTS profile_name TEXT",
    f"ALTER TABLE {_SCHEMA}.messages ADD COLUMN IF NOT EXISTS effect_disposition TEXT",
    f"ALTER TABLE {_SCHEMA}.messages ADD COLUMN IF NOT EXISTS api_content TEXT",
    f"ALTER TABLE {_SCHEMA}.sessions ADD COLUMN IF NOT EXISTS system_prompt_hash TEXT",
    f"ALTER TABLE {_SCHEMA}.sessions ADD COLUMN IF NOT EXISTS "
    "git_metadata_generation BIGINT NOT NULL DEFAULT 0",
    f"ALTER TABLE {_SCHEMA}.sessions ADD COLUMN IF NOT EXISTS title_source TEXT",
    f"ALTER TABLE {_SCHEMA}.sessions ADD COLUMN IF NOT EXISTS last_activity_at DOUBLE PRECISION",
    f"ALTER TABLE {_SCHEMA}.sessions ADD COLUMN IF NOT EXISTS last_activity_description TEXT",
    f"ALTER TABLE {_SCHEMA}.sessions ADD COLUMN IF NOT EXISTS last_activity_provenance TEXT",
    f"ALTER TABLE {_SCHEMA}.sessions ADD COLUMN IF NOT EXISTS "
    "compression_ineffective_count BIGINT NOT NULL DEFAULT 0",
    f"ALTER TABLE {_SCHEMA}.sessions ADD COLUMN IF NOT EXISTS pinned BIGINT NOT NULL DEFAULT 0",
    f"ALTER TABLE {_SCHEMA}.sessions ADD COLUMN IF NOT EXISTS hidden BIGINT NOT NULL DEFAULT 0",
    f"ALTER TABLE {_SCHEMA}.sessions ADD COLUMN IF NOT EXISTS last_read_at DOUBLE PRECISION",
    f"ALTER TABLE {_SCHEMA}.messages ADD COLUMN IF NOT EXISTS display_kind TEXT",
    f"ALTER TABLE {_SCHEMA}.messages ADD COLUMN IF NOT EXISTS display_metadata TEXT",
    # AE-183 owner identity (see the async_delegations DDL above). Legacy rows
    # keep NULL, which the recovery pass reads as "legacy, same host".
    f"ALTER TABLE {_SCHEMA}.async_delegations ADD COLUMN IF NOT EXISTS "
    "owner_instance TEXT",
    f"ALTER TABLE {_SCHEMA}.async_delegations ADD COLUMN IF NOT EXISTS "
    "origin_session_id TEXT NOT NULL DEFAULT ''",
]

# Online-safe v26 surface expansion used by the explicit AE-240 migration.
# A constant-default column addition is backward-compatible with the current
# runtime.  Runtime boot remains observation-only; only the migration entrypoint
# executes this statement.
PG_V26_FORWARD_SURFACE_SQL = (
    f"ALTER TABLE {_SCHEMA}.messages ADD COLUMN IF NOT EXISTS "
    "_compressed_summary BIGINT NOT NULL DEFAULT 0"
)

# v25 adds a content-addressed system-prompt reference. Postgres lacks
# ``ALTER TABLE ... ADD CONSTRAINT IF NOT EXISTS``, so the explicit migration
# guards the semantic constraint in a DO block. Fresh databases get the same
# constraint inline in PG_SCHEMA_SQL.
PG_V26_CONSTRAINT_SQL: List[str] = [
    f"""DO $$
    BEGIN
        IF NOT EXISTS (
            SELECT 1
            FROM pg_constraint
            WHERE conname = 'sessions_system_prompt_hash_fkey'
              AND conrelid = '{_SCHEMA}.sessions'::regclass
        ) THEN
            ALTER TABLE {_SCHEMA}.sessions
            ADD CONSTRAINT sessions_system_prompt_hash_fkey
            FOREIGN KEY (system_prompt_hash)
            REFERENCES {_SCHEMA}.system_prompts(hash)
            DEFERRABLE INITIALLY IMMEDIATE;
        END IF;
    END $$""",
]

# The v26 unique title index cannot be created until legacy duplicates are
# repaired. Retain the newest session deterministically (started_at, then id)
# and preserve every older session with a NULL alias.
PG_V26_DEDUPLICATE_TITLES_SQL = f"""
    WITH ranked AS (
        SELECT id,
               ROW_NUMBER() OVER (
                   PARTITION BY title
                   ORDER BY started_at DESC, id DESC
               ) AS title_rank
        FROM {_SCHEMA}.sessions
        WHERE title IS NOT NULL
    )
    UPDATE {_SCHEMA}.sessions AS session
    SET title = NULL
    FROM ranked
    WHERE session.id = ranked.id
      AND ranked.title_rank > 1
"""

_REQUIRED_COLUMNS: Dict[str, frozenset[str]] = {
    "schema_version": frozenset({"version"}),
    "system_prompts": frozenset({"hash", "prompt"}),
    "sessions": frozenset(
        {
            "id", "source", "user_id", "session_key", "chat_id",
            "chat_type", "thread_id", "display_name", "origin_json",
            "expiry_finalized", "model", "model_config", "system_prompt",
            "system_prompt_hash", "parent_session_id", "started_at",
            "ended_at", "end_reason", "message_count", "tool_call_count",
            "input_tokens", "output_tokens", "cache_read_tokens",
            "cache_write_tokens", "reasoning_tokens", "cwd", "git_branch",
            "git_repo_root", "git_metadata_generation", "billing_provider",
            "billing_base_url", "billing_mode", "estimated_cost_usd",
            "actual_cost_usd", "cost_status", "cost_source",
            "pricing_version", "title", "title_source", "last_activity_at",
            "last_activity_description", "last_activity_provenance",
            "api_call_count", "handoff_state", "handoff_platform",
            "handoff_error", "compression_failure_cooldown_until",
            "compression_failure_error", "compression_fallback_streak",
            "compression_ineffective_count", "profile_name", "rewind_count",
            "archived", "pinned", "hidden", "last_read_at",
        }
    ),
    "messages": frozenset(
        {
            "id", "session_id", "role", "content", "tool_call_id",
            "tool_calls", "tool_name", "effect_disposition", "timestamp",
            "token_count", "finish_reason", "reasoning",
            "reasoning_content", "reasoning_details",
            "codex_reasoning_items", "codex_message_items",
            "platform_message_id", "observed", "active", "compacted",
            "api_content", "display_kind", "display_metadata",
        }
    ),
    "session_model_usage": frozenset(
        {
            "session_id", "model", "billing_provider", "billing_base_url",
            "billing_mode", "task", "api_call_count", "input_tokens",
            "output_tokens", "cache_read_tokens", "cache_write_tokens",
            "reasoning_tokens", "estimated_cost_usd", "actual_cost_usd",
            "cost_status", "cost_source", "first_seen", "last_seen",
        }
    ),
    "async_delegations": frozenset(
        {
            "delegation_id", "origin_session", "origin_ui_session_id",
            "parent_session_id", "state", "dispatched_at", "completed_at",
            "updated_at", "event_json", "result_json", "delivery_state",
            "delivery_attempts", "delivered_at", "owner_pid",
            "owner_started_at", "task_json", "delivery_claim",
            "delivery_claimed_at", "owner_instance", "origin_session_id",
        }
    ),
    "state_meta": frozenset({"key", "value"}),
    "gateway_routing": frozenset(
        {"scope", "session_key", "entry_json", "updated_at"}
    ),
    "gateway_hygiene_state": frozenset({"session_key", "failure_streak"}),
    "compression_locks": frozenset(
        {"session_id", "holder", "acquired_at", "expires_at"}
    ),
    "session_turn_leases": frozenset(
        {"conversation_id", "holder", "acquired_at", "expires_at"}
    ),
    "telegram_dm_topic_mode": frozenset(
        {
            "chat_id", "user_id", "enabled", "activated_at", "updated_at",
            "has_topics_enabled", "allows_users_to_create_topics",
            "capability_checked_at", "intro_message_id", "pinned_message_id",
        }
    ),
    "telegram_dm_topic_bindings": frozenset(
        {
            "chat_id", "thread_id", "user_id", "session_key", "session_id",
            "managed_mode", "linked_at", "updated_at",
        }
    ),
}

_FORWARD_REQUIRED_COLUMNS: Dict[str, frozenset[str]] = {
    **_REQUIRED_COLUMNS,
    "messages": _REQUIRED_COLUMNS["messages"] | {"_compressed_summary"},
}

# Exact catalog accepted by the one-shot migration preflight. These sets
# describe the fork's audited v22 Postgres surface, including owner_instance
# (the IdeaRoom async-delegation extension) and excluding every v23-v26
# addition. A marker alone is not proof of shape: an interrupted/manual DDL
# attempt can leave the marker at v22 while the catalog is already widened.
_V22_REQUIRED_COLUMNS: Dict[str, frozenset[str]] = {
    "schema_version": frozenset({"version"}),
    "sessions": frozenset(
        {
            "id",
            "source",
            "user_id",
            "session_key",
            "chat_id",
            "chat_type",
            "thread_id",
            "display_name",
            "origin_json",
            "expiry_finalized",
            "model",
            "model_config",
            "system_prompt",
            "parent_session_id",
            "started_at",
            "ended_at",
            "end_reason",
            "message_count",
            "tool_call_count",
            "input_tokens",
            "output_tokens",
            "cache_read_tokens",
            "cache_write_tokens",
            "reasoning_tokens",
            "cwd",
            "git_branch",
            "git_repo_root",
            "billing_provider",
            "billing_base_url",
            "billing_mode",
            "estimated_cost_usd",
            "actual_cost_usd",
            "cost_status",
            "cost_source",
            "pricing_version",
            "title",
            "api_call_count",
            "handoff_state",
            "handoff_platform",
            "handoff_error",
            "compression_failure_cooldown_until",
            "compression_failure_error",
            "compression_fallback_streak",
            "profile_name",
            "rewind_count",
            "archived",
        }
    ),
    "messages": frozenset(
        {
            "id",
            "session_id",
            "role",
            "content",
            "tool_call_id",
            "tool_calls",
            "tool_name",
            "effect_disposition",
            "timestamp",
            "token_count",
            "finish_reason",
            "reasoning",
            "reasoning_content",
            "reasoning_details",
            "codex_reasoning_items",
            "codex_message_items",
            "platform_message_id",
            "observed",
            "active",
            "compacted",
            "api_content",
        }
    ),
    "session_model_usage": frozenset(
        {
            "session_id",
            "model",
            "billing_provider",
            "billing_base_url",
            "billing_mode",
            "task",
            "api_call_count",
            "input_tokens",
            "output_tokens",
            "cache_read_tokens",
            "cache_write_tokens",
            "reasoning_tokens",
            "estimated_cost_usd",
            "actual_cost_usd",
            "cost_status",
            "cost_source",
            "first_seen",
            "last_seen",
        }
    ),
    "async_delegations": frozenset(
        {
            "delegation_id",
            "origin_session",
            "origin_ui_session_id",
            "parent_session_id",
            "state",
            "dispatched_at",
            "completed_at",
            "updated_at",
            "event_json",
            "result_json",
            "delivery_state",
            "delivery_attempts",
            "delivered_at",
            "owner_pid",
            "owner_started_at",
            "task_json",
            "delivery_claim",
            "delivery_claimed_at",
            "owner_instance",
        }
    ),
    "state_meta": frozenset({"key", "value"}),
    "gateway_routing": frozenset(
        {"scope", "session_key", "entry_json", "updated_at"}
    ),
    "compression_locks": frozenset(
        {"session_id", "holder", "acquired_at", "expires_at"}
    ),
    "telegram_dm_topic_mode": frozenset(
        {
            "chat_id",
            "user_id",
            "enabled",
            "activated_at",
            "updated_at",
            "has_topics_enabled",
            "allows_users_to_create_topics",
            "capability_checked_at",
            "intro_message_id",
            "pinned_message_id",
        }
    ),
    "telegram_dm_topic_bindings": frozenset(
        {
            "chat_id",
            "thread_id",
            "user_id",
            "session_key",
            "session_id",
            "managed_mode",
            "linked_at",
            "updated_at",
        }
    ),
}

@dataclass(frozen=True)
class _ColumnSpec:
    """Semantic shape exposed by ``information_schema.columns``."""

    data_type: str
    nullable: bool
    default: Optional[str] = None
    identity_generation: Optional[str] = None


_BIGINT_COLUMNS = frozenset(
    {
        ("schema_version", "version"),
        *(
            ("sessions", column)
            for column in (
                "expiry_finalized",
                "message_count",
                "tool_call_count",
                "input_tokens",
                "output_tokens",
                "cache_read_tokens",
                "cache_write_tokens",
                "reasoning_tokens",
                "git_metadata_generation",
                "api_call_count",
                "compression_fallback_streak",
                "compression_ineffective_count",
                "rewind_count",
                "archived",
                "pinned",
                "hidden",
            )
        ),
        *(
            ("messages", column)
            for column in (
                "id",
                "token_count",
                "observed",
                "_compressed_summary",
                "active",
                "compacted",
            )
        ),
        *(
            ("session_model_usage", column)
            for column in (
                "api_call_count",
                "input_tokens",
                "output_tokens",
                "cache_read_tokens",
                "cache_write_tokens",
                "reasoning_tokens",
            )
        ),
        ("async_delegations", "delivery_attempts"),
        ("async_delegations", "owner_pid"),
        ("async_delegations", "owner_started_at"),
        ("gateway_hygiene_state", "failure_streak"),
        ("telegram_dm_topic_mode", "enabled"),
        ("telegram_dm_topic_mode", "has_topics_enabled"),
        ("telegram_dm_topic_mode", "allows_users_to_create_topics"),
    }
)

_DOUBLE_PRECISION_COLUMNS = frozenset(
    {
        *(
            ("sessions", column)
            for column in (
                "started_at",
                "ended_at",
                "estimated_cost_usd",
                "actual_cost_usd",
                "last_activity_at",
                "compression_failure_cooldown_until",
                "last_read_at",
            )
        ),
        ("messages", "timestamp"),
        *(
            ("session_model_usage", column)
            for column in (
                "estimated_cost_usd",
                "actual_cost_usd",
                "first_seen",
                "last_seen",
            )
        ),
        *(
            ("async_delegations", column)
            for column in (
                "dispatched_at",
                "completed_at",
                "updated_at",
                "delivered_at",
                "delivery_claimed_at",
            )
        ),
        ("gateway_routing", "updated_at"),
        ("compression_locks", "acquired_at"),
        ("compression_locks", "expires_at"),
        ("session_turn_leases", "acquired_at"),
        ("session_turn_leases", "expires_at"),
        ("telegram_dm_topic_mode", "activated_at"),
        ("telegram_dm_topic_mode", "updated_at"),
        ("telegram_dm_topic_mode", "capability_checked_at"),
        ("telegram_dm_topic_bindings", "linked_at"),
        ("telegram_dm_topic_bindings", "updated_at"),
    }
)

_NOT_NULL_COLUMNS = frozenset(
    {
        ("schema_version", "version"),
        ("system_prompts", "hash"),
        ("system_prompts", "prompt"),
        *(
            ("sessions", column)
            for column in (
                "id",
                "source",
                "started_at",
                "git_metadata_generation",
                "compression_fallback_streak",
                "compression_ineffective_count",
                "rewind_count",
                "archived",
                "pinned",
                "hidden",
            )
        ),
        *(
            ("messages", column)
            for column in (
                "id",
                "session_id",
                "role",
                "timestamp",
                "_compressed_summary",
                "active",
                "compacted",
            )
        ),
        *(
            ("session_model_usage", column)
            for column in (
                "session_id",
                "model",
                "billing_provider",
                "billing_base_url",
                "billing_mode",
                "task",
                "api_call_count",
                "input_tokens",
                "output_tokens",
                "cache_read_tokens",
                "cache_write_tokens",
                "reasoning_tokens",
                "estimated_cost_usd",
                "actual_cost_usd",
            )
        ),
        *(
            ("async_delegations", column)
            for column in (
                "delegation_id",
                "origin_session",
                "origin_ui_session_id",
                "state",
                "dispatched_at",
                "updated_at",
                "delivery_state",
                "delivery_attempts",
                "origin_session_id",
            )
        ),
        ("state_meta", "key"),
        *(
            ("gateway_routing", column)
            for column in ("scope", "session_key", "entry_json", "updated_at")
        ),
        ("gateway_hygiene_state", "session_key"),
        ("gateway_hygiene_state", "failure_streak"),
        *(
            ("compression_locks", column)
            for column in ("session_id", "holder", "acquired_at", "expires_at")
        ),
        *(
            ("session_turn_leases", column)
            for column in ("conversation_id", "holder", "acquired_at", "expires_at")
        ),
        *(
            ("telegram_dm_topic_mode", column)
            for column in ("chat_id", "user_id", "enabled", "activated_at", "updated_at")
        ),
        *(
            ("telegram_dm_topic_bindings", column)
            for column in (
                "chat_id",
                "thread_id",
                "user_id",
                "session_key",
                "session_id",
                "managed_mode",
                "linked_at",
                "updated_at",
            )
        ),
    }
)

_COLUMN_DEFAULTS: Dict[tuple[str, str], str] = {
    **{
        ("sessions", column): "0"
        for column in (
            "expiry_finalized",
            "message_count",
            "tool_call_count",
            "input_tokens",
            "output_tokens",
            "cache_read_tokens",
            "cache_write_tokens",
            "reasoning_tokens",
            "git_metadata_generation",
            "api_call_count",
            "compression_fallback_streak",
            "compression_ineffective_count",
            "rewind_count",
            "archived",
            "pinned",
            "hidden",
        )
    },
    ("messages", "observed"): "0",
    ("messages", "_compressed_summary"): "0",
    ("messages", "active"): "1",
    ("messages", "compacted"): "0",
    **{
        ("session_model_usage", column): "''::text"
        for column in ("billing_provider", "billing_base_url", "billing_mode", "task")
    },
    **{
        ("session_model_usage", column): "0"
        for column in (
            "api_call_count",
            "input_tokens",
            "output_tokens",
            "cache_read_tokens",
            "cache_write_tokens",
            "reasoning_tokens",
            "estimated_cost_usd",
            "actual_cost_usd",
        )
    },
    ("async_delegations", "origin_ui_session_id"): "''::text",
    ("async_delegations", "delivery_state"): "'pending'::text",
    ("async_delegations", "delivery_attempts"): "0",
    ("async_delegations", "origin_session_id"): "''::text",
    ("gateway_routing", "scope"): "''::text",
    ("gateway_hygiene_state", "failure_streak"): "0",
    ("telegram_dm_topic_mode", "enabled"): "1",
    ("telegram_dm_topic_bindings", "managed_mode"): "'auto'::text",
}


def _column_specs(
    required_columns: Dict[str, frozenset[str]],
) -> Dict[tuple[str, str], _ColumnSpec]:
    specs: Dict[tuple[str, str], _ColumnSpec] = {}
    for table, columns in required_columns.items():
        for column in columns:
            key = (table, column)
            if key in _BIGINT_COLUMNS:
                data_type = "bigint"
            elif key in _DOUBLE_PRECISION_COLUMNS:
                data_type = "double precision"
            else:
                data_type = "text"
            specs[key] = _ColumnSpec(
                data_type=data_type,
                nullable=key not in _NOT_NULL_COLUMNS,
                default=_COLUMN_DEFAULTS.get(key),
                identity_generation=(
                    "BY DEFAULT" if key == ("messages", "id") else None
                ),
            )
    return specs


_REQUIRED_COLUMN_SPECS = _column_specs(_REQUIRED_COLUMNS)
_FORWARD_REQUIRED_COLUMN_SPECS = _column_specs(_FORWARD_REQUIRED_COLUMNS)
_V22_REQUIRED_COLUMN_SPECS = _column_specs(_V22_REQUIRED_COLUMNS)


_REQUIRED_INDEXES = frozenset(
    {
        "idx_telegram_dm_topic_bindings_session",
        "idx_telegram_dm_topic_bindings_user",
        "idx_sessions_source",
        "idx_sessions_source_id",
        "idx_sessions_parent",
        "idx_sessions_started",
        "idx_messages_session",
        "idx_messages_session_id",
        "idx_messages_assistant_calls_by_session",
        "idx_compression_locks_expires",
        "idx_session_turn_leases_expires",
        "idx_session_model_usage_session",
        "idx_session_model_usage_model",
        "idx_async_delegations_delivery",
        "idx_messages_session_active",
        "idx_messages_active_null",
        "idx_sessions_session_key",
        "idx_sessions_gateway_peer",
        "idx_sessions_handoff_state",
        "idx_sessions_system_prompt_hash",
        "idx_sessions_title_unique",
        "idx_messages_platform_msg_id",
        "idx_messages_search_tsv",
    }
)

_V22_REQUIRED_INDEXES = frozenset(
    {
        "idx_telegram_dm_topic_bindings_session",
        "idx_telegram_dm_topic_bindings_user",
        "idx_sessions_source",
        "idx_sessions_source_id",
        "idx_sessions_parent",
        "idx_sessions_started",
        "idx_messages_session",
        "idx_compression_locks_expires",
        "idx_session_model_usage_session",
        "idx_session_model_usage_model",
        "idx_async_delegations_delivery",
        "idx_messages_session_active",
        "idx_messages_active_null",
        "idx_sessions_session_key",
        "idx_sessions_gateway_peer",
        "idx_sessions_handoff_state",
        "idx_messages_platform_msg_id",
        "idx_messages_search_tsv",
    }
)


@dataclass(frozen=True)
class _IndexSpec:
    """Semantic shape of one non-primary Postgres index."""

    table: str
    unique: bool
    access_method: str
    keys: tuple[str, ...]
    options: tuple[int, ...]
    opclasses: tuple[str, ...]
    predicate: Optional[str] = None
    expression: Optional[str] = None


def _index_spec(
    table: str,
    *keys: str,
    unique: bool = False,
    access_method: str = "btree",
    descending: Collection[int] = (),
    opclasses: Optional[tuple[str, ...]] = None,
    predicate: Optional[str] = None,
    expression: Optional[str] = None,
) -> _IndexSpec:
    descending_positions = set(descending)
    return _IndexSpec(
        table=table,
        unique=unique,
        access_method=access_method,
        keys=tuple(keys),
        # pg_index.indoption uses bit 0 for DESC and bit 1 for NULLS FIRST.
        # Postgres's default for DESC is NULLS FIRST, hence 3.
        options=tuple(
            3 if position in descending_positions else 0
            for position in range(len(keys))
        ),
        opclasses=opclasses or tuple("text_ops" for _ in keys),
        predicate=predicate,
        expression=expression,
    )


_SEARCH_TEXT_CATALOG_EXPRESSION = (
    '"left"((((COALESCE(content, \'\'::text) || \' \'::text) || '
    "COALESCE(tool_name, ''::text)) || ' '::text) || "
    "COALESCE(tool_calls, ''::text), 500000)"
)
_SEARCH_TSV_CATALOG_EXPRESSION = (
    "to_tsvector('simple'::regconfig, "
    f"{_SEARCH_TEXT_CATALOG_EXPRESSION})"
)

_REQUIRED_INDEX_SPECS: Dict[str, _IndexSpec] = {
    "idx_telegram_dm_topic_bindings_session": _index_spec(
        "telegram_dm_topic_bindings", "session_id", unique=True
    ),
    "idx_telegram_dm_topic_bindings_user": _index_spec(
        "telegram_dm_topic_bindings", "user_id", "chat_id"
    ),
    "idx_sessions_source": _index_spec("sessions", "source"),
    "idx_sessions_source_id": _index_spec("sessions", "source", "id"),
    "idx_sessions_parent": _index_spec("sessions", "parent_session_id"),
    "idx_sessions_started": _index_spec(
        "sessions",
        "started_at",
        descending=(0,),
        opclasses=("float8_ops",),
    ),
    "idx_messages_session": _index_spec(
        "messages", "session_id", '"timestamp"',
        opclasses=("text_ops", "float8_ops"),
    ),
    "idx_messages_session_id": _index_spec(
        "messages", "session_id", "id",
        opclasses=("text_ops", "int8_ops"),
    ),
    "idx_messages_assistant_calls_by_session": _index_spec(
        "messages",
        "session_id",
        predicate="role = 'assistant'::text AND tool_calls IS NOT NULL",
    ),
    "idx_compression_locks_expires": _index_spec(
        "compression_locks", "expires_at", opclasses=("float8_ops",)
    ),
    "idx_session_turn_leases_expires": _index_spec(
        "session_turn_leases", "expires_at", opclasses=("float8_ops",)
    ),
    "idx_session_model_usage_session": _index_spec(
        "session_model_usage", "session_id"
    ),
    "idx_session_model_usage_model": _index_spec(
        "session_model_usage", "model"
    ),
    "idx_async_delegations_delivery": _index_spec(
        "async_delegations", "delivery_state", "completed_at",
        opclasses=("text_ops", "float8_ops"),
    ),
    "idx_messages_session_active": _index_spec(
        "messages", "session_id", "active", '"timestamp"',
        opclasses=("text_ops", "int8_ops", "float8_ops"),
    ),
    "idx_messages_active_null": _index_spec(
        "messages", "active", opclasses=("int8_ops",),
        predicate="active IS NULL",
    ),
    "idx_sessions_session_key": _index_spec(
        "sessions", "session_key", "started_at", descending=(1,),
        opclasses=("text_ops", "float8_ops"),
    ),
    "idx_sessions_gateway_peer": _index_spec(
        "sessions", "source", "user_id", "chat_id", "chat_type",
        "thread_id", "started_at", descending=(5,),
        opclasses=(
            "text_ops", "text_ops", "text_ops", "text_ops", "text_ops",
            "float8_ops",
        ),
    ),
    "idx_sessions_handoff_state": _index_spec(
        "sessions", "handoff_state", "started_at",
        opclasses=("text_ops", "float8_ops"),
    ),
    "idx_sessions_system_prompt_hash": _index_spec(
        "sessions", "system_prompt_hash"
    ),
    "idx_sessions_title_unique": _index_spec(
        "sessions", "title", unique=True, predicate="title IS NOT NULL"
    ),
    "idx_messages_platform_msg_id": _index_spec(
        "messages", "session_id", "platform_message_id",
        predicate="platform_message_id IS NOT NULL",
    ),
    "idx_messages_search_tsv": _index_spec(
        "messages",
        _SEARCH_TSV_CATALOG_EXPRESSION,
        access_method="gin",
        opclasses=("tsvector_ops",),
        expression=_SEARCH_TSV_CATALOG_EXPRESSION,
    ),
}

_V22_REQUIRED_INDEX_SPECS = {
    name: spec
    for name, spec in _REQUIRED_INDEX_SPECS.items()
    if name in _V22_REQUIRED_INDEXES
}

# pg_trgm is best-effort. Its absence is valid, but if the name exists it must
# still be the audited expression GIN index rather than a same-named decoy.
_OPTIONAL_INDEX_SPECS = {
    "idx_messages_search_trgm": _index_spec(
        "messages",
        _SEARCH_TEXT_CATALOG_EXPRESSION,
        access_method="gin",
        opclasses=("gin_trgm_ops",),
        expression=_SEARCH_TEXT_CATALOG_EXPRESSION,
    )
}

_REQUIRED_PRIMARY_KEYS = {
    "system_prompts": ("hash",),
    "sessions": ("id",),
    "messages": ("id",),
    "session_model_usage": (
        "session_id", "model", "billing_provider", "billing_base_url",
        "billing_mode", "task",
    ),
    "async_delegations": ("delegation_id",),
    "state_meta": ("key",),
    "gateway_routing": ("scope", "session_key"),
    "gateway_hygiene_state": ("session_key",),
    "compression_locks": ("session_id",),
    "session_turn_leases": ("conversation_id",),
    "telegram_dm_topic_mode": ("chat_id",),
    "telegram_dm_topic_bindings": ("chat_id", "thread_id"),
}

_V22_REQUIRED_PRIMARY_KEYS = {
    table: columns
    for table, columns in _REQUIRED_PRIMARY_KEYS.items()
    if table in _V22_REQUIRED_COLUMNS
}

@dataclass(frozen=True)
class _ConstraintSpec:
    """Semantic shape of a table constraint from ``pg_constraint``."""

    kind: str
    columns: tuple[str, ...]
    referenced_schema: Optional[str] = None
    referenced_table: Optional[str] = None
    referenced_columns: tuple[str, ...] = ()
    update_action: Optional[str] = None
    delete_action: Optional[str] = None
    match_type: Optional[str] = None
    deferrable: bool = False
    initially_deferred: bool = False
    validated: bool = True


def _constraint_specs(
    primary_keys: Dict[str, tuple[str, ...]],
    *,
    include_system_prompt: bool,
) -> Dict[tuple[str, str], _ConstraintSpec]:
    specs = {
        (table, f"{table}_pkey"): _ConstraintSpec("p", columns)
        for table, columns in primary_keys.items()
    }
    foreign_keys = {
        (
            "sessions",
            "sessions_parent_session_id_fkey",
        ): _ConstraintSpec(
            "f",
            ("parent_session_id",),
            referenced_schema=_SCHEMA,
            referenced_table="sessions",
            referenced_columns=("id",),
            update_action="a",
            delete_action="a",
            match_type="s",
            deferrable=True,
        ),
        ("messages", "messages_session_id_fkey"): _ConstraintSpec(
            "f",
            ("session_id",),
            referenced_schema=_SCHEMA,
            referenced_table="sessions",
            referenced_columns=("id",),
            update_action="a",
            delete_action="a",
            match_type="s",
            deferrable=True,
        ),
        (
            "session_model_usage",
            "session_model_usage_session_id_fkey",
        ): _ConstraintSpec(
            "f",
            ("session_id",),
            referenced_schema=_SCHEMA,
            referenced_table="sessions",
            referenced_columns=("id",),
            update_action="a",
            delete_action="c",
            match_type="s",
            deferrable=True,
        ),
        (
            "telegram_dm_topic_bindings",
            "telegram_dm_topic_bindings_session_id_fkey",
        ): _ConstraintSpec(
            "f",
            ("session_id",),
            referenced_schema=_SCHEMA,
            referenced_table="sessions",
            referenced_columns=("id",),
            update_action="a",
            delete_action="c",
            match_type="s",
            deferrable=True,
        ),
    }
    if include_system_prompt:
        foreign_keys[(
            "sessions",
            "sessions_system_prompt_hash_fkey",
        )] = _ConstraintSpec(
            "f",
            ("system_prompt_hash",),
            referenced_schema=_SCHEMA,
            referenced_table="system_prompts",
            referenced_columns=("hash",),
            update_action="a",
            delete_action="a",
            match_type="s",
            deferrable=True,
        )
    specs.update(foreign_keys)
    return specs


_REQUIRED_CONSTRAINT_SPECS = _constraint_specs(
    _REQUIRED_PRIMARY_KEYS,
    include_system_prompt=True,
)
_V22_REQUIRED_CONSTRAINT_SPECS = _constraint_specs(
    _V22_REQUIRED_PRIMARY_KEYS,
    include_system_prompt=False,
)

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
_JSON_TYPE_SESSION_RESET_RE = re.compile(
    r"json_type\(\s*sessions\.model_config\s*,\s*'\$\._reset_from'\s*\)"
)
_JSON_REMOVE_SESSION_RESET_RE = re.compile(
    r"json_remove\(\s*sessions\.model_config\s*,\s*'\$\._reset_from'\s*\)"
)
_JSON_SET_EXCLUDED_RESET_RE = re.compile(
    r"json_set\(\s*excluded\.model_config\s*,\s*'\$\._reset_from'\s*,\s*"
    r"json_extract\(\s*sessions\.model_config\s*,\s*'\$\._reset_from'\s*\)\s*\)"
)
_JSON_SET_CHILD_RESET_RE = re.compile(
    r"json_set\(\s*COALESCE\(\s*child\.model_config\s*,\s*'\{\}'\s*\)\s*,\s*"
    r"'\$\._reset_from'\s*,\s*child\.parent_session_id\s*\)"
)
_INSERT_OR_IGNORE_RE = re.compile(r"^(\s*)INSERT\s+OR\s+IGNORE\s+INTO\b", re.IGNORECASE)
_LIKE_RE = re.compile(r"\bLIKE\b")
# SQLite's null-safe comparison against a bind parameter — ``col IS ?`` /
# ``col IS NOT ?`` — has no direct Postgres spelling. Rewrite it only on the
# parameterized execution path, before the generic qmark conversion.
_IS_PARAM_RE = re.compile(r"\bIS\s+(NOT\s+)?\?", re.IGNORECASE)
_LIMIT_OFFSET_PARAM_RE = re.compile(
    r"\bLIMIT\s+\?\s+OFFSET\s+\?", re.IGNORECASE
)
_INSERT_MESSAGES_RE = re.compile(r"^\s*INSERT\s+INTO\s+messages\s*\(", re.IGNORECASE)

# ── Whole-statement overrides ──────────────────────────────────────────────
# A handful of inherited statements are not mechanically translatable because
# SQLite and Postgres disagree on *scoping*, not syntax. Rewriting those by
# pattern would mean parsing SQL in the hot path; instead each one is replaced
# wholesale, keyed by its whitespace-normalized upstream text. If an upstream
# rebase edits the original by so much as a column, the key stops matching —
# ``tests/gateway/test_session_store_pg_unit.py`` asserts every key is still
# present in ``hermes_state.py``, so that drift fails a unit test instead of
# surfacing as a runtime error in production.


def _normalize_statement(sql: str) -> str:
    return " ".join(sql.split())


# update_token_counts / record_auxiliary_usage (upstream v20 cb7f6bbb2 +
# v22 eb6aa0360). Inside ``DO UPDATE SET`` both the target table and
# ``excluded`` are in scope on Postgres, so SQLite's bare self-reference
# ("api_call_count = api_call_count + excluded.api_call_count", meaning the
# row already stored) raises `column reference "..." is ambiguous`. The
# rewrite only qualifies those reads with the target table; the accumulate
# semantics are identical.
_USAGE_UPSERT_SQLITE = """INSERT INTO session_model_usage (
                   session_id, model, billing_provider, billing_base_url, billing_mode,
                   task, api_call_count, input_tokens, output_tokens,
                   cache_read_tokens, cache_write_tokens, reasoning_tokens,
                   estimated_cost_usd, actual_cost_usd, cost_status, cost_source,
                   first_seen, last_seen
               ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(session_id, model, billing_provider, billing_base_url, billing_mode, task)
               DO UPDATE SET
                   api_call_count = api_call_count + excluded.api_call_count,
                   input_tokens = input_tokens + excluded.input_tokens,
                   output_tokens = output_tokens + excluded.output_tokens,
                   cache_read_tokens = cache_read_tokens + excluded.cache_read_tokens,
                   cache_write_tokens = cache_write_tokens + excluded.cache_write_tokens,
                   reasoning_tokens = reasoning_tokens + excluded.reasoning_tokens,
                   estimated_cost_usd = estimated_cost_usd + excluded.estimated_cost_usd,
                   actual_cost_usd = actual_cost_usd + excluded.actual_cost_usd,
                   cost_status = COALESCE(excluded.cost_status, cost_status),
                   cost_source = COALESCE(excluded.cost_source, cost_source),
                   last_seen = excluded.last_seen"""

_USAGE_UPSERT_PG = """INSERT INTO session_model_usage (
                   session_id, model, billing_provider, billing_base_url, billing_mode,
                   task, api_call_count, input_tokens, output_tokens,
                   cache_read_tokens, cache_write_tokens, reasoning_tokens,
                   estimated_cost_usd, actual_cost_usd, cost_status, cost_source,
                   first_seen, last_seen
               ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(session_id, model, billing_provider, billing_base_url, billing_mode, task)
               DO UPDATE SET
                   api_call_count = session_model_usage.api_call_count + excluded.api_call_count,
                   input_tokens = session_model_usage.input_tokens + excluded.input_tokens,
                   output_tokens = session_model_usage.output_tokens + excluded.output_tokens,
                   cache_read_tokens = session_model_usage.cache_read_tokens + excluded.cache_read_tokens,
                   cache_write_tokens = session_model_usage.cache_write_tokens + excluded.cache_write_tokens,
                   reasoning_tokens = session_model_usage.reasoning_tokens + excluded.reasoning_tokens,
                   estimated_cost_usd = session_model_usage.estimated_cost_usd + excluded.estimated_cost_usd,
                   actual_cost_usd = session_model_usage.actual_cost_usd + excluded.actual_cost_usd,
                   cost_status = COALESCE(excluded.cost_status, session_model_usage.cost_status),
                   cost_source = COALESCE(excluded.cost_source, session_model_usage.cost_source),
                   last_seen = excluded.last_seen"""

STATEMENT_OVERRIDES: Dict[str, str] = {
    _normalize_statement(_USAGE_UPSERT_SQLITE): _USAGE_UPSERT_PG,
}


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
    ASCII-case-insensitive), null-safe ``IS ?``/``IS NOT ?`` →
    ``IS [NOT] DISTINCT FROM %s`` and ``%``→``%%`` escaping only apply
    outside single-quoted string literals; ``json_extract``/hex-literal/
    ``INSERT OR IGNORE`` rewrites run on the raw statement first because
    their patterns intentionally span literals.
    """
    sql = STATEMENT_OVERRIDES.get(_normalize_statement(sql), sql)
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
    # v26 preserves reset lineage inside model_config. SQLite's JSON1 names
    # have no PostgreSQL equivalents, so port the two exact runtime shapes
    # before the broader scalar json_extract rewrite below.
    sql = _JSON_SET_EXCLUDED_RESET_RE.sub(
        "jsonb_set(excluded.model_config::jsonb, '{_reset_from}', "
        "sessions.model_config::jsonb -> '_reset_from', true)::text",
        sql,
    )
    sql = _JSON_SET_CHILD_RESET_RE.sub(
        "jsonb_set(COALESCE(child.model_config, '{}')::jsonb, "
        "'{_reset_from}', to_jsonb(child.parent_session_id), true)::text",
        sql,
    )
    sql = _JSON_TYPE_SESSION_RESET_RE.sub(
        "(sessions.model_config::jsonb -> '_reset_from')", sql
    )
    sql = _JSON_REMOVE_SESSION_RESET_RE.sub(
        "(sessions.model_config::jsonb - '_reset_from')", sql
    )
    sql = _JSON_EXTRACT_COALESCE_RE.sub(
        r"(COALESCE(\1, '{}')::jsonb ->> '\2')", sql
    )
    sql = _JSON_EXTRACT_PLAIN_RE.sub(r"(COALESCE(\1, '{}')::jsonb ->> '\2')", sql)
    # SessionDB uses SQLite's sentinel LIMIT -1 for offset-only paging.
    # Postgres rejects a negative LIMIT; NULL means unbounded and preserves
    # the same two-parameter shape for both bounded and unbounded callers.
    sql = _LIMIT_OFFSET_PARAM_RE.sub("LIMIT NULLIF(?, -1) OFFSET ?", sql)

    out: List[str] = []
    for is_literal, seg in _split_literals(sql):
        if is_literal:
            out.append(seg.replace("%", "%%") if with_params else seg)
            continue
        seg = _LIKE_RE.sub("ILIKE", seg)
        if with_params:
            # Escape existing percents first so the placeholder emitted by the
            # null-safe rewrite is not escaped a second time.
            seg = seg.replace("%", "%%")
            seg = _IS_PARAM_RE.sub(
                lambda match: (
                    "IS DISTINCT FROM %s"
                    if match.group(1)
                    else "IS NOT DISTINCT FROM %s"
                ),
                seg,
            )
            seg = seg.replace("?", "%s")
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
        self._last_change_count = 0

    def execute(self, sql: str, params=None) -> _Result:
        if sql.strip().rstrip(";").lower() == "select changes()":
            return _Result(
                [_Row(["changes()"], (self._last_change_count,))],
                rowcount=1,
                lastrowid=None,
            )
        result = _run_statement(self._raw, sql, params)
        statement = sql.lstrip().upper()
        if statement.startswith(("INSERT", "UPDATE", "DELETE", "REPLACE")) or (
            statement.startswith("WITH")
            and re.search(r"\b(?:INSERT|UPDATE|DELETE)\b", statement)
        ):
            self._last_change_count = max(0, int(result.rowcount))
        return result

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
        allow_schema_migration: bool = False,
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
        self.db_path = db_path or hermes_state._default_db_path()
        self.read_only = read_only
        # Reentrant: inherited read paths take ``with self._lock`` around
        # what is now a per-statement pooled checkout.
        self._lock = threading.RLock()
        self._write_count = 0
        # tsvector search is always available; the ILIKE CJK path replaces
        # the trigram FTS5 table (works with or without pg_trgm).
        self._fts_enabled = True
        self._fts_stale = False
        self._trigram_available = True
        self._fts_cjk_available = False
        self._fts_unavailable_warned = False
        self._trigram_unavailable_warned = False
        self._closed = False
        self._allow_schema_migration = bool(allow_schema_migration)
        self._message_columns_cache: Optional[List[str]] = None
        self._storage_attestation: Optional[Dict[str, Any]] = None
        # Inherited async token-accounting methods require the same lifecycle
        # state as SessionDB. The writer is backend-agnostic; only its writes
        # flow through our Postgres transaction seam below.
        self._token_queue: deque = deque()
        self._token_queue_cond = threading.Condition(threading.Lock())
        self._token_writer_thread: Optional[threading.Thread] = None
        self._token_writer_stop = False
        self._token_writer_busy = False
        self._token_atexit_hook: Optional[Callable[[], None]] = None

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

    def storage_attestation(self) -> Dict[str, Any]:
        """Return the verified, non-secret storage identity for health checks."""
        if self._storage_attestation is None:
            raise RuntimeError(
                "Postgres session-store attestation requested before schema verification"
            )
        return dict(self._storage_attestation)

    # ── Schema ──────────────────────────────────────────────────────────

    # Serializes concurrent cold-boot bootstrap (see _init_pg_schema).
    # Stable 63-bit advisory-lock key derived from the schema name.
    _BOOTSTRAP_LOCK_KEY = int.from_bytes(
        hashlib.sha256(f"hermes_state_pg:{_SCHEMA}".encode()).digest()[:8],
        "big",
        signed=True,
    )

    def _init_pg_schema(self) -> None:
        """Create, verify, or explicitly migrate the Postgres schema.

        The advisory lock serializes cold starts and the one-shot migration.
        Existing stores are inspected before any DDL: ordinary boot against
        v22 or a drifted v26 catalog is observation-only and fails closed.
        """
        with self._pool.connection() as conn:
            conn.execute(
                "SELECT pg_advisory_lock(%s)", (self._BOOTSTRAP_LOCK_KEY,)
            )
            try:
                surface_marker = EXPECTED_SCHEMA_SURFACE_SHA256
                namespace = conn.execute(
                    "SELECT to_regnamespace(%s)", (_SCHEMA,)
                ).fetchone()
                schema_exists = bool(namespace and namespace[0] is not None)
                install_search_acceleration = (
                    not schema_exists or self._allow_schema_migration
                )

                if self._allow_schema_migration and not schema_exists:
                    raise RuntimeError(
                        "Postgres v22→v26 migration requires an existing "
                        "hermes_state schema at the audited v22 shape; refusing "
                        "to initialize an empty database in migration mode."
                    )

                with conn.transaction():
                    if not schema_exists:
                        self._create_fresh_schema(conn)
                    else:
                        version_table = conn.execute(
                            "SELECT to_regclass(%s)",
                            (f"{_SCHEMA}.schema_version",),
                        ).fetchone()
                        if not version_table or version_table[0] is None:
                            raise RuntimeError(
                                "Postgres session-store schema exists but its "
                                "schema_version table is missing; refusing to "
                                "repair an ambiguous partial store."
                            )
                        db_version = self._read_schema_version(conn)
                        if self._allow_schema_migration:
                            # _migrate_v22_to_v26 performs the complete,
                            # observation-only v22 preflight before its first
                            # DDL statement. Migration mode is intentionally
                            # not an idempotent "ensure current" operation: a
                            # wrong DSN pointing at v26 must fail too.
                            self._migrate_v22_to_v26(conn)
                        elif db_version == EXPECTED_SCHEMA_VERSION:
                            surface_marker = self._assert_persisted_schema_markers(
                                conn
                            )
                            self._verify_catalog(
                                conn, surface_marker=surface_marker
                            )
                        else:
                            # Stable fail-closed errors (including the explicit
                            # drain instruction for v22) live at this seam.
                            self._assert_persisted_schema_markers(conn)

                self._storage_attestation = {
                    "backend": "postgres",
                    "schema_version": EXPECTED_SCHEMA_VERSION,
                    "surface_marker": surface_marker,
                }

                if install_search_acceleration:
                    self._ensure_optional_search_acceleration(conn)
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

    def _create_fresh_schema(self, conn) -> None:
        for stmt in PG_SCHEMA_SQL:
            conn.execute(stmt)
        conn.execute(
            f"INSERT INTO {_SCHEMA}.schema_version (version) VALUES (%s)",
            (EXPECTED_SCHEMA_VERSION,),
        )
        conn.execute(
            f"INSERT INTO {_SCHEMA}.state_meta (key, value) VALUES (%s, %s)",
            (_META_SURFACE_KEY, EXPECTED_SCHEMA_SURFACE_SHA256),
        )
        self._assert_persisted_schema_markers(conn)
        self._verify_catalog(conn)

    @staticmethod
    def _ensure_optional_search_acceleration(conn) -> None:
        """Install optional trigram search only during create/migrate paths."""
        try:
            with conn.transaction():
                for stmt in PG_TRGM_SQL:
                    conn.execute(stmt)
        except Exception as exc:
            # This runs after the correctness transaction. A savepoint keeps an
            # unavailable extension or index from poisoning the connection;
            # search then degrades to unindexed ILIKE.
            logger.info(
                "pg_trgm acceleration unavailable for the session store "
                "(CJK search falls back to unindexed ILIKE): %s",
                exc,
            )

    @staticmethod
    def _read_surface_marker(conn) -> Optional[str]:
        row = conn.execute(
            f"SELECT value FROM {_SCHEMA}.state_meta WHERE key = %s",
            (_META_SURFACE_KEY,),
        ).fetchone()
        return None if row is None else str(row[0])

    @staticmethod
    def _read_schema_version(conn) -> int:
        row = conn.execute(
            f"SELECT COUNT(*), MIN(version), MAX(version) "
            f"FROM {_SCHEMA}.schema_version"
        ).fetchone()
        if row is None or int(row[0]) != 1 or row[1] != row[2]:
            count = 0 if row is None else int(row[0])
            if count == 0:
                raise RuntimeError(
                    "Postgres session store is missing schema_version; "
                    "refusing to infer it."
                )
            raise RuntimeError(
                "Postgres session store must contain exactly one "
                f"schema_version row (found {count}); refusing to infer it."
            )
        return int(row[1])

    def _migrate_v22_to_v26(self, conn) -> None:
        """Run the drained, transactional v22→v26 migration."""
        self._assert_v22_migration_precondition(conn)

        # Tables precede column expansion; indexes follow duplicate repair.
        for stmt in PG_SCHEMA_SQL:
            if "CREATE TABLE" in stmt or "CREATE SCHEMA" in stmt:
                conn.execute(stmt)
        for stmt in PG_EXPAND_SQL:
            conn.execute(stmt)
        for stmt in PG_V26_CONSTRAINT_SQL:
            conn.execute(stmt)
        self._migrate_system_prompts(conn)
        conn.execute(PG_V26_DEDUPLICATE_TITLES_SQL)
        for stmt in PG_SCHEMA_SQL:
            if " INDEX " in stmt:
                conn.execute(stmt)

        # Validate the live transaction's catalog before claiming v26.
        self._verify_catalog(conn)
        conn.execute(
            f"UPDATE {_SCHEMA}.schema_version SET version = %s",
            (EXPECTED_SCHEMA_VERSION,),
        )
        conn.execute(
            f"UPDATE {_SCHEMA}.state_meta SET value = %s WHERE key = %s",
            (EXPECTED_SCHEMA_SURFACE_SHA256, _META_SURFACE_KEY),
        )
        self._assert_persisted_schema_markers(conn)

    @staticmethod
    def _assert_v22_migration_precondition(conn) -> None:
        """Require the one exact source state accepted by the migration.

        This method is observation-only and must remain ahead of every DDL
        statement in :meth:`_migrate_v22_to_v26`. It is also reused by the
        migration command's read-only dry-run.
        """
        db_version = PgSessionDB._read_schema_version(conn)
        if db_version != 22:
            raise RuntimeError(
                "Postgres v22→v26 migration requires schema_version 22 "
                f"exactly (found {db_version}); refusing to mutate this store."
            )
        surface = PgSessionDB._read_surface_marker(conn)
        if surface != _V22_SCHEMA_SURFACE_SHA256:
            raise RuntimeError(
                "Refusing v22→v26 migration: persisted v22 schema surface "
                f"is {surface!r}, expected {_V22_SCHEMA_SURFACE_SHA256}."
            )
        PgSessionDB._verify_catalog_shape(
            conn,
            required_column_specs=_V22_REQUIRED_COLUMN_SPECS,
            required_index_specs=_V22_REQUIRED_INDEX_SPECS,
            required_constraint_specs=_V22_REQUIRED_CONSTRAINT_SPECS,
            label="v22 migration source",
            exact_relations=True,
        )

    @staticmethod
    def _migrate_system_prompts(conn) -> None:
        cursor = conn.execute(
            f"SELECT id, system_prompt FROM {_SCHEMA}.sessions "
            "WHERE system_prompt IS NOT NULL ORDER BY id"
        )
        while True:
            rows = cursor.fetchmany(250)
            if not rows:
                return
            for session_id, prompt in rows:
                prompt_hash = hashlib.sha256(str(prompt).encode("utf-8")).hexdigest()
                conn.execute(
                    f"INSERT INTO {_SCHEMA}.system_prompts (hash, prompt) "
                    "VALUES (%s, %s) ON CONFLICT (hash) DO NOTHING",
                    (prompt_hash, prompt),
                )
                conn.execute(
                    f"UPDATE {_SCHEMA}.sessions "
                    "SET system_prompt_hash = %s, system_prompt = NULL "
                    "WHERE id = %s",
                    (prompt_hash, session_id),
                )

    @staticmethod
    def _verify_catalog(
        conn,
        *,
        surface_marker: str = EXPECTED_SCHEMA_SURFACE_SHA256,
    ) -> None:
        """Prove the live store matches one exact audited v26 surface."""
        if surface_marker == EXPECTED_SCHEMA_SURFACE_SHA256:
            column_specs = _REQUIRED_COLUMN_SPECS
            label = "v26"
        elif surface_marker == FORWARD_SCHEMA_SURFACE_SHA256:
            column_specs = _FORWARD_REQUIRED_COLUMN_SPECS
            label = "v26 + compressed-summary marker"
        else:
            raise RuntimeError(
                "Postgres session-store catalog verification received an "
                f"unknown surface marker {surface_marker!r}."
            )
        PgSessionDB._verify_catalog_shape(
            conn,
            required_column_specs=column_specs,
            required_index_specs=_REQUIRED_INDEX_SPECS,
            required_constraint_specs=_REQUIRED_CONSTRAINT_SPECS,
            label=label,
            exact_relations=True,
        )

    @staticmethod
    def _verify_catalog_shape(
        conn,
        *,
        required_column_specs: Dict[tuple[str, str], _ColumnSpec],
        required_index_specs: Dict[str, _IndexSpec],
        required_constraint_specs: Dict[tuple[str, str], _ConstraintSpec],
        label: str,
        exact_relations: bool,
    ) -> None:
        """Verify one named Postgres catalog contract without mutating it."""
        column_rows = conn.execute(
            "SELECT table_name, column_name, data_type, is_nullable, "
            "column_default, is_identity, identity_generation "
            "FROM information_schema.columns "
            "WHERE table_schema = %s",
            (_SCHEMA,),
        ).fetchall()
        actual_column_specs: Dict[tuple[str, str], _ColumnSpec] = {}
        malformed_identity_columns: List[str] = []
        for (
            table,
            column,
            data_type,
            is_nullable,
            column_default,
            is_identity,
            identity_generation,
        ) in column_rows:
            key = (str(table), str(column))
            normalized_default = (
                " ".join(str(column_default).split())
                if column_default is not None
                else None
            )
            normalized_identity = (
                str(identity_generation)
                if identity_generation is not None
                else None
            )
            if (str(is_identity) == "YES") != (normalized_identity is not None):
                malformed_identity_columns.append(f"{key[0]}.{key[1]}")
            actual_column_specs[key] = _ColumnSpec(
                data_type=str(data_type),
                nullable=str(is_nullable) == "YES",
                default=normalized_default,
                identity_generation=normalized_identity,
            )

        required_columns: Dict[str, set[str]] = {}
        for table, column in required_column_specs:
            required_columns.setdefault(table, set()).add(column)
        actual_columns: Dict[str, set[str]] = {}
        for table, column in actual_column_specs:
            actual_columns.setdefault(table, set()).add(column)

        problems: List[str] = []
        if malformed_identity_columns:
            problems.append(
                "columns with inconsistent identity metadata "
                f"{sorted(malformed_identity_columns)}"
            )
        if exact_relations:
            expected_tables = set(required_columns)
            actual_tables = set(actual_columns)
            if actual_tables != expected_tables:
                missing_tables = sorted(expected_tables.difference(actual_tables))
                unexpected_tables = sorted(actual_tables.difference(expected_tables))
                if missing_tables:
                    problems.append(f"missing tables {missing_tables}")
                if unexpected_tables:
                    problems.append(f"unexpected tables {unexpected_tables}")

        for table, required in required_columns.items():
            actual = actual_columns.get(table, set())
            missing = required.difference(actual)
            if missing:
                problems.append(f"{table} missing columns {sorted(missing)}")
            if exact_relations:
                unexpected = actual.difference(required)
                if unexpected:
                    problems.append(
                        f"{table} has unexpected columns {sorted(unexpected)}"
                    )
        for key, expected in required_column_specs.items():
            actual = actual_column_specs.get(key)
            if actual is not None and actual != expected:
                problems.append(
                    f"column {key[0]}.{key[1]} is {actual!r}, "
                    f"expected {expected!r}"
                )

        index_rows = conn.execute(
            "SELECT table_class.relname, index_class.relname, "
            "index_catalog.indisunique, index_catalog.indisvalid, "
            "index_catalog.indisready, index_catalog.indislive, "
            "access_method.amname, index_catalog.indnatts, "
            "index_catalog.indnkeyatts, "
            "ARRAY(SELECT pg_get_indexdef(index_catalog.indexrelid, "
            "position, TRUE) FROM generate_series("
            "1, index_catalog.indnkeyatts) AS position ORDER BY position), "
            "index_catalog.indoption::smallint[], "
            "ARRAY(SELECT operator_class.opcname FROM "
            "unnest(index_catalog.indclass::oid[]) WITH ORDINALITY "
            "AS indexed_class(operator_class_oid, position) "
            "JOIN pg_opclass AS operator_class "
            "ON operator_class.oid = indexed_class.operator_class_oid "
            "ORDER BY indexed_class.position), "
            "pg_get_expr(index_catalog.indpred, index_catalog.indrelid, TRUE), "
            "pg_get_expr(index_catalog.indexprs, index_catalog.indrelid, TRUE) "
            "FROM pg_index AS index_catalog "
            "JOIN pg_class AS index_class "
            "ON index_class.oid = index_catalog.indexrelid "
            "JOIN pg_class AS table_class "
            "ON table_class.oid = index_catalog.indrelid "
            "JOIN pg_namespace AS table_schema "
            "ON table_schema.oid = table_class.relnamespace "
            "JOIN pg_am AS access_method "
            "ON access_method.oid = index_class.relam "
            "WHERE table_schema.nspname = %s "
            "AND NOT index_catalog.indisprimary",
            (_SCHEMA,),
        ).fetchall()

        def _catalog_expression(value) -> Optional[str]:
            if value is None:
                return None
            return " ".join(str(value).split())

        actual_indexes: Dict[str, _IndexSpec] = {}
        invalid_indexes: List[str] = []
        included_column_indexes: List[str] = []
        for row in index_rows:
            (
                table,
                index_name,
                unique,
                valid,
                ready,
                live,
                access_method,
                attribute_count,
                key_count,
                keys,
                options,
                opclasses,
                predicate,
                expression,
            ) = row
            name = str(index_name)
            if not (bool(valid) and bool(ready) and bool(live)):
                invalid_indexes.append(name)
            if int(attribute_count) != int(key_count):
                included_column_indexes.append(name)
            actual_indexes[name] = _IndexSpec(
                table=str(table),
                unique=bool(unique),
                access_method=str(access_method),
                keys=tuple(_catalog_expression(key) or "" for key in keys),
                options=tuple(int(option) for option in options),
                opclasses=tuple(str(opclass) for opclass in opclasses),
                predicate=_catalog_expression(predicate),
                expression=_catalog_expression(expression),
            )

        allowed_index_names = set(required_index_specs).union(
            _OPTIONAL_INDEX_SPECS
        )
        actual_index_names = set(actual_indexes)
        missing_indexes = set(required_index_specs).difference(actual_index_names)
        unexpected_indexes = actual_index_names.difference(allowed_index_names)
        if missing_indexes:
            problems.append(f"missing indexes {sorted(missing_indexes)}")
        if unexpected_indexes:
            problems.append(f"unexpected indexes {sorted(unexpected_indexes)}")
        if invalid_indexes:
            problems.append(f"invalid indexes {sorted(invalid_indexes)}")
        if included_column_indexes:
            problems.append(
                "indexes with unexpected included columns "
                f"{sorted(included_column_indexes)}"
            )
        expected_present_indexes = dict(required_index_specs)
        for name, spec in _OPTIONAL_INDEX_SPECS.items():
            if name in actual_indexes:
                expected_present_indexes[name] = spec
        for name, expected in expected_present_indexes.items():
            actual = actual_indexes.get(name)
            if actual is not None and actual != expected:
                problems.append(
                    f"index {name} is {actual!r}, expected {expected!r}"
                )

        constraint_rows = conn.execute(
            "SELECT table_class.relname, constraint_catalog.conname, "
            "constraint_catalog.contype, constraint_catalog.condeferrable, "
            "constraint_catalog.condeferred, constraint_catalog.convalidated, "
            "ARRAY(SELECT column_attribute.attname FROM "
            "unnest(constraint_catalog.conkey) WITH ORDINALITY "
            "AS constrained_column(attribute_number, position) "
            "JOIN pg_attribute AS column_attribute "
            "ON column_attribute.attrelid = constraint_catalog.conrelid "
            "AND column_attribute.attnum = constrained_column.attribute_number "
            "ORDER BY constrained_column.position), "
            "referenced_schema.nspname, referenced_table.relname, "
            "ARRAY(SELECT referenced_attribute.attname FROM "
            "unnest(constraint_catalog.confkey) WITH ORDINALITY "
            "AS referenced_column(attribute_number, position) "
            "JOIN pg_attribute AS referenced_attribute "
            "ON referenced_attribute.attrelid = constraint_catalog.confrelid "
            "AND referenced_attribute.attnum = referenced_column.attribute_number "
            "ORDER BY referenced_column.position), "
            "constraint_catalog.confupdtype::text, "
            "constraint_catalog.confdeltype::text, "
            "constraint_catalog.confmatchtype::text "
            "FROM pg_constraint AS constraint_catalog "
            "JOIN pg_class AS table_class "
            "ON table_class.oid = constraint_catalog.conrelid "
            "JOIN pg_namespace AS table_schema "
            "ON table_schema.oid = table_class.relnamespace "
            "LEFT JOIN pg_class AS referenced_table "
            "ON referenced_table.oid = constraint_catalog.confrelid "
            "LEFT JOIN pg_namespace AS referenced_schema "
            "ON referenced_schema.oid = referenced_table.relnamespace "
            "WHERE table_schema.nspname = %s "
            "AND constraint_catalog.contype IN ('p', 'f', 'u', 'c', 'x')",
            (_SCHEMA,),
        ).fetchall()
        actual_constraint_specs: Dict[
            tuple[str, str], _ConstraintSpec
        ] = {}
        for (
            table,
            constraint_name,
            kind,
            deferrable,
            initially_deferred,
            validated,
            columns,
            referenced_schema,
            referenced_table,
            referenced_columns,
            update_action,
            delete_action,
            match_type,
        ) in constraint_rows:
            normalized_kind = str(kind)
            key = (str(table), str(constraint_name))
            actual_constraint_specs[key] = _ConstraintSpec(
                kind=normalized_kind,
                columns=tuple(str(column) for column in (columns or ())),
                referenced_schema=(
                    str(referenced_schema)
                    if referenced_schema is not None
                    else None
                ),
                referenced_table=(
                    str(referenced_table)
                    if referenced_table is not None
                    else None
                ),
                referenced_columns=tuple(
                    str(column) for column in (referenced_columns or ())
                ),
                update_action=(
                    str(update_action) if normalized_kind == "f" else None
                ),
                delete_action=(
                    str(delete_action) if normalized_kind == "f" else None
                ),
                match_type=(
                    str(match_type) if normalized_kind == "f" else None
                ),
                deferrable=bool(deferrable),
                initially_deferred=bool(initially_deferred),
                validated=bool(validated),
            )

        actual_constraint_names = set(actual_constraint_specs)
        required_constraint_names = set(required_constraint_specs)
        missing_constraints = required_constraint_names.difference(
            actual_constraint_names
        )
        unexpected_constraints = actual_constraint_names.difference(
            required_constraint_names
        )
        if missing_constraints:
            problems.append(f"missing constraints {sorted(missing_constraints)}")
        if unexpected_constraints:
            problems.append(
                f"unexpected constraints {sorted(unexpected_constraints)}"
            )
        for key, expected in required_constraint_specs.items():
            actual = actual_constraint_specs.get(key)
            if actual is not None and actual != expected:
                problems.append(
                    f"constraint {key[0]}.{key[1]} is {actual!r}, "
                    f"expected {expected!r}"
                )

        if problems:
            raise RuntimeError(
                f"Postgres session-store {label} catalog verification failed: "
                + "; ".join(problems)
            )

    def _assert_persisted_schema_markers(self, conn) -> str:
        """Verify persisted markers without mutating them.

        Fresh-store marker creation and the explicit v22→v26 transition live
        on separate paths. Keeping this runtime gate observation-only prevents
        an ordinary task boot from partly widening a v22 database before it
        refuses to serve it.
        """
        db_version = self._read_schema_version(conn)
        if db_version != EXPECTED_SCHEMA_VERSION:
            if db_version == 22:
                raise RuntimeError(
                    "Postgres session store schema migration required "
                    "(v22→v26). Drain every gateway task to zero and run the "
                    "explicit migration before starting this image."
                )
            raise RuntimeError(
                f"Postgres session store schema_version is {db_version}, but "
                f"this build expects {EXPECTED_SCHEMA_VERSION}. Refusing to "
                "write into a store owned by a different Hermes build."
            )
        row = conn.execute(
            f"SELECT value FROM {_SCHEMA}.state_meta WHERE key = %s",
            (_META_SURFACE_KEY,),
        ).fetchone()
        if row is None:
            raise RuntimeError(
                "Postgres session store is missing its schema surface marker; "
                "refusing to infer or repair it during runtime boot."
            )
        surface_marker = str(row[0])
        if surface_marker not in {
            EXPECTED_SCHEMA_SURFACE_SHA256,
            FORWARD_SCHEMA_SURFACE_SHA256,
        }:
            raise RuntimeError(
                "Postgres session store was initialized against a different "
                f"upstream schema surface (db={surface_marker}, accepted="
                f"{EXPECTED_SCHEMA_SURFACE_SHA256} or "
                f"{FORWARD_SCHEMA_SURFACE_SHA256}). Rebase drift — "
                "re-audit hermes_state_pg.py and migrate deliberately."
            )
        return surface_marker

    def _init_schema(self) -> None:  # pragma: no cover - safety net
        self._init_pg_schema()

    @contextmanager
    def _read_ctx(self):
        """Hold one pooled connection for a complete logical read operation."""
        with self._pool.connection() as raw:
            with raw.transaction():
                yield _TxnConn(raw)

    # ── Write executor ──────────────────────────────────────────────────

    def _execute_write(
        self,
        fn: Callable[[Any], T],
        patience_s: Optional[float] = None,
    ) -> T:
        """One transaction per write closure, with deadlock retry.

        Replaces SQLite's BEGIN IMMEDIATE + busy-retry: Postgres MVCC makes
        writer convoys structurally impossible; only deadlocks/serialization
        failures are worth retrying.
        """
        if patience_s is None:
            patience_s = self._WRITE_PATIENCE_S
        deadline = time.monotonic() + max(0.0, float(patience_s))

        def _sleep_before_retry() -> bool:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return False
            time.sleep(
                min(
                    remaining,
                    random.uniform(
                        self._WRITE_RETRY_MIN_S,
                        self._WRITE_RETRY_MAX_S,
                    ),
                )
            )
            return True

        while True:
            try:
                with self._pool.connection() as raw:
                    with raw.transaction():
                        result = fn(_TxnConn(raw))
                self._write_count += 1
                return result
            except sqlite3.Error as exc:
                if _is_retryable_pg_error(exc) and _sleep_before_retry():
                    continue
                raise
            except Exception as exc:
                import psycopg

                if isinstance(exc, psycopg.Error):
                    translated = _translate_exception(exc)
                    if (
                        _is_retryable_pg_error(translated)
                        and _sleep_before_retry()
                    ):
                        continue
                    raise translated from exc
                raise

    # ── Lifecycle / SQLite maintenance obsoleted by Postgres ────────────

    def close(self) -> None:
        if self._closed:
            return
        self._stop_token_writer()
        hook, self._token_atexit_hook = self._token_atexit_hook, None
        if hook is not None:
            hermes_state.atexit.unregister(hook)
        self._closed = True
        self._conn = None
        try:
            self._pool.close()
        except Exception:
            pass

    @staticmethod
    def _lock_holder_process_is_dead(holder: str) -> bool:
        """Distributed Postgres leases recover by TTL, never local PID probes."""
        return False

    def _message_column_names(self, conn) -> List[str]:
        """Return Postgres message columns in physical order, cached per handle."""
        if self._message_columns_cache:
            return self._message_columns_cache
        rows = conn.execute(
            "SELECT column_name FROM information_schema.columns "
            "WHERE table_schema = ? AND table_name = 'messages' "
            "ORDER BY ordinal_position",
            (_SCHEMA,),
        ).fetchall()
        self._message_columns_cache = [str(row[0]) for row in rows]
        return self._message_columns_cache

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

    def _search_messages_impl(
        self,
        query: str,
        source_filter: List[str] = None,
        exclude_sources: List[str] = None,
        role_filter: List[str] = None,
        limit: int = 20,
        offset: int = 0,
        sort: str = None,
        include_inactive: bool = False,
        fields: Optional[Collection[str]] = None,
    ) -> List[Dict[str, Any]]:
        """Postgres port of :meth:`SessionDB.search_messages`.

        Same contract and result shape; tsvector('simple') replaces the FTS5
        unicode61 table and per-token ILIKE replaces the trigram/LIKE CJK
        paths (ILIKE substring match has no 3-char minimum, so short-CJK
        queries take the same path as long ones).
        """
        result_fields = self._search_message_fields(fields)
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

        return self._finalize_search_matches(
            matches,
            result_fields=result_fields,
        )


def inspect_v22_migration_precondition(dsn: str) -> Dict[str, Any]:
    """Read-only proof that *dsn* names the exact v22 migration source.

    The apply path repeats this proof under the bootstrap advisory lock. This
    separate snapshot exists so the migration command's default dry-run can
    catch a wrong DSN without owning any create/repair behavior.
    """
    dsn = (dsn or "").strip()
    if not dsn:
        raise ValueError("Postgres migration preflight requires a DSN")
    assert_schema_compat()

    import psycopg
    from psycopg import ClientCursor

    with psycopg.connect(_normalize_dsn(dsn), autocommit=True) as conn:
        conn.cursor_factory = ClientCursor
        with conn.transaction():
            conn.execute(
                "SET TRANSACTION ISOLATION LEVEL REPEATABLE READ READ ONLY"
            )
            namespace = conn.execute(
                "SELECT to_regnamespace(%s)", (_SCHEMA,)
            ).fetchone()
            if not namespace or namespace[0] is None:
                raise RuntimeError(
                    "Postgres v22→v26 migration requires an existing "
                    "hermes_state schema; the selected database has none."
                )
            version_table = conn.execute(
                "SELECT to_regclass(%s)", (f"{_SCHEMA}.schema_version",)
            ).fetchone()
            if not version_table or version_table[0] is None:
                raise RuntimeError(
                    "Postgres v22→v26 migration requires an existing "
                    "hermes_state.schema_version table; the selected database "
                    "does not have one."
                )
            PgSessionDB._assert_v22_migration_precondition(conn)

    return {
        "backend": "postgres",
        "schema_version": 22,
        "surface_marker": _V22_SCHEMA_SURFACE_SHA256,
    }


def _inspect_v26_surface(conn) -> Dict[str, Any]:
    """Verify and classify one of the two audited v26 catalog surfaces."""
    version = PgSessionDB._read_schema_version(conn)
    if version != EXPECTED_SCHEMA_VERSION:
        raise RuntimeError(
            "Postgres v26 surface migration requires schema_version 26 "
            f"exactly (found {version}); refusing to mutate this store."
        )
    surface = PgSessionDB._read_surface_marker(conn)
    if surface == EXPECTED_SCHEMA_SURFACE_SHA256:
        PgSessionDB._verify_catalog(
            conn, surface_marker=EXPECTED_SCHEMA_SURFACE_SHA256
        )
        migration_required = True
    elif surface == FORWARD_SCHEMA_SURFACE_SHA256:
        PgSessionDB._verify_catalog(
            conn, surface_marker=FORWARD_SCHEMA_SURFACE_SHA256
        )
        migration_required = False
    else:
        raise RuntimeError(
            "Postgres v26 surface migration requires the exact current or "
            f"destination marker (found {surface!r}); refusing unknown drift."
        )
    return {
        "backend": "postgres",
        "schema_version": EXPECTED_SCHEMA_VERSION,
        "surface_marker": surface,
        "target_surface_marker": FORWARD_SCHEMA_SURFACE_SHA256,
        "migration_required": migration_required,
    }


def inspect_v26_surface_migration_precondition(dsn: str) -> Dict[str, Any]:
    """Read-only proof of the exact AE-240 migration source or destination."""
    dsn = (dsn or "").strip()
    if not dsn:
        raise ValueError("Postgres migration preflight requires a DSN")
    assert_schema_compat()

    import psycopg
    from psycopg import ClientCursor

    with psycopg.connect(_normalize_dsn(dsn), autocommit=True) as conn:
        conn.cursor_factory = ClientCursor
        with conn.transaction():
            conn.execute(
                "SET TRANSACTION ISOLATION LEVEL REPEATABLE READ READ ONLY"
            )
            namespace = conn.execute(
                "SELECT to_regnamespace(%s)", (_SCHEMA,)
            ).fetchone()
            if not namespace or namespace[0] is None:
                raise RuntimeError(
                    "Postgres v26 surface migration requires an existing "
                    "hermes_state schema; the selected database has none."
                )
            version_table = conn.execute(
                "SELECT to_regclass(%s)", (f"{_SCHEMA}.schema_version",)
            ).fetchone()
            if not version_table or version_table[0] is None:
                raise RuntimeError(
                    "Postgres v26 surface migration requires an existing "
                    "hermes_state.schema_version table."
                )
            return _inspect_v26_surface(conn)


def _migrate_v26_surface_on_connection(conn) -> Dict[str, Any]:
    """Apply or verify the AE-240 expansion in the caller's transaction."""
    conn.execute(
        "SELECT set_config('lock_timeout', %s, true)",
        (_V26_MIGRATION_LOCK_TIMEOUT,),
    )
    conn.execute(
        "SELECT set_config('statement_timeout', %s, true)",
        (_V26_MIGRATION_STATEMENT_TIMEOUT,),
    )
    current = _inspect_v26_surface(conn)
    if not current["migration_required"]:
        return current

    conn.execute(PG_V26_FORWARD_SURFACE_SQL)
    PgSessionDB._verify_catalog(
        conn, surface_marker=FORWARD_SCHEMA_SURFACE_SHA256
    )
    updated = conn.execute(
        f"UPDATE {_SCHEMA}.state_meta SET value = %s "
        "WHERE key = %s AND value = %s RETURNING value",
        (
            FORWARD_SCHEMA_SURFACE_SHA256,
            _META_SURFACE_KEY,
            EXPECTED_SCHEMA_SURFACE_SHA256,
        ),
    ).fetchone()
    if updated is None:
        raise RuntimeError(
            "Postgres v26 surface marker changed during migration; "
            "rolling back instead of inferring ownership."
        )
    destination = _inspect_v26_surface(conn)
    destination["migration_applied"] = True
    return destination


def migrate_v26_surface(dsn: str) -> Dict[str, Any]:
    """Apply the one-column AE-240 expansion under the bootstrap lock.

    The transaction is idempotent only at the exact audited destination.
    Runtime boot never calls this function.
    """
    dsn = (dsn or "").strip()
    if not dsn:
        raise ValueError("Postgres migration requires a DSN")
    assert_schema_compat()

    import psycopg
    from psycopg import ClientCursor

    with psycopg.connect(_normalize_dsn(dsn), autocommit=True) as conn:
        conn.cursor_factory = ClientCursor
        conn.execute(
            "SELECT set_config('lock_timeout', %s, false)",
            (_V26_MIGRATION_ADVISORY_TIMEOUT,),
        )
        conn.execute(
            "SELECT pg_advisory_lock(%s)", (PgSessionDB._BOOTSTRAP_LOCK_KEY,)
        )
        try:
            with conn.transaction():
                return _migrate_v26_surface_on_connection(conn)
        finally:
            try:
                conn.execute(
                    "SELECT pg_advisory_unlock(%s)",
                    (PgSessionDB._BOOTSTRAP_LOCK_KEY,),
                )
            except Exception as exc:
                logger.warning(
                    "failed to release session-store migration advisory lock "
                    "(%s, sqlstate=%s)",
                    type(exc).__name__,
                    getattr(exc, "sqlstate", None) or "unknown",
                )
