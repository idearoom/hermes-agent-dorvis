"""Postgres-backed Responses API store (IdeaRoom D6 / AE-61).

A drop-in replacement for the SQLite ``ResponseStore`` in ``api_server.py`` that
keeps the exact same 8-method interface but persists to PostgreSQL instead of a
SQLite file on the agent home.

Why: on AWS the agent home is an EFS (NFS) mount, and SQLite over NFS is ~1000x
slower than local disk (measured 0.2 vs 201 txn/s, 8 concurrent writers). Every
stored turn writes the full transcript here, so under concurrency the SQLite
store serializes on the NFS write lock and turns stall for 100-240s with the CPU
idle. Postgres MVCC + a connection pool removes the single-writer bottleneck.
See docs/platform/ae-61-efs-sqlite-diagnostic-2026-06-13.md (in idearoom-agents).

Additive by design: ``api_server`` selects this backend only when
``HERMES_RESPONSE_STORE_DSN`` is set (the AWS Hermes task sets it to the RDS DSN);
absent, it uses the SQLite ``ResponseStore`` unchanged, so local dev and the Mac
Mini profiles are untouched. ``psycopg`` is imported lazily for the same reason —
environments that never set the DSN don't need the dependency installed.

Tables live in a dedicated ``hermes_gw`` schema so the gateway's storage never
collides with the web app's drizzle-managed tables in the same database (D6:
gateway + web share one RDS, JOIN-able by ids, but own their DDL separately).
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# ``max_size`` is still accepted by ``PgResponseStore.__init__`` for interface
# parity with the SQLite store, but Postgres is the durable response store on
# AWS. It must not evict model-visible conversation state just because more
# turns arrived later.
DEFAULT_MAX_STORED_RESPONSES = None

_SCHEMA = "hermes_gw"


def _dumps(obj: Any) -> str:
    """json.dumps with default=str — parity with the SQLite store's put()."""
    return json.dumps(obj, default=str)


def _normalize_dsn(dsn: str) -> str:
    """Make a node-pg-shaped DSN safe for libpq/psycopg.

    The shared RDS secret is written for the web app's node ``pg`` driver and
    carries ``sslmode=no-verify`` (encrypt, skip cert verification). libpq has no
    ``no-verify`` value and would reject it; ``require`` is the libpq equivalent
    (encrypt without CA/hostname verification). One secret, two drivers.
    """
    return dsn.replace("sslmode=no-verify", "sslmode=require")


class PgResponseStore:
    """Postgres-backed durable store for Responses API state.

    Interface parity with ``api_server.ResponseStore``:
    ``get / put / delete / get_conversation / set_conversation / close / __len__``.
    Thread-safe via a psycopg connection pool (the gateway calls these
    synchronously from async handlers across threads, like the SQLite store with
    ``check_same_thread=False``).
    """

    def __init__(
        self,
        dsn: str,
        max_size: Optional[int] = DEFAULT_MAX_STORED_RESPONSES,
        *,
        min_pool: int = 1,
        max_pool: int = 8,
    ) -> None:
        # Kept for constructor compatibility only. Unlike the SQLite fallback,
        # the Postgres adapter is durable state and intentionally does not apply
        # the old 100-response LRU cap.
        self._max_size = max_size
        # Lazy imports: only the Postgres path pulls these in.
        from psycopg import errors as psycopg_errors
        from psycopg.types.json import Jsonb  # noqa: F401  (used in put)
        from psycopg_pool import ConnectionPool

        self._Jsonb = Jsonb
        self._schema_recovery_errors = (
            psycopg_errors.AdminShutdown,
            psycopg_errors.InvalidSchemaName,
            psycopg_errors.UndefinedTable,
        )
        # open=True connects eagerly so a bad DSN fails fast at construction
        # (the factory catches and falls back to SQLite with a loud error).
        self._pool = ConnectionPool(
            conninfo=_normalize_dsn(dsn),
            min_size=min_pool,
            max_size=max_pool,
            kwargs={"autocommit": True},
            open=True,
            name="hermes-response-store",
        )
        self._closed = False
        self._init_schema()

    def _init_schema(self) -> None:
        with self._pool.connection() as conn:
            conn.execute(f"CREATE SCHEMA IF NOT EXISTS {_SCHEMA}")
            conn.execute(
                f"""CREATE TABLE IF NOT EXISTS {_SCHEMA}.responses (
                    response_id TEXT PRIMARY KEY,
                    data JSONB NOT NULL,
                    accessed_at DOUBLE PRECISION NOT NULL
                )"""
            )
            # accessed_at remains useful for diagnostics and future explicit
            # retention policies, even though this durable store does not apply
            # automatic LRU eviction.
            conn.execute(
                f"CREATE INDEX IF NOT EXISTS idx_responses_accessed_at "
                f"ON {_SCHEMA}.responses (accessed_at)"
            )
            conn.execute(
                f"""CREATE TABLE IF NOT EXISTS {_SCHEMA}.conversations (
                    name TEXT PRIMARY KEY,
                    response_id TEXT NOT NULL
                )"""
            )

    def _with_schema_retry(self, operation: str, fn):
        try:
            return fn()
        except self._schema_recovery_errors:
            logger.warning(
                "ResponseStore: Postgres schema unavailable during %s; "
                "reinitializing schema and retrying once.",
                operation,
                exc_info=True,
            )
            self._init_schema()
            return fn()

    def get(self, response_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a stored response by ID and refresh diagnostic access time."""
        def _get():
            with self._pool.connection() as conn:
                row = conn.execute(
                    f"SELECT data FROM {_SCHEMA}.responses WHERE response_id = %s",
                    (response_id,),
                ).fetchone()
                if row is None:
                    return None
                conn.execute(
                    f"UPDATE {_SCHEMA}.responses SET accessed_at = %s WHERE response_id = %s",
                    (time.time(), response_id),
                )
                # psycopg adapts jsonb -> Python dict/list directly; no json.loads.
                # (Postgres enforces valid JSON on write, so the SQLite store's
                # corrupted-JSON self-heal path is structurally impossible here.)
                return row[0]

        return self._with_schema_retry("get", _get)

    def put(self, response_id: str, data: Dict[str, Any]) -> None:
        """Store a response without automatic eviction."""
        def _put():
            with self._pool.connection() as conn:
                conn.execute(
                    f"""INSERT INTO {_SCHEMA}.responses (response_id, data, accessed_at)
                        VALUES (%s, %s, %s)
                        ON CONFLICT (response_id)
                        DO UPDATE SET data = EXCLUDED.data, accessed_at = EXCLUDED.accessed_at""",
                    (response_id, self._Jsonb(data, dumps=_dumps), time.time()),
                )

        self._with_schema_retry("put", _put)

    def delete(self, response_id: str) -> bool:
        """Remove a response from the store. Returns True if found and deleted."""
        def _delete():
            with self._pool.connection() as conn:
                conn.execute(
                    f"DELETE FROM {_SCHEMA}.conversations WHERE response_id = %s",
                    (response_id,),
                )
                cur = conn.execute(
                    f"DELETE FROM {_SCHEMA}.responses WHERE response_id = %s",
                    (response_id,),
                )
                return cur.rowcount > 0

        return self._with_schema_retry("delete", _delete)

    def get_conversation(self, name: str) -> Optional[str]:
        """Get the latest response_id for a conversation name."""
        def _get_conversation():
            with self._pool.connection() as conn:
                row = conn.execute(
                    f"SELECT response_id FROM {_SCHEMA}.conversations WHERE name = %s",
                    (name,),
                ).fetchone()
                return row[0] if row else None

        return self._with_schema_retry("get_conversation", _get_conversation)

    def set_conversation(self, name: str, response_id: str) -> None:
        """Map a conversation name to its latest response_id."""
        def _set_conversation():
            with self._pool.connection() as conn:
                conn.execute(
                    f"""INSERT INTO {_SCHEMA}.conversations (name, response_id)
                        VALUES (%s, %s)
                        ON CONFLICT (name)
                        DO UPDATE SET response_id = EXCLUDED.response_id""",
                    (name, response_id),
                )

        self._with_schema_retry("set_conversation", _set_conversation)

    def close(self) -> None:
        """Close the connection pool."""
        if self._closed:
            return
        self._closed = True
        try:
            self._pool.close()
        except Exception:
            pass

    def __len__(self) -> int:
        def _len():
            with self._pool.connection() as conn:
                row = conn.execute(
                    f"SELECT COUNT(*) FROM {_SCHEMA}.responses"
                ).fetchone()
                return row[0] if row else 0

        return self._with_schema_retry("len", _len)
