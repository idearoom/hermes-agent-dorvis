"""Postgres-backed Responses API store (IdeaRoom D6 / AE-61).

A drop-in replacement for the SQLite ``ResponseStore`` in ``api_server.py`` that
keeps the same data and lifecycle interface but persists to PostgreSQL instead
of a SQLite file on the agent home.

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
_SCHEMA_CONTRACT_VERSION = 2
_TERMINAL_STATUSES = frozenset(
    {"completed", "failed", "cancelled", "canceled", "incomplete"}
)
_LEGACY_TERMINAL_FUNCTION_BODY = """
BEGIN
    IF NEW.owner_id IS NULL AND NEW.owner_epoch IS NULL THEN
        NEW.terminal := COALESCE(
            NEW.data -> 'response' ->> 'status', 'completed'
        ) IN (
            'completed', 'failed', 'cancelled',
            'canceled', 'incomplete'
        );
    END IF;
    RETURN NEW;
END
"""
_OWNED_RESPONSE_FENCE_FUNCTION_BODY = """
BEGIN
    IF TG_OP = 'DELETE' THEN
        IF (OLD.owner_id IS NOT NULL OR OLD.owner_epoch IS NOT NULL)
           AND NOT OLD.terminal THEN
            RETURN NULL;
        END IF;
        RETURN OLD;
    END IF;

    IF OLD.owner_id IS NULL AND OLD.owner_epoch IS NULL THEN
        RETURN NEW;
    END IF;

    IF NEW.response_id IS NOT DISTINCT FROM OLD.response_id
       AND NEW.data IS NOT DISTINCT FROM OLD.data
       AND NEW.owner_id IS NOT DISTINCT FROM OLD.owner_id
       AND NEW.owner_epoch IS NOT DISTINCT FROM OLD.owner_epoch
       AND NEW.terminal IS NOT DISTINCT FROM OLD.terminal THEN
        RETURN NEW;
    END IF;

    IF current_setting(
           'hermes_gw.response_write_owner_id', true
       ) = OLD.owner_id
       AND current_setting(
           'hermes_gw.response_write_owner_epoch', true
       ) = OLD.owner_epoch
       AND NEW.response_id IS NOT DISTINCT FROM OLD.response_id
       AND NEW.owner_id IS NOT DISTINCT FROM OLD.owner_id
       AND NEW.owner_epoch IS NOT DISTINCT FROM OLD.owner_epoch THEN
        RETURN NEW;
    END IF;

    RAISE EXCEPTION 'unauthorized owned response mutation for %', OLD.response_id;
END
"""
_CONVERSATION_DELETE_FENCE_FUNCTION_BODY = """
DECLARE
    mapped_owner_id TEXT;
    mapped_owner_epoch TEXT;
    mapped_terminal BOOLEAN;
BEGIN
    SELECT owner_id, owner_epoch, terminal
    INTO mapped_owner_id, mapped_owner_epoch, mapped_terminal
    FROM hermes_gw.responses
    WHERE response_id = OLD.response_id;

    IF NOT FOUND
       OR (mapped_owner_id IS NULL AND mapped_owner_epoch IS NULL)
       OR mapped_terminal THEN
        RETURN OLD;
    END IF;
    RETURN NULL;
END
"""


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


def _record_is_terminal(data: Dict[str, Any]) -> bool:
    response = data.get("response") if isinstance(data, dict) else None
    if not isinstance(response, dict) or not response.get("status"):
        return True
    return str(response.get("status")) in _TERMINAL_STATUSES


class PgResponseStore:
    """Postgres-backed durable store for Responses API state.

    Interface parity with ``api_server.ResponseStore``, including owned
    response claim/transition and terminal-only deletion operations.
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
        try:
            self._init_schema()
        except Exception:
            self._closed = True
            try:
                self._pool.close()
            except Exception:
                logger.debug(
                    "ResponseStore: failed to close pool after schema init error",
                    exc_info=True,
                )
            raise

    def _init_schema(self) -> None:
        with self._pool.connection() as conn:
            # Ordinary boots must remain read-only with respect to the schema.
            # The version row is written only after the full migration and its
            # exact trigger/FK checks commit. Core relation probes keep a stale
            # marker from hiding a partially removed schema; an operation that
            # later detects a missing object re-enters this path via the
            # one-shot schema retry.
            try:
                current = conn.execute(
                    f"""SELECT /* hermes_response_schema_contract_fast_path */
                               contract_version,
                               to_regclass('{_SCHEMA}.responses') IS NOT NULL,
                               to_regclass('{_SCHEMA}.conversations') IS NOT NULL,
                               to_regprocedure(
                                   '{_SCHEMA}.sync_legacy_response_terminal()'
                               ) IS NOT NULL,
                               EXISTS (
                                   SELECT 1 FROM pg_trigger
                                   WHERE tgname = 'sync_legacy_response_terminal'
                                     AND tgrelid =
                                         '{_SCHEMA}.responses'::regclass
                               ),
                               to_regprocedure(
                                   '{_SCHEMA}.fence_owned_response()'
                               ) IS NOT NULL,
                               EXISTS (
                                   SELECT 1 FROM pg_trigger
                                   WHERE tgname = 'fence_owned_response'
                                     AND tgrelid =
                                         '{_SCHEMA}.responses'::regclass
                               ),
                               to_regprocedure(
                                   '{_SCHEMA}.fence_owned_response_conversation_delete()'
                               ) IS NOT NULL,
                               EXISTS (
                                   SELECT 1 FROM pg_trigger
                                   WHERE tgname =
                                       'fence_owned_response_conversation_delete'
                                     AND tgrelid =
                                         '{_SCHEMA}.conversations'::regclass
                               ),
                               EXISTS (
                                   SELECT 1 FROM pg_constraint
                                   WHERE conname =
                                       'conversations_response_id_fkey'
                                     AND conrelid =
                                         '{_SCHEMA}.conversations'::regclass
                               ),
                               to_regclass(
                                   '{_SCHEMA}.idx_responses_accessed_at'
                               ) IS NOT NULL,
                               EXISTS (
                                   SELECT 1
                                   FROM pg_attribute
                                   WHERE attrelid =
                                       '{_SCHEMA}.responses'::regclass
                                     AND attname = 'owner_heartbeat_at'
                                     AND atttypid = 'float8'::regtype
                                     AND NOT attisdropped
                               )
                        FROM {_SCHEMA}.schema_contract
                        WHERE singleton = TRUE"""
                ).fetchone()
            except self._schema_recovery_errors:
                current = None
            if (
                current is not None
                and current[0] == _SCHEMA_CONTRACT_VERSION
                and all(bool(value) for value in current[1:])
            ):
                # The marker avoids every DDL statement and migration lock,
                # while these exact read-only checks retain fail-closed
                # protection against a same-named trigger/FK with altered
                # semantics.
                self._ensure_legacy_terminal_function(conn)
                self._ensure_legacy_terminal_trigger(conn)
                self._ensure_owned_response_fence_function(conn)
                self._ensure_owned_response_fence_trigger(conn)
                self._ensure_conversation_delete_fence_function(conn)
                self._ensure_conversation_delete_fence_trigger(conn)
                self._ensure_conversation_response_fk(conn)
                return
            if (
                current is not None
                and current[0] is not None
                and int(current[0]) > _SCHEMA_CONTRACT_VERSION
            ):
                raise RuntimeError(
                    "Responses schema contract is newer than this runtime: "
                    f"installed={current[0]} supported={_SCHEMA_CONTRACT_VERSION}"
                )

            conn.execute(f"CREATE SCHEMA IF NOT EXISTS {_SCHEMA}")
            conn.execute(
                f"""CREATE TABLE IF NOT EXISTS {_SCHEMA}.responses (
                    response_id TEXT PRIMARY KEY,
                    data JSONB NOT NULL,
                    accessed_at DOUBLE PRECISION NOT NULL,
                    owner_id TEXT,
                    owner_epoch TEXT,
                    owner_heartbeat_at DOUBLE PRECISION,
                    terminal BOOLEAN NOT NULL DEFAULT TRUE
                )"""
            )
            conn.execute(
                f"""CREATE TABLE IF NOT EXISTS {_SCHEMA}.conversations (
                    name TEXT PRIMARY KEY,
                    response_id TEXT NOT NULL REFERENCES {_SCHEMA}.responses(response_id)
                        ON DELETE CASCADE
                )"""
            )
            # Perform every compatibility change while old writers are blocked.
            # This avoids exposing DEFAULT TRUE without the legacy classifier,
            # and closes the cleanup/FK window where an old task could recreate
            # a dangling conversation mapping.
            with conn.transaction():
                conn.execute(
                    f"LOCK TABLE {_SCHEMA}.responses, {_SCHEMA}.conversations "
                    "IN SHARE ROW EXCLUSIVE MODE"
                )
                conn.execute(
                    f"ALTER TABLE {_SCHEMA}.responses "
                    "ADD COLUMN IF NOT EXISTS owner_id TEXT"
                )
                conn.execute(
                    f"ALTER TABLE {_SCHEMA}.responses "
                    "ADD COLUMN IF NOT EXISTS owner_epoch TEXT"
                )
                conn.execute(
                    f"ALTER TABLE {_SCHEMA}.responses "
                    "ADD COLUMN IF NOT EXISTS owner_heartbeat_at "
                    "DOUBLE PRECISION"
                )
                conn.execute(
                    f"ALTER TABLE {_SCHEMA}.responses "
                    "ADD COLUMN IF NOT EXISTS terminal "
                    "BOOLEAN NOT NULL DEFAULT TRUE"
                )
                conn.execute(
                    f"ALTER TABLE {_SCHEMA}.responses "
                    "ALTER COLUMN terminal SET DEFAULT TRUE"
                )
                self._ensure_legacy_terminal_function(conn)
                self._ensure_legacy_terminal_trigger(conn)
                self._ensure_owned_response_fence_function(conn)
                self._ensure_owned_response_fence_trigger(conn)
                self._ensure_conversation_delete_fence_function(conn)
                self._ensure_conversation_delete_fence_trigger(conn)
                # Backstop rows written before the trigger existed while old
                # and new gateway tasks overlap.
                conn.execute(
                    f"""UPDATE {_SCHEMA}.responses
                        SET terminal = COALESCE(
                            data -> 'response' ->> 'status', 'completed'
                        ) IN (
                            'completed', 'failed', 'cancelled',
                            'canceled', 'incomplete'
                        )
                        WHERE owner_id IS NULL
                          AND owner_epoch IS NULL
                          AND terminal IS DISTINCT FROM (
                              COALESCE(
                                  data -> 'response' ->> 'status', 'completed'
                              ) IN (
                                  'completed', 'failed', 'cancelled',
                                  'canceled', 'incomplete'
                              )
                          )"""
                )
                # Run after terminal classification: columns added to a legacy
                # table start with terminal=TRUE, so doing this first would
                # miss every ownerless in-progress row.
                conn.execute(
                    f"""UPDATE {_SCHEMA}.responses
                        SET owner_heartbeat_at = %s
                        WHERE terminal = FALSE
                          AND owner_heartbeat_at IS NULL""",
                    (time.time(),),
                )
                conn.execute(
                    f"""DELETE FROM {_SCHEMA}.conversations AS c
                        WHERE NOT EXISTS (
                            SELECT 1 FROM {_SCHEMA}.responses AS r
                            WHERE r.response_id = c.response_id
                        )"""
                )
                self._ensure_conversation_response_fk(conn)
                conn.execute(
                    f"""CREATE TABLE IF NOT EXISTS {_SCHEMA}.schema_contract (
                            singleton BOOLEAN PRIMARY KEY DEFAULT TRUE
                                CHECK (singleton),
                            contract_version INTEGER NOT NULL
                        )"""
                )
                conn.execute(
                    f"""INSERT INTO {_SCHEMA}.schema_contract
                            (singleton, contract_version)
                        VALUES (TRUE, %s)
                        ON CONFLICT (singleton) DO UPDATE
                        SET contract_version = GREATEST(
                            {_SCHEMA}.schema_contract.contract_version,
                            EXCLUDED.contract_version
                        )""",
                    (_SCHEMA_CONTRACT_VERSION,),
                )
            # accessed_at remains useful for diagnostics and future explicit
            # retention policies, even though this durable store does not apply
            # automatic LRU eviction.
            conn.execute(
                f"CREATE INDEX IF NOT EXISTS idx_responses_accessed_at "
                f"ON {_SCHEMA}.responses (accessed_at)"
            )

    @staticmethod
    def _ensure_legacy_terminal_function(conn) -> None:
        """Create the compatibility function, or reject a namesake contract."""
        normalized_body = " ".join(_LEGACY_TERMINAL_FUNCTION_BODY.split())
        row = conn.execute(
            f"""SELECT /* hermes_response_legacy_function_contract */
                       l.lanname = 'plpgsql'
                   AND p.prokind = 'f'
                   AND p.prorettype = 'pg_catalog.trigger'::regtype
                   AND p.pronargs = 0
                   AND p.provolatile = 'v'
                   AND NOT p.proisstrict
                   AND NOT p.prosecdef
                   AND NOT p.proleakproof
                   AND p.proparallel = 'u'
                   AND p.proconfig IS NULL
                   AND btrim(
                           regexp_replace(
                               p.prosrc, '[[:space:]]+', ' ', 'g'
                           )
                       ) = %s
                FROM pg_proc AS p
                JOIN pg_namespace AS n ON n.oid = p.pronamespace
                JOIN pg_language AS l ON l.oid = p.prolang
                WHERE n.nspname = '{_SCHEMA}'
                  AND p.proname = 'sync_legacy_response_terminal'
                  AND p.pronargs = 0""",
            (normalized_body,),
        ).fetchone()
        if row is not None:
            if not bool(row[0]):
                raise RuntimeError(
                    "Existing Hermes legacy terminal function has unsafe semantics"
                )
            return
        conn.execute(
            f"""CREATE FUNCTION {_SCHEMA}.sync_legacy_response_terminal()
                RETURNS trigger
                LANGUAGE plpgsql
                VOLATILE
                CALLED ON NULL INPUT
                SECURITY INVOKER
                PARALLEL UNSAFE
                AS $function${_LEGACY_TERMINAL_FUNCTION_BODY}$function$"""
        )

    @staticmethod
    def _ensure_legacy_terminal_trigger(conn) -> None:
        """Install the exact old-writer classifier trigger, never a namesake."""
        row = conn.execute(
            f"""SELECT /* hermes_response_legacy_trigger_contract */
                       t.tgfoid =
                           '{_SCHEMA}.sync_legacy_response_terminal()'::regprocedure
                   AND t.tgtype = 23
                   AND t.tgenabled = 'O'
                   AND NOT t.tgisinternal
                   AND t.tgqual IS NULL
                   AND t.tgnargs = 0
                   AND t.tgoldtable IS NULL
                   AND t.tgnewtable IS NULL
                   AND t.tgattr::text = concat_ws(
                       ' ',
                       (
                           SELECT attnum FROM pg_attribute
                           WHERE attrelid = '{_SCHEMA}.responses'::regclass
                             AND attname = 'data'
                       ),
                       (
                           SELECT attnum FROM pg_attribute
                           WHERE attrelid = '{_SCHEMA}.responses'::regclass
                             AND attname = 'owner_id'
                       ),
                       (
                           SELECT attnum FROM pg_attribute
                           WHERE attrelid = '{_SCHEMA}.responses'::regclass
                             AND attname = 'owner_epoch'
                       )
                   )
                FROM pg_trigger AS t
                WHERE t.tgname = 'sync_legacy_response_terminal'
                  AND t.tgrelid = '{_SCHEMA}.responses'::regclass"""
        ).fetchone()
        if row is not None:
            if not bool(row[0]):
                raise RuntimeError(
                    "Existing Hermes legacy terminal trigger has unsafe semantics"
                )
            return
        conn.execute(
            f"""CREATE TRIGGER sync_legacy_response_terminal
                BEFORE INSERT OR UPDATE OF data, owner_id, owner_epoch
                ON {_SCHEMA}.responses
                FOR EACH ROW
                EXECUTE FUNCTION {_SCHEMA}.sync_legacy_response_terminal()"""
        )

    @staticmethod
    def _ensure_owned_response_fence_function(conn) -> None:
        """Install the exact mixed-version mutation/delete fence function.

        Unauthorized semantic updates raise instead of becoming silent no-ops:
        the old cancel handler ignores ``put`` rowcount and would otherwise
        return a false 200. Active deletes return a no-op so its old DELETE
        handler observes rowcount zero/404 while the row remains intact.
        """
        PgResponseStore._ensure_trigger_function(
            conn,
            function_name="fence_owned_response",
            function_body=_OWNED_RESPONSE_FENCE_FUNCTION_BODY,
            contract_marker="hermes_response_owned_fence_function_contract",
            error_label="owned response fence function",
        )

    @staticmethod
    def _ensure_owned_response_fence_trigger(conn) -> None:
        """Protect owned rows from legacy updates and active deletion."""
        row = conn.execute(
            f"""SELECT /* hermes_response_owned_fence_trigger_contract */
                       t.tgfoid =
                           '{_SCHEMA}.fence_owned_response()'::regprocedure
                   AND t.tgtype = 27
                   AND t.tgenabled = 'O'
                   AND NOT t.tgisinternal
                   AND t.tgqual IS NULL
                   AND t.tgnargs = 0
                   AND t.tgoldtable IS NULL
                   AND t.tgnewtable IS NULL
                   AND t.tgattr::text = ''
                FROM pg_trigger AS t
                WHERE t.tgname = 'fence_owned_response'
                  AND t.tgrelid = '{_SCHEMA}.responses'::regclass"""
        ).fetchone()
        if row is not None:
            if not bool(row[0]):
                raise RuntimeError(
                    "Existing owned response fence trigger has unsafe semantics"
                )
            return
        conn.execute(
            f"""CREATE TRIGGER fence_owned_response
                BEFORE UPDATE OR DELETE ON {_SCHEMA}.responses
                FOR EACH ROW
                EXECUTE FUNCTION {_SCHEMA}.fence_owned_response()"""
        )

    @staticmethod
    def _ensure_conversation_delete_fence_function(conn) -> None:
        """Install the exact active-owned mapping delete fence function."""
        PgResponseStore._ensure_trigger_function(
            conn,
            function_name="fence_owned_response_conversation_delete",
            function_body=_CONVERSATION_DELETE_FENCE_FUNCTION_BODY,
            contract_marker=(
                "hermes_response_conversation_delete_fence_function_contract"
            ),
            error_label="conversation delete fence function",
        )

    @staticmethod
    def _ensure_conversation_delete_fence_trigger(conn) -> None:
        """Keep legacy DELETE from detaching an active owned response."""
        row = conn.execute(
            f"""SELECT /* hermes_response_conversation_delete_fence_trigger_contract */
                       t.tgfoid =
                           '{_SCHEMA}.fence_owned_response_conversation_delete()'
                           ::regprocedure
                   AND t.tgtype = 11
                   AND t.tgenabled = 'O'
                   AND NOT t.tgisinternal
                   AND t.tgqual IS NULL
                   AND t.tgnargs = 0
                   AND t.tgoldtable IS NULL
                   AND t.tgnewtable IS NULL
                   AND t.tgattr::text = ''
                FROM pg_trigger AS t
                WHERE t.tgname = 'fence_owned_response_conversation_delete'
                  AND t.tgrelid = '{_SCHEMA}.conversations'::regclass"""
        ).fetchone()
        if row is not None:
            if not bool(row[0]):
                raise RuntimeError(
                    "Existing conversation delete fence trigger has unsafe semantics"
                )
            return
        conn.execute(
            f"""CREATE TRIGGER fence_owned_response_conversation_delete
                BEFORE DELETE ON {_SCHEMA}.conversations
                FOR EACH ROW
                EXECUTE FUNCTION
                    {_SCHEMA}.fence_owned_response_conversation_delete()"""
        )

    @staticmethod
    def _ensure_trigger_function(
        conn,
        *,
        function_name: str,
        function_body: str,
        contract_marker: str,
        error_label: str,
    ) -> None:
        """Create an exact trigger function or fail closed on a namesake."""
        normalized_body = " ".join(function_body.split())
        row = conn.execute(
            f"""SELECT /* {contract_marker} */
                       l.lanname = 'plpgsql'
                   AND p.prokind = 'f'
                   AND p.prorettype = 'pg_catalog.trigger'::regtype
                   AND p.pronargs = 0
                   AND p.provolatile = 'v'
                   AND NOT p.proisstrict
                   AND NOT p.prosecdef
                   AND NOT p.proleakproof
                   AND p.proparallel = 'u'
                   AND p.proconfig IS NULL
                   AND btrim(
                           regexp_replace(
                               p.prosrc, '[[:space:]]+', ' ', 'g'
                           )
                       ) = %s
                FROM pg_proc AS p
                JOIN pg_namespace AS n ON n.oid = p.pronamespace
                JOIN pg_language AS l ON l.oid = p.prolang
                WHERE n.nspname = '{_SCHEMA}'
                  AND p.proname = %s
                  AND p.pronargs = 0""",
            (normalized_body, function_name),
        ).fetchone()
        if row is not None:
            if not bool(row[0]):
                raise RuntimeError(
                    f"Existing {error_label} has unsafe semantics"
                )
            return
        conn.execute(
            f"""CREATE FUNCTION {_SCHEMA}.{function_name}()
                RETURNS trigger
                LANGUAGE plpgsql
                VOLATILE
                CALLED ON NULL INPUT
                SECURITY INVOKER
                PARALLEL UNSAFE
                AS $function${function_body}$function$"""
        )

    @staticmethod
    def _ensure_conversation_response_fk(conn) -> None:
        """Install the exact cascading FK, or reject a misleading namesake."""
        row = conn.execute(
            f"""SELECT /* hermes_response_conversation_fk_contract */
                       c.contype = 'f'
                   AND c.confrelid = '{_SCHEMA}.responses'::regclass
                   AND cardinality(c.conkey) = 1
                   AND c.conkey[1] = (
                       SELECT attnum FROM pg_attribute
                       WHERE attrelid = '{_SCHEMA}.conversations'::regclass
                         AND attname = 'response_id'
                   )
                   AND cardinality(c.confkey) = 1
                   AND c.confkey[1] = (
                       SELECT attnum FROM pg_attribute
                       WHERE attrelid = '{_SCHEMA}.responses'::regclass
                         AND attname = 'response_id'
                   )
                   AND c.confmatchtype = 's'
                   AND c.confupdtype = 'a'
                   AND c.confdeltype = 'c'
                   AND NOT c.condeferrable
                   AND NOT c.condeferred
                   AND c.convalidated
                FROM pg_constraint AS c
                WHERE c.conname = 'conversations_response_id_fkey'
                  AND c.conrelid = '{_SCHEMA}.conversations'::regclass"""
        ).fetchone()
        if row is not None:
            if not bool(row[0]):
                raise RuntimeError(
                    "Existing conversation response foreign key has unsafe semantics"
                )
            return
        conn.execute(
            f"""ALTER TABLE {_SCHEMA}.conversations
                ADD CONSTRAINT conversations_response_id_fkey
                FOREIGN KEY (response_id)
                REFERENCES {_SCHEMA}.responses(response_id)
                ON DELETE CASCADE"""
        )

    def claim(
        self,
        response_id: str,
        data: Dict[str, Any],
        *,
        owner_id: str,
        owner_epoch: str,
        conversation: Optional[str] = None,
        terminal: bool = False,
    ) -> bool:
        """Atomically create an owned response and its conversation mapping."""
        def _claim():
            with self._pool.connection() as conn, conn.transaction():
                row = conn.execute(
                    f"""INSERT INTO {_SCHEMA}.responses
                        (response_id, data, accessed_at, owner_id, owner_epoch,
                         owner_heartbeat_at, terminal)
                        VALUES (%s, %s, %s, %s, %s, %s, %s)
                        ON CONFLICT (response_id) DO NOTHING
                        RETURNING response_id""",
                    (
                        response_id,
                        self._Jsonb(data, dumps=_dumps),
                        time.time(),
                        owner_id,
                        owner_epoch,
                        time.time(),
                        terminal,
                    ),
                ).fetchone()
                if row is None:
                    return False
                if conversation:
                    conn.execute(
                        f"""INSERT INTO {_SCHEMA}.conversations (name, response_id)
                            VALUES (%s, %s)
                            ON CONFLICT (name)
                            DO UPDATE SET response_id = EXCLUDED.response_id""",
                        (conversation, response_id),
                    )
                return True

        return self._with_schema_retry("claim", _claim)

    def get_control(self, response_id: str) -> Optional[Dict[str, Any]]:
        """Return durable owner/epoch and terminal state."""
        def _get_control():
            with self._pool.connection() as conn:
                row = conn.execute(
                    f"""SELECT owner_id, owner_epoch, terminal
                        FROM {_SCHEMA}.responses WHERE response_id = %s""",
                    (response_id,),
                ).fetchone()
                if row is None:
                    return None
                return {
                    "owner_id": row[0],
                    "owner_epoch": row[1],
                    "terminal": bool(row[2]),
                }

        return self._with_schema_retry("get_control", _get_control)

    def transition(
        self,
        response_id: str,
        data: Dict[str, Any],
        *,
        owner_id: str,
        owner_epoch: str,
        terminal: bool,
    ) -> bool:
        """CAS an owned nonterminal response; terminal states are monotonic."""
        def _transition():
            with self._pool.connection() as conn, conn.transaction():
                # Mixed-version database triggers reject data mutation on an
                # owned row unless this exact owner/epoch authorizes it for the
                # current transaction.  SET LOCAL semantics prevent pooled
                # connections from leaking authority to a later legacy write.
                conn.execute(
                    """SELECT
                           set_config(
                               'hermes_gw.response_write_owner_id', %s, true
                           ),
                           set_config(
                               'hermes_gw.response_write_owner_epoch', %s, true
                           )""",
                    (owner_id, owner_epoch),
                )
                cur = conn.execute(
                    f"""UPDATE {_SCHEMA}.responses
                        SET data = %s, accessed_at = %s,
                            owner_heartbeat_at = %s, terminal = %s
                        WHERE response_id = %s
                          AND owner_id = %s
                          AND owner_epoch = %s
                          AND terminal = FALSE""",
                    (
                        self._Jsonb(data, dumps=_dumps),
                        time.time(),
                        time.time(),
                        terminal,
                        response_id,
                        owner_id,
                        owner_epoch,
                    ),
                )
                return cur.rowcount == 1

        return self._with_schema_retry("transition", _transition)

    def heartbeat(
        self, response_id: str, *, owner_id: str, owner_epoch: str
    ) -> bool:
        """Renew a live response owner lease without changing its payload."""
        def _heartbeat():
            with self._pool.connection() as conn:
                cur = conn.execute(
                    f"""UPDATE {_SCHEMA}.responses
                        SET owner_heartbeat_at = %s
                        WHERE response_id = %s
                          AND owner_id = %s
                          AND owner_epoch = %s
                          AND terminal = FALSE""",
                    (time.time(), response_id, owner_id, owner_epoch),
                )
                return cur.rowcount == 1

        return self._with_schema_retry("heartbeat", _heartbeat)

    def recover_stale_owned(self, response_id: str, *, stale_before: float) -> bool:
        """Fence and terminalize a response whose owner lease expired."""
        def _recover():
            with self._pool.connection() as conn, conn.transaction():
                row = conn.execute(
                    f"""SELECT /* hermes_response_stale_recovery */
                               data, owner_id, owner_epoch,
                               owner_heartbeat_at, terminal
                        FROM {_SCHEMA}.responses
                        WHERE response_id = %s
                        FOR UPDATE""",
                    (response_id,),
                ).fetchone()
                if row is None or bool(row[4]):
                    return False
                heartbeat_at = row[3]
                if heartbeat_at is None or float(heartbeat_at) >= stale_before:
                    return False
                record = row[0]
                response = record.get("response") if isinstance(record, dict) else None
                if not isinstance(response, dict):
                    return False
                response["status"] = "incomplete"
                response["incomplete_details"] = {"reason": "owner_lost"}
                owner_id, owner_epoch = row[1], row[2]
                conn.execute(
                    """SELECT
                           set_config(
                               'hermes_gw.response_write_owner_id', %s, true
                           ),
                           set_config(
                               'hermes_gw.response_write_owner_epoch', %s, true
                           )""",
                    (owner_id or "", owner_epoch or ""),
                )
                cur = conn.execute(
                    f"""UPDATE /* hermes_response_stale_recovery_commit */
                               {_SCHEMA}.responses
                        SET data = %s, accessed_at = %s, terminal = TRUE
                        WHERE response_id = %s
                          AND owner_id IS NOT DISTINCT FROM %s
                          AND owner_epoch IS NOT DISTINCT FROM %s
                          AND owner_heartbeat_at < %s
                          AND terminal = FALSE""",
                    (
                        self._Jsonb(record, dumps=_dumps),
                        time.time(),
                        response_id,
                        owner_id,
                        owner_epoch,
                        stale_before,
                    ),
                )
                return cur.rowcount == 1

        return self._with_schema_retry("stale recovery", _recover)

    def delete_terminal(self, response_id: str) -> str:
        """Lock, verify terminality, and delete with FK-cascaded mappings."""
        def _delete_terminal():
            with self._pool.connection() as conn, conn.transaction():
                row = conn.execute(
                    f"""SELECT terminal FROM {_SCHEMA}.responses
                        WHERE response_id = %s FOR UPDATE""",
                    (response_id,),
                ).fetchone()
                if row is None:
                    return "not_found"
                if not bool(row[0]):
                    return "active"
                conn.execute(
                    f"DELETE FROM {_SCHEMA}.responses WHERE response_id = %s",
                    (response_id,),
                )
                return "deleted"

        return self._with_schema_retry("delete_terminal", _delete_terminal)

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
            terminal = _record_is_terminal(data)
            with self._pool.connection() as conn:
                conn.execute(
                    f"""INSERT INTO {_SCHEMA}.responses
                        (response_id, data, accessed_at, owner_id, owner_epoch,
                         owner_heartbeat_at, terminal)
                        VALUES (%s, %s, %s, NULL, NULL, %s, %s)
                        ON CONFLICT (response_id)
                        DO UPDATE SET data = EXCLUDED.data,
                                      accessed_at = EXCLUDED.accessed_at,
                                      owner_id = NULL,
                                      owner_epoch = NULL,
                                      owner_heartbeat_at =
                                          EXCLUDED.owner_heartbeat_at,
                                      terminal = EXCLUDED.terminal""",
                    (
                        response_id,
                        self._Jsonb(data, dumps=_dumps),
                        time.time(),
                        None if terminal else time.time(),
                        terminal,
                    ),
                )

        self._with_schema_retry("put", _put)

    def delete(self, response_id: str) -> bool:
        """Remove a response from the store. Returns True if found and deleted."""
        def _delete():
            with self._pool.connection() as conn, conn.transaction():
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

    def set_conversation(self, name: str, response_id: str) -> bool:
        """Map a name only while the referenced response still exists."""
        def _set_conversation():
            with self._pool.connection() as conn, conn.transaction():
                response = conn.execute(
                    f"""SELECT response_id FROM {_SCHEMA}.responses
                        WHERE response_id = %s FOR KEY SHARE""",
                    (response_id,),
                ).fetchone()
                if response is None:
                    return False
                row = conn.execute(
                    f"""INSERT INTO {_SCHEMA}.conversations (name, response_id)
                        VALUES (%s, %s)
                        ON CONFLICT (name)
                        DO UPDATE SET response_id = EXCLUDED.response_id
                        RETURNING name""",
                    (name, response_id),
                ).fetchone()
                return row is not None

        return self._with_schema_retry("set_conversation", _set_conversation)

    def close(self) -> None:
        """Close the connection pool."""
        if self._closed:
            return
        self._closed = True
        try:
            self._pool.close()
        except Exception:
            pass

    def storage_attestation(self) -> Dict[str, str]:
        """Return a non-secret, machine-readable backend identity."""
        return {"backend": "postgres"}

    def __len__(self) -> int:
        def _len():
            with self._pool.connection() as conn:
                row = conn.execute(
                    f"SELECT COUNT(*) FROM {_SCHEMA}.responses"
                ).fetchone()
                return row[0] if row else 0

        return self._with_schema_retry("len", _len)
