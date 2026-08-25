"""Unit regression tests for PgResponseStore retention behavior.

These tests avoid a live Postgres dependency so the D6 durability contract is
checked in ordinary local/CI runs. The DSN-backed tests in
``test_response_store_pg.py`` still cover real Postgres behavior when a test
database is provided.
"""

import importlib.util
import json
import pathlib
import sys
import time
import types

import pytest


_MODULE_PATH = (
    pathlib.Path(__file__).resolve().parents[2]
    / "gateway"
    / "platforms"
    / "response_store_pg.py"
)
_spec = importlib.util.spec_from_file_location("response_store_pg", _MODULE_PATH)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
PgResponseStore = _mod.PgResponseStore


class _FakeAdminShutdown(Exception):
    pass


class _FakeInvalidSchemaName(Exception):
    pass


class _FakeUndefinedTable(Exception):
    pass


class _FakeJsonb:
    def __init__(self, value, *, dumps):
        self.value = json.loads(dumps(value))


class _FakeCursor:
    def __init__(self, rows=None, rowcount=0):
        self._rows = rows or []
        self.rowcount = rowcount

    def fetchone(self):
        return self._rows[0] if self._rows else None

    def fetchall(self):
        return self._rows


class _FakeConnection:
    def __init__(self):
        self.responses = {}
        self.conversations = {}
        self._clock = 0.0
        self.fail_next_response_insert = False
        self.schema_init_count = 0
        self.schema_lock_count = 0
        self.schema_contract_version = None
        self.raise_on_schema_fast_path = False
        self.schema_mutation_count = 0
        self.responses_relation_exists = False
        self.conversations_relation_exists = False
        self.access_index_exists = False
        self.owner_heartbeat_column_exists = True
        self.terminal_default = False
        self.legacy_function_exists = False
        self.legacy_function_definition_valid = True
        self.trigger_exists = False
        self.trigger_definition_valid = True
        self.fk_exists = False
        self.fk_definition_valid = True
        self.owned_fence_function_exists = False
        self.owned_fence_function_definition_valid = True
        self.owned_fence_trigger_exists = False
        self.owned_fence_trigger_definition_valid = True
        self.conversation_fence_function_exists = False
        self.conversation_fence_function_definition_valid = True
        self.conversation_fence_trigger_exists = False
        self.conversation_fence_trigger_definition_valid = True
        self.response_write_owner_id = None
        self.response_write_owner_epoch = None

    def execute(self, sql, params=None):
        normalized = " ".join(sql.split()).lower()
        params = params or ()
        if "hermes_response_schema_contract_version" in normalized:
            return _FakeCursor(
                [(self.schema_contract_version,)]
                if self.schema_contract_version is not None
                else []
            )
        if "hermes_response_schema_contract_fast_path" in normalized:
            if self.raise_on_schema_fast_path:
                raise _FakeUndefinedTable()
            return _FakeCursor(
                [(
                    self.schema_contract_version,
                    self.responses_relation_exists,
                    self.conversations_relation_exists,
                    self.legacy_function_exists,
                    self.trigger_exists,
                    self.owned_fence_function_exists,
                    self.owned_fence_trigger_exists,
                    self.conversation_fence_function_exists,
                    self.conversation_fence_trigger_exists,
                    self.fk_exists,
                    self.access_index_exists,
                    self.owner_heartbeat_column_exists,
                )]
                if self.schema_contract_version is not None
                else []
            )
        if "hermes_response_legacy_function_contract" in normalized:
            return _FakeCursor(
                [(self.legacy_function_definition_valid,)]
                if self.legacy_function_exists
                else []
            )
        if "hermes_response_legacy_trigger_contract" in normalized:
            return _FakeCursor(
                [(self.trigger_definition_valid,)] if self.trigger_exists else []
            )
        if "hermes_response_conversation_fk_contract" in normalized:
            return _FakeCursor(
                [(self.fk_definition_valid,)] if self.fk_exists else []
            )
        if "hermes_response_owned_fence_function_contract" in normalized:
            return _FakeCursor(
                [(self.owned_fence_function_definition_valid,)]
                if self.owned_fence_function_exists
                else []
            )
        if "hermes_response_owned_fence_trigger_contract" in normalized:
            return _FakeCursor(
                [(self.owned_fence_trigger_definition_valid,)]
                if self.owned_fence_trigger_exists
                else []
            )
        if (
            "hermes_response_conversation_delete_fence_function_contract"
            in normalized
        ):
            return _FakeCursor(
                [(self.conversation_fence_function_definition_valid,)]
                if self.conversation_fence_function_exists
                else []
            )
        if (
            "hermes_response_conversation_delete_fence_trigger_contract"
            in normalized
        ):
            return _FakeCursor(
                [(self.conversation_fence_trigger_definition_valid,)]
                if self.conversation_fence_trigger_exists
                else []
            )
        if normalized.startswith("select set_config"):
            self.response_write_owner_id, self.response_write_owner_epoch = params
            return _FakeCursor([(self.response_write_owner_id, self.response_write_owner_epoch)])
        if normalized.startswith("create schema"):
            self.schema_init_count += 1
            self.schema_mutation_count += 1
            return _FakeCursor()
        if normalized.startswith("create table if not exists hermes_gw.responses"):
            self.responses_relation_exists = True
            self.schema_mutation_count += 1
            return _FakeCursor()
        if normalized.startswith("create table if not exists hermes_gw.conversations"):
            self.conversations_relation_exists = True
            self.schema_mutation_count += 1
            return _FakeCursor()
        if normalized.startswith("create index if not exists idx_responses_accessed_at"):
            self.access_index_exists = True
            self.schema_mutation_count += 1
            return _FakeCursor()
        if normalized.startswith("insert into hermes_gw.schema_contract"):
            self.schema_contract_version = params[0]
            self.schema_mutation_count += 1
            return _FakeCursor()
        if normalized.startswith(
            "create function hermes_gw.sync_legacy_response_terminal"
        ):
            self.legacy_function_exists = True
            self.legacy_function_definition_valid = True
            return _FakeCursor()
        if normalized.startswith("create function hermes_gw.fence_owned_response()"):
            self.owned_fence_function_exists = True
            self.owned_fence_function_definition_valid = True
            return _FakeCursor()
        if normalized.startswith(
            "create function hermes_gw.fence_owned_response_conversation_delete()"
        ):
            self.conversation_fence_function_exists = True
            self.conversation_fence_function_definition_valid = True
            return _FakeCursor()
        if normalized.startswith("create trigger sync_legacy_response_terminal"):
            self.trigger_exists = True
            self.trigger_definition_valid = True
            return _FakeCursor()
        if normalized.startswith(
            "create trigger fence_owned_response_conversation_delete"
        ):
            self.conversation_fence_trigger_exists = True
            self.conversation_fence_trigger_definition_valid = True
            return _FakeCursor()
        if normalized.startswith("create trigger fence_owned_response"):
            self.owned_fence_trigger_exists = True
            self.owned_fence_trigger_definition_valid = True
            return _FakeCursor()
        if normalized.startswith("create "):
            self.schema_mutation_count += 1
            return _FakeCursor()
        if (
            normalized.startswith("alter table hermes_gw.conversations")
            and "add constraint conversations_response_id_fkey" in normalized
        ):
            self.fk_exists = True
            self.fk_definition_valid = True
            return _FakeCursor()
        if normalized.startswith("alter table"):
            self.schema_mutation_count += 1
            if "alter column terminal set default true" in normalized:
                self.terminal_default = True
            return _FakeCursor()
        if normalized.startswith("do $migration$") or normalized.startswith(
            "do $trigger$"
        ):
            return _FakeCursor()
        if normalized.startswith("lock table"):
            self.schema_lock_count += 1
            return _FakeCursor()
        if normalized.startswith("update hermes_gw.responses set terminal = coalesce"):
            for row in self.responses.values():
                if row["owner_id"] is not None or row["owner_epoch"] is not None:
                    continue
                response = row["data"].get("response", {})
                row["terminal"] = response.get("status", "completed") in {
                    "completed",
                    "failed",
                    "cancelled",
                    "canceled",
                    "incomplete",
                }
            return _FakeCursor()
        if normalized.startswith(
            "update hermes_gw.responses set owner_heartbeat_at = %s where terminal = false"
        ):
            heartbeat_at = params[0]
            for row in self.responses.values():
                if (
                    not row["terminal"]
                    and row.get("owner_heartbeat_at") is None
                ):
                    row["owner_heartbeat_at"] = heartbeat_at
            return _FakeCursor()
        if normalized.startswith("delete from hermes_gw.conversations as c"):
            self.conversations = {
                name: response_id
                for name, response_id in self.conversations.items()
                if response_id in self.responses
            }
            return _FakeCursor()
        if normalized.startswith("insert into hermes_gw.responses"):
            if self.fail_next_response_insert:
                self.fail_next_response_insert = False
                self.responses_relation_exists = False
                raise _FakeUndefinedTable("relation hermes_gw.responses does not exist")
            if "do nothing" in normalized:
                (
                    response_id,
                    data,
                    accessed_at,
                    owner_id,
                    owner_epoch,
                    owner_heartbeat_at,
                    terminal,
                ) = params
                if response_id in self.responses:
                    return _FakeCursor(rowcount=0)
            elif len(params) == 3:
                response_id, data, accessed_at = params
                decoded = data.value if isinstance(data, _FakeJsonb) else data
                existing = self.responses.get(response_id)
                if existing and (
                    existing["owner_id"] is not None
                    or existing["owner_epoch"] is not None
                ):
                    # The old ON CONFLICT clause only assigns data/accessed_at;
                    # ownership and terminality remain the target row's values.
                    owner_id = existing["owner_id"]
                    owner_epoch = existing["owner_epoch"]
                    owner_heartbeat_at = existing.get("owner_heartbeat_at")
                    terminal = existing["terminal"]
                else:
                    owner_id = None
                    owner_epoch = None
                    owner_heartbeat_at = None
                    status = decoded.get("response", {}).get("status", "completed")
                    terminal = status in {
                        "completed",
                        "failed",
                        "cancelled",
                        "canceled",
                        "incomplete",
                    }
            else:
                response_id, data, accessed_at, owner_heartbeat_at, terminal = params
                owner_id = None
                owner_epoch = None
            decoded = data.value if isinstance(data, _FakeJsonb) else data
            existing = self.responses.get(response_id)
            if existing and (
                existing["owner_id"] is not None
                or existing["owner_epoch"] is not None
            ):
                semantic_noop = (
                    decoded == existing["data"]
                    and owner_id == existing["owner_id"]
                    and owner_epoch == existing["owner_epoch"]
                    and terminal == existing["terminal"]
                )
                if not semantic_noop:
                    raise RuntimeError("unauthorized owned response mutation")
                existing["accessed_at"] = accessed_at
                return _FakeCursor([(response_id,)], rowcount=1)
            self.responses[response_id] = {
                "data": decoded,
                "accessed_at": accessed_at,
                "owner_id": owner_id,
                "owner_epoch": owner_epoch,
                "owner_heartbeat_at": owner_heartbeat_at,
                "terminal": terminal,
            }
            return _FakeCursor([(response_id,)], rowcount=1)
        if normalized.startswith("select count(*) from hermes_gw.responses"):
            return _FakeCursor([(len(self.responses),)])
        if normalized.startswith("select data from hermes_gw.responses"):
            response_id = params[0]
            row = self.responses.get(response_id)
            return _FakeCursor([(row["data"],)] if row else [])
        if normalized.startswith("select owner_id, owner_epoch, terminal"):
            response_id = params[0]
            row = self.responses.get(response_id)
            return _FakeCursor(
                [(row["owner_id"], row["owner_epoch"], row["terminal"])]
                if row
                else []
            )
        if normalized.startswith("select terminal from hermes_gw.responses"):
            response_id = params[0]
            row = self.responses.get(response_id)
            return _FakeCursor([(row["terminal"],)] if row else [])
        if normalized.startswith("select /* hermes_response_stale_recovery */"):
            response_id = params[0]
            row = self.responses.get(response_id)
            return _FakeCursor(
                [(
                    row["data"],
                    row["owner_id"],
                    row["owner_epoch"],
                    row.get("owner_heartbeat_at"),
                    row["accessed_at"],
                    row["terminal"],
                )]
                if row
                else []
            )
        if normalized.startswith(
            "update /* hermes_response_stale_recovery_commit */ hermes_gw.responses"
        ):
            (
                data,
                accessed_at,
                response_id,
                owner_id,
                owner_epoch,
                stale_before,
                legacy_stale_before,
            ) = params
            row = self.responses.get(response_id)
            matches = bool(
                row
                and row["owner_id"] == owner_id
                and row["owner_epoch"] == owner_epoch
                and not row["terminal"]
                and (
                    (
                        row.get("owner_heartbeat_at") is not None
                        and row["owner_heartbeat_at"] < stale_before
                    )
                    or (
                        row["owner_id"] is None
                        and row["owner_epoch"] is None
                        and row.get("owner_heartbeat_at") is None
                        and row["accessed_at"] < legacy_stale_before
                    )
                )
            )
            if matches:
                row.update(
                    data=data.value if isinstance(data, _FakeJsonb) else data,
                    accessed_at=accessed_at,
                    terminal=True,
                )
            return _FakeCursor(rowcount=int(matches))
        if normalized.startswith("update hermes_gw.responses set data"):
            (
                data,
                accessed_at,
                owner_heartbeat_at,
                terminal,
                response_id,
                owner_id,
                owner_epoch,
            ) = params
            row = self.responses.get(response_id)
            matches = bool(
                row
                and row["owner_id"] == owner_id
                and row["owner_epoch"] == owner_epoch
                and not row["terminal"]
            )
            if matches and (
                self.response_write_owner_id != owner_id
                or self.response_write_owner_epoch != owner_epoch
            ):
                raise RuntimeError("unauthorized owned response mutation")
            if matches:
                row.update(
                    data=data.value if isinstance(data, _FakeJsonb) else data,
                    accessed_at=accessed_at,
                    owner_heartbeat_at=owner_heartbeat_at,
                    terminal=terminal,
                )
            return _FakeCursor(rowcount=int(matches))
        if normalized.startswith(
            "update hermes_gw.responses set owner_heartbeat_at = %s where response_id = %s"
        ):
            heartbeat_at, response_id, owner_id, owner_epoch = params
            row = self.responses.get(response_id)
            matches = bool(
                row
                and row["owner_id"] == owner_id
                and row["owner_epoch"] == owner_epoch
                and not row["terminal"]
            )
            if matches:
                row["owner_heartbeat_at"] = heartbeat_at
            return _FakeCursor(rowcount=int(matches))
        if normalized.startswith("update hermes_gw.responses set accessed_at"):
            accessed_at, response_id = params
            if response_id in self.responses:
                self.responses[response_id]["accessed_at"] = accessed_at
            return _FakeCursor(rowcount=int(response_id in self.responses))
        if normalized.startswith("select response_id from hermes_gw.responses"):
            if "where response_id = %s" in normalized:
                response_id = params[0]
                return _FakeCursor(
                    [(response_id,)] if response_id in self.responses else []
                )
            limit = params[0]
            rows = sorted(
                self.responses.items(),
                key=lambda item: item[1]["accessed_at"],
            )[:limit]
            return _FakeCursor([(response_id,) for response_id, _ in rows])
        if normalized.startswith("delete from hermes_gw.conversations where response_id = any"):
            evict_ids = set(params[0])
            self.conversations = {
                key: value
                for key, value in self.conversations.items()
                if value not in evict_ids
            }
            return _FakeCursor()
        if normalized.startswith("delete from hermes_gw.responses where response_id = any"):
            evict_ids = set(params[0])
            for response_id in evict_ids:
                self.responses.pop(response_id, None)
            return _FakeCursor()
        if normalized.startswith("insert into hermes_gw.conversations"):
            name, response_id = params
            if response_id not in self.responses:
                return _FakeCursor(rowcount=0)
            self.conversations[name] = response_id
            return _FakeCursor([(name,)], rowcount=1)
        if normalized.startswith("select response_id from hermes_gw.conversations"):
            name = params[0]
            response_id = self.conversations.get(name)
            return _FakeCursor([(response_id,)] if response_id else [])
        if normalized.startswith("delete from hermes_gw.conversations where response_id = %s"):
            response_id = params[0]
            response = self.responses.get(response_id)
            if response and (
                response["owner_id"] is not None
                or response["owner_epoch"] is not None
            ) and not response["terminal"]:
                return _FakeCursor(rowcount=0)
            self.conversations = {
                name: mapped_id
                for name, mapped_id in self.conversations.items()
                if mapped_id != response_id
            }
            return _FakeCursor()
        if normalized.startswith("delete from hermes_gw.responses where response_id = %s"):
            response_id = params[0]
            response = self.responses.get(response_id)
            if response and (
                response["owner_id"] is not None
                or response["owner_epoch"] is not None
            ) and not response["terminal"]:
                return _FakeCursor(rowcount=0)
            existed = response_id in self.responses
            self.responses.pop(response_id, None)
            self.conversations = {
                name: mapped_id
                for name, mapped_id in self.conversations.items()
                if mapped_id != response_id
            }
            return _FakeCursor(rowcount=int(existed))
        raise AssertionError(f"unhandled SQL: {sql}")

    def transaction(self):
        return _TransactionContext(self)


class _ConnectionContext:
    def __init__(self, conn):
        self._conn = conn

    def __enter__(self):
        return self._conn

    def __exit__(self, *args):
        return False


class _TransactionContext(_ConnectionContext):
    def __exit__(self, *args):
        self._conn.response_write_owner_id = None
        self._conn.response_write_owner_epoch = None
        return False


class _FakePool:
    last_instance = None

    def __init__(self, **_kwargs):
        self.conn = _FakeConnection()
        self.closed = False
        _FakePool.last_instance = self

    def connection(self):
        return _ConnectionContext(self.conn)

    def close(self):
        self.closed = True


def _install_fake_psycopg_modules(monkeypatch):
    psycopg = types.ModuleType("psycopg")
    psycopg.errors = types.SimpleNamespace(
        AdminShutdown=_FakeAdminShutdown,
        InvalidSchemaName=_FakeInvalidSchemaName,
        UndefinedTable=_FakeUndefinedTable,
    )
    psycopg_types = types.ModuleType("psycopg.types")
    psycopg_json = types.ModuleType("psycopg.types.json")
    psycopg_json.Jsonb = _FakeJsonb
    psycopg_pool = types.ModuleType("psycopg_pool")
    psycopg_pool.ConnectionPool = _FakePool
    monkeypatch.setitem(sys.modules, "psycopg", psycopg)
    monkeypatch.setitem(sys.modules, "psycopg.types", psycopg_types)
    monkeypatch.setitem(sys.modules, "psycopg.types.json", psycopg_json)
    monkeypatch.setitem(sys.modules, "psycopg_pool", psycopg_pool)


def test_pg_response_store_keeps_more_than_constructor_max_size(monkeypatch):
    _install_fake_psycopg_modules(monkeypatch)
    store = PgResponseStore("postgresql://fake", max_size=3)
    try:
        for index in range(4):
            store.put(f"r{index}", {"index": index})

        assert len(store) == 4
        assert store.get("r0") == {"index": 0}
        assert store.get("r3") == {"index": 3}
    finally:
        store.close()


def test_pg_response_store_exposes_positive_backend_attestation(monkeypatch):
    _install_fake_psycopg_modules(monkeypatch)
    store = PgResponseStore("postgresql://fake")
    try:
        assert store.storage_attestation() == {"backend": "postgres"}
    finally:
        store.close()


def test_pg_current_schema_second_boot_avoids_ddl_lock_and_backfill(monkeypatch):
    _install_fake_psycopg_modules(monkeypatch)
    store = PgResponseStore("postgresql://fake")
    try:
        conn = _FakePool.last_instance.conn
        mutations = conn.schema_mutation_count
        locks = conn.schema_lock_count

        store._init_schema()

        assert conn.schema_mutation_count == mutations
        assert conn.schema_lock_count == locks
    finally:
        store.close()


def test_pg_current_schema_missing_heartbeat_column_reenters_migration(monkeypatch):
    _install_fake_psycopg_modules(monkeypatch)
    store = PgResponseStore("postgresql://fake")
    try:
        conn = _FakePool.last_instance.conn
        locks = conn.schema_lock_count
        conn.owner_heartbeat_column_exists = False

        store._init_schema()

        assert conn.schema_lock_count == locks + 1
    finally:
        store.close()


def test_pg_future_schema_contract_fails_closed_without_mutation(monkeypatch):
    _install_fake_psycopg_modules(monkeypatch)
    store = PgResponseStore("postgresql://fake")
    try:
        conn = _FakePool.last_instance.conn
        conn.schema_contract_version = 99
        conn.raise_on_schema_fast_path = True
        mutations = conn.schema_mutation_count

        with pytest.raises(RuntimeError, match="newer than this runtime"):
            store._init_schema()

        assert conn.schema_contract_version == 99
        assert conn.schema_mutation_count == mutations
    finally:
        store.close()


def test_pg_response_store_reinitializes_missing_schema_once(monkeypatch):
    _install_fake_psycopg_modules(monkeypatch)
    store = PgResponseStore("postgresql://fake", max_size=3)
    try:
        conn = _FakePool.last_instance.conn
        conn.fail_next_response_insert = True

        store.put("r1", {"value": 1})

        assert conn.schema_init_count == 2
        assert store.get("r1") == {"value": 1}
    finally:
        store.close()


def test_pg_owned_terminal_transition_cannot_be_overwritten_or_resurrected(
    monkeypatch,
):
    _install_fake_psycopg_modules(monkeypatch)
    store = PgResponseStore("postgresql://fake")
    try:
        active = {"response": {"id": "r1", "status": "in_progress"}}
        cancelled = {"response": {"id": "r1", "status": "cancelled"}}
        completed = {"response": {"id": "r1", "status": "completed"}}

        assert store.claim(
            "r1",
            active,
            owner_id="owner-a",
            owner_epoch="epoch-1",
            conversation="conv-1",
        )
        assert store.delete_terminal("r1") == "active"
        assert store.transition(
            "r1",
            cancelled,
            owner_id="owner-a",
            owner_epoch="epoch-1",
            terminal=True,
        )
        assert not store.transition(
            "r1",
            completed,
            owner_id="owner-a",
            owner_epoch="epoch-1",
            terminal=True,
        )

        assert store.delete_terminal("r1") == "deleted"
        assert store.get("r1") is None
        assert store.get_conversation("conv-1") is None
        assert not store.set_conversation("late", "r1")
        assert store.get_conversation("late") is None
        assert not store.transition(
            "r1",
            completed,
            owner_id="owner-a",
            owner_epoch="epoch-1",
            terminal=True,
        )
    finally:
        store.close()


def test_pg_stale_owner_recovery_is_heartbeat_fenced(monkeypatch):
    _install_fake_psycopg_modules(monkeypatch)
    store = PgResponseStore("postgresql://fake")
    try:
        for response_id in ("orphan", "live"):
            assert store.claim(
                response_id,
                {"response": {"id": response_id, "status": "in_progress"}},
                owner_id=f"owner-{response_id}",
                owner_epoch=f"epoch-{response_id}",
            )
        conn = _FakePool.last_instance.conn
        conn.responses["orphan"]["owner_heartbeat_at"] = 1.0

        assert store.heartbeat(
            "live", owner_id="owner-live", owner_epoch="epoch-live"
        )
        assert store.recover_stale_owned("orphan", stale_before=2.0)
        assert not store.recover_stale_owned("live", stale_before=2.0)
        assert store.get("orphan")["response"]["status"] == "incomplete"
        assert store.get_control("orphan")["terminal"] is True
        assert store.get_control("live")["terminal"] is False
    finally:
        store.close()


def test_pg_ownerless_legacy_put_expires(monkeypatch):
    _install_fake_psycopg_modules(monkeypatch)
    store = PgResponseStore("postgresql://fake")
    try:
        store.put(
            "legacy",
            {"response": {"id": "legacy", "status": "in_progress"}},
        )
        conn = _FakePool.last_instance.conn
        conn.responses["legacy"]["owner_heartbeat_at"] = None
        conn.responses["legacy"]["accessed_at"] = 1.0

        assert store.recover_stale_owned("legacy", stale_before=2.0)
        assert store.get("legacy")["response"]["status"] == "incomplete"
        assert store.delete_terminal("legacy") == "deleted"
    finally:
        store.close()


def test_pg_existing_table_migration_backfills_state_and_removes_dangling_mappings(
    monkeypatch,
):
    _install_fake_psycopg_modules(monkeypatch)
    store = PgResponseStore("postgresql://fake")
    try:
        conn = _FakePool.last_instance.conn
        conn.responses.update(
            {
                "legacy-complete": {
                    "data": {
                        "response": {
                            "id": "legacy-complete",
                            "status": "completed",
                        }
                    },
                    "accessed_at": 1.0,
                    "owner_id": None,
                    "owner_epoch": None,
                    "terminal": False,
                },
                "legacy-active": {
                    "data": {
                        "response": {
                            "id": "legacy-active",
                            "status": "in_progress",
                        }
                    },
                    "accessed_at": 2.0,
                    "owner_id": None,
                    "owner_epoch": None,
                    "terminal": False,
                },
            }
        )
        conn.conversations.update(
            {
                "valid": "legacy-complete",
                "dangling": "missing-response",
            }
        )

        # Simulate a database created by the pre-marker schema version.
        conn.schema_contract_version = None

        store._init_schema()

        assert conn.responses["legacy-complete"]["terminal"] is True
        assert conn.responses["legacy-active"]["terminal"] is False
        assert conn.conversations == {"valid": "legacy-complete"}
        assert conn.terminal_default is True
        # One lock transaction on construction and one on this simulated
        # rolling-upgrade migration.
        assert conn.schema_lock_count == 2
    finally:
        store.close()


def test_pg_migration_classifies_old_three_column_writes(monkeypatch):
    """A blue/green sibling still emits the pre-lifecycle INSERT shape."""
    _install_fake_psycopg_modules(monkeypatch)
    store = PgResponseStore("postgresql://fake")
    try:
        conn = _FakePool.last_instance.conn
        conn.execute(
            """INSERT INTO hermes_gw.responses (response_id, data, accessed_at)
               VALUES (%s, %s, %s)
               ON CONFLICT (response_id)
               DO UPDATE SET data = EXCLUDED.data,
                             accessed_at = EXCLUDED.accessed_at""",
            (
                "old-completed",
                _FakeJsonb(
                    {"response": {"id": "old-completed", "status": "completed"}},
                    dumps=json.dumps,
                ),
                1.0,
            ),
        )
        conn.execute(
            """INSERT INTO hermes_gw.responses (response_id, data, accessed_at)
               VALUES (%s, %s, %s)
               ON CONFLICT (response_id)
               DO UPDATE SET data = EXCLUDED.data,
                             accessed_at = EXCLUDED.accessed_at""",
            (
                "old-stream",
                _FakeJsonb(
                    {"response": {"id": "old-stream", "status": "in_progress"}},
                    dumps=json.dumps,
                ),
                2.0,
            ),
        )

        assert conn.responses["old-completed"]["terminal"] is True
        assert conn.responses["old-stream"]["terminal"] is False

        # The same old ON CONFLICT update must move its own stream to terminal.
        conn.execute(
            """INSERT INTO hermes_gw.responses (response_id, data, accessed_at)
               VALUES (%s, %s, %s)
               ON CONFLICT (response_id)
               DO UPDATE SET data = EXCLUDED.data,
                             accessed_at = EXCLUDED.accessed_at""",
            (
                "old-stream",
                _FakeJsonb(
                    {"response": {"id": "old-stream", "status": "completed"}},
                    dumps=json.dumps,
                ),
                3.0,
            ),
        )
        assert conn.responses["old-stream"]["terminal"] is True
    finally:
        store.close()


@pytest.mark.parametrize(
    ("validity_attribute", "error_fragment"),
    (
        ("legacy_function_definition_valid", "legacy terminal function"),
        ("fk_definition_valid", "conversation response foreign key"),
        ("trigger_definition_valid", "legacy terminal trigger"),
        ("owned_fence_function_definition_valid", "owned response fence function"),
        ("owned_fence_trigger_definition_valid", "owned response fence trigger"),
        (
            "conversation_fence_function_definition_valid",
            "conversation delete fence function",
        ),
        (
            "conversation_fence_trigger_definition_valid",
            "conversation delete fence trigger",
        ),
    ),
)
def test_pg_migration_rejects_same_named_objects_with_wrong_semantics(
    monkeypatch,
    validity_attribute,
    error_fragment,
):
    """A familiar catalog name must never be mistaken for the safety contract."""
    _install_fake_psycopg_modules(monkeypatch)
    store = PgResponseStore("postgresql://fake")
    try:
        conn = _FakePool.last_instance.conn
        setattr(conn, validity_attribute, False)

        with pytest.raises(RuntimeError, match=error_fragment):
            store._init_schema()
    finally:
        store.close()


def test_old_pg_writer_cannot_mutate_owned_row_but_access_touch_and_cas_work(
    monkeypatch,
):
    _install_fake_psycopg_modules(monkeypatch)
    store = PgResponseStore("postgresql://fake")
    try:
        conn = _FakePool.last_instance.conn
        active = {"response": {"id": "owned", "status": "in_progress"}}
        completed = {"response": {"id": "owned", "status": "completed"}}
        assert store.claim(
            "owned",
            active,
            owner_id="gateway-new",
            owner_epoch="epoch-new",
            conversation="owned-conversation",
        )

        with pytest.raises(RuntimeError, match="owned response mutation"):
            conn.execute(
                """INSERT INTO hermes_gw.responses (response_id, data, accessed_at)
                   VALUES (%s, %s, %s)
                   ON CONFLICT (response_id)
                   DO UPDATE SET data = EXCLUDED.data,
                                 accessed_at = EXCLUDED.accessed_at""",
                ("owned", _FakeJsonb(completed, dumps=json.dumps), 10.0),
            )
        assert store.get("owned") == active
        assert store.get_control("owned") == {
            "owner_id": "gateway-new",
            "owner_epoch": "epoch-new",
            "terminal": False,
        }

        conn.execute(
            "UPDATE hermes_gw.responses SET accessed_at = %s WHERE response_id = %s",
            (20.0, "owned"),
        )
        assert conn.responses["owned"]["accessed_at"] == 20.0
        assert store.transition(
            "owned",
            completed,
            owner_id="gateway-new",
            owner_epoch="epoch-new",
            terminal=True,
        )
        assert store.get("owned") == completed
        with pytest.raises(RuntimeError, match="owned response mutation"):
            store.put(
                "owned",
                {"response": {"id": "owned", "status": "failed"}},
            )
    finally:
        store.close()


def test_old_pg_delete_cannot_remove_active_owned_row_or_mapping_but_legacy_works(
    monkeypatch,
):
    _install_fake_psycopg_modules(monkeypatch)
    store = PgResponseStore("postgresql://fake")
    try:
        conn = _FakePool.last_instance.conn
        active = {"response": {"id": "owned", "status": "in_progress"}}
        assert store.claim(
            "owned",
            active,
            owner_id="gateway-new",
            owner_epoch="epoch-new",
            conversation="owned-conversation",
        )

        conn.execute(
            "DELETE FROM hermes_gw.conversations WHERE response_id = %s",
            ("owned",),
        )
        conn.execute(
            "DELETE FROM hermes_gw.responses WHERE response_id = %s",
            ("owned",),
        )
        assert store.get("owned") == active
        assert store.get_conversation("owned-conversation") == "owned"

        store.put(
            "legacy",
            {"response": {"id": "legacy", "status": "completed"}},
        )
        assert store.set_conversation("legacy-conversation", "legacy")
        conn.execute(
            "DELETE FROM hermes_gw.conversations WHERE response_id = %s",
            ("legacy",),
        )
        conn.execute(
            "DELETE FROM hermes_gw.responses WHERE response_id = %s",
            ("legacy",),
        )
        assert store.get("legacy") is None
        assert store.get_conversation("legacy-conversation") is None
    finally:
        store.close()
