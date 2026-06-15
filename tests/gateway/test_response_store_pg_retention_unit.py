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
import types


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

    def execute(self, sql, params=None):
        normalized = " ".join(sql.split()).lower()
        params = params or ()
        if normalized.startswith("create "):
            return _FakeCursor()
        if normalized.startswith("insert into hermes_gw.responses"):
            response_id, data, accessed_at = params
            self.responses[response_id] = {
                "data": data.value if isinstance(data, _FakeJsonb) else data,
                "accessed_at": accessed_at,
            }
            return _FakeCursor(rowcount=1)
        if normalized.startswith("select count(*) from hermes_gw.responses"):
            return _FakeCursor([(len(self.responses),)])
        if normalized.startswith("select data from hermes_gw.responses"):
            response_id = params[0]
            row = self.responses.get(response_id)
            return _FakeCursor([(row["data"],)] if row else [])
        if normalized.startswith("update hermes_gw.responses set accessed_at"):
            accessed_at, response_id = params
            if response_id in self.responses:
                self.responses[response_id]["accessed_at"] = accessed_at
            return _FakeCursor(rowcount=int(response_id in self.responses))
        if normalized.startswith("select response_id from hermes_gw.responses"):
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
        raise AssertionError(f"unhandled SQL: {sql}")


class _ConnectionContext:
    def __init__(self, conn):
        self._conn = conn

    def __enter__(self):
        return self._conn

    def __exit__(self, *args):
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
