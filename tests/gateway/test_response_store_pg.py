"""Behavioral tests for the Postgres Responses API store (IdeaRoom D6 / AE-61).

Parity with the SQLite ``ResponseStore`` in ``gateway/platforms/api_server.py``:
get/put/delete/get_conversation/set_conversation/__len__ plus LRU eviction and
accessed_at bump.

Requires a reachable Postgres; provide its DSN via ``HERMES_D6_TEST_DSN``
(e.g. ``postgresql://user@localhost:5432/hermes_d6_test``). The test is skipped
when psycopg or the DSN is unavailable, so the default CI run is unaffected.
The module is loaded by file path to avoid the heavy ``gateway`` package
``__init__`` (which pulls aiohttp/yaml and is irrelevant to this storage class).
"""

import importlib.util
import os
import pathlib

import pytest

_DSN = os.environ.get("HERMES_D6_TEST_DSN", "").strip()

psycopg = pytest.importorskip("psycopg")
pytest.importorskip("psycopg_pool")
if not _DSN:
    pytest.skip("HERMES_D6_TEST_DSN not set", allow_module_level=True)

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


@pytest.fixture()
def store():
    # Clean slate: drop the gateway schema so each test starts empty.
    with psycopg.connect(_DSN, autocommit=True) as conn:
        conn.execute("DROP SCHEMA IF EXISTS hermes_gw CASCADE")
    s = PgResponseStore(_DSN, max_size=3)
    try:
        yield s
    finally:
        s.close()


def test_put_get_roundtrip(store):
    data = {"response": {"id": "r1"}, "history": [1, 2, 3], "nested": {"a": {"b": 1}}}
    store.put("r1", data)
    assert store.get("r1") == data


def test_get_missing_returns_none(store):
    assert store.get("nope") is None


def test_put_serializes_non_json_native_with_default_str(store):
    # The SQLite store uses json.dumps(..., default=str); parity here means a
    # non-JSON-native value is coerced to its str() rather than raising.
    import datetime

    ts = datetime.datetime(2026, 6, 13, 12, 0, 0)
    store.put("r1", {"when": ts, "tags": ["x"]})
    got = store.get("r1")
    assert got["when"] == str(ts)
    assert got["tags"] == ["x"]


def test_len(store):
    assert len(store) == 0
    store.put("a", {"v": 1})
    store.put("b", {"v": 2})
    assert len(store) == 2


def test_put_replaces_same_id(store):
    store.put("r1", {"v": 1})
    store.put("r1", {"v": 2})
    assert store.get("r1") == {"v": 2}
    assert len(store) == 1


def test_lru_eviction_by_accessed_at(store):
    # max_size=3. Insert 3, touch the first so it is most-recent, then insert a
    # 4th — the least-recently-accessed of the remaining should be evicted.
    import time

    store.put("a", {"v": "a"})
    time.sleep(0.01)
    store.put("b", {"v": "b"})
    time.sleep(0.01)
    store.put("c", {"v": "c"})
    time.sleep(0.01)
    store.get("a")  # bump a's accessed_at -> a is now newest
    time.sleep(0.01)
    store.put("d", {"v": "d"})  # over capacity -> evict oldest (b)
    assert len(store) == 3
    assert store.get("b") is None
    assert store.get("a") == {"v": "a"}
    assert store.get("c") == {"v": "c"}
    assert store.get("d") == {"v": "d"}


def test_conversations_set_get(store):
    assert store.get_conversation("conv1") is None
    store.set_conversation("conv1", "r1")
    assert store.get_conversation("conv1") == "r1"
    store.set_conversation("conv1", "r2")  # upsert
    assert store.get_conversation("conv1") == "r2"


def test_delete(store):
    store.put("r1", {"v": 1})
    store.set_conversation("conv1", "r1")
    assert store.delete("r1") is True
    assert store.get("r1") is None
    assert store.get_conversation("conv1") is None  # mapping cleared
    assert store.delete("r1") is False  # already gone


def test_eviction_clears_conversation_mappings(store):
    import time

    for rid in ("a", "b", "c"):
        store.put(rid, {"v": rid})
        store.set_conversation(f"conv-{rid}", rid)
        time.sleep(0.01)
    store.put("d", {"v": "d"})  # evicts a
    assert store.get_conversation("conv-a") is None
    assert store.get_conversation("conv-b") == "b"
