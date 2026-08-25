"""Behavioral tests for the Postgres Responses API store (IdeaRoom D6 / AE-61).

Parity with the SQLite ``ResponseStore`` in ``gateway/platforms/api_server.py``:
get/put/delete/get_conversation/set_conversation/__len__. Unlike the SQLite
fallback, the Postgres adapter is durable state and does not apply the old LRU
cap.

Requires a reachable Postgres; provide its DSN via ``HERMES_D6_TEST_DSN``
(e.g. ``postgresql://user@localhost:5432/hermes_d6_test``). The test is skipped
when psycopg or the DSN is unavailable, so the default CI run is unaffected.
The module is loaded by file path to avoid the heavy ``gateway`` package
``__init__`` (which pulls aiohttp/yaml and is irrelevant to this storage class).
"""

import importlib.util
import json
import os
import pathlib
import threading
from concurrent.futures import ThreadPoolExecutor

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


def test_max_size_argument_does_not_evict_durable_responses(store):
    store.put("a", {"v": "a"})
    store.put("b", {"v": "b"})
    store.put("c", {"v": "c"})
    store.put("d", {"v": "d"})
    assert len(store) == 4
    assert store.get("a") == {"v": "a"}
    assert store.get("b") == {"v": "b"}
    assert store.get("c") == {"v": "c"}
    assert store.get("d") == {"v": "d"}


def test_conversations_set_get(store):
    assert store.get_conversation("conv1") is None
    store.put("r1", {"response": {"id": "r1", "status": "completed"}})
    assert store.set_conversation("conv1", "r1")
    assert store.get_conversation("conv1") == "r1"
    store.put("r2", {"response": {"id": "r2", "status": "completed"}})
    assert store.set_conversation("conv1", "r2")  # upsert
    assert store.get_conversation("conv1") == "r2"


def test_delete(store):
    store.put("r1", {"v": 1})
    store.set_conversation("conv1", "r1")
    assert store.delete("r1") is True
    assert store.get("r1") is None
    assert store.get_conversation("conv1") is None  # mapping cleared
    assert store.delete("r1") is False  # already gone


def test_inserting_later_responses_preserves_conversation_mappings(store):
    for rid in ("a", "b", "c"):
        store.put(rid, {"v": rid})
        store.set_conversation(f"conv-{rid}", rid)
    store.put("d", {"v": "d"})
    assert store.get_conversation("conv-a") == "a"
    assert store.get_conversation("conv-b") == "b"


def test_owned_terminal_transition_and_delete_are_monotonic(store):
    active = {"response": {"id": "owned", "status": "in_progress"}}
    cancelled = {"response": {"id": "owned", "status": "cancelled"}}
    completed = {"response": {"id": "owned", "status": "completed"}}

    assert store.claim(
        "owned",
        active,
        owner_id="gateway-a",
        owner_epoch="epoch-1",
        conversation="owned-conversation",
    )
    assert store.delete_terminal("owned") == "active"
    assert store.transition(
        "owned",
        cancelled,
        owner_id="gateway-a",
        owner_epoch="epoch-1",
        terminal=True,
    )
    assert not store.transition(
        "owned",
        completed,
        owner_id="gateway-a",
        owner_epoch="epoch-1",
        terminal=True,
    )
    assert store.delete_terminal("owned") == "deleted"
    assert store.get("owned") is None
    assert store.get_conversation("owned-conversation") is None
    assert not store.set_conversation("late", "owned")
    assert store.get_conversation("late") is None
    assert not store.transition(
        "owned",
        completed,
        owner_id="gateway-a",
        owner_epoch="epoch-1",
        terminal=True,
    )


def test_existing_tables_are_migrated_and_fk_is_validated():
    with psycopg.connect(_DSN, autocommit=True) as conn:
        conn.execute("DROP SCHEMA IF EXISTS hermes_gw CASCADE")
        conn.execute("CREATE SCHEMA hermes_gw")
        conn.execute(
            """CREATE TABLE hermes_gw.responses (
                response_id TEXT PRIMARY KEY,
                data JSONB NOT NULL,
                accessed_at DOUBLE PRECISION NOT NULL
            )"""
        )
        conn.execute(
            """CREATE TABLE hermes_gw.conversations (
                name TEXT PRIMARY KEY,
                response_id TEXT NOT NULL
            )"""
        )
        for response_id, status in (
            ("legacy-complete", "completed"),
            ("legacy-active", "in_progress"),
        ):
            conn.execute(
                """INSERT INTO hermes_gw.responses
                   (response_id, data, accessed_at) VALUES (%s, %s::jsonb, 1)""",
                (
                    response_id,
                    json.dumps(
                        {"response": {"id": response_id, "status": status}}
                    ),
                ),
            )
        conn.execute(
            "INSERT INTO hermes_gw.conversations VALUES ('valid', 'legacy-complete')"
        )
        conn.execute(
            "INSERT INTO hermes_gw.conversations VALUES ('dangling', 'missing')"
        )

    migrated = PgResponseStore(_DSN)
    try:
        assert migrated.get_control("legacy-complete")["terminal"] is True
        assert migrated.get_control("legacy-active")["terminal"] is False
        assert migrated.get_conversation("valid") == "legacy-complete"
        assert migrated.get_conversation("dangling") is None
        with psycopg.connect(_DSN, autocommit=True) as conn:
            constraint = conn.execute(
                """SELECT convalidated, confdeltype
                   FROM pg_constraint
                   WHERE conname = 'conversations_response_id_fkey'
                     AND conrelid = 'hermes_gw.conversations'::regclass"""
            ).fetchone()
            terminal_default = conn.execute(
                """SELECT pg_get_expr(d.adbin, d.adrelid)
                   FROM pg_attribute AS a
                   JOIN pg_attrdef AS d
                     ON d.adrelid = a.attrelid AND d.adnum = a.attnum
                   WHERE a.attrelid = 'hermes_gw.responses'::regclass
                     AND a.attname = 'terminal'"""
            ).fetchone()

            old_write = """INSERT INTO hermes_gw.responses
                (response_id, data, accessed_at) VALUES (%s, %s::jsonb, 1)
                ON CONFLICT (response_id)
                DO UPDATE SET data = EXCLUDED.data,
                              accessed_at = EXCLUDED.accessed_at"""
            conn.execute(
                old_write,
                (
                    "old-completed-after-migration",
                    json.dumps(
                        {
                            "response": {
                                "id": "old-completed-after-migration",
                                "status": "completed",
                            }
                        }
                    ),
                ),
            )
            conn.execute(
                old_write,
                (
                    "old-stream-after-migration",
                    json.dumps(
                        {
                            "response": {
                                "id": "old-stream-after-migration",
                                "status": "in_progress",
                            }
                        }
                    ),
                ),
            )
        assert constraint == (True, "c")
        assert terminal_default and terminal_default[0].lower().startswith("true")
        assert migrated.get_control("old-completed-after-migration")["terminal"] is True
        assert migrated.get_control("old-stream-after-migration")["terminal"] is False

        # The old writer's ON CONFLICT completion update must also flip its
        # initial in-progress row to terminal during the overlap.
        with psycopg.connect(_DSN, autocommit=True) as conn:
            conn.execute(
                old_write,
                (
                    "old-stream-after-migration",
                    json.dumps(
                        {
                            "response": {
                                "id": "old-stream-after-migration",
                                "status": "completed",
                            }
                        }
                    ),
                ),
            )
        assert migrated.get_control("old-stream-after-migration")["terminal"] is True
    finally:
        migrated.close()
        with psycopg.connect(_DSN, autocommit=True) as conn:
            conn.execute("DROP SCHEMA IF EXISTS hermes_gw CASCADE")


@pytest.mark.parametrize(
    ("object_kind", "error_fragment"),
    (
        ("function", "legacy terminal function"),
        ("trigger", "legacy terminal trigger"),
        ("foreign_key", "conversation response foreign key"),
        ("owned_fence_function", "owned response fence function"),
        ("owned_fence_trigger", "owned response fence trigger"),
        ("conversation_fence_function", "conversation delete fence function"),
        ("conversation_fence_trigger", "conversation delete fence trigger"),
    ),
)
def test_schema_init_rejects_same_named_objects_with_wrong_semantics(
    store,
    object_kind,
    error_fragment,
):
    with psycopg.connect(_DSN, autocommit=True) as conn:
        if object_kind == "function":
            conn.execute(
                """CREATE OR REPLACE FUNCTION
                       hermes_gw.sync_legacy_response_terminal()
                   RETURNS trigger
                   LANGUAGE plpgsql
                   AS $function$
                   BEGIN
                       RETURN OLD;
                   END
                   $function$"""
            )
        elif object_kind == "trigger":
            conn.execute(
                """DROP TRIGGER sync_legacy_response_terminal
                   ON hermes_gw.responses"""
            )
            conn.execute(
                """CREATE TRIGGER sync_legacy_response_terminal
                   BEFORE INSERT ON hermes_gw.responses
                   FOR EACH ROW
                   EXECUTE FUNCTION
                       hermes_gw.sync_legacy_response_terminal()"""
            )
        elif object_kind == "foreign_key":
            conn.execute(
                """ALTER TABLE hermes_gw.conversations
                   DROP CONSTRAINT conversations_response_id_fkey"""
            )
            conn.execute(
                """ALTER TABLE hermes_gw.conversations
                   ADD CONSTRAINT conversations_response_id_fkey
                   FOREIGN KEY (response_id)
                   REFERENCES hermes_gw.responses(response_id)"""
            )
        elif object_kind == "owned_fence_function":
            conn.execute(
                """CREATE OR REPLACE FUNCTION hermes_gw.fence_owned_response()
                   RETURNS trigger
                   LANGUAGE plpgsql
                   AS $function$
                   BEGIN
                       RETURN NEW;
                   END
                   $function$"""
            )
        elif object_kind == "owned_fence_trigger":
            conn.execute(
                "DROP TRIGGER fence_owned_response ON hermes_gw.responses"
            )
            conn.execute(
                """CREATE TRIGGER fence_owned_response
                   BEFORE UPDATE ON hermes_gw.responses
                   FOR EACH ROW
                   EXECUTE FUNCTION hermes_gw.fence_owned_response()"""
            )
        elif object_kind == "conversation_fence_function":
            conn.execute(
                """CREATE OR REPLACE FUNCTION
                       hermes_gw.fence_owned_response_conversation_delete()
                   RETURNS trigger
                   LANGUAGE plpgsql
                   AS $function$
                   BEGIN
                       RETURN OLD;
                   END
                   $function$"""
            )
        else:
            conn.execute(
                """DROP TRIGGER fence_owned_response_conversation_delete
                   ON hermes_gw.conversations"""
            )
            conn.execute(
                """CREATE TRIGGER fence_owned_response_conversation_delete
                   AFTER DELETE ON hermes_gw.conversations
                   FOR EACH ROW
                   EXECUTE FUNCTION
                       hermes_gw.fence_owned_response_conversation_delete()"""
            )

    with pytest.raises(RuntimeError, match=error_fragment):
        store._init_schema()


def test_old_writer_cannot_mutate_or_delete_active_owned_response(store):
    active = {"response": {"id": "owned", "status": "in_progress"}}
    completed = {"response": {"id": "owned", "status": "completed"}}
    assert store.claim(
        "owned",
        active,
        owner_id="gateway-new",
        owner_epoch="epoch-new",
        conversation="owned-conversation",
    )

    old_write = """INSERT INTO hermes_gw.responses
        (response_id, data, accessed_at) VALUES (%s, %s::jsonb, %s)
        ON CONFLICT (response_id)
        DO UPDATE SET data = EXCLUDED.data,
                      accessed_at = EXCLUDED.accessed_at"""
    with psycopg.connect(_DSN, autocommit=True) as conn:
        with pytest.raises(psycopg.errors.RaiseException, match="owned response mutation"):
            conn.execute(old_write, ("owned", json.dumps(completed), 10.0))

        row = conn.execute(
            """SELECT data, accessed_at, owner_id, owner_epoch, terminal
               FROM hermes_gw.responses WHERE response_id = 'owned'"""
        ).fetchone()
        assert row[0] == active
        assert row[2:] == ("gateway-new", "epoch-new", False)
        assert row[1] != 10.0

        conn.execute(
            """UPDATE hermes_gw.responses SET accessed_at = 20
               WHERE response_id = 'owned'"""
        )
        assert conn.execute(
            """SELECT accessed_at FROM hermes_gw.responses
               WHERE response_id = 'owned'"""
        ).fetchone() == (20.0,)

        mapping_delete = conn.execute(
            """DELETE FROM hermes_gw.conversations
               WHERE response_id = 'owned'"""
        )
        response_delete = conn.execute(
            """DELETE FROM hermes_gw.responses
               WHERE response_id = 'owned'"""
        )
        assert mapping_delete.rowcount == 0
        assert response_delete.rowcount == 0

    assert store.get("owned") == active
    assert store.get_conversation("owned-conversation") == "owned"
    assert store.transition(
        "owned",
        completed,
        owner_id="gateway-new",
        owner_epoch="epoch-new",
        terminal=True,
    )
    # The current writer's authorization is SET LOCAL: returning its pooled
    # connection cannot grant a later compatibility put authority.
    with pytest.raises(psycopg.errors.RaiseException, match="owned response mutation"):
        store.put(
            "owned",
            {"response": {"id": "owned", "status": "failed"}},
        )
    assert store.delete_terminal("owned") == "deleted"
    assert store.get_conversation("owned-conversation") is None


def test_old_writer_can_still_update_and_delete_legacy_owner_null_response(store):
    old_write = """INSERT INTO hermes_gw.responses
        (response_id, data, accessed_at) VALUES (%s, %s::jsonb, %s)
        ON CONFLICT (response_id)
        DO UPDATE SET data = EXCLUDED.data,
                      accessed_at = EXCLUDED.accessed_at"""
    with psycopg.connect(_DSN, autocommit=True) as conn:
        conn.execute(
            old_write,
            (
                "legacy",
                json.dumps(
                    {"response": {"id": "legacy", "status": "in_progress"}}
                ),
                1.0,
            ),
        )
        conn.execute(
            "INSERT INTO hermes_gw.conversations VALUES ('legacy-conversation', 'legacy')"
        )
        conn.execute(
            old_write,
            (
                "legacy",
                json.dumps(
                    {"response": {"id": "legacy", "status": "completed"}}
                ),
                2.0,
            ),
        )
        assert conn.execute(
            """SELECT data -> 'response' ->> 'status', terminal
               FROM hermes_gw.responses WHERE response_id = 'legacy'"""
        ).fetchone() == ("completed", True)
        assert conn.execute(
            """DELETE FROM hermes_gw.conversations
               WHERE response_id = 'legacy'"""
        ).rowcount == 1
        assert conn.execute(
            """DELETE FROM hermes_gw.responses
               WHERE response_id = 'legacy'"""
        ).rowcount == 1

    assert store.get("legacy") is None
    assert store.get_conversation("legacy-conversation") is None


def test_concurrent_terminal_cas_has_exactly_one_winner(store):
    for index in range(12):
        response_id = f"cas-{index}"
        owner_epoch = f"epoch-{index}"
        assert store.claim(
            response_id,
            {"response": {"id": response_id, "status": "in_progress"}},
            owner_id="gateway-a",
            owner_epoch=owner_epoch,
        )
        barrier = threading.Barrier(2)

        def transition(status):
            barrier.wait(timeout=5)
            return store.transition(
                response_id,
                {"response": {"id": response_id, "status": status}},
                owner_id="gateway-a",
                owner_epoch=owner_epoch,
                terminal=True,
            )

        with ThreadPoolExecutor(max_workers=2) as executor:
            cancel = executor.submit(transition, "cancelled")
            complete = executor.submit(transition, "completed")
            cancel_won = cancel.result(timeout=5)
            complete_won = complete.result(timeout=5)

        assert int(cancel_won) + int(complete_won) == 1
        stored_status = store.get(response_id)["response"]["status"]
        assert stored_status == ("cancelled" if cancel_won else "completed")


def test_concurrent_mapping_set_and_terminal_delete_never_dangles(store):
    for index in range(12):
        response_id = f"delete-race-{index}"
        conversation = f"conversation-{index}"
        store.put(
            response_id,
            {"response": {"id": response_id, "status": "completed"}},
        )
        barrier = threading.Barrier(2)

        def set_mapping():
            barrier.wait(timeout=5)
            return store.set_conversation(conversation, response_id)

        def delete_response():
            barrier.wait(timeout=5)
            return store.delete_terminal(response_id)

        with ThreadPoolExecutor(max_workers=2) as executor:
            mapping = executor.submit(set_mapping)
            deletion = executor.submit(delete_response)
            mapping.result(timeout=5)
            assert deletion.result(timeout=5) == "deleted"

        assert store.get(response_id) is None
        assert store.get_conversation(conversation) is None
