"""Behavioral tests for the Postgres SessionDB backend (D6b foundation, AE-61).

Per-turn hot-path parity with the SQLite SessionDB: sessions + token counts,
append/get messages (incl. multimodal content + tool_calls round-trip),
state_meta, compression locks, and Postgres full-text search.

Requires a Postgres DSN via HERMES_D6_TEST_DSN; skipped otherwise so the default
CI run is unaffected. Module loaded by file path to avoid heavy package imports.
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

_MODULE_PATH = pathlib.Path(__file__).resolve().parents[1] / "hermes_state_pg.py"
_spec = importlib.util.spec_from_file_location("hermes_state_pg", _MODULE_PATH)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
PgSessionDB = _mod.PgSessionDB


@pytest.fixture()
def db():
    with psycopg.connect(_DSN, autocommit=True) as conn:
        conn.execute("DROP SCHEMA IF EXISTS hermes_gw CASCADE")
    d = PgSessionDB(_DSN)
    try:
        yield d
    finally:
        d.close()


def test_create_and_get_session(db):
    db.create_session("s1", "gateway", model="gpt-5.5", user_id="u1")
    s = db.get_session("s1")
    assert s["id"] == "s1" and s["source"] == "gateway"
    assert s["model"] == "gpt-5.5" and s["user_id"] == "u1"
    assert s["message_count"] == 0
    assert db.get_session("missing") is None


def test_create_session_idempotent(db):
    db.create_session("s1", "gateway", model="a")
    db.create_session("s1", "gateway", model="b")  # ON CONFLICT DO NOTHING
    assert db.get_session("s1")["model"] == "a"
    assert db.session_count() == 1


def test_append_and_get_messages_order_and_counts(db):
    db.create_session("s1", "gateway")
    db.append_message("s1", "user", content="hello")
    db.append_message("s1", "assistant", content="hi there")
    msgs = db.get_messages("s1")
    assert [m["role"] for m in msgs] == ["user", "assistant"]
    assert [m["content"] for m in msgs] == ["hello", "hi there"]
    assert db.get_session("s1")["message_count"] == 2
    assert db.message_count("s1") == 2


def test_tool_call_count_and_tool_calls_roundtrip(db):
    db.create_session("s1", "gateway")
    tc = [{"id": "c1", "function": {"name": "terminal", "arguments": "{}"}}]
    db.append_message("s1", "assistant", content=None, tool_calls=tc)
    msgs = db.get_messages("s1")
    assert msgs[0]["tool_calls"] == tc  # JSON round-trip
    assert db.get_session("s1")["tool_call_count"] == 1


def test_multimodal_content_roundtrip(db):
    db.create_session("s1", "gateway")
    content = [{"type": "text", "text": "look"},
               {"type": "image_url", "image_url": {"url": "http://x/y.png"}}]
    db.append_message("s1", "user", content=content)
    got = db.get_messages("s1")[0]["content"]
    assert got == content  # encoded via the \x00json: prefix, decoded back to a list


def test_token_counts_incremental_and_absolute(db):
    db.create_session("s1", "gateway")
    db.update_token_counts("s1", input_tokens=10, output_tokens=5, api_call_count=1)
    db.update_token_counts("s1", input_tokens=3, output_tokens=2, api_call_count=1)
    s = db.get_session("s1")
    assert s["input_tokens"] == 13 and s["output_tokens"] == 7 and s["api_call_count"] == 2
    db.update_token_counts("s1", input_tokens=100, output_tokens=50, api_call_count=9, absolute=True)
    s = db.get_session("s1")
    assert s["input_tokens"] == 100 and s["output_tokens"] == 50 and s["api_call_count"] == 9


def test_token_counts_autocreates_session(db):
    # SQLite parity: update_token_counts INSERT OR IGNOREs the session row first.
    db.update_token_counts("ghost", input_tokens=5)
    assert db.get_session("ghost")["input_tokens"] == 5


def test_meta_upsert(db):
    assert db.get_meta("k") is None
    db.set_meta("k", "v1")
    assert db.get_meta("k") == "v1"
    db.set_meta("k", "v2")
    assert db.get_meta("k") == "v2"


def test_compression_lock_mutual_exclusion_and_release(db):
    assert db.try_acquire_compression_lock("s1", "holderA") is True
    assert db.try_acquire_compression_lock("s1", "holderB") is False  # A holds it
    assert db.get_compression_lock_holder("s1") == "holderA"
    db.release_compression_lock("s1", "holderB")  # wrong holder -> no-op
    assert db.get_compression_lock_holder("s1") == "holderA"
    db.release_compression_lock("s1", "holderA")
    assert db.get_compression_lock_holder("s1") is None
    assert db.try_acquire_compression_lock("s1", "holderB") is True  # now free


def test_compression_lock_expiry_reclaim(db):
    assert db.try_acquire_compression_lock("s1", "old", ttl_seconds=-1) is True  # already expired
    # expired -> reclaimable by a new holder
    assert db.try_acquire_compression_lock("s1", "new") is True
    assert db.get_compression_lock_holder("s1") == "new"


def test_search_messages_fulltext_and_substring(db):
    db.create_session("s1", "gateway")
    db.append_message("s1", "user", content="Deploy the staging gateway and verify the swarm")
    db.append_message("s1", "assistant", content="调试中文搜索 substring matching")
    db.append_message("s1", "user", content="unrelated pricing note")
    full = db.search_messages("swarm")
    assert any("swarm" in m["content"] for m in full)
    cjk = db.search_messages("中文")
    assert any("中文" in m["content"] for m in cjk)
    sub = db.search_messages("gatew")  # substring via trigram/ILIKE
    assert any("gateway" in m["content"] for m in sub)


def test_replace_messages(db):
    db.create_session("s1", "gateway")
    db.append_message("s1", "user", content="old1")
    db.append_message("s1", "user", content="old2")
    db.replace_messages("s1", [{"role": "user", "content": "new1"},
                               {"role": "assistant", "content": "new2"}])
    msgs = db.get_messages("s1")
    assert [m["content"] for m in msgs] == ["new1", "new2"]
