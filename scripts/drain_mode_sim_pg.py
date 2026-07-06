#!/usr/bin/env python3
"""Two-process SHARED-POSTGRES gateway drain simulation (AE-115 + AE-117).

The shared-store companion to ``scripts/drain_mode_sim.py`` and the
acceptance evidence parent-repo ADR 0177 calls for before the ECS blue/green
flip: two gateway API-server processes run against the SAME Postgres session
store (``HERMES_STATE_STORE_DSN`` → ``PgSessionDB`` via the
``SessionDB.__new__`` seam) while one of them drains.

Scenario (same drain choreography as drain_mode_sim.py, plus store work):

  * Process A ("outgoing"): carries an in-flight slow run that writes
    session/message rows to the shared store and HOLDS the shared
    compression lock for the probe session while it sleeps; A is drained via
    POST /admin/drain, refuses new launches with the 503
    ``{"error": {"code": "gateway_draining"}}`` contract, finishes the
    in-flight run, streams the terminal event, and self-exits 0.
  * Process B ("incoming"): accepts and completes new runs against the same
    Postgres store throughout A's drain window. B's mid-drain run must LOSE
    the compression-lock contention against A's holder (single winner across
    processes); B's post-exit run must WIN it (A released cleanly).

After the choreography the orchestrator connects to Postgres directly and
asserts no corruption:

  * ``compression_locks`` is empty (every holder released),
  * every simulated session row exists exactly once with
    ``message_count`` == its actual ``messages`` row count,
  * exactly one ``schema_version`` row at the pinned version,
  * A's rows were not disturbed by B and vice versa.

Usage (point it at a throwaway database — the ``hermes_state`` schema is
dropped first):

    .venv/bin/python scripts/drain_mode_sim_pg.py \\
        --dsn postgresql://postgres:test@127.0.0.1:54331/postgres

Exit code 0 with a JSON evidence summary on stdout when every assertion
holds; non-zero with a failure message otherwise.
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import tempfile
import time

_SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_SCRIPTS_DIR)
for _p in (_SCRIPTS_DIR, _REPO_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import drain_mode_sim as base  # reuse the SQLite sim's HTTP/orchestration helpers

API_KEY = base.API_KEY

# One well-known session id both processes contend on for the compression
# lock — proves the Pg lock table gives a single winner ACROSS processes.
SHARED_LOCK_SESSION = "drain-sim-pg-shared-lock-probe"


# ---------------------------------------------------------------------------
# Child: one API-server gateway process, fake agent that works the shared store
# ---------------------------------------------------------------------------


def run_child(port: int, settle_seconds: float, cap_seconds: float) -> int:
    """Run one gateway API-server process against the shared Postgres store."""
    import asyncio
    import threading

    from gateway.config import PlatformConfig
    from gateway.drain_mode import drain_coordinator_loop, get_drain_mode
    from gateway.platforms.api_server import APIServerAdapter
    from hermes_state import SessionDB
    from hermes_state_pg import PgSessionDB

    # The whole point: the bare SessionDB() every gateway component uses must
    # resolve to the Postgres backend through the __new__ env seam.
    state_db = SessionDB()
    if not isinstance(state_db, PgSessionDB):
        print(
            f"child[{port}]: FATAL — SessionDB() resolved to "
            f"{type(state_db).__name__}, expected PgSessionDB "
            f"(HERMES_STATE_STORE_DSN missing?)",
            file=sys.stderr,
        )
        return 3
    print(
        f"child[{port}]: state backend {type(state_db).__name__} (pid={os.getpid()})",
        flush=True,
    )
    state_db.ensure_session(SHARED_LOCK_SESSION, source="api_server")

    pid = os.getpid()
    seq_lock = threading.Lock()
    seq = {"n": 0}

    class FakeAgent:
        """Deterministic AIAgent stand-in that exercises the shared store.

        Every run creates its own session row and appends a user+assistant
        message pair. ``sleep:N`` runs additionally HOLD the shared
        compression lock for the whole N seconds (the outgoing gateway's
        in-flight work); all other runs PROBE that lock and report whether
        they contended, via the final_response (which lands in the run's
        SSE events for the orchestrator to assert on).
        """

        session_prompt_tokens = 1
        session_completion_tokens = 1
        session_total_tokens = 2
        _session_messages: list = []

        def __init__(self) -> None:
            self._interrupted = threading.Event()

        def run_conversation(self, user_message="", conversation_history=None, task_id=None):
            with seq_lock:
                seq["n"] += 1
                n = seq["n"]
            sim_session = f"drain-sim-pg-{pid}-{n}"
            state_db.create_session(sim_session, source="api_server")
            state_db.append_message(sim_session, "user", str(user_message))

            delay = 0.5
            holder = f"sim-{pid}-{n}"
            lock_note = ""
            if isinstance(user_message, str) and user_message.startswith("sleep:"):
                try:
                    delay = float(user_message.split(":", 1)[1])
                except ValueError:
                    pass
                # Outgoing gateway's slow run: hold the shared lock while
                # sleeping so the concurrent process must lose contention.
                got = state_db.try_acquire_compression_lock(
                    SHARED_LOCK_SESSION, holder, ttl_seconds=120.0
                )
                lock_note = f" shared_lock_held={got}"
                try:
                    self._interrupted.wait(timeout=delay)
                finally:
                    if got:
                        state_db.release_compression_lock(SHARED_LOCK_SESSION, holder)
            else:
                got = state_db.try_acquire_compression_lock(
                    SHARED_LOCK_SESSION, holder, ttl_seconds=120.0
                )
                if got:
                    state_db.release_compression_lock(SHARED_LOCK_SESSION, holder)
                lock_note = f" shared_lock_contended={not got}"
                self._interrupted.wait(timeout=delay)

            state = "interrupted" if self._interrupted.is_set() else "done"
            final = f"{state} pid={pid} session={sim_session}{lock_note}"
            state_db.append_message(sim_session, "assistant", final)
            return {"final_response": final}

        def interrupt(self, message=None):
            self._interrupted.set()

        def shutdown_memory_provider(self, *args, **kwargs):
            return None

        def close(self):
            return None

    APIServerAdapter._create_agent = lambda self, **kwargs: FakeAgent()  # type: ignore[method-assign]

    async def _main() -> int:
        adapter = APIServerAdapter(
            PlatformConfig(
                enabled=True,
                extra={"host": "127.0.0.1", "port": port, "key": API_KEY},
            )
        )
        if not await adapter.connect():
            print(f"child[{port}]: failed to start API server", file=sys.stderr)
            return 2

        drained = asyncio.Event()
        coordinator = asyncio.create_task(
            drain_coordinator_loop(
                get_drain_mode(),
                shutdown_cb=drained.set,
                poll_interval=0.2,
                cap_seconds=cap_seconds,
                settle_seconds=settle_seconds,
                write_marker=True,
            )
        )
        print(f"child[{port}]: ready (pid={os.getpid()})", flush=True)
        await drained.wait()
        coordinator.cancel()
        await adapter.disconnect()
        state_db.close()
        print(f"child[{port}]: drained to zero — exiting cleanly", flush=True)
        return 0

    return asyncio.run(_main())


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


def _spawn_child(port: int, home: str, dsn: str, settle: float, cap: float) -> subprocess.Popen:
    repo_root = os.path.dirname(_SCRIPTS_DIR)
    env = dict(os.environ)
    env["HERMES_HOME"] = home
    env["API_SERVER_KEY"] = API_KEY
    env["HERMES_STATE_STORE_DSN"] = dsn  # BOTH children share this store
    env["PYTHONPATH"] = repo_root + (
        os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else ""
    )
    return subprocess.Popen(
        [
            sys.executable, os.path.abspath(__file__), "child",
            "--port", str(port),
            "--settle-seconds", str(settle),
            "--cap-seconds", str(cap),
        ],
        env=env,
        cwd=repo_root,
    )


def _store_snapshot(dsn: str) -> dict:
    """Read the shared store's consistency-relevant state via psycopg."""
    import psycopg

    with psycopg.connect(dsn) as conn:
        locks = conn.execute(
            "SELECT session_id, holder FROM hermes_state.compression_locks"
        ).fetchall()
        sessions = conn.execute(
            "SELECT id, message_count FROM hermes_state.sessions"
            r" WHERE id ~ '^drain-sim-pg-\d+-\d+$' ORDER BY id"
        ).fetchall()
        msg_counts = dict(
            conn.execute(
                "SELECT session_id, COUNT(*) FROM hermes_state.messages"
                r" WHERE session_id ~ '^drain-sim-pg-\d+-\d+$' GROUP BY session_id"
            ).fetchall()
        )
        versions = conn.execute(
            "SELECT version FROM hermes_state.schema_version"
        ).fetchall()
    return {
        "locks": locks,
        "sessions": sessions,
        "msg_counts": msg_counts,
        "versions": [v[0] for v in versions],
    }


def run_orchestrator(port_a: int, port_b: int, dsn: str) -> int:
    from hermes_state_pg import EXPECTED_SCHEMA_VERSION, _SCHEMA

    evidence: dict = {
        "scenario": "AE-115 + AE-117 two-process SHARED-POSTGRES drain simulation",
        "dsn_host": dsn.split("@")[-1] if "@" in dsn else dsn,
    }
    procs: list[subprocess.Popen] = []
    _request, _wait_healthy, _read_events = base._request, base._wait_healthy, base._read_events
    try:
        # Clean slate: both children bootstrap the schema on first touch.
        import psycopg

        with psycopg.connect(dsn, autocommit=True) as conn:
            conn.execute(f"DROP SCHEMA IF EXISTS {_SCHEMA} CASCADE")

        with tempfile.TemporaryDirectory(prefix="drain-sim-pg-a-") as home_a, \
                tempfile.TemporaryDirectory(prefix="drain-sim-pg-b-") as home_b:
            a = _spawn_child(port_a, home_a, dsn, settle=1.0, cap=120.0)
            b = _spawn_child(port_b, home_b, dsn, settle=1.0, cap=120.0)
            procs = [a, b]
            _wait_healthy(port_a)
            _wait_healthy(port_b)

            # Pre-drain readiness: A reports draining=false.
            _, ready_before = _request("GET", f"http://127.0.0.1:{port_a}/ready")
            assert ready_before["draining"] is False, ready_before
            evidence["a_ready_before"] = {
                "draining": ready_before["draining"],
                "active_runs": ready_before["active_runs"],
            }

            # 1. In-flight slow run on A — writes rows to the shared store
            #    and holds the shared compression lock while it sleeps.
            status, launched = _request(
                "POST", f"http://127.0.0.1:{port_a}/v1/runs", {"input": "sleep:8"}
            )
            assert status == 202, (status, launched)
            run_a = launched["run_id"]
            time.sleep(1.0)  # let the executor thread enter the run + take the lock

            # 2. Drain A.
            status, drain_resp = _request(
                "POST", f"http://127.0.0.1:{port_a}/admin/drain",
                {"reason": "sim-blue-green-shared-pg"},
            )
            assert status == 202 and drain_resp["draining"] is True, (status, drain_resp)
            assert drain_resp["active_runs"] == 1, drain_resp
            evidence["a_drain_engaged"] = {
                "active_runs": drain_resp["active_runs"],
                "reason": drain_resp["drain_reason"],
            }

            # 3. A refuses new launches with the recognizable contract.
            status, refused = _request(
                "POST", f"http://127.0.0.1:{port_a}/v1/runs", {"input": "hello"}
            )
            assert status == 503, (status, refused)
            assert refused["error"]["code"] == "gateway_draining", refused
            evidence["a_refusal"] = {"status": status, "code": refused["error"]["code"]}

            # 4. A's readiness reports draining with one active run.
            status, ready = _request("GET", f"http://127.0.0.1:{port_a}/ready")
            assert status == 503 and ready["draining"] is True, (status, ready)
            assert ready["active_runs"] == 1, ready
            evidence["a_ready_during_drain"] = {
                "status": status,
                "draining": ready["draining"],
                "active_runs": ready["active_runs"],
            }

            # 5. Mid-drain: B accepts + completes a run against the SAME
            #    store, and must LOSE the shared-lock contention (A holds it).
            status, b_run = _request(
                "POST", f"http://127.0.0.1:{port_b}/v1/runs", {"input": "quick"}
            )
            assert status == 202, (status, b_run)
            b_events = _read_events(port_b, b_run["run_id"])
            assert "run.completed" in b_events, b_events
            assert "shared_lock_contended=True" in b_events, b_events
            evidence["b_mid_drain_run"] = {
                "status": status,
                "run_completed": True,
                "lost_lock_contention_to_a": True,
            }

            # 6. A's established stream serves the in-flight run to a clean
            #    end (finished naturally, lock was actually held).
            a_events = _read_events(port_a, run_a, timeout=30.0)
            assert "run.completed" in a_events, a_events
            assert "done" in a_events, a_events
            assert "shared_lock_held=True" in a_events, a_events
            evidence["a_inflight_run"] = {
                "run_completed": True,
                "finished_naturally": True,
                "held_shared_lock": True,
            }

            # 7. A self-exits cleanly at zero.
            a_exit = a.wait(timeout=30.0)
            assert a_exit == 0, f"process A exited {a_exit}, expected 0"
            evidence["a_self_exit"] = {"returncode": a_exit}

            # 8. B still accepts after A is gone, and now WINS the shared
            #    lock (A released it on the way out — no stale holder).
            status, b_run2 = _request(
                "POST", f"http://127.0.0.1:{port_b}/v1/runs", {"input": "again"}
            )
            assert status == 202, (status, b_run2)
            b2_events = _read_events(port_b, b_run2["run_id"])
            assert "run.completed" in b2_events, b2_events
            assert "shared_lock_contended=False" in b2_events, b2_events
            evidence["b_post_exit_run"] = {
                "status": status,
                "run_completed": True,
                "acquired_lock_after_a_release": True,
            }

            b.send_signal(signal.SIGTERM)
            try:
                b.wait(timeout=10.0)
            except subprocess.TimeoutExpired:
                b.kill()

        # 9. Store consistency: no lock leaks, no session/message skew.
        snap = _store_snapshot(dsn)
        assert snap["locks"] == [], f"compression_locks not clean: {snap['locks']}"
        # 3 simulated runs total: A's slow run, B's mid-drain run, B's
        # post-exit run. Each wrote exactly one session with 2 messages.
        assert len(snap["sessions"]) == 3, snap["sessions"]
        for session_id, message_count in snap["sessions"]:
            actual = snap["msg_counts"].get(session_id, 0)
            assert message_count == 2 == actual, (
                f"{session_id}: message_count={message_count}, rows={actual}"
            )
        assert snap["versions"] == [EXPECTED_SCHEMA_VERSION], snap["versions"]
        pids = sorted({s.split("-")[3] for s, _ in snap["sessions"]})
        assert len(pids) == 2, f"expected rows from 2 distinct pids: {snap['sessions']}"
        evidence["shared_store_consistency"] = {
            "compression_locks_clean": True,
            "sessions": [
                {"session_id": s, "message_count": c} for s, c in snap["sessions"]
            ],
            "message_rows_match_counts": True,
            "schema_version": snap["versions"][0],
            "writer_pids": pids,
        }

        evidence["result"] = "PASS"
        print(json.dumps(evidence, indent=2))
        return 0
    except AssertionError as exc:
        evidence["result"] = "FAIL"
        evidence["failure"] = str(exc)
        print(json.dumps(evidence, indent=2), file=sys.stderr)
        return 1
    finally:
        for proc in procs:
            if proc.poll() is None:
                proc.kill()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode")
    child = sub.add_parser("child", help="internal: run one gateway process")
    child.add_argument("--port", type=int, required=True)
    child.add_argument("--settle-seconds", type=float, default=1.0)
    child.add_argument("--cap-seconds", type=float, default=120.0)
    parser.add_argument("--a-port", type=int, default=8795)
    parser.add_argument("--b-port", type=int, default=8796)
    parser.add_argument(
        "--dsn",
        default=os.environ.get("HERMES_STATE_TEST_DSN", "").strip()
        or os.environ.get("HERMES_D6_TEST_DSN", "").strip(),
        help="Throwaway Postgres DSN (the hermes_state schema is DROPPED first). "
        "Defaults to $HERMES_STATE_TEST_DSN / $HERMES_D6_TEST_DSN.",
    )
    args = parser.parse_args()

    if args.mode == "child":
        # DSN arrives via HERMES_STATE_STORE_DSN in the child environment —
        # that is the seam under test.
        return run_child(args.port, args.settle_seconds, args.cap_seconds)
    if not args.dsn:
        parser.error("--dsn (or HERMES_STATE_TEST_DSN) is required")
    return run_orchestrator(args.a_port, args.b_port, args.dsn)


if __name__ == "__main__":
    sys.exit(main())
