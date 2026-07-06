#!/usr/bin/env python3
"""Two-process gateway drain simulation (AE-117, parent-repo ADR 0177).

Proves the drain-based blue/green handoff shape locally with two isolated
API-server gateway processes (separate HERMES_HOME → separate SQLite state):

  * Process A ("outgoing"): carries an in-flight slow run, is drained via
    POST /admin/drain, REFUSES new launches with the 503
    ``{"error": {"code": "gateway_draining"}}`` contract, finishes its
    in-flight run, streams the terminal event to its established SSE
    consumer, and self-exits with code 0 at zero active runs.
  * Process B ("incoming"): keeps accepting and completing new launches
    concurrently throughout A's drain window.

The shared-Postgres session-store variant of this simulation is
``scripts/drain_mode_sim_pg.py`` (AE-115 + AE-117 combined evidence); this
script deliberately uses per-process SQLite state.

Usage:
    .venv/bin/python scripts/drain_mode_sim.py            # orchestrator
    .venv/bin/python scripts/drain_mode_sim.py child ...  # internal

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
import urllib.error
import urllib.request

API_KEY = "drain-sim-key-0123456789abcdef"  # test-only; >=16 chars for the startup guard


# ---------------------------------------------------------------------------
# Child: a single API-server gateway process with a fake slow agent
# ---------------------------------------------------------------------------


def run_child(port: int, settle_seconds: float, cap_seconds: float) -> int:
    """Run one gateway API-server process until drained or killed."""
    import asyncio
    import threading

    # HERMES_HOME must already be set by the orchestrator (isolated per child).
    from gateway.config import PlatformConfig
    from gateway.drain_mode import drain_coordinator_loop, get_drain_mode
    from gateway.platforms.api_server import APIServerAdapter

    class FakeAgent:
        """Deterministic stand-in for AIAgent: 'sleep:N' inputs run N seconds."""

        session_prompt_tokens = 1
        session_completion_tokens = 1
        session_total_tokens = 2
        _session_messages: list = []

        def __init__(self) -> None:
            self._interrupted = threading.Event()

        def run_conversation(self, user_message="", conversation_history=None, task_id=None):
            delay = 0.5
            if isinstance(user_message, str) and user_message.startswith("sleep:"):
                try:
                    delay = float(user_message.split(":", 1)[1])
                except ValueError:
                    pass
            self._interrupted.wait(timeout=delay)
            state = "interrupted" if self._interrupted.is_set() else "done"
            return {"final_response": f"{state} pid={os.getpid()}"}

        def interrupt(self, message=None):
            self._interrupted.set()

        def shutdown_memory_provider(self, *args, **kwargs):
            return None

        def close(self):
            return None

    # Swap the real agent factory for the fake before any run starts.
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
                write_marker=True,  # exercises the relay drain-marker path too
            )
        )
        print(f"child[{port}]: ready (pid={os.getpid()})", flush=True)
        await drained.wait()
        coordinator.cancel()
        await adapter.disconnect()
        print(f"child[{port}]: drained to zero — exiting cleanly", flush=True)
        return 0

    return asyncio.run(_main())


# ---------------------------------------------------------------------------
# Orchestrator helpers
# ---------------------------------------------------------------------------


def _request(method: str, url: str, body=None, timeout: float = 10.0):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(
        url,
        data=data,
        method=method,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {API_KEY}",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, json.loads(resp.read().decode() or "{}")
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read().decode() or "{}")


def _wait_healthy(port: int, deadline_s: float = 30.0) -> None:
    deadline = time.monotonic() + deadline_s
    while time.monotonic() < deadline:
        try:
            status, _ = _request("GET", f"http://127.0.0.1:{port}/health", timeout=2.0)
            if status == 200:
                return
        except Exception:
            pass
        time.sleep(0.25)
    raise AssertionError(f"gateway on :{port} never became healthy")


def _read_events(port: int, run_id: str, timeout: float = 30.0) -> str:
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/runs/{run_id}/events",
        headers={"Authorization": f"Bearer {API_KEY}"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read().decode()


def _spawn_child(port: int, home: str, settle: float, cap: float) -> subprocess.Popen:
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    env = dict(os.environ)
    env["HERMES_HOME"] = home
    env["API_SERVER_KEY"] = API_KEY
    # THIS checkout must win over any editable hermes install in the venv:
    # sys.path[0] for a script is scripts/, not the repo root.
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
        cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    )


# ---------------------------------------------------------------------------
# Orchestrator scenario
# ---------------------------------------------------------------------------


def run_orchestrator(port_a: int, port_b: int) -> int:
    evidence: dict = {"scenario": "AE-117 two-process drain simulation"}
    procs: list[subprocess.Popen] = []
    try:
        with tempfile.TemporaryDirectory(prefix="drain-sim-a-") as home_a, \
                tempfile.TemporaryDirectory(prefix="drain-sim-b-") as home_b:
            a = _spawn_child(port_a, home_a, settle=1.0, cap=120.0)
            b = _spawn_child(port_b, home_b, settle=1.0, cap=120.0)
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

            # 1. In-flight slow run on A.
            status, launched = _request(
                "POST", f"http://127.0.0.1:{port_a}/v1/runs", {"input": "sleep:8"}
            )
            assert status == 202, (status, launched)
            run_a = launched["run_id"]
            time.sleep(0.5)  # let the executor thread enter the run

            # 2. Drain A.
            status, drain_resp = _request(
                "POST", f"http://127.0.0.1:{port_a}/admin/drain",
                {"reason": "sim-blue-green"},
            )
            assert status == 202 and drain_resp["draining"] is True, (status, drain_resp)
            evidence["a_drain_engaged"] = {
                "active_runs": drain_resp["active_runs"],
                "reason": drain_resp["drain_reason"],
            }
            assert drain_resp["active_runs"] == 1, drain_resp

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
            assert ready["checks"]["gateway_not_draining"] is False, ready
            evidence["a_ready_during_drain"] = {
                "status": status,
                "draining": ready["draining"],
                "active_runs": ready["active_runs"],
            }

            # 5. B accepts and completes new work concurrently.
            status, b_run = _request(
                "POST", f"http://127.0.0.1:{port_b}/v1/runs", {"input": "quick"}
            )
            assert status == 202, (status, b_run)
            b_events = _read_events(port_b, b_run["run_id"])
            assert "run.completed" in b_events, b_events
            evidence["b_accepts_during_a_drain"] = {
                "status": status,
                "run_completed": "run.completed" in b_events,
            }

            # 6. A's established stream serves the in-flight run to a clean end.
            a_events = _read_events(port_a, run_a, timeout=30.0)
            assert "run.completed" in a_events, a_events
            assert "done" in a_events, a_events  # finished naturally, NOT interrupted
            evidence["a_inflight_run"] = {
                "run_completed": True,
                "finished_naturally": "done" in a_events,
            }

            # 7. A self-exits cleanly at zero (settle 1s + margin).
            a_exit = a.wait(timeout=30.0)
            assert a_exit == 0, f"process A exited {a_exit}, expected 0"
            evidence["a_self_exit"] = {"returncode": a_exit}

            # 8. B is still healthy and still accepting after A is gone.
            status, b_run2 = _request(
                "POST", f"http://127.0.0.1:{port_b}/v1/runs", {"input": "again"}
            )
            assert status == 202, (status, b_run2)
            evidence["b_still_accepting_after_a_exit"] = {"status": status}

            b.send_signal(signal.SIGTERM)
            try:
                b.wait(timeout=10.0)
            except subprocess.TimeoutExpired:
                b.kill()

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
    parser.add_argument("--a-port", type=int, default=8791)
    parser.add_argument("--b-port", type=int, default=8792)
    args = parser.parse_args()

    if args.mode == "child":
        return run_child(args.port, args.settle_seconds, args.cap_seconds)
    return run_orchestrator(args.a_port, args.b_port)


if __name__ == "__main__":
    sys.exit(main())
