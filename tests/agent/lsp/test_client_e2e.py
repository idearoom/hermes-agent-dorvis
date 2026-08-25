"""End-to-end client tests against the in-process mock LSP server.

Spins up :file:`_mock_lsp_server.py` as an actual subprocess, drives
it through real LSP traffic, and asserts diagnostic flow.  This is
the closest thing we have to integration coverage without requiring
pyright/gopls/etc. to be installed in CI.
"""
from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

import pytest

import agent.lsp.client as lsp_client
from agent.lsp.client import LSPClient


MOCK_SERVER = str(Path(__file__).parent / "_mock_lsp_server.py")


def _client(
    workspace: Path,
    script: str = "clean",
    *,
    exit_marker: Path | None = None,
) -> LSPClient:
    env = {"MOCK_LSP_SCRIPT": script, "PYTHONPATH": os.environ.get("PYTHONPATH", "")}
    if exit_marker is not None:
        env["MOCK_LSP_EXIT_MARKER"] = str(exit_marker)
    return LSPClient(
        server_id=f"mock-{script}",
        workspace_root=str(workspace),
        command=[sys.executable, MOCK_SERVER],
        env=env,
        cwd=str(workspace),
    )


@pytest.mark.asyncio
async def test_client_lifecycle_clean(tmp_path: Path):
    """Full lifecycle: spawn, initialize, open, get clean diagnostics, shutdown."""
    f = tmp_path / "x.py"
    f.write_text("print('hi')\n")

    client = _client(tmp_path, "clean")
    await client.start()
    try:
        assert client.is_running
        version = await client.open_file(str(f), language_id="python")
        assert version == 0
        await client.wait_for_diagnostics(str(f), version, mode="document")
        diags = client.diagnostics_for(str(f))
        assert diags == []
    finally:
        await client.shutdown()
    assert not client.is_running


@pytest.mark.asyncio
async def test_shutdown_allows_cooperative_server_to_exit_cleanly(tmp_path: Path):
    """A server that accepted ``exit`` gets a grace period before SIGTERM."""
    marker = tmp_path / "clean-exit.marker"
    client = _client(tmp_path, "delayed_exit", exit_marker=marker)
    await client.start()

    await client.shutdown()

    assert marker.read_text() == "clean\n"
    assert not client.is_running


@pytest.mark.asyncio
async def test_shutdown_terminates_server_that_ignores_exit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    """The clean-exit wait remains bounded when a server does not cooperate."""
    monkeypatch.setattr(lsp_client, "GRACEFUL_EXIT_WAIT", 0.05)
    monkeypatch.setattr(lsp_client, "SHUTDOWN_GRACE", 0.1)
    client = _client(tmp_path, "ignore_exit")
    await client.start()

    await asyncio.wait_for(client.shutdown(), timeout=1.0)

    assert not client.is_running


@pytest.mark.asyncio
async def test_client_receives_published_errors(tmp_path: Path):
    f = tmp_path / "x.py"
    f.write_text("print('hi')\n")

    client = _client(tmp_path, "errors")
    await client.start()
    try:
        version = await client.open_file(str(f), language_id="python")
        await client.wait_for_diagnostics(str(f), version, mode="document")
        diags = client.diagnostics_for(str(f))
        assert len(diags) == 1
        d = diags[0]
        assert d["severity"] == 1
        assert d["code"] == "MOCK001"
        assert d["source"] == "mock-lsp"
        assert "synthetic error" in d["message"]
    finally:
        await client.shutdown()





