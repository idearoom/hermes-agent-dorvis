"""Fail-closed routing tests for the gateway's configured session store."""

from unittest.mock import MagicMock, patch

import pytest

from gateway.session import SessionStore


def test_configured_postgres_init_failure_never_falls_back_to_jsonl(
    monkeypatch, tmp_path
):
    monkeypatch.setenv("HERMES_STATE_STORE_DSN", "postgresql://configured")
    config = MagicMock()
    config.write_sessions_json = True

    with (
        patch(
            "hermes_state.SessionDB",
            side_effect=RuntimeError("postgres unavailable"),
        ),
        pytest.raises(
            RuntimeError,
            match="HERMES_SESSION_STORE_POSTGRES_REQUIRED",
        ),
    ):
        SessionStore(tmp_path, config)


def test_gateway_runner_configured_postgres_init_failure_is_fatal(monkeypatch):
    from gateway.config import GatewayConfig
    from gateway.run import GatewayRunner

    monkeypatch.setenv("HERMES_STATE_STORE_DSN", "postgresql://configured")

    with (
        patch("gateway.run.SessionStore", return_value=MagicMock()),
        patch.object(
            GatewayRunner,
            "_open_session_db_for_active_scope",
            side_effect=RuntimeError("postgres unavailable"),
        ),
        pytest.raises(
            RuntimeError,
            match="HERMES_SESSION_STORE_POSTGRES_REQUIRED",
        ),
    ):
        GatewayRunner(GatewayConfig())
