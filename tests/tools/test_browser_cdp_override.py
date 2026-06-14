from unittest.mock import Mock, patch


HOST = "example-host"
PORT = 9223
WS_URL = f"ws://{HOST}:{PORT}/devtools/browser/abc123"
HTTP_URL = f"http://{HOST}:{PORT}"
VERSION_URL = f"{HTTP_URL}/json/version"


class TestResolveCdpOverride:
    def test_keeps_full_devtools_websocket_url(self):
        from tools.browser_tool import _resolve_cdp_override

        assert _resolve_cdp_override(WS_URL) == WS_URL

    def test_resolves_http_discovery_endpoint_to_websocket(self):
        from tools.browser_tool import _resolve_cdp_override

        response = Mock()
        response.raise_for_status.return_value = None
        response.json.return_value = {"webSocketDebuggerUrl": WS_URL}

        with patch("tools.browser_tool.requests.get", return_value=response) as mock_get:
            resolved = _resolve_cdp_override(HTTP_URL)

        assert resolved == WS_URL
        mock_get.assert_called_once_with(VERSION_URL, timeout=10)

    def test_resolves_bare_ws_hostport_to_discovery_websocket(self):
        from tools.browser_tool import _resolve_cdp_override

        response = Mock()
        response.raise_for_status.return_value = None
        response.json.return_value = {"webSocketDebuggerUrl": WS_URL}

        with patch("tools.browser_tool.requests.get", return_value=response) as mock_get:
            resolved = _resolve_cdp_override(f"ws://{HOST}:{PORT}")

        assert resolved == WS_URL
        mock_get.assert_called_once_with(VERSION_URL, timeout=10)

    def test_falls_back_to_raw_url_when_discovery_fails(self):
        from tools.browser_tool import _resolve_cdp_override

        with patch("tools.browser_tool.requests.get", side_effect=RuntimeError("boom")):
            assert _resolve_cdp_override(HTTP_URL) == HTTP_URL

    def test_normalizes_provider_returned_http_cdp_url_when_creating_session(self, monkeypatch):
        import tools.browser_tool as browser_tool

        provider = Mock()
        provider.create_session.return_value = {
            "session_name": "cloud-session",
            "bb_session_id": "bu_123",
            "cdp_url": "https://cdp.browser-use.example/session",
            "features": {"browser_use": True},
        }

        response = Mock()
        response.raise_for_status.return_value = None
        response.json.return_value = {"webSocketDebuggerUrl": WS_URL}

        monkeypatch.setattr(browser_tool, "_active_sessions", {})
        monkeypatch.setattr(browser_tool, "_session_last_activity", {})
        monkeypatch.setattr(browser_tool, "_start_browser_cleanup_thread", lambda: None)
        monkeypatch.setattr(browser_tool, "_update_session_activity", lambda task_id: None)
        monkeypatch.setattr(browser_tool, "_get_cdp_override", lambda: "")
        monkeypatch.setattr(browser_tool, "_get_cloud_provider", lambda: provider)

        with patch("tools.browser_tool.requests.get", return_value=response) as mock_get:
            session_info = browser_tool._get_session_info("task-browser-use")

        assert session_info["cdp_url"] == WS_URL
        provider.create_session.assert_called_once_with("task-browser-use")
        mock_get.assert_called_once_with(
            "https://cdp.browser-use.example/session/json/version",
            timeout=10,
        )


class TestGetCdpOverride:
    def test_prefers_env_var_over_config(self, monkeypatch):
        import tools.browser_tool as browser_tool

        monkeypatch.setenv("BROWSER_CDP_URL", HTTP_URL)
        monkeypatch.setattr(
            browser_tool,
            "read_raw_config",
            lambda: {"browser": {"cdp_url": "http://config-host:9222"}},
            raising=False,
        )

        response = Mock()
        response.raise_for_status.return_value = None
        response.json.return_value = {"webSocketDebuggerUrl": WS_URL}

        with patch("tools.browser_tool.requests.get", return_value=response) as mock_get:
            resolved = browser_tool._get_cdp_override()

        assert resolved == WS_URL
        mock_get.assert_called_once_with(VERSION_URL, timeout=10)


class TestCdpSupervisorEnabled:
    def test_enabled_by_default(self, monkeypatch):
        import tools.browser_tool as browser_tool

        monkeypatch.delenv("BROWSER_CDP_SUPERVISOR_ENABLED", raising=False)

        with patch("hermes_cli.config.read_raw_config", return_value={}):
            assert browser_tool._is_cdp_supervisor_enabled() is True

    def test_env_false_disables_supervisor(self, monkeypatch):
        import tools.browser_tool as browser_tool

        monkeypatch.setenv("BROWSER_CDP_SUPERVISOR_ENABLED", "false")

        assert browser_tool._is_cdp_supervisor_enabled() is False

    def test_config_false_disables_supervisor_when_env_missing(self, monkeypatch):
        import tools.browser_tool as browser_tool

        monkeypatch.delenv("BROWSER_CDP_SUPERVISOR_ENABLED", raising=False)

        with patch(
            "hermes_cli.config.read_raw_config",
            return_value={"browser": {"cdp_supervisor_enabled": False}},
        ):
            assert browser_tool._is_cdp_supervisor_enabled() is False

    def test_disabled_supervisor_does_not_start_registry(self, monkeypatch):
        import sys
        import types

        import tools.browser_tool as browser_tool

        calls = []

        class Registry:
            def get_or_start(self, **kwargs):
                calls.append(kwargs)

        monkeypatch.setenv("BROWSER_CDP_URL", "ws://browserless.internal/?launch=%7B%7D")
        monkeypatch.setenv("BROWSER_CDP_SUPERVISOR_ENABLED", "false")
        monkeypatch.setattr(
            browser_tool,
            "_get_dialog_policy_config",
            lambda: ("must_respond", 300.0),
        )

        monkeypatch.setitem(
            sys.modules,
            "tools.browser_supervisor",
            types.SimpleNamespace(SUPERVISOR_REGISTRY=Registry()),
        )

        browser_tool._ensure_cdp_supervisor("task-raw-cdp")

        assert calls == []

    def test_enabled_supervisor_still_starts_registry(self, monkeypatch):
        import sys
        import types

        import tools.browser_tool as browser_tool

        calls = []

        class Registry:
            def get_or_start(self, **kwargs):
                calls.append(kwargs)

        monkeypatch.setenv("BROWSER_CDP_URL", "ws://browserless.internal/?launch=%7B%7D")
        monkeypatch.setenv("BROWSER_CDP_SUPERVISOR_ENABLED", "true")
        monkeypatch.setattr(
            browser_tool,
            "_get_dialog_policy_config",
            lambda: ("must_respond", 300.0),
        )

        monkeypatch.setitem(
            sys.modules,
            "tools.browser_supervisor",
            types.SimpleNamespace(SUPERVISOR_REGISTRY=Registry()),
        )

        browser_tool._ensure_cdp_supervisor("task-raw-cdp")

        assert calls == [
            {
                "task_id": "task-raw-cdp",
                "cdp_url": "ws://browserless.internal/?launch=%7B%7D",
                "dialog_policy": "must_respond",
                "dialog_timeout_s": 300.0,
            }
        ]

    def test_uses_config_browser_cdp_url_when_env_missing(self, monkeypatch):
        import tools.browser_tool as browser_tool

        monkeypatch.delenv("BROWSER_CDP_URL", raising=False)

        response = Mock()
        response.raise_for_status.return_value = None
        response.json.return_value = {"webSocketDebuggerUrl": WS_URL}

        with patch("hermes_cli.config.read_raw_config", return_value={"browser": {"cdp_url": HTTP_URL}}), \
             patch("tools.browser_tool.requests.get", return_value=response) as mock_get:
            resolved = browser_tool._get_cdp_override()

        assert resolved == WS_URL
        mock_get.assert_called_once_with(VERSION_URL, timeout=10)
