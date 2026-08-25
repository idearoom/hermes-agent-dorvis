"""Per-run compression event callback on ``_emit_compression_hook``.

The gateway's streaming Responses path sets ``agent._compression_event_callback``
so the compaction lifecycle reaches the client SSE stream. The callback is
isolated from the plugin hook: either side failing must not suppress the other.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

from agent.conversation_compression import _emit_compression_hook


def _agent(**attrs):
    return SimpleNamespace(session_id="sess-1", _request_metadata={}, **attrs)


class TestCompressionEventCallback:
    def test_callback_receives_hook_name_and_payload_with_session_id(self):
        seen = []
        agent = _agent(_compression_event_callback=lambda name, payload: seen.append((name, payload)))

        _emit_compression_hook(agent, "context_compression_started", pre_message_count=42, pre_tokens=1000)

        assert len(seen) == 1
        name, payload = seen[0]
        assert name == "context_compression_started"
        assert payload["pre_message_count"] == 42
        assert payload["pre_tokens"] == 1000
        assert payload["session_id"] == "sess-1"

    def test_missing_callback_is_a_noop(self):
        _emit_compression_hook(_agent(), "context_compression_started", pre_message_count=1)

    def test_callback_exception_is_swallowed(self):
        def _boom(_name, _payload):
            raise RuntimeError("callback exploded")

        agent = _agent(_compression_event_callback=_boom)
        _emit_compression_hook(agent, "context_compression_completed", post_message_count=5)

    def test_plugin_hook_failure_does_not_suppress_callback(self):
        seen = []
        agent = _agent(_compression_event_callback=lambda name, payload: seen.append(name))

        with patch("hermes_cli.plugins.invoke_hook", side_effect=RuntimeError("observer down")):
            _emit_compression_hook(agent, "context_compression_aborted", pre_message_count=3)

        assert seen == ["context_compression_aborted"]
