# Dorvis Runtime Patch Manifest

This fork carries IdeaRoom-specific runtime patches on top of NousResearch
`hermes-agent` upstream. Keep this file current whenever a patch is added,
removed, rebased, or promoted.

Current upstream base for the carried branch: `d1383a6b1450c6c139720b1b01f8b99cc130453f`.

## Patch Inventory

| Patch | Status | Purpose | Main files | Conflict surface | Owner |
|---|---|---|---|---|---|
| Request metadata and tool-dispatch observability | Promoted in fork history | Preserve request/session/turn metadata through hooks and tool execution so gateway, traces, and downstream plugins can correlate turns reliably. | `run_agent.py`, `model_tools.py`, `hermes_cli/hooks.py`, `hermes_cli/plugins.py`, `tools/delegate_tool.py` | Medium: agent loop and hook dispatch | IdeaRoom Agentic Systems |
| Runtime footer preservation | Promoted in fork history | Preserve runtime footer usage fields across gateway streaming and response handling. | `gateway/runtime_footer.py`, `gateway/run.py`, `gateway/platforms/api_server.py` | Medium: gateway stream/result formatting | IdeaRoom Agentic Systems |
| Hindsight retained-chat tagging | Promoted in fork history | Tag retained web chats with useful provenance for Hindsight recall and auditability. | `plugins/memory/hindsight/__init__.py` | Low: Hindsight plugin provider surface | IdeaRoom Agentic Systems |
| Stateful compression quality gate | Promoted in fork history | Improve context-compression safety by preserving ledger/state details and validating compression quality. | `agent/context_compressor.py`, `agent/conversation_compression.py`, `agent/conversation_loop.py`, `agent/turn_context.py`, `agent/turn_finalizer.py` | High: agent context and compression flow | IdeaRoom Agentic Systems |
| Responses API Postgres store (D6a) | Promoted in fork history | Move the gateway Responses API store off EFS SQLite and into Postgres for concurrent web traffic; Postgres is durable state and does not use the old SQLite LRU cap. | `gateway/platforms/api_server.py`, `gateway/platforms/response_store_pg.py` | Medium: gateway response-store boundary | IdeaRoom Agentic Systems |
| Browser raw-CDP supervisor disable | Promoted in AWS runtime `69731a494` | Let AWS Browserless raw-CDP stages disable the second persistent CDP supervisor client, avoiding an idle extra Chrome per browser-using chat. | `tools/browser_tool.py`, `tests/tools/test_browser_cdp_override.py` | Low: browser CDP setup path | IdeaRoom Agentic Systems |
| MCP Hermes context keying | Promoted in AWS runtime `69731a494` | Let trusted MCP servers opt in to `_meta.hermes.task_id` on `tools/call` so sandbox orchestrators can key one container per chat without trusting model arguments. | `tools/mcp_tool.py`, `tests/tools/test_mcp_tool.py`, `website/docs/reference/mcp-config-reference.md`, `website/docs/user-guide/features/mcp.md` | Low: opt-in MCP call metadata | IdeaRoom Agentic Systems |
| execute_code per-turn budget | Promoted in AWS runtime `69731a494` | Cap aggregate `execute_code` wall-clock time per assistant turn and pass `turn_id` into the tool handler, reducing runaway multi-call CPU exposure without adding a concurrency cap. | `model_tools.py`, `tools/code_execution_tool.py`, `tests/tools/test_code_execution.py`, `website/docs/user-guide/features/code-execution.md` | Medium: tool dispatch and execution timeout path | IdeaRoom Agentic Systems |
| Runtime source revision health field | Promoted in AWS runtime `69731a494` | Expose the loaded Hermes source commit through gateway health/status so promotion audits can compare running process code to the tested fork commit. | `gateway/status.py`, `gateway/platforms/api_server.py`, `tests/gateway/test_status.py`, `tests/gateway/test_api_server.py` | Low: health/status payloads | IdeaRoom Agentic Systems |
| Responses API usage transparency (AE-24) | Source-only, not deployed | Preserve context-window usage, compaction count, cost estimate, and structured failure type in `/v1/responses` terminal envelopes for Hermes Web observability. | `gateway/platforms/api_server.py`, `tests/gateway/test_api_server.py` | Low: additive Responses API envelope fields | IdeaRoom Agentic Systems |

## Maintenance Rules

- Prefer additive modules and narrow hook-point edits over broad upstream rewrites.
- Keep model-visible tool schemas stable unless the product behavior truly changes.
- Add or update focused tests before implementation changes.
- After rebasing, run the focused tests named by each affected patch plus the
  parent repo promotion gate before deploying the fork runtime.
- When a source-only patch is committed and promoted, update its status here and
  the parent repo's `packages/hermes-config/hermes-fork-state.json`.
