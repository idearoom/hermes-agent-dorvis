# Dorvis Runtime Patch Manifest

This fork carries IdeaRoom-specific runtime patches on top of NousResearch
`hermes-agent` upstream. Keep this file current whenever a patch is added,
removed, rebased, or promoted.

Current upstream base for the carried branch: `2ebf9a90b762f21e33318b342e921b17e3d81946`
(merged into fork `main` as `37bc3ce4a`; previous base `388268ecde085a22c15474fea1723db161a930da`).

### 2026-07-06 rebase notes (388268ec → 2ebf9a90)

All carried patches survived; none were absorbed or dropped. Conflicted and
notable resolutions:

- **Stateful compression quality gate** — conflicted in
  `agent/context_compressor.py` and `agent/conversation_compression.py`.
  Upstream stopped eagerly resetting `_last_summary_auth_failure` /
  `_last_summary_network_failure` at the top of `compress()` (data-loss fix
  #29559); the patch now follows that semantics and only resets its own
  quality-gate/diagnostic fields. The `context_compression_aborted` /
  `context_compression_completed` hook emissions were re-grafted into
  upstream's new try/finally compression-lock structure (lock lease refresher,
  in-place compaction #38763); the completed hook now also carries `in_place`.
  `quality_gate_enabled` is read via `getattr(..., False)` because upstream
  tests construct `ContextCompressor` via `__new__`, bypassing `__init__`.
- **MCP Hermes context keying** — additive collisions with upstream's
  `skip_preflight` doc row and new `TestFilterMCPChildren` tests; both sides
  kept.
- **Large output handoff** — trivial import-block collision in
  `tests/tools/test_code_execution.py`; both sides kept.
- Hook-surface audit: upstream added a `pre_verify` hook
  (`hermes_cli/plugins.py`) and now fires `on_session_end` memory extraction
  before soft-evicting finalizable sessions under LRU cache pressure
  (`gateway/run.py`). Both are additive; no fork hook call sites changed.
- Upstream consolidated gateway session metadata and the routing index into
  `state.db` (schema v19, `gateway_routing` table, `AsyncSessionDB` facade);
  `sessions.json` is now an optional legacy mirror. No carried patch touches
  that surface, but D6b (Postgres session store) builds on it.

## Patch Inventory

| Patch | Status | Purpose | Main files | Conflict surface | Owner |
|---|---|---|---|---|---|
| Request metadata and tool-dispatch observability | Promoted in fork history | Preserve request/session/turn metadata through hooks and tool execution so gateway, traces, and downstream plugins can correlate turns reliably. | `run_agent.py`, `model_tools.py`, `hermes_cli/hooks.py`, `hermes_cli/plugins.py`, `tools/delegate_tool.py` | Medium: agent loop and hook dispatch | IdeaRoom Agentic Systems |
| Runtime footer preservation | Promoted in fork history | Preserve runtime footer usage fields across gateway streaming and response handling. | `gateway/runtime_footer.py`, `gateway/run.py`, `gateway/platforms/api_server.py` | Medium: gateway stream/result formatting | IdeaRoom Agentic Systems |
| Hindsight retained-chat tagging and document metadata | Promoted in fork history | Tag retained web chats with useful provenance and preserve document-level Hindsight metadata for recall/auditability, including chat/session/user/environment/profile fields when traffic comes through Hermes Web. | `agent/agent_init.py`, `agent/turn_context.py`, `plugins/memory/hindsight/__init__.py` | Low: Hindsight plugin provider surface and request metadata handoff | IdeaRoom Agentic Systems |
| Stateful compression quality gate | Promoted in fork history | Improve context-compression safety by preserving ledger/state details and validating compression quality. | `agent/context_compressor.py`, `agent/conversation_compression.py`, `agent/conversation_loop.py`, `agent/turn_context.py`, `agent/turn_finalizer.py` | High: agent context and compression flow | IdeaRoom Agentic Systems |
| Responses API Postgres store (D6a) | Promoted in fork history | Move the gateway Responses API store off EFS SQLite and into Postgres for concurrent web traffic; Postgres is durable state, does not use the old SQLite LRU cap, and can recover the required schema on fresh databases. | `gateway/platforms/api_server.py`, `gateway/platforms/response_store_pg.py` | Medium: gateway response-store boundary | IdeaRoom Agentic Systems |
| Browser raw-CDP supervisor disable | Promoted previously; carried in fork history | Let AWS Browserless raw-CDP stages disable the second persistent CDP supervisor client, avoiding an idle extra Chrome per browser-using chat. | `tools/browser_tool.py`, `tests/tools/test_browser_cdp_override.py` | Low: browser CDP setup path | IdeaRoom Agentic Systems |
| MCP Hermes context keying | Promoted previously; carried in fork history | Let trusted MCP servers opt in to `_meta.hermes.task_id` on `tools/call` so sandbox orchestrators can key one container per chat without trusting model arguments. | `tools/mcp_tool.py`, `tests/tools/test_mcp_tool.py`, `website/docs/reference/mcp-config-reference.md`, `website/docs/user-guide/features/mcp.md` | Low: opt-in MCP call metadata | IdeaRoom Agentic Systems |
| execute_code per-turn budget | Promoted previously; carried in fork history | Cap aggregate `execute_code` wall-clock time per assistant turn and pass `turn_id` into the tool handler, reducing runaway multi-call CPU exposure without adding a concurrency cap. | `model_tools.py`, `tools/code_execution_tool.py`, `tests/tools/test_code_execution.py`, `website/docs/user-guide/features/code-execution.md` | Medium: tool dispatch and execution timeout path | IdeaRoom Agentic Systems |
| Large output handoff | Carried in fork history | Replace oversized terminal and `execute_code` stdout truncation with sanitized file handoff references so JSON/CSV and other parseable output is not corrupted. | `tools/large_output_handoff.py`, `tools/code_execution_tool.py`, `tests/tools/test_large_output_handoff.py`, `tests/tools/test_code_execution.py`, `tests/tools/test_terminal_output_transform_hook.py` | Medium: tool output serialization and redaction | IdeaRoom Agentic Systems |
| Runtime source revision health field | Promoted previously; carried in fork history | Expose the loaded Hermes source commit through gateway health/status so promotion audits can compare running process code to the tested fork commit. | `gateway/status.py`, `gateway/platforms/api_server.py`, `tests/gateway/test_status.py`, `tests/gateway/test_api_server.py` | Low: health/status payloads | IdeaRoom Agentic Systems |
| Gateway readiness endpoint (AE-77) | Promoted in fork history | Expose a deploy/readiness check that reports API server state, runtime gateway state, active agents, runtime PID, and gateway owner for local, staging, production, and PR-environment validation. | `gateway/platforms/api_server.py`, `tests/gateway/test_api_server.py` | Low: additive health/readiness endpoint | IdeaRoom Agentic Systems |
| Responses API usage transparency (AE-24) | Promoted in fork history | Preserve context-window usage, compaction count, cost estimate, and structured failure type in `/v1/responses` terminal envelopes for Hermes Web observability. | `gateway/platforms/api_server.py`, `tests/gateway/test_api_server.py` | Low: additive Responses API envelope fields | IdeaRoom Agentic Systems |
| Skill review gate removal (AE-81) | Promoted in fork history | Keep AWS runtime-created skills in the current free-write posture by removing the pending skill review API surface and tests that would reintroduce the old approval queue. | `gateway/platforms/api_server.py`, `tests/gateway/test_api_server.py` | Medium: API surface and product workflow posture | IdeaRoom Agentic Systems |
| API server memory cleanup (AE-47) | Carried in fork history | Ensure API-server and runs flows clean up memory providers when runs end and skip next-turn memory prefetch when a client sends a follow-up before the previous turn finishes. | `gateway/platforms/api_server.py`, `run_agent.py`, `tests/gateway/test_api_server.py`, `tests/gateway/test_api_server_runs.py`, `tests/run_agent/test_memory_sync_interrupted.py` | Medium: gateway run lifecycle and memory provider cleanup | IdeaRoom Agentic Systems |
| Local Codex auth adoption | Carried in fork history | Recover local parity on fresh profile volumes by adopting a valid mounted Codex CLI auth snapshot only when `CODEX_HOME` is explicit and Hermes Codex auth state is absent. | `hermes_cli/auth.py`, `tests/hermes_cli/test_auth_codex_self_heal.py` | Low: Codex OAuth credential resolution | IdeaRoom Agentic Systems |
| Postgres session store (D6b, AE-115) | Carried in fork history | Move the shared session store (`state.db`) off EFS SQLite into a dedicated `hermes_state` Postgres schema so blue/green gateway deploys can run two tasks against shared state (ADR 0177 in idearoom-agents). `SessionDB.__new__` dispatches to `PgSessionDB` when `HERMES_STATE_STORE_DSN` is set and no explicit `db_path` is given; the SQLite path is unchanged otherwise. `PgSessionDB` inherits SessionDB method bodies over a SQLite→Postgres statement translator (tsvector/ILIKE replace FTS5/trigram search) and pins the upstream `SCHEMA_VERSION` + DDL surface hash, refusing to boot on rebase drift. Includes a one-time `state.db` → Postgres migration script. | `hermes_state.py` (the `__new__` seam only), `hermes_state_pg.py`, `scripts/migrate_state_to_postgres.py`, `tests/gateway/test_session_store_pg.py`, `tests/gateway/test_session_store_pg_unit.py`, `tests/conftest.py` | Medium: the only upstream-file edit is `SessionDB.__new__`, but any upstream change to `SCHEMA_SQL` / `DEFERRED_INDEX_SQL` or to SessionDB SQL dialect requires re-auditing `hermes_state_pg.py` and updating its pinned `EXPECTED_SCHEMA_VERSION` / `EXPECTED_SCHEMA_SURFACE_SHA256` (the boot guard enforces this loudly). | IdeaRoom Agentic Systems |

## Maintenance Rules

- Prefer additive modules and narrow hook-point edits over broad upstream rewrites.
- Keep model-visible tool schemas stable unless the product behavior truly changes.
- Add or update focused tests before implementation changes.
- After rebasing, run the focused tests named by each affected patch plus the
  parent repo promotion gate before deploying the fork runtime.
- When a source-only patch is committed and promoted, update its status here and
  the parent repo's `packages/hermes-config/hermes-fork-state.json`.
