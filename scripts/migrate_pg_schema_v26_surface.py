#!/usr/bin/env python3
"""Expand the Dorvis Postgres v26 surface for Hermes v2026.8.27.

The default is a read-only preflight. ``--apply`` adds only the persisted
compressed-summary marker column and advances the exact surface marker. The
v2026.8.19 compatibility bridge and v2026.8.27 runtime can both serve the
expanded catalog, so this migration does not require a zero-writer window.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from hermes_state_pg import (  # noqa: E402
    ENV_VAR,
    FORWARD_SCHEMA_SURFACE_SHA256,
    inspect_v26_surface_migration_precondition,
    migrate_v26_surface,
)


def _safe_failure_detail(exc: Exception) -> str:
    """Describe a failure without rendering untrusted driver error text."""
    name = type(exc).__name__
    sqlstate = getattr(exc, "sqlstate", None)
    state = (
        f", SQLSTATE {sqlstate}"
        if isinstance(sqlstate, str)
        and len(sqlstate) == 5
        and sqlstate.isalnum()
        else ""
    )
    return (
        f"database operation failed ({name}{state}); "
        "connection details suppressed"
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dsn",
        help=f"Postgres DSN (defaults to ${ENV_VAR})",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="perform the transactional migration (default: print plan only)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    dsn = (args.dsn or os.environ.get(ENV_VAR, "")).strip()
    if not dsn:
        print(f"error: --dsn or ${ENV_VAR} is required", file=sys.stderr)
        return 2

    try:
        evidence = (
            migrate_v26_surface(dsn)
            if args.apply
            else inspect_v26_surface_migration_precondition(dsn)
        )
    except Exception as exc:
        phase = "apply" if args.apply else "preflight"
        print(
            f"Migration {phase} failed: {_safe_failure_detail(exc)}",
            file=sys.stderr,
        )
        return 1

    if args.apply:
        action = (
            "applied" if evidence.get("migration_applied") else "already current"
        )
        print(
            f"Migration {action}: backend={evidence['backend']} "
            f"schema_version={evidence['schema_version']} "
            f"surface_marker={evidence['surface_marker']}"
        )
    else:
        print(
            "Dry run passed without database writes: "
            f"backend={evidence['backend']} "
            f"schema_version={evidence['schema_version']} "
            f"surface_marker={evidence['surface_marker']} "
            f"migration_required={str(evidence['migration_required']).lower()}"
        )
        print(f"Target surface: {FORWARD_SCHEMA_SURFACE_SHA256}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
