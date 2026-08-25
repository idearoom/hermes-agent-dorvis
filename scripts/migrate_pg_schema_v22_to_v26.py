#!/usr/bin/env python3
"""Explicit, drain-only Postgres session-store migration from v22 to v26.

This command is intentionally separate from runtime boot. Before ``--apply``:
drain every Hermes gateway task to zero and take a schema-scoped backup of
``hermes_state``. Old v22 and new v26 images must never share this database.
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
    EXPECTED_SCHEMA_SURFACE_SHA256,
    EXPECTED_SCHEMA_VERSION,
    PgSessionDB,
    inspect_v22_migration_precondition,
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

    if not args.apply:
        try:
            source = inspect_v22_migration_precondition(dsn)
        except Exception as exc:
            print(f"Preflight failed: {exc}", file=sys.stderr)
            return 1
        print(
            "Dry run passed without database writes: "
            f"backend={source['backend']} "
            f"schema_version={source['schema_version']} "
            f"surface_marker={source['surface_marker']}"
        )
        print(
            f"Target: schema v{EXPECTED_SCHEMA_VERSION}, "
            f"surface {EXPECTED_SCHEMA_SURFACE_SHA256}"
        )
        print(
            "Drain all Hermes tasks to zero and back up the hermes_state "
            "schema before re-running with --apply."
        )
        return 0

    store = PgSessionDB(dsn=dsn, allow_schema_migration=True)
    try:
        attestation = store.storage_attestation()
    finally:
        store.close()
    print(
        "Migration complete: "
        f"backend={attestation['backend']} "
        f"schema_version={attestation['schema_version']} "
        f"surface_marker={attestation['surface_marker']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
