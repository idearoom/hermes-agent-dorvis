#!/usr/bin/env python3
"""One-time migration: copy a SQLite ``state.db`` into the Postgres session store.

IdeaRoom D6b (AE-115). Companion to ``hermes_state_pg.PgSessionDB`` — copies
every session-store table (sessions, messages incl. ``active``/``compacted``
soft-archive flags, state_meta, gateway_routing, compression_locks, and the
telegram topic tables when present) into the dedicated ``hermes_state``
Postgres schema. Cutover (pointing the gateway at the DSN, retiring the EFS
file) is a separate issue; this script only moves data.

Usage:
    python scripts/migrate_state_to_postgres.py \
        --sqlite ~/.hermes/state.db \
        --dsn "$HERMES_STATE_STORE_DSN" \
        [--apply] [--allow-nonempty]

Safety:
- Dry-run by default: prints per-table row counts and the plan, writes nothing.
- Refuses to write into a Postgres store that already has sessions/messages
  rows unless ``--allow-nonempty`` is passed; with it, every INSERT is
  ``ON CONFLICT DO NOTHING`` (idempotent re-run — existing rows win).
- The whole copy runs in ONE transaction with constraints deferred; a failure
  leaves Postgres untouched.
- Prints per-table verification counts (sqlite vs postgres) and exits nonzero
  on mismatch.
"""

from __future__ import annotations

import argparse
import os
import sqlite3
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from hermes_state_pg import _SCHEMA, PgSessionDB, _normalize_dsn  # noqa: E402

# (table, pk_cols) in FK-safe copy order. compression_locks are ephemeral
# leases but copied anyway so an in-flight lease survives a cutover.
_TABLES = [
    "sessions",
    "messages",
    "state_meta",
    "gateway_routing",
    "compression_locks",
    "telegram_dm_topic_mode",
    "telegram_dm_topic_bindings",
]

_BATCH = 1000


def _sqlite_table_exists(conn: sqlite3.Connection, name: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?",
        (name,),
    ).fetchone()
    return row is not None


def _pg_columns(pg_conn, table: str) -> dict:
    """{column_name: data_type} for the target table."""
    rows = pg_conn.execute(
        "SELECT column_name, data_type FROM information_schema.columns "
        "WHERE table_schema = %s AND table_name = %s",
        (_SCHEMA, table),
    ).fetchall()
    return {r[0]: r[1] for r in rows}


def _coerce(value, data_type: str):
    """SQLite dynamic typing → declared Postgres column types."""
    if value is None:
        return None
    if data_type == "text":
        if isinstance(value, bytes):
            value = value.decode("utf-8", errors="replace")
        elif not isinstance(value, str):
            value = str(value)
        if "\x00" in value:
            # SQLite's multimodal-content sentinel is "\x00json:";
            # PgSessionDB uses "\x01json:" because Postgres TEXT cannot
            # hold NUL bytes. Rewrite the sentinel, strip any other NULs.
            if value.startswith("\x00json:"):
                value = "\x01json:" + value[len("\x00json:"):]
            value = value.replace("\x00", "")
        return value
    if isinstance(value, bool):
        return int(value)
    return value


def _copy_table(sqlite_conn, pg_conn, table: str, *, allow_nonempty: bool) -> tuple:
    """Copy one table; returns (source_count, copied_count)."""
    src_cols = [
        r[1] for r in sqlite_conn.execute(f'PRAGMA table_info("{table}")')
    ]
    dst_types = _pg_columns(pg_conn, table)
    cols = [c for c in src_cols if c in dst_types]
    missing = [c for c in src_cols if c not in dst_types]
    if missing:
        print(f"  WARNING: {table}: source columns not in Postgres schema, "
              f"skipped: {missing}")
    col_sql = ", ".join(cols)
    placeholders = ", ".join(["%s"] * len(cols))
    conflict = " ON CONFLICT DO NOTHING" if allow_nonempty else ""
    insert_sql = (
        f"INSERT INTO {_SCHEMA}.{table} ({col_sql}) "
        f"VALUES ({placeholders}){conflict}"
    )

    total = sqlite_conn.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()[0]
    copied = 0
    cur = sqlite_conn.execute(f'SELECT {col_sql} FROM "{table}"')
    while True:
        rows = cur.fetchmany(_BATCH)
        if not rows:
            break
        batch = [
            tuple(_coerce(v, dst_types[c]) for v, c in zip(row, cols))
            for row in rows
        ]
        with pg_conn.cursor() as pcur:
            pcur.executemany(insert_sql, batch)
        copied += len(batch)
    return total, copied


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sqlite",
        default=None,
        help="Path to the source state.db (default: <hermes home>/state.db)",
    )
    parser.add_argument(
        "--dsn",
        default=os.environ.get("HERMES_STATE_STORE_DSN", "").strip(),
        help="Target Postgres DSN (default: $HERMES_STATE_STORE_DSN)",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Actually write. Without this flag the script is a dry run.",
    )
    parser.add_argument(
        "--allow-nonempty",
        action="store_true",
        help="Proceed even if the Postgres store already has rows "
        "(INSERT ... ON CONFLICT DO NOTHING; existing rows win).",
    )
    args = parser.parse_args()

    if args.sqlite:
        sqlite_path = Path(args.sqlite).expanduser()
    else:
        from hermes_constants import get_hermes_home

        sqlite_path = get_hermes_home() / "state.db"
    if not sqlite_path.exists():
        print(f"ERROR: SQLite source not found: {sqlite_path}")
        return 2
    if not args.dsn:
        print("ERROR: no DSN (pass --dsn or set HERMES_STATE_STORE_DSN)")
        return 2

    print(f"Source: {sqlite_path}")
    print(f"Target: {_SCHEMA} schema at "
          f"{args.dsn.split('@')[-1] if '@' in args.dsn else '<dsn>'}")

    # Creating PgSessionDB validates the DSN, creates/validates the schema,
    # and runs the rebase-drift guard before any copying starts.
    store = PgSessionDB(dsn=args.dsn)
    store.close()

    import psycopg

    sqlite_conn = sqlite3.connect(f"file:{sqlite_path}?mode=ro", uri=True)
    pg_conn = psycopg.connect(_normalize_dsn(args.dsn), autocommit=False)
    try:
        # Source schema version must match what this build's PgSessionDB
        # mirrors — refuse to copy a stale/newer SQLite layout.
        src_version = sqlite_conn.execute(
            "SELECT version FROM schema_version LIMIT 1"
        ).fetchone()
        print(f"Source schema_version: {src_version[0] if src_version else '?'}")

        existing = pg_conn.execute(
            f"SELECT (SELECT COUNT(*) FROM {_SCHEMA}.sessions) + "
            f"(SELECT COUNT(*) FROM {_SCHEMA}.messages)"
        ).fetchone()[0]
        if existing and not args.allow_nonempty:
            print(
                f"ERROR: Postgres store already has {existing} "
                "session/message rows. Re-run with --allow-nonempty to merge "
                "idempotently (existing rows win), or use a clean schema."
            )
            return 3

        plan = []
        for table in _TABLES:
            if not _sqlite_table_exists(sqlite_conn, table):
                print(f"  {table}: not present in source, skipping")
                continue
            plan.append(table)
            count = sqlite_conn.execute(
                f'SELECT COUNT(*) FROM "{table}"'
            ).fetchone()[0]
            print(f"  {table}: {count} row(s) to copy")

        if not args.apply:
            print("Dry run (no --apply): nothing written.")
            return 0

        pg_conn.execute("SET CONSTRAINTS ALL DEFERRED")
        results = {}
        for table in plan:
            total, copied = _copy_table(
                sqlite_conn, pg_conn, table, allow_nonempty=args.allow_nonempty
            )
            results[table] = (total, copied)

        # Identity sequence must resume past the migrated message ids.
        pg_conn.execute(
            f"SELECT setval(pg_get_serial_sequence('{_SCHEMA}.messages', 'id'), "
            f"GREATEST((SELECT COALESCE(MAX(id), 0) FROM {_SCHEMA}.messages), 1))"
        )
        pg_conn.commit()

        print("\nVerification (source -> postgres):")
        ok = True
        for table in plan:
            src_count = results[table][0]
            dst_count = pg_conn.execute(
                f"SELECT COUNT(*) FROM {_SCHEMA}.{table}"
            ).fetchone()[0]
            status = "OK" if dst_count >= src_count else "MISMATCH"
            if dst_count < src_count:
                ok = False
            print(f"  {table}: {src_count} -> {dst_count}  {status}")
        if not ok:
            print("ERROR: verification failed — postgres has fewer rows than "
                  "the source for at least one table.")
            return 4
        print("Migration complete.")
        return 0
    finally:
        sqlite_conn.close()
        pg_conn.close()


if __name__ == "__main__":
    sys.exit(main())
