from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import mysql.connector
from dotenv import load_dotenv

THIS_FILE = Path(__file__).resolve()
_API_DIR = THIS_FILE.parent
_ROOT_DIR = _API_DIR.parent
load_dotenv(_ROOT_DIR / ".env")

MYSQL_CONFIG = {
    "host": os.getenv("MYSQL_HOST", "127.0.0.1"),
    "port": int(os.getenv("MYSQL_PORT", "3306")),
    "user": os.getenv("MYSQL_USER", "root"),
    "password": os.getenv("MYSQL_PASSWORD", ""),
    "database": os.getenv("MYSQL_DB", "rethink_ai_boston"),
}

INACTIVE_HOURS = int(os.getenv("GUEST_CLEANUP_INACTIVE_HOURS", "24"))
BATCH_SIZE = int(os.getenv("GUEST_CLEANUP_BATCH_SIZE", "500"))
MAX_BATCHES = int(os.getenv("GUEST_CLEANUP_MAX_BATCHES", "20"))


def _cleanup_enabled() -> bool:
    return os.getenv("GUEST_CLEANUP_ENABLED", "false").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def get_db_connection():
    return mysql.connector.connect(**MYSQL_CONFIG)


def find_stale_guest_user_ids(cursor, *, limit: int) -> list[str]:
    cursor.execute(
        """
        SELECT u.id
        FROM users u
        WHERE u.is_guest = TRUE
          AND NOT EXISTS (
            SELECT 1
            FROM auth_identities ai
            WHERE ai.user_id = u.id
          )
          AND NOT EXISTS (
            SELECT 1
            FROM web_sessions ws
            WHERE ws.user_id = u.id
              AND ws.revoked_at IS NULL
              AND ws.expires_at > UTC_TIMESTAMP()
              AND GREATEST(
                    COALESCE(ws.last_seen_at, ws.created_at),
                    ws.created_at
                  ) > DATE_SUB(UTC_TIMESTAMP(), INTERVAL %s HOUR)
          )
        ORDER BY u.created_at ASC
        LIMIT %s
        """,
        (INACTIVE_HOURS, limit),
    )
    return [row[0] for row in cursor.fetchall()]


def count_stale_guest_users(cursor) -> int:
    cursor.execute(
        """
        SELECT COUNT(*)
        FROM users u
        WHERE u.is_guest = TRUE
          AND NOT EXISTS (
            SELECT 1
            FROM auth_identities ai
            WHERE ai.user_id = u.id
          )
          AND NOT EXISTS (
            SELECT 1
            FROM web_sessions ws
            WHERE ws.user_id = u.id
              AND ws.revoked_at IS NULL
              AND ws.expires_at > UTC_TIMESTAMP()
              AND GREATEST(
                    COALESCE(ws.last_seen_at, ws.created_at),
                    ws.created_at
                  ) > DATE_SUB(UTC_TIMESTAMP(), INTERVAL %s HOUR)
          )
        """,
        (INACTIVE_HOURS,),
    )
    row = cursor.fetchone()
    return int(row[0] if row else 0)


def delete_guest_users(cursor, user_ids: list[str]) -> tuple[int, int]:
    if not user_ids:
        return 0, 0

    placeholders = ", ".join(["%s"] * len(user_ids))
    cursor.execute(
        f"DELETE FROM interaction_log WHERE user_id IN ({placeholders})",
        tuple(user_ids),
    )
    interaction_rows = cursor.rowcount

    cursor.execute(
        f"DELETE FROM users WHERE id IN ({placeholders}) AND is_guest = TRUE",
        tuple(user_ids),
    )
    user_rows = cursor.rowcount
    return interaction_rows, user_rows


def run_cleanup(*, dry_run: bool) -> int:
    conn = None
    cursor = None
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        stale_total = count_stale_guest_users(cursor)

        print("Guest user cleanup")
        print("-" * 32)
        print(f"Inactive threshold:   {INACTIVE_HOURS} hour(s)")
        print(f"Batch size:           {BATCH_SIZE}")
        print(f"Stale guests matched: {stale_total}")

        if dry_run or stale_total == 0:
            if dry_run:
                print("\nDry run only. No data was deleted.")
            return 0

        deleted_users = 0
        deleted_interactions = 0
        batches = 0

        while batches < MAX_BATCHES:
            user_ids = find_stale_guest_user_ids(cursor, limit=BATCH_SIZE)
            if not user_ids:
                break

            interaction_rows, user_rows = delete_guest_users(cursor, user_ids)
            conn.commit()
            deleted_users += user_rows
            deleted_interactions += interaction_rows
            batches += 1

            if len(user_ids) < BATCH_SIZE:
                break

        print(f"\nDeleted guest users:        {deleted_users}")
        print(f"Deleted interaction rows:   {deleted_interactions}")
        return deleted_users
    except mysql.connector.Error as exc:
        if conn:
            conn.rollback()
        raise SystemExit(
            "Could not connect to MySQL using the current .env settings. "
            f"Host={MYSQL_CONFIG['host']} Port={MYSQL_CONFIG['port']} "
            f"Database={MYSQL_CONFIG['database']}. Original error: {exc}"
        ) from exc
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Delete orphaned guest users with no recently active sessions "
            "and purge related interaction_log rows."
        )
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report how many stale guest users would be removed without deleting anything.",
    )
    args = parser.parse_args()

    if not args.dry_run and not _cleanup_enabled():
        print("Guest cleanup disabled (set GUEST_CLEANUP_ENABLED=true to run deletions).")
        return

    run_cleanup(dry_run=args.dry_run)


if __name__ == "__main__":
    main()
