from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any

import mysql.connector
from dotenv import load_dotenv

THIS_FILE = Path(__file__).resolve()
_API_DIR = THIS_FILE.parent
_ROOT_DIR = _API_DIR.parent
load_dotenv(_ROOT_DIR / ".env")

_ON_THE_PORCH_DIR = _ROOT_DIR / "on_the_porch"
if str(_ON_THE_PORCH_DIR) not in sys.path:
    sys.path.insert(0, str(_ON_THE_PORCH_DIR))

_RAG_DIR = _ON_THE_PORCH_DIR / "rag stuff"
if str(_RAG_DIR) not in sys.path:
    sys.path.insert(0, str(_RAG_DIR))

import chromadb  # noqa: E402
import retrieval  # noqa: E402

retrieval.VECTORDB_DIR = (_ON_THE_PORCH_DIR / "vectordb_new").resolve()


MYSQL_CONFIG = {
    "host": os.getenv("MYSQL_HOST", "127.0.0.1"),
    "port": int(os.getenv("MYSQL_PORT", "3306")),
    "user": os.getenv("MYSQL_USER", "root"),
    "password": os.getenv("MYSQL_PASSWORD", ""),
    "database": os.getenv("MYSQL_DB", "rethink_ai_boston"),
}
print(MYSQL_CONFIG)
BATCH_SIZE = 500


def get_db_connection():
    return mysql.connector.connect(**MYSQL_CONFIG)


def count_admin_knowledge_rows() -> int:
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        cursor.execute("SELECT COUNT(*) FROM admin_knowledge")
        row = cursor.fetchone()
        return int(row[0] if row else 0)
    finally:
        cursor.close()
        conn.close()


def clear_admin_knowledge_table(dry_run: bool = False) -> int:
    row_count = count_admin_knowledge_rows()
    if dry_run or row_count == 0:
        return row_count

    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        cursor.execute("TRUNCATE TABLE admin_knowledge")
        conn.commit()
        return row_count
    except Exception:
        conn.rollback()
        raise
    finally:
        cursor.close()
        conn.close()


def _matches_admin_knowledge_doc(metadata: dict[str, Any] | None) -> bool:
    if not metadata:
        return False
    return (
        metadata.get("doc_type") == "community_note"
        or metadata.get("source") == "admin_knowledge"
    )


def get_admin_knowledge_doc_ids() -> list[str]:
    client = chromadb.PersistentClient(path=str(retrieval.VECTORDB_DIR))
    collection = client.get_collection("langchain")
    total = collection.count()
    if total <= 0:
        return []

    matching_ids: list[str] = []
    offset = 0
    while offset < total:
        batch = collection.get(
            limit=BATCH_SIZE,
            offset=offset,
            include=["metadatas"],
        )
        ids = batch.get("ids") or []
        metadatas = batch.get("metadatas") or []
        for doc_id, metadata in zip(ids, metadatas):
            if _matches_admin_knowledge_doc(metadata):
                matching_ids.append(doc_id)
        offset += BATCH_SIZE

    return matching_ids


def clear_admin_knowledge_from_chroma(dry_run: bool = False) -> int:
    doc_ids = get_admin_knowledge_doc_ids()
    if dry_run or not doc_ids:
        return len(doc_ids)

    client = chromadb.PersistentClient(path=str(retrieval.VECTORDB_DIR))
    collection = client.get_collection("langchain")

    for start in range(0, len(doc_ids), BATCH_SIZE):
        batch_ids = doc_ids[start:start + BATCH_SIZE]
        collection.delete(ids=batch_ids)

    return len(doc_ids)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Clear the admin_knowledge MySQL table and remove its documents from Chroma."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show how many rows/documents would be removed without deleting anything.",
    )
    args = parser.parse_args()

    try:
        table_rows = count_admin_knowledge_rows()
    except mysql.connector.Error as exc:
        raise SystemExit(
            "Could not connect to MySQL using the current .env settings. "
            f"Host={MYSQL_CONFIG['host']} Port={MYSQL_CONFIG['port']} Database={MYSQL_CONFIG['database']}. "
            f"Original error: {exc}"
        ) from exc

    chroma_docs = get_admin_knowledge_doc_ids()

    print("Admin knowledge cleanup")
    print("-" * 32)
    print(f"MySQL rows matched:   {table_rows}")
    print(f"Chroma docs matched:  {len(chroma_docs)}")

    if args.dry_run:
        print("\nDry run only. No data was deleted.")
        return

    removed_rows = clear_admin_knowledge_table(dry_run=False)
    removed_docs = clear_admin_knowledge_from_chroma(dry_run=False)

    print("\nCleanup complete")
    print("-" * 32)
    print(f"MySQL rows removed:   {removed_rows}")
    print(f"Chroma docs removed:  {removed_docs}")


if __name__ == "__main__":
    main()