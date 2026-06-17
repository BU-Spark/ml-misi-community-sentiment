import os
import re
import sqlite3
import threading
import time
from pathlib import Path

import chromadb
import google.generativeai as genai  # type: ignore

try:
    from langchain_chroma import Chroma
except ImportError:
    from langchain_community.vectorstores import Chroma

# Resolve to an absolute path anchored at this file (on_the_porch/vectordb_new)
# so the location does not depend on the process working directory. The live
# server still overrides this via _fix_retrieval_vectordb_path(), but resolving
# here keeps standalone scripts pointed at the same store the server reads.
VECTORDB_DIR = (Path(__file__).resolve().parent / "../vectordb_new").resolve()
GEMINI_EMBED_MODEL = os.getenv("GEMINI_EMBED_MODEL", "models/gemini-embedding-001")
# How many times to retry a vector-store read when SQLite is momentarily locked
# (e.g. the nightly ingestion cron is writing to the same store).
_CHROMA_MAX_ATTEMPTS = max(1, int(os.getenv("CHROMA_MAX_ATTEMPTS", "3")))
# Tracks which vector-store paths have already had WAL mode enabled.
_WAL_ENABLED_PATHS: set[str] = set()

# Reuse a single Chroma client + embeddings object instead of rebuilding them on
# every retrieval call (the old behavior rebuilt them 4-6x per question). The
# client is safe to share across threads for reads. Disable with
# VECTORDB_SINGLETON=false to fall back to per-call construction.
_VECTORDB_SINGLETON_ENABLED = os.getenv("VECTORDB_SINGLETON", "true").strip().lower() in ("1", "true", "yes")
_vectordb_singleton = None
_vectordb_singleton_path: str | None = None
_embeddings_singleton = None
_vectordb_lock = threading.Lock()
_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "how",
    "i", "in", "is", "it", "me", "of", "on", "or", "that", "the", "this",
    "to", "was", "what", "when", "where", "which", "who", "why", "with",
}


def _configure_gemini() -> None:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY not configured")
    genai.configure(api_key=api_key)


class GeminiEmbeddings:
    """
    Minimal embeddings wrapper using Gemini's embedding API, compatible with LangChain's interface.
    """

    def __init__(self, model: str | None = None) -> None:
        _configure_gemini()
        self.model = model or GEMINI_EMBED_MODEL

    def _embed(self, text: str):
        res = genai.embed_content(model=self.model, content=text)
        return res["embedding"]

    def embed_documents(self, texts):
        return [self._embed(t) for t in texts]

    def embed_query(self, text):
        return self._embed(text)


def _is_locked_error(exc: Exception) -> bool:
    """True when an exception looks like a transient SQLite lock/contention."""
    message = str(exc).lower()
    return (
        "database is locked" in message
        or "database is busy" in message
        or "disk i/o error" in message
    )


def _ensure_sqlite_wal() -> None:
    """Best-effort: put the Chroma SQLite store into WAL mode.

    WAL (Write-Ahead Logging) lets the live server keep *reading* while the
    nightly ingestion cron *writes* to the same store, instead of failing with
    "database is locked". journal_mode=WAL is persisted in the database header,
    so this only needs to succeed once; we track it per path and retry until it
    does. This is intentionally non-fatal — if it can't be set we simply rely on
    the read-retry logic below.
    """
    try:
        path_str = str(Path(VECTORDB_DIR).resolve())
    except Exception:
        return
    if path_str in _WAL_ENABLED_PATHS:
        return
    db_path = Path(VECTORDB_DIR) / "chroma.sqlite3"
    if not db_path.exists():
        return
    try:
        conn = sqlite3.connect(str(db_path), timeout=30)
        try:
            conn.execute("PRAGMA journal_mode=WAL;")
            conn.execute("PRAGMA busy_timeout=30000;")
        finally:
            conn.close()
        _WAL_ENABLED_PATHS.add(path_str)
    except Exception as exc:
        print(f"  ⚠️ Could not enable SQLite WAL on vector store: {exc}")


def load_vectordb():
    """Load the unified vector database (policies, transcripts, client uploads, etc.).

    Returns a cached singleton keyed by the current VECTORDB_DIR. If the path
    changes (the live server overrides it at startup), the client is rebuilt.
    """
    _ensure_sqlite_wal()

    if not _VECTORDB_SINGLETON_ENABLED:
        return Chroma(persist_directory=str(VECTORDB_DIR), embedding_function=GeminiEmbeddings())

    global _vectordb_singleton, _vectordb_singleton_path, _embeddings_singleton
    current_path = str(VECTORDB_DIR)
    with _vectordb_lock:
        if _vectordb_singleton is None or _vectordb_singleton_path != current_path:
            if _embeddings_singleton is None:
                _embeddings_singleton = GeminiEmbeddings()
            _vectordb_singleton = Chroma(
                persist_directory=current_path,
                embedding_function=_embeddings_singleton,
            )
            _vectordb_singleton_path = current_path
        return _vectordb_singleton


def _normalize_query_terms(text: str) -> list[str]:
    terms = re.findall(r"[a-z0-9]{3,}", (text or "").lower())
    return [term for term in terms if term not in _STOPWORDS]


def _matches_doc_type(metadata, doc_type) -> bool:
    if not doc_type:
        return True
    current = metadata.get("doc_type")
    if isinstance(doc_type, (list, tuple)):
        allowed = {value for value in doc_type if value}
        return current in allowed if allowed else True
    return current == doc_type


def _matches_source(metadata, source) -> bool:
    if not source:
        return True
    return metadata.get("source") == source


def _matches_tags(metadata, tags) -> bool:
    if not tags:
        return True
    raw_tags = metadata.get("tags")
    if not raw_tags:
        return False
    doc_tags = {tag.strip().lower() for tag in raw_tags.split(",") if tag.strip()}
    requested = {tag.strip().lower() for tag in tags if tag and tag.strip()}
    return bool(doc_tags & requested) if requested else True


def _score_keyword_match(query: str, document: str, metadata) -> float:
    query_lower = (query or "").lower().strip()
    if not query_lower:
        return 0.0

    content_lower = (document or "").lower()
    source_lower = str(metadata.get("source", "")).lower()
    folder_lower = str(metadata.get("folder_category", "")).lower()
    terms = _normalize_query_terms(query_lower)

    score = 0.0

    if query_lower in content_lower:
        score += 10.0
    if query_lower in source_lower:
        score += 8.0

    for term in terms:
        score += min(content_lower.count(term), 5)
        if term in source_lower:
            score += 2.0
        if term in folder_lower:
            score += 1.0

    return score


def _load_raw_documents():
    client = chromadb.PersistentClient(path=str(VECTORDB_DIR))
    collection = client.get_collection("langchain")
    count = collection.count()
    if count <= 0:
        return []
    rows = collection.get(limit=count, include=["documents", "metadatas"])
    documents = rows.get("documents") or []
    metadatas = rows.get("metadatas") or []
    return list(zip(documents, metadatas))


def _keyword_retrieve(query, k=5, doc_type=None, tags=None, source=None):
    rows = []
    tag_matched_rows = []
    for document, metadata in _load_raw_documents():
        metadata = metadata or {}
        if not _matches_doc_type(metadata, doc_type):
            continue
        if not _matches_source(metadata, source):
            continue
        score = _score_keyword_match(query, document, metadata)
        if score <= 0:
            continue
        row = (score, document, metadata)
        rows.append(row)
        if _matches_tags(metadata, tags):
            tag_matched_rows.append(row)

    ranked_rows = tag_matched_rows if tag_matched_rows else rows
    ranked_rows.sort(key=lambda item: item[0], reverse=True)
    top_rows = ranked_rows[:k]
    return {
        "chunks": [document for _, document, _ in top_rows],
        "metadata": [metadata for _, _, metadata in top_rows],
        "scores": [score for score, _, _ in top_rows],
        "query": query,
    }


def retrieve(query, k=5, doc_type=None, tags=None, source=None, min_score=None, vectordb=None):
    """
    Universal retrieval with flexible metadata filtering.
    
    Args:
        query: Search query string
        k: Number of results to return
        doc_type: Filter by document type ('transcript', 'policy', or 'client_upload')
        tags: Filter by tags (list of tags, e.g., ['media', 'community'])
               For transcripts only - uses OR logic (chunk must have ANY tag)
        source: Filter by specific source filename
        min_score: Optional minimum similarity score threshold (lower is more similar)
    
    Returns:
        dict with chunks, metadata, and optional scores
    
    Examples:
        # Only transcripts
        retrieve(query, doc_type='transcript')
        
        # Transcripts with specific tags
        retrieve(query, doc_type='transcript', tags=['media', 'community'])
        
        # Only policy docs
        retrieve(query, doc_type='policy')
        
        # Specific source file
        retrieve(query, source='Boston Anti-Displacement Plan Analysis.txt')
        
        # Everything
        retrieve(query)
    """
    # Defensive clamp: Chroma requires k >= 1
    try:
        k = int(k)
    except (TypeError, ValueError):
        k = 5
    if k <= 0:
        k = 5

    # Build filter dictionary (Chroma requires $and / $or for combinations)
    filter_dict = None

    # Allow doc_type to be a single value or a list of values
    doc_filter = None
    if isinstance(doc_type, (list, tuple)):
        doc_types = [dt for dt in doc_type if dt]
        if len(doc_types) == 1:
            doc_filter = {"doc_type": doc_types[0]}
        elif len(doc_types) > 1:
            doc_filter = {"$or": [{"doc_type": dt} for dt in doc_types]}
    elif doc_type:
        doc_filter = {"doc_type": doc_type}

    if doc_filter and source:
        filter_dict = {
            "$and": [
                doc_filter,
                {"source": source},
            ]
        }
    elif doc_filter:
        filter_dict = doc_filter
    elif source:
        filter_dict = {"source": source}

    def _do_semantic_search(vectordb):
        # Retrieve with or without score threshold
        if min_score is not None:
            results_with_scores = vectordb.similarity_search_with_score(
                query,
                k=k * 3 if tags else k,
                filter=filter_dict if filter_dict else None,
            )

            # Post-process tag filtering if needed (soft filter: fall back to unfiltered if no matches)
            if tags:
                filtered_results = []
                for doc, score in results_with_scores:
                    if "tags" in doc.metadata:
                        doc_tags = [t.strip() for t in doc.metadata["tags"].split(",")]
                        if any(tag in doc_tags for tag in tags):
                            filtered_results.append((doc, score))
                            if len(filtered_results) >= k:
                                break
                if filtered_results:
                    results_with_scores = filtered_results

            filtered_results = [(doc, score) for doc, score in results_with_scores if score <= min_score]

            return {
                "chunks": [doc.page_content for doc, _ in filtered_results[:k]],
                "metadata": [doc.metadata for doc, _ in filtered_results[:k]],
                "scores": [score for _, score in filtered_results[:k]],
                "query": query,
            }

        results = vectordb.similarity_search(
            query,
            k=k * 3 if tags else k,
            filter=filter_dict if filter_dict else None
        )

        if tags:
            filtered_results = []
            for doc in results:
                if "tags" in doc.metadata:
                    doc_tags = [t.strip() for t in doc.metadata["tags"].split(",")]
                    if any(tag in doc_tags for tag in tags):
                        filtered_results.append(doc)
                        if len(filtered_results) >= k:
                            break
            if filtered_results:
                results = filtered_results

        return {
            "chunks": [doc.page_content for doc in results[:k]],
            "metadata": [doc.metadata for doc in results[:k]],
            "scores": None,
            "query": query,
        }

    # The nightly ingestion cron writes to the same SQLite-backed store. A
    # concurrent write can momentarily lock the DB; rather than immediately
    # degrading to keyword search, wait out a transient lock and retry.
    for attempt in range(1, _CHROMA_MAX_ATTEMPTS + 1):
        try:
            db = vectordb if vectordb is not None else load_vectordb()
            return _do_semantic_search(db)
        except Exception as exc:
            if _is_locked_error(exc) and attempt < _CHROMA_MAX_ATTEMPTS:
                print(f"  ⏳ Vector store busy (attempt {attempt}/{_CHROMA_MAX_ATTEMPTS}): {exc}; retrying")
                time.sleep(min(2 ** (attempt - 1), 4) * 0.25)
                continue
            print(f"  ⚠️ Semantic retrieval failed, falling back to keyword search: {exc}")
            return _keyword_retrieve(query, k=k, doc_type=doc_type, tags=tags, source=source)


def retrieve_transcripts(query, tags=None, k=5):
    """
    Convenience function for transcript-only search.
    
    Args:
        query: Search query
        tags: Optional list of tags to filter by (e.g., ['media', 'community'])
        k: Number of results
    
    Example:
        retrieve_transcripts("What do people say about safety?", tags=['safety'])
    """
    return retrieve(query, k=k, doc_type='transcript', tags=tags)


def retrieve_policies(query, k=5, source=None):
    """
    Convenience function for policy-only search.
    
    Args:
        query: Search query
        k: Number of results
        source: Optional specific policy document to search
    
    Example:
        retrieve_policies("anti-displacement strategies")
    """
    return retrieve(query, k=k, doc_type='policy', source=source)



def format_results(result_dict):
    """Format retrieval results for display."""
    formatted = []
    formatted.append(f"Query: {result_dict['query']}")
    formatted.append(f"Total results: {len(result_dict['chunks'])}\n")
    
    for i, (chunk, meta) in enumerate(zip(result_dict['chunks'], result_dict['metadata']), 1):
        formatted.append(f"\n{'='*80}")
        formatted.append(f"Result {i}")
        formatted.append(f"Source: {meta.get('source', 'Unknown')}")
        formatted.append(f"Type: {meta.get('doc_type', 'Unknown')}")
        
        if meta.get('tags'):
            formatted.append(f"Tags: {meta['tags']}")
        if meta.get('Heading'):
            formatted.append(f"Section: {meta['Heading']}")
        
        if result_dict.get('scores'):
            formatted.append(f"Score: {result_dict['scores'][i-1]:.4f}")
        
        formatted.append(f"\nContent:\n{chunk[:300]}...")
    
    return '\n'.join(formatted)


def retrieve_rss(query, k=5, source=None):
    """
    Search RSS-ingested content in ChromaDB (doc_type='calendar_event' with feed_url set).
    
    Args:
        query: Search query
        k: Number of results
        source: Optional specific feed source name (e.g., 'Dorchester Reporter')
    
    Example:
        retrieve_rss("upcoming events Codman Square")
    """
    # RSS docs have doc_type='calendar_event' AND a 'feed_url' metadata field.
    # We filter by calendar_event and post-filter to only RSS-sourced docs
    # (newsletter PDFs also use calendar_event but don't have feed_url).
    results = retrieve(query, k=k * 2, doc_type="calendar_event", source=source)

    # Post-filter: keep only RSS docs (they have feed_url in metadata)
    rss_chunks = []
    rss_meta = []
    for chunk, meta in zip(results["chunks"], results["metadata"]):
        if meta.get("feed_url"):
            rss_chunks.append(chunk)
            rss_meta.append(meta)
            if len(rss_chunks) >= k:
                break

    # If source filter returned nothing, fall back to unfiltered RSS docs
    if not rss_chunks:
        fallback = retrieve(query, k=k * 2, doc_type="calendar_event")
        for chunk, meta in zip(fallback["chunks"], fallback["metadata"]):
            if meta.get("feed_url"):
                rss_chunks.append(chunk)
                rss_meta.append(meta)
                if len(rss_chunks) >= k:
                    break

    return {
        "chunks": rss_chunks,
        "metadata": rss_meta,
        "scores": None,
        "query": query,
    }

if __name__ == "__main__":
    # Example usage
    print("=== Example 1: Search transcripts only ===")
    results = retrieve_transcripts("What do people say about media representation?", k=3)
    print(format_results(results))
    
    print("\n\n=== Example 2: Search with specific tags ===")
    results = retrieve_transcripts("community safety", tags=['safety'], k=3)
    print(format_results(results))
    
    print("\n\n=== Example 3: Search policy docs ===")
    results = retrieve_policies("housing affordability", k=3)
    print(format_results(results))
