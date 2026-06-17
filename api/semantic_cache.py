"""Semantic response cache.

Caches final answers keyed by the *meaning* of the question (its embedding), so a
near-duplicate question can be answered instantly without re-running the whole
routing/retrieval/generation pipeline.

This is deliberately conservative and OFF by default (set SEMANTIC_CACHE_ENABLED=true):
- Only first-turn questions are cached/served (follow-ups depend on conversation
  context, so they must never hit the cache).
- Time-sensitive questions are excluded by the caller.
- Entries expire (TTL) so answers can't go stale indefinitely.
- A Gemini embedding failure degrades to "no cache" rather than raising.
"""

from __future__ import annotations

import math
import threading
import time
from typing import Any, Callable, Dict, List, Optional, Tuple


def _cosine(a: List[float], b: List[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = na = nb = 0.0
    for x, y in zip(a, b):
        dot += x * y
        na += x * x
        nb += y * y
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (math.sqrt(na) * math.sqrt(nb))


class SemanticResponseCache:
    def __init__(
        self,
        *,
        enabled: bool,
        threshold: float,
        ttl_seconds: int,
        max_entries: int,
        embed_fn: Callable[[str], List[float]],
    ) -> None:
        self.enabled = enabled
        self.threshold = threshold
        self.ttl_seconds = ttl_seconds
        self.max_entries = max_entries
        self._embed_fn = embed_fn
        self._entries: List[Dict[str, Any]] = []
        self._lock = threading.Lock()
        # Reuse the embedding computed during get() for an immediately following
        # put() of the same question, so a cache miss only embeds once.
        self._last_embedding: Optional[Tuple[str, List[float]]] = None

    def _embed(self, question: str) -> Optional[List[float]]:
        if self._last_embedding and self._last_embedding[0] == question:
            return self._last_embedding[1]
        try:
            vec = self._embed_fn(question)
        except Exception as exc:  # noqa: BLE001
            print(f"  ⚠️ Semantic cache embedding failed: {exc}")
            return None
        if vec:
            self._last_embedding = (question, vec)
        return vec

    def get(self, question: str) -> Optional[Dict[str, Any]]:
        if not self.enabled:
            return None
        vec = self._embed(question)
        if not vec:
            return None
        now = time.time()
        with self._lock:
            self._entries = [e for e in self._entries if e["expires_at"] > now]
            best: Optional[Dict[str, Any]] = None
            best_sim = 0.0
            for entry in self._entries:
                sim = _cosine(vec, entry["vec"])
                if sim > best_sim:
                    best_sim = sim
                    best = entry
            if best is not None and best_sim >= self.threshold:
                best["last_used"] = now
                payload = dict(best["payload"])
                payload["_similarity"] = round(best_sim, 4)
                return payload
        return None

    def put(self, question: str, payload: Dict[str, Any]) -> None:
        if not self.enabled:
            return
        vec = self._embed(question)
        if not vec:
            return
        now = time.time()
        with self._lock:
            self._entries.append(
                {
                    "vec": vec,
                    "payload": dict(payload),
                    "question": question,
                    "expires_at": now + self.ttl_seconds,
                    "last_used": now,
                }
            )
            if len(self._entries) > self.max_entries:
                # Drop least-recently-used down to capacity.
                self._entries.sort(key=lambda e: e["last_used"])
                self._entries = self._entries[-self.max_entries:]
