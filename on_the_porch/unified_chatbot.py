import os
import re
import sys
import json
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv


# Ensure we can import RAG utilities from the directory with a space in its name
_THIS_FILE = Path(__file__).resolve()
_REAL_DIR = _THIS_FILE.parent
_ROOT_DIR = _REAL_DIR.parent.parent
load_dotenv(_ROOT_DIR / ".env")
# RAG utilities live in `on_the_porch/rag stuff`
_RAG_DIR = _REAL_DIR / "rag stuff"
if str(_RAG_DIR) not in sys.path:
    sys.path.insert(0, str(_RAG_DIR))


# Import RAG retrieval helpers; import SQL pipeline lazily only when needed
import retrieval  # type: ignore  # noqa: E402
import boston_gov  # type: ignore  # noqa: E402

# Local Gemini client config (avoid importing app3 at module load)
try:
    import google.generativeai as genai  # type: ignore
except Exception:  # pragma: no cover
    genai = None  # type: ignore

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash-lite")
GEMINI_SUMMARY_MODEL = os.getenv("GEMINI_SUMMARY_MODEL", GEMINI_MODEL)
# Cheaper/faster model for internal classification tasks (routing, the
# needs-new-data gate, relevance classification). Defaults to GEMINI_MODEL so
# behavior is unchanged until you point it at a lighter model (e.g. a -lite).
GEMINI_FAST_MODEL = os.getenv("GEMINI_FAST_MODEL", GEMINI_MODEL)
GEMINI_REQUEST_TIMEOUT = int(os.getenv("GEMINI_REQUEST_TIMEOUT", "20"))
_GEMINI_SUPPORTS_REQUEST_OPTIONS = True
# Run independent retrievals (and hybrid SQL+RAG) concurrently for lower latency.
_PARALLEL_RETRIEVAL = os.getenv("PARALLEL_RETRIEVAL", "true").strip().lower() in ("1", "true", "yes")
FALLBACK_TRIGGER_PREFIX = "i did not find exact information"
SQL_FALLBACK_TRIGGER_PHRASES = (
    FALLBACK_TRIGGER_PREFIX,
    "i don't see",
    "i do not see",
    "i don't currently see",
    "i do not currently see",
    "i couldn't find",
    "i could not find",
)
BOSTON_GOV_BOILERPLATE_PATTERNS = (
    "search results below",
    "links to relevant pages",
    "links to relevant",
    "relevant pages and more information",
    "find links",
)

# Short greetings / pleasantries — answered instantly without routing or retrieval.
_SMALL_TALK_EXACT = frozenset({
    "hi", "hello", "hey", "yo", "hiya", "howdy", "sup",
    "thanks", "thank you", "thx", "ty",
    "bye", "goodbye", "cya", "see ya", "see you",
    "ok", "okay", "cool", "great", "help",
})
_SMALL_TALK_PATTERNS = (
    r"^(hi|hello|hey|yo|hiya|howdy)\s*(there|everyone|friend)?$",
    r"^how are you$",
    r"^how r u$",
    r"^how(?:'re| is) you$",
    r"^how(?:'s| is) it going$",
    r"^what(?:'s| is) up$",
    r"^good (morning|afternoon|evening|night)$",
    r"^who are you$",
    r"^what are you$",
    r"^what can you (do|help with)$",
    r"^how does this work$",
    r"^how do i use this$",
    r"^what is this$",
    r"^what do you do$",
)
_SMALL_TALK_DOMAIN_HINTS = (
    "event", "311", "911", "crime", "safety", "dorchester", "news",
    "meeting", "happening", "request", "neighborhood", "budget", "policy",
    "arrest", "shooting", "calendar", "schedule", "activity", "service",
)


def _normalize_small_talk(message: str) -> str:
    text = (message or "").strip().lower()
    text = re.sub(r"[^\w\s']", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def is_small_talk(message: str) -> bool:
    """True when the message is a brief greeting or pleasantry, not a data question."""
    raw = (message or "").strip()
    if not raw or len(raw) > 120:
        return False
    lower = raw.lower()
    if any(hint in lower for hint in _SMALL_TALK_DOMAIN_HINTS):
        return False
    normalized = _normalize_small_talk(raw)
    if not normalized:
        return False
    if normalized in _SMALL_TALK_EXACT:
        return True
    return any(re.fullmatch(pat, normalized) for pat in _SMALL_TALK_PATTERNS)


def small_talk_response(message: str) -> str:
    """Instant reply for greetings and other brief non-data messages."""
    n = _normalize_small_talk(message)
    if any(x in n for x in ("how are you", "how r u", "how is it going", "how's it going")):
        return (
            "I'm doing well — thanks for asking! I'm here to help with Dorchester community "
            "info: events, 311 activity, safety trends, and neighborhood news. "
            "What would you like to know?"
        )
    if n.startswith(("thanks", "thank you", "thx", "ty")) or n in ("thanks", "thank you", "thx", "ty"):
        return (
            "You're welcome! Ask anytime about events, city services, safety, or what's "
            "happening in the neighborhood."
        )
    if n.startswith(("bye", "goodbye", "cya", "see ya", "see you")):
        return "Goodbye! Come back anytime you have questions about Dorchester."
    if "who are you" in n or "what are you" in n or "what can you" in n or "what do you do" in n:
        return (
            "I'm your Dorchester community assistant. I can help with local events, 311 requests, "
            "safety data, and neighborhood news. Try “Events this week” or “311 activity” to get started."
        )
    if n.startswith("good "):
        return (
            "Good to see you! Ask me about Dorchester events, services, safety, or neighborhood trends — "
            "what's on your mind?"
        )
    if n in ("help",) or "what can you help" in n or "how does this work" in n or "how do i use this" in n or n == "what is this":
        return (
            "I can answer questions about Dorchester events, 311 service requests, safety trends, "
            "and community news. Try one of the suggestion chips below, or ask in your own words."
        )
    return (
        "Hi! I'm here to help with Dorchester community questions — events, services, safety, "
        "and local news. What would you like to know?"
    )


def build_small_talk_result(
    message: str,
    retrieval_cache: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    answer = small_talk_response(message)
    cache = retrieval_cache or create_empty_cache()
    return {
        "answer": answer,
        "mode": "chitchat",
        "sources": [],
        "result": {"answer": answer},
        "retrieval_cache": cache,
    }


def _bootstrap_env() -> None:
    """Ensure environment variables are loaded from the repo root .env."""
    try:
        load_dotenv(_ROOT_DIR / ".env")
    except Exception:
        pass


def _fix_retrieval_vectordb_path() -> None:
    # retrieval.VECTORDB_DIR is relative; ensure it points to on_the_porch/vectordb_new
    try:
        expected = _REAL_DIR / "vectordb_new"
        retrieval.VECTORDB_DIR = expected  # type: ignore[attr-defined]
    except Exception:
        pass


def _get_llm_client():
    if genai is None:
        raise RuntimeError("gemini client not installed: pip install google-generativeai")
    api_key = os.getenv("GEMINI_API_KEY")
    if api_key:
        genai.configure(api_key=api_key)
    return genai


def _generate_content(model, prompt, generation_config):
    """Call Gemini with a per-request timeout when the SDK supports it."""
    global _GEMINI_SUPPORTS_REQUEST_OPTIONS
    if _GEMINI_SUPPORTS_REQUEST_OPTIONS:
        try:
            return model.generate_content(
                prompt,
                generation_config=generation_config,
                request_options={"timeout": GEMINI_REQUEST_TIMEOUT},
            )
        except TypeError:
            _GEMINI_SUPPORTS_REQUEST_OPTIONS = False
    return model.generate_content(prompt, generation_config=generation_config)


def _should_trigger_boston_gov_fallback(answer: str) -> bool:
    if not answer:
        return False
    first_two_lines = [line.strip() for line in answer.splitlines() if line.strip()][:2]
    return any(FALLBACK_TRIGGER_PREFIX in line.lower() for line in first_two_lines)


def _should_trigger_sql_fallback(answer: str) -> bool:
    if not answer:
        return False
    first_two_lines = [line.strip().lower() for line in answer.splitlines() if line.strip()][:2]
    sentence_text = " ".join(first_two_lines)
    return any(phrase in sentence_text for phrase in SQL_FALLBACK_TRIGGER_PHRASES)


def _clean_boston_gov_fallback_text(text: str) -> str:
    """Remove Boston.gov search UI boilerplate before exposing text to users."""
    cleaned_lines: List[str] = []
    for line in (text or "").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        lowered = stripped.lower()
        if any(pattern in lowered for pattern in BOSTON_GOV_BOILERPLATE_PATTERNS):
            continue
        cleaned_lines.append(stripped)
    return "\n".join(cleaned_lines).strip()


def _classify_boston_gov_exact_match(question: str, page_text: str) -> bool:
    excerpt = (page_text or "").strip()[:4000]
    if not excerpt:
        return False

    client = _get_llm_client()
    model = client.GenerativeModel(GEMINI_FAST_MODEL)
    system_prompt = (
        "You are a relevance classifier.\n"
        "Your task is to decide whether the provided Boston.gov AI answer answers the user's question well enough to be used as the main answer.\n"
        "Return only one word: yes or no.\n"
        "Do not be overly strict.\n"
        "Return yes if the answer is clearly relevant, substantially answers the question, or provides the practical information the user is looking for, even if it is not a perfect exact match.\n"
        "Return no only if the answer is mostly unrelated, too vague, or missing the key information needed to answer the question."
    )
    user_prompt = (
        "Question:\n"
        f"{question}\n\n"
        "Boston.gov AI answer:\n"
        f"{excerpt}\n\n"
        "Does this AI answer answer the user's question well enough to be used as the main answer?"
    )

    try:
        resp = _generate_content(
            model,
            system_prompt + "\n\n" + user_prompt,
            generation_config={"temperature": 0},
        )
        label = (resp.text or "").strip().lower()
        print(f"  🤖 Boston.gov fallback classifier: {label!r}")
        return label == "yes"
    except Exception as exc:
        print(f"  ⚠️ Boston.gov fallback classifier failed: {exc}")
        return False


def _regenerate_with_boston_gov_context(question: str, original_answer: str, ai_text: str) -> str:
    client = _get_llm_client()
    model = client.GenerativeModel(GEMINI_MODEL)
    system_prompt = (
        "You are a friendly, non-technical assistant helping people understand Dorchester community information.\n"
        "You are revising an answer using two temporary context sources:\n"
        "1. The assistant's original answer.\n"
        "2. A related Boston.gov AI answer.\n\n"
        "Write one clean final answer for the user.\n"
        "Use the Boston.gov information only as supporting context when it is relevant.\n"
        "Do not claim Boston.gov directly answers the question if it does not.\n"
        "Remove search-page boilerplate such as references to links, relevant pages, or search results below.\n"
        "Do not mention internal tools, fallback logic, classifiers, or vector databases.\n"
        "If the exact answer is still not available, say so clearly and then share the most helpful related information."
    )
    user_prompt = (
        "Question:\n"
        f"{question}\n\n"
        "Original assistant answer:\n"
        f"{original_answer}\n\n"
        "Related Boston.gov AI answer:\n"
        f"{ai_text[:4000]}\n\n"
        "Please produce a single/combined improved answer for the user:"
    )
    try:
        resp = _generate_content(
            model,
            system_prompt + "\n\n" + user_prompt,
            generation_config={"temperature": 0.2},
        )
        regenerated = (resp.text or "").strip()
        if regenerated:
            print("  🤖 Boston.gov fallback: regenerated final answer using original + Boston.gov context")
            return regenerated
    except Exception as exc:
        print(f"  ⚠️ Boston.gov fallback regeneration failed: {exc}")
    return original_answer


def _build_boston_gov_fallback_answer(question: str, original_answer: str) -> str:
    print("  🏛️ Boston.gov fallback: trigger detected from model answer")
    ai_result = boston_gov.get_boston_gov_ai_answer(question)
    # Defensive: get_boston_gov_ai_answer should always return a dict, but if a
    # future change or unexpected path returns something else, never crash the
    # request — just keep the original answer.
    if not isinstance(ai_result, dict):
        print(f"  ⚠️ Boston.gov fallback: unexpected result type {type(ai_result).__name__}; keeping original answer")
        return original_answer
    ai_text = str(ai_result.get("text", "") or "").strip()
    search_url = ai_result.get("search_url", "")

    # Prefer the first real link from the AI summary; fall back to search URL
    scraped_links = ai_result.get("links", []) or []
    primary_link = ""
    for link_obj in scraped_links:
        href = (link_obj.get("href") or "").strip()
        if href and href.startswith(("http://", "https://", "/")):
            if href.startswith("/"):
                href = f"https://www.boston.gov{href}"
            primary_link = href
            break
    if not primary_link:
        primary_link = search_url

    if not ai_text:
        print("  ⚠️ Boston.gov fallback: no AI answer text found")
        return original_answer
    cleaned_ai_text = _clean_boston_gov_fallback_text(ai_text)
    if not cleaned_ai_text:
        print("  ⚠️ Boston.gov fallback: AI answer only contained boilerplate")
        return original_answer

    print("  🏛️ Boston.gov fallback: scraped AI answer text:")
    for line in cleaned_ai_text.splitlines():
        if line.strip():
            print(f"     {line}")
    print(f"  🔗 Boston.gov fallback: using link {primary_link}")

    excerpt = cleaned_ai_text[:2000].strip()
    is_exact_match = _classify_boston_gov_exact_match(question, ai_text)
    if is_exact_match:
        # Saving to the vector store is a cache-for-next-time side effect and
        # involves a remote embedding call + a SQLite write. Do it in the
        # background so the user is not blocked waiting on it; the answer we
        # return (excerpt) does not depend on the save succeeding.
        def _save_boston_gov_answer() -> None:
            try:
                boston_gov.add_boston_gov_answer_to_vectordb(
                    question,
                    cleaned_ai_text,
                    link=primary_link,
                )
            except Exception as exc:  # noqa: BLE001
                print(f"  ⚠️ Boston.gov fallback vectordb save failed: {exc}")

        threading.Thread(target=_save_boston_gov_answer, daemon=True).start()
        return excerpt
    print("  ⚠️ Boston.gov fallback: classifier said scraped AI answer is not an exact match")
    return _regenerate_with_boston_gov_context(question, original_answer, cleaned_ai_text)


def _apply_default_fallback_if_needed(question: str, answer: str) -> str:
    if not _should_trigger_boston_gov_fallback(answer):
        return answer
    try:
        return _build_boston_gov_fallback_answer(question, answer)
    except Exception as exc:  # noqa: BLE001
        print(f"  ⚠️ Boston.gov fallback failed; returning original answer: {exc}")
        return answer


def _apply_sql_fallback_if_needed(question: str, answer: str) -> str:
    if not _should_trigger_sql_fallback(answer):
        return answer
    try:
        return _build_boston_gov_fallback_answer(question, answer)
    except Exception as exc:  # noqa: BLE001
        print(f"  ⚠️ Boston.gov fallback failed; returning original answer: {exc}")
        return answer


def _safe_json_loads(text: str, default: Dict[str, Any]) -> Dict[str, Any]:
    try:
        return json.loads(text)
    except Exception:
        return default


# ---------------------------------------------------------------------------
# Retrieval Cache: stores the most recent retrieval results for follow-up use
# ---------------------------------------------------------------------------

def create_empty_cache() -> Dict[str, Any]:
    """Create an empty retrieval cache structure."""
    return {
        "mode": None,  # "sql", "rag", or "hybrid"
        "timestamp": None,
        "question": None,
        "sql_result": None,  # Raw SQL rows/data
        "sql_query": None,
        "rag_chunks": None,  # List of text chunks
        "rag_metadata": None,  # List of metadata dicts
        "answer": None,  # The generated answer
    }


def build_retrieval_cache(
    mode: str,
    question: str,
    answer: str,
    sql_result: Optional[Dict[str, Any]] = None,
    sql_query: Optional[str] = None,
    rag_chunks: Optional[List[str]] = None,
    rag_metadata: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Build a retrieval cache from the results of a query."""
    return {
        "mode": mode,
        "timestamp": datetime.now().isoformat(),
        "question": question,
        "sql_result": sql_result,
        "sql_query": sql_query,
        "rag_chunks": rag_chunks[:20] if rag_chunks else None,  # Cap chunks
        "rag_metadata": rag_metadata[:20] if rag_metadata else None,
        "answer": answer,
    }


def summarize_cache(cache: Optional[Dict[str, Any]]) -> str:
    """Create a concise text summary of what's in the cache for the LLM."""
    if not cache or not cache.get("mode"):
        return "(No cached data available)"
    
    parts = []
    mode = cache.get("mode", "unknown")
    question = cache.get("question", "")
    timestamp = cache.get("timestamp", "")
    
    parts.append(f"Cached data from mode '{mode}' for question: \"{question}\"")
    if timestamp:
        parts.append(f"Retrieved at: {timestamp}")
    
    # Summarize SQL results
    sql_result = cache.get("sql_result")
    if sql_result:
        rows = sql_result.get("rows", [])
        columns = sql_result.get("columns", [])
        row_count = len(rows) if isinstance(rows, list) else 0
        col_names = ", ".join(columns[:10]) if columns else "unknown columns"
        parts.append(f"SQL data: {row_count} rows with columns [{col_names}]")
        
        # Include actual data preview (first few rows)
        if rows and row_count > 0:
            preview_rows = rows[:10]  # First 10 rows for preview
            parts.append(f"Data preview (first {len(preview_rows)} rows):")
            for i, row in enumerate(preview_rows, 1):
                if isinstance(row, dict):
                    row_str = ", ".join(f"{k}: {v}" for k, v in list(row.items())[:6])
                elif isinstance(row, (list, tuple)):
                    row_str = ", ".join(str(v) for v in row[:6])
                else:
                    row_str = str(row)[:200]
                parts.append(f"  Row {i}: {row_str}")
    
    # Summarize RAG chunks
    rag_meta = cache.get("rag_metadata")
    rag_chunks = cache.get("rag_chunks")
    if rag_meta:
        chunk_count = len(rag_meta)
        sources = list(set(m.get("source", "unknown") for m in rag_meta[:10]))
        doc_types = list(set(m.get("doc_type", "unknown") for m in rag_meta[:10]))
        parts.append(f"RAG data: {chunk_count} chunks from sources: {sources[:5]}, types: {doc_types}")
        
        # Include chunk previews
        if rag_chunks:
            parts.append(f"Chunk previews (first {min(5, len(rag_chunks))}):")
            for i, chunk in enumerate(rag_chunks[:5], 1):
                preview = chunk[:300] + "..." if len(chunk) > 300 else chunk
                parts.append(f"  Chunk {i}: {preview}")
    
    return "\n".join(parts)


def _check_if_needs_new_data(
    question: str,
    conversation_history: Optional[List[Dict[str, str]]] = None,
    retrieval_cache: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Check if the question can be answered from conversation history and/or cached data.
    Returns: {"needs_new_data": bool, "reason": str}
    """
    # If no history and no cache, always need new data
    has_history = conversation_history and len(conversation_history) > 0
    has_cache = retrieval_cache and retrieval_cache.get("mode")
    
    if not has_history and not has_cache:
        return {"needs_new_data": True, "reason": "No conversation history or cached data available"}
    
    client = _get_llm_client()
    
    # Build conversation context for analysis
    history_context = ""
    if conversation_history:
        for msg in conversation_history[-10:]:  # Last 10 messages
            role = msg.get("role", "")
            content = msg.get("content", "")
            if role and content:
                history_context += f"{role.upper()}: {content}\n\n"
    
    # Build cache summary
    cache_summary = summarize_cache(retrieval_cache)
    
    system_prompt = (
    "You analyze if a user's question can be answered from conversation history and/or cached retrieval data, or if it needs new data retrieval.\n\n"
    "You have access to:\n"
    "1. Conversation history (previous Q&A exchanges)\n"
    "2. Cached data (the actual data rows/chunks from the most recent retrieval)\n\n"
    "CRITICAL RULES (check these FIRST):\n"
    "- If the question names a specific event, person, place, or entity by name, and that exact name is NOT visibly present in the cached data or conversation history → needs_new_data = true\n"
    "- If the question references a specific date, day of the week, or time period (e.g., 'April 25', 'this Saturday', 'tomorrow', 'next week') and the cached data does not already contain matching data for that date → needs_new_data = true\n"
    "- If the question asks for factual details (location, time, description, contact info, schedule) about an entity and those specific details are NOT in the cache → needs_new_data = true\n\n"
    "Only after checking the critical rules above, apply these:\n"
    "- If the cached data contains the information needed to answer the question → needs_new_data = false\n"
    "- If question is a pure follow-up referencing items already shown in the cache (e.g., 'the second one', 'the one in Dorchester') AND that item is visible in the cache → needs_new_data = false\n"
    "- If question is a clarification or rephrasing of a previous answer → needs_new_data = false\n"
    "- If question asks for new data, different time period not in cache, different metrics, or a completely new topic → needs_new_data = true\n\n"
    "When in doubt, prefer needs_new_data = true. It is much worse to answer with stale or missing data than to fetch fresh data.\n\n"
    "Return ONLY valid JSON with keys: needs_new_data (boolean) and reason (brief string explaining your decision)."
)
    
    user_prompt = (
        "Conversation History:\n" + (history_context if history_context else "(No previous conversation)") + "\n\n"
        "Cached Data:\n" + cache_summary + "\n\n"
        "Current Question: " + question + "\n\n"
        "Analyze if this question can be answered from the conversation history and/or cached data above, or if it needs new data retrieval.\n"
        "Return JSON only."
    )
    
    default_result = {"needs_new_data": True, "reason": "Error analyzing question, defaulting to new data"}
    
    try:
        model = client.GenerativeModel(GEMINI_FAST_MODEL)
        prompt = f"{system_prompt}\n\n{user_prompt}"
        resp = _generate_content(
            model,
            prompt,
            generation_config={"temperature": 0},
        )
        content = (resp.text or "").strip()
        
        # Remove code fences if present
        if content.startswith("```"):
            content = content.strip("`").strip()
            lines = content.splitlines()
            if lines and lines[0].strip().lower() in ("json", "javascript", "js"):
                content = "\n".join(lines[1:]).strip()
        
        result = _safe_json_loads(content, default_result)
        
        # Ensure needs_new_data is boolean
        needs_new = result.get("needs_new_data", True)
        if isinstance(needs_new, str):
            needs_new = needs_new.lower() in ("true", "yes", "1")
        result["needs_new_data"] = bool(needs_new)
        
        return result
    except Exception:
        return default_result


def _route_question(question: str) -> Dict[str, Any]:
    """
    Decide whether to answer via SQL, RAG, or HYBRID.
    Returns a dict like: {"mode": "sql|rag|hybrid", "transcript_tags": [..]|null, "policy_sources": [..]|null, "k": int}
    """
    client = _get_llm_client()

    system_prompt = (
        f"Today's date is {date.today().strftime('%A, %B %d, %Y')}.\n\n"
        "You are a STRICT routing classifier for a chatbot that combines SQL (structured data) and RAG (text documents).\n"
        "RAG includes transcripts, policy documents, RSS/news items, and cached Boston.gov answers.\n"
        "Whenever a question is routed to RAG or hybrid, cached Boston.gov answers will be searched along with the other RAG sources.\n"
        "You MUST classify the user's question into EXACTLY one of three modes: 'sql', 'rag', or 'hybrid'.\n"
        "These rules are MANDATORY and NON-NEGOTIABLE. Follow them EXACTLY.\n\n"
        "═══════════════════════════════════════════════════════════════════════════════\n"
        "CRITICAL ROUTING RULES - ABSOLUTE PRIORITY (CHECK IN THIS ORDER):\n"
        "═══════════════════════════════════════════════════════════════════════════════\n\n"
        "RULE 0: NEIGHBORHOOD NEWS / RSS QUESTIONS → 'rag' or 'hybrid'\n"
        "   - If question uses phrases like 'what's going on in [neighborhood]', 'what's new in', 'lately', 'recent news about', 'updates from [neighborhood]', 'what's happening in [neighborhood]' without asking for specific event schedules\n"
        "   - AND does not mention a specific day/week/date → mode MUST be 'hybrid' (SQL for 311 activity + RAG for RSS news)\n"
        "   - If question explicitly names a feed source (Dorchester Reporter, CSNDC, etc.) → mode MUST be 'rag'\n\n"
        "RULE 1: CRIME-RELATED QUESTIONS → Route based on question type\n"
        "   - If the question mentions ANY of: crime, crimes, arrest, arrests, offense, offenses, homicide, homicides, shooting, shootings, shots fired, safety incident, safety incidents, criminal activity, violence, violent\n"
        "   - THEN apply these sub-rules:\n"
        "     a) If asking for STATISTICS/NUMBERS ONLY (counts, trends, comparisons, breakdowns) → mode MUST be 'sql'\n"
        "        Examples: 'How many arrests were there?', 'What is the trend in shots fired?', 'Show me crime statistics', 'Which areas have highest arrests?' → sql\n"
        "     b) If asking for OPINIONS/CONTEXT ONLY (what people think/say about crime) → mode MUST be 'hybrid'\n"
        "        Examples: 'What do people say about crime?', 'How do residents feel about safety?' → hybrid\n"
        "     c) If asking for BOTH statistics AND context/opinions → mode MUST be 'hybrid'\n"
        "        Examples: 'How many homicides and what concerns come up?', 'Show crime trends and community concerns' → hybrid\n"
        "   - DO NOT use 'rag' alone for crime questions\n\n"
        "RULE 2: EVENT/CALENDAR/ACTIVITY QUESTIONS → ALWAYS 'sql' mode\n"
        "   - If the question contains the word 'event' or 'events' anywhere, mode MUST be 'sql' no matter what.\n"
        "   - If the question mentions ANY of: event, events, happening, schedule, calendar, activity, activities, 'what's on', 'what is on', 'going on', meeting, meetings, workshop, workshops, 'this week', 'next week', 'today', 'tomorrow', 'weekend', day of week\n"
        "   - THEN mode MUST be 'sql' (NO EXCEPTIONS)\n"
        "   - DO NOT use 'rag' or 'hybrid' for event/calendar questions\n"
        "   - Examples that MUST be 'sql':\n"
        "     * 'What events are happening this week?' → sql\n"
        "     * 'Show me fun activities for kids' → sql\n"
        "     * 'What public meetings are scheduled?' → sql\n"
        "     * 'What's happening on Saturday?' → sql\n"
        "     * 'Are there any community events?' → sql\n\n"
        "RULE 3: OPINION/PERSPECTIVE QUESTIONS → ALWAYS 'rag' mode\n"
        "   - If the question asks for: opinions, perspectives, feelings, views, what people think/say/believe/feel/describe, community views, resident views\n"
        "   - AND the question is NOT about crime (see Rule 1)\n"
        "   - THEN mode MUST be 'rag' (NO EXCEPTIONS)\n"
        "   - DO NOT use 'sql' or 'hybrid' for pure opinion questions\n"
        "   - Examples that MUST be 'rag':\n"
        "     * 'What do people think about displacement?' → rag\n"
        "     * 'How do community members feel about housing?' → rag\n"
        "     * 'What are people's opinions on media representation?' → rag\n"
        "     * 'What do residents say about the neighborhood?' → rag\n\n"
        "═══════════════════════════════════════════════════════════════════════════════\n"
        "SECONDARY ROUTING RULES (Apply if Rules 1-3 don't match):\n"
        "═══════════════════════════════════════════════════════════════════════════════\n\n"
        "RULE 4: PURE STATISTICS/NUMBERS → 'sql' mode\n"
        "   - Questions asking ONLY for: counts, numbers, statistics, trends, comparisons, breakdowns, aggregations\n"
        "   - Questions that can be answered with numeric data from tables\n"
        "   - This INCLUDES crime statistics (see Rule 1a)\n"
        "   - Examples: 'How many 311 requests?', 'What is the trend in shots fired?', 'Which areas have highest arrests?', 'How many homicides last year?'\n"
        "   - DO NOT use 'rag' or 'hybrid' if the question is purely numeric\n\n"
        "RULE 5: POLICY/DOCUMENT CONTENT → 'rag' mode\n"
        "   - Questions asking about: what a policy/document says, what a program aims to achieve, document content, newsletter content\n"
        "   - Examples: 'What does Slow Streets aim to achieve?', 'What strategies does the Anti-Displacement Plan propose?', 'What was in the newsletter?'\n"
        "   - DO NOT use 'sql' for document content questions\n\n"
        "RULE 6: COMBINED DATA + CONTEXT → 'hybrid' mode\n"
        "   - Questions that explicitly ask for BOTH numbers/data AND context/explanation\n"
        "   - Examples: 'How many homicides and what concerns come up?', 'Show trends and how policies address them'\n"
        "   - DO NOT use 'sql' or 'rag' alone if question explicitly requires both\n\n"
        "═══════════════════════════════════════════════════════════════════════════════\n"
        "STRICT VALIDATION REQUIREMENTS:\n"
        "═══════════════════════════════════════════════════════════════════════════════\n\n"
        "1. Mode MUST be exactly one of: 'sql', 'rag', or 'hybrid' (lowercase, no quotes in JSON)\n"
        "2. If mode is 'sql': transcript_tags, policy_sources, and folder_categories MUST be null\n"
        "3. If mode is 'rag' or 'hybrid':\n"
        "   - transcript_tags: array of 0-2 strings OR null (valid tags: safety, violence, youth, media, community, displacement, government, structural racism)\n"
        "   - policy_sources: array of strings OR null (valid: 'Boston Anti-Displacement Plan Analysis.txt', 'Boston Slow Streets Plan Analysis.txt', 'Imagine Boston 2030 Analysis.txt')\n"
        "   - folder_categories: array of strings OR null (valid: newsletters, policy, transcripts)\n"
        "   - k: integer between 3 and 10 (default 5, minimum 5 for event queries)\n"
        "4. For crime questions using 'hybrid' mode (Rule 1b or 1c): transcript_tags MUST include at least one of: 'safety' or 'violence'\n"
        "   For crime questions using 'sql' mode (Rule 1a): transcript_tags, policy_sources, and folder_categories MUST be null\n"
        "5. For opinion questions (Rule 3): transcript_tags should include relevant tags like 'community', 'displacement', 'youth', 'media'\n"
        "6. For event questions (Rule 2): k MUST be at least 5\n\n"
        "═══════════════════════════════════════════════════════════════════════════════\n"
        "OUTPUT FORMAT (STRICT):\n"
        "═══════════════════════════════════════════════════════════════════════════════\n\n"
        "Return ONLY valid JSON with EXACTLY these keys: mode, transcript_tags, policy_sources, folder_categories, k\n"
        "DO NOT include any explanatory text, markdown, code blocks, or additional content.\n"
        "DO NOT use backticks or markdown formatting.\n"
        "Example valid output:\n"
        '{"mode": "hybrid", "transcript_tags": ["safety"], "policy_sources": null, "folder_categories": null, "k": 5}\n\n'
        "NOTE: This system is configured for DORCHESTER ONLY. All SQL queries automatically filter to Dorchester data only."
    )

    user_prompt = (
        "Question:\n" + question + "\n\n"
        "Policy sources include: 'Boston Anti-Displacement Plan Analysis.txt', 'Boston Slow Streets Plan Analysis.txt', 'Imagine Boston 2030 Analysis.txt'.\n"
        "Transcript tags include: safety, violence, youth, media, community, displacement, government, structural racism.\n"
        "Folder categories (for client uploads): newsletters, policy, transcripts.\n"
        "Output JSON only."
    )

    default_plan = {
        "mode": "hybrid",
        "transcript_tags": None,
        "policy_sources": None,
        "folder_categories": None,
        "k": 5,
    }

    try:
        model = client.GenerativeModel(GEMINI_FAST_MODEL)
        prompt = f"{system_prompt}\n\n{user_prompt}"
        resp = _generate_content(
            model,
            prompt,
            generation_config={"temperature": 0},
        )
        content = (resp.text or "").strip()
        # Remove code fences if present
        if content.startswith("```"):
            content = content.strip("`").strip()
            # If the first line is a language tag, drop it
            lines = content.splitlines()
            if lines:
                if lines[0].strip().lower() in ("json", "javascript", "js"):
                    content = "\n".join(lines[1:]).strip()
        plan = _safe_json_loads(content, default_plan)
    except Exception:
        plan = default_plan

    # Normalize values
    mode = str(plan.get("mode", "hybrid")).lower()
    if mode not in {"sql", "rag", "hybrid"}:
        mode = "hybrid"  # Default to hybrid for safety
    tags = plan.get("transcript_tags")
    sources = plan.get("policy_sources")
    folders = plan.get("folder_categories")
    k = plan.get("k", 5)
    
    # Normalize and validate k
    try:
        k = int(k)
    except (ValueError, TypeError):
        k = 5
    
    # Ensure k is at least 3 (minimum for useful retrieval)
    if k < 3:
        k = 3
    
    # Force higher k for calendar questions to ensure good event coverage
    if _is_calendar_question(question):
        # Ensure at least 5 results for calendar queries
        if k < 5:
            k = 5
    
    # Cap k at reasonable maximum (20)
    if k > 20:
        k = 20

    print("Initial routing plan from LLM:", {"mode": mode, "transcript_tags": tags, "policy_sources": sources, "folder_categories": folders, "k": k})
    if _looks_like_rss_query(question):
        if _question_mentions_rss_source(question):
            mode = "rag"
        elif mode == "sql":
            mode = "hybrid"
        if not folders:
            folders = ["newsletters"]
        print("Final routing plan from LLM:", {"mode": mode, "transcript_tags": tags, "policy_sources": sources, "folder_categories": folders, "k": k})
    return {
        "mode": mode,
        "transcript_tags": tags if isinstance(tags, list) or tags is None else None,
        "policy_sources": sources if isinstance(sources, list) or sources is None else None,
        "folder_categories": folders if isinstance(folders, list) or folders is None else None,
        "k": k,
    }


def _build_rag_prompt(
    question: str,
    chunks: List[str],
    metadatas: List[Dict[str, Any]],
    conversation_history: Optional[List[Dict[str, str]]] = None,
) -> Tuple[Optional[str], List[str]]:
    """Build the full RAG answer prompt.

    Returns (full_prompt, context_parts). full_prompt is None when there are no
    chunks. context_parts is kept for the non-streaming fallback.
    """
    if not chunks:
        return None, []

    context_parts: List[str] = []
    for chunk, meta in zip(chunks, metadatas):
        source = meta.get("source", "Unknown")
        context_parts.append(f"[{source}]")
        context_parts.append(chunk)
        context_parts.append("")
    context = "\n".join(context_parts)

    system_prompt = (
        "You are a friendly, non-technical assistant helping people understand Dorchester community data and policies.\n"
        "This system is configured for DORCHESTER ONLY. All data queries are automatically filtered to Dorchester only.\n"
        "Use clear, everyday language and imagine you are talking to a neighbor, not a technical expert.\n"
        "Use only the provided SOURCES and do not add information that is not supported by the text.\n\n"
        "If the sources do not contain the user's exact answer, the first line of your response must begin with "
        "'I did not find exact information about ...' and briefly name the missing topic. After that first line, "
        "you may share closely related information from the sources if it is helpful.\n"
        "When you cite sources, use the source name naturally in the sentence (e.g. 'According to CSNDC...'). "
        "Do not use numbered source citations like (Source 1). Avoid technical jargon, and do not mention SQL, databases, RAG, "
        "retrieval methods, or internal tools.\n"
        "If the question involves numbers, be honest when the sources are limited and avoid inventing precise figures.\n"
        + ("\n\nYou are in a conversation. Use previous messages for context when the current question references earlier topics or asks for follow-ups." if conversation_history else "")
    )

    user_prompt = (
        "SOURCES:\n" + context + "\n\n" +
        "QUESTION: " + question + "\n\n" +
        "Please answer for the user in clear, everyday language:"
    )

    full_prompt = system_prompt + "\n\n"
    if conversation_history:
        for msg in conversation_history[-10:]:
            role = msg.get("role", "")
            content = msg.get("content", "")
            full_prompt += f"{role.upper()}: {content}\n\n"
    full_prompt += user_prompt
    return full_prompt, context_parts


def _compose_rag_answer(question: str, chunks: List[str], metadatas: List[Dict[str, Any]], conversation_history: Optional[List[Dict[str, str]]] = None) -> str:
    full_prompt, context_parts = _build_rag_prompt(question, chunks, metadatas, conversation_history)
    if full_prompt is None:
        return "No relevant information found."

    client = _get_llm_client()
    model = client.GenerativeModel(GEMINI_MODEL)
    try:
        resp = _generate_content(
            model,
            full_prompt,
            generation_config={"temperature": 0.3},
        )
        return (resp.text or "").strip()
    except Exception:
        return "\n\n".join(context_parts[:10])  # fallback: show a sample of context


def _answer_from_history(
    question: str,
    conversation_history: Optional[List[Dict[str, str]]] = None,
    retrieval_cache: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Generate an answer from conversation history and/or cached retrieval data.
    This is used for follow-up questions that can be answered from previous context.
    """
    full_prompt = _build_history_prompt(question, conversation_history, retrieval_cache)
    if full_prompt is None:
        return "I don't have any previous conversation or data to reference. Could you ask your question again?"

    client = _get_llm_client()
    model = client.GenerativeModel(GEMINI_MODEL)
    try:
        resp = _generate_content(
            model,
            full_prompt,
            generation_config={"temperature": 0.3},
        )
        return (resp.text or "").strip()
    except Exception:
        return "I encountered an error answering from the available information. Could you rephrase your question?"


def _build_history_prompt(
    question: str,
    conversation_history: Optional[List[Dict[str, str]]] = None,
    retrieval_cache: Optional[Dict[str, Any]] = None,
) -> Optional[str]:
    """Build the full 'answer from history/cache' prompt, or None if no data."""
    has_history = conversation_history and len(conversation_history) > 0
    has_cache = retrieval_cache and retrieval_cache.get("mode")
    if not has_history and not has_cache:
        return None

    cache_context = ""
    if has_cache:
        cache_context = _build_cache_context_for_answer(retrieval_cache)

    system_prompt = (
        "You are a friendly, non-technical assistant helping people understand Dorchester community data and policies.\n"
        "This system is configured for DORCHESTER ONLY. All data queries are automatically filtered to Dorchester only.\n"
        "Use clear, everyday language and imagine you are talking to a neighbor, not a technical expert.\n\n"
        "Answer the user's question based on the conversation history and cached data provided. "
        "Do not mention that you're using cached data or conversation history - just answer naturally as if continuing the conversation.\n"
        "If the question asks about specific items (e.g., 'tell me more about event #2', 'what about the first one'), "
        "use the cached data to provide detailed information about those specific items.\n"
        "If the question references previous answers, numbers, or statistics, use those in your response.\n"
        "If you cannot answer the user's exact question from the available information, the first line of your response must begin "
        "with 'I did not find exact information about ...' and briefly name the missing topic. After that first line, you may share "
        "closely related information from the available context if it helps.\n"
        "Avoid technical jargon, and do not mention SQL, databases, RAG, retrieval methods, or internal tools."
    )

    history_text = ""
    if conversation_history:
        for msg in conversation_history[-20:]:  # Last 20 messages for context
            role = msg.get("role", "")
            content = msg.get("content", "")
            if role and content:
                history_text += f"{role.upper()}: {content}\n\n"

    user_prompt = ""
    if history_text:
        user_prompt += "Conversation History:\n" + history_text + "\n\n"
    if cache_context:
        user_prompt += "Available Data (from recent retrieval):\n" + cache_context + "\n\n"
    user_prompt += "Current Question: " + question + "\n\n"
    user_prompt += "Please answer the current question using the available information:"

    return system_prompt + "\n\n" + user_prompt


def _build_cache_context_for_answer(cache: Dict[str, Any]) -> str:
    """Build a detailed context string from cache for answering questions."""
    parts = []
    
    # Include SQL results
    sql_result = cache.get("sql_result")
    if sql_result:
        rows = sql_result.get("rows", [])
        columns = sql_result.get("columns", [])
        
        if rows and columns:
            parts.append(f"Data table with {len(rows)} entries:")
            parts.append(f"Columns: {', '.join(columns)}")
            parts.append("")
            
            # Include all rows (up to a reasonable limit) with numbering
            for i, row in enumerate(rows[:50], 1):
                if isinstance(row, dict):
                    row_items = [f"{k}: {v}" for k, v in row.items()]
                    parts.append(f"Entry {i}: {', '.join(row_items)}")
                elif isinstance(row, (list, tuple)) and columns:
                    row_items = [f"{columns[j]}: {row[j]}" for j in range(min(len(columns), len(row)))]
                    parts.append(f"Entry {i}: {', '.join(row_items)}")
                else:
                    parts.append(f"Entry {i}: {row}")
            
            if len(rows) > 50:
                parts.append(f"... and {len(rows) - 50} more entries")
    
    # Include RAG chunks
    rag_chunks = cache.get("rag_chunks", [])
    rag_meta = cache.get("rag_metadata", [])
    
    if rag_chunks:
        parts.append("")
        parts.append(f"Document excerpts ({len(rag_chunks)} chunks):")
        for i, (chunk, meta) in enumerate(zip(rag_chunks, rag_meta or [{}] * len(rag_chunks)), 1):
            source = meta.get("source", "Unknown source") if meta else "Unknown source"
            parts.append(f"\nExcerpt {i} (from {source}):")
            parts.append(chunk)
    
    return "\n".join(parts)


def _is_calendar_question(question: str) -> bool:
    """Check if the question is about events, calendar, or schedules."""
    calendar_keywords = [
        "event", "events", "happening", "schedule", "calendar", "activity", "activities",
        "this week", "next week", "today", "tomorrow", "weekend", "saturday", "sunday",
        "monday", "tuesday", "wednesday", "thursday", "friday", "what's on", "what is on",
        "going on", "things to do", "community event", "meeting", "workshop"
    ]
    question_lower = question.lower()
    return any(kw in question_lower for kw in calendar_keywords)


_RSS_SOURCE_ALIASES = (
    "dot reporter",
    "dotnews",
    "csndc",
    "codman square neighborhood development corporation",
    "codman square library",
    "bpl codman square",
    "codman square health center",   
    "codman.org",                   
    "codman square neighborhood council", 
    "codman council"
    "south dorchester",                
)

_RSS_NEWS_HINTS = (
    "what's new",
    "whats new",
    "recent news",
    "latest news",
    "news about",
    "news from",
    "updates from",
    "recent updates",
    "what's going on in",
    "whats going on in",
    "what's happening in",
    "whats happening in",
    "what is happening in",
    "what is going on in",
    "lately in",
    "dorchester",
    "south dorchester",
    "codman square",
    "csndc",
    "tell me about",
    "what has",
    "who is",
    "what is",
)
_SCHEDULE_HINTS = (
    "event",
    "events",
    "calendar",
    "schedule",
    "meeting",
    "meetings",
    "workshop",
    "workshops",
    "today",
    "tomorrow",
    "this week",
    "next week",
    "weekend",
    "monday",
    "tuesday",
    "wednesday",
    "thursday",
    "friday",
    "saturday",
    "sunday",
)


def _question_mentions_rss_source(question: str) -> bool:
    question_lower = (question or "").lower()
    if "event" in question_lower:
        return False
    return any(alias in question_lower for alias in _RSS_SOURCE_ALIASES)


def _looks_like_rss_query(question: str) -> bool:
    question_lower = (question or "").lower()
    if _question_mentions_rss_source(question):
        return True
    if any(hint in question_lower for hint in _RSS_NEWS_HINTS):
        return not any(schedule_hint in question_lower for schedule_hint in _SCHEDULE_HINTS)
    return False


def _should_include_rss(question: str, folder_categories: Optional[List[str]]) -> bool:
    normalized = {str(value).strip().lower() for value in (folder_categories or []) if value}
    return "newsletters" in normalized or _looks_like_rss_query(question)


def _retrieve_rag(
    question: str,
    plan: Dict[str, Any],
) -> Tuple[List[str], List[Dict[str, Any]]]:
    """Run all RAG retrieval sources (concurrently when enabled).

    Returns (combined_chunks, combined_meta). Kept separate from answer
    composition so the streaming path can retrieve, then stream the answer.
    """
    k = int(plan.get("k", 5))
    tags = plan.get("transcript_tags")
    sources = plan.get("policy_sources")
    folders = plan.get("folder_categories") if isinstance(plan.get("folder_categories"), list) else None

    combined_chunks: List[str] = []
    combined_meta: List[Dict[str, Any]] = []

    # Each retrieval source is independent, so run them concurrently. Returns
    # (label, chunks, metadata). These mostly wait on network (embeddings) and
    # the vector store, so threads give real concurrency.
    def _task_rss():
        rss_res = retrieval.retrieve_rss(question, k=k)
        return ("📰 RSS", rss_res.get("chunks", []), rss_res.get("metadata", []))

    def _task_boston():
        boston_res = retrieval.retrieve(question, k=k, doc_type="boston_gov_answer")
        return ("🏛️ BostonGov", boston_res.get("chunks", []), boston_res.get("metadata", []))

    def _task_transcripts():
        t_res = retrieval.retrieve_transcripts(question, tags=tags, k=k)
        return ("📝 Transcripts", t_res.get("chunks", []), t_res.get("metadata", []))

    def _task_policies():
        chunks: List[str] = []
        meta: List[Dict[str, Any]] = []
        if sources:
            for src in sources:
                p_res = retrieval.retrieve_policies(question, k=k, source=src)
                chunks.extend(p_res.get("chunks", []))
                meta.extend(p_res.get("metadata", []))
        else:
            p_res = retrieval.retrieve_policies(question, k=k)
            chunks = p_res.get("chunks", [])
            meta = p_res.get("metadata", [])
        return ("📋 Policies", chunks, meta)

    # Fixed order so combined results are deterministic regardless of which task
    # finishes first.
    tasks = []
    if _should_include_rss(question, folders):
        tasks.append(_task_rss)
    tasks.extend([_task_boston, _task_transcripts, _task_policies])

    def _safe_run(fn):
        try:
            return fn()
        except Exception as e:  # noqa: BLE001
            print(f"  ⚠️ Retrieval task error: {e}")
            return ("(failed)", [], [])

    if _PARALLEL_RETRIEVAL and len(tasks) > 1:
        with ThreadPoolExecutor(max_workers=len(tasks)) as executor:
            results = list(executor.map(_safe_run, tasks))
    else:
        results = [_safe_run(fn) for fn in tasks]

    for label, chunks, meta in results:
        print(f"  {label}: {len(chunks)} chunks found")
        combined_chunks.extend(chunks)
        combined_meta.extend(meta)

    return combined_chunks, combined_meta


def _run_rag(
    question: str,
    plan: Dict[str, Any],
    conversation_history: Optional[List[Dict[str, str]]] = None,
    apply_fallback: bool = True,
) -> Dict[str, Any]:
    combined_chunks, combined_meta = _retrieve_rag(question, plan)
    answer = _compose_rag_answer(question, combined_chunks, combined_meta, conversation_history)
    if apply_fallback:
        answer = _apply_default_fallback_if_needed(question, answer)
    return {"answer": answer, "chunks": combined_chunks, "metadata": combined_meta}


def _run_sql(
    question: str,
    conversation_history: Optional[List[Dict[str, str]]] = None,
    apply_fallback: bool = True,
) -> Dict[str, Any]:
    # Import app4 (MySQL) only when SQL path is actually used
    import sql_chat.app4 as app4  # noqa: WPS433

    if app4.is_generic_events_list_question(question):
        try:
            out = app4.run_generic_events_query(question, conversation_history)
            if apply_fallback:
                out["answer"] = _apply_sql_fallback_if_needed(question, out.get("answer", ""))
            return out
        except Exception as exc:  # noqa: BLE001
            print(f"  ⚠️ Fast events path failed ({exc}); falling back to LLM SQL")

    database = os.environ.get("PGSCHEMA", "public")
    schema = app4._fetch_schema_snapshot(database)
    # Base metadata from catalog selection
    metadata = app4._build_question_metadata(question)
    # Strongly encourage maps for location-related queries and many data queries
    location_keywords = ["map", "maps", "where", "location", "locations", "hotspot", "cluster", "show on a map", "geo", "geography", "near", "place", "places", "area", "neighborhood", "neighborhoods"]
    data_visualization_keywords = ["show", "display", "visualize", "see", "find", "list"]
    question_lower = (question or "").lower()
    want_map = any(w in question_lower for w in location_keywords) or any(w in question_lower for w in data_visualization_keywords)
    
    # Default to including location when possible
    if metadata:
        try:
            meta_obj = json.loads(metadata)
        except Exception:
            meta_obj = {}
        hints = (meta_obj.get("hints") if isinstance(meta_obj, dict) else None) or {}
        if want_map:
            hints.update({"need_location": True, "max_points": 500})
        else:
            # Even if not explicitly asked, suggest including location when tables have coordinates
            hints.update({"prefer_location": True, "max_points": 500})
        if isinstance(meta_obj, dict):
            meta_obj["hints"] = hints
        else:
            meta_obj = {"hints": hints}
        try:
            metadata = json.dumps(meta_obj, ensure_ascii=False)
        except Exception:
            pass
    else:
        # Even without metadata, create hints to encourage maps
        try:
            meta_obj = {"hints": {"prefer_location": True, "max_points": 500}}
            metadata = json.dumps(meta_obj, ensure_ascii=False)
        except Exception:
            pass
    try:
        sql = app4._llm_generate_sql(
            question,
            schema,
            os.getenv("GEMINI_MODEL", getattr(app4, "GEMINI_MODEL", GEMINI_MODEL)),
            metadata,
            conversation_history,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"  ⚠️ SQL generation failed: {exc}")
        if app4.is_generic_events_list_question(question):
            out = app4.run_generic_events_query(question, conversation_history)
            if apply_fallback:
                out["answer"] = _apply_sql_fallback_if_needed(question, out.get("answer", ""))
            return out
        return {
            "answer": (
                "I'm having trouble reaching the AI service right now. "
                "Please try again in a moment."
            ),
            "sql": "",
            "result": {"columns": [], "rows": [], "error": str(exc)},
        }
    exec_out = app4._execute_with_retries(
        initial_sql=sql,
        question=question,
        schema=schema,
        metadata=metadata,
    )
    final_sql = exec_out.get("sql", sql)
    result = exec_out.get("result", {})
    try:
        answer = app4._llm_generate_answer(
            question,
            final_sql,
            result,
            os.getenv("GEMINI_SUMMARY_MODEL", getattr(app4, "GEMINI_SUMMARY_MODEL", GEMINI_SUMMARY_MODEL)),
            conversation_history,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"  ⚠️ Answer generation failed: {exc}")
        if app4.is_generic_events_list_question(question):
            answer = app4._format_events_answer_fallback(question, result)
        else:
            answer = (
                "I'm having trouble generating a response right now. "
                "Please try again in a moment."
            )
    if apply_fallback:
        answer = _apply_sql_fallback_if_needed(question, answer)
    return {"answer": answer, "sql": final_sql, "result": result}


def _run_hybrid_parts(
    question: str,
    plan: Dict[str, Any],
    conversation_history: Optional[List[Dict[str, str]]] = None,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Run the SQL and RAG halves of a hybrid query (concurrently when enabled)."""
    # SQL and RAG are independent. SQL uses its own thread-local DB connection
    # and RAG uses the shared (read-only) vector store, so this is safe.
    if _PARALLEL_RETRIEVAL:
        with ThreadPoolExecutor(max_workers=2) as executor:
            sql_future = executor.submit(_run_sql, question, conversation_history, False)
            rag_future = executor.submit(_run_rag, question, plan, conversation_history, False)
            return sql_future.result(), rag_future.result()
    sql_part = _run_sql(question, conversation_history, apply_fallback=False)
    rag_part = _run_rag(question, plan, conversation_history, apply_fallback=False)
    return sql_part, rag_part


def _build_hybrid_merge_prompt(
    question: str,
    sql_part: Dict[str, Any],
    rag_part: Dict[str, Any],
    conversation_history: Optional[List[Dict[str, str]]] = None,
) -> str:
    """Build the hybrid merge prompt that blends the SQL and RAG answers."""
    merge_system = (
        "You are a friendly, non-technical assistant explaining information about DORCHESTER ONLY to a general audience.\n"
        "This system is configured for DORCHESTER ONLY. All data queries are automatically filtered to Dorchester only.\n"
        "Use clear, everyday language and speak as if you are talking directly to the user.\n"
        "You have access to both numeric data (counts, trends, patterns) and contextual information (people's experiences, policy documents, community perspectives).\n\n"
        "Weave these together naturally into a single, cohesive answer that tells a complete story.\n"
        "Blend the numbers with the context so the user understands both what is happening and why it matters.\n"
        "Focus on what the information means for people in Dorchester, not on technical details or data sources.\n"
        "If you see any data from other neighborhoods, ignore it completely and only discuss Dorchester.\n\n"
        "If the inputs do not contain the user's exact answer, the first line of your response must begin with "
        "'I did not find exact information about ...' and briefly name the missing topic. After that first line, "
        "you may give the most relevant related information from the inputs.\n"
        "Do NOT mention SQL, databases, RAG, retrieval, or any internal tools. Just speak as a helpful information bot.\n"
        "Never invent data or trends not present in the inputs."
        + ("\n\nYou are in a conversation. Reference previous questions naturally when it helps the user." if conversation_history else "")
    )
    blob = {
        "sql_answer": sql_part.get("answer"),
        "sql_result": sql_part.get("result"),
        "rag_answer": rag_part.get("answer"),
        "rag_sources": [m.get("source", "?") for m in rag_part.get("metadata", [])][:10],
    }
    merge_user = (
        "Question:\n" + question + "\n\n" +
        "Inputs (JSON):\n" + json.dumps(blob, ensure_ascii=False, default=str)
    )
    full_prompt = merge_system + "\n\n"
    if conversation_history:
        for msg in conversation_history[-10:]:
            role = msg.get("role", "")
            content = msg.get("content", "")
            full_prompt += f"{role.upper()}: {content}\n\n"
    full_prompt += merge_user
    return full_prompt


def _run_hybrid(question: str, plan: Dict[str, Any], conversation_history: Optional[List[Dict[str, str]]] = None) -> Dict[str, Any]:
    sql_part, rag_part = _run_hybrid_parts(question, plan, conversation_history)
    full_prompt = _build_hybrid_merge_prompt(question, sql_part, rag_part, conversation_history)

    client = _get_llm_client()
    model = client.GenerativeModel(GEMINI_MODEL)
    try:
        resp = _generate_content(
            model,
            full_prompt,
            generation_config={"temperature": 0},
        )
        answer = (resp.text or "").strip()
    except Exception:
        answer = (sql_part.get("answer") or "") + "\n\n" + (rag_part.get("answer") or "")

    answer = _apply_default_fallback_if_needed(question, answer)
    return {"answer": answer, "sql": sql_part, "rag": rag_part}


def _stream_model_text(model_name: str, prompt: str, temperature: float):
    """Yield text deltas from a streaming Gemini generation. Defensive on errors."""
    client = _get_llm_client()
    model = client.GenerativeModel(model_name)
    try:
        try:
            stream = model.generate_content(
                prompt,
                generation_config={"temperature": temperature},
                stream=True,
                request_options={"timeout": GEMINI_REQUEST_TIMEOUT},
            )
        except TypeError:
            stream = model.generate_content(
                prompt,
                generation_config={"temperature": temperature},
                stream=True,
            )
    except Exception as exc:  # noqa: BLE001
        print(f"  ⚠️ Streaming generation failed to start: {exc}")
        return
    for chunk in stream:
        try:
            piece = getattr(chunk, "text", "") or ""
        except Exception:  # noqa: BLE001
            piece = ""
        if piece:
            yield piece


def _pseudo_stream(text: str):
    """Emit an already-computed answer in word chunks for progressive rendering."""
    if not text:
        return
    words = text.split(" ")
    for i, word in enumerate(words):
        yield word if i == 0 else " " + word


def stream_agent_response(
    message: str,
    conversation_history: Optional[List[Dict[str, str]]] = None,
    retrieval_cache: Optional[Dict[str, Any]] = None,
):
    """Stream an agent answer.

    Yields ('delta', text) chunks as the answer is produced, then exactly one
    ('final', payload) where payload = {answer, mode, result, retrieval_cache}.

    This mirrors `_execute_agent_response`'s routing/caching decisions but
    streams the final generation. The RAG / hybrid / history answers are truly
    token-streamed from Gemini; SQL answers are computed then emitted
    progressively. The Boston.gov post-hoc fallback is intentionally skipped
    here (it requires inspecting the complete answer first).
    """
    cache = retrieval_cache or create_empty_cache()

    if is_small_talk(message):
        answer = small_talk_response(message)
        yield ("delta", answer)
        yield ("final", {
            "answer": answer,
            "mode": "chitchat",
            "result": {"answer": answer},
            "retrieval_cache": cache,
        })
        return

    has_history = bool(conversation_history)
    has_cache = bool(cache and cache.get("mode"))

    if has_history or has_cache:
        history_check = _check_if_needs_new_data(message, conversation_history, cache)
    else:
        history_check = {"needs_new_data": True}

    # Follow-up that can be answered from history/cache.
    if not history_check.get("needs_new_data", True) and (has_history or has_cache):
        prompt = _build_history_prompt(message, conversation_history, cache)
        if prompt is None:
            answer = "I don't have any previous conversation or data to reference. Could you ask your question again?"
            yield ("delta", answer)
        else:
            acc: List[str] = []
            for piece in _stream_model_text(GEMINI_MODEL, prompt, 0.3):
                acc.append(piece)
                yield ("delta", piece)
            answer = "".join(acc).strip()
            if not answer:
                answer = "I encountered an error answering from the available information. Could you rephrase your question?"
                yield ("delta", answer)
        yield ("final", {"answer": answer, "mode": "history", "result": {"answer": answer}, "retrieval_cache": cache})
        return

    plan = _route_question(message)
    mode = plan.get("mode", "hybrid")

    if mode == "sql":
        try:
            out = _run_sql(message, conversation_history)
        except Exception as exc:  # noqa: BLE001
            print(f"  ⚠️ SQL pipeline failed: {exc}")
            answer = (
                "I couldn't load that information right now. "
                "Please try again in a moment."
            )
            yield ("delta", answer)
            yield ("final", {
                "answer": answer,
                "mode": "sql",
                "result": {"answer": answer, "error": str(exc)},
                "retrieval_cache": cache,
            })
            return
        answer = out.get("answer", "")
        for piece in _pseudo_stream(answer):
            yield ("delta", piece)
        next_cache = build_retrieval_cache(
            mode="sql", question=message, answer=answer,
            sql_result=out.get("result"), sql_query=out.get("sql"),
        )
        yield ("final", {"answer": answer, "mode": "sql", "result": out, "retrieval_cache": next_cache})
        return

    if mode == "rag":
        chunks, meta = _retrieve_rag(message, plan)
        prompt, context_parts = _build_rag_prompt(message, chunks, meta, conversation_history)
        if prompt is None:
            answer = "No relevant information found."
            yield ("delta", answer)
        else:
            acc = []
            for piece in _stream_model_text(GEMINI_MODEL, prompt, 0.3):
                acc.append(piece)
                yield ("delta", piece)
            answer = "".join(acc).strip()
            if not answer:
                answer = "\n\n".join(context_parts[:10])
                yield ("delta", answer)
        result = {"answer": answer, "chunks": chunks, "metadata": meta}
        next_cache = build_retrieval_cache(
            mode="rag", question=message, answer=answer,
            rag_chunks=chunks, rag_metadata=meta,
        )
        yield ("final", {"answer": answer, "mode": "rag", "result": result, "retrieval_cache": next_cache})
        return

    # hybrid
    sql_part, rag_part = _run_hybrid_parts(message, plan, conversation_history)
    prompt = _build_hybrid_merge_prompt(message, sql_part, rag_part, conversation_history)
    acc = []
    for piece in _stream_model_text(GEMINI_MODEL, prompt, 0):
        acc.append(piece)
        yield ("delta", piece)
    answer = "".join(acc).strip()
    if not answer:
        answer = (sql_part.get("answer") or "") + "\n\n" + (rag_part.get("answer") or "")
        yield ("delta", answer)
    result = {"answer": answer, "sql": sql_part, "rag": rag_part}
    next_cache = build_retrieval_cache(
        mode="hybrid", question=message, answer=answer,
        sql_result=sql_part.get("result"), sql_query=sql_part.get("sql"),
        rag_chunks=rag_part.get("chunks"), rag_metadata=rag_part.get("metadata"),
    )
    yield ("final", {"answer": answer, "mode": "hybrid", "result": result, "retrieval_cache": next_cache})


def apply_post_stream_fallback(question: str, answer: str, mode: str) -> str:
    """Apply Boston.gov fallback after streaming completes (mirrors non-streaming path)."""
    if not answer:
        return answer
    if mode == "sql":
        return _apply_sql_fallback_if_needed(question, answer)
    if mode in {"rag", "hybrid", "history"}:
        return _apply_default_fallback_if_needed(question, answer)
    return answer


def _ensure_gemini_ready() -> None:
    if not os.getenv("GEMINI_API_KEY"):
        raise SystemExit("GEMINI_API_KEY not configured")


def main() -> None:
    _bootstrap_env()
    _fix_retrieval_vectordb_path()
    _ensure_gemini_ready()

    print("\nUnified SQL + RAG Chatbot (type 'exit' to quit)\n")
    while True:
        try:
            question = input("Question> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not question:
            continue
        if question.lower() in {"exit", "quit", ":q", "q"}:
            break

        plan = _route_question(question)
        mode = plan.get("mode", "rag")
        
        # Print the routing plan
        print(f"\n🧭 Routing Plan: {json.dumps(plan, indent=2)}\n")

        try:
            if mode == "sql":
                # Validate DB env only when needed
                if not os.environ.get("DATABASE_URL"):
                    print("DATABASE_URL not set; falling back to RAG.")
                    out = _run_rag(question, plan)
                    print("\nAnswer:\n" + out.get("answer", ""))
                else:
                    out = _run_sql(question)
                    print("\nAnswer:\n" + out.get("answer", ""))
            elif mode == "hybrid":
                if not os.environ.get("DATABASE_URL"):
                    print("DATABASE_URL not set; running RAG only.")
                    out = _run_rag(question, plan)
                    print("\nAnswer:\n" + out.get("answer", ""))
                else:
                    out = _run_hybrid(question, plan)
                    print("\nAnswer:\n" + out.get("answer", ""))
            else:  # rag
                out = _run_rag(question, plan)
                print("\nAnswer:\n" + out.get("answer", ""))
        except Exception as exc:  # noqa: BLE001
            print(f"Error: {exc}")


if __name__ == "__main__":
    main()
