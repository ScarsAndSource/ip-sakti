import asyncio
import hashlib

import asyncpg

from app.cache import TTLCache
from app.config import settings
from app.llm.groq_client import generate_answer
from app.retrieval.db import search_chunks
from app.retrieval.embeddings import embed
from app.retrieval.hybrid import hybrid_search

ABSTAIN_MESSAGE = (
    "The corpus doesn't have a clear answer to this specific fact pattern. "
    "Please consult a registered patent agent or AYUSH-recognized IP cell."
)

_embedding_cache = TTLCache(max_size=settings.CACHE_MAX_SIZE, ttl_seconds=settings.CACHE_TTL_SECONDS)
_answer_cache = TTLCache(max_size=settings.CACHE_MAX_SIZE, ttl_seconds=settings.CACHE_TTL_SECONDS)

# Bounds concurrent embedding computation -- see config.py's
# EMBED_CONCURRENCY_LIMIT comment for why unbounded concurrency doesn't
# actually buy throughput on CPU-bound local inference.
_embed_semaphore = asyncio.Semaphore(settings.EMBED_CONCURRENCY_LIMIT)

# Single-flight map for in-progress embedding computations, keyed the same
# way as _embedding_cache. This is NOT the same thing as the semaphore
# above, and conflating them was a real bug caught by testing: a semaphore
# with N permits lets N callers past the cache-check simultaneously before
# any of them has finished and populated the cache, so N (not 1) identical
# concurrent queries all triggered a real encode() call. Verified this with
# 5 concurrent identical queries against EMBED_CONCURRENCY_LIMIT=4 -- 4 real
# computations happened, not 1. A semaphore limits how many things run at
# once; it does not deduplicate "the same thing running more than once".
# This dict is what actually deduplicates: the second-and-later callers for
# the same key await the first caller's in-flight future instead of
# starting their own computation. Safe without a lock because there's no
# `await` between checking this dict and inserting into it -- asyncio only
# switches coroutines at an `await` point, so that check-then-insert is
# atomic with respect to other coroutines on the same event loop.
_embed_in_flight: dict[str, asyncio.Future] = {}


def _answer_cache_key(query: str, jurisdiction: str, category: str | None, previous_query: str | None) -> str:
    raw = f"{jurisdiction}|{category or ''}|{(previous_query or '').strip().lower()}|{query.strip().lower()}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


async def _embed_cached(query: str) -> list[float]:
    key = query.strip().lower()
    cached = _embedding_cache.get(key)
    if cached is not None:
        return cached

    existing = _embed_in_flight.get(key)
    if existing is not None:
        return await existing

    loop = asyncio.get_event_loop()
    fut: asyncio.Future = loop.create_future()
    _embed_in_flight[key] = fut
    try:
        async with _embed_semaphore:
            vector = await asyncio.to_thread(embed, query)
        _embedding_cache.set(key, vector)
        fut.set_result(vector)
        return vector
    except Exception as exc:
        fut.set_exception(exc)
        raise
    finally:
        if _embed_in_flight.get(key) is fut:
            del _embed_in_flight[key]


async def answer_query(
    pool: asyncpg.Pool,
    query: str,
    jurisdiction: str,
    category: str | None = None,
    previous_query: str | None = None,
) -> dict:
    cache_key = _answer_cache_key(query, jurisdiction, category, previous_query)
    cached_answer = _answer_cache.get(cache_key)
    if cached_answer is not None:
        return cached_answer

    # Expand a decontextualised follow-up so the embedding captures the full
    # intent. "What about the penalty?" alone matches nothing; prepending the
    # previous question gives the embedder enough signal to pull the right rows.
    retrieval_query = (
        f"{previous_query.strip()} {query.strip()}"
        if previous_query and previous_query.strip()
        else query
    )
    query_embedding = await _embed_cached(retrieval_query)

    if settings.HYBRID_SEARCH_ENABLED:
        rows = await hybrid_search(
            # ``category`` is a formulation classification (for example,
            # ``classical_asu``), while chunks are tagged with legal-source
            # topics (for example, ``patent_law``). They are not the same
            # vocabulary, so applying it as a SQL filter hides the whole
            # corpus for users who completed the assessment.
            pool, query, query_embedding, jurisdiction, None,
            settings.RETRIEVAL_TOP_K, search_chunks,
        )
    else:
        rows = await search_chunks(pool, query_embedding, jurisdiction, None, settings.RETRIEVAL_TOP_K)

    if not rows:
        result = {
            "answer": ABSTAIN_MESSAGE,
            "citations": [],
            "confidence": 0.0,
            "abstained": True,
            "generation_mode": "abstained",
        }
        _answer_cache.set(cache_key, result)
        return result

    rows = [dict(r) for r in rows]

    top_scores = [s for s in (r.get("score") for r in rows[:3]) if s is not None]
    confidence = sum(top_scores) / len(top_scores) if top_scores else 0.0

    if confidence < settings.CONFIDENCE_THRESHOLD:
        result = {
            "answer": ABSTAIN_MESSAGE,
            "citations": rows,
            "confidence": confidence,
            "abstained": True,
            "generation_mode": "abstained",
        }
        _answer_cache.set(cache_key, result)
        return result

    sources = rows
    answer_text, model_generated = await asyncio.to_thread(generate_answer, query, sources, previous_query)

    result = {
        "answer": answer_text,
        "citations": sources,
        "confidence": confidence,
        "abstained": False,
        "generation_mode": "model" if model_generated else "source_fallback",
    }
    _answer_cache.set(cache_key, result)
    return result
