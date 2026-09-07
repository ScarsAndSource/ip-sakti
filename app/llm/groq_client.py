import logging
import time

from groq import Groq

from app.config import settings

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are IP-SAKTI Sahayak, a legal-regulatory assistant for Ayurveda IP and \
regulatory questions in India. You must ONLY answer using the provided source excerpts. Every \
factual claim must cite the exact section/rule reference given in the sources. If the sources do \
not clearly answer the question, say so explicitly and recommend consulting a registered patent \
agent or AYUSH-recognized IP cell. Never invent a section number, and never state a section number \
that is not present verbatim in the provided sources."""

LLM_FAILURE_MESSAGE = (
    "Retrieval found relevant sources, but the answer-generation step failed "
    "(the model backend didn't respond). The citations below are still real "
    "and safe to read directly; please retry the query, or consult a "
    "registered patent agent or AYUSH-recognized IP cell in the meantime."
)


def get_client() -> Groq:
    return Groq(api_key=settings.GROQ_API_KEY, timeout=settings.GROQ_TIMEOUT_SECONDS, max_retries=0)


def _build_context(sources: list[dict]) -> str:
    return "\n\n".join(
        f"[{s['act_name']} - {s['section_ref']}]\n{s['chunk_text']}" for s in sources
    )


def _call_model(model: str, query: str, context: str, previous_query: str | None = None) -> str:
    client = get_client()
    messages: list[dict] = [
        {"role": "system", "content": SYSTEM_PROMPT},
    ]
    # If there is a previous turn, add it as a prior user message so the
    # model understands pronouns / references in the follow-up ("that",
    # "it", "the same act", etc.). We don't persist the previous answer
    # because it may have abstained; re-attaching only the question is
    # enough to resolve the reference.
    if previous_query and previous_query.strip():
        messages.append({"role": "user", "content": previous_query.strip()})
        messages.append({
            "role": "assistant",
            "content": "[Prior turn — see current sources for the full answer.]",
        })
    messages.append({
        "role": "user",
        "content": f"Sources:\n{context}\n\nQuestion: {query}",
    })
    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.1,
    )
    answer = resp.choices[0].message.content
    if not answer or not answer.strip():
        raise ValueError(f"model {model} returned an empty completion")
    return answer


def _extractive_fallback(query: str, sources: list[dict]) -> str:
    """Last resort when every model call fails: hand back the real,
    already-verified citations directly instead of a bare error message."""
    lines = [
        "The answer-generation model is temporarily unavailable, so here are "
        "the most relevant source excerpts directly (unsynthesized) instead "
        "of a generated answer:",
        "",
    ]
    for s in sources[:3]:
        lines.append(f"[{s['act_name']} - {s['section_ref']}]")
        snippet = s["chunk_text"].strip()
        if len(snippet) > 400:
            snippet = snippet[:400].rsplit(" ", 1)[0] + "..."
        lines.append(snippet)
        lines.append("")
    lines.append(
        "Please verify against the full section text and consult a registered "
        "patent agent or AYUSH-recognized IP cell before relying on this."
    )
    return "\n".join(lines)


def generate_answer(query: str, sources: list[dict], previous_query: str | None = None) -> tuple[str, bool]:
    """Return the answer and whether Groq actually generated it.

    The boolean prevents a source-excerpt fallback from being presented to a
    user as though it were a successful LLM response.
    """
    context = _build_context(sources)

    attempts = [(settings.GROQ_MODEL, settings.GROQ_MAX_RETRIES)]
    if settings.GROQ_FALLBACK_MODEL and settings.GROQ_FALLBACK_MODEL != settings.GROQ_MODEL:
        attempts.append((settings.GROQ_FALLBACK_MODEL, 1))

    last_error: Exception | None = None
    for model, max_retries in attempts:
        for attempt in range(1, max_retries + 1):
            try:
                return _call_model(model, query, context, previous_query), True
            except Exception as exc:
                last_error = exc
                logger.warning(
                    "Groq call failed (model=%s, attempt=%d/%d): %s",
                    model, attempt, max_retries, exc,
                )
                if attempt < max_retries:
                    time.sleep(settings.GROQ_RETRY_BACKOFF_SECONDS * attempt)

    logger.error(
        "All Groq models exhausted for query=%r; falling back to extractive answer. Last error: %s",
        query, last_error,
    )
    return _extractive_fallback(query, sources), False
