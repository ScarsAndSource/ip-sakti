import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.audit.logger import get_recent_queries, log_query
from app.classifier.classifier import classify
from app.config import settings
from app.models.schemas import ClassificationResult, ClassifierAnswers, QueryRequest, QueryResponse
from app.rag import _answer_cache, _embedding_cache, answer_query
from app.ratelimit import FixedWindowRateLimiter
from app.retrieval.db import get_pool
from app.retrieval.embeddings import preload_model
from app.retrieval.hybrid import clear_bm25_cache

logger = logging.getLogger(__name__)

pool = None
rate_limiter = FixedWindowRateLimiter(settings.RATE_LIMIT_PER_MINUTE)
_model_ready = False


@asynccontextmanager
async def lifespan(app: FastAPI):
    global pool, _model_ready
    pool = await get_pool()
    preload_model()
    _model_ready = True
    yield
    await pool.close()


app = FastAPI(title="IP-SAKTI Sahayak Backend", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    if request.url.path == "/query":
        client_key = request.client.host if request.client else "unknown"
        allowed, retry_after = rate_limiter.allow(client_key)
        if not allowed:
            return JSONResponse(
                status_code=429,
                content={
                    "detail": "Too many requests. Please wait a moment before trying again.",
                    "retry_after_seconds": retry_after,
                },
                headers={"Retry-After": str(retry_after)},
            )
    return await call_next(request)


@app.post("/classify", response_model=ClassificationResult)
def classify_formulation(answers: ClassifierAnswers):
    return classify(answers)


@app.post("/query", response_model=QueryResponse)
async def query(req: QueryRequest):
    category = None
    if req.classification and req.classification.category != "unclassified":
        category = req.classification.category

    try:
        result = await asyncio.wait_for(
            answer_query(pool, req.query, req.jurisdiction, category, req.previous_query),
            timeout=settings.QUERY_TIMEOUT_SECONDS,
        )
    except asyncio.TimeoutError:
        logger.error("Query timed out after %ss: %r", settings.QUERY_TIMEOUT_SECONDS, req.query)
        raise HTTPException(
            status_code=504,
            detail=(
                f"The query took longer than {settings.QUERY_TIMEOUT_SECONDS}s to answer. "
                "Please try again, or rephrase the question."
            ),
        )

    try:
        await log_query(
            pool,
            req.query,
            req.classification.category if req.classification else None,
            req.jurisdiction,
            result["citations"],
            result["confidence"],
            result["abstained"],
        )
    except Exception:
        logger.exception("Failed to write audit_log entry for query=%r", req.query)

    return QueryResponse(
        answer=result["answer"],
        citations=result["citations"],
        confidence=result["confidence"],
        abstained=result["abstained"],
    )


@app.get("/audit/recent")
async def audit_recent(limit: int = 20):
    limit = min(limit, 100)
    return await get_recent_queries(pool, limit)


@app.post("/admin/reindex-bm25")
async def reindex_bm25():
    cleared = clear_bm25_cache()
    return {"cleared_indexes": cleared}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/health/ready")
async def readiness():
    db_ok = False
    if pool is not None:
        try:
            async with pool.acquire() as conn:
                await conn.fetchval("select 1")
            db_ok = True
        except Exception:
            logger.exception("Readiness check: DB ping failed")

    ready = db_ok and _model_ready
    body = {"ready": ready, "db_ok": db_ok, "model_ready": _model_ready}
    return JSONResponse(status_code=200 if ready else 503, content=body)


@app.get("/admin/cache-stats")
def cache_stats():
    return {"embedding_cache": _embedding_cache.stats(), "answer_cache": _answer_cache.stats()}
