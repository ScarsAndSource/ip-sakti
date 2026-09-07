import logging

import asyncpg
from pgvector import Vector
from pgvector.asyncpg import register_vector

from app.config import settings

logger = logging.getLogger(__name__)


async def get_pool() -> asyncpg.Pool:
    # `init` runs on every connection the pool opens (including connections
    # opened later to replace a dropped one) -- registering per-connection
    # here is the only place guaranteed to fire before that connection is
    # ever handed out with a `vector` parameter.  Registering once on a
    # single connection and assuming it "sticks" for the whole pool is the
    # mistake that makes this bug intermittent rather than obvious: it works
    # on whichever connection you tested with and silently fails on the next
    # one the pool hands out.
    #
    # We also keep the ivfflat.probes setting from the previous version:
    # it controls ANN recall vs speed at query time and should be set on
    # every connection for the same reason as register_vector.
    async def _init_connection(conn: asyncpg.Connection) -> None:
        await register_vector(conn)
        await conn.execute(f"SET ivfflat.probes = {settings.PGVECTOR_PROBES}")

    return await asyncpg.create_pool(
        settings.DATABASE_URL,
        min_size=settings.DB_POOL_MIN_SIZE,
        max_size=settings.DB_POOL_MAX_SIZE,
        command_timeout=settings.DB_COMMAND_TIMEOUT_SECONDS,
        init=_init_connection,
    )


async def search_chunks(
    pool: asyncpg.Pool,
    query_embedding: list[float],
    jurisdiction: str,
    category: str | None,
    top_k: int,
) -> list[asyncpg.Record]:
    """Jurisdiction (and optionally category) is a hard SQL WHERE filter, not
    a prompt instruction -- an India-mode query must be structurally unable to
    retrieve international-mode chunks regardless of what the LLM does."""
    if top_k < 1:
        raise ValueError(f"top_k must be >= 1, got {top_k}")

    # A normal Python list is encoded by asyncpg as ``double precision[]``,
    # not pgvector's ``vector`` type. PostgreSQL then rejects ``vector <=>
    # double precision[]`` at runtime. Wrap the embedding explicitly so this
    # is a vector parameter on every supported pgvector/asyncpg version.
    filters = ["c.jurisdiction = $2"]
    params: list = [Vector(query_embedding), jurisdiction]
    next_param = 3
    if category:
        filters.append(f"c.category = ${next_param}")
        params.append(category)
        next_param += 1

    top_k_param = next_param
    params.append(top_k)

    where_clause = " AND ".join(filters)
    sql = f"""
        select c.id, c.chunk_text, c.section_ref, c.jurisdiction, c.category,
               s.act_name, s.source_url,
               1 - (c.embedding <=> $1) as score
        from chunks c
        join statutes s on s.id = c.statute_id
        where {where_clause}
        order by c.embedding <=> $1
        limit ${top_k_param}
    """
    async with pool.acquire() as conn:
        rows = await conn.fetch(sql, *params)
    return rows


async def fetch_all_chunks_for_bm25(
    pool: asyncpg.Pool,
    jurisdiction: str,
    category: str | None,
) -> list[dict]:
    """Pulls every chunk in scope (same jurisdiction/category filter as
    search_chunks) so hybrid.py builds a BM25 index over exactly the set
    dense search is allowed to return from -- never a broader corpus than
    the hard filter permits."""
    filters = ["c.jurisdiction = $1"]
    params: list = [jurisdiction]
    if category:
        filters.append("c.category = $2")
        params.append(category)

    where_clause = " AND ".join(filters)
    sql = f"""
        select c.id, c.chunk_text, c.section_ref, c.jurisdiction, c.category,
               s.act_name, s.source_url
        from chunks c
        join statutes s on s.id = c.statute_id
        where {where_clause}
    """
    async with pool.acquire() as conn:
        rows = await conn.fetch(sql, *params)
    return [dict(r) for r in rows]


async def get_corpus_stats(pool: asyncpg.Pool) -> dict:
    """Return a small, non-sensitive deployment diagnostic."""
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            "select (select count(*) from statutes) as statutes, "
            "(select count(*) from chunks) as chunks"
        )
    return dict(row)
