"""
Run with: python -m app.ingestion.ingest

Drop plain-text statute files into app/ingestion/corpus/ and list them in
CORPUS_MANIFEST below. This is the actual bottleneck of the whole build -
see README. Do not let this script's simplicity fool you into thinking corpus
prep is quick; sourcing and cleaning the statute text is the real work.
"""

import asyncio
from pathlib import Path

from app.ingestion.chunker import chunk_by_section
from app.retrieval.db import get_pool
from app.retrieval.embeddings import embed_batch

CORPUS_MANIFEST = [
    ("patents_act_1970.txt", "Patents Act, 1970", "india", "patent_law", None),
    ("drugs_and_cosmetics_act_1940.txt", "Drugs & Cosmetics Act, 1940", "india", "asu_regulatory", None),
    ("biological_diversity_act_2002.txt", "Biological Diversity Act, 2002", "india", "biodiversity", None),
]


async def ingest_file(pool, filepath: Path, act_name: str, jurisdiction: str, category: str, source_url):
    full_text = filepath.read_text(encoding="utf-8")
    chunks = chunk_by_section(full_text, act_name)  # raises before we touch the DB if malformed

    # Optimization: embed all chunks for this statute in one batched call
    # instead of one model.encode() per chunk in a loop.
    vectors = embed_batch([c["chunk_text"] for c in chunks])

    async with pool.acquire() as conn:
        async with conn.transaction():
            statute_id = await conn.fetchval(
                """insert into statutes (act_name, jurisdiction, category, source_url, full_text)
                   values ($1, $2, $3, $4, $5) returning id""",
                act_name,
                jurisdiction,
                category,
                source_url,
                full_text,
            )
            # executemany batches the inserts into far fewer round trips.
            # Wrapped in the same transaction as the statute insert above,
            # so a failure partway through doesn't leave an orphaned
            # statute row with only some of its chunks.
            rows = [
                (statute_id, jurisdiction, category, c["section_ref"], c["chunk_text"], str(vec))
                for c, vec in zip(chunks, vectors)
            ]
            await conn.executemany(
                """insert into chunks (statute_id, jurisdiction, category, section_ref, chunk_text, embedding)
                   values ($1, $2, $3, $4, $5, $6::vector)""",
                rows,
            )
    print(f"Ingested {len(chunks)} chunks from {act_name}")


async def _detect_embedding_dim(pool) -> int | None:
    """Return the vector dimension stored in the chunks table, or None if the
    table is empty or the embedding column is null."""
    async with pool.acquire() as conn:
        # vector_dims() is a pgvector helper that works on any vector column.
        raw = await conn.fetchval(
            "select vector_dims(embedding) from chunks where embedding is not null limit 1"
        )
    return int(raw) if raw is not None else None


async def ensure_corpus(pool) -> int:
    """Populate the bundled statutes exactly once for an empty database.

    Hosting the API without separately running the ingestion command leaves
    retrieval empty, so every request abstains and Groq is never invoked.
    This makes a fresh deployment self-contained while preserving any
    existing indexed corpus unchanged.

    Dimension mismatch guard
    ------------------------
    If the chunks table is non-empty but the stored embedding dimension
    doesn't match the live model (e.g., because the corpus was ingested by
    the old sentence-transformers backend and the service was later switched
    to fastembed), the cosine scores will be meaningless and every query will
    abstain.  We detect this and wipe + re-ingest automatically so the
    embedding space is always consistent without requiring a manual
    ``truncate table chunks`` command.
    """
    # Derive the expected dimension from the live model.
    from app.retrieval.embeddings import embed as _embed
    probe = _embed("dimension probe")
    expected_dim = len(probe)

    async with pool.acquire() as conn:
        existing_chunks = await conn.fetchval("select count(*) from chunks")

    if existing_chunks:
        stored_dim = await _detect_embedding_dim(pool)
        if stored_dim is not None and stored_dim != expected_dim:
            print(
                f"Embedding dimension mismatch: stored={stored_dim}, model={expected_dim}. "
                "Wiping stale corpus and re-ingesting with the current model."
            )
            async with pool.acquire() as conn:
                async with conn.transaction():
                    await conn.execute("truncate table chunks cascade")
                    await conn.execute("truncate table statutes cascade")
        else:
            return existing_chunks

    corpus_dir = Path(__file__).parent / "corpus"
    for filename, act_name, jurisdiction, category, url in CORPUS_MANIFEST:
        filepath = corpus_dir / filename
        if not filepath.exists():
            raise FileNotFoundError(f"Bundled corpus file is missing: {filepath}")
        await ingest_file(pool, filepath, act_name, jurisdiction, category, url)

    async with pool.acquire() as conn:
        return await conn.fetchval("select count(*) from chunks")


async def main():
    pool = await get_pool()
    await ensure_corpus(pool)


if __name__ == "__main__":
    asyncio.run(main())
