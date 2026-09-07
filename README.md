# IP-SAKTI Sahayak

**AI-assisted legal and regulatory guidance for Ayurveda, traditional knowledge, and Indian IP law.**

Built for Smart India Hackathon 2026 — Problem Statement **SIH26045**.

---

## The problem

An Ayurveda entrepreneur or small manufacturer building a new formulation has no
easy way to know, before they spend money finding out the hard way:

- Is this product patentable, or does it already belong to public traditional
  knowledge?
- Is it a "new drug" that needs government approval before it can be sold —
  and if so, which approval pathway?
- If it's made from an Indian plant or biological material, is there a
  separate government clearance required before *any* patent filing,
  anywhere in the world?

Generic AI chatbots make this worse, not better: a Stanford RegLab study
found general-purpose LLMs give hallucinated or unreliable answers to legal
questions **58–88% of the time**. In this domain, a wrong answer isn't a bad
UX — it's a rejected patent application or, in the biodiversity-law case, a
criminal penalty.

**IP-SAKTI Sahayak** is the complementary gap to India's existing
Traditional Knowledge Digital Library (TKDL): TKDL protects patent
**examiners** from granting bad patents by giving them prior art to cite.
This tool helps the **innovator** understand their own legal position
*before* an examiner or a lawyer is ever involved.

---

## What it actually does

Four layers, not one chatbot call:

1. **Classifier** — a transparent, hand-written decision tree (not a
   black-box model) that sorts any formulation into one of six legal
   categories, purely from structured yes/no answers. The classification
   decision is never made by an LLM — only free-text disambiguation, when
   genuinely needed, feeds into the deterministic branching.
2. **Grounded retrieval (RAG)** — answers free-text legal questions using
   hybrid dense + keyword search over the *actual* statutory text, never
   from the model's memory. Every claim is expected to trace back to a
   real, cited section.
3. **Jurisdiction filter** — India vs. international law is enforced as a
   hard `WHERE` clause on the retrieval query, not a prompt instruction an
   LLM could ignore.
4. **Trust layer** — every answer carries a real retrieval-confidence score
   and a citation list; when confidence is too low, the system says so and
   points to a registered patent agent instead of guessing. Every query is
   logged to an audit trail.

---

## The six-bucket classifier

Every Ayurveda formulation is routed into exactly one of these, based on
four structured questions (a fifth free-text question appears only when
needed to disambiguate):

| Category | Trigger | Legal pathway |
|---|---|---|
| `classical_asu` | Follows a named classical text exactly (Charaka Samhita, etc.) | Section 3(a), D&C Act — no DCGI pre-approval needed |
| `patent_proprietary_asu` | Disease claim, classical base with a new combination/indication | Section 3(a)/(h), D&C Act — needs Rule 158B pilot-study proof |
| `phytopharmaceutical` | Disease claim, non-classical, description matches purified/standardized-extract markers | CDSCO Phytopharmaceutical guidelines — full new-drug pathway |
| `cosmetic` | External use, no disease claim | Cosmetic Rules — lighter regime, no GMP requirement |
| `nutraceutical` | Internal use, wellness claim, no disease claim | FSSAI jurisdiction, not CDSCO/AYUSH |
| `unclassified` | Answers don't cleanly resolve, or a disambiguating description is missing | System explicitly abstains rather than guess, and asks a follow-up |

**Cross-cutting flag (`bda_flag`):** independent of category, if the
formulation uses a biological resource sourced from India, the system flags
that **National Biodiversity Authority clearance under BDA Section 6 is
required before any patent filing, in India or abroad** — skipping it
carries a penalty of up to 5 years' imprisonment or a ₹10 lakh fine. This is
the branch most competing tools miss because they stop at "is it
patentable" and never ask where the raw material came from.

The classifier also returns `needs_review`, and when a free-text keyword
match drove the decision, the exact `matched_keyword` and surrounding
`review_snippet` — so a human can verify an edge-case result in seconds
instead of re-reading the whole submission.

---

## How the grounded answer engine works

- **Retrieval:** hybrid dense (embedding cosine similarity) + BM25 keyword
  search over the ingested statute chunks, fused with reciprocal rank
  fusion, filtered by jurisdiction and (when known) formulation category at
  the SQL level before anything reaches the language model.
- **Chunking:** statutes are split at section/sub-section boundaries, never
  by arbitrary token windows — a citation that quotes half of a section is
  treated as worse than no citation at all.
- **Generation:** Groq (`llama-3.3-70b-versatile`, with an automatic
  fallback to `llama-3.1-8b-instant` on repeated failures) answers strictly
  from the retrieved chunks.
- **Confidence:** a real number computed from retrieval agreement — not a
  number the model makes up about its own certainty. Below the configured
  threshold, the system abstains and recommends a registered patent agent
  or AYUSH-recognized IP cell instead of answering.
- **Caching:** embeddings and full answers are cached (single-flight, so
  concurrent identical queries don't trigger duplicate embedding calls),
  since legal Q&A has heavy query overlap in practice.
- **Every response** carries a fixed disclaimer: *"This is not legal
  advice. Consult a registered patent agent or AYUSH-recognized IP cell."*

---

## Tech stack

| Layer | Choice |
|---|---|
| API framework | FastAPI + Uvicorn |
| Database | PostgreSQL with `pgvector`, hosted on Supabase |
| Embeddings | `sentence-transformers` (`BAAI/bge-small-en-v1.5`, 384-dim) |
| Keyword search | `rank-bm25` |
| LLM generation | Groq (`llama-3.3-70b-versatile`, primary) |
| Async DB driver | `asyncpg` |
| Validation | Pydantic |
| Frontend | Static HTML/CSS/vanilla JS, no build step, no framework |

---

## Project structure

```
ip-sakti-backend/
├── app/
│   ├── main.py                 # FastAPI app, routes, CORS, rate limiting
│   ├── config.py                # env-driven settings, fails fast if misconfigured
│   ├── rag.py                    # orchestrates retrieval -> generation -> confidence
│   ├── ratelimit.py               # fixed-window rate limiter for /query
│   ├── cache.py                    # TTL cache with single-flight dedup
│   ├── classifier/
│   │   └── classifier.py            # the six-bucket decision tree
│   ├── retrieval/
│   │   ├── embeddings.py              # dense embedding model wrapper
│   │   ├── bm25.py                     # keyword search index
│   │   ├── hybrid.py                    # RRF fusion + jurisdiction/category filters
│   │   └── db.py                         # asyncpg pool + pgvector codec registration
│   ├── llm/
│   │   └── groq_client.py                 # Groq call with model fallback ladder
│   ├── audit/
│   │   └── logger.py                       # writes/reads audit_log
│   ├── ingestion/
│   │   ├── chunker.py                       # section-boundary-aware chunking
│   │   ├── ingest.py                         # embeds + inserts the corpus
│   │   └── corpus/                            # raw statute text files
│   └── models/
│       └── schemas.py                          # all Pydantic request/response models
├── frontend/
│   ├── config.js                                # API_BASE + shared helpers
│   ├── index.html                                # landing + jurisdiction selector
│   ├── wizard.html                                # 4(-5) step classifier flow
│   ├── result.html                                 # classification result screen
│   ├── query.html                                   # grounded Q&A + citations
│   └── audit.html                                    # audit log viewer
├── schema.sql                                          # Postgres/pgvector schema
├── requirements.txt
└── .env.example
```

---

## API reference

### `POST /classify`
```json
// Request
{
  "internal_or_external": "internal",
  "follows_classical_text_exactly": true,
  "makes_disease_claim": false,
  "uses_indian_biological_resource": true,
  "free_text_description": null
}
```
```json
// Response
{
  "category": "classical_asu",
  "legal_pathway": "Classical ASU drug, Section 3(a) D&C Act — ...",
  "bda_flag": true,
  "reasoning": "Made exactly per an authoritative classical text ...",
  "needs_review": false,
  "matched_keyword": null,
  "review_snippet": null
}
```

### `POST /query`
```json
// Request
{
  "query": "Can I patent a modified turmeric extract with a new ratio?",
  "jurisdiction": "india",
  "classification": null
}
```
```json
// Response
{
  "answer": "...",
  "citations": [
    {
      "act_name": "The Patents Act, 1970",
      "section_ref": "Section 3(p)",
      "jurisdiction": "india",
      "source_url": null,
      "chunk_text": "an invention which in effect is traditional knowledge ...",
      "score": 0.87
    }
  ],
  "confidence": 0.81,
  "abstained": false,
  "disclaimer": "This is not legal advice. Consult a registered patent agent or AYUSH-recognized IP cell."
}
```

### `GET /audit/recent?limit=20`
Returns the most recent logged queries: `query`, `classification_branch`,
`jurisdiction`, `confidence_score`, `abstained`, `created_at`.

### Operational endpoints
- `GET /health` — liveness
- `GET /health/ready` — readiness (checks DB connection + embedding model loaded)
- `GET /admin/cache-stats` — embedding/answer cache hit rates
- `POST /admin/reindex-bm25` — clears the in-memory BM25 index (call after re-ingesting)

---

## Corpus currently ingested

| Act | Sections ingested | Jurisdiction |
|---|---|---|
| The Patents Act, 1970 | Full text, as amended till 01-08-2024 | India |
| The Drugs and Cosmetics Act, 1940 | Full text | India |
| The Biological Diversity Act, 2002 | Full text | India |

Statute text is chunked at section boundaries and cross-checked against the
official IPIndia gazette PDF, not reconstructed from model memory.

---

## Setup

```bash
git clone <this-repo>
cd ip-sakti-backend
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # fill in DATABASE_URL (Supabase session-pooler URI) and GROQ_API_KEY
```

Run the schema once in the Supabase SQL editor (`schema.sql`), then:

```bash
python -m app.ingestion.ingest     # embeds and loads the corpus
uvicorn app.main:app --reload      # starts the API on :8000
```

Serve the frontend separately (any static file server, e.g.
`python -m http.server` from `frontend/`) and point `frontend/config.js`
at your running backend URL.

---

## Honest project status

This project treats an inflated completion percentage as a bigger
credibility risk than an honest gap — the whole point of the product is
trustworthy disclosure, so the README holds itself to that standard too.

**Working end-to-end today:**
- Classifier (all six categories, tested against the real decision logic)
- Grounded retrieval + generation against the ingested India corpus
- Confidence scoring and abstention
- Audit logging
- Frontend wired to the real API contract (no fabricated data anywhere in
  the UI — the "show in plain English" citation feature is deliberately
  disabled rather than faked, since it isn't backed by a real LLM call yet)

---

## Team : code4life, Manipal University Jaipur
