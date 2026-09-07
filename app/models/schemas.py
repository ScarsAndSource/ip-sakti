from typing import Literal, Optional

from pydantic import BaseModel

Jurisdiction = Literal["india", "international"]

FormulationCategory = Literal[
    "classical_asu",
    "patent_proprietary_asu",
    "phytopharmaceutical",
    "cosmetic",
    "nutraceutical",
    "unclassified",
]


class ClassifierAnswers(BaseModel):
    internal_or_external: Literal["internal", "external"]
    follows_classical_text_exactly: bool
    makes_disease_claim: bool
    uses_indian_biological_resource: bool
    free_text_description: Optional[str] = None


class ClassificationResult(BaseModel):
    category: FormulationCategory
    legal_pathway: str
    bda_flag: bool
    reasoning: str
    needs_review: bool = False
    # Additive fields -- default None so any existing frontend code that
    # doesn't know about them keeps working unchanged (None renders as
    # "nothing to show", not a crash).  Only populated when the
    # phytopharmaceutical-keyword branch fires; a UI can render
    # "Flagged: <keyword>" + snippet when present, and fall back to just
    # `reasoning` otherwise.
    matched_keyword: Optional[str] = None
    review_snippet: Optional[str] = None


class Citation(BaseModel):
    act_name: str
    section_ref: str
    jurisdiction: Jurisdiction
    source_url: Optional[str] = None
    chunk_text: str
    # Optional, not required: hybrid retrieval can surface a chunk that only
    # matched via BM25 (exact-term match), which never computed a dense
    # cosine similarity at all. Making this required broke serialization the
    # moment such a row reached the API response -- confirmed by testing
    # Citation(**row_without_score) directly, not just reasoned about.
    score: Optional[float] = None


class QueryRequest(BaseModel):
    query: str
    jurisdiction: Jurisdiction = "india"
    classification: Optional[ClassificationResult] = None
    # Previous turn's question text (plain string, not an object).
    # When supplied, it is prepended to the current query before embedding so
    # that a decontextualised follow-up ("what about the penalty for that?")
    # retrieves the right corpus rows instead of getting a low-confidence miss.
    previous_query: Optional[str] = None


class QueryResponse(BaseModel):
    answer: str
    citations: list[Citation]
    confidence: float
    abstained: bool
    generation_mode: Literal["model", "source_fallback", "abstained"] = "abstained"
    disclaimer: str = (
        "This is not legal advice. Consult a registered patent agent or "
        "AYUSH-recognized IP cell."
    )
