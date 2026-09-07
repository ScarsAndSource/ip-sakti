"""
Embeddings with automatic backend selection:

  HF_TOKEN set  →  HuggingFace Inference API (zero local model files, zero OOM risk)
  HF_TOKEN unset →  fastembed ONNX (local dev / CI, downloads model on first use)

WHY the API path exists
-----------------------
Render's free tier has a hard 512 MB RSS limit.  fastembed downloads five ONNX
model files at runtime and then loads them into memory; that download + load
together exceed the limit and the process is OOM-killed before it can serve a
single request.  The render.yaml buildCommand workaround requires the Render
service to have been created via IaC; services created through the dashboard
ignore the file entirely, so the model is still fetched at startup.

Using the HF Inference API instead means no model files ever touch the
running container — the embedding computation happens on HuggingFace's servers.
The same BAAI/bge-small-en-v1.5 model is used, so existing DB vectors stay
fully compatible (no re-ingestion required after this change).

How to enable on Render
-----------------------
Add a single environment variable in the Render dashboard:
  HF_TOKEN = hf_...   (free account token from huggingface.co/settings/tokens)

BGE query prefix
----------------
BAAI/bge-small-en-v1.5 performs better on asymmetric retrieval when queries
are prepended with the standard instruction.  The HF feature-extraction
endpoint does NOT add it automatically, so we do it here for single-text
embed() calls (used by the query path) but NOT for embed_batch() (used by
the ingestion path, which embeds passage text, not queries).

Public API — unchanged from the previous fastembed version:
    get_model()      -> backend object
    preload_model()  -> None  (no-op for the API path; validates token)
    embed(text)      -> list[float]
    embed_batch(...) -> list[list[float]]
"""

import os
import threading
from typing import Iterable

import numpy as np

from app.config import settings

_HF_TOKEN: str | None = os.environ.get("HF_TOKEN") or None
_CACHE_DIR: str | None = os.environ.get("FASTEMBED_CACHE_PATH") or None

# BGE-small instruction prefix for query-side encoding.
_BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


# ---------------------------------------------------------------------------
# HuggingFace Inference API backend
# ---------------------------------------------------------------------------

class _HFInferenceEmbedder:
    """Wraps the HF feature-extraction inference endpoint with the same
    ``embed()`` interface fastembed exposes so the rest of the codebase
    needs zero changes."""

    def __init__(self, model_name: str, token: str) -> None:
        import requests as _req
        self._session = _req.Session()
        self._session.headers.update({"Authorization": f"Bearer {token}"})
        self._url = (
            f"https://api-inference.huggingface.co"
            f"/pipeline/feature-extraction/{model_name}"
        )

    def embed(self, texts: list[str], **_kw) -> Iterable[np.ndarray]:
        payload = {"inputs": texts, "options": {"wait_for_model": True}}
        resp = self._session.post(self._url, json=payload, timeout=60)
        resp.raise_for_status()
        # API returns list[list[float]] for batch input
        for vec in resp.json():
            yield np.array(vec, dtype=np.float32)


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_model: "_HFInferenceEmbedder | object | None" = None
_model_lock = threading.Lock()


def get_model():
    global _model
    if _model is None:
        with _model_lock:
            if _model is None:
                if _HF_TOKEN:
                    _model = _HFInferenceEmbedder(settings.EMBEDDING_MODEL, _HF_TOKEN)
                else:
                    # Local ONNX fallback — requires fastembed; OOMs on Render
                    # free tier unless FASTEMBED_CACHE_PATH points to pre-built files.
                    from fastembed import TextEmbedding  # type: ignore[import]
                    kwargs: dict = {"model_name": settings.EMBEDDING_MODEL}
                    if _CACHE_DIR:
                        kwargs["cache_dir"] = _CACHE_DIR
                    _model = TextEmbedding(**kwargs)
    return _model


def preload_model() -> None:
    """Call once at startup.  For the API backend this is a no-op (there is
    nothing to download); for the ONNX backend it triggers the one-time
    weight load so the first request doesn't pay that cost."""
    get_model()


def embed(text: str) -> list[float]:
    """Embed a single query string.

    Applies the BGE instruction prefix when using the HF API backend so
    query-side vectors are in the same space as corpus-side vectors (which
    were embedded without a prefix during ingestion).
    """
    model = get_model()
    if isinstance(model, _HFInferenceEmbedder):
        input_text = _BGE_QUERY_PREFIX + text
    else:
        input_text = text
    result: Iterable[np.ndarray] = model.embed([input_text])
    vector = next(iter(result))
    return vector.tolist()


def embed_batch(texts: list[str], batch_size: int = 32) -> list[list[float]]:
    """Batch-embed passage texts (ingestion path — no query prefix).

    For the HF API backend, requests are chunked at ``batch_size`` to stay
    within the endpoint's payload limit.  For the fastembed backend the
    parameter is forwarded as-is (fastembed manages its own internal batching).
    """
    if not texts:
        return []
    model = get_model()

    if isinstance(model, _HFInferenceEmbedder):
        results: list[list[float]] = []
        for i in range(0, len(texts), batch_size):
            chunk = texts[i : i + batch_size]
            results.extend(vec.tolist() for vec in model.embed(chunk))
        return results

    result: Iterable[np.ndarray] = model.embed(texts, batch_size=batch_size)
    return [vec.tolist() for vec in result]
