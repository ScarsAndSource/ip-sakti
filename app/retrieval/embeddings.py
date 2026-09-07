"""
ONNX-based embeddings via fastembed -- no torch, no CUDA, ~50 MB RSS at startup.

Public API is identical to the old sentence-transformers version so main.py,
rag.py, and ingest.py need zero changes:
    get_model()      -> TextEmbedding
    preload_model()  -> None
    embed(text)      -> list[float]
    embed_batch(...) -> list[list[float]]

FASTEMBED_CACHE_PATH (env var, optional)
    If set, fastembed stores / reads its downloaded model files from this
    directory instead of its default ~/.cache/fastembed location.  Set this
    to a path inside the Render workspace (e.g. /opt/render/project/src/.fastembed_cache)
    so that model files downloaded during the Render *build phase* are still
    present when the *run phase* starts -- avoiding a second download that
    would exceed the 512 MB free-tier memory limit.
"""

import os
import threading
from typing import Iterable

import numpy as np
from fastembed import TextEmbedding

from app.config import settings

_model: TextEmbedding | None = None
# Guards first-load only; fastembed's embed() / passage_embed() are stateless
# once the model is loaded, so no lock is needed after that.
_model_lock = threading.Lock()

# Optional override for the fastembed file-cache directory.  Must be read
# once at module import time (before the model is loaded) so that both the
# build-phase preload and the runtime load point to the same path.
_CACHE_DIR: str | None = os.environ.get("FASTEMBED_CACHE_PATH") or None


def get_model() -> TextEmbedding:
    global _model
    # Double-checked locking: avoids lock acquisition on every call once
    # the model is warm (the hot path). Inner check is the safety net for
    # two threads racing past the outer None check simultaneously.
    if _model is None:
        with _model_lock:
            if _model is None:
                kwargs: dict = {"model_name": settings.EMBEDDING_MODEL}
                if _CACHE_DIR:
                    kwargs["cache_dir"] = _CACHE_DIR
                _model = TextEmbedding(**kwargs)
    return _model


def preload_model() -> None:
    """Call once at startup so the ONNX weight download/load cost is paid
    before the first request, not during it."""
    get_model()


def embed(text: str) -> list[float]:
    """Embed a single string and return a normalised float list."""
    model = get_model()
    # embed() returns a generator of numpy arrays; we take the first (and only) one.
    result: Iterable[np.ndarray] = model.embed([text])
    vector = next(iter(result))
    return vector.tolist()


def embed_batch(texts: list[str], batch_size: int = 32) -> list[list[float]]:
    """Batch-embed a list of strings.

    fastembed.embed() already batches internally and streams results as a
    generator of numpy arrays — we just materialise them into a plain list.
    batch_size is accepted for API compatibility but fastembed controls its
    own internal batch size; passing it is a no-op here.
    """
    if not texts:
        return []
    model = get_model()
    result: Iterable[np.ndarray] = model.embed(texts, batch_size=batch_size)
    return [vec.tolist() for vec in result]
