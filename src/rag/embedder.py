"""
Embedding helper for the RAG pipeline.

Switched from the old hash trick to sentence-transformers (all-MiniLM-L6-v2).
MiniLM is small, fast, runs fine on CPU, and produces actually-useful embeddings
unlike the hash approach which was just a placeholder.

Falls back to hash-based embeddings if sentence-transformers isn't installed,
so tests still pass in minimal environments.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class EmbedConfig:
    model_name: str = "all-MiniLM-L6-v2"
    # dim gets updated automatically once the model loads
    dim: int = 384


class SimpleEmbedder:
    """
    Text embedder backed by sentence-transformers.

    "Simple" is a bit of a misnomer now, but keeping the name
    so nothing breaks. Might rename to CodeEmbedder later if I
    switch to a code-specific model (e.g. codet5-embeddings).
    """

    def __init__(self, cfg: EmbedConfig | None = None):
        self.cfg = cfg or EmbedConfig()
        self._model = None
        self._try_load_model()
        # expose dim at top level for callers
        self.dim = self.cfg.dim

    def _try_load_model(self) -> None:
        try:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.cfg.model_name)
            actual_dim = self._model.get_sentence_embedding_dimension()
            self.cfg.dim = actual_dim
            logger.info(
                f"loaded embedding model '{self.cfg.model_name}' "
                f"(dim={actual_dim})"
            )
        except ImportError:
            logger.warning(
                "sentence-transformers not installed, using hash-based fallback. "
                "Install it with: pip install sentence-transformers"
            )
            self._model = None
            self.cfg.dim = 256  # hash-based fallback dim

    def _hash_embed(self, text: str) -> np.ndarray:
        # old fallback - deterministic but not semantically meaningful
        dim = self.cfg.dim
        if not text:
            return np.zeros(dim, dtype="float32")
        toks = text.split()
        vecs = []
        for tok in toks:
            h = hashlib.sha256(tok.encode("utf-8")).digest()
            raw = (h * ((dim // len(h)) + 1))[:dim]
            arr = np.frombuffer(raw, dtype=np.uint8).astype("float32")
            vecs.append((arr / 127.5) - 1.0)
        mat = np.stack(vecs, axis=0)
        out = mat.mean(axis=0)
        norm = np.linalg.norm(out) + 1e-8
        return (out / norm).astype("float32")

    def embed_text(self, text: str) -> np.ndarray:
        if not text:
            return np.zeros(self.cfg.dim, dtype="float32")

        if self._model is not None:
            emb = self._model.encode(
                text,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            return emb.astype("float32")

        return self._hash_embed(text)
