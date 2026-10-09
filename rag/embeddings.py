"""Embedding model loader. Uses sentence-transformers when installed, else a dependency-free hashing embedder."""
import re
import zlib
import numpy as np


class HashingEmbedder:
    name = "hashing-384"
    dim = 384

    def embed(self, texts):
        out = np.zeros((len(texts), self.dim), dtype="float32")
        for i, t in enumerate(texts):
            toks = re.findall(r"[a-z0-9]+", t.lower())
            for tok in toks + [a + "_" + b for a, b in zip(toks, toks[1:])]:
                h = zlib.crc32(tok.encode())
                out[i, h % self.dim] += 1.0 if h & 1 else -1.0
        n = np.linalg.norm(out, axis=1, keepdims=True)
        return (out / np.maximum(n, 1e-9)).tolist()


class STEmbedder:
    def __init__(self, model="all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer
        self.model, self.name = SentenceTransformer(model), model

    def embed(self, texts):
        return self.model.encode(texts, normalize_embeddings=True).tolist()


def load_embedding_model(prefer="auto"):
    if prefer in ("auto", "sentence-transformers"):
        try:
            return STEmbedder()
        except Exception:
            pass
    return HashingEmbedder()
