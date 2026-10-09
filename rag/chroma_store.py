"""Vector store: ChromaDB (in-memory) with a NumPy fallback exposing the same interface."""
import uuid
import numpy as np


class NumpyStore:
    backend = "numpy"

    def __init__(self, embedder):
        self.e, self.texts, self.vecs = embedder, [], None

    def add(self, chunks):
        self.texts = [c["text"] for c in chunks]
        self.vecs = np.array(self.e.embed(self.texts))

    def query(self, text, k=5):
        if self.vecs is None or not len(self.texts):
            return []
        sims = self.vecs @ np.array(self.e.embed([text])[0])
        return [self.texts[i] for i in np.argsort(-sims)[:k]]


class ChromaStore:
    backend = "chromadb"

    def __init__(self, embedder):
        import chromadb
        self.e = embedder
        self.client = chromadb.EphemeralClient()
        self.col = self.client.create_collection(f"dataset_{uuid.uuid4().hex[:10]}", metadata={"hnsw:space": "cosine"})

    def add(self, chunks):
        texts = [c["text"] for c in chunks]
        self.col.add(ids=[str(i) for i in range(len(texts))], documents=texts,
                     embeddings=self.e.embed(texts), metadatas=[{"source": c["source"]} for c in chunks])

    def query(self, text, k=5):
        r = self.col.query(query_embeddings=self.e.embed([text]), n_results=min(k, self.col.count()))
        return r["documents"][0]


def build_store(chunks, embedder):
    try:
        s = ChromaStore(embedder)
        s.add(chunks)
        return s
    except Exception:
        s = NumpyStore(embedder)
        s.add(chunks)
        return s
