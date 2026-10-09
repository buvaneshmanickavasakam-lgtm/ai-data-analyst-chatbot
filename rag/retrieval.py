from .parser import build_documents
from .chunking import chunk_documents
from .chroma_store import build_store


def build_index(df, profile, embedder):
    return build_store(chunk_documents(build_documents(df, profile)), embedder)


def retrieve(store, question, k=5):
    if store is None:
        return []
    try:
        return store.query(question, k)
    except Exception:
        return []
