import re


def chunk_documents(docs, max_chars=500):
    chunks = []
    for d in docs:
        buf = ""
        for sent in re.split(r"(?<=[.!?])\s+", d["text"]):
            if buf and len(buf) + len(sent) > max_chars:
                chunks.append({"source": d["source"], "text": buf.strip()})
                buf = ""
            buf += sent + " "
        if buf.strip():
            chunks.append({"source": d["source"], "text": buf.strip()})
    return chunks
