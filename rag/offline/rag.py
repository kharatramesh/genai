"""Simple offline RAG: Ollama embeddings + Ollama small language model.

Setup:
    ollama pull nomic-embed-text
    ollama pull llama3.2:3b
    pip install -r requirements.txt
Run:
    python rag.py "your question"        # one-shot
    python rag.py                        # interactive
"""
import glob
import re
import sys

import numpy as np
import requests
from pypdf import PdfReader

OLLAMA = "http://localhost:11434"
EMBED_MODEL = "nomic-embed-text"
LLM_MODEL = "qwen3:8b"
DOCS_GLOB = "docs/*.[tp][xd][tf]"  # .txt and .pdf
CHUNK_SIZE, TOP_K = 500, 3


def chunk(text, size=CHUNK_SIZE):
    paras, out, cur = [], [], ""
    for p in text.split("\n\n"):
        p = p.strip()
        # PDF text has no blank lines: break oversized paragraphs at sentence ends
        paras += re.split(r"(?<=[.!?])\s+", p) if len(p) > size else [p] if p else []
    for p in paras:
        if cur and len(cur) + len(p) > size:
            out.append(cur)
            cur = ""
        cur += p + "\n\n"
    if cur:
        out.append(cur.strip())
    return out


def embed(texts):
    r = requests.post(f"{OLLAMA}/api/embed", json={"model": EMBED_MODEL, "input": texts})
    r.raise_for_status()
    v = np.array(r.json()["embeddings"], dtype=np.float32)
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def read_doc(path):
    if path.lower().endswith(".pdf"):
        return "\n\n".join(page.extract_text() or "" for page in PdfReader(path).pages)
    with open(path, encoding="utf-8") as f:
        return f.read()


def build_index():
    chunks = []
    for path in glob.glob(DOCS_GLOB):
        chunks += chunk(read_doc(path))
    if not chunks:
        sys.exit(f"No documents found at {DOCS_GLOB}")
    return chunks, embed(chunks)


def ask(question, chunks, vecs):
    scores = vecs @ embed([question])[0]
    context = "\n---\n".join(chunks[i] for i in np.argsort(-scores)[:TOP_K])
    prompt = (
        "Answer using only the context below. If the answer is not there, say you don't know.\n\n"
        f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"
    )
    r = requests.post(
        f"{OLLAMA}/api/generate",
        json={"model": LLM_MODEL, "prompt": prompt, "stream": False},
    )
    r.raise_for_status()
    return r.json()["response"].strip()


if __name__ == "__main__":
    chunks, vecs = build_index()
    print(f"Indexed {len(chunks)} chunks.")
    if len(sys.argv) > 1:
        print(ask(" ".join(sys.argv[1:]), chunks, vecs))
    else:
        while (q := input("\nQuestion (empty to quit): ").strip()):
            print(ask(q, chunks, vecs))
