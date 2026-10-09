"""Simple online RAG: local TF-IDF retrieval + hosted LLM (Groq by default, or xAI Grok).

Setup:
    pip install -r requirements.txt
    PowerShell: $env:LLM_API_KEY="your_key"
Run:
    python rag.py "your question"        # one-shot
    python rag.py                        # interactive
"""
import glob
import re
import os
import sys

import requests
from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Defaults to Groq (gsk_ keys). For xAI Grok set:
#   LLM_API_URL=https://api.x.ai/v1/chat/completions  LLM_MODEL=grok-4  (key: xai-...)
API_URL = os.getenv("LLM_API_URL", "https://api.groq.com/openai/v1/chat/completions")
MODEL = os.getenv("LLM_MODEL", "openai/gpt-oss-120b")
API_KEY = os.getenv("LLM_API_KEY")
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
    vec = TfidfVectorizer(stop_words="english")
    return chunks, vec, vec.fit_transform(chunks)


def ask(question, chunks, vec, matrix):
    scores = cosine_similarity(vec.transform([question]), matrix)[0]
    context = "\n---\n".join(chunks[i] for i in scores.argsort()[::-1][:TOP_K])
    r = requests.post(
        API_URL,
        headers={"Authorization": f"Bearer {API_KEY}"},
        json={
            "model": MODEL,
            "messages": [
                {"role": "system", "content": "Answer using only the provided context. If the answer is not there, say you don't know."},
                {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"},
            ],
        },
        timeout=120,
    )
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"].strip()


if __name__ == "__main__":
    if not API_KEY:
        sys.exit("Set the LLM_API_KEY environment variable first.")
    chunks, vec, matrix = build_index()
    print(f"Indexed {len(chunks)} chunks.")
    if len(sys.argv) > 1:
        print(ask(" ".join(sys.argv[1:]), chunks, vec, matrix))
    else:
        while (q := input("\nQuestion (empty to quit): ").strip()):
            print(ask(q, chunks, vec, matrix))
