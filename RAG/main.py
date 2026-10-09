"""Minimal RAG API: ingest documents -> embed (Ollama) -> store (Qdrant) -> retrieve -> answer (Ollama).

Endpoints:
  GET  /            test UI
  GET  /status      health of Ollama, models and Qdrant, number of stored chunks
  POST /ingest/samples   ingest the bundled sample documents
  POST /ingest      upload a .txt/.md/.pdf file (multipart field "file")
  POST /query       {"question": "...", "use_rag": true, "top_k": 4}
  DELETE /collection    wipe the vector store
"""
import asyncio
import io
import logging
import os
import re
import uuid
from pathlib import Path

import httpx
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

# ---------------------------------------------------------------- config
# All settings come from environment variables (set in the Kubernetes manifests);
# the defaults point at the in-cluster Services in the "llm" namespace.
OLLAMA = os.getenv("OLLAMA_URL", "http://ollama.llm.svc.cluster.local:11434")   # LLM server (embeddings + chat)
QDRANT = os.getenv("QDRANT_URL", "http://qdrant.llm.svc.cluster.local:6333")    # vector database
EMBED_MODEL = os.getenv("EMBED_MODEL", "nomic-embed-text")  # turns text into vectors
CHAT_MODEL = os.getenv("CHAT_MODEL", "llama3.2:3b")         # generates the final answer
COLLECTION = os.getenv("COLLECTION", "docs")                # Qdrant collection name
SAMPLE_DIR = Path(os.getenv("SAMPLE_DOCS_DIR", "/sample-docs"))  # folder mounted with demo documents
DOCS_DIR = Path(os.getenv("DOCS_DIR", "/docs"))                  # PVC with your own (large) documents
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "800"))       # max characters per chunk
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "100")) # characters shared between neighbouring chunks

log = logging.getLogger("rag")
logging.basicConfig(level=logging.INFO)

# root_path lets the app work behind an ingress that serves it under a sub-path (e.g. /rag).
app = FastAPI(title="RAG demo", root_path=os.getenv("ROOT_PATH", ""))
# One shared async HTTP client for Ollama and Qdrant; long timeout because LLM calls on CPU are slow.
http = httpx.AsyncClient(timeout=httpx.Timeout(600.0))


# ---------------------------------------------------------------- startup
async def pull_models():
    """Make sure both models exist in Ollama (no-op if already pulled)."""
    for model in (EMBED_MODEL, CHAT_MODEL):
        for attempt in range(30):  # up to ~5 minutes of retries per model
            try:
                r = await http.post(f"{OLLAMA}/api/pull", json={"name": model, "stream": False})
                r.raise_for_status()
                log.info("model ready: %s", model)
                break
            except Exception as e:  # Ollama may still be starting
                log.warning("pull %s failed (%s), retry %d", model, e, attempt + 1)
                await asyncio.sleep(10)


@app.on_event("startup")
async def startup():
    # Run in the background so the API starts serving immediately while models download.
    asyncio.create_task(pull_models())


# ---------------------------------------------------------------- helpers
def chunk_text(text: str) -> list[str]:
    """Paragraph-aware chunking with a small character overlap between chunks."""
    # Step 1: split on blank lines into paragraphs; paragraphs longer than CHUNK_SIZE
    # are cut into CHUNK_SIZE pieces that overlap, so no sentence is lost at a cut.
    pieces = []
    for p in re.split(r"\n\s*\n", text):
        p = p.strip()
        while len(p) > CHUNK_SIZE:  # hard-split very long paragraphs
            pieces.append(p[:CHUNK_SIZE])
            p = p[CHUNK_SIZE - CHUNK_OVERLAP:]
        if p:
            pieces.append(p)
    # Step 2: pack consecutive pieces into chunks up to CHUNK_SIZE. When a chunk is full,
    # start the next one with the tail of the previous (overlap) to keep context across boundaries.
    chunks, cur = [], ""
    for p in pieces:
        if cur and len(cur) + len(p) + 2 > CHUNK_SIZE:  # +2 = the "\n\n" separator
            chunks.append(cur)
            cur = cur[-CHUNK_OVERLAP:] + "\n\n" + p
        else:
            cur = f"{cur}\n\n{p}" if cur else p
    if cur:
        chunks.append(cur)
    return chunks


async def embed(texts: list[str]) -> list[list[float]]:
    """Convert texts to embedding vectors via Ollama, 16 texts per request."""
    out = []
    for i in range(0, len(texts), 16):
        r = await http.post(f"{OLLAMA}/api/embed", json={"model": EMBED_MODEL, "input": texts[i:i + 16]})
        if r.status_code != 200:
            raise HTTPException(503, f"Embedding failed (is {EMBED_MODEL} pulled yet?): {r.text[:200]}")
        out.extend(r.json()["embeddings"])
    return out


async def ensure_collection(dim: int):
    """Create the Qdrant collection on first use; `dim` must match the embedding model's output size."""
    r = await http.get(f"{QDRANT}/collections/{COLLECTION}")
    if r.status_code == 200:  # already exists
        return
    r = await http.put(f"{QDRANT}/collections/{COLLECTION}",
                       json={"vectors": {"size": dim, "distance": "Cosine"}})
    r.raise_for_status()


async def ingest_text(source: str, text: str) -> int:
    """Chunk -> embed -> upsert into Qdrant. Returns the number of chunks stored."""
    chunks = chunk_text(text)
    if not chunks:
        return 0
    vectors = await embed(chunks)
    await ensure_collection(len(vectors[0]))
    # Each point = vector + payload (original text and where it came from, used later for citations).
    points = [{
        "id": str(uuid.uuid5(uuid.NAMESPACE_URL, f"{source}:{i}")),  # stable ids: re-ingest overwrites
        "vector": v,
        "payload": {"source": source, "chunk": i, "text": c},
    } for i, (c, v) in enumerate(zip(chunks, vectors))]
    r = await http.put(f"{QDRANT}/collections/{COLLECTION}/points?wait=true", json={"points": points})
    r.raise_for_status()
    return len(points)


async def search(question: str, top_k: int) -> list[dict]:
    """Retrieval step: embed the question and return the top_k most similar chunks (cosine similarity)."""
    [vec] = await embed([question])
    r = await http.post(f"{QDRANT}/collections/{COLLECTION}/points/search",
                        json={"vector": vec, "limit": top_k, "with_payload": True})
    if r.status_code == 404:  # nothing ingested yet
        return []
    r.raise_for_status()
    return [{"source": h["payload"]["source"], "score": round(h["score"], 3), "text": h["payload"]["text"]}
            for h in r.json()["result"]]


async def chat(messages: list[dict]) -> str:
    """Send a chat to Ollama and return the reply text. Low temperature = more factual, less random."""
    r = await http.post(f"{OLLAMA}/api/chat", json={
        "model": CHAT_MODEL, "messages": messages, "stream": False, "options": {"temperature": 0.2}})
    if r.status_code != 200:
        raise HTTPException(503, f"Chat model not ready: {r.text[:200]}")
    return r.json()["message"]["content"]


# ---------------------------------------------------------------- API
class Query(BaseModel):
    """Request body for /query."""
    question: str
    use_rag: bool = True  # False = ask the LLM directly, to compare with RAG
    top_k: int = 4        # how many chunks to retrieve as context


@app.get("/status")
async def status():
    """Report whether Ollama/Qdrant are reachable, which models are pulled, and how many chunks are stored."""
    res = {"ollama": False, "qdrant": False, "models": {}, "chunks": 0}
    try:
        tags = (await http.get(f"{OLLAMA}/api/tags")).json()
        names = [m["name"] for m in tags.get("models", [])]
        res["ollama"] = True
        res["models"] = {m: any(n.startswith(m) for n in names) for m in (EMBED_MODEL, CHAT_MODEL)}
    except Exception as e:
        res["ollama_error"] = str(e)
    try:
        r = await http.get(f"{QDRANT}/collections/{COLLECTION}")
        res["qdrant"] = True
        if r.status_code == 200:
            res["chunks"] = r.json()["result"]["points_count"]
    except Exception as e:
        res["qdrant_error"] = str(e)
    return res


def extract_text(name: str, data: bytes) -> str:
    """PDFs are text-extracted page by page, anything else is read as UTF-8."""
    if name.lower().endswith(".pdf"):
        from pypdf import PdfReader  # imported lazily; only needed for PDFs
        return "\n\n".join(page.extract_text() or "" for page in PdfReader(io.BytesIO(data)).pages)
    return data.decode("utf-8", errors="replace")


@app.post("/ingest/samples")
async def ingest_samples():
    """Ingest every file in SAMPLE_DIR (ConfigMap) and DOCS_DIR (PVC, recursive); skips hidden/'..' entries."""
    files = [p for p in SAMPLE_DIR.glob("*") if p.is_file()]
    files += [p for p in DOCS_DIR.rglob("*") if p.is_file()]
    files = sorted(p for p in files if not any(part.startswith(".") for part in p.parts[-3:]))
    if not files:
        raise HTTPException(404, f"No docs in {SAMPLE_DIR} or {DOCS_DIR}")
    result = {}
    for p in files:
        source = str(p.relative_to(DOCS_DIR)) if DOCS_DIR in p.parents else p.name
        result[source] = await ingest_text(source, extract_text(p.name, p.read_bytes()))
    return {"ingested_chunks": result}


@app.post("/ingest")
async def ingest(file: UploadFile = File(...)):
    """Ingest an uploaded file."""
    text = extract_text(file.filename, await file.read())
    n = await ingest_text(file.filename, text)
    if n == 0:
        raise HTTPException(400, "No text found in file")
    return {"source": file.filename, "chunks": n}


@app.post("/query")
async def query(q: Query):
    """Answer a question, either with RAG or directly from the LLM."""
    if not q.use_rag:  # baseline: no retrieval, the model answers from its own training only
        answer = await chat([{"role": "user", "content": q.question}])
        return {"mode": "plain-llm", "answer": answer, "sources": []}

    # RAG: 1) retrieve relevant chunks, 2) put them in the prompt, 3) let the LLM answer from them.
    hits = await search(q.question, q.top_k)
    if not hits:
        return {"mode": "rag", "answer": "No documents found. Ingest some documents first.", "sources": []}
    # Number the chunks so the model can cite them as [1], [2], ...
    context = "\n\n".join(f"[{i + 1}] (source: {h['source']})\n{h['text']}" for i, h in enumerate(hits))
    system = ("You answer questions using ONLY the context below. If the answer is not in the context, "
              "say you don't know. Cite sources like [1].\n\nContext:\n" + context)
    answer = await chat([{"role": "system", "content": system}, {"role": "user", "content": q.question}])
    return {"mode": "rag", "answer": answer, "sources": hits}


@app.delete("/collection")
async def wipe():
    """Delete the whole Qdrant collection (it is recreated on the next ingest)."""
    r = await http.delete(f"{QDRANT}/collections/{COLLECTION}")
    return {"deleted": r.status_code == 200}


@app.get("/", response_class=HTMLResponse)
async def ui():
    return UI_HTML


# Single-page test UI served at "/". Plain HTML + JS, no build step.
# JS calls use paths relative to the current URL so it works behind an ingress sub-path.
UI_HTML = """<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>RAG demo</title>
<style>
body{font:16px/1.5 system-ui,sans-serif;max-width:800px;margin:2rem auto;padding:0 16px}
textarea,input[type=text]{width:100%;padding:.5rem;font:inherit;box-sizing:border-box}
button{padding:.5rem 1rem;margin:.25rem .25rem .25rem 0;font:inherit;cursor:pointer}
pre{white-space:pre-wrap;background:#f3f5f7;padding:.75rem;border-radius:6px}
.src{border-left:3px solid #0b6e4f;padding-left:.75rem;margin:.5rem 0;font-size:.9rem}
</style></head><body>
<h1>RAG demo</h1>
<p><button onclick="post('ingest/samples')">1. Ingest sample docs</button>
<input type="file" id="f"><button onclick="upload()">Upload file</button>
<button onclick="status()">Status</button></p>
<pre id="log">Ready.</pre>
<h3>Ask a question</h3>
<input type="text" id="q" value="How many days per week can employees work remotely?">
<label><input type="checkbox" id="rag" checked> Use RAG (uncheck to compare with the plain LLM)</label><br>
<button onclick="ask()">Ask</button>
<h3>Answer</h3><pre id="ans"></pre><div id="srcs"></div>
<script>
const base = location.pathname.replace(/\\/+$/, '') + '/';
const $ = id => document.getElementById(id);
async function call(path, opts){ const r = await fetch(base+path, opts); const t = await r.text();
  try { return JSON.parse(t) } catch { return t } }
async function post(p){ $('log').textContent='Working...'; $('log').textContent = JSON.stringify(await call(p,{method:'POST'}),null,2) }
async function status(){ $('log').textContent = JSON.stringify(await call('status'),null,2) }
async function upload(){ const f=$('f').files[0]; if(!f) return; const fd=new FormData(); fd.append('file',f);
  $('log').textContent='Uploading...'; $('log').textContent = JSON.stringify(await call('ingest',{method:'POST',body:fd}),null,2) }
async function ask(){ $('ans').textContent='Thinking...'; $('srcs').textContent='';
  const res = await call('query',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({question:$('q').value,use_rag:$('rag').checked})});
  $('ans').textContent = '['+(res.mode||'error')+'] '+(res.answer||JSON.stringify(res));
  (res.sources||[]).forEach((s,i)=>{ const d=document.createElement('div'); d.className='src';
    d.textContent='['+(i+1)+'] '+s.source+' (score '+s.score+'): '+s.text; $('srcs').appendChild(d) }) }
</script></body></html>"""
