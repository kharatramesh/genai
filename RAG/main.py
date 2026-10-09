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

OLLAMA = os.getenv("OLLAMA_URL", "http://ollama.llm.svc.cluster.local:11434")
QDRANT = os.getenv("QDRANT_URL", "http://qdrant.llm.svc.cluster.local:6333")
EMBED_MODEL = os.getenv("EMBED_MODEL", "nomic-embed-text")
CHAT_MODEL = os.getenv("CHAT_MODEL", "llama3.2:3b")
COLLECTION = os.getenv("COLLECTION", "docs")
SAMPLE_DIR = Path(os.getenv("SAMPLE_DOCS_DIR", "/sample-docs"))
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "800"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "100"))

log = logging.getLogger("rag")
logging.basicConfig(level=logging.INFO)

app = FastAPI(title="RAG demo", root_path=os.getenv("ROOT_PATH", ""))
http = httpx.AsyncClient(timeout=httpx.Timeout(600.0))


# ---------------------------------------------------------------- startup
async def pull_models():
    """Make sure both models exist in Ollama (no-op if already pulled)."""
    for model in (EMBED_MODEL, CHAT_MODEL):
        for attempt in range(30):
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
    asyncio.create_task(pull_models())


# ---------------------------------------------------------------- helpers
def chunk_text(text: str) -> list[str]:
    """Paragraph-aware chunking with a small character overlap between chunks."""
    pieces = []
    for p in re.split(r"\n\s*\n", text):
        p = p.strip()
        while len(p) > CHUNK_SIZE:  # hard-split very long paragraphs
            pieces.append(p[:CHUNK_SIZE])
            p = p[CHUNK_SIZE - CHUNK_OVERLAP:]
        if p:
            pieces.append(p)
    chunks, cur = [], ""
    for p in pieces:
        if cur and len(cur) + len(p) + 2 > CHUNK_SIZE:
            chunks.append(cur)
            cur = cur[-CHUNK_OVERLAP:] + "\n\n" + p
        else:
            cur = f"{cur}\n\n{p}" if cur else p
    if cur:
        chunks.append(cur)
    return chunks


async def embed(texts: list[str]) -> list[list[float]]:
    out = []
    for i in range(0, len(texts), 16):
        r = await http.post(f"{OLLAMA}/api/embed", json={"model": EMBED_MODEL, "input": texts[i:i + 16]})
        if r.status_code != 200:
            raise HTTPException(503, f"Embedding failed (is {EMBED_MODEL} pulled yet?): {r.text[:200]}")
        out.extend(r.json()["embeddings"])
    return out


async def ensure_collection(dim: int):
    r = await http.get(f"{QDRANT}/collections/{COLLECTION}")
    if r.status_code == 200:
        return
    r = await http.put(f"{QDRANT}/collections/{COLLECTION}",
                       json={"vectors": {"size": dim, "distance": "Cosine"}})
    r.raise_for_status()


async def ingest_text(source: str, text: str) -> int:
    chunks = chunk_text(text)
    if not chunks:
        return 0
    vectors = await embed(chunks)
    await ensure_collection(len(vectors[0]))
    points = [{
        "id": str(uuid.uuid5(uuid.NAMESPACE_URL, f"{source}:{i}")),  # stable ids: re-ingest overwrites
        "vector": v,
        "payload": {"source": source, "chunk": i, "text": c},
    } for i, (c, v) in enumerate(zip(chunks, vectors))]
    r = await http.put(f"{QDRANT}/collections/{COLLECTION}/points?wait=true", json={"points": points})
    r.raise_for_status()
    return len(points)


async def search(question: str, top_k: int) -> list[dict]:
    [vec] = await embed([question])
    r = await http.post(f"{QDRANT}/collections/{COLLECTION}/points/search",
                        json={"vector": vec, "limit": top_k, "with_payload": True})
    if r.status_code == 404:  # nothing ingested yet
        return []
    r.raise_for_status()
    return [{"source": h["payload"]["source"], "score": round(h["score"], 3), "text": h["payload"]["text"]}
            for h in r.json()["result"]]


async def chat(messages: list[dict]) -> str:
    r = await http.post(f"{OLLAMA}/api/chat", json={
        "model": CHAT_MODEL, "messages": messages, "stream": False, "options": {"temperature": 0.2}})
    if r.status_code != 200:
        raise HTTPException(503, f"Chat model not ready: {r.text[:200]}")
    return r.json()["message"]["content"]


# ---------------------------------------------------------------- API
class Query(BaseModel):
    question: str
    use_rag: bool = True
    top_k: int = 4


@app.get("/status")
async def status():
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


@app.post("/ingest/samples")
async def ingest_samples():
    files = sorted(p for p in SAMPLE_DIR.glob("*") if p.is_file() and not p.name.startswith(".."))
    if not files:
        raise HTTPException(404, f"No sample docs in {SAMPLE_DIR}")
    result = {p.name: await ingest_text(p.name, p.read_text(encoding="utf-8")) for p in files}
    return {"ingested_chunks": result}


@app.post("/ingest")
async def ingest(file: UploadFile = File(...)):
    data = await file.read()
    if file.filename.lower().endswith(".pdf"):
        from pypdf import PdfReader
        text = "\n\n".join(page.extract_text() or "" for page in PdfReader(io.BytesIO(data)).pages)
    else:
        text = data.decode("utf-8", errors="replace")
    n = await ingest_text(file.filename, text)
    if n == 0:
        raise HTTPException(400, "No text found in file")
    return {"source": file.filename, "chunks": n}


@app.post("/query")
async def query(q: Query):
    if not q.use_rag:
        answer = await chat([{"role": "user", "content": q.question}])
        return {"mode": "plain-llm", "answer": answer, "sources": []}

    hits = await search(q.question, q.top_k)
    if not hits:
        return {"mode": "rag", "answer": "No documents found. Ingest some documents first.", "sources": []}
    context = "\n\n".join(f"[{i + 1}] (source: {h['source']})\n{h['text']}" for i, h in enumerate(hits))
    system = ("You answer questions using ONLY the context below. If the answer is not in the context, "
              "say you don't know. Cite sources like [1].\n\nContext:\n" + context)
    answer = await chat([{"role": "system", "content": system}, {"role": "user", "content": q.question}])
    return {"mode": "rag", "answer": answer, "sources": hits}


@app.delete("/collection")
async def wipe():
    r = await http.delete(f"{QDRANT}/collections/{COLLECTION}")
    return {"deleted": r.status_code == 200}


@app.get("/", response_class=HTMLResponse)
async def ui():
    return UI_HTML


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
