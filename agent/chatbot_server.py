# FastAPI web server that exposes the file agent in file.py as a chatbot.
# Install:  pip install fastapi uvicorn
# Run:      python chatbot_server.py   then open http://localhost:8000
import json
import uuid

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

# Reuse the model call and tools from the agent
from file import TOOLS_MAP, call_local_ollama, format_tools

SYSTEM_PROMPT = {
    "role": "system",
    "content": (
        "You are a local file assistant. "
        "Use relative filenames only. Do not claim a file was created "
        "unless the tool reports success."
    ),
}
MAX_ITERATIONS = 5

app = FastAPI(title="File Agent Chatbot")

# In-memory conversation history keyed by session id
sessions: dict[str, list] = {}


class ChatRequest(BaseModel):
    message: str
    session_id: str | None = None
    model: str = "qwen3:8b"


class ChatResponse(BaseModel):
    session_id: str
    reply: str


def run_turn(messages: list, model_name: str) -> str:
    """Run the agent loop on the history and return the final reply text."""
    for _ in range(MAX_ITERATIONS):
        result = call_local_ollama(messages, model_name)
        message = result["choices"][0]["message"]
        messages.append(message)

        tool_calls = message.get("tool_calls")
        if not tool_calls:
            return message.get("content") or ""

        for tool_call in tool_calls:
            name = tool_call["function"]["name"]
            if name not in TOOLS_MAP:
                tool_result = f"Error: Unknown tool {name}"
            else:
                try:
                    arguments = json.loads(tool_call["function"]["arguments"])
                    tool_result = TOOLS_MAP[name](**arguments)
                except (ValueError, TypeError, OSError) as error:
                    tool_result = f"Tool execution failed: {error}"
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call["id"],
                "content": str(tool_result),
            })
    return "Stopped: maximum agent iterations reached."


# Plain `def` so FastAPI runs the blocking Ollama call in a worker thread
@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    session_id = req.session_id or uuid.uuid4().hex
    # Slash command handled locally: never sent to the model or stored in history
    if req.message.strip() == "/tools":
        return ChatResponse(session_id=session_id, reply=format_tools())
    messages = sessions.setdefault(session_id, [SYSTEM_PROMPT])
    messages.append({"role": "user", "content": req.message})
    try:
        reply = run_turn(messages, req.model)
    except RuntimeError as error:  # Ollama unreachable
        raise HTTPException(status_code=502, detail=str(error))
    return ChatResponse(session_id=session_id, reply=reply)


@app.delete("/chat/{session_id}")
def reset(session_id: str):
    sessions.pop(session_id, None)
    return {"status": "cleared"}


PAGE = """<!doctype html>
<html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>File Agent Chat</title>
<style>
body{font-family:system-ui,sans-serif;max-width:720px;margin:0 auto;padding:16px;display:flex;flex-direction:column;height:100vh;box-sizing:border-box}
#log{flex:1;overflow-y:auto;border:1px solid #ccc;border-radius:8px;padding:12px}
.m{margin:8px 0;padding:8px 12px;border-radius:8px;white-space:pre-wrap;max-width:85%}
.user{background:#dbeafe;margin-left:auto}.bot{background:#f1f5f9}.err{background:#fee2e2}
form{display:flex;gap:8px;margin-top:12px}input{flex:1;padding:10px;font-size:16px}button{padding:10px 18px;font-size:16px}
</style></head><body>
<h2>File Agent Chat</h2><div id="log"></div>
<form id="f"><input id="t" autocomplete="off" placeholder="e.g. create hello.txt with some text, or /tools" autofocus><button>Send</button></form>
<script>
let sid=null;const log=document.getElementById('log'),t=document.getElementById('t');
function add(c,x){const d=document.createElement('div');d.className='m '+c;d.textContent=x;log.appendChild(d);log.scrollTop=log.scrollHeight;return d}
document.getElementById('f').onsubmit=async e=>{e.preventDefault();const m=t.value.trim();if(!m)return;t.value='';add('user',m);
const w=add('bot','...');
try{const r=await fetch('/chat',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({message:m,session_id:sid})});
const j=await r.json();if(!r.ok)throw new Error(j.detail);sid=j.session_id;w.textContent=j.reply}
catch(err){w.className='m err';w.textContent=err.message}};
</script></body></html>"""


@app.get("/", response_class=HTMLResponse)
def index():
    return PAGE


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
