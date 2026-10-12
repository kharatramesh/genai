"""MCP client + local Ollama model.

Flow: user prompt -> Ollama (sees MCP tools) -> tool call -> MCP server -> result -> Ollama -> answer

Usage:  python client.py "What is the weather in Pune?"
        python client.py                (interactive chat)
        set OLLAMA_MODEL=qwen3:8b       (default: llama3.2)

Langfuse tracing is ON when LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY are set (env or .env).
Each user turn = one trace; each Ollama call = a generation; each MCP call = a tool span.
"""
import asyncio
import os
import sys
import uuid
from contextlib import AsyncExitStack
from pathlib import Path

import ollama
from dotenv import load_dotenv
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

load_dotenv()

TRACING = bool(os.getenv("LANGFUSE_PUBLIC_KEY") and os.getenv("LANGFUSE_SECRET_KEY"))
if TRACING:
    from langfuse import get_client, observe, propagate_attributes
    langfuse = get_client()
else:                                   # no-op fallbacks so the code below needs no branching
    from contextlib import nullcontext

    class _Off:
        def __getattr__(self, _):
            return lambda *a, **k: nullcontext(_Off())
        def update(self, **_): pass

    langfuse = _Off()
    def propagate_attributes(**_):
        return nullcontext()
    def observe(**_):
        return lambda f: f

MODEL = os.environ.get("OLLAMA_MODEL", "llama3.2")
HERE = Path(__file__).parent
SERVERS = ["weather_server.py", "hardware_server.py", "file_server.py"]


async def connect_all(stack: AsyncExitStack):
    """Start every server as a subprocess; return {tool_name: session} and Ollama tool specs."""
    routes, specs = {}, []
    for script in SERVERS:
        params = StdioServerParameters(command=sys.executable, args=[str(HERE / script)])
        read, write = await stack.enter_async_context(stdio_client(params))
        session = await stack.enter_async_context(ClientSession(read, write))
        await session.initialize()
        for t in (await session.list_tools()).tools:          # discovery
            routes[t.name] = session
            specs.append({"type": "function", "function": {   # MCP schema -> Ollama schema
                "name": t.name, "description": t.description or "",
                "parameters": t.inputSchema}})
    return routes, specs


def _plain(messages):
    """Ollama Message objects -> plain dicts so Langfuse can serialise them."""
    return [m if isinstance(m, dict) else m.model_dump(exclude_none=True) for m in messages]


@observe(name="mcp-chat-turn")
async def ask(client, routes, specs, messages, session_id=None):
    """Run the tool loop until the model answers in plain text."""
    with propagate_attributes(session_id=session_id,
                              metadata={"model": MODEL, "tools": ",".join(routes)}):
        langfuse.set_current_trace_io(input=messages[-1]["content"])
        while True:
            with langfuse.start_as_current_observation(
                    as_type="generation", name="ollama-chat", model=MODEL,
                    input=_plain(messages)) as gen:
                resp = await client.chat(model=MODEL, messages=messages, tools=specs)
                gen.update(output=resp.message.model_dump(exclude_none=True),
                           usage_details={"input": resp.prompt_eval_count or 0,
                                          "output": resp.eval_count or 0})
            msg = resp.message
            messages.append(msg)
            if not msg.tool_calls:
                langfuse.set_current_trace_io(output=msg.content)
                return msg.content
            for call in msg.tool_calls:
                name, args = call.function.name, dict(call.function.arguments)
                print(f"  [tool] {name}({args})")
                with langfuse.start_as_current_observation(
                        as_type="tool", name=f"tool:{name}", input=args) as span:
                    try:
                        result = await routes[name].call_tool(name, args)    # MCP call
                        text = "".join(c.text for c in result.content if c.type == "text")
                        span.update(output=text)
                    except Exception as e:
                        text = f"Error: {e}"
                        span.update(output=text, level="ERROR", status_message=str(e))
                print(f"  [result] {text[:200]}")
                messages.append({"role": "tool", "tool_name": name, "content": text})


async def main():
    client = ollama.AsyncClient()
    one_shot = " ".join(sys.argv[1:]) or None
    async with AsyncExitStack() as stack:
        routes, specs = await connect_all(stack)
        print(f"Model: {MODEL} | tools: {', '.join(routes)} | Langfuse: {'ON' if TRACING else 'OFF'}\n")
        session_id = uuid.uuid4().hex[:8]
        messages = [{"role": "system", "content":
                     "You are a helpful assistant. Use the tools when needed. "
                     "File tools only work on a small sandbox folder."}]
        while True:
            q = one_shot or input("you> ").strip()
            if not q:
                break
            messages.append({"role": "user", "content": q})
            print("assistant>", await ask(client, routes, specs, messages, session_id), "\n")
            if one_shot:
                break


if __name__ == "__main__":
    try:
        asyncio.run(main())
    finally:
        if TRACING:
            langfuse.flush()
