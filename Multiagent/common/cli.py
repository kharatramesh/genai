"""STEP 11 - Terminal driver: streams progress and answers human-approval interrupts."""
import sys

from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from common.tracing import flush, langfuse_enabled, make_config

sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # Windows consoles default to cp1252


def ask_human(req: dict) -> dict:
    print(f"\n  HUMAN APPROVAL NEEDED -> {req.get('action')}\n     {req.get('details')}")
    ans = input("     approve? [y/N, or type a reason to reject]: ").strip()
    if ans.lower() in {"y", "yes"}:
        return {"approved": True}
    return {"approved": False, "feedback": "" if ans.lower() in {"", "n", "no"} else ans}


def run_turn(app, payload, config) -> str:
    """Run one user request to completion, looping while the graph asks for approval."""
    final = ""
    while True:
        pending = []
        for chunk in app.stream(payload, config, stream_mode="updates"):
            for node, update in chunk.items():
                if node == "__interrupt__":
                    pending.extend(update)  # several, if the model asked for 2+ risky tools at once
                    continue
                print(f"  [{node}]")
                for m in (update or {}).get("messages", []):
                    if isinstance(m, AIMessage) and m.content and not m.tool_calls:
                        final = m.content if isinstance(m.content, str) else str(m.content)
        if not pending:
            return final
        payload = Command(resume={i.id: ask_human(i.value) for i in pending})  # answer each interrupt by id


def chat(app_builder, title: str):
    """Interactive loop for any compiled graph. app_builder(checkpointer) -> compiled graph."""
    app = app_builder(InMemorySaver())
    config = make_config()
    print(f"=== {title} === (Langfuse tracing: {'ON' if langfuse_enabled() else 'off'}; blank line to quit)")
    while True:
        try:
            q = input("\nYou: ").strip()
        except EOFError:
            break
        if not q:
            break
        print("\nAnswer:", run_turn(app, {"messages": [{"role": "user", "content": q}]}, config))
    flush()
