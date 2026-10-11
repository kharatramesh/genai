# LangGraph reflection loop: writer -> critic -> (revise or finish), with a custom state
# Concepts: custom State, conditional edges that loop, stopping rule, no tools needed
import sys
from typing import TypedDict
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, START, END

llm = ChatOllama(model="qwen3:8b", base_url="http://localhost:11434")
MAX_ROUNDS = 3


class State(TypedDict):
    topic: str
    draft: str
    feedback: str
    rounds: int


def writer(state: State):
    # First pass writes from scratch; later passes revise using the critic's feedback
    if state["feedback"]:
        prompt = (f"Topic: {state['topic']}\nDraft:\n{state['draft']}\n\n"
                  f"Critic feedback:\n{state['feedback']}\n\nRewrite the draft fixing the feedback.")
    else:
        prompt = f"Write a short 4-sentence explanation of: {state['topic']}"
    return {"draft": llm.invoke(prompt).content, "rounds": state["rounds"] + 1}


def critic(state: State):
    reply = llm.invoke(
        f"Review this draft for clarity and accuracy:\n{state['draft']}\n\n"
        "If it is good, reply with exactly APPROVED. Otherwise give 2 concrete fixes."
    ).content
    return {"feedback": reply}


def route(state: State):
    # Stop when approved or when we hit the round limit (prevents endless loops)
    if "APPROVED" in state["feedback"].upper() or state["rounds"] >= MAX_ROUNDS:
        return END
    return "writer"


graph = StateGraph(State)
graph.add_node("writer", writer)
graph.add_node("critic", critic)
graph.add_edge(START, "writer")
graph.add_edge("writer", "critic")
graph.add_conditional_edges("critic", route)
app = graph.compile()

if __name__ == "__main__":
    topic = " ".join(sys.argv[1:]) or "how a transformer model works"
    result = app.invoke({"topic": topic, "draft": "", "feedback": "", "rounds": 0})
    print(f"=== Final draft (after {result['rounds']} round(s)) ===\n{result['draft']}")
