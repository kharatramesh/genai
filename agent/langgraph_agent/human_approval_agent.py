# LangGraph human-in-the-loop: the graph PAUSES before a destructive tool and waits for approval
# Concepts: interrupt(), Command(resume=...), checkpointer, thread_id
from pathlib import Path
from langchain_core.tools import tool
from langchain_ollama import ChatOllama
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import StateGraph, MessagesState, START
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.types import Command, interrupt

workspace_dir = Path(__file__).parent / "workspace"
workspace_dir.mkdir(exist_ok=True)


@tool
def create_file(filename: str, content: str) -> str:
    """Create a text file in the workspace with the given content."""
    (workspace_dir / filename).write_text(content, encoding="utf-8")
    return f"Created {filename}"


@tool
def delete_file(filename: str) -> str:
    """Delete a file from the workspace (needs human approval)."""
    # interrupt() stops the graph and surfaces this payload to the caller
    answer = interrupt(f"Delete '{filename}'? [y/N]")
    if str(answer).strip().lower() != "y":
        return "Deletion rejected by the user"
    (workspace_dir / filename).unlink()
    return f"Deleted {filename}"


tools = [create_file, delete_file]
llm = ChatOllama(model="qwen3:8b", base_url="http://localhost:11434").bind_tools(tools)


def call_model(state: MessagesState):
    return {"messages": [llm.invoke(state["messages"])]}


graph = StateGraph(MessagesState)
graph.add_node("model", call_model)
graph.add_node("tools", ToolNode(tools))
graph.add_edge(START, "model")
graph.add_conditional_edges("model", tools_condition)
graph.add_edge("tools", "model")
# Pausing needs saved state, so a checkpointer is required
app = graph.compile(checkpointer=InMemorySaver())

if __name__ == "__main__":
    config = {"configurable": {"thread_id": "1"}}
    while True:
        try:
            question = input("You (blank to quit): ").strip()
        except EOFError:
            break
        if not question:
            break
        result = app.invoke({"messages": [{"role": "user", "content": question}]}, config)
        # While the graph is paused, result contains "__interrupt__"; resume with the human's answer
        while "__interrupt__" in result:
            decision = input(result["__interrupt__"][0].value + " ").strip()
            result = app.invoke(Command(resume=decision), config)
        print("Agent:", result["messages"][-1].content)
