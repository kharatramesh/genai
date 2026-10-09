# LangGraph agent (no FastAPI): explicit graph of model node <-> tool node
from pathlib import Path
from langchain_core.tools import tool
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import ToolNode, tools_condition

workspace_dir = Path(__file__).parent / "workspace"
workspace_dir.mkdir(exist_ok=True)


@tool
def create_file(filename: str, content: str) -> str:
    """Create a text file in the workspace with the given content."""
    path = workspace_dir / filename
    path.write_text(content, encoding="utf-8")
    return f"Created {path.resolve()}"


@tool
def read_file(filename: str) -> str:
    """Read a text file from the workspace."""
    return (workspace_dir / filename).read_text(encoding="utf-8")


@tool
def list_files() -> str:
    """List the files in the workspace."""
    return "\n".join(p.name for p in workspace_dir.iterdir()) or "(empty)"


tools = [create_file, read_file, list_files]
llm = ChatOllama(model="qwen3:8b", base_url="http://localhost:11434").bind_tools(tools)


def call_model(state: MessagesState):
    # Node 1: the model decides to answer or call a tool
    return {"messages": [llm.invoke(state["messages"])]}


graph = StateGraph(MessagesState)
graph.add_node("model", call_model)
graph.add_node("tools", ToolNode(tools))       # Node 2: runs requested tools
graph.add_edge(START, "model")
graph.add_conditional_edges("model", tools_condition)  # tool call -> "tools", else END
graph.add_edge("tools", "model")                # loop back with the tool result
app = graph.compile()

if __name__ == "__main__":
    while True:
        try:
            question = input("You (blank to quit): ").strip()
        except EOFError:
            break
        if not question:
            break
        result = app.invoke({"messages": [{"role": "user", "content": question}]})
        print("Agent:", result["messages"][-1].content)
