# LangChain agent (no FastAPI): a tool-calling agent running in the terminal
from pathlib import Path
from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain_ollama import ChatOllama

# File operations are confined to this folder
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


llm = ChatOllama(model="qwen3:8b", base_url="http://localhost:11434")
agent = create_agent(llm, tools=[create_file, read_file, list_files])

if __name__ == "__main__":
    while True:
        try:
            question = input("You (blank to quit): ").strip()
        except EOFError:
            break
        if not question:
            break
        result = agent.invoke({"messages": [{"role": "user", "content": question}]})
        print("Agent:", result["messages"][-1].content)
