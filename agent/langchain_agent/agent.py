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

# ---------------------------------------------------------------------------
# DAG = Directed Acyclic Graph
#   Directed : data flows one way, from a node to the next (A -> B)
#   Acyclic  : it never loops back, so every run is guaranteed to finish
# A DAG lets independent steps run IN PARALLEL (fan-out) and then combine
# their results in one step (fan-in):
#
#            +--> pros_agent --+
#   topic ---+                 +--> writer_agent --> final answer
#            +--> cons_agent --+
#
# In LangChain a DAG is built with LCEL (LangChain Expression Language):
#   |                 -> sequential edge        (A | B)
#   RunnableParallel  -> fan-out, same input to many branches, outputs in a dict
#   RunnableLambda    -> wraps any python function / agent as a node
# Agents below are plain nodes: a node only needs "input in, output out".
# ---------------------------------------------------------------------------
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnableParallel


def ask(agent_, text: str) -> str:
    """Run an agent on one message and return only its final text."""
    result = agent_.invoke({"messages": [{"role": "user", "content": text}]})
    return result["messages"][-1].content


# Two specialist agents with different system prompts (the parallel branches)
pros_agent = create_agent(llm, tools=[], system_prompt="List the 3 strongest advantages of the topic. Be brief.")
cons_agent = create_agent(llm, tools=[], system_prompt="List the 3 strongest drawbacks of the topic. Be brief.")

# DAG 1 - agents as nodes: topic -> {pros || cons} -> writer
analysis_dag = (
    RunnableParallel(  # fan-out: both agents get the same topic and run at the same time
        pros=RunnableLambda(lambda topic: ask(pros_agent, topic)),
        cons=RunnableLambda(lambda topic: ask(cons_agent, topic)),
    )
    | ChatPromptTemplate.from_template(  # fan-in: this node receives {"pros": ..., "cons": ...}
        "Write a balanced 3-sentence verdict.\nPros:\n{pros}\n\nCons:\n{cons}"
    )
    | llm
    | StrOutputParser()
)

# DAG 2 - a pure chain DAG with a branch that also passes the input through:
#   topic -> {title || summary || original topic} -> report
report_dag = (
    RunnableParallel(
        topic=lambda t: t,  # a plain function is auto-wrapped as a node (passthrough)
        title=RunnableLambda(lambda t: {"t": t})
        | ChatPromptTemplate.from_template("Give one catchy title for: {t}") | llm | StrOutputParser(),
        summary=RunnableLambda(lambda t: {"t": t})
        | ChatPromptTemplate.from_template("Summarize in one sentence: {t}") | llm | StrOutputParser(),
    )
    | RunnableLambda(lambda d: f"# {d['title']}\n\nTopic: {d['topic']}\n\n{d['summary']}")
)

# ---------------------------------------------------------------------------
# MEMORY agent: a checkpointer remembers the conversation per thread_id
# ---------------------------------------------------------------------------
import ast
import operator
from langgraph.checkpoint.memory import InMemorySaver

# Whitelisted operators so the calculator never uses eval() on model-written text
OPS = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
    ast.Div: operator.truediv, ast.Pow: operator.pow, ast.USub: operator.neg,
}


def _eval(node):
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in OPS:
        return OPS[type(node.op)](_eval(node.left), _eval(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in OPS:
        return OPS[type(node.op)](_eval(node.operand))
    raise ValueError("Unsupported expression")


@tool
def calculator(expression: str) -> str:
    """Evaluate a math expression such as '(12 + 8) * 3'."""
    try:
        return str(_eval(ast.parse(expression, mode="eval").body))
    except (ValueError, SyntaxError, ZeroDivisionError) as error:
        return f"Error: {error}"


memory_agent = create_agent(
    llm,
    tools=[calculator],
    system_prompt="You are a friendly assistant. Use the calculator for any math.",
    checkpointer=InMemorySaver(),  # stores messages between invoke() calls
)

if __name__ == "__main__":
    print("Modes: /file (tool agent, default) | /dag (pros||cons -> verdict) | /report (title||summary -> report)"
          " | /memory (remembers per thread; /new starts a fresh one)")
    mode = "/file"
    thread = 1
    while True:
        try:
            question = input(f"[{mode}] You (blank to quit): ").strip()
        except EOFError:
            break
        if not question:
            break
        if question in ("/file", "/dag", "/report", "/memory"):
            mode = question
            continue
        if question == "/new":
            thread += 1  # a new thread_id means a brand-new, empty conversation
            print(f"(started thread {thread})")
            continue
        if mode == "/memory":
            config = {"configurable": {"thread_id": str(thread)}}
            result = memory_agent.invoke({"messages": [{"role": "user", "content": question}]}, config)
            print("Agent:", result["messages"][-1].content)
        elif mode == "/dag":
            print(analysis_dag.get_graph().draw_ascii())  # prints the DAG structure
            print("Verdict:", analysis_dag.invoke(question))
        elif mode == "/report":
            print(report_dag.invoke(question))
        else:
            result = agent.invoke({"messages": [{"role": "user", "content": question}]})
            print("Agent:", result["messages"][-1].content)
