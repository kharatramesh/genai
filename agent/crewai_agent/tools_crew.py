# CrewAI with a CUSTOM TOOL and a file output: an analyst reads workspace files, a reporter writes a report
# Concepts: @tool, Agent(tools=...), Task(output_file=...), context between tasks
import sys
from pathlib import Path

# Windows consoles default to cp1252 and crash on CrewAI's emoji logging
sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

from crewai import Agent, Task, Crew, Process, LLM
from crewai.tools import tool

workspace_dir = Path(__file__).parent / "workspace"
workspace_dir.mkdir(exist_ok=True)


@tool("list_files")
def list_files() -> str:
    """List the files in the workspace."""
    return "\n".join(p.name for p in workspace_dir.iterdir()) or "(empty)"


@tool("read_file")
def read_file(filename: str) -> str:
    """Read a text file from the workspace."""
    path = (workspace_dir / filename).resolve()
    if workspace_dir.resolve() not in path.parents:
        return "Error: path must stay inside the workspace"
    return path.read_text(encoding="utf-8") if path.is_file() else f"Error: {filename} not found"


llm = LLM(model="ollama/qwen3:8b", base_url="http://localhost:11434")

analyst = Agent(
    role="File Analyst",
    goal="Find out what the workspace files contain",
    backstory="You inspect files with your tools and never guess their contents.",
    tools=[list_files, read_file],  # only this agent may touch files
    llm=llm,
    verbose=True,
)
reporter = Agent(
    role="Reporter",
    goal="Write a clear markdown report from the analyst's findings",
    backstory="You turn raw findings into a tidy report for managers.",
    llm=llm,
    verbose=True,
)

analyse_task = Task(
    description="List the workspace files, read each one, and note what each contains.",
    expected_output="One bullet per file: name and a one-line description of its content.",
    agent=analyst,
)
report_task = Task(
    description="Write a short markdown report titled 'Workspace Report' from the findings.",
    expected_output="A markdown report with a title and a bullet list.",
    agent=reporter,
    context=[analyse_task],
    output_file=str(workspace_dir / "report.md"),  # the result is also saved to disk
)

crew = Crew(
    agents=[analyst, reporter],
    tasks=[analyse_task, report_task],
    process=Process.sequential,
)

if __name__ == "__main__":
    result = crew.kickoff()
    print("\n=== Final result ===\n", result)
