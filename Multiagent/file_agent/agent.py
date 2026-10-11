"""Agent 1 - FILE MANAGER (OFFLINE model: Ollama)."""
from common.models import offline_llm
from common.react import build_react_agent
from file_agent.tools import TOOLS

DESCRIPTION = "create, read, search, list and delete files in the workspace folder"
PROMPT = ("You are the File Manager agent. You only manage files inside the workspace using your tools. "
          "Never invent file contents you have not read. Be concise.")


def build(checkpointer=None):
    return build_react_agent(offline_llm(), TOOLS, PROMPT, name="file_agent", checkpointer=checkpointer)
