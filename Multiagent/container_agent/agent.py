"""Agent 3 - CONTAINER MANAGER (ONLINE model: Groq). Lifecycle: create -> start -> stop -> restart -> remove."""
from common.models import online_llm
from common.react import build_react_agent
from container_agent.tools import TOOLS

DESCRIPTION = "create and manage Docker container lifecycle (run, start, stop, restart, remove, logs, stats)"
PROMPT = ("You are the Container Manager agent. Use the docker tools to manage the container lifecycle. "
          "Inspect state (list_containers) before changing it. Destructive tools ask a human for approval; "
          "if rejected, do not retry. If docker is missing, report the error.")


def build(checkpointer=None):
    return build_react_agent(online_llm(), TOOLS, PROMPT, name="container_agent", checkpointer=checkpointer)
