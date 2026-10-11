"""Agent 2 - SYSTEM MONITOR (ONLINE model: Groq)."""
from common.models import online_llm
from common.react import build_react_agent
from system_agent.tools import TOOLS

DESCRIPTION = "report host CPU, memory, disk, network, uptime and top processes"
PROMPT = ("You are the System Monitor agent. Call tools to get real numbers, then summarise them. "
          "Flag anything above 85% utilisation as a warning.")


def build(checkpointer=None):
    return build_react_agent(online_llm(), TOOLS, PROMPT, name="system_agent", checkpointer=checkpointer)
