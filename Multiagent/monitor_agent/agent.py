"""Agent 4 - AGENT MONITOR / OBSERVER (ONLINE model: Claude). Reads Langfuse traces of all agents."""
from common.models import online_llm
from common.react import build_react_agent
from monitor_agent.tools import TOOLS

DESCRIPTION = "report on the health, latency, cost and errors of the other agents using Langfuse traces"
PROMPT = ("You are the Agent Monitor. Use the Langfuse tools to inspect how the agents have performed. "
          "Report latency, cost, usage per agent and any errors. If Langfuse is not configured, say so plainly.")


def build(checkpointer=None):
    return build_react_agent(online_llm(), TOOLS, PROMPT, name="monitor_agent", checkpointer=checkpointer)
