"""Run this agent on its own:  python -m monitor_agent.run"""
from common.cli import chat
from monitor_agent.agent import build

if __name__ == "__main__":
    chat(lambda cp: build(cp), "Agent Monitor (online)")
