"""Run this agent on its own:  python -m system_agent.run"""
from common.cli import chat
from system_agent.agent import build

if __name__ == "__main__":
    chat(lambda cp: build(cp), "System Monitor (offline)")
