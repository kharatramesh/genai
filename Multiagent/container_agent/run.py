"""Run this agent on its own:  python -m container_agent.run"""
from common.cli import chat
from container_agent.agent import build

if __name__ == "__main__":
    chat(lambda cp: build(cp), "Container Manager (online)")
