"""Run this agent on its own:  python -m file_agent.run"""
from common.cli import chat
from file_agent.agent import build

if __name__ == "__main__":
    chat(lambda cp: build(cp), "File Manager (offline)")
