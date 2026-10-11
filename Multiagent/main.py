"""Run the whole multi-agent system:  python main.py"""
from common.cli import chat
from supervisor.graph import build_graph

if __name__ == "__main__":
    chat(build_graph, "Multi-agent system (supervisor + 4 agents)")
