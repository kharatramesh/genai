"""STEP 4 - Human-in-the-loop. Any risky tool calls require_approval() first.

interrupt() PAUSES the whole graph (state is saved by the checkpointer) and returns
control to the caller. The caller resumes with Command(resume=<answer>).
"""
import os

from langgraph.types import interrupt


def require_approval(action: str, details: str) -> tuple[bool, str]:
    """Ask a human. Returns (approved, feedback)."""
    if os.getenv("AUTO_APPROVE") == "1":
        return True, ""
    answer = interrupt({"type": "approval", "action": action, "details": details})
    if isinstance(answer, dict):
        return bool(answer.get("approved")), str(answer.get("feedback", ""))
    return str(answer).strip().lower() in {"y", "yes", "approve", "approved", "ok"}, ""


def rejected(feedback: str = "") -> str:
    """Text handed back to the LLM so it can react to a 'no'."""
    return "REJECTED by human operator." + (f" Reason: {feedback}" if feedback else "") + \
        " Do not retry this action; tell the user it was not performed."
