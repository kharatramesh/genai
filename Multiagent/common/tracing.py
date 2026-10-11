"""STEP 8 - Langfuse tracing. One handler is attached to the run config and every
LLM call / tool call / agent node below it is recorded automatically."""
import os
import uuid

from dotenv import load_dotenv

load_dotenv()


def langfuse_enabled() -> bool:
    return bool(os.getenv("LANGFUSE_PUBLIC_KEY") and os.getenv("LANGFUSE_SECRET_KEY"))


def make_config(session_id: str | None = None, **extra) -> dict:
    """Build the LangGraph run config: thread id (for memory/interrupts) + Langfuse callback."""
    session_id = session_id or uuid.uuid4().hex[:8]
    config = {"configurable": {"thread_id": session_id}, "recursion_limit": 40, **extra}
    if langfuse_enabled():
        from langfuse.langchain import CallbackHandler

        config["callbacks"] = [CallbackHandler()]
        config["metadata"] = {"langfuse_session_id": session_id}
    return config


def flush():
    if langfuse_enabled():
        from langfuse import get_client

        get_client().flush()
