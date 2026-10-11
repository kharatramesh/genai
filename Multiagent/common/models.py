"""STEP 2 - Model factory: one place that decides OFFLINE (Ollama) vs ONLINE (Anthropic)."""
import os

from dotenv import load_dotenv
from langchain_ollama import ChatOllama

load_dotenv()


def offline_llm(temperature: float = 0):
    """Local model served by Ollama. No internet, no API key, data never leaves the machine."""
    return ChatOllama(
        model=os.getenv("OLLAMA_MODEL", "qwen3:8b"),
        base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        temperature=temperature,
    )


def online_llm(temperature: float = 0):
    """Hosted Claude model. Falls back to the local model if no key / FORCE_OFFLINE=1."""
    if os.getenv("FORCE_OFFLINE") == "1" or not os.getenv("ANTHROPIC_API_KEY"):
        print("[models] online model unavailable -> using offline fallback")
        return offline_llm(temperature)
    from langchain_anthropic import ChatAnthropic

    return ChatAnthropic(
        model=os.getenv("ANTHROPIC_MODEL", "claude-haiku-5-5"),
        temperature=temperature,
        max_tokens=2048,
    )
