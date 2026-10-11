"""STEP 2 - Model factory: one place that decides which model every agent uses (Groq online by default)."""
import os

from dotenv import load_dotenv

load_dotenv()


def offline_llm(temperature: float = 0):
    """Local model served by Ollama. No internet, no API key, data never leaves the machine."""
    from langchain_ollama import ChatOllama

    return ChatOllama(
        model=os.getenv("OLLAMA_MODEL", "qwen3:8b"),
        base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        temperature=temperature,
    )


def online_llm(temperature: float = 0):
    """Hosted model on Groq (needs GROQ_API_KEY). Set FORCE_OFFLINE=1 to use local Ollama instead."""
    if os.getenv("FORCE_OFFLINE") == "1":
        print("[models] FORCE_OFFLINE=1 -> using local Ollama model")
        return offline_llm(temperature)
    if not os.getenv("GROQ_API_KEY"):
        raise RuntimeError("GROQ_API_KEY is not set. Add it to .env (get one at https://console.groq.com/keys).")
    from langchain_groq import ChatGroq

    return ChatGroq(
        model=os.getenv("GROQ_MODEL", "openai/gpt-oss-120b"),
        temperature=temperature,
        max_tokens=2048,
    )
