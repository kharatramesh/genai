# Multiagent - LangGraph multi-agent system

Four specialist agents (each in its own folder) orchestrated by a LangGraph plan-and-execute graph,
with human-in-the-loop approvals and Langfuse tracing. Full step-by-step guide: open `docs.html`.
All agents run on an online Groq model (default `llama-3.3-70b-versatile`).

| Folder | Agent | Model |
|---|---|---|
| `file_agent/` | File manager (sandboxed to `workspace/`) | Online - Groq |
| `system_agent/` | System monitor (psutil) | Online - Groq |
| `container_agent/` | Docker container lifecycle | Online - Groq |
| `monitor_agent/` | Monitors all agents via Langfuse | Online - Groq |
| `supervisor/` | LangGraph orchestrator (planner -> workers -> summary) | Online - Groq |
| `common/` | Model factory, HITL, tracing, ReAct builder, CLI | - |

## Quick start (Windows / PowerShell)

```powershell
cd Multiagent
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env      # then fill in GROQ_API_KEY and LANGFUSE_* keys

python -m file_agent.run        # one agent on its own
python -m system_agent.run
python -m container_agent.run
python -m monitor_agent.run
python main.py                  # the whole multi-agent system
```

Get a key at https://console.groq.com/keys. Change the model with `GROQ_MODEL`.
To run fully local instead, set `FORCE_OFFLINE=1` (needs `pip install langchain-ollama` and `ollama pull qwen3:8b`).
Set `APPROVE_PLAN=1` to also approve each plan before it runs.
