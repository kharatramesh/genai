# Multiagent - LangGraph multi-agent system

Four specialist agents (each in its own folder) orchestrated by a LangGraph plan-and-execute graph,
with human-in-the-loop approvals and Langfuse tracing. Full step-by-step guide: open `docs.html`.

| Folder | Agent | Model |
|---|---|---|
| `file_agent/` | File manager (sandboxed to `workspace/`) | Offline - Ollama `qwen3:8b` |
| `system_agent/` | System monitor (psutil) | Offline - Ollama `qwen3:8b` |
| `container_agent/` | Docker container lifecycle | Online - Claude (`claude-haiku-5-5`) |
| `monitor_agent/` | Monitors all agents via Langfuse | Online - Claude (`claude-haiku-5-5`) |
| `supervisor/` | LangGraph orchestrator (planner -> workers -> summary) | Offline |
| `common/` | Model factory, HITL, tracing, ReAct builder, CLI | - |

## Quick start (Windows / PowerShell)

```powershell
cd Multiagent
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env      # then fill in ANTHROPIC_API_KEY and LANGFUSE_* keys
ollama pull qwen3:8b

python -m file_agent.run        # one agent on its own
python -m system_agent.run
python -m container_agent.run
python -m monitor_agent.run
python main.py                  # the whole multi-agent system
```

No Anthropic key? Set `FORCE_OFFLINE=1` and the online agents fall back to Ollama.
Set `APPROVE_PLAN=1` to also approve each plan before it runs.
