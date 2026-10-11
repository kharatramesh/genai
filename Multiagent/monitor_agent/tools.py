"""Read Langfuse traces so an agent can reason about how the OTHER agents are behaving."""
from langchain_core.tools import tool

from common.tracing import flush, langfuse_enabled

AGENTS = ["file_agent", "system_agent", "container_agent", "monitor_agent"]
NOT_CONFIGURED ="Langfuse is not configured (set LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY in .env)."


def _client():
    from langfuse import get_client

    flush()  # make sure traces from this very session have been sent
    return get_client()


def _recent(limit: int):
    return _client().api.trace.list(limit=limit).data


@tool
def recent_traces(limit: int = 10) -> str:
    """List the latest traces (one per user request): id, time, name, latency, cost, tags."""
    if not langfuse_enabled():
        return NOT_CONFIGURED
    rows = [f"{t.id[:8]} | {t.timestamp:%H:%M:%S} | {t.name} | latency {getattr(t, 'latency', None) or 0:.1f}s | "
            f"cost ${getattr(t, 'total_cost', None) or 0:.4f} | tags {t.tags}" for t in _recent(limit)]
    return "\n".join(rows) or "no traces yet"


@tool
def agent_health_report(limit: int = 30) -> str:
    """Aggregate recent traces (request count, avg/max latency, total cost) plus runs and latency per agent."""
    if not langfuse_enabled():
        return NOT_CONFIGURED
    traces = _recent(limit)
    if not traces:
        return "no traces yet"
    lat = [getattr(t, "latency", None) or 0 for t in traces]
    cost = sum(getattr(t, "total_cost", None) or 0 for t in traces)
    lines = [f"{len(traces)} traces | avg latency {sum(lat) / len(lat):.1f}s | max {max(lat):.1f}s | total cost ${cost:.4f}"]
    # Each worker is a LangGraph node, so Langfuse records an observation named after it
    for agent in AGENTS:
        obs = _client().api.observations.get_many(name=agent, limit=limit, fields="core,basic,metrics").data
        if obs:
            avg = sum(o.latency or 0 for o in obs) / len(obs)
            lines.append(f"{agent}: {len(obs)} runs, avg {avg:.1f}s")
        else:
            lines.append(f"{agent}: no runs")
    return "\n".join(lines)


@tool
def failed_observations(limit: int = 20) -> str:
    """Show recent observations (LLM/tool calls) that ended with level ERROR."""
    if not langfuse_enabled():
        return NOT_CONFIGURED
    obs = _client().api.observations.get_many(limit=limit, level="ERROR").data
    return "\n".join(f"{o.name} | {o.start_time:%H:%M:%S} | trace {(o.trace_id or '?')[:8]} | {o.status_message}"
                     for o in obs) or "no errors found"


TOOLS = [recent_traces, agent_health_report, failed_observations]
