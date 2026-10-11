"""STEP 10 - LangGraph orchestrator (plan-and-execute): plan once, run each worker in order, summarise.

  START -> begin -> planner -> approve_plan --> file_agent / system_agent / container_agent / monitor_agent
                                     |              (each worker returns to the "next step" router)
                                     +--> summarize -> END      (when the plan is finished)
"""
import os
from typing import Literal

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.types import interrupt
from pydantic import BaseModel, Field

import container_agent.agent as container
import file_agent.agent as files
import monitor_agent.agent as monitor
import system_agent.agent as system
from common.models import offline_llm

WORKERS = {"file_agent": files, "system_agent": system, "container_agent": container, "monitor_agent": monitor}
AgentName = Literal["file_agent", "system_agent", "container_agent", "monitor_agent"]


class Step(BaseModel):
    agent: AgentName
    task: str = Field(description="self-contained instruction for that agent, with all names/values it needs")


class Plan(BaseModel):
    """Ordered steps needed for the request. Empty list if no agent is needed."""
    steps: list[Step]


class State(MessagesState):
    plan: list[dict]
    idx: int
    reports: list[str]


PLANNER_PROMPT = (
    "You are a planner. Split the user's request into the minimum ordered steps, one per agent action.\n"
    + "\n".join(f"- {n}: {m.DESCRIPTION}" for n, m in WORKERS.items())
    + "\nIf a later step needs the output of an earlier step, say so in its task "
      "(e.g. 'save the memory summary reported by the previous step to ram.txt'). "
      "Use an empty list for greetings or questions that need no agent."
)


def last_user_text(messages) -> str:
    return next(m.content for m in reversed(messages) if isinstance(m, HumanMessage))


def build_graph(checkpointer):
    llm = offline_llm()  # planning + summarising run on the OFFLINE model
    planner_llm = llm.with_structured_output(Plan)

    def begin(state: State):
        return {"plan": [], "idx": 0, "reports": []}  # reset for each new user request

    def planner(state: State):
        plan = planner_llm.invoke([SystemMessage(PLANNER_PROMPT), HumanMessage(last_user_text(state["messages"]))])
        steps = [s.model_dump() for s in plan.steps]
        print("  plan: " + (" -> ".join(s["agent"] for s in steps) or "(no agent needed)"))
        return {"plan": steps}

    def approve_plan(state: State):
        # HUMAN IN THE LOOP #2. Separate node on purpose: on resume only this node re-runs,
        # so the human approves exactly the plan that was saved (the planner LLM is not called again).
        if not state["plan"] or os.getenv("APPROVE_PLAN") != "1":
            return {}
        answer = interrupt({"type": "approval", "action": "execute_plan",
                            "details": "\n     ".join(f"{i + 1}. {s['agent']}: {s['task']}" for i, s in enumerate(state["plan"]))})
        if not (isinstance(answer, dict) and answer.get("approved")):
            return {"plan": [], "reports": ["Plan rejected by human operator."]}
        return {}

    def next_step(state: State) -> str:
        return state["plan"][state["idx"]]["agent"] if state["idx"] < len(state["plan"]) else "summarize"

    def make_worker(name: str, module):
        agent = module.build()

        def worker(state: State, config):
            step = state["plan"][state["idx"]]
            prior = "\n".join(state["reports"])
            task = step["task"] + (f"\n\nResults of previous steps:\n{prior}" if prior else "")
            result = agent.invoke({"messages": [HumanMessage(task)]}, config)
            report = f"[{name}] {result['messages'][-1].content}"
            return {"idx": state["idx"] + 1, "reports": state["reports"] + [report],
                    "messages": [AIMessage(report, name=name)]}

        return worker

    def summarize(state: State):
        question = last_user_text(state["messages"])
        reports = "\n".join(state["reports"]) or "(no agent was needed)"
        answer = llm.invoke([SystemMessage("Write a short final answer for the user using ONLY the agent reports. "
                                           "If a report says an action was rejected or failed, say so."),
                             HumanMessage(f"Request: {question}\n\nAgent reports:\n{reports}")])
        return {"messages": [AIMessage(answer.content, name="orchestrator")]}

    g = StateGraph(State)
    g.add_node("begin", begin)
    g.add_node("planner", planner)
    g.add_node("approve_plan", approve_plan)
    g.add_node("summarize", summarize)
    g.add_edge(START, "begin")
    g.add_edge("begin", "planner")
    g.add_edge("planner", "approve_plan")
    targets = [*WORKERS, "summarize"]
    g.add_conditional_edges("approve_plan", next_step, targets)
    for name, module in WORKERS.items():
        g.add_node(name, make_worker(name, module))
        g.add_conditional_edges(name, next_step, targets)
    g.add_edge("summarize", END)
    return g.compile(checkpointer=checkpointer)
