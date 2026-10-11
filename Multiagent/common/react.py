"""STEP 3 - A tiny reusable ReAct agent: model node <-> tool node, built with LangGraph."""
from langchain_core.messages import SystemMessage
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition


def build_react_agent(llm, tools, system_prompt: str, name: str, checkpointer=None):
    llm_with_tools = llm.bind_tools(tools)

    def call_model(state: MessagesState):
        msgs = [SystemMessage(system_prompt)] + state["messages"]
        return {"messages": [llm_with_tools.invoke(msgs)]}

    g = StateGraph(MessagesState)
    g.add_node("model", call_model)
    g.add_node("tools", ToolNode(tools, handle_tool_errors=True))  # tool exceptions -> error text for the LLM
    g.add_edge(START, "model")
    g.add_conditional_edges("model", tools_condition)  # tool call -> "tools", else END
    g.add_edge("tools", "model")
    return g.compile(name=name, checkpointer=checkpointer)  # None when used as a node: inherits the parent's
