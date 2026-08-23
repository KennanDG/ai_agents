from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from ai_agents.agents.voice.nodes import (
    custom_tools_node,
    gather_context_node,
    intake_node,
)
from ai_agents.agents.voice.state import VoiceAgentState


def build_voice_agent_graph():
    builder = StateGraph(VoiceAgentState)

    builder.add_node("gather_context", gather_context_node)
    builder.add_node("custom_tools", custom_tools_node)
    builder.add_node("intake", intake_node)

    # Keep custom tools outside the intake-model call. Approved tools enrich the
    # backend-owned evidence first; the intake model still has no callable tools.
    builder.add_edge(START, "gather_context")
    builder.add_edge("gather_context", "custom_tools")
    builder.add_edge("custom_tools", "intake")
    builder.add_edge("intake", END)

    return builder.compile()
