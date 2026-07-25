from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from ai_agents.agents.voice.nodes import gather_context_node, intake_node
from ai_agents.agents.voice.state import VoiceAgentState


def build_voice_agent_graph():
    builder = StateGraph(VoiceAgentState)

    builder.add_node("gather_context", gather_context_node)
    builder.add_node("intake", intake_node)

    builder.add_edge(START, "gather_context")
    builder.add_edge("gather_context", "intake")
    builder.add_edge("intake", END)

    return builder.compile()
