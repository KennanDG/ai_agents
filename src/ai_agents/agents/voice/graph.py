from __future__ import annotations

from typing import TypedDict, Optional
import io

from langgraph.graph import END, START, StateGraph

from ai_agents.agents.voice.nodes import intake_node
from ai_agents.agents.voice.state import VoiceAgentState


def build_voice_agent_graph():
    builder = StateGraph(VoiceAgentState)

    builder.add_node("intake", intake_node)

    builder.add_edge(START, "intake")
    builder.add_edge("intake", END)

    return builder.compile()