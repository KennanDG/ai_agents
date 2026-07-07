from __future__ import annotations

from typing import TypedDict, Optional
import io

from groq import Groq
from langgraph.graph import END, START, StateGraph
from langgraph.types import RetryPolicy

from ai_agents.config.settings import settings