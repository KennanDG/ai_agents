from __future__ import annotations

from typing import TypeVar

from langchain_core.output_parsers import PydanticOutputParser
from langchain_groq import ChatGroq
from openrouter import OpenRouter
from google import genai
from pydantic import BaseModel

from ai_agents.agents.coding.utils.constants import LLM_DECISION_MAX_ATTEMPTS
from ai_agents.agents.coding.runtime import node_config
from ai_agents.agents.coding.state import CodingAgentState
from ai_agents.agents.coding.utils.text import message_content_to_text
from ai_agents.config.settings import settings as config_settings


llm = ChatGroq(
    model=config_settings.reasoning_model,
    api_key=config_settings.resolved_groq_api_key(),
)

DecisionT = TypeVar("DecisionT", bound=BaseModel)


def invoke_parsed_decision(
    *,
    schema: type[DecisionT],
    node_name: str,
    state: CodingAgentState,
    system_prompt: str,
    user_prompt: str,
    max_attempts: int = LLM_DECISION_MAX_ATTEMPTS,
) -> DecisionT:
    """
    Calls the model as a normal chat completion and parses the response locally.

    This intentionally avoids llm.with_structured_output(...), because Groq may
    convert that into provider tool-calling. The graph runner owns all repository
    operations; the model only returns a structured decision object.
    """
    parser = PydanticOutputParser(pydantic_object=schema)
    last_error: Exception | None = None

    for attempt in range(1, max_attempts + 1):
        retry_feedback = ""

        if last_error is not None:
            retry_feedback = (
                "\n\nPrevious attempt failed to parse or execute cleanly. "
                "Fix the response and return only the structured object.\n"
                f"Parser/runtime error:\n{last_error}"
            )

        response = llm.invoke(
            [
                (
                    "system",
                    f"{system_prompt}\n\n"
                    "Do not call tools or functions. The LangGraph runner executes "
                    "all repository operations after your response is parsed.\n"
                    "Return only the structured object requested below. Do not wrap "
                    "the response in markdown fences.\n\n"
                    f"{parser.get_format_instructions()}",
                ),
                ("human", f"{user_prompt}{retry_feedback}"),
            ],
            config=node_config(
                node_name,
                state,
                {"llm_attempt": attempt, "llm_max_attempts": max_attempts},
            ),
        )

        try:
            return parser.parse(message_content_to_text(response.content))
        except Exception as exc:
            last_error = exc

    raise RuntimeError(
        f"{node_name} LLM decision failed after {max_attempts} attempts: {last_error}"
    )
