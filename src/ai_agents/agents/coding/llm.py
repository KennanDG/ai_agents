from __future__ import annotations

from typing import TypeVar

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel

from ai_agents.agents.coding.runtime import node_config
from ai_agents.agents.coding.state import CodingAgentState
from ai_agents.agents.coding.utils.text import message_content_to_text


DecisionT = TypeVar("DecisionT", bound=BaseModel)


def invoke_parsed_decision(
    *,
    model: BaseChatModel,
    schema: type[DecisionT],
    node_name: str,
    state: CodingAgentState,
    system_prompt: str,
    user_prompt: str,
    max_attempts: int = 1,
) -> DecisionT:
    """Invoke a model and parse one bounded structured decision.

    Graph-level retries and provider retries already exist. Keeping parser retries at one
    by default prevents a malformed response from multiplying latency across every node.
    The patch node may explicitly request two attempts because it is the only expensive
    decision that directly produces repository edits.
    """

    parser = PydanticOutputParser(pydantic_object=schema)
    last_error: Exception | None = None

    for attempt in range(1, max_attempts + 1):
        retry_feedback = ""
        if last_error is not None:
            retry_feedback = (
                "\n\nPrevious response could not be parsed. Return only the corrected "
                "structured object.\n"
                f"Parser/runtime error:\n{last_error}"
            )

        response = model.invoke(
            [
                (
                    "system",
                    f"{system_prompt}\n\n"
                    "Do not call tools or functions. The LangGraph runner executes "
                    "repository operations after parsing your response.\n"
                    "Return only the requested structured object without markdown fences.\n\n"
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
