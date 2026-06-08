from __future__ import annotations

import os
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI

from ai_agents.config.settings import settings as config_settings


load_dotenv()


model = ChatGroq(
    model=config_settings.coding_model,
    api_key=config_settings.resolved_groq_api_key(),
)

reasoning_model = ChatOpenAI(
    model=config_settings.reasoning_model,
    api_key=os.environ["OPEN_ROUTER_API_KEY"],
    base_url=os.environ.get("OPEN_ROUTER_URL", "https://openrouter.ai/api/v1"),
    max_retries=2,
)


def invoke_parsed_decision(model, schema, node_name, state, system_prompt, user_prompt):
    messages = [
        ("system", system_prompt),
        ("human", user_prompt),
    ]
    return model.with_structured_output(schema).invoke(messages)
