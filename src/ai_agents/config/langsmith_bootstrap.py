# ai_agents/config/langsmith_bootstrap.py
import os
from ai_agents.config.settings import settings

def ensure_langsmith_env() -> None:
    # Resolve key (env first, secret fallback)
    api_key = settings.resolved_langchain_api_key()

    if api_key:
        os.environ["LANGCHAIN_API_KEY"] = api_key

    # Endpoint: LangSmith prefers LANGCHAIN_ENDPOINT
    # Your Settings currently uses LANGSMITH_API_URL, so support both.
    endpoint = getattr(settings, "langsmith_api_url", None) or os.environ.get("LANGCHAIN_ENDPOINT")

    if endpoint:
        os.environ["LANGCHAIN_ENDPOINT"] = endpoint

    # Force tracing on (must be STRING in Lambda/ECS env anyway)
    os.environ.setdefault("LANGCHAIN_TRACING_V2", "true")
