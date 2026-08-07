import os

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import List, Literal

from .secrets import get_secret_json


class Settings(BaseSettings):

    def resolved_groq_api_key(self) -> str | None:
        if self.groq_api_key:
            return self.groq_api_key
        
        if self.groq_secret_arn:
            self.groq_api_key = get_secret_json(self.groq_secret_arn).get("GROQ_API_KEY")
            return self.groq_api_key
        
        return None


    def resolved_deepseek_api_key(self) -> str | None:
        if self.deepseek_api_key:
            return self.deepseek_api_key

        if self.deepseek_secret_arn:
            self.deepseek_api_key = get_secret_json(self.deepseek_secret_arn).get("DEEPSEEK_API_KEY")
            return self.deepseek_api_key

        return None


    def resolved_openrouter_api_key(self) -> str | None:
        if self.openrouter_api_key:
            return self.openrouter_api_key

        if self.openrouter_secret_arn:
            self.openrouter_api_key = get_secret_json(self.openrouter_secret_arn).get("OPENROUTER_API_KEY")
            return self.openrouter_api_key

        return None


    def resolved_openai_api_key(self) -> str | None:
        if self.openai_api_key:
            return self.openai_api_key

        if self.openai_secret_arn:
            self.openai_api_key = get_secret_json(self.openai_secret_arn).get("OPENAI_API_KEY")
            return self.openai_api_key

        return None


    def resolved_anthropic_api_key(self) -> str | None:
        if self.anthropic_api_key:
            return self.anthropic_api_key

        if self.anthropic_secret_arn:
            self.anthropic_api_key = get_secret_json(self.anthropic_secret_arn).get("ANTHROPIC_API_KEY")
            return self.anthropic_api_key

        return None


    def resolved_qdrant_api_key(self) -> str | None:
        if self.qdrant_api_key:
            return self.qdrant_api_key
        
        if self.qdrant_secret_arn:
            self.qdrant_api_key = get_secret_json(self.qdrant_secret_arn).get("QDRANT_API_KEY")
            return self.qdrant_api_key
        
        return None
    

    def resolved_langchain_api_key(self) -> str | None:
        if self.langchain_api_key:
            return self.langchain_api_key
        
        if self.langchain_secret_arn:
            self.langchain_api_key = get_secret_json(self.langchain_secret_arn).get("LANGCHAIN_API_KEY")
            return self.langchain_api_key
        
        return None
    

    def resolved_jina_api_key(self) -> str | None:
        if self.jina_api_key:
            return self.jina_api_key
        
        if self.jina_secret_arn:
            self.jina_api_key = get_secret_json(self.jina_secret_arn).get("JINA_API_KEY")
            return self.jina_api_key
        
        return None
    


    def resolved_ai_agents_api_key(self) -> str | None:
        if self.ai_agents_api_key:
            return self.ai_agents_api_key
        
        if self.ai_agents_secret_arn:
            self.ai_agents_api_key = get_secret_json(self.ai_agents_secret_arn).get("AI_AGENTS_API_KEY")
            return self.ai_agents_api_key
        
        return None
    

    
    def resolved_github_token(self) -> str | None:
        if self.github_token:
            return self.github_token

        if self.github_secret_arn:
            self.github_token = get_secret_json(self.github_secret_arn).get("GITHUB_TOKEN")
            return self.github_token

        return None


    def resolved_google_api_key(self) -> str | None:
        if self.google_api_key:
            return self.google_api_key

        if self.google_secret_arn:
            self.google_api_key = get_secret_json(self.google_secret_arn).get("GOOGLE_API_KEY")
            return self.google_api_key

        return None

    model_config = SettingsConfigDict(
        env_file=os.getenv("ENV_FILE", ".env"),
        extra="ignore"
        )


    # App API key
    ai_agents_api_key: str | None = Field(default=None, alias="AI_AGENTS_API_KEY")
    ai_agents_secret_arn: str | None = Field(default=None, alias="AI_AGENTS_SECRET_ARN")

    # GitHub repository integration
    github_token: str | None = Field(default=None, alias="GITHUB_TOKEN")
    github_secret_arn: str | None = Field(default=None, alias="GITHUB_SECRET_ARN")
    github_token_kind: Literal["user", "installation"] = Field(
        default="user",
        alias="GITHUB_TOKEN_KIND",
    )
    github_api_url: str = Field(default="https://api.github.com", alias="GITHUB_API_URL")
    github_api_version: str = Field(default="2026-03-10", alias="GITHUB_API_VERSION")
    github_workspace_root: str = Field(
        default=".ai-agents/github-workspaces",
        alias="GITHUB_WORKSPACE_ROOT",
    )
    github_timeout_seconds: int = Field(default=120, alias="GITHUB_TIMEOUT_SECONDS")
    github_commit_author_name: str = Field(
        default="AI Agents",
        alias="GITHUB_COMMIT_AUTHOR_NAME",
    )
    github_commit_author_email: str = Field(
        default="ai-agents@users.noreply.github.com",
        alias="GITHUB_COMMIT_AUTHOR_EMAIL",
    )
    github_allow_default_branch_push: bool = Field(
        default=False,
        alias="GITHUB_ALLOW_DEFAULT_BRANCH_PUSH",
    )
    github_max_commit_files: int = Field(
        default=100,
        ge=1,
        le=500,
        alias="GITHUB_MAX_COMMIT_FILES",
    )
    github_max_file_size_bytes: int = Field(
        default=5_000_000,
        ge=1,
        alias="GITHUB_MAX_FILE_SIZE_BYTES",
    )
    github_blocked_path_patterns: List[str] = Field(
        default_factory=lambda: [
            ".env",
            ".env.*",
            "*.pem",
            "*.key",
            "id_rsa",
            "id_ed25519",
            "*credentials*",
            "*secrets*",
        ],
        alias="GITHUB_BLOCKED_PATH_PATTERNS",
    )

    # LangChain
    langchain_api_key: str | None = Field(default=None, alias="LANGCHAIN_API_KEY")
    langsmith_api_url: str | None = Field(default="https://api.smith.langchain.com", alias="LANGCHAIN_ENDPOINT")
    langchain_secret_arn: str | None = Field(default=None, alias="LANGCHAIN_SECRET_ARN")
    langchain_project : str = Field(default="ai-agents-dev", alias="LANGCHAIN_PROJECT")


    # Chat model routing. Model IDs can be overridden by the runtime admin API.
    coding_provider: Literal["groq", "deepseek", "openrouter", "openai", "anthropic", "google"] = Field(
        default="groq",
        alias="CODING_PROVIDER",
    )
    reasoning_provider: Literal["groq", "deepseek", "openrouter", "openai", "anthropic", "google"] = Field(
        default="deepseek",
        alias="REASONING_PROVIDER",
    )
    caption_provider: Literal["groq", "openrouter", "openai", "anthropic", "google"] = Field(
        default="groq",
        alias="CAPTION_PROVIDER",
    )
    voice_chat_provider: Literal["groq", "deepseek", "openrouter", "openai", "anthropic", "google"] = Field(
        default="groq",
        alias="VOICE_CHAT_PROVIDER",
    )
    voice_stt_provider: Literal["groq", "openai"] = Field(
        default="groq",
        alias="VOICE_STT_PROVIDER",
    )
    voice_tts_provider: Literal["groq", "openai"] = Field(
        default="groq",
        alias="VOICE_TTS_PROVIDER",
    )

    # Groq
    chat_model: str = Field(default="llama-3.1-8b-instant", alias="CHAT_MODEL")                   
    query_model: str = Field(default="llama-3.1-8b-instant", alias="QUERY_MODEL")         
    caption_model: str = Field(default="meta-llama/llama-4-scout-17b-16e-instruct", alias="CAPTION_MODEL")  # VLM
    verify_model: str = Field(default="llama-3.1-8b-instant", alias="VERIFY_MODEL")
    verify_docs_model: str = Field(default="llama-3.1-8b-instant", alias="VERIFY_DOCS_MODEL")
    coding_model: str = Field(default="openai/gpt-oss-120b", alias="CODING_MODEL") 
    reasoning_model: str = Field(default="deepseek-v4-pro", alias="REASONING_MODEL") 
    groq_api_key: str | None = Field(default=None, alias="GROQ_API_KEY")
    groq_api_url: str = Field(default="https://api.groq.com/openai/v1", alias="GROQ_URL")
    groq_secret_arn: str | None = Field(default=None, alias="GROQ_SECRET_ARN")

    # OpenAI-compatible chat providers used by the coding and reasoning slots.
    deepseek_api_key: str | None = Field(default=None, alias="DEEPSEEK_API_KEY")
    deepseek_api_url: str = Field(default="https://api.deepseek.com", alias="DEEPSEEK_URL")
    deepseek_secret_arn: str | None = Field(default=None, alias="DEEPSEEK_SECRET_ARN")

    openrouter_api_key: str | None = Field(default=None, alias="OPENROUTER_API_KEY")
    openrouter_api_url: str = Field(default="https://openrouter.ai/api/v1", alias="OPENROUTER_URL")
    openrouter_secret_arn: str | None = Field(default=None, alias="OPENROUTER_SECRET_ARN")

    openai_api_key: str | None = Field(default=None, alias="OPENAI_API_KEY")
    openai_api_url: str = Field(default="https://api.openai.com/v1", alias="OPENAI_URL")
    openai_secret_arn: str | None = Field(default=None, alias="OPENAI_SECRET_ARN")

    anthropic_api_key: str | None = Field(default=None, alias="ANTHROPIC_API_KEY")
    anthropic_api_url: str = Field(default="https://api.anthropic.com", alias="ANTHROPIC_URL")
    anthropic_secret_arn: str | None = Field(default=None, alias="ANTHROPIC_SECRET_ARN")    
    google_api_key: str | None = Field(default=None, alias="GOOGLE_API_KEY")
    google_api_url: str = Field(default="https://generativelanguage.googleapis.com/v1beta", alias="GOOGLE_URL")
    google_secret_arn: str | None = Field(default=None, alias="GOOGLE_SECRET_ARN")

    # Non-secret runtime model selections are persisted here. Provider secrets remain
    # in environment/Secrets Manager or in the current backend process only.
    runtime_agent_config_path: str = Field(
        default=".ai-agents/runtime-agent-config.json",
        alias="AI_AGENTS_RUNTIME_CONFIG_PATH",
    )

    # Coding-agent execution profile. These defaults can be changed through the
    # admin UI and are applied to new runs. Hard request bounds remain server-side.
    coding_subagent_count: int = Field(
        default=3, ge=1, le=6, alias="CODING_AGENT_MAX_CONTEXT_WORKERS"
    )
    coding_route_max_tokens: int = Field(
        default=700, ge=256, le=2_000, alias="CODING_AGENT_ROUTE_MAX_TOKENS"
    )
    coding_planner_max_tokens: int = Field(
        default=2_400, ge=512, le=6_000, alias="CODING_AGENT_PLANNER_MAX_TOKENS"
    )
    coding_repo_navigation_max_tokens: int = Field(
        default=1_600, ge=512, le=4_000,
        alias="CODING_AGENT_REPO_NAVIGATION_MAX_TOKENS",
    )
    coding_simple_patch_max_tokens: int = Field(
        default=6_000, ge=2_000, le=16_000,
        alias="CODING_AGENT_SIMPLE_PATCH_MAX_TOKENS",
    )
    coding_patch_max_tokens: int = Field(
        default=12_000, ge=4_000, le=32_000, alias="CODING_AGENT_PATCH_MAX_TOKENS"
    )
    coding_progress_max_tokens: int = Field(
        default=1_200, ge=512, le=4_000, alias="CODING_AGENT_PROGRESS_MAX_TOKENS"
    )
    

    # Qdrant
    qdrant_url: str = Field(default="http://localhost:6333", alias="QDRANT_URL")
    qdrant_api_key: str | None = Field(default=None, alias="QDRANT_API_KEY")
    qdrant_secret_arn: str | None = Field(default=None, alias="QDRANT_SECRET_ARN")
    qdrant_collection: str = Field(default="rag-default", alias="QDRANT_COLLECTION")

    # FastEmbed
    
    rerank_device: str = Field(default="cpu", alias="RERANK_DEVICE") 


    # Jina AI
    embedding_model: str = Field(default="google_genai:gemini-embedding-2", alias="EMBEDDING_MODEL")    # Doc embedding
    rerank_model: str = Field(default="jina-reranker-v3", alias="RERANK_MODEL")
    jina_api_key: str | None = Field(default=None, alias="JINA_API_KEY")
    jina_api_url: str | None = Field(default="https://api.jina.ai/v1", alias="JINA_URL")
    jina_secret_arn: str | None = Field(default=None, alias="JINA_SECRET_ARN")


    # DB
    database_url: str = Field(
        default="postgresql+psycopg://ai_agents:ai_agents@localhost:5432/ai_agents",
        alias="DATABASE_URL",
    )

    # Retrieval
    k: int = Field(default=8, alias="K")
    candidate_k: int = Field(default=30, alias="CANDIDATE_K")   # docs kept after RRF before rerank
    k_per_query: int = Field(default=8, alias="K_PER_QUERY")    # docs retrieved per expanded query
    rrf_k: int = Field(default=60, alias="RRF_K")               # RRF constant
    min_docs_for_success: int = Field(default=2, alias="MIN_DOCS_FOR_SUCCESS")
    max_collection_fallbacks: int = Field(default=3, alias="MAX_COLLECTION_FALLBACKS")
    retrieve_workers: int = Field(default=8, alias="RETRIEVE_WORKERS")
    preferred_collections: List[str] = Field(default=["rag-engineering", "rag-robotics", "rag-cs"], alias="PREFERRED_COLLECTIONS")
    enable_parallel_collection_retrieval: bool = Field(default=True, alias="ENABLE_PARALLEL_COLLECTION_RETRIEVAL")
    parallel_collection_workers: int = Field(default=2, alias="PARALLEL_COLLECTION_WORKERS")
    
    n_query_expansions: int = Field(default=2, alias="N_QUERY_EXPANSIONS")
    enable_query_expansion: bool = Field(default=True, alias="ENABLE_QUERY_EXPANSION")
    min_question_chars_for_expansion: int = Field(default=25, alias="MIN_QUESTION_CHARS_FOR_EXPANSION")


    # Generation
    max_rag_attempts: int = Field(default=2, alias="MAX_RAG_ATTEMPTS")
    retrieve_attempts: int = Field(default=2, alias="RETRIEVE_ATTEMPTS")
    generate_attempts: int = Field(default=2, alias="GENERATE_ATTEMPTS")
    verify_attempts: int = Field(default=2, alias="VERIFY_ATTEMPTS")
    verify_max_chars: int = Field(default=6_000, alias="VERIFY_MAX_CHARS")
    verify_docs_attempts: int = Field(default=2, alias="VERIFY_DOCS_ATTEMPTS")
    verify_docs_max_chars: int = Field(default=6_000, alias="VERIFY_DOCS_MAX_CHARS")

    # Voice Agent
    voice_stt_model: str = Field(default="whisper-large-v3-turbo", alias="VOICE_STT_MODEL")
    voice_chat_model: str = Field(default="llama-3.1-8b-instant", alias="VOICE_CHAT_MODEL")
    voice_chat_max_tokens: int = Field(default=2_048, alias="VOICE_CHAT_MAX_TOKENS")
    voice_tts_model: str = Field(default="canopylabs/orpheus-v1-english", alias="VOICE_TTS_MODEL")
    voice_tts_voice: str = Field(default="hannah", alias="VOICE_TTS_VOICE")
    voice_tts_enabled: bool = Field(default=True, alias="VOICE_TTS_ENABLED")
    voice_tts_max_chars: int = Field(default=200, alias="VOICE_TTS_MAX_CHARS")
    voice_max_clarifications: int = Field(default=2, alias="VOICE_MAX_CLARIFICATIONS")
    voice_max_audio_mb: int = Field(default=15, alias="VOICE_MAX_AUDIO_MB")


settings = Settings()