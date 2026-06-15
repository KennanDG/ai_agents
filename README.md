# ai_agents

A personal agentic AI project focused on a **LangGraph-based coding agent** that can inspect a repository, plan changes, gather context, propose or apply patches, run validation, and produce a run report.

The broader repo also contains RAG infrastructure, but the current main focus of the project is the coding agent under `ai_agents.agents.coding`.


---

## Planned Future Development

- [ ] Improve the frontend/UI for running the coding agent.
- [ ] Expand the skill library.
- [ ] Improve validation profiles.
- [ ] Add stronger evaluation coverage.
- [ ] Refine long-term memory behavior.
- [ ] Add deployment documentation.
- [ ] Create a Voice Agent for verbal queries. 

---

## Current Focus: Coding Agent

The coding agent is designed to work like a careful repository assistant. It reads the codebase, routes the request to an appropriate skill, gathers relevant files, creates targeted edits, validates the result, and reports what happened.

By default, the agent runs in **dry-run mode**. It only writes files when explicitly called with `--write`.

### Current workflow

```text
User request
  -> skill routing
  -> long-term memory recall
  -> planning
  -> optional web search routing
  -> repo navigator
  -> context gathering
  -> patch generation
  -> validation
  -> report
  -> long-term memory save
```

---

## Coding Agent Functionality

- **Skill routing**
  - Selects the best coding skill for the request.
  - Supports general implementation, debugging, testing, web-search-backed tasks, and a placeholder Gmail-access route.
  - Falls back to deterministic routing if model routing fails.

- **Planning**
  - Produces a concise implementation plan.
  - Generates structured repository search requests.
  - Selects validation commands when useful.

- **Repository navigation**
  - Uses a read-only repo navigator node to choose the most relevant files before patching.
  - Supports follow-up searches when the first pass is not enough.
  - Avoids logs, cache files, build artifacts, environment files, and other unsafe context.

- **Structured repository search**
  - Supports search modes such as `all`, `any`, `exact`, and `symbol`.
  - Supports path includes, path excludes, and file extension filters.
  - Ranks and formats results before they are passed into the context-selection flow.

- **Context gathering**
  - Reads a small set of relevant files instead of dumping the whole repo into context.
  - Adds retrieved long-term memories and optional web search results to the working context.

- **Patch generation**
  - Produces structured file edits.
  - Supports exact text replacement for existing files.
  - Supports creating new files when the request and inspected context justify it.
  - Uses idempotent handling for create operations that were already successfully applied.
  - Blocks unsafe write paths such as secrets, environment files, generated files, caches, and logs.

- **Validation**
  - Runs safe validation commands through the coding agent validation runner.
  - Can use requested validation commands or project defaults.
  - Reports failures honestly instead of claiming success when validation did not pass.

- **Reports**
  - Prints a concise final report.
  - Can write a detailed Markdown report with the plan, search requests, memory status, inspected files, diffs, validation output, and errors.

- **Persistent memory**
  - Uses LangGraph checkpointers for thread-scoped continuity.
  - Uses LangGraph Store for cross-thread coding memories.
  - Supports semantic memory when embeddings are configured.
  - Saves a compact memory after each run, including request, selected skill, inspected files, changed files, validation, and patch summary.

- **Observability**
  - Uses LangSmith tracing around the coding agent run.
  - Adds node-specific run names, tags, and metadata for easier debugging.

---

## Dependencies

### Core runtime

- Python 3.11+
- `uv`
- `python-dotenv`
- `pydantic`
- LangChain
- LangGraph
- LangSmith

### Model providers

- Groq via `langchain-groq`
- OpenRouter-compatible chat models via `langchain-openai`
- DeepSeek/OpenRouter-style reasoning model configuration

### Coding-agent persistence

- Postgres
- `psycopg`
- `langgraph-checkpoint-postgres`
- Optional semantic embeddings through:
  - `langchain-google-genai`, or
  - `langchain.embeddings.init_embeddings`

### RAG subsystem

- Ollama for local chat and embedding models
- Qdrant for vector search
- Postgres for metadata and idempotency
- SQLAlchemy

---

## Environment Variables

Create a `.env` file in the project root.

```env
# LangSmith tracing
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=your_langsmith_key
LANGCHAIN_PROJECT=ai-agents-dev
LANGSMITH_ENDPOINT=https://api.smith.langchain.com

# Model providers
GROQ_API_KEY=your_groq_key
OPEN_ROUTER_API_KEY=your_openrouter_key
DEEPSEEK_API_KEY=your_deepseek_key

# Coding agent memory
CODING_AGENT_MEMORY_DB_URI=postgresql://user:password@localhost:5432/ai_agents
CODING_AGENT_MEMORY_ENABLED=true
CODING_AGENT_MEMORY_SETUP=false
CODING_AGENT_MEMORY_USER_ID=default
CODING_AGENT_MEMORY_NAMESPACE=default
CODING_AGENT_MEMORY_SEARCH_LIMIT=5

# Semantic memory
GOOGLE_API_KEY=your_google_api_key
CODING_AGENT_MEMORY_SEMANTIC=true
EMBEDDING_MODEL=google_genai:gemini-embedding-2
CODING_AGENT_MEMORY_EMBEDDING_DIMS=768
CODING_AGENT_MEMORY_INDEX_FIELDS=text,request,summary

# RAG
OLLAMA_HOST=http://localhost:11434
CHAT_MODEL=llama3.1:8b
QDRANT_URL=http://localhost:6333
QDRANT_COLLECTION=rag-default
DATABASE_URL=postgresql+psycopg://ai_agents:ai_agents@localhost:5432/ai_agents
K=8
```

If you do not want persistent memory for a run, use `--no-memory`.

---

## CLI Usage

Run the agent in dry-run mode:

```bash
uv run python -m ai_agents.agents.coding.main \
  --repo-root ./src/ai_agents \
  --workspace-root . \
  "Find where the coding agent validates patches and explain how it works"
```

Allow the agent to write changes:

```bash
uv run python -m ai_agents.agents.coding.main \
  --repo-root ./src/ai_agents \
  --workspace-root . \
  --write \
  "Fix the reporting node so it does not claim files were created when a retry only hit an idempotent no-op"
```

Write a Markdown run report:

```bash
uv run python -m ai_agents.agents.coding.main \
  --repo-root ./src/ai_agents \
  --workspace-root . \
  --write \
  --markdown-report \
  "Add tests for the structured repository search service"
```

Initialize memory tables before running:

```bash
uv run python -m ai_agents.agents.coding.main \
  --repo-root ./src/ai_agents \
  --workspace-root . \
  --setup-memory \
  "Smoke test memory setup"
```

Reuse a checkpoint thread:

```bash
uv run python -m ai_agents.agents.coding.main \
  --repo-root ./src/ai_agents \
  --workspace-root . \
  --thread-id coding-run-example \
  "Continue the previous coding-agent task"
```

Disable memory for a single run:

```bash
uv run python -m ai_agents.agents.coding.main \
  --repo-root ./src/ai_agents \
  --workspace-root . \
  --no-memory \
  "Run this task without recalling or saving long-term memory"
```

### CLI options

```text
--repo-root            Repository root to inspect and patch.
--workspace-root       Project root used for validation commands.
--write                Allow the agent to write changes. Omit for dry-run mode.
--markdown-report      Write a detailed Markdown report under logs/runs.
--report-path          Custom Markdown report path.
--thread-id            Optional LangGraph checkpoint thread id.
--memory-user-id       User id segment for cross-thread memory namespaces.
--memory-namespace     Namespace segment for grouping coding memories.
--setup-memory         Run Postgres checkpointer/store setup before invoking the graph.
--no-memory            Disable persistent memory for the run.
```

---

## Recommended Local Setup

Install dependencies:

```bash
uv sync
```

Run Postgres if using persistent memory:

```bash
podman-compose up -d postgres  # or use docker compose
```

Run the memory setup once:

```bash
uv run python -m ai_agents.agents.coding.main \
  --repo-root ./src/ai_agents \
  --workspace-root . \
  --setup-memory \
  "Smoke test memory setup"
```

Then run normal coding-agent tasks with or without `--write`.

---


## RAG Subsystem

The RAG portion of this project is still part of the broader platform. It provides document ingestion, chunking, embeddings, retrieval, and grounded answering.

### RAG goals

- Load documents from local files.
- Convert PDFs to text or Markdown before ingestion.
- Chunk documents deterministically.
- Store metadata and hashes in Postgres.
- Store vectors in Qdrant.
- Avoid duplicate embeddings when source content has not changed.
- Answer questions only from retrieved context.

### Example ingestion

```python
from ai_agents.rag.ingest import ingest_files
from ai_agents.rag.settings import RagSettings

settings = RagSettings()
count = ingest_files(["docs/example.md"], settings)
print(f"Ingested {count} chunks")
```

### Example query

```python
from ai_agents.rag.query import answer
from ai_agents.rag.settings import RagSettings

settings = RagSettings()
response = answer("What is this document about?", settings)
print(response)
```

---

## Design Principles

- Inspect before editing.
- Keep patches small and targeted.
- Prefer structured model outputs over free-form text.
- Keep repository operations in deterministic runner code.
- Treat memory as helpful context, not unquestioned truth.
- Validate before reporting success.
- Make dry-run mode the default.

---

## Status

Active development. Expect breaking changes while the coding agent, memory system, and RAG infrastructure continue to evolve.

---

## License

MIT
