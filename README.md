# ai_agents

A personal, long-term project to build a **production-grade agentic AI platform**: reusable AI assistants, tools, and workflows with Retrieval-Augmented Generation (RAG), memory, evaluation, and deployment in mind.

> **Goal:** A robust, modular framework that can power multiple assistants (personal + project-specific) with shared infrastructure and strong guarantees around grounding, idempotency, and observability.

---

## What this repo is (and isn’t)

### ✅ This repo *is* for
- Building agentic systems (single-agent or multi-agent)
- Tool-driven agents (APIs, DBs, webhooks, vector stores)
- End-to-end RAG pipelines (ingestion → chunking → embeddings → retrieval → verification)
- Idempotent ingestion with Postgres + Qdrant
- Evaluation, tracing, and debugging via LangSmith
- Local-first development with Ollama
- Deployment-ready structure (Docker / devcontainer)

### ❌ This repo is *not* for
- Storing secrets or private keys in git
- Storing large datasets or embeddings directly in the repo
- One-off scripts with no path to reuse or production

---

## High-Level Architecture

```
┌──────────────┐
│   User / UI  │
└──────┬───────┘
       │
       ▼
┌──────────────┐
│   Agent(s)   │  ← role + tools + guardrails
└──────┬───────┘
       │
       ▼
┌──────────────┐
│ RAG Pipeline │
│              │
│  Loaders     │ → Text / PDF / future: HTML, images
│  Splitter    │ → deterministic chunking
│  Embeddings  │ → Ollama
│  Vector DB   │ → Qdrant
│  Metadata DB │ → Postgres
└──────┬───────┘
       │
       ▼
┌──────────────┐
│   LLM (Chat) │ → grounded answer only
└──────────────┘
```

### Core concepts
- **Agents** – LLMs with role instructions + tool access
- **Tools** – Typed, testable functions (DB, APIs, filesystem, etc.)
- **RAG** – Deterministic ingestion and retrieval with provenance
- **Memory** – Durable state (planned / incremental)
- **Orchestration** – Retry loops, verification, routing, guardrails
- **Observability** – Tracing with LangSmith (`@traceable` everywhere)

---

## Tech Stack

- **Python 3.11+**
- **LangChain / LangGraph** (agent + RAG orchestration)
- **Ollama** (local LLMs + embeddings)
- **Qdrant** (vector database)
- **Postgres** (metadata + idempotency)
- **SQLAlchemy** (ORM)
- **Pydantic v2** (configs + schemas)
- **LangSmith** (tracing & evaluation)
- **Docker / Devcontainers** (reproducible dev)

---

## Repository Structure

```
ai_agents/
├─ src/ai_agents/
│  ├─ config/         # global settings & constants
│  ├─ rag/            # RAG pipeline (load, split, embed, retrieve)
│  ├─ db/             # Postgres models & session
│  ├─ agents/         # (future) agent definitions
│  └─ tools/          # (future) reusable tools
│
├─ docker-compose.yml # Qdrant, Postgres
├─ devcontainer.json  # VS Code devcontainer
├─ README.md
└─ .env.example
```

---

## Getting Started

### Prerequisites
- Git
- Docker + Docker Compose
- Python 3.11+
- VS Code (recommended, for devcontainer)

---

## Option A: Devcontainer (Recommended)

This is the **primary supported workflow**.

1. Install **VS Code** + **Dev Containers** extension
2. Clone the repo
3. Open in VS Code → *Reopen in Container*

The container will automatically:
- Start Postgres, Qdrant, and Ollama
- Install Python deps via `uv`
- Configure the Python interpreter and paths

---

## Option B: Local Python (Advanced)

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip uv
uv sync
```

You must also run Postgres & Qdrant yourself (Docker recommended).

---

## Environment Variables

Create a `.env` (or `.env.docker`) file:

```env
# Ollama
OLLAMA_HOST=http://localhost:11434
CHAT_MODEL=llama3.1:8b
EMBEDDING_MODEL=nomic-embed-text

# Qdrant
QDRANT_URL=http://localhost:6333
QDRANT_COLLECTION=rag-default

# Database
DATABASE_URL=postgresql+psycopg://ai_agents:ai_agents@localhost:5432/ai_agents

# Retrieval
K=8
```

---

## RAG: Ingesting Documents

```python
from ai_agents.rag.ingest import ingest_files
from ai_agents.rag.settings import RagSettings

settings = RagSettings()
count = ingest_files(["docs/example.md"], settings)
print(f"Ingested {count} chunks")
```

### What ingestion guarantees
- Stable `source_uri` per file
- Deterministic chunk indices per source
- Content hashing (file + chunk)
- Idempotency (no re-embedding if unchanged)
- Postgres is the source of truth; Qdrant mirrors it

---

## Asking Questions (RAG)

```python
from ai_agents.rag.query import answer
from ai_agents.rag.settings import RagSettings

settings = RagSettings()
response = answer("What is this document about?", settings)
print(response)
```

The model is instructed to answer **only from retrieved context**.

---

## PDF Support

Basic PDF → Markdown conversion is supported:

```python
from ai_agents.utils.pdf_to_markdown import pdf_to_markdown, PdfToMarkdownRequest

req = PdfToMarkdownRequest(pdf_path="paper.pdf")
res = pdf_to_markdown(req)
print(res.markdown)
```

Useful for converting PDFs before ingestion.

---

## Observability & Tracing

All critical paths are decorated with `@traceable`:
- Ingestion
- Retrieval
- Chain execution
- Embeddings

Use **LangSmith** to:
- Inspect latency bottlenecks
- Debug hallucinations
- Evaluate regressions

---

## Design Principles

- **Deterministic ingestion**: same input → same IDs
- **Separation of concerns**: Postgres = truth, Qdrant = index
- **Modular**: agents, tools, and chains are reusable
- **Production-minded**: observability, retries, verification

---

## Roadmap (Living)

- [ ] Agent orchestration graph (LangGraph)
- [ ] Answer verification + retry loops
- [ ] Query expansion & reranking
- [ ] Multi-modal ingestion (.pdf, .docx, images)
- [ ] Durable user memory
- [ ] Evaluation harness
- [ ] Deployment targets (AWS / self-hosted)

---

## Status

🚧 **Active development** — expect breaking changes.

This repo is intentionally opinionated and evolving toward a reusable, production-quality agentic platform.

---

## License

MIT 

