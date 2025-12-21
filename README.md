# ai_agents

A personal, long-term project to build an **agentic AI platform**: reusable AI assistants, tools, and workflows (eventually with RAG, memory, and deployment).

> Goal: a robust framework that can power multiple assistants (personal + project-specific) with shared infrastructure.

---

## What this repo is (and isn’t)

### ✅ This repo *is* for
- Building an agentic framework (multi-agent or single-agent + tools)
- Adding RAG (knowledge base + retrieval + citations)
- Integrations/tools (webhooks, APIs, DB, vector store, etc.)
- Evaluation + testing (so agents don’t regress)
- Deployment-ready structure

### ❌ This repo is *not* for
- Storing secrets or private keys
- Storing large datasets or embeddings directly in git

---

## Planned Architecture (high level)

**Core concepts**
- **Agents**: “brains” with role instructions + tool access
- **Tools**: typed functions the agent can call (DB, web, filesystem, APIs)
- **RAG**: ingestion → chunking → embeddings → retrieval → grounded answers
- **Memory**: optional durable user preferences + session state
- **Orchestration**: routing, guardrails, retries, tool error handling


---

## Getting Started

### Prereqs
- Git
- Python 3.11+ (recommended)
- (Optional) Node 18+ if you add a web UI later

### Setup (Python-first)
```bash
# from repo root
python -m venv .venv
source .venv/bin/activate  # (WSL/Linux/macOS)
# .venv\Scripts\activate   # (Windows PowerShell)

pip install -U pip