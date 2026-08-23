# ai_agents

`ai_agents` is a work in progress agent harness built around LangGraph, FastAPI, React, and Electron. The project currently centers on a repository-aware coding agent, a voice intake agent, a desktop/web workspace, GitHub source-control integration, configurable model routing, local persistent coding memory, and user-managed skills and tools.

The repository also contains a broader RAG subsystem. The coding/voice agent platform and the RAG stack share the repository, but they have different persistence and runtime requirements. The agentic platform will eventually be packaged into its own managed repository.

> **Project status:** active development. Interfaces, configuration fields, and agent workflows may change while the harness is being hardened.

---

## What Is Implemented Today

### Coding agent

- LangGraph workflow for routing, planning, repository navigation, context gathering, patch generation, validation, progress assessment, reporting, and memory.
- Multi-skill routing with up to three selected skills per run.
- Fast path for simple edits and parallel read-only context workers for broader tasks.
- Structured repository search and chunked context selection for large files.
- Human approval flow before generated changes are applied from the frontend.
- Dry-run support for CLI usage.
- Approved custom coding tools that can contribute bounded context before repository navigation.
- Local SQLite checkpoints and long-term memory.
- Local semantic memory with FastEmbed.
- LangSmith tracing.

### Voice agent

- Audio transcription and optional text-to-speech responses.
- Repository-aware intake and clarification flow.
- Attachment and image-caption context.
- Produces a detailed coding-agent request instead of directly modifying the repository.
- Provider-selectable chat, STT, and TTS models.
- Separate voice skill/tool directories exposed through the admin UI.

### Desktop/frontend

- React + TypeScript + Tailwind UI.
- Electron packaging for Windows, macOS, and Linux.
- Local repository selection with a desktop directory picker.
- Managed GitHub repository selection and checkout.
- Repository tree/file preview and coding-agent task workflow.
- Source-control page for branches, pull, commit, push, and pull requests.
- Agent Settings modal for model providers, models, credentials, token budgets, and sub-agent count.
- Skills and Tools page for custom skill authoring/import and custom tool review.

---

## High-Level Architecture

```text
React / Electron frontend
        |
        v
Local FastAPI backend
        |
        +-------------------------+
        |                         |
        v                         v
Coding Agent                 Voice Agent
LangGraph                    LangGraph intake
        |                         |
        |                         +--> STT / chat / optional TTS
        |
        +--> Skill Registry
        +--> Approved Tool Registry
        +--> Repository/Search runners
        +--> Validation runners
        +--> SQLite checkpointer
        +--> SQLite long-term store + FastEmbed
        |
        v
Local repo or managed GitHub checkout
```

The frontend does not receive provider secrets or the GitHub token. Backend services resolve credentials from environment variables, optional Secrets Manager configuration, or runtime-only secret updates.

---

# Coding Agent

The coding agent is designed to inspect before editing, keep changes targeted, validate its work, and require approval before frontend-generated patches are applied.

## Current workflow

```text
START
  |\
  | +--> skill routing
  | +--> long-term memory recall
  |          |
  +----------+
             v
            plan
             |
             +--> optional web search
             +--> optional Gmail/connector route
             |
             v
      approved custom tools
             |
             v
       repo navigator
             |
       dynamic fan-out
             v
   read-only context workers
             |
       dynamic fan-in
             v
       gather context
             |
             v
           patch
             |
             v
         validate
             |
             v
     assess progress
       |         |
       |         +--> gather more context / retry loop
       v
      report
       |
       v
   remember run
       |
      END
```

Routing and memory recall begin in the same LangGraph super-step. Context workers are deterministic/read-only workers; they do not edit the repository and do not make their own model calls.

## Coding-agent capabilities

### Skill routing

- Loads Markdown playbooks from the coding skill registry on every registry load.
- Supports multiple complementary skills in one run.
- Deterministic routing can identify common debugging, testing, frontend, repository, web-search, and connector-related requests.
- Custom skills can be selected explicitly by name and can also be ranked by name/purpose overlap.
- The default maximum is **3 selected skills**.
- Custom skill files are treated as data only; Markdown is never executed.

### Planning and task modes

The planner can classify work into a simple, standard, or parallel path. Broader tasks can be split into isolated repository concerns and assigned to read-only context workers.

### Repository navigation and context engineering

- Structured repository search instead of dumping the entire repository into the prompt.
- Large files can be retained at intake and reduced into relevant line windows later.
- Configurable limits for total context, per-file context, chunks, overlaps, worker counts, and attachments.
- Repository noise such as `.git`, caches, build output, virtual environments, and `node_modules` is excluded from normal inspection.

### Patch generation

- Structured file-change output.
- Supports exact replacements and new files when justified by inspected context.
- Uses a lower-token simple-patch path for narrow changes.
- Uses a larger patch budget for multi-file changes.
- Repository writes remain separated from model reasoning and are performed by deterministic runner code.

### Validation and progress assessment

- Runs focused validation commands through the validation runner.
- Tracks blocking versus advisory validation failures.
- Can request another context/navigation pass when validation or patch generation shows that more evidence is needed.
- Final reporting is expected to preserve failed validation instead of claiming success.

### Human approval

Frontend runs can return an approval-required state before generated repository changes are applied. The client can explicitly apply or reject the proposed changes through the coding-agent WebSocket lifecycle.

CLI runs remain dry-run by default and require `--write` to permit file writes.

---

# Persistent Coding Memory

Coding-agent persistence is now **local-first**.

## SQLite layout

The coding agent uses two SQLite databases:

```text
.ai-agents/memory/checkpoints.sqlite3
.ai-agents/memory/store.sqlite3
```

- **Checkpoint database** — LangGraph thread-scoped state/continuity.
- **Store database** — cross-thread coding memories.

The databases are separated to reduce lock contention between graph checkpointing and long-term-memory access.

## Local semantic memory

Semantic memory uses `FastEmbedEmbeddings` locally. The default model is:

```text
BAAI/bge-small-en-v1.5
```

Default embedding size: **384 dimensions**.

FastEmbed model files are cached under:

```text
.ai-agents/memory/fastembed-cache
```

Memory namespaces include the coding agent, user, logical scope, and a stable repository identifier. Repository identity is based on the original repository root rather than a temporary sandbox path.

### Coding-memory environment variables

```env
CODING_AGENT_MEMORY_DIR=.ai-agents/memory
CODING_AGENT_MEMORY_CHECKPOINT_DB=.ai-agents/memory/checkpoints.sqlite3
CODING_AGENT_MEMORY_STORE_DB=.ai-agents/memory/store.sqlite3
CODING_AGENT_MEMORY_ENABLED=true
CODING_AGENT_MEMORY_SETUP=true
CODING_AGENT_MEMORY_USER_ID=local
CODING_AGENT_MEMORY_NAMESPACE=default
CODING_AGENT_MEMORY_SEARCH_LIMIT=5

CODING_AGENT_MEMORY_SEMANTIC=true
CODING_AGENT_MEMORY_EMBEDDING_MODEL=BAAI/bge-small-en-v1.5
CODING_AGENT_MEMORY_EMBEDDING_DIMS=384
CODING_AGENT_MEMORY_EMBEDDING_CACHE_DIR=.ai-agents/memory/fastembed-cache
CODING_AGENT_MEMORY_INDEX_FIELDS=text,request,summary
```

> The coding agent no longer requires Postgres for its checkpoint/store memory. The separate RAG subsystem still has its own Qdrant/Postgres configuration.

---

# Agent Settings

The frontend Agent Settings modal configures model routing and coding execution limits for new runs.

## Supported providers

| Capability                | Supported providers                                   |
| ------------------------- | ----------------------------------------------------- |
| Coding chat               | Groq, DeepSeek, OpenRouter, OpenAI, Anthropic, Google |
| Reasoning chat            | Groq, DeepSeek, OpenRouter, OpenAI, Anthropic, Google |
| Vision / image captioning | Groq, OpenRouter, OpenAI, Anthropic, Google           |
| Voice chat                | Groq, DeepSeek, OpenRouter, OpenAI, Anthropic, Google |
| Speech-to-text            | Groq, OpenAI                                          |
| Text-to-speech            | Groq, OpenAI                                          |

Model selectors are capability-aware. When credentials are available, the backend attempts live account-aware model discovery. When live discovery is unavailable, the UI can use the backend fallback catalog.

## Provider credentials

The settings UI accepts API keys for:

- Groq
- DeepSeek
- OpenRouter
- OpenAI
- Anthropic
- Google

Secrets are **not returned to the renderer**. Runtime-entered secrets remain process/session-only unless you also configure them through environment variables or the repository's Secrets Manager integration.

Non-secret model selections are persisted by the backend under `.ai-agents/` so new runs can reuse the chosen configuration.

## Coding execution settings

| Setting                            | Default / recommended | Allowed range |
| ---------------------------------- | --------------------: | ------------: |
| Max sub-agent/context-worker count |                     3 |          1–6 |
| Router max tokens                  |                   700 |    256–2,000 |
| Planner max tokens                 |                 2,400 |    512–6,000 |
| Repo navigator max tokens          |                 1,600 |    512–4,000 |
| Simple patch max tokens            |                 6,000 | 2,000–16,000 |
| Standard patch max tokens          |                12,000 | 4,000–32,000 |
| Progress max tokens                |                 1,200 |    512–4,000 |

These values can also be supplied as per-run API overrides, but the backend enforces the same hard bounds.

The sub-agent setting controls parallel **read-only context workers**, not independent patch-writing agents. Increasing it is most useful when a task can be decomposed into independent repository concerns.

## Prompt caching and latency controls

The coding-agent settings also support:

```env
CODING_AGENT_FAST_PATH_ENABLED=true
CODING_AGENT_LLM_SKILL_ROUTING_ENABLED=false
CODING_AGENT_LLM_NAVIGATION_ENABLED=false
CODING_AGENT_MODEL_TIMEOUT_SECONDS=120
CODING_AGENT_PROMPT_CACHING_ENABLED=true
CODING_AGENT_PROMPT_CACHE_VERSION=v1
CODING_AGENT_ANTHROPIC_PROMPT_CACHE_TTL=5m
```

Deterministic routing/navigation are intentionally available to avoid unnecessary model calls. The reasoning model can then be reserved for the expensive planning/patch/repair portions of a run.

---

# Custom Skills and Tools

The Skills and Tools page is an admin surface for extending agent behavior without editing built-in playbooks directly.

The feature deliberately treats **skills** and **tools** differently:

- A **skill** is Markdown guidance. It is never executed.
- A **tool** is Python code. **Approved tools execute inside the backend process and therefore require a stronger validation and human-review lifecycle.**

## Current support matrix

| Feature                                      | Coding agent | Voice agent                                               |
| -------------------------------------------- | ------------ | --------------------------------------------------------- |
| List built-in/custom skills in admin UI      | Yes          | Yes                                                       |
| Create/edit custom skill Markdown            | Yes          | Yes                                                       |
| Generate skill with AI                       | Yes          | Yes                                                       |
| Import + normalize Markdown skill            | Yes          | Yes                                                       |
| Dynamic runtime`SkillRegistry` routing     | Yes          | Voice intake still uses its existing recommendation logic |
| Upload tool to quarantine/review             | Yes          | Yes                                                       |
| Approve custom tool for runtime execution    | Yes          | Yes                                                       |
| Invoke approved custom tool during agent run | Yes          | Yes                                                       |

---

## Custom Skill Lifecycle

### 1. Start from a template, import Markdown, or generate with AI

The UI supports three ways to begin:

1. Create a new canonical skill manually.
2. Import a `.md` file and ask the backend to normalize it.
3. Enter a natural-language prompt and use **Generate with AI**.

AI drafting currently uses the configured coding provider/model with a low-temperature structured-output request.

### 2. Canonical skill format

Every saved skill must contain the following contract:

```md
# Skill: Display name

Purpose: One-line description of what the skill does.

Use when:
- Condition that should cause the router to select this skill.

Allowed tools:
- exact_registered_tool_name

Steps:
1. Concrete step.
2. Concrete step.

Rules:
- Safety or behavior rule.
```

Required elements:

- First non-empty line starts with `# Skill: <display name>`.
- A non-empty top-level `Purpose:` line.
- `Use when` section.
- `Allowed tools` section.
- `Steps` section.
- `Rules` section.
- No null bytes.

The UI validates the canonical shape before save, and the backend validates it again.

### 3. Custom skill naming rules

User-created skills must use a registry name beginning with:

```text
custom_
```

Names are normalized to lowercase and use the repository name pattern:

```text
^[a-z][a-z0-9_-]{1,63}$
```

AI/imported names are normalized toward lowercase snake_case and automatically receive the `custom_` prefix when it is missing.

Built-in skills are read-only in the UI. Only `custom_` skills may be overwritten or deleted through the admin endpoints.

### 4. Tool dependency rules

A skill's `Allowed tools` list is a capability boundary, not documentation-only text.

Rules:

- AI generation may select tools **only from the executable tool catalog supplied by the backend**.
- Tool names must match registered tools exactly.
- Pending-review tools do **not** count as executable.
- A skill cannot be saved if it references a tool that is not executable for that agent.
- Imported Markdown may name unavailable tools, but unavailable dependencies are removed from `Allowed tools` and returned as warnings.
- When unavailable imported/generated dependencies are detected, the rendered skill receives an additional rule telling the agent not to assume those tools exist.

This prevents skill generation from hallucinating a tool dependency and silently granting itself a capability that the runtime does not expose.

### 5. Runtime loading

Coding skills are loaded from disk on registry load, so a newly saved custom coding skill can be available to the next coding-agent run without restarting the process.

The coding registry can combine multiple selected skills into one bounded prompt and derives the union of their allowed tool names.

### Skill limits

- Skill content: up to **50,000 characters**.
- AI-generation prompt: **3–8,000 characters**.
- Generated `use_when`: up to 8 items.
- Generated `allowed_tools`: up to 20 items.
- Generated steps: up to 12 items.
- Generated rules: up to 12 items.

---

## Custom Tool Lifecycle

Custom tools are intentionally more restrictive because they execute Python in the backend process.

### 1. Upload to quarantine

A custom tool is submitted with:

- agent
- tool name
- purpose
- Python source

The source is written to the agent's `custom_pending` directory and is **not imported by the runtime**.

A pending tool cannot be selected by a saved skill as an executable dependency.

### 2. Quarantine validation

Before a file can enter the review queue:

- Source must parse as valid Python.
- It must define a public function whose name matches the submitted tool name.
- Only supported top-level constructs are accepted.
- A top-level expression may only be the module docstring.
- Obvious top-level execution is rejected.
- Existing built-in or approved tool names cannot be replaced by uploading another tool with the same executable name.

If no module docstring is supplied, the submitted purpose is inserted as the module docstring.

### 3. Human review

The review endpoint returns:

- normalized tool metadata
- full source
- `approval_ready`
- validation errors

A human should review the source before approval. Static validation is defense in depth; it is **not an OS sandbox**.

### 4. Strong approval validation

Approved custom coding tools must satisfy the stricter runtime contract:

- Synchronous Python function only; async tools are not supported yet.
- Exactly one public function is exposed.
- Public function name must match the module/file name.
- No decorators.
- No executable function calls in default argument expressions.
- No executable function calls in top-level assignments.
- Only the safe import allowlist is permitted.
- No dunder name/attribute access.
- Dangerous dynamic calls such as `eval`, `exec`, `compile`, `open`, `getattr`, `setattr`, `globals`, `locals`, and similar operations are rejected.
- File-mutation/process-loading calls such as `.write_text()`, `.unlink()`, `.rename()`, `.system()`, `.popen()`, and related calls are rejected.
- Function signatures may use named parameters only; positional-only parameters, `*args`, and `**kwargs` are rejected.

The current safe import roots include standard-library modules such as `collections`, `dataclasses`, `datetime`, `enum`, `fnmatch`, `functools`, `hashlib`, `itertools`, `json`, `math`, `operator`, `pathlib`, `re`, `statistics`, `string`, `textwrap`, `tomllib`, and `typing`.

Expand that allowlist only for a concrete, reviewed requirement.

### 5. Atomic activation

Approval does not immediately move the file into the executable directory.

The backend first:

1. Re-runs quarantine validation.
2. Runs the stronger approved-tool static validation.
3. Copies the candidate into a temporary directory.
4. Loads it through an `ApprovedCustomToolRegistry`.
5. Verifies the runtime can discover and bind the tool.
6. Only then moves it from `custom_pending` to `custom_approved`.

This keeps activation atomic from the coding runtime's perspective.

### 6. Runtime invocation

The coding agent loads only approved tools.

Additional runtime boundaries include:

- `repo_root` is runtime-owned; model output cannot override it.
- Arguments are bound against the inspected Python signature before invocation.
- Custom tool output is JSON-rendered when possible.
- Tool output is truncated after **24,000 characters**.
- A coding run is limited to a small bounded number of custom tool calls (currently **4**).

Custom tools are intended primarily to provide narrow, deterministic context or transformations. They should not become a second unrestricted shell/runtime layer.

### 7. Reject

Rejecting a pending tool deletes the quarantined source. It is never moved to the approved registry.

### Tool limits

- Tool source: up to **100,000 characters**.
- Purpose: up to **500 characters**.
- Name: lowercase repository name pattern; use snake_case for custom tools.

---

# GitHub Integration / Source Control

The application includes a backend-managed GitHub integration designed to work with the coding agent's local filesystem workflow.

## Managed checkout model

1. Backend lists repositories visible to the configured GitHub credential.
2. User selects a repository and branch in the frontend.
3. Backend verifies Git transport access.
4. Repository is cloned under `GITHUB_WORKSPACE_ROOT` or an existing checkout is reused.
5. Coding-agent repository browsing, patching, validation, and approval run against that checkout.
6. Source-control actions operate on the same managed working tree.

The GitHub token stays on the backend. Git HTTPS authentication is passed through an ephemeral Git configuration header rather than embedding the token in the clone URL.

## Supported source-control operations

The current backend supports:

- connection test
- repository discovery, including private repositories when the credential has access
- branch discovery
- managed clone/import
- repository status
- branch creation
- branch switching
- pull / fast-forward update
- commit selected changed files
- push
- create pull request

The Source Control page also displays staged, unstaged, untracked, and most recently committed files.

## Branch switching with local changes

Changing branches does not require users to throw away staged/unstaged/untracked work.

The managed checkout stores branch-specific work in an internal Git stash using the `ai-agents:auto-stash:` prefix, switches branches, then restores the saved snapshot when the user returns to that branch.

If restoration fails, the backend retains the stash for manual recovery and returns a conflict instead of silently losing the work.

## PR-first publishing policy

Direct publishing to the repository default branch is disabled by default:

```env
GITHUB_ALLOW_DEFAULT_BRANCH_PUSH=false
```

The intended workflow is:

```text
create agent/* branch
  -> coding-agent changes
  -> human approval
  -> commit
  -> push
  -> pull request
```

## Commit safety

Agent commits are constrained by backend policy:

- Only paths that are currently changed can be committed.
- Requested paths must remain inside the managed repository.
- Default maximum: **100 files per commit**.
- Default maximum file size: **5 MB**.
- Sensitive path patterns are blocked by default, including `.env`, `.env.*`, PEM/key files, SSH private-key names, and paths containing credentials/secrets.
- Push requires repository push permission.
- Push is rejected when the remote branch has commits not present locally.
- Pull requests require a clean working tree and a pushed head branch.

## GitHub backend routes

```text
GET  /github/status
GET  /github/connection-test
GET  /github/repositories
POST /github/repositories/import
GET  /github/repositories/branches
GET  /github/repositories/status
POST /github/repositories/branches/create
POST /github/repositories/pull
POST /github/repositories/commit
POST /github/repositories/push
POST /github/repositories/pull-requests
```

## GitHub configuration

```env
GITHUB_TOKEN=your_backend_only_token
GITHUB_TOKEN_KIND=user
GITHUB_API_URL=https://api.github.com
GITHUB_API_VERSION=2026-03-10
GITHUB_WORKSPACE_ROOT=.ai-agents/github-workspaces
GITHUB_TIMEOUT_SECONDS=120

GITHUB_COMMIT_AUTHOR_NAME=AI Agents
GITHUB_COMMIT_AUTHOR_EMAIL=ai-agents@users.noreply.github.com
GITHUB_ALLOW_DEFAULT_BRANCH_PUSH=false
GITHUB_MAX_COMMIT_FILES=100
GITHUB_MAX_FILE_SIZE_BYTES=5000000
```

For a local single-user installation, a fine-grained PAT is sufficient when it has the required repository access. For a multi-user deployment, prefer GitHub App installation tokens instead of sharing one long-lived user credential.

Typical permissions:

- Read-only repository inspection: **Contents: Read**.
- Commit/push workflow: **Contents: Read and write**.
- Pull-request workflow: **Pull requests: Read and write** in addition to the required contents access.
- Organization repositories may also require SSO authorization.

---

# Voice Agent

The voice agent is an intake/orchestration layer for coding tasks.

Its job is to turn spoken or typed requests plus repository evidence into a detailed coding-agent request.

## Current flow

```text
voice/text input
  -> optional STT
  -> attachment + repository context gathering
  -> clarification / intake decision
  -> detailed coding request
  -> coding agent
  -> optional TTS reply
```

The intake model is explicitly told that repository inspection has already happened in deterministic backend code and that it has no callable tools during the structured intake step. This helps avoid malformed provider tool calls in JSON mode.

The voice flow includes retry handling for malformed structured output and unwanted `tool_use_failed` / model-tool-call behavior.

## Voice configuration

```env
VOICE_CHAT_MODEL=llama-3.1-8b-instant
VOICE_CHAT_MAX_TOKENS=2048
VOICE_STT_MODEL=whisper-large-v3-turbo
VOICE_TTS_MODEL=canopylabs/orpheus-v1-english
VOICE_TTS_VOICE=hannah
VOICE_TTS_ENABLED=true
VOICE_TTS_MAX_CHARS=200
VOICE_MAX_CLARIFICATIONS=2
VOICE_MAX_AUDIO_MB=15
```

The provider for each voice slot is configured separately through Agent Settings.

---

# Model Provider Environment Variables

Configure only the providers you intend to use.

```env
# Groq
GROQ_API_KEY=...
GROQ_URL=https://api.groq.com/openai/v1

# DeepSeek
DEEPSEEK_API_KEY=...
DEEPSEEK_URL=https://api.deepseek.com

# OpenRouter
OPENROUTER_API_KEY=...
OPENROUTER_URL=https://openrouter.ai/api/v1

# OpenAI
OPENAI_API_KEY=...
OPENAI_URL=https://api.openai.com/v1

# Anthropic
ANTHROPIC_API_KEY=...
ANTHROPIC_URL=https://api.anthropic.com

# Google
GOOGLE_API_KEY=...
GOOGLE_URL=https://generativelanguage.googleapis.com/v1beta
```

Default routing can also be set from environment variables:

```env
CODING_PROVIDER=groq
CODING_MODEL=openai/gpt-oss-120b
REASONING_PROVIDER=deepseek
REASONING_MODEL=deepseek-v4-pro
CAPTION_PROVIDER=groq
CAPTION_MODEL=meta-llama/llama-4-scout-17b-16e-instruct
VOICE_CHAT_PROVIDER=groq
VOICE_STT_PROVIDER=groq
VOICE_TTS_PROVIDER=groq
```

The Agent Settings API can override the non-secret model selections at runtime.

---

# LangSmith

```env
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=your_langsmith_key
LANGCHAIN_PROJECT=ai-agents-dev
LANGSMITH_ENDPOINT=https://api.smith.langchain.com
```

The coding agent uses run names/tags/metadata to make graph and node behavior easier to inspect while developing.

---

# CLI Usage

Run the coding agent in dry-run mode:

```bash
uv run python -m ai_agents.agents.coding.main \
  --repo-root ./src/ai_agents \
  --workspace-root . \
  "Find where the coding agent validates patches and explain how it works"
```

Allow the CLI agent to write changes:

```bash
uv run python -m ai_agents.agents.coding.main \
  --repo-root ./src/ai_agents \
  --workspace-root . \
  --write \
  "Fix the reporting node so it does not claim success when validation fails"
```

Write a Markdown run report:

```bash
uv run python -m ai_agents.agents.coding.main \
  --repo-root ./src/ai_agents \
  --workspace-root . \
  --markdown-report \
  "Add tests for the structured repository search service"
```

Reuse a checkpoint thread:

```bash
uv run python -m ai_agents.agents.coding.main \
  --repo-root ./src/ai_agents \
  --workspace-root . \
  --thread-id coding-run-example \
  "Continue the previous coding-agent task"
```

Disable persistent memory for one run:

```bash
uv run python -m ai_agents.agents.coding.main \
  --repo-root ./src/ai_agents \
  --workspace-root . \
  --no-memory \
  "Run this task without recalling or saving long-term memory"
```

`--setup-memory` can force SQLite setup for a run. SQLite setup is otherwise enabled by default and is cheap/idempotent.

### CLI options

```text
--repo-root            Repository root to inspect and patch.
--workspace-root       Project root used for validation commands.
--write                Allow file writes; omit for dry-run mode.
--markdown-report      Write a detailed Markdown report.
--report-path          Custom Markdown report path.
--thread-id            LangGraph checkpoint thread id.
--memory-user-id       User segment for long-term memory namespaces.
--memory-namespace     Logical memory namespace/scope.
--setup-memory         Force persistence setup before the run.
--no-memory            Disable persistent memory for the run.
```

---

# Local Development

## Python/backend

Requirements:

- Python `>=3.10,<3.14`
- `uv`
- Git for managed GitHub checkouts/source control

Install dependencies:

```bash
uv sync
```

The core project dependencies include FastAPI, LangChain/LangGraph, provider SDKs/integrations, `langgraph-checkpoint-sqlite`, and `fastembed`.

## Frontend / Electron

```bash
cd src/ai_agents/agents/frontend
npm install
npm run dev
```

Useful frontend scripts:

```text
npm run dev
npm run build
npm run typecheck
npm run preview
npm run desktop:build
```

Electron Builder targets:

- macOS: DMG
- Windows: NSIS
- Linux: AppImage

The desktop bridge also provides the native local-directory picker used by the Source Control page.

---

# RAG Subsystem

The RAG subsystem remains part of the broader repository and is separate from the coding agent's local SQLite memory.

Current RAG components include:

- document ingestion
- chunking
- Qdrant vector search
- Postgres metadata/idempotency storage
- query expansion and retrieval controls
- reranking/verification infrastructure

Example environment variables:

```env
QDRANT_URL=http://localhost:6333
QDRANT_COLLECTION=rag-default
DATABASE_URL=postgresql+psycopg://ai_agents:ai_agents@localhost:5432/ai_agents
K=8
```

Do not confuse `DATABASE_URL` for the RAG subsystem with coding-agent memory; coding-agent checkpoints/store now use local SQLite files.

---

# Design Principles

- Inspect before editing.
- Keep patches small and reviewable.
- Prefer structured outputs over free-form control messages.
- Keep repository mutation in deterministic runner code.
- Keep context workers read-only.
- Treat memory as useful context, not unquestioned truth.
- Validate before reporting success.
- Require explicit approval before applying frontend-generated changes.
- Keep secrets on the backend.
- Use PR-first GitHub publishing by default.
- Treat custom skills as data and custom tools as executable code requiring human review.
- Never make a pending-review tool executable.

---

# Planned Future Development

- [X] Build the coding-agent LangGraph workflow.
- [X] Add parallel read-only context workers and simple-task fast path.
- [X] Replace coding-agent Postgres memory with local SQLite.
- [X] Add local FastEmbed semantic coding memory.
- [X] Add a voice intake agent.
- [X] Add provider-agnostic model configuration for coding, reasoning, vision, and voice slots.
- [X] Add Anthropic/OpenAI/Google provider support to runtime model configuration.
- [X] Add local-directory repository selection in the desktop app.
- [X] Add managed GitHub repository discovery/checkouts and source-control operations.
- [X] Add custom skill authoring, AI drafting, and Markdown import normalization.
- [X] Add custom coding-tool quarantine, review, approval, and runtime execution.
- [ ] Complete dynamic custom skill routing inside the voice runtime.
- [ ] Add an approved custom-tool runtime/approval registry for the voice agent.
- [ ] Add optional AI-assisted custom tool scaffolding while preserving human review.
- [ ] Expand validation profiles and evaluation coverage.
- [ ] Continue improving long-term memory quality and retrieval evaluation.
- [ ] Harden desktop packaging/deployment documentation.
- [ ] Expand the built-in skill and tool libraries.

---

# Security Notes

This project is an agent-development harness, not a hardened multi-tenant sandbox.

In particular:

- Approved custom tools run as Python in the local backend process.
- Static AST checks reduce risk but do not create an OS security boundary.
- GitHub credentials should remain backend-only and should use the minimum required repository permissions.
- Default-branch pushes are disabled unless explicitly enabled.
- Sensitive GitHub commit paths are blocked by default.
- Provider secrets are never included in the public agent-configuration snapshot.

For a production multi-user deployment, add stronger process isolation, identity/authorization controls, short-lived provider/GitHub credentials, audit logging, and platform-specific sandboxing.

---

# License

MIT
