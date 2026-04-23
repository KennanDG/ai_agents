from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Annotated, TypedDict, Literal

from langgraph.graph import StateGraph, START, END
from langchain_groq import ChatGroq

from ai_agents.config.settings import settings


# ----------------------------
# Config
# ----------------------------

REPO_ROOT = Path(os.getenv("CODING_REPO_ROOT", ".")).resolve()
GROQ_API_KEY = settings.resolved_groq_api_key()

if not GROQ_API_KEY:
    raise RuntimeError("GROQ_API_KEY is not set")

llm = ChatGroq(
    model=settings.chat_model,
    api_key=GROQ_API_KEY,
    temperature=0.0,
)


# ----------------------------
# Safe repo tools
# ----------------------------

TEXT_EXTENSIONS = {
    ".py", ".md", ".txt", ".toml", ".yaml", ".yml",
    ".json", ".tsx", ".ts", ".js", ".jsx"
}

BLOCKED_PARTS = {
    ".git", ".venv", "node_modules", "__pycache__", ".mypy_cache", ".pytest_cache"
}


def is_allowed(path: Path) -> bool:
    return not any(part in BLOCKED_PARTS for part in path.parts)


def list_repo_files(max_files: int = 300) -> list[str]:
    files: list[str] = []

    for path in REPO_ROOT.rglob("*"):
        if path.is_file() and is_allowed(path):
            files.append(str(path.relative_to(REPO_ROOT)))

        if len(files) >= max_files:
            break

    return files


def read_file(relative_path: str, max_chars: int = 12000) -> str:
    path = (REPO_ROOT / relative_path).resolve()

    if not path.exists() or not path.is_file():
        return f"[ERROR] File not found: {relative_path}"
    
    if REPO_ROOT not in path.parents and path != REPO_ROOT:
        return f"[ERROR] Path escapes repo root: {relative_path}"
    
    if not is_allowed(path):
        return f"[ERROR] Access denied: {relative_path}"

    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
        return text[:max_chars]
    
    except Exception as e:
        return f"[ERROR] Failed reading {relative_path}: {e}"


def search_repo(pattern: str, max_results: int = 20) -> list[str]:
    results: list[str] = []

    try:
        regex = re.compile(pattern, re.IGNORECASE)
    except re.error:
        regex = re.compile(re.escape(pattern), re.IGNORECASE)

    for path in REPO_ROOT.rglob("*"):
        if not path.is_file() or not is_allowed(path):
            continue
        if path.suffix not in TEXT_EXTENSIONS:
            continue

        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue

        for line_no, line in enumerate(text.splitlines(), start=1):
            if regex.search(line):
                results.append(f"{path.relative_to(REPO_ROOT)}:{line_no}: {line.strip()}")
                if len(results) >= max_results:
                    return results

    return results


# ----------------------------
# Agent state
# ----------------------------

class CodingState(TypedDict, total=False):
    task: str
    task_type: Literal["explain", "search", "summarize", "unknown"]
    repo_files: list[str]
    search_hits: list[str]
    selected_files: list[str]
    file_contents: dict[str, str]
    answer: str


# ----------------------------
# Nodes
# ----------------------------

def classify_task(state: CodingState) -> CodingState:
    task = state["task"].lower()

    prompt = f"""
Classify the user's coding-repo task into one of:
- explain
- search
- summarize
- unknown

User task:
{task}

Return only one label.
""".strip()

    result = llm.invoke(prompt).content.strip().lower()

    if result not in {"explain", "search", "summarize", "unknown"}:
        result = "unknown"

    state["task_type"] = result 

    return state


def gather_context(state: CodingState) -> CodingState:
    task = state["task"]
    files = list_repo_files()
    hits = search_repo(task)

    selected_files: list[str] = []

    for hit in hits:
        path = hit.split(":", 1)[0]

        if path not in selected_files:
            selected_files.append(path)

    # Fallback
    if not selected_files:
        selected_files = [f for f in files if f.endswith(".py")]

    contents = {path: read_file(path) for path in selected_files}

    state["repo_files"] = files[:80]
    state["search_hits"] = hits
    state["selected_files"] = selected_files
    state["file_contents"] = contents
    
    return state


def answer_task(state: CodingState) -> CodingState:
    task = state["task"]
    task_type = state.get("task_type", "unknown")
    search_hits = "\n".join(state.get("search_hits", []))
    selected_files = state.get("selected_files", [])
    file_contents = state.get("file_contents", {})

    context_blocks = []

    for path in selected_files:
        context_blocks.append(f"\n### FILE: {path}\n{file_contents.get(path, '')}")

    context = "\n".join(context_blocks)

    prompt = f"""
You are a careful coding assistant working on a local repository.

Task type: {task_type}
User request: {task}

Repo root: {REPO_ROOT}

Search hits:
{search_hits if search_hits else "[none]"}

Loaded file context:
{context}

Instructions:
- Answer only from the provided repo context.
- If the answer is uncertain, say so clearly.
- Be concrete and reference file paths when relevant.
- For simple changes, suggest the exact files to edit and why.
- Do not invent files or functions not shown in context.
""".strip()

    response = llm.invoke(prompt).content
    state["answer"] = response

    return state


# ----------------------------
# Graph
# ----------------------------

graph = StateGraph(CodingState)
graph.add_node("classify_task", classify_task)
graph.add_node("gather_context", gather_context)
graph.add_node("answer_task", answer_task)

graph.add_edge(START, "classify_task")
graph.add_edge("classify_task", "gather_context")
graph.add_edge("gather_context", "answer_task")
graph.add_edge("answer_task", END)

app = graph.compile()


# ----------------------------
# CLI
# ----------------------------

if __name__ == "__main__":
    print(f"Simple Coding Agent")
    print(f"Repo root: {REPO_ROOT}")
    print("Type a question about your codebase. Ctrl+C to exit.\n")

    while True:
        try:
            task = input(">> ").strip()
            if not task:
                continue

            result = app.invoke({"task": task})
            print("\n" + result["answer"] + "\n")

        except KeyboardInterrupt:
            print("\nExiting.")
            break