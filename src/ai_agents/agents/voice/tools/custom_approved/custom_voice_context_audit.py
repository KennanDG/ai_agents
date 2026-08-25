"""Summarize the repository evidence gathered for a voice-agent turn."""


def custom_voice_context_audit(
    transcript: str,
    prompt_text: str,
    repo_context: dict,
    active_path: str | None = None,
) -> dict:
    """Return a compact audit of repository context available to the voice agent."""

    transcript = (transcript or "").strip()
    prompt_text = (prompt_text or "").strip()

    repository_tree = repo_context.get("repository_tree") or []
    explicit_files = repo_context.get("explicit_files") or []
    search_matches = repo_context.get("search_matches") or []
    attachment_context = repo_context.get("attachment_context") or []

    relevant_paths: list[str] = []

    if active_path:
        relevant_paths.append(active_path)

    for item in explicit_files:
        if not isinstance(item, dict):
            continue

        path = item.get("path")
        if isinstance(path, str) and path and path not in relevant_paths:
            relevant_paths.append(path)

    ranked_matches: list[dict] = []

    for item in search_matches[:10]:
        if not isinstance(item, dict):
            continue

        path = item.get("path")
        if not isinstance(path, str) or not path:
            continue

        if path not in relevant_paths:
            relevant_paths.append(path)

        matched_terms = item.get("matched_terms")
        if not isinstance(matched_terms, list):
            matched_terms = []

        ranked_matches.append(
            {
                "path": path,
                "score": item.get("score", 0),
                "matched_terms": [
                    str(term)
                    for term in matched_terms[:8]
                    if str(term).strip()
                ],
            }
        )

    request_text = " ".join(
        part
        for part in (prompt_text, transcript)
        if part
    )

    return {
        "request_text": request_text[:1500],
        "active_path": active_path,
        "repository_file_count": len(repository_tree),
        "explicit_file_count": len(explicit_files),
        "search_match_count": len(search_matches),
        "attachment_context_count": (
            len(attachment_context)
            if isinstance(attachment_context, list)
            else 1 if attachment_context else 0
        ),
        "relevant_paths": relevant_paths[:12],
        "ranked_matches": ranked_matches,
        "context_available": bool(
            repository_tree
            or explicit_files
            or search_matches
            or attachment_context
        ),
    }
