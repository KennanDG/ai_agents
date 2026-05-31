from __future__ import annotations

import os
from langchain_core.tools import tool


@tool
def gmail_search(query: str, max_results: int = 5) -> str:
    """Search the user's Gmail inbox using the Gmail API.

    This tool requires the following environment variables to be set:
    - GMAIL_OAUTH_CLIENT_ID
    - GMAIL_OAUTH_CLIENT_SECRET
    - GMAIL_OAUTH_TOKEN_FILE or GMAIL_OAUTH_TOKEN_JSON

    Args:
        query: The Gmail search query (supports Gmail search operators).
        max_results: Maximum number of matching emails to return (default 5).

    Returns:
        A formatted string containing email metadata (subject, sender, date, snippet).
    """
    try:
        from google.oauth2.credentials import Credentials
        from googleapiclient.discovery import build
    except ImportError as exc:
        return (
            "Gmail access libraries are not installed. "
            "Please install google-auth and google-api-python-client."
        )

    client_id = os.getenv("GMAIL_OAUTH_CLIENT_ID")
    client_secret = os.getenv("GMAIL_OAUTH_CLIENT_SECRET")
    token_file = os.getenv("GMAIL_OAUTH_TOKEN_FILE")
    token_json = os.getenv("GMAIL_OAUTH_TOKEN_JSON")

    if not client_id or not client_secret:
        return "Missing GMAIL_OAUTH_CLIENT_ID or GMAIL_OAUTH_CLIENT_SECRET environment variables."

    if not token_file and not token_json:
        return "Missing GMAIL_OAUTH_TOKEN_FILE or GMAIL_OAUTH_TOKEN_JSON environment variables."

    try:
        if token_file and os.path.exists(token_file):
            credentials = Credentials.from_authorized_user_file(token_file)
        elif token_json:
            import json
            token_data = json.loads(token_json)
            credentials = Credentials.from_authorized_user_info(token_data)
        else:
            return "No valid token source found."

        service = build("gmail", "v1", credentials=credentials)

        response = (
            service.users()
            .messages()
            .list(userId="me", q=query, maxResults=max_results)
            .execute()
        )

        messages = response.get("messages", [])
        if not messages:
            return "No emails found matching the query."

        lines: list[str] = []
        for msg in messages[:max_results]:
            msg_data = (
                service.users()
                .messages()
                .get(userId="me", id=msg["id"], format="metadata",
                     metadataHeaders=["From", "Date", "Subject"])
                .execute()
            )
            headers = {h["name"]: h["value"] for h in msg_data.get("payload", {}).get("headers", [])}
            snippet = msg_data.get("snippet", "")
            sender = headers.get("From", "Unknown")
            subject = headers.get("Subject", "(no subject)")
            date = headers.get("Date", "Unknown date")
            lines.append(
                f"From: {sender}\nSubject: {subject}\nDate: {date}\nSnippet: {snippet}\n"
            )

        return "\n".join(lines)

    except Exception as e:
        # Do not leak token or credential details
        return f"Gmail search failed: {type(e).__name__}"