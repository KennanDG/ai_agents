from __future__ import annotations

from typing import TypedDict, Optional
import io

from groq import Groq
from langgraph.graph import END, START, StateGraph
from langgraph.types import RetryPolicy

from ai_agents.config.settings import settings


class VoiceAgentState(TypedDict, total=False):
    """
    State used by the voice agent LangGraph.

    Attributes:
        audio_bytes: Raw audio data (e.g., from a file or stream).
        transcription: The resulting text transcription.
        error: An error message if transcription fails.
    """
    audio_bytes: bytes
    transcription: str
    error: Optional[str]


def transcribe_node(state: VoiceAgentState) -> dict:
    """
    Transcribe audio using Groq's Whisper-large-v3-turbo model.

    Expects `audio_bytes` in state. Returns the transcription text
    or an error message.

    Args:
        state: Current VoiceAgentState.

    Returns:
        A dict with keys 'transcription' and 'error' to merge into state.
    """
    audio_bytes = state.get("audio_bytes")
    if not audio_bytes:
        return {"transcription": "", "error": "No audio bytes provided."}

    api_key = settings.resolved_groq_api_key()
    if not api_key:
        return {"transcription": "", "error": "GROQ_API_KEY not available from settings."}

    client = Groq(api_key=api_key)
    model_name = "whisper-large-v3-turbo"

    try:
        # Wrap bytes in a binary file-like object.
        # The Whisper endpoint infers the format from the file extension,
        # so assign a plausible name.
        audio_file = io.BytesIO(audio_bytes)
        audio_file.name = "audio.wav"

        transcription_response = client.audio.transcriptions.create(
            file=audio_file,
            model=model_name,
            response_format="json",
            # language="en"  # let the model auto‑detect by default
        )

        return {"transcription": transcription_response.text, "error": None}
    except Exception as e:
        return {"transcription": "", "error": f"Transcription failed: {type(e).__name__}: {e}"}


def build_voice_agent_graph():
    """
    Build and compile the voice agent LangGraph.

    The graph contains a single transcription node with transient
    retry support. The compiled graph can be invoked with a
    VoiceAgentState dictionary containing `audio_bytes`.

    Returns:
        A compiled LangGraph `CompiledStateGraph`.
    """
    builder = StateGraph(VoiceAgentState)

    # Retry transient failures (network, rate limits, etc.)
    transient_retry = RetryPolicy(
        max_attempts=3,
        initial_interval=1.0,
        backoff_factor=2.0,
        max_interval=8.0,
    )

    builder.add_node("transcribe", transcribe_node, retry_policy=transient_retry)

    builder.add_edge(START, "transcribe")
    builder.add_edge("transcribe", END)

    return builder.compile()


# Convenience singleton – can be imported and called directly.
voice_agent_app = build_voice_agent_graph()
