from __future__ import annotations

from typing import TypedDict, Optional
import io

from groq import Groq
from langgraph.graph import END, START, StateGraph
from langgraph.types import RetryPolicy

from ai_agents.config.settings import settings
from ai_agents.agents.voice.utils.audio_io import load_audio_file
from ai_agents.agents.voice.utils.vad import trim_silence


class VoiceAgentState(TypedDict, total=False):
    """
    State used by the voice agent LangGraph.

    Attributes:
        audio_file_path: Optional path to an audio file to load.
        audio_bytes: Raw audio data (e.g., from a file or stream).
        transcription: The resulting text transcription.
        error: An error message if any step fails.
    """
    audio_file_path: Optional[str]
    audio_bytes: bytes
    transcription: str
    error: Optional[str]


def load_audio_node(state: VoiceAgentState) -> dict:
    """
    Load audio from a file path if provided and audio_bytes is empty.

    Args:
        state: Current VoiceAgentState.

    Returns:
        Dictionary with updated audio_bytes or an error.
    """
    file_path = state.get("audio_file_path")
    if not file_path:
        return {}  # nothing to load
    if state.get("audio_bytes"):
        # Audio bytes already present; skip loading
        return {}
    try:
        audio_bytes = load_audio_file(file_path)
        return {"audio_bytes": audio_bytes, "error": None}
    except Exception as e:
        return {"error": f"Failed to load audio file: {e}"}


def vad_node(state: VoiceAgentState) -> dict:
    """
    Trim silence from audio_bytes using energy-based VAD.

    Args:
        state: Current VoiceAgentState.

    Returns:
        Dictionary with updated audio_bytes or an error.
    """
    audio_bytes = state.get("audio_bytes")
    if not audio_bytes:
        return {"error": "No audio bytes to process."}
    try:
        trimmed = trim_silence(audio_bytes)
        return {"audio_bytes": trimmed, "error": None}
    except Exception as e:
        # If VAD fails, keep original audio and continue
        return {"error": f"VAD error: {e}"}


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

    The graph contains:
    1. load_audio: Loads audio from a file if `audio_file_path` is given.
    2. vad: Removes silence using energy-based VAD.
    3. transcribe: Runs Groq Whisper STT.

    All nodes support transient retry.

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

    builder.add_node("load_audio", load_audio_node, retry_policy=transient_retry)
    builder.add_node("vad", vad_node, retry_policy=transient_retry)
    builder.add_node("transcribe", transcribe_node, retry_policy=transient_retry)

    builder.add_edge(START, "load_audio")
    builder.add_edge("load_audio", "vad")
    builder.add_edge("vad", "transcribe")
    builder.add_edge("transcribe", END)

    return builder.compile()


# Convenience singleton – can be imported and called directly.
voice_agent_app = build_voice_agent_graph()
