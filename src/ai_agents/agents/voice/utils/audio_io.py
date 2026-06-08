"""Utility functions for audio file I/O."""

from pathlib import Path
from typing import Optional


def load_audio_file(file_path: str) -> bytes:
    """Load an audio file as raw bytes.

    Supports any audio format accepted by the downstream ASR service
    (e.g., WAV, MP3, FLAC).

    Args:
        file_path: Path to the audio file.

    Returns:
        Raw file contents as bytes.

    Raises:
        FileNotFoundError: If the file does not exist.
        IOError: If reading fails.
    """
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"Audio file not found: {file_path}")
    with open(path, "rb") as f:
        return f.read()


def save_audio_file(audio_bytes: bytes, file_path: str) -> None:
    """Save audio bytes to a file.

    Args:
        audio_bytes: Audio data.
        file_path: Destination path.
    """
    with open(file_path, "wb") as f:
        f.write(audio_bytes)
