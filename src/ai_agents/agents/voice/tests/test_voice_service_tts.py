from __future__ import annotations

import io
import wave

from ai_agents.agents.voice.service import VoiceAgentService
from ai_agents.config.settings import settings


def _wav_bytes(*, frames: int, sample_rate: int = 16_000) -> bytes:
    output = io.BytesIO()
    with wave.open(output, "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(sample_rate)
        writer.writeframes(b"\x00\x00" * frames)
    return output.getvalue()


def test_tts_inputs_split_without_losing_text(monkeypatch) -> None:
    monkeypatch.setattr(settings, "voice_tts_max_chars", 200, raising=False)
    text = ("This sentence should be spoken completely and never truncated. " * 12).strip()

    chunks = VoiceAgentService._tts_inputs(text)

    assert len(chunks) > 1
    assert all(0 < len(chunk) <= 200 for chunk in chunks)
    assert " ".join(chunks) == " ".join(text.split())


def test_merge_wav_chunks_preserves_all_frames() -> None:
    first = _wav_bytes(frames=1_600)
    second = _wav_bytes(frames=3_200)

    merged = VoiceAgentService._merge_wav_chunks([first, second])

    with wave.open(io.BytesIO(merged), "rb") as reader:
        assert reader.getnchannels() == 1
        assert reader.getsampwidth() == 2
        assert reader.getframerate() == 16_000
        assert reader.getnframes() == 4_800
