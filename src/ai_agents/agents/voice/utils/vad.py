"""Voice Activity Detection (VAD) utilities using energy thresholding."""

import io
import wave
import audioop
from typing import List, Optional, Tuple


# Default WAV parameters
DEFAULT_SAMPLE_RATE = 16000
DEFAULT_SAMPLE_WIDTH = 2  # 16-bit
DEFAULT_CHANNELS = 1

def decode_wav(audio_bytes: bytes) -> Tuple[bytes, int, int, int]:
    """Decode a WAV file into raw PCM data and parameters.

    Args:
        audio_bytes: WAV file content.

    Returns:
        Tuple of (pcm_data, sample_rate, sample_width, num_channels).

    Raises:
        ValueError: If the audio is not a valid WAV file.
    """
    with wave.open(io.BytesIO(audio_bytes), 'rb') as wf:
        params = (wf.getnchannels(), wf.getsampwidth(), wf.getframerate(), wf.getnframes())
        pcm = wf.readframes(wf.getnframes())
    return pcm, params[2], params[1], params[0]

def encode_wav(pcm_data: bytes, sample_rate: int = DEFAULT_SAMPLE_RATE,
               sample_width: int = DEFAULT_SAMPLE_WIDTH,
               channels: int = DEFAULT_CHANNELS) -> bytes:
    """Encode raw PCM data into a WAV file in memory.

    Args:
        pcm_data: Raw PCM audio bytes.
        sample_rate: Sampling rate in Hz.
        sample_width: Bytes per sample (2 for 16-bit).
        channels: Number of audio channels.

    Returns:
        WAV file as bytes.
    """
    buf = io.BytesIO()
    with wave.open(buf, 'wb') as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(sample_width)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm_data)
    return buf.getvalue()

def detect_speech_segments(
    pcm_data: bytes,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    sample_width: int = DEFAULT_SAMPLE_WIDTH,
    frame_duration_ms: int = 20,
    threshold: float = 500.0,
    min_speech_duration_ms: int = 300,
) -> List[Tuple[float, float]]:
    """Detect speech segments using RMS energy threshold.

    Args:
        pcm_data: Raw PCM audio (16-bit mono expected).
        sample_rate: Sample rate in Hz.
        sample_width: Bytes per sample.
        frame_duration_ms: Length of each analysis frame in ms.
        threshold: RMS energy threshold above which a frame is considered speech.
        min_speech_duration_ms: Minimum duration of a speech segment to be kept.

    Returns:
        List of (start_time_sec, end_time_sec) tuples.
    """
    frame_size = int(sample_rate * (frame_duration_ms / 1000.0))
    num_bytes_per_sec = sample_width * sample_rate
    min_speech_bytes = int(num_bytes_per_sec * min_speech_duration_ms / 1000.0)

    segments = []
    speech_start = None
    pos = 0

    while pos + frame_size <= len(pcm_data):
        frame = pcm_data[pos:pos + frame_size]
        rms = audioop.rms(frame, sample_width)
        if rms > threshold:
            if speech_start is None:
                speech_start = pos
        else:
            if speech_start is not None:
                speech_end = pos
                if (speech_end - speech_start) >= min_speech_bytes:
                    start_sec = speech_start / num_bytes_per_sec
                    end_sec = speech_end / num_bytes_per_sec
                    segments.append((start_sec, end_sec))
                speech_start = None
        pos += frame_size

    # Handle trailing speech
    if speech_start is not None:
        speech_end = len(pcm_data)
        if (speech_end - speech_start) >= min_speech_bytes:
            start_sec = speech_start / num_bytes_per_sec
            end_sec = speech_end / num_bytes_per_sec
            segments.append((start_sec, end_sec))

    return segments

def trim_silence(
    audio_bytes: bytes,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    sample_width: int = DEFAULT_SAMPLE_WIDTH,
    threshold: float = 500.0,
    padding_ms: int = 200,
) -> bytes:
    """Trim silence from the beginning and end of a WAV audio file.

    Args:
        audio_bytes: WAV file content.
        sample_rate: Expected sample rate (used if WAV header is missing; otherwise overridden).
        sample_width: Expected sample width.
        threshold: RMS energy threshold.
        padding_ms: Amount of silence to keep at edges (ms).

    Returns:
        Trimmed WAV audio bytes, or the original audio if unable to decode. If no speech
        is detected, returns empty WAV.
    """
    try:
        pcm, sr, sw, ch = decode_wav(audio_bytes)
    except Exception:
        # Not a valid WAV; pass through unchanged
        return audio_bytes

    # Override with actual params from WAV
    sample_rate = sr
    sample_width = sw

    segments = detect_speech_segments(
        pcm, sample_rate, sample_width, threshold=threshold,
        min_speech_duration_ms=0,
    )
    if not segments:
        # No speech detected; return silence (empty)
        return encode_wav(b'', sample_rate, sample_width, ch)

    padding_bytes = int(sample_rate * sample_width * padding_ms / 1000.0)
    start_byte = max(0, int(segments[0][0] * sample_rate * sample_width) - padding_bytes)
    end_byte = min(len(pcm), int(segments[-1][1] * sample_rate * sample_width) + padding_bytes)

    trimmed_pcm = pcm[start_byte:end_byte]
    return encode_wav(trimmed_pcm, sample_rate, sample_width, ch)
