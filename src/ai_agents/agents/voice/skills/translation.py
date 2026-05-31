from agents.voice.tools.translation import translate


def run(text: str, target: str = "en") -> str:
    """Translate text and return a voice-friendly result."""
    result = translate(text, target=target)
    if isinstance(result, dict):
        if "error" in result:
            return f"Translation error: {result['error']}"
        return f"Translation to {target}: {result['translated_text']}"
    return "Unexpected translation output."
