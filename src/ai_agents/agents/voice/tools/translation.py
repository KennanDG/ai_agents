import logging
from typing import Any, Dict

import requests

logger = logging.getLogger(__name__)


def translate(text: str, source: str = "auto", target: str = "en") -> Dict[str, Any]:
    """Translate text using the MyMemory free translation API."""
    try:
        langpair = f"{source}|{target}"
        url = "https://api.mymemory.translated.net/get"
        params = {"q": text, "langpair": langpair}
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        translated = data.get("responseData", {}).get("translatedText", "")
        return {"translated_text": translated}
    except Exception as e:
        logger.exception("Translation failed")
        return {"error": str(e)}
