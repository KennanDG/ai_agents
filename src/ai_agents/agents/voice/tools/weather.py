import os
import logging
from typing import Any, Dict

import requests

logger = logging.getLogger(__name__)


def get_weather(city: str) -> Dict[str, Any]:
    """Fetch current weather for a city via OpenWeatherMap."""
    api_key = os.getenv("OPENWEATHERMAP_API_KEY")
    if not api_key:
        logger.error("OPENWEATHERMAP_API_KEY not set")
        return {"error": "Weather API key not configured"}
    try:
        url = "https://api.openweathermap.org/data/2.5/weather"
        params = {"q": city, "appid": api_key, "units": "metric"}
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        return {
            "city": data.get("name", city),
            "temperature": data["main"]["temp"],
            "description": data["weather"][0]["description"],
            "humidity": data["main"]["humidity"],
            "wind_speed": data["wind"]["speed"],
        }
    except Exception as e:
        logger.exception("Weather fetch failed")
        return {"error": str(e)}
