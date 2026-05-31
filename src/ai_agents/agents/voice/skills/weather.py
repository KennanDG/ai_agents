from agents.voice.tools.weather import get_weather


def run(city: str) -> str:
    """Return current weather in a voice-friendly format."""
    data = get_weather(city)
    if isinstance(data, dict):
        if "error" in data:
            return f"Weather error: {data['error']}"
        return (
            f"Current weather in {data['city']}: "
            f"{data['description']}, {data['temperature']}°C, "
            f"humidity {data['humidity']}%, wind {data['wind_speed']} m/s."
        )
    return "Unexpected weather output."
