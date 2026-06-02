# Skill: Get Weather

Purpose: Retrieve current weather conditions or forecast for a given location.

Use when:
- The user asks about the weather (temperature, conditions, forecast).
- The user mentions a city, region, or asks "what's the weather like today?"

Allowed tools:
- get_weather (weather API), geocode_location (location lookup)

Steps:
1. Extract the location from the user's request. If no location is given, use the user's default location if available.
2. Determine whether the user wants current weather, hourly, or daily forecast.
3. Use a geocoding tool to resolve location names to coordinates if required.
4. Call the weather API with the resolved location and parameters.
5. Format the response with temperature, conditions, wind, humidity, and forecast as needed.

Rules:
- Default to metric units (Celsius, km/h) unless the user requests imperial.
- If the location is ambiguous, ask for clarification (city, country).
- Report the source of the weather data if asked.
- Handle API errors gracefully: tell the user if weather data is unavailable.
- Do not store or log location data beyond the session unless required for personalization.
