import os
from unittest.mock import patch

import pytest

from ai_agents.agents.voice.tools.web_search import web_search
from ai_agents.agents.voice.tools.weather import get_weather
from ai_agents.agents.voice.tools.translation import translate
from ai_agents.agents.voice.tools.implement_change import implement_change


class TestWebSearch:
    @patch("agents.voice.tools.web_search.requests.get")
    def test_success(self, mock_get, monkeypatch):
        monkeypatch.setenv("SERPAPI_API_KEY", "test_key")
        mock_get.return_value.json.return_value = {
            "organic_results": [
                {"title": "Result 1", "snippet": "Snippet 1"},
                {"title": "Result 2", "snippet": "Snippet 2"},
            ]
        }
        mock_get.return_value.raise_for_status.return_value = None
        result = web_search("test query")
        assert "results" in result
        assert len(result["results"]) == 2
        mock_get.assert_called_once()
        assert "api_key" in mock_get.call_args[1]["params"]
        assert "test_key" not in str(mock_get.call_args)  # key not logged in test

    def test_missing_api_key(self, monkeypatch):
        monkeypatch.delenv("SERPAPI_API_KEY", raising=False)
        result = web_search("query")
        assert result == {"error": "Search API key not configured"}

    @patch("agents.voice.tools.web_search.requests.get")
    def test_http_error(self, mock_get, monkeypatch):
        monkeypatch.setenv("SERPAPI_API_KEY", "test_key")
        mock_get.side_effect = Exception("Connection error")
        result = web_search("query")
        assert "error" in result
        assert "Connection error" in result["error"]


class TestWeather:
    @patch("agents.voice.tools.weather.requests.get")
    def test_success(self, mock_get, monkeypatch):
        monkeypatch.setenv("OPENWEATHERMAP_API_KEY", "weather_key")
        mock_get.return_value.json.return_value = {
            "name": "London",
            "main": {"temp": 15, "humidity": 70},
            "weather": [{"description": "cloudy"}],
            "wind": {"speed": 5},
        }
        mock_get.return_value.raise_for_status.return_value = None
        result = get_weather("London")
        assert result["city"] == "London"
        assert result["temperature"] == 15
        assert result["description"] == "cloudy"

    def test_missing_api_key(self, monkeypatch):
        monkeypatch.delenv("OPENWEATHERMAP_API_KEY", raising=False)
        result = get_weather("London")
        assert result == {"error": "Weather API key not configured"}

    @patch("agents.voice.tools.weather.requests.get")
    def test_http_error(self, mock_get, monkeypatch):
        monkeypatch.setenv("OPENWEATHERMAP_API_KEY", "weather_key")
        mock_get.side_effect = Exception("Timeout")
        result = get_weather("London")
        assert "error" in result
        assert "Timeout" in result["error"]


class TestTranslation:
    @patch("agents.voice.tools.translation.requests.get")
    def test_success(self, mock_get):
        mock_get.return_value.json.return_value = {
            "responseData": {"translatedText": "Bonjour"}
        }
        mock_get.return_value.raise_for_status.return_value = None
        result = translate("Hello", source="en", target="fr")
        assert result == {"translated_text": "Bonjour"}

    @patch("agents.voice.tools.translation.requests.get")
    def test_http_error(self, mock_get):
        mock_get.side_effect = Exception("Network down")
        result = translate("Hello")
        assert "error" in result
        assert "Network down" in result["error"]


class TestImplementChange:
    def test_return_placeholder(self):
        result = implement_change("fix a bug")
        assert result["status"] == "not_implemented"
        assert "voice agent" in result["message"]
