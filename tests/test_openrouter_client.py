import pytest
from unittest.mock import patch, MagicMock
import os
import json
from src.llm_interpreter import OpenRouterClient

class TestOpenRouterClient:
    def test_init_with_api_key(self):
        client = OpenRouterClient(model="test-model", api_key="test-key")
        assert client.model == "test-model"
        assert client.api_key == "test-key"

    def test_init_with_env_var(self):
        with patch.dict(os.environ, {"OPENROUTER_API_KEY": "env-key"}):
            client = OpenRouterClient(model="test-model")
            assert client.api_key == "env-key"

    @patch("requests.post")
    def test_complete_success(self, mock_post):
        # Mock response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [
                {
                    "message": {
                        "content": '{"minimum": 18}'
                    }
                }
            ]
        }
        mock_post.return_value = mock_response

        client = OpenRouterClient(model="test-model", api_key="test-key")
        result = client.complete("test prompt")

        assert result == '{"minimum": 18}'
        mock_post.assert_called_once()
        args, kwargs = mock_post.call_args
        assert kwargs["headers"]["Authorization"] == "Bearer test-key"
        data = json.loads(kwargs["data"])
        assert data["model"] == "test-model"
        assert data["messages"][0]["content"] == "test prompt"

    @patch("requests.post")
    def test_complete_no_api_key(self, mock_post):
        with patch.dict(os.environ, {}, clear=True):
            client = OpenRouterClient(model="test-model")
            result = client.complete("test prompt")
            assert result == ""
            mock_post.assert_not_called()

    @patch("requests.post")
    def test_complete_error(self, mock_post):
        mock_post.side_effect = Exception("API Error")
        client = OpenRouterClient(model="test-model", api_key="test-key")
        result = client.complete("test prompt")
        assert result == ""

    @patch("requests.post")
    def test_records_model_and_provider_that_served_each_call(self, mock_post):
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "id": "gen-123",
            "model": "deepseek/deepseek-chat-v3-0324",
            "provider": "DeepInfra",
            "choices": [{"message": {"content": '{"minimum": 18}'}}],
        }
        mock_post.return_value = mock_response

        client = OpenRouterClient(model="deepseek/deepseek-chat", api_key="test-key")
        client.complete("test prompt")

        assert client.call_log == [{
            "requested_model": "deepseek/deepseek-chat",
            "model": "deepseek/deepseek-chat-v3-0324",
            "provider": "DeepInfra",
            "id": "gen-123",
        }]

    @patch("requests.post")
    def test_failed_call_is_not_logged_as_served(self, mock_post):
        mock_post.side_effect = Exception("API Error")
        client = OpenRouterClient(model="test-model", api_key="test-key")
        client.complete("test prompt")
        assert client.call_log == []

    @patch("requests.post")
    def test_requests_identify_this_project(self, mock_post):
        mock_post.return_value.json.return_value = {
            "choices": [{"message": {"content": "{}"}}]
        }
        OpenRouterClient(model="test-model", api_key="test-key").complete("p")

        headers = mock_post.call_args.kwargs["headers"]
        assert headers["X-Title"] == "constraints_2_json"
        assert headers["HTTP-Referer"] == "https://github.com/carlhuxley/constraints_2_json"
