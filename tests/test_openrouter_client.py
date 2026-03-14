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
