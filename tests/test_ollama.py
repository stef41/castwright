"""Tests for castwright OllamaProvider."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from castwright._types import ProviderError
from castwright.providers import OllamaProvider


class TestOllamaProviderInit:
    def test_default_init(self):
        p = OllamaProvider()
        assert p._model == "llama3"
        assert p._base_url == "http://localhost:11434/v1"

    def test_custom_model(self):
        p = OllamaProvider(model="mistral")
        assert p._model == "mistral"

    def test_custom_base_url(self):
        p = OllamaProvider(base_url="http://myhost:8000/v1")
        assert p._base_url == "http://myhost:8000/v1"

    def test_custom_host(self):
        p = OllamaProvider(host="http://myhost:11434")
        assert p._base_url == "http://myhost:11434/v1"


class TestOllamaProviderGenerate:
    def _mock_response(self, text: str, prompt_tokens: int = 10, completion_tokens: int = 20):
        return {
            "choices": [{"message": {"content": text}}],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
            },
        }

    @patch("castwright.providers.urlopen")
    def test_basic_generation(self, mock_urlopen):
        resp_data = self._mock_response("Hello world")
        mock_resp = MagicMock()
        mock_resp.read.return_value = json.dumps(resp_data).encode()
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_resp

        p = OllamaProvider(model="llama3")
        text, in_tok, out_tok = p.generate("Say hello")

        assert text == "Hello world"
        assert in_tok == 10
        assert out_tok == 20

    @patch("castwright.providers.urlopen")
    def test_with_system_prompt(self, mock_urlopen):
        resp_data = self._mock_response("response")
        mock_resp = MagicMock()
        mock_resp.read.return_value = json.dumps(resp_data).encode()
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_resp

        p = OllamaProvider()
        p.generate("prompt", system="You are helpful")

        # Verify the request was made
        call_args = mock_urlopen.call_args
        req = call_args[0][0]
        body = json.loads(req.data)
        assert body["messages"][0]["role"] == "system"
        assert body["messages"][0]["content"] == "You are helpful"
        assert body["messages"][1]["role"] == "user"

    @patch("castwright.providers.urlopen")
    def test_temperature_and_max_tokens(self, mock_urlopen):
        resp_data = self._mock_response("ok")
        mock_resp = MagicMock()
        mock_resp.read.return_value = json.dumps(resp_data).encode()
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_resp

        p = OllamaProvider()
        p.generate("hi", temperature=0.1, max_tokens=512)

        call_args = mock_urlopen.call_args
        req = call_args[0][0]
        body = json.loads(req.data)
        assert body["temperature"] == 0.1
        assert body["max_tokens"] == 512

    @patch("castwright.providers.urlopen")
    def test_no_usage_field(self, mock_urlopen):
        resp_data = {"choices": [{"message": {"content": "hi"}}]}
        mock_resp = MagicMock()
        mock_resp.read.return_value = json.dumps(resp_data).encode()
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_resp

        p = OllamaProvider()
        text, in_tok, out_tok = p.generate("prompt")
        assert text == "hi"
        assert in_tok == 0
        assert out_tok == 0

    @patch("castwright.providers.urlopen")
    def test_empty_choices(self, mock_urlopen):
        resp_data = {"choices": [{"message": {"content": ""}}], "usage": {"prompt_tokens": 5, "completion_tokens": 0}}
        mock_resp = MagicMock()
        mock_resp.read.return_value = json.dumps(resp_data).encode()
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_resp

        p = OllamaProvider()
        text, _, _ = p.generate("prompt")
        assert text == ""

    @patch("castwright.providers.urlopen")
    def test_network_error_raises_provider_error(self, mock_urlopen):
        from urllib.error import URLError
        mock_urlopen.side_effect = URLError("Connection refused")

        p = OllamaProvider()
        with pytest.raises(ProviderError, match="Ollama API error"):
            p.generate("prompt")

    @patch("castwright.providers.urlopen")
    def test_parse_json_array_inherited(self, mock_urlopen):
        """OllamaProvider should inherit parse_json_array from LLMProvider."""
        p = OllamaProvider()
        result = p.parse_json_array('[{"a": 1}]')
        assert result == [{"a": 1}]

    @patch("castwright.providers.urlopen")
    def test_without_system(self, mock_urlopen):
        resp_data = self._mock_response("response")
        mock_resp = MagicMock()
        mock_resp.read.return_value = json.dumps(resp_data).encode()
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_resp

        p = OllamaProvider()
        p.generate("prompt")

        call_args = mock_urlopen.call_args
        req = call_args[0][0]
        body = json.loads(req.data)
        assert len(body["messages"]) == 1
        assert body["messages"][0]["role"] == "user"

    @patch("castwright.providers.urlopen")
    def test_model_in_request(self, mock_urlopen):
        resp_data = self._mock_response("ok")
        mock_resp = MagicMock()
        mock_resp.read.return_value = json.dumps(resp_data).encode()
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_resp

        p = OllamaProvider(model="phi3")
        p.generate("hi")

        call_args = mock_urlopen.call_args
        req = call_args[0][0]
        body = json.loads(req.data)
        assert body["model"] == "phi3"
