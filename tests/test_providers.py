"""Tests for castwright.providers."""

import json

import pytest

from castwright._types import ProviderError
from castwright.providers import LLMProvider, MockProvider


class TestParseJsonArray:
    """Test the static parse_json_array method on LLMProvider."""

    def test_plain_array(self):
        text = '[{"instruction": "Q", "output": "A"}]'
        result = LLMProvider.parse_json_array(text)
        assert len(result) == 1
        assert result[0]["instruction"] == "Q"

    def test_markdown_code_block(self):
        text = '```json\n[{"instruction": "Q", "output": "A"}]\n```'
        result = LLMProvider.parse_json_array(text)
        assert len(result) == 1

    def test_markdown_block_no_language(self):
        text = '```\n[{"instruction": "Q", "output": "A"}]\n```'
        result = LLMProvider.parse_json_array(text)
        assert len(result) == 1

    def test_trailing_comma(self):
        text = '[{"instruction": "Q", "output": "A",}]'
        result = LLMProvider.parse_json_array(text)
        assert len(result) == 1

    def test_trailing_comma_in_array(self):
        text = '[{"instruction": "Q", "output": "A"},]'
        result = LLMProvider.parse_json_array(text)
        assert len(result) == 1

    def test_multiple_trailing_commas(self):
        text = '[{"a": "1", "b": "2",}, {"c": "3",},]'
        result = LLMProvider.parse_json_array(text)
        assert len(result) == 2

    def test_multiple_items(self):
        text = json.dumps([
            {"instruction": "Q1", "output": "A1"},
            {"instruction": "Q2", "output": "A2"},
            {"instruction": "Q3", "output": "A3"},
        ])
        result = LLMProvider.parse_json_array(text)
        assert len(result) == 3

    def test_text_around_json(self):
        text = 'Here are the examples:\n[{"instruction": "Q", "output": "A"}]\nDone!'
        result = LLMProvider.parse_json_array(text)
        assert len(result) == 1

    def test_no_array(self):
        with pytest.raises(ProviderError, match="No JSON array"):
            LLMProvider.parse_json_array("Just some text without JSON")

    def test_empty_array(self):
        result = LLMProvider.parse_json_array("[]")
        assert result == []

    def test_malformed_json(self):
        with pytest.raises(ProviderError, match="Failed to parse"):
            LLMProvider.parse_json_array("[{bad json}]")

    def test_not_array(self):
        # This path is hard to trigger since [ ] extraction always yields a list
        # if json.loads succeeds. Just verify the method handles it gracefully.
        result = LLMProvider.parse_json_array("[]")
        assert result == []

    def test_nested_objects(self):
        text = json.dumps([
            {"instruction": "Q", "output": "A", "metadata": {"source": "wiki"}},
        ])
        result = LLMProvider.parse_json_array(text)
        assert result[0]["metadata"]["source"] == "wiki"

    def test_unicode(self):
        text = json.dumps([{"instruction": "日本語", "output": "答え"}])
        result = LLMProvider.parse_json_array(text)
        assert result[0]["instruction"] == "日本語"


class TestMockProvider:
    def test_default_response(self):
        mock = MockProvider()
        text, in_tok, out_tok = mock.generate("Some prompt")
        data = json.loads(text)
        assert isinstance(data, list)
        assert len(data) == 3
        assert "instruction" in data[0]
        assert in_tok > 0
        assert out_tok > 0

    def test_custom_responses(self):
        responses = [
            json.dumps([{"instruction": "Custom Q", "output": "Custom A"}]),
            json.dumps([{"instruction": "Second Q", "output": "Second A"}]),
        ]
        mock = MockProvider(responses=responses)

        text1, _, _ = mock.generate("prompt 1")
        assert "Custom Q" in text1

        text2, _, _ = mock.generate("prompt 2")
        assert "Second Q" in text2

    def test_response_cycling(self):
        responses = [json.dumps([{"instruction": "Only", "output": "One"}])]
        mock = MockProvider(responses=responses)
        text1, _, _ = mock.generate("p1")
        text2, _, _ = mock.generate("p2")
        assert text1 == text2

    def test_call_count(self):
        mock = MockProvider()
        assert mock._call_count == 0
        mock.generate("a")
        assert mock._call_count == 1
        mock.generate("b")
        assert mock._call_count == 2

    def test_system_and_temperature_accepted(self):
        mock = MockProvider()
        text, _, _ = mock.generate("p", system="sys", temperature=0.5, max_tokens=1000)
        assert text  # just check it doesn't error
