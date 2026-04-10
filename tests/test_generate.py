"""Tests for castwright.generate."""

import json

import pytest

from castwright._types import (
    CastwrightError,
    GeneratedExample,
    GenerationConfig,
    GenerationResult,
    OutputFormat,
    Seed,
)
from castwright.generate import (
    _parse_generated,
    generate,
    generate_multiturn,
    load_seeds,
    save_results,
)
from castwright.providers import MockProvider

SEEDS = [
    Seed(instruction="Explain recursion in programming", output="Recursion is a technique where a function calls itself to solve smaller subproblems."),
    Seed(instruction="What is a binary search tree?", output="A BST is a tree data structure where each node has at most two children..."),
]


# --- _parse_generated ---


class TestParseGenerated:
    def test_basic_item(self):
        raw = [{"instruction": "Q1", "output": "A1"}]
        result = _parse_generated(raw, seed_index=0, model="gpt-4o")
        assert len(result) == 1
        assert result[0].instruction == "Q1"
        assert result[0].output == "A1"
        assert result[0].seed_index == 0
        assert result[0].generation_model == "gpt-4o"

    def test_with_input_and_system(self):
        raw = [{"instruction": "Q", "output": "A", "input": "ctx", "system": "sys"}]
        result = _parse_generated(raw, seed_index=None, model="m")
        assert result[0].input == "ctx"
        assert result[0].system == "sys"

    def test_multiturn_fallback(self):
        raw = [{
            "conversations": [
                {"from": "human", "value": "Hello"},
                {"from": "gpt", "value": "Hi there!"},
            ]
        }]
        result = _parse_generated(raw, seed_index=None, model="m")
        assert len(result) == 1
        assert result[0].instruction == "Hello"
        assert result[0].output == "Hi there!"

    def test_skips_non_dict(self):
        raw = ["not a dict", {"instruction": "Q", "output": "A"}]
        result = _parse_generated(raw, seed_index=None, model="m")
        assert len(result) == 1

    def test_empty_instruction_and_output_skipped(self):
        raw = [{"instruction": "", "output": ""}]
        result = _parse_generated(raw, seed_index=None, model="m")
        assert len(result) == 0

    def test_empty_list(self):
        assert _parse_generated([], seed_index=None, model="m") == []

    def test_multiple_items(self):
        raw = [
            {"instruction": f"Q{i}", "output": f"A{i}"}
            for i in range(5)
        ]
        result = _parse_generated(raw, seed_index=None, model="m")
        assert len(result) == 5


# --- generate ---


class TestGenerate:
    def test_basic_generation(self):
        mock = MockProvider()
        result = generate(SEEDS, mock, GenerationConfig(n=3))
        assert isinstance(result, GenerationResult)
        assert result.n_generated > 0

    def test_empty_seeds_raises(self):
        with pytest.raises(CastwrightError, match="At least one seed"):
            generate([], MockProvider())

    def test_default_config(self):
        mock = MockProvider()
        result = generate(SEEDS, mock)
        assert isinstance(result, GenerationResult)

    def test_custom_mock_response(self):
        response = json.dumps([
            {"instruction": "Custom question about algorithms?", "output": "Detailed answer about algorithms."},
            {"instruction": "Another custom question here?", "output": "Another detailed answer here."},
        ])
        mock = MockProvider(responses=[response])
        result = generate(SEEDS, mock, GenerationConfig(n=2))
        assert isinstance(result, GenerationResult)

    def test_dedup_against_seeds(self):
        # If mock generates something matching a seed, it gets deduped
        response = json.dumps([
            {"instruction": "Explain recursion in programming", "output": "Duplicate of seed"},
            {"instruction": "Brand new unique question?", "output": "Fresh answer here."},
        ])
        mock = MockProvider(responses=[response])
        result = generate(SEEDS, mock, GenerationConfig(n=2))
        # The seed-matching one should be deduped
        for ex in result.examples:
            assert ex.instruction != "Explain recursion in programming"

    def test_result_has_model(self):
        mock = MockProvider()
        config = GenerationConfig(model="test-model")
        result = generate(SEEDS, mock, config)
        assert result.model == "test-model"

    def test_result_has_token_counts(self):
        mock = MockProvider()
        result = generate(SEEDS, mock)
        assert result.total_input_tokens >= 0
        assert result.total_output_tokens >= 0

    def test_large_n(self):
        mock = MockProvider()
        result = generate(SEEDS, mock, GenerationConfig(n=50))
        assert isinstance(result, GenerationResult)


# --- generate_multiturn ---


class TestGenerateMultiturn:
    def test_basic(self):
        response = json.dumps([{
            "conversations": [
                {"from": "human", "value": "Tell me about sorting"},
                {"from": "gpt", "value": "Sorting algorithms arrange data..."},
                {"from": "human", "value": "What about quicksort?"},
                {"from": "gpt", "value": "Quicksort uses a pivot element..."},
            ]
        }])
        mock = MockProvider(responses=[response])
        result = generate_multiturn(SEEDS, mock, n=1, turns=2)
        assert isinstance(result, GenerationResult)

    def test_empty_seeds_raises(self):
        with pytest.raises(CastwrightError, match="At least one seed"):
            generate_multiturn([], MockProvider())


# --- load_seeds ---


class TestLoadSeeds:
    def test_load_jsonl(self, tmp_path):
        f = tmp_path / "seeds.jsonl"
        f.write_text(
            '{"instruction": "Q1", "output": "A1"}\n'
            '{"instruction": "Q2", "output": "A2"}\n'
        )
        seeds = load_seeds(f)
        assert len(seeds) == 2
        assert seeds[0].instruction == "Q1"
        assert seeds[1].instruction == "Q2"

    def test_load_json_array(self, tmp_path):
        f = tmp_path / "seeds.json"
        f.write_text(json.dumps([
            {"instruction": "Q1", "output": "A1"},
            {"instruction": "Q2", "output": "A2"},
        ]))
        seeds = load_seeds(f)
        assert len(seeds) == 2

    def test_load_prompt_response_format(self, tmp_path):
        f = tmp_path / "seeds.jsonl"
        f.write_text('{"prompt": "P1", "response": "R1"}\n')
        seeds = load_seeds(f)
        assert seeds[0].instruction == "P1"
        assert seeds[0].output == "R1"

    def test_with_input_and_system(self, tmp_path):
        f = tmp_path / "seeds.jsonl"
        f.write_text('{"instruction": "Q", "output": "A", "input": "ctx", "system": "sys"}\n')
        seeds = load_seeds(f)
        assert seeds[0].input == "ctx"
        assert seeds[0].system == "sys"

    def test_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            load_seeds("/nonexistent/file.jsonl")

    def test_empty_lines_ignored(self, tmp_path):
        f = tmp_path / "seeds.jsonl"
        f.write_text('{"instruction": "Q", "output": "A"}\n\n\n')
        seeds = load_seeds(f)
        assert len(seeds) == 1

    def test_string_path(self, tmp_path):
        f = tmp_path / "seeds.jsonl"
        f.write_text('{"instruction": "Q", "output": "A"}\n')
        seeds = load_seeds(str(f))
        assert len(seeds) == 1


# --- save_results ---


class TestSaveResults:
    def test_save_alpaca(self, tmp_path):
        result = GenerationResult(
            examples=[
                GeneratedExample(instruction="Q1", output="A1"),
                GeneratedExample(instruction="Q2", output="A2"),
            ],
            n_generated=2, n_filtered=0, model="m",
        )
        out = tmp_path / "out.jsonl"
        save_results(result, out)

        lines = out.read_text().strip().split("\n")
        assert len(lines) == 2
        data = json.loads(lines[0])
        assert data["instruction"] == "Q1"

    def test_save_sharegpt(self, tmp_path):
        result = GenerationResult(
            examples=[GeneratedExample(instruction="Q", output="A")],
            n_generated=1, n_filtered=0, model="m",
        )
        out = tmp_path / "out.jsonl"
        save_results(result, out, OutputFormat.SHAREGPT)

        data = json.loads(out.read_text().strip())
        assert "conversations" in data

    def test_save_openai(self, tmp_path):
        result = GenerationResult(
            examples=[GeneratedExample(instruction="Q", output="A")],
            n_generated=1, n_filtered=0, model="m",
        )
        out = tmp_path / "out.jsonl"
        save_results(result, out, OutputFormat.OPENAI)

        data = json.loads(out.read_text().strip())
        assert "messages" in data

    def test_creates_parent_dirs(self, tmp_path):
        result = GenerationResult(
            examples=[GeneratedExample(instruction="Q", output="A")],
            n_generated=1, n_filtered=0, model="m",
        )
        out = tmp_path / "sub" / "dir" / "out.jsonl"
        save_results(result, out)
        assert out.exists()

    def test_empty_result(self, tmp_path):
        result = GenerationResult(examples=[], n_generated=0, n_filtered=0, model="m")
        out = tmp_path / "out.jsonl"
        save_results(result, out)
        assert out.read_text() == ""

    def test_unicode_content(self, tmp_path):
        result = GenerationResult(
            examples=[GeneratedExample(instruction="日本語", output="答え")],
            n_generated=1, n_filtered=0, model="m",
        )
        out = tmp_path / "out.jsonl"
        save_results(result, out)
        data = json.loads(out.read_text().strip())
        assert data["instruction"] == "日本語"
