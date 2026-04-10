"""Tests for castwright.prompts."""

from castwright._types import Seed
from castwright.prompts import (
    build_generation_prompt,
    build_multiturn_prompt,
    build_quality_check_prompt,
    format_seed_examples,
)

SEEDS = [
    Seed(instruction="Explain recursion", output="Recursion is when a function calls itself."),
    Seed(instruction="What is a hash table?", output="A data structure mapping keys to values."),
    Seed(instruction="Define polymorphism", output="Polymorphism allows objects to take many forms."),
]


class TestFormatSeedExamples:
    def test_formats_all_seeds(self):
        text = format_seed_examples(SEEDS)
        assert "Example 1:" in text
        assert "Example 2:" in text
        assert "Example 3:" in text
        assert "Explain recursion" in text
        assert "hash table" in text

    def test_with_input_field(self):
        seeds = [Seed(instruction="Q", output="A", input="some context")]
        text = format_seed_examples(seeds)
        assert "Input: some context" in text

    def test_max_examples_limits(self):
        many = [Seed(instruction=f"Q{i}", output=f"A{i}") for i in range(20)]
        text = format_seed_examples(many, max_examples=3)
        # Should only have 3 examples
        count = text.count("Example ")
        assert count == 3

    def test_single_seed(self):
        text = format_seed_examples([SEEDS[0]])
        assert "Example 1:" in text
        assert "Explain recursion" in text


class TestBuildGenerationPrompt:
    def test_contains_seeds(self):
        prompt = build_generation_prompt(SEEDS, n=10)
        assert "Explain recursion" in prompt
        assert "hash table" in prompt

    def test_contains_n(self):
        prompt = build_generation_prompt(SEEDS, n=42)
        assert "42" in prompt

    def test_high_diversity(self):
        prompt = build_generation_prompt(SEEDS, n=5, diversity_factor=0.8)
        assert "significantly different" in prompt

    def test_medium_diversity(self):
        prompt = build_generation_prompt(SEEDS, n=5, diversity_factor=0.5)
        assert "same general domain" in prompt

    def test_low_diversity(self):
        prompt = build_generation_prompt(SEEDS, n=5, diversity_factor=0.2)
        assert "very similar" in prompt

    def test_with_system_context(self):
        prompt = build_generation_prompt(SEEDS, n=5, system_context="Medical domain")
        assert "Medical domain" in prompt

    def test_json_array_mentioned(self):
        prompt = build_generation_prompt(SEEDS, n=5)
        assert "JSON array" in prompt

    def test_no_copy_instruction(self):
        prompt = build_generation_prompt(SEEDS, n=5)
        assert "NOT copy" in prompt


class TestBuildMultiturnPrompt:
    def test_contains_turns(self):
        prompt = build_multiturn_prompt(SEEDS, n=10, turns=4)
        assert "4 turns" in prompt

    def test_contains_n(self):
        prompt = build_multiturn_prompt(SEEDS, n=15)
        assert "15" in prompt

    def test_has_example_format(self):
        prompt = build_multiturn_prompt(SEEDS, n=5)
        assert "conversations" in prompt
        assert "human" in prompt
        assert "gpt" in prompt


class TestBuildQualityCheckPrompt:
    def test_contains_instruction_and_output(self):
        prompt = build_quality_check_prompt("Do X", "Done X in detail")
        assert "Do X" in prompt
        assert "Done X in detail" in prompt

    def test_asks_for_rating(self):
        prompt = build_quality_check_prompt("Q", "A")
        assert "1" in prompt and "10" in prompt
