"""Tests for castwright._types."""

import pytest

from castwright._types import (
    CastwrightError,
    GeneratedExample,
    GenerationConfig,
    GenerationResult,
    OutputFormat,
    ProviderError,
    Seed,
)

# --- Seed ---


class TestSeed:
    def test_basic(self):
        s = Seed(instruction="Do X", output="Done X")
        assert s.instruction == "Do X"
        assert s.output == "Done X"
        assert s.input == ""
        assert s.system == ""
        assert s.metadata == {}

    def test_with_input(self):
        s = Seed(instruction="Do X", output="Done X", input="context")
        assert s.input == "context"

    def test_to_dict_minimal(self):
        s = Seed(instruction="Q", output="A")
        d = s.to_dict()
        assert d == {"instruction": "Q", "output": "A"}
        assert "input" not in d

    def test_to_dict_with_input(self):
        s = Seed(instruction="Q", output="A", input="ctx")
        d = s.to_dict()
        assert d == {"instruction": "Q", "output": "A", "input": "ctx"}

    def test_metadata(self):
        s = Seed(instruction="Q", output="A", metadata={"source": "wiki"})
        assert s.metadata["source"] == "wiki"


# --- GeneratedExample ---


class TestGeneratedExample:
    def test_basic(self):
        ex = GeneratedExample(instruction="Explain X", output="X is...")
        assert ex.instruction == "Explain X"
        assert ex.output == "X is..."
        assert ex.input == ""
        assert ex.system == ""
        assert ex.seed_index is None
        assert ex.quality_score is None
        assert ex.generation_model is None

    def test_to_alpaca_minimal(self):
        ex = GeneratedExample(instruction="Q", output="A")
        d = ex.to_alpaca()
        assert d == {"instruction": "Q", "output": "A"}

    def test_to_alpaca_with_input_and_system(self):
        ex = GeneratedExample(instruction="Q", output="A", input="ctx", system="sys")
        d = ex.to_alpaca()
        assert d == {"instruction": "Q", "output": "A", "input": "ctx", "system": "sys"}

    def test_to_sharegpt_no_system(self):
        ex = GeneratedExample(instruction="Hi", output="Hello")
        d = ex.to_sharegpt()
        assert "conversations" in d
        assert len(d["conversations"]) == 2
        assert d["conversations"][0] == {"from": "human", "value": "Hi"}
        assert d["conversations"][1] == {"from": "gpt", "value": "Hello"}

    def test_to_sharegpt_with_system(self):
        ex = GeneratedExample(instruction="Hi", output="Hello", system="Be kind")
        d = ex.to_sharegpt()
        assert len(d["conversations"]) == 3
        assert d["conversations"][0] == {"from": "system", "value": "Be kind"}

    def test_to_openai_no_system(self):
        ex = GeneratedExample(instruction="Hi", output="Hello")
        d = ex.to_openai()
        assert "messages" in d
        assert len(d["messages"]) == 2
        assert d["messages"][0] == {"role": "user", "content": "Hi"}
        assert d["messages"][1] == {"role": "assistant", "content": "Hello"}

    def test_to_openai_with_system(self):
        ex = GeneratedExample(instruction="Hi", output="Hello", system="Be kind")
        d = ex.to_openai()
        assert len(d["messages"]) == 3
        assert d["messages"][0] == {"role": "system", "content": "Be kind"}

    def test_to_dict_default_alpaca(self):
        ex = GeneratedExample(instruction="Q", output="A")
        assert ex.to_dict() == ex.to_alpaca()

    def test_to_dict_sharegpt(self):
        ex = GeneratedExample(instruction="Q", output="A")
        assert ex.to_dict(OutputFormat.SHAREGPT) == ex.to_sharegpt()

    def test_to_dict_openai(self):
        ex = GeneratedExample(instruction="Q", output="A")
        assert ex.to_dict(OutputFormat.OPENAI) == ex.to_openai()


# --- GenerationConfig ---


class TestGenerationConfig:
    def test_defaults(self):
        c = GenerationConfig()
        assert c.n == 10
        assert c.model == "gpt-4o-mini"
        assert c.temperature == 0.9
        assert c.max_retries == 3
        assert c.diversity_factor == 0.7
        assert c.output_format == OutputFormat.ALPACA

    def test_custom(self):
        c = GenerationConfig(n=50, model="claude-sonnet-4-20250514", temperature=0.5)
        assert c.n == 50
        assert c.model == "claude-sonnet-4-20250514"
        assert c.temperature == 0.5

    def test_invalid_n(self):
        with pytest.raises(ValueError, match="n must be >= 1"):
            GenerationConfig(n=0)

    def test_invalid_n_negative(self):
        with pytest.raises(ValueError, match="n must be >= 1"):
            GenerationConfig(n=-5)

    def test_invalid_temperature_low(self):
        with pytest.raises(ValueError, match="temperature"):
            GenerationConfig(temperature=-0.1)

    def test_invalid_temperature_high(self):
        with pytest.raises(ValueError, match="temperature"):
            GenerationConfig(temperature=2.5)

    def test_invalid_diversity_low(self):
        with pytest.raises(ValueError, match="diversity_factor"):
            GenerationConfig(diversity_factor=-0.1)

    def test_invalid_diversity_high(self):
        with pytest.raises(ValueError, match="diversity_factor"):
            GenerationConfig(diversity_factor=1.5)

    def test_edge_values(self):
        c = GenerationConfig(n=1, temperature=0.0, diversity_factor=0.0)
        assert c.n == 1
        c2 = GenerationConfig(temperature=2.0, diversity_factor=1.0)
        assert c2.temperature == 2.0


# --- GenerationResult ---


class TestGenerationResult:
    def test_basic(self):
        ex = GeneratedExample(instruction="Q", output="A")
        r = GenerationResult(examples=[ex], n_generated=5, n_filtered=4, model="gpt-4o")
        assert r.examples == [ex]
        assert r.n_generated == 5
        assert r.n_filtered == 4
        assert r.model == "gpt-4o"
        assert r.total_input_tokens == 0
        assert r.total_output_tokens == 0

    def test_with_tokens(self):
        r = GenerationResult(
            examples=[], n_generated=0, n_filtered=0, model="x",
            total_input_tokens=100, total_output_tokens=50,
        )
        assert r.total_input_tokens == 100
        assert r.total_output_tokens == 50


# --- OutputFormat ---


class TestOutputFormat:
    def test_values(self):
        assert OutputFormat.ALPACA.value == "alpaca"
        assert OutputFormat.SHAREGPT.value == "sharegpt"
        assert OutputFormat.OPENAI.value == "openai"

    def test_from_string(self):
        assert OutputFormat("alpaca") == OutputFormat.ALPACA
        assert OutputFormat("sharegpt") == OutputFormat.SHAREGPT
        assert OutputFormat("openai") == OutputFormat.OPENAI

    def test_invalid(self):
        with pytest.raises(ValueError):
            OutputFormat("csv")


# --- Exceptions ---


class TestExceptions:
    def test_castwright_error(self):
        e = CastwrightError("bad")
        assert str(e) == "bad"
        assert isinstance(e, Exception)

    def test_provider_error_inherits(self):
        e = ProviderError("oops")
        assert isinstance(e, CastwrightError)
        assert isinstance(e, Exception)
