"""Tests for the castwright public API."""

import castwright


class TestPublicAPI:
    def test_version(self):
        assert castwright.__version__ == "0.1.0"

    def test_types_exported(self):
        assert hasattr(castwright, "Seed")
        assert hasattr(castwright, "GeneratedExample")
        assert hasattr(castwright, "GenerationConfig")
        assert hasattr(castwright, "GenerationResult")
        assert hasattr(castwright, "OutputFormat")
        assert hasattr(castwright, "CastwrightError")
        assert hasattr(castwright, "ProviderError")

    def test_generation_exported(self):
        assert hasattr(castwright, "generate")
        assert hasattr(castwright, "generate_multiturn")
        assert hasattr(castwright, "load_seeds")
        assert hasattr(castwright, "save_results")

    def test_providers_exported(self):
        assert hasattr(castwright, "LLMProvider")
        assert hasattr(castwright, "OpenAIProvider")
        assert hasattr(castwright, "AnthropicProvider")
        assert hasattr(castwright, "MockProvider")

    def test_filters_exported(self):
        assert hasattr(castwright, "filter_examples")
        assert hasattr(castwright, "deduplicate_generated")

    def test_all_list(self):
        for name in castwright.__all__:
            assert hasattr(castwright, name), f"{name} in __all__ but not importable"
