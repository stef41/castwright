"""castwright — generate synthetic instruction-tuning data from seed examples."""

from castwright._types import (
    CastwrightError,
    GeneratedExample,
    GenerationConfig,
    GenerationResult,
    OutputFormat,
    ProviderError,
    Seed,
)
from castwright.filters import (
    deduplicate_generated,
    filter_examples,
)
from castwright.generate import (
    generate,
    generate_multiturn,
    load_seeds,
    save_results,
)
from castwright.multiturn import (
    Conversation,
    ConversationTurn,
    extend_conversation,
    format_openai,
    format_sharegpt,
    generate_conversation,
)
from castwright.providers import (
    AnthropicProvider,
    LLMProvider,
    MockProvider,
    OllamaProvider,
    OpenAIProvider,
)

__version__ = "0.2.0"

__all__ = [
    "__version__",
    # Types
    "Seed",
    "GeneratedExample",
    "GenerationConfig",
    "GenerationResult",
    "OutputFormat",
    "CastwrightError",
    "ProviderError",
    # Generation
    "generate",
    "generate_multiturn",
    "load_seeds",
    "save_results",
    # Multi-turn
    "ConversationTurn",
    "Conversation",
    "generate_conversation",
    "extend_conversation",
    "format_sharegpt",
    "format_openai",
    # Providers
    "LLMProvider",
    "OpenAIProvider",
    "AnthropicProvider",
    "OllamaProvider",
    "MockProvider",
    # Filters
    "filter_examples",
    "deduplicate_generated",
]
