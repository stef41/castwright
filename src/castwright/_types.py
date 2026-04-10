"""Core types for castwright."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional


class OutputFormat(str, Enum):
    """Output format for generated examples."""

    ALPACA = "alpaca"
    SHAREGPT = "sharegpt"
    OPENAI = "openai"


@dataclass
class Seed:
    """A seed example for generation.

    Provide at least ``instruction`` and ``output``.
    """

    instruction: str
    output: str
    input: str = ""
    system: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, str]:
        d: Dict[str, str] = {
            "instruction": self.instruction,
            "output": self.output,
        }
        if self.input:
            d["input"] = self.input
        return d


@dataclass
class GeneratedExample:
    """A single generated example with metadata."""

    instruction: str
    output: str
    input: str = ""
    system: str = ""
    seed_index: Optional[int] = None
    quality_score: Optional[float] = None
    generation_model: Optional[str] = None

    def to_alpaca(self) -> Dict[str, str]:
        d: Dict[str, str] = {"instruction": self.instruction, "output": self.output}
        if self.input:
            d["input"] = self.input
        if self.system:
            d["system"] = self.system
        return d

    def to_sharegpt(self) -> Dict[str, Any]:
        conversations: list[dict[str, str]] = []
        if self.system:
            conversations.append({"from": "system", "value": self.system})
        conversations.append({"from": "human", "value": self.instruction})
        conversations.append({"from": "gpt", "value": self.output})
        return {"conversations": conversations}

    def to_openai(self) -> Dict[str, Any]:
        messages: list[dict[str, str]] = []
        if self.system:
            messages.append({"role": "system", "content": self.system})
        messages.append({"role": "user", "content": self.instruction})
        messages.append({"role": "assistant", "content": self.output})
        return {"messages": messages}

    def to_dict(self, fmt: OutputFormat = OutputFormat.ALPACA) -> Dict[str, Any]:
        if fmt == OutputFormat.SHAREGPT:
            return self.to_sharegpt()
        if fmt == OutputFormat.OPENAI:
            return self.to_openai()
        return self.to_alpaca()


@dataclass
class GenerationConfig:
    """Configuration for data generation."""

    n: int = 10
    model: str = "gpt-4o-mini"
    temperature: float = 0.9
    max_retries: int = 3
    system_prompt: str = ""
    diversity_factor: float = 0.7
    min_quality: float = 0.0
    output_format: OutputFormat = OutputFormat.ALPACA

    def __post_init__(self) -> None:
        if self.n < 1:
            raise ValueError("n must be >= 1")
        if not (0.0 <= self.temperature <= 2.0):
            raise ValueError("temperature must be between 0.0 and 2.0")
        if not (0.0 <= self.diversity_factor <= 1.0):
            raise ValueError("diversity_factor must be between 0.0 and 1.0")


@dataclass
class GenerationResult:
    """Result of a generation run."""

    examples: List[GeneratedExample]
    n_generated: int
    n_filtered: int
    model: str
    total_input_tokens: int = 0
    total_output_tokens: int = 0


class CastwrightError(Exception):
    """Base exception."""


class ProviderError(CastwrightError):
    """Error communicating with the LLM provider."""
