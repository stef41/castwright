"""LLM provider abstraction for generation.

Supports OpenAI-compatible APIs and Anthropic. Each provider is optional;
an ImportError is raised if the required SDK is not installed.
"""

from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple
from urllib.request import Request, urlopen

from castwright._types import ProviderError


class LLMProvider(ABC):
    """Abstract base class for LLM providers."""

    @abstractmethod
    def generate(
        self,
        prompt: str,
        system: str = "",
        temperature: float = 0.9,
        max_tokens: int = 4096,
    ) -> Tuple[str, int, int]:
        """Generate a completion.

        Returns (text, input_tokens, output_tokens).
        """
        ...

    @staticmethod
    def parse_json_array(text: str) -> List[Dict[str, Any]]:
        """Extract a JSON array from the LLM response.

        Handles common issues: markdown code blocks, trailing commas, etc.
        """
        # Strip markdown code blocks
        cleaned = text.strip()
        if cleaned.startswith("```"):
            lines = cleaned.split("\n")
            # Remove first and last lines (``` markers)
            lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            cleaned = "\n".join(lines)

        # Try to find JSON array
        cleaned = cleaned.strip()

        # Find the first [ and last ]
        start = cleaned.find("[")
        end = cleaned.rfind("]")
        if start == -1 or end == -1 or end <= start:
            raise ProviderError(f"No JSON array found in response: {text[:200]}")

        json_str = cleaned[start : end + 1]

        # Remove trailing commas before ] or }
        json_str = re.sub(r",\s*([}\]])", r"\1", json_str)

        try:
            data = json.loads(json_str)
        except json.JSONDecodeError as e:
            raise ProviderError(f"Failed to parse JSON: {e}\nText: {json_str[:500]}") from e

        if not isinstance(data, list):
            raise ProviderError(f"Expected JSON array, got {type(data).__name__}")

        return data


class OpenAIProvider(LLMProvider):
    """OpenAI API provider (works with any OpenAI-compatible API)."""

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
    ) -> None:
        try:
            import openai
        except ImportError:
            raise ImportError(
                "OpenAI provider requires the openai package. "
                "Install with: pip install castwright[openai]"
            ) from None

        kwargs: Dict[str, Any] = {}
        if api_key:
            kwargs["api_key"] = api_key
        if base_url:
            kwargs["base_url"] = base_url

        self._client = openai.OpenAI(**kwargs)
        self._model = model

    def generate(
        self,
        prompt: str,
        system: str = "",
        temperature: float = 0.9,
        max_tokens: int = 4096,
    ) -> Tuple[str, int, int]:
        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        try:
            response = self._client.chat.completions.create(
                model=self._model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
        except Exception as e:
            raise ProviderError(f"OpenAI API error: {e}") from e

        text = response.choices[0].message.content or ""
        usage = response.usage
        input_tokens = usage.prompt_tokens if usage else 0
        output_tokens = usage.completion_tokens if usage else 0

        return text, input_tokens, output_tokens


class AnthropicProvider(LLMProvider):
    """Anthropic Claude API provider."""

    def __init__(
        self,
        model: str = "claude-sonnet-4-20250514",
        api_key: Optional[str] = None,
    ) -> None:
        try:
            import anthropic
        except ImportError:
            raise ImportError(
                "Anthropic provider requires the anthropic package. "
                "Install with: pip install castwright[anthropic]"
            ) from None

        kwargs: Dict[str, Any] = {}
        if api_key:
            kwargs["api_key"] = api_key

        self._client = anthropic.Anthropic(**kwargs)
        self._model = model

    def generate(
        self,
        prompt: str,
        system: str = "",
        temperature: float = 0.9,
        max_tokens: int = 4096,
    ) -> Tuple[str, int, int]:
        kwargs: Dict[str, Any] = {
            "model": self._model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": [{"role": "user", "content": prompt}],
        }
        if system:
            kwargs["system"] = system

        try:
            response = self._client.messages.create(**kwargs)
        except Exception as e:
            raise ProviderError(f"Anthropic API error: {e}") from e

        text = response.content[0].text if response.content else ""
        input_tokens = response.usage.input_tokens
        output_tokens = response.usage.output_tokens

        return text, input_tokens, output_tokens


class OllamaProvider(LLMProvider):
    """Ollama local LLM provider (uses its OpenAI-compatible API).

    Connects to Ollama at ``http://localhost:11434`` by default.
    No extra dependencies required — uses only ``urllib``.
    """

    def __init__(
        self,
        model: str = "llama3",
        host: Optional[str] = None,
        base_url: Optional[str] = None,
    ) -> None:
        if base_url:
            self._base_url = base_url.rstrip("/")
        elif host:
            self._base_url = host.rstrip("/") + "/v1"
        else:
            self._base_url = "http://localhost:11434/v1"
        self._model = model

    def generate(
        self,
        prompt: str,
        system: str = "",
        temperature: float = 0.9,
        max_tokens: int = 4096,
    ) -> Tuple[str, int, int]:
        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        payload = json.dumps({
            "model": self._model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }).encode()

        req = Request(
            f"{self._base_url}/chat/completions",
            data=payload,
            headers={"Content-Type": "application/json"},
        )

        try:
            with urlopen(req) as resp:
                data = json.loads(resp.read())
        except Exception as e:
            raise ProviderError(f"Ollama API error: {e}") from e

        text = data["choices"][0]["message"]["content"] if data.get("choices") else ""
        usage = data.get("usage", {})
        input_tokens = usage.get("prompt_tokens", 0)
        output_tokens = usage.get("completion_tokens", 0)

        return text, input_tokens, output_tokens


class MockProvider(LLMProvider):
    """Mock provider for testing without API calls.

    Generates deterministic fake data based on seed examples.
    """

    def __init__(self, responses: Optional[List[str]] = None) -> None:
        self._responses = responses or []
        self._call_count = 0

    def generate(
        self,
        prompt: str,
        system: str = "",
        temperature: float = 0.9,
        max_tokens: int = 4096,
    ) -> Tuple[str, int, int]:
        if self._responses:
            idx = self._call_count % len(self._responses)
            self._call_count += 1
            return self._responses[idx], len(prompt.split()), 100
        # Default: generate a simple response
        self._call_count += 1
        result = json.dumps([
            {
                "instruction": f"Generated question {i+1} from prompt",
                "output": f"Generated answer {i+1} with detailed explanation.",
            }
            for i in range(3)
        ])
        return result, len(prompt.split()), len(result.split())
