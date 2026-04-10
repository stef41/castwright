"""Core generation engine."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

from castwright._types import (
    CastwrightError,
    GeneratedExample,
    GenerationConfig,
    GenerationResult,
    OutputFormat,
    ProviderError,
    Seed,
)
from castwright.filters import deduplicate_generated, filter_examples
from castwright.prompts import build_generation_prompt, build_multiturn_prompt
from castwright.providers import LLMProvider, MockProvider


def _parse_generated(
    raw_items: List[Dict[str, Any]],
    seed_index: Optional[int],
    model: str,
) -> List[GeneratedExample]:
    """Parse raw JSON items into GeneratedExample objects."""
    examples: list[GeneratedExample] = []
    for item in raw_items:
        if isinstance(item, dict):
            # Handle both single-turn and multi-turn
            instruction = str(item.get("instruction", ""))
            output = str(item.get("output", ""))
            inp = str(item.get("input", ""))
            system = str(item.get("system", ""))

            # If it's a multi-turn format, flatten to instruction/output
            if "conversations" in item and not instruction:
                convs = item["conversations"]
                for turn in convs:
                    role = turn.get("from", turn.get("role", ""))
                    value = turn.get("value", turn.get("content", ""))
                    if role in ("human", "user") and not instruction:
                        instruction = value
                    elif role in ("gpt", "assistant") and not output:
                        output = value

            if instruction or output:
                examples.append(
                    GeneratedExample(
                        instruction=instruction,
                        output=output,
                        input=inp,
                        system=system,
                        seed_index=seed_index,
                        generation_model=model,
                    )
                )
    return examples


def generate(
    seeds: Sequence[Seed],
    provider: LLMProvider,
    config: Optional[GenerationConfig] = None,
) -> GenerationResult:
    """Generate synthetic instruction-tuning data from seed examples.

    This is the main entry point. Given a set of seed examples and an LLM
    provider, it generates new instruction-output pairs, applies quality
    filtering, and deduplicates the results.

    Parameters
    ----------
    seeds:
        Seed examples to base the generation on.
    provider:
        An LLM provider instance (OpenAI, Anthropic, or Mock).
    config:
        Generation configuration. Uses defaults if None.

    Returns
    -------
    GenerationResult
        The generated examples with metadata.
    """
    if not seeds:
        raise CastwrightError("At least one seed example is required.")

    if config is None:
        config = GenerationConfig()

    all_generated: list[GeneratedExample] = []
    total_input_tokens = 0
    total_output_tokens = 0

    # Generate in batches to avoid hitting token limits
    batch_size = min(config.n, 10)
    n_batches = math.ceil(config.n / batch_size)
    # Generate extra to account for filtering
    extra_factor = 1.3

    system = config.system_prompt or (
        "You are an expert data curator creating high-quality instruction-tuning "
        "examples for training a language model. Generate diverse, accurate, and "
        "well-formatted examples."
    )

    for batch_idx in range(n_batches):
        remaining = config.n - len(all_generated)
        if remaining <= 0:
            break

        batch_n = min(batch_size, int(remaining * extra_factor) + 1)

        prompt = build_generation_prompt(
            seeds=seeds,
            n=batch_n,
            diversity_factor=config.diversity_factor,
            system_context=config.system_prompt if config.system_prompt else "",
        )

        retries = 0
        while retries < config.max_retries:
            try:
                text, in_tok, out_tok = provider.generate(
                    prompt=prompt,
                    system=system,
                    temperature=config.temperature,
                )
                total_input_tokens += in_tok
                total_output_tokens += out_tok

                raw_items = provider.parse_json_array(text)
                examples = _parse_generated(raw_items, seed_index=None, model=config.model)
                all_generated.extend(examples)
                break
            except ProviderError:
                retries += 1
                if retries >= config.max_retries:
                    # Silently continue with what we have
                    break

    # Dedup against seeds
    seed_instructions = {s.instruction for s in seeds}
    deduplicated = deduplicate_generated(all_generated, seed_instructions)

    # Quality filtering
    filtered = filter_examples(deduplicated)

    # Trim to requested count
    final = filtered[: config.n]

    return GenerationResult(
        examples=final,
        n_generated=len(all_generated),
        n_filtered=len(all_generated) - len(filtered),
        model=config.model,
        total_input_tokens=total_input_tokens,
        total_output_tokens=total_output_tokens,
    )


def generate_multiturn(
    seeds: Sequence[Seed],
    provider: LLMProvider,
    n: int = 10,
    turns: int = 3,
    temperature: float = 0.9,
    max_retries: int = 3,
) -> GenerationResult:
    """Generate multi-turn conversation data.

    Parameters
    ----------
    seeds:
        Seed examples showing the domain and style.
    provider:
        LLM provider instance.
    n:
        Number of conversations to generate.
    turns:
        Number of turns per conversation.
    temperature:
        Sampling temperature.
    max_retries:
        Max retries on parse failure.
    """
    if not seeds:
        raise CastwrightError("At least one seed example is required.")

    prompt = build_multiturn_prompt(seeds, n=n, turns=turns)
    system = (
        "You are an expert data curator creating high-quality multi-turn "
        "conversations for training a language model."
    )

    total_input = 0
    total_output = 0
    all_generated: list[GeneratedExample] = []

    retries = 0
    while retries < max_retries:
        try:
            text, in_tok, out_tok = provider.generate(
                prompt=prompt,
                system=system,
                temperature=temperature,
            )
            total_input += in_tok
            total_output += out_tok

            raw = provider.parse_json_array(text)
            for item in raw:
                examples = _parse_generated([item], seed_index=None, model="")
                all_generated.extend(examples)
            break
        except ProviderError:
            retries += 1

    return GenerationResult(
        examples=all_generated[:n],
        n_generated=len(all_generated),
        n_filtered=0,
        model="",
        total_input_tokens=total_input,
        total_output_tokens=total_output,
    )


def save_results(
    result: GenerationResult,
    path: Union[str, Path],
    fmt: OutputFormat = OutputFormat.ALPACA,
) -> None:
    """Save generated examples to a JSONL file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        for ex in result.examples:
            f.write(json.dumps(ex.to_dict(fmt), ensure_ascii=False) + "\n")


def load_seeds(path: Union[str, Path]) -> List[Seed]:
    """Load seed examples from a JSONL or JSON file."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Seed file not found: {path}")

    text = path.read_text(encoding="utf-8").strip()
    if text.startswith("["):
        data = json.loads(text)
    else:
        data = [json.loads(line) for line in text.splitlines() if line.strip()]

    seeds: list[Seed] = []
    for item in data:
        seeds.append(
            Seed(
                instruction=item.get("instruction", item.get("prompt", "")),
                output=item.get("output", item.get("response", "")),
                input=item.get("input", ""),
                system=item.get("system", ""),
            )
        )
    return seeds
