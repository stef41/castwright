"""Prompt templates for synthetic data generation.

These prompts instruct the LLM to generate new instruction-output pairs
that are diverse, high-quality, and similar in style to the seed examples.
"""

from __future__ import annotations

import json
import random
from typing import List, Sequence

from castwright._types import Seed


def format_seed_examples(seeds: Sequence[Seed], max_examples: int = 5) -> str:
    """Format seed examples as a numbered list for the prompt."""
    selected = list(seeds)
    if len(selected) > max_examples:
        selected = random.sample(selected, max_examples)

    lines = []
    for i, seed in enumerate(selected, 1):
        lines.append(f"Example {i}:")
        lines.append(f"  Instruction: {seed.instruction}")
        if seed.input:
            lines.append(f"  Input: {seed.input}")
        lines.append(f"  Output: {seed.output}")
        lines.append("")
    return "\n".join(lines)


def build_generation_prompt(
    seeds: Sequence[Seed],
    n: int,
    diversity_factor: float = 0.7,
    system_context: str = "",
) -> str:
    """Build the prompt that asks the LLM to generate new examples.

    Parameters
    ----------
    seeds:
        Seed examples to base the generation on.
    n:
        Number of new examples to generate.
    diversity_factor:
        0.0 = very similar to seeds, 1.0 = very diverse.
    system_context:
        Optional domain context to include.
    """
    seed_text = format_seed_examples(seeds)

    diversity_instruction = ""
    if diversity_factor > 0.6:
        diversity_instruction = (
            "Make each example significantly different from the seeds and from each other. "
            "Vary the topic, complexity, style, and length. "
        )
    elif diversity_factor > 0.3:
        diversity_instruction = (
            "Vary the examples while staying in the same general domain as the seeds. "
        )
    else:
        diversity_instruction = (
            "Keep the examples very similar in style and topic to the seeds. "
        )

    context = f"\nDomain context: {system_context}\n" if system_context else ""

    return f"""Generate {n} new high-quality instruction-output pairs for fine-tuning a language model.

Here are some seed examples that show the desired style and quality:

{seed_text}
{context}
Requirements:
- Each example must have an "instruction" field and an "output" field.
- Optionally include an "input" field if the instruction needs additional context.
- {diversity_instruction}
- Make sure outputs are detailed, accurate, and well-formatted.
- Do NOT copy the seed examples. Generate entirely new ones.
- Output ONLY a JSON array of objects.

Generate exactly {n} examples as a JSON array:"""


def build_multiturn_prompt(
    seeds: Sequence[Seed],
    n: int,
    turns: int = 3,
) -> str:
    """Build a prompt for multi-turn conversation generation."""
    seed_text = format_seed_examples(seeds)

    return f"""Generate {n} new multi-turn conversations for fine-tuning a language model.

Seed examples (showing the domain and style):

{seed_text}

Requirements:
- Each conversation must have {turns} turns (alternating user and assistant).
- The first user message should be similar in domain to the seed instructions.
- Each subsequent turn should build on the previous context.
- Assistant responses must be detailed and helpful.
- Output ONLY a JSON array of conversation objects.

Each object should have this format:
{{
  "conversations": [
    {{"from": "human", "value": "..."}},
    {{"from": "gpt", "value": "..."}},
    {{"from": "human", "value": "..."}},
    {{"from": "gpt", "value": "..."}}
  ]
}}

Generate exactly {n} conversations as a JSON array:"""


def build_quality_check_prompt(instruction: str, output: str) -> str:
    """Build a prompt to check the quality of a generated example."""
    return f"""Rate the quality of this instruction-output pair on a scale of 1-10.
Consider: clarity of instruction, accuracy of output, usefulness, and formatting.

Instruction: {instruction}
Output: {output}

Respond with ONLY a single number from 1 to 10."""
