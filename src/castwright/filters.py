"""Quality filtering for generated examples.

Applies fast heuristic checks to filter out low-quality generations.
All checks are CPU-only and deterministic.
"""

from __future__ import annotations

import re
from typing import Callable, List, Optional, Sequence

from castwright._types import GeneratedExample


def _check_not_empty(ex: GeneratedExample) -> bool:
    """Instruction and output must both be non-empty."""
    return bool(ex.instruction.strip()) and bool(ex.output.strip())


def _check_min_length(ex: GeneratedExample, min_chars: int = 10) -> bool:
    """Instruction must be at least min_chars."""
    return len(ex.instruction.strip()) >= min_chars


def _check_not_repetitive(ex: GeneratedExample) -> bool:
    """Output should not be excessively repetitive."""
    words = ex.output.lower().split()
    if len(words) < 5:
        return True
    # Check consecutive repeats
    consecutive = 0
    for i in range(1, len(words)):
        if words[i] == words[i - 1]:
            consecutive += 1
    return consecutive / len(words) < 0.3


def _check_not_refusal(ex: GeneratedExample) -> bool:
    """Filter out raw refusals (unless that's what you want)."""
    refusal_patterns = [
        r"^I('m| am) (sorry|unable|not able)",
        r"^(Sorry|Unfortunately),? I (can't|cannot|am not able)",
        r"^As an AI",
    ]
    for pattern in refusal_patterns:
        if re.match(pattern, ex.output.strip(), re.IGNORECASE):
            return False
    return True


def _check_no_meta_talk(ex: GeneratedExample) -> bool:
    """Filter out examples where the model talks about generating data."""
    meta_patterns = [
        r"here('s| is) (an |another )?(example|instruction|training)",
        r"as requested",
        r"I('ll| will) generate",
        r"here are \d+ (more )?examples",
    ]
    full_text = f"{ex.instruction} {ex.output}".lower()
    for pattern in meta_patterns:
        if re.search(pattern, full_text, re.IGNORECASE):
            return False
    return True


def _check_balanced_formatting(ex: GeneratedExample) -> bool:
    """Check for balanced code blocks and quotes."""
    text = ex.output
    if text.count("```") % 2 != 0:
        return False
    return True


# Default filter chain
DEFAULT_FILTERS: list[Callable[[GeneratedExample], bool]] = [
    _check_not_empty,
    _check_min_length,
    _check_not_repetitive,
    _check_not_refusal,
    _check_no_meta_talk,
    _check_balanced_formatting,
]


def filter_examples(
    examples: Sequence[GeneratedExample],
    filters: Optional[List[Callable[[GeneratedExample], bool]]] = None,
) -> List[GeneratedExample]:
    """Apply quality filters to generated examples.

    Parameters
    ----------
    examples:
        The generated examples to filter.
    filters:
        Custom filter functions. Each takes a GeneratedExample and returns
        True to keep, False to discard. If None, uses DEFAULT_FILTERS.

    Returns
    -------
    list
        Examples that passed all filters.
    """
    if filters is None:
        filters = DEFAULT_FILTERS

    result: list[GeneratedExample] = []
    for ex in examples:
        if all(f(ex) for f in filters):
            result.append(ex)
    return result


def deduplicate_generated(
    examples: Sequence[GeneratedExample],
    existing_instructions: Optional[set[str]] = None,
) -> List[GeneratedExample]:
    """Remove duplicates from generated examples.

    Also removes examples whose instruction matches anything in
    ``existing_instructions`` (e.g., the seed examples).
    """
    seen: set[str] = set()
    if existing_instructions:
        seen.update(s.lower().strip() for s in existing_instructions)

    result: list[GeneratedExample] = []
    for ex in examples:
        key = ex.instruction.lower().strip()
        if key not in seen:
            seen.add(key)
            result.append(ex)
    return result
