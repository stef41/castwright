"""Deduplication against seed examples.

Detects and removes generated examples that are too similar to the
original seed set, preventing the model from parroting back training data.
All comparisons are CPU-only using character n-gram Jaccard similarity.
No external dependencies required.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field


@dataclass
class DedupConfig:
    """Configuration for seed deduplication."""

    similarity_threshold: float = 0.85
    method: str = "ngram"
    ngram_size: int = 3
    case_sensitive: bool = False

    def __post_init__(self) -> None:
        if not 0.0 <= self.similarity_threshold <= 1.0:
            raise ValueError("similarity_threshold must be between 0.0 and 1.0")
        if self.method not in ("ngram", "exact"):
            raise ValueError(f"method must be 'ngram' or 'exact', got {self.method!r}")
        if self.ngram_size < 1:
            raise ValueError("ngram_size must be >= 1")


@dataclass
class DedupResult:
    """Result of a deduplication pass."""

    original_count: int
    deduplicated_count: int
    removed_count: int
    removed_indices: list[int] = field(default_factory=list)
    similarity_scores: list[float] = field(default_factory=list)

    @property
    def removal_rate(self) -> float:
        if self.original_count == 0:
            return 0.0
        return self.removed_count / self.original_count


# ---------------------------------------------------------------------------
# Similarity helpers
# ---------------------------------------------------------------------------


def _char_ngrams(text: str, n: int) -> set[str]:
    """Return the set of character n-grams for *text*."""
    if len(text) < n:
        return {text} if text else set()
    return {text[i : i + n] for i in range(len(text) - n + 1)}


def ngram_similarity(text_a: str, text_b: str, n: int = 3) -> float:
    """Character n-gram Jaccard similarity between two strings.

    Returns a float in [0, 1].
    """
    if not text_a and not text_b:
        return 1.0
    if not text_a or not text_b:
        return 0.0
    grams_a = _char_ngrams(text_a, n)
    grams_b = _char_ngrams(text_b, n)
    intersection = len(grams_a & grams_b)
    union = len(grams_a | grams_b)
    if union == 0:
        return 1.0
    return intersection / union


# ---------------------------------------------------------------------------
# Exact dedup (fast path)
# ---------------------------------------------------------------------------


def exact_dedup(
    examples: Sequence[dict],
    field: str = "instruction",
) -> tuple[list[dict], DedupResult]:
    """Remove exact-duplicate examples based on *field*.

    Returns (deduplicated_list, result).
    """
    seen: set[str] = set()
    kept: list[dict] = []
    removed_indices: list[int] = []
    scores: list[float] = []

    for idx, ex in enumerate(examples):
        text = ex.get(field, "")
        if text in seen:
            removed_indices.append(idx)
            scores.append(1.0)
        else:
            seen.add(text)
            kept.append(ex)
            scores.append(0.0)

    return kept, DedupResult(
        original_count=len(examples),
        deduplicated_count=len(kept),
        removed_count=len(removed_indices),
        removed_indices=removed_indices,
        similarity_scores=scores,
    )


# ---------------------------------------------------------------------------
# Seed-aware deduplicator
# ---------------------------------------------------------------------------


class SeedDeduplicator:
    """Compare generated examples against a known seed set.

    Parameters
    ----------
    seed_examples:
        The original seed examples to compare against.
    config:
        Optional :class:`DedupConfig` (defaults are used when *None*).
    """

    def __init__(
        self,
        seed_examples: list[dict],
        config: DedupConfig | None = None,
    ) -> None:
        self.config = config or DedupConfig()
        self._seeds: list[dict] = list(seed_examples)

    # -- public API ---------------------------------------------------------

    def add_seeds(self, examples: list[dict]) -> None:
        """Add more seed examples to the comparison set."""
        self._seeds.extend(examples)

    @property
    def seeds(self) -> list[dict]:
        return list(self._seeds)

    def is_duplicate(
        self,
        example: dict,
        field: str = "instruction",
    ) -> tuple[bool, float]:
        """Check whether *example* is too similar to any seed.

        Returns ``(is_dup, max_similarity)``.
        """
        text = example.get(field, "")
        text = self._normalise(text)
        max_sim = 0.0

        for seed in self._seeds:
            seed_text = self._normalise(seed.get(field, ""))
            sim = self._compare(text, seed_text)
            if sim > max_sim:
                max_sim = sim
            if max_sim >= self.config.similarity_threshold:
                return True, max_sim

        return max_sim >= self.config.similarity_threshold, max_sim

    def deduplicate(
        self,
        examples: list[dict],
        field: str = "instruction",
    ) -> tuple[list[dict], DedupResult]:
        """Remove examples that are too similar to any seed.

        Returns ``(kept, result)``.
        """
        kept: list[dict] = []
        removed_indices: list[int] = []
        scores: list[float] = []

        for idx, ex in enumerate(examples):
            is_dup, sim = self.is_duplicate(ex, field=field)
            scores.append(sim)
            if is_dup:
                removed_indices.append(idx)
            else:
                kept.append(ex)

        return kept, DedupResult(
            original_count=len(examples),
            deduplicated_count=len(kept),
            removed_count=len(removed_indices),
            removed_indices=removed_indices,
            similarity_scores=scores,
        )

    def find_near_duplicates(
        self,
        examples: list[dict],
        field: str = "instruction",
    ) -> list[tuple[int, int, float]]:
        """Find near-duplicate pairs *within* *examples*.

        Returns list of ``(i, j, similarity)`` tuples whose similarity
        meets or exceeds the threshold.
        """
        pairs: list[tuple[int, int, float]] = []
        n = len(examples)
        for i in range(n):
            text_i = self._normalise(examples[i].get(field, ""))
            for j in range(i + 1, n):
                text_j = self._normalise(examples[j].get(field, ""))
                sim = self._compare(text_i, text_j)
                if sim >= self.config.similarity_threshold:
                    pairs.append((i, j, sim))
        return pairs

    # -- internals ----------------------------------------------------------

    def _normalise(self, text: str) -> str:
        if not self.config.case_sensitive:
            return text.lower()
        return text

    def _compare(self, a: str, b: str) -> float:
        if self.config.method == "exact":
            return 1.0 if a == b else 0.0
        return ngram_similarity(a, b, n=self.config.ngram_size)


# ---------------------------------------------------------------------------
# Report formatting
# ---------------------------------------------------------------------------


def format_dedup_report(result: DedupResult) -> str:
    """Return a human-readable deduplication summary."""
    lines = [
        "Seed Deduplication Report",
        "=" * 40,
        f"Original examples : {result.original_count}",
        f"After dedup       : {result.deduplicated_count}",
        f"Removed           : {result.removed_count}",
        f"Removal rate      : {result.removal_rate:.1%}",
    ]
    if result.removed_indices:
        lines.append(f"Removed indices   : {result.removed_indices}")
    if result.similarity_scores:
        max_s = max(result.similarity_scores)
        avg_s = sum(result.similarity_scores) / len(result.similarity_scores)
        lines.append(f"Max similarity    : {max_s:.4f}")
        lines.append(f"Mean similarity   : {avg_s:.4f}")
    return "\n".join(lines)
