"""Quality scoring rubric for generated instruction-tuning data.

Scores examples against configurable quality dimensions using
fast CPU-only heuristics (no external dependencies).
"""

from __future__ import annotations

import math
import re
import string
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence


@dataclass
class QualityDimension:
    """A single dimension in a quality rubric."""

    name: str
    weight: float
    description: str


@dataclass
class ScoreResult:
    """Score for one dimension on one example."""

    dimension: str
    score: float  # 0-1
    detail: str


class QualityRubric:
    """Configurable rubric that scores instruction-tuning examples.

    Each *dimension* has a ``weight`` and a scoring function.  Built-in
    dimensions use fast heuristics; you can also register custom scorers.
    """

    def __init__(
        self,
        dimensions: Optional[List[QualityDimension]] = None,
    ) -> None:
        if dimensions is not None:
            self.dimensions = list(dimensions)
        else:
            self.dimensions = [
                QualityDimension("coherence", 0.25, "Logical flow and grammatical correctness"),
                QualityDimension("relevance", 0.25, "Output addresses the instruction"),
                QualityDimension("diversity", 0.15, "Vocabulary richness and varied expression"),
                QualityDimension("completeness", 0.20, "Presence of expected fields and sufficient detail"),
                QualityDimension("format_compliance", 0.15, "Proper formatting (balanced delimiters, etc.)"),
            ]
        self._scorers: Dict[str, Any] = {
            "coherence": _score_coherence,
            "relevance": _score_relevance,
            "diversity": _score_diversity,
            "completeness": _score_completeness,
            "format_compliance": _score_format_compliance,
        }

    # ------------------------------------------------------------------
    def add_dimension(
        self,
        name: str,
        weight: float,
        description: str,
    ) -> None:
        """Add a new quality dimension to the rubric."""
        self.dimensions.append(QualityDimension(name, weight, description))

    # ------------------------------------------------------------------
    def score_example(self, example: dict) -> List[ScoreResult]:
        """Score a single example against every dimension."""
        results: List[ScoreResult] = []
        for dim in self.dimensions:
            scorer = self._scorers.get(dim.name, _score_generic)
            score, detail = scorer(example)
            results.append(ScoreResult(dimension=dim.name, score=_clamp(score), detail=detail))
        return results

    # ------------------------------------------------------------------
    def score_dataset(self, examples: List[dict]) -> Dict[str, Dict[str, float]]:
        """Aggregate per-dimension scores across a dataset.

        Returns ``{dimension: {"mean": …, "min": …, "max": …}}``.
        """
        if not examples:
            return {}

        accum: Dict[str, List[float]] = {d.name: [] for d in self.dimensions}
        for ex in examples:
            for sr in self.score_example(ex):
                accum[sr.dimension].append(sr.score)

        result: Dict[str, Dict[str, float]] = {}
        for dim_name, scores in accum.items():
            result[dim_name] = {
                "mean": sum(scores) / len(scores),
                "min": min(scores),
                "max": max(scores),
            }
        return result

    # ------------------------------------------------------------------
    def overall_score(self, example: dict) -> float:
        """Return a single weighted 0-1 score for *example*."""
        results = self.score_example(example)
        weight_map = {d.name: d.weight for d in self.dimensions}
        total_weight = sum(weight_map.get(r.dimension, 0.0) for r in results)
        if total_weight == 0:
            return 0.0
        weighted = sum(r.score * weight_map.get(r.dimension, 0.0) for r in results)
        return _clamp(weighted / total_weight)


# ======================================================================
# Factory
# ======================================================================

def default_rubric() -> QualityRubric:
    """Create a ``QualityRubric`` with sensible defaults for instruction-response data."""
    return QualityRubric()


# ======================================================================
# Report formatting
# ======================================================================

def format_rubric_report(scores: Dict[str, Dict[str, float]]) -> str:
    """Format dataset-level scores into a human-readable text report."""
    if not scores:
        return "No scores to report."
    lines = ["Quality Rubric Report", "=" * 40]
    for dim, stats in scores.items():
        lines.append(
            f"  {dim:<22s}  mean={stats['mean']:.3f}  min={stats['min']:.3f}  max={stats['max']:.3f}"
        )
    overall_mean = sum(s["mean"] for s in scores.values()) / len(scores)
    lines.append("-" * 40)
    lines.append(f"  {'overall':<22s}  mean={overall_mean:.3f}")
    return "\n".join(lines)


# ======================================================================
# Internal scoring heuristics
# ======================================================================

def _clamp(v: float) -> float:
    return max(0.0, min(1.0, v))


def _word_count(text: str) -> int:
    return len(text.split())


def _sentence_count(text: str) -> int:
    return max(1, len(re.split(r"[.!?]+", text.strip())) - 1) if text.strip() else 0


def _unique_word_ratio(text: str) -> float:
    words = [w.lower().strip(string.punctuation) for w in text.split() if w.strip(string.punctuation)]
    if not words:
        return 0.0
    return len(set(words)) / len(words)


def _get_text(example: dict) -> str:
    """Concatenate instruction + output (+ input if present)."""
    parts = [
        example.get("instruction", ""),
        example.get("input", ""),
        example.get("output", ""),
    ]
    return " ".join(p for p in parts if p)


# ------------------------------------------------------------------

def _score_coherence(example: dict) -> tuple[float, str]:
    """Heuristic: sentence count, avg sentence length, no excessive repetition."""
    output = example.get("output", "")
    if not output.strip():
        return 0.0, "empty output"

    sentences = _sentence_count(output)
    words = _word_count(output)
    avg_sent_len = words / max(sentences, 1)

    # Penalise very short or very long average sentences
    len_score = 1.0
    if avg_sent_len < 3:
        len_score = 0.4
    elif avg_sent_len > 60:
        len_score = 0.5

    # Penalise consecutive duplicate words
    word_list = output.lower().split()
    consec = sum(1 for i in range(1, len(word_list)) if word_list[i] == word_list[i - 1])
    rep_ratio = consec / max(len(word_list), 1)
    rep_score = max(0.0, 1.0 - rep_ratio * 3)

    score = 0.5 * len_score + 0.5 * rep_score
    detail = f"sentences={sentences} avg_len={avg_sent_len:.1f} rep_ratio={rep_ratio:.2f}"
    return score, detail


def _score_relevance(example: dict) -> tuple[float, str]:
    """Heuristic: word overlap between instruction and output."""
    instruction = example.get("instruction", "").lower()
    output = example.get("output", "").lower()
    if not instruction.strip() or not output.strip():
        return 0.0, "missing instruction or output"

    instr_words = set(instruction.split()) - _STOPWORDS
    out_words = set(output.split()) - _STOPWORDS
    if not instr_words:
        return 0.5, "instruction has only stopwords"
    overlap = len(instr_words & out_words) / len(instr_words)
    score = min(1.0, overlap * 2)  # 50 % overlap → 1.0
    detail = f"keyword_overlap={overlap:.2f}"
    return score, detail


def _score_diversity(example: dict) -> tuple[float, str]:
    """Heuristic: unique-word ratio in the output."""
    output = example.get("output", "")
    if not output.strip():
        return 0.0, "empty output"
    ratio = _unique_word_ratio(output)
    # ratio ≥ 0.7 → 1.0; ratio ≤ 0.3 → 0.0
    score = _clamp((ratio - 0.3) / 0.4)
    detail = f"unique_word_ratio={ratio:.2f}"
    return score, detail


def _score_completeness(example: dict) -> tuple[float, str]:
    """Heuristic: expected fields present and non-trivially filled."""
    checks = 0
    passed = 0

    # instruction present & non-trivial
    checks += 1
    instr = example.get("instruction", "")
    if instr.strip() and _word_count(instr) >= 3:
        passed += 1

    # output present & non-trivial
    checks += 1
    out = example.get("output", "")
    if out.strip() and _word_count(out) >= 5:
        passed += 1

    # output has reasonable length (≥ 20 words)
    checks += 1
    if _word_count(out) >= 20:
        passed += 1

    # instruction ends with punctuation or question mark
    checks += 1
    if instr.strip() and instr.strip()[-1] in ".?!:":
        passed += 1

    score = passed / checks if checks else 0.0
    detail = f"passed={passed}/{checks}"
    return score, detail


def _score_format_compliance(example: dict) -> tuple[float, str]:
    """Heuristic: balanced delimiters, no junk artefacts."""
    output = example.get("output", "")
    penalties = 0
    reasons: list[str] = []

    # Balanced code fences
    if output.count("```") % 2 != 0:
        penalties += 1
        reasons.append("unbalanced_code_fences")

    # Balanced parentheses / brackets
    for open_c, close_c, name in [("(", ")", "parens"), ("[", "]", "brackets"), ("{", "}", "braces")]:
        if output.count(open_c) != output.count(close_c):
            penalties += 1
            reasons.append(f"unbalanced_{name}")

    # No null bytes or weird control chars
    if re.search(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", output):
        penalties += 1
        reasons.append("control_chars")

    max_penalties = 5
    score = max(0.0, 1.0 - penalties / max_penalties)
    detail = ", ".join(reasons) if reasons else "ok"
    return score, detail


def _score_generic(example: dict) -> tuple[float, str]:
    """Fallback scorer for custom dimensions without a registered scorer."""
    text = _get_text(example)
    if not text.strip():
        return 0.0, "empty"
    wc = _word_count(text)
    score = _clamp(wc / 50)
    return score, f"word_count={wc}"


# Minimal English stopwords for relevance overlap
_STOPWORDS = frozenset(
    "a an the is are was were be been being have has had do does did will would "
    "shall should may might can could to of in for on with at by from as into "
    "through during before after above below between out off over under again "
    "further then once here there when where why how all each every both few "
    "more most other some such no nor not only own same so than too very and "
    "but or if it its i me my we our you your he him his she her they them their "
    "this that these those what which who whom".split()
)
