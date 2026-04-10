"""Tests for castwright.scoring module."""

from __future__ import annotations

import pytest

from castwright.scoring import (
    QualityDimension,
    QualityRubric,
    ScoreResult,
    default_rubric,
    format_rubric_report,
)

# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------

@pytest.fixture()
def good_example() -> dict:
    return {
        "instruction": "Explain the difference between a list and a tuple in Python.",
        "output": (
            "A list is a mutable, ordered collection of elements in Python. "
            "You can add, remove, and change items after creation. A tuple, "
            "on the other hand, is immutable — once created you cannot modify "
            "its contents. Tuples are generally faster and use less memory. "
            "Use a list when you need to change the data and a tuple when the "
            "data should remain constant."
        ),
    }


@pytest.fixture()
def bad_example() -> dict:
    return {
        "instruction": "hi",
        "output": "ok",
    }


@pytest.fixture()
def rubric() -> QualityRubric:
    return default_rubric()


# ------------------------------------------------------------------
# QualityDimension
# ------------------------------------------------------------------

def test_quality_dimension_fields():
    dim = QualityDimension("coherence", 0.25, "Logical flow")
    assert dim.name == "coherence"
    assert dim.weight == 0.25
    assert dim.description == "Logical flow"


# ------------------------------------------------------------------
# ScoreResult
# ------------------------------------------------------------------

def test_score_result_fields():
    sr = ScoreResult(dimension="coherence", score=0.8, detail="ok")
    assert sr.dimension == "coherence"
    assert 0.0 <= sr.score <= 1.0
    assert sr.detail == "ok"


# ------------------------------------------------------------------
# QualityRubric — defaults
# ------------------------------------------------------------------

def test_default_rubric_has_five_dimensions(rubric: QualityRubric):
    assert len(rubric.dimensions) == 5
    names = {d.name for d in rubric.dimensions}
    assert names == {"coherence", "relevance", "diversity", "completeness", "format_compliance"}


def test_default_rubric_weights_sum_to_one(rubric: QualityRubric):
    total = sum(d.weight for d in rubric.dimensions)
    assert abs(total - 1.0) < 1e-9


# ------------------------------------------------------------------
# add_dimension
# ------------------------------------------------------------------

def test_add_dimension(rubric: QualityRubric):
    rubric.add_dimension("custom", 0.1, "A custom dimension")
    assert any(d.name == "custom" for d in rubric.dimensions)


# ------------------------------------------------------------------
# score_example
# ------------------------------------------------------------------

def test_score_example_returns_all_dimensions(rubric: QualityRubric, good_example: dict):
    results = rubric.score_example(good_example)
    assert len(results) == len(rubric.dimensions)
    for sr in results:
        assert 0.0 <= sr.score <= 1.0
        assert isinstance(sr.detail, str)


def test_score_example_good_vs_bad(rubric: QualityRubric, good_example: dict, bad_example: dict):
    good_scores = rubric.score_example(good_example)
    bad_scores = rubric.score_example(bad_example)
    good_avg = sum(s.score for s in good_scores) / len(good_scores)
    bad_avg = sum(s.score for s in bad_scores) / len(bad_scores)
    assert good_avg > bad_avg


def test_score_example_empty_output(rubric: QualityRubric):
    ex = {"instruction": "Do something.", "output": ""}
    results = rubric.score_example(ex)
    avg = sum(s.score for s in results) / len(results)
    assert avg < 0.3


# ------------------------------------------------------------------
# overall_score
# ------------------------------------------------------------------

def test_overall_score_range(rubric: QualityRubric, good_example: dict):
    score = rubric.overall_score(good_example)
    assert 0.0 <= score <= 1.0


def test_overall_score_bad_is_lower(rubric: QualityRubric, good_example: dict, bad_example: dict):
    assert rubric.overall_score(good_example) > rubric.overall_score(bad_example)


def test_overall_score_empty():
    rubric = QualityRubric(dimensions=[])
    assert rubric.overall_score({"instruction": "x", "output": "y"}) == 0.0


# ------------------------------------------------------------------
# score_dataset
# ------------------------------------------------------------------

def test_score_dataset_basic(rubric: QualityRubric, good_example: dict, bad_example: dict):
    scores = rubric.score_dataset([good_example, bad_example])
    assert set(scores.keys()) == {d.name for d in rubric.dimensions}
    for dim_scores in scores.values():
        assert "mean" in dim_scores
        assert "min" in dim_scores
        assert "max" in dim_scores
        assert dim_scores["min"] <= dim_scores["mean"] <= dim_scores["max"]


def test_score_dataset_empty(rubric: QualityRubric):
    assert rubric.score_dataset([]) == {}


def test_score_dataset_single(rubric: QualityRubric, good_example: dict):
    scores = rubric.score_dataset([good_example])
    for dim_scores in scores.values():
        assert dim_scores["min"] == dim_scores["max"] == dim_scores["mean"]


# ------------------------------------------------------------------
# format_rubric_report
# ------------------------------------------------------------------

def test_format_rubric_report(rubric: QualityRubric, good_example: dict, bad_example: dict):
    scores = rubric.score_dataset([good_example, bad_example])
    report = format_rubric_report(scores)
    assert "Quality Rubric Report" in report
    assert "coherence" in report
    assert "overall" in report


def test_format_rubric_report_empty():
    report = format_rubric_report({})
    assert "No scores" in report


# ------------------------------------------------------------------
# Scoring heuristics
# ------------------------------------------------------------------

def test_unbalanced_code_fences(rubric: QualityRubric):
    ex = {"instruction": "Show code.", "output": "```python\nprint('hi')"}
    results = {s.dimension: s for s in rubric.score_example(ex)}
    assert results["format_compliance"].score < 1.0


def test_repetitive_output(rubric: QualityRubric):
    ex = {"instruction": "Say something.", "output": "word " * 80}
    results = {s.dimension: s for s in rubric.score_example(ex)}
    assert results["diversity"].score < 0.5


def test_custom_dimension_gets_generic_scorer(rubric: QualityRubric, good_example: dict):
    rubric.add_dimension("novelty", 0.1, "How novel the response is")
    results = rubric.score_example(good_example)
    novelty = [r for r in results if r.dimension == "novelty"]
    assert len(novelty) == 1
    assert 0.0 <= novelty[0].score <= 1.0


# ------------------------------------------------------------------
# default_rubric factory
# ------------------------------------------------------------------

def test_default_rubric_factory():
    r = default_rubric()
    assert isinstance(r, QualityRubric)
    assert len(r.dimensions) == 5
