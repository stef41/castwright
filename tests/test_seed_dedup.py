"""Tests for castwright.seed_dedup module."""

from __future__ import annotations

import pytest

from castwright.seed_dedup import (
    DedupConfig,
    DedupResult,
    SeedDeduplicator,
    exact_dedup,
    format_dedup_report,
    ngram_similarity,
)


# ---------------------------------------------------------------------------
# ngram_similarity
# ---------------------------------------------------------------------------


class TestNgramSimilarity:
    def test_identical_strings(self):
        assert ngram_similarity("hello world", "hello world") == 1.0

    def test_completely_different(self):
        sim = ngram_similarity("aaa", "zzz")
        assert sim == 0.0

    def test_empty_strings(self):
        assert ngram_similarity("", "") == 1.0

    def test_one_empty(self):
        assert ngram_similarity("abc", "") == 0.0
        assert ngram_similarity("", "abc") == 0.0

    def test_partial_overlap(self):
        sim = ngram_similarity("abcdef", "abcxyz")
        assert 0.0 < sim < 1.0

    def test_custom_ngram_size(self):
        sim = ngram_similarity("hello", "hello", n=2)
        assert sim == 1.0


# ---------------------------------------------------------------------------
# DedupConfig validation
# ---------------------------------------------------------------------------


class TestDedupConfig:
    def test_defaults(self):
        cfg = DedupConfig()
        assert cfg.similarity_threshold == 0.85
        assert cfg.method == "ngram"
        assert cfg.ngram_size == 3
        assert cfg.case_sensitive is False

    def test_invalid_threshold(self):
        with pytest.raises(ValueError, match="similarity_threshold"):
            DedupConfig(similarity_threshold=1.5)

    def test_invalid_method(self):
        with pytest.raises(ValueError, match="method"):
            DedupConfig(method="cosine")

    def test_invalid_ngram_size(self):
        with pytest.raises(ValueError, match="ngram_size"):
            DedupConfig(ngram_size=0)


# ---------------------------------------------------------------------------
# DedupResult
# ---------------------------------------------------------------------------


class TestDedupResult:
    def test_removal_rate(self):
        r = DedupResult(original_count=10, deduplicated_count=7, removed_count=3)
        assert r.removal_rate == pytest.approx(0.3)

    def test_removal_rate_zero(self):
        r = DedupResult(original_count=0, deduplicated_count=0, removed_count=0)
        assert r.removal_rate == 0.0


# ---------------------------------------------------------------------------
# exact_dedup
# ---------------------------------------------------------------------------


class TestExactDedup:
    def test_no_duplicates(self):
        examples = [{"instruction": "a"}, {"instruction": "b"}, {"instruction": "c"}]
        kept, result = exact_dedup(examples)
        assert len(kept) == 3
        assert result.removed_count == 0

    def test_with_duplicates(self):
        examples = [
            {"instruction": "hello"},
            {"instruction": "world"},
            {"instruction": "hello"},
        ]
        kept, result = exact_dedup(examples)
        assert len(kept) == 2
        assert result.removed_count == 1
        assert 2 in result.removed_indices

    def test_all_duplicates(self):
        examples = [{"instruction": "same"}] * 5
        kept, result = exact_dedup(examples)
        assert len(kept) == 1
        assert result.removed_count == 4

    def test_custom_field(self):
        examples = [{"text": "a"}, {"text": "a"}]
        kept, result = exact_dedup(examples, field="text")
        assert len(kept) == 1


# ---------------------------------------------------------------------------
# SeedDeduplicator
# ---------------------------------------------------------------------------


class TestSeedDeduplicator:
    def _seeds(self):
        return [
            {"instruction": "Write a poem about the ocean"},
            {"instruction": "Explain quantum computing"},
        ]

    def test_is_duplicate_exact(self):
        dedup = SeedDeduplicator(self._seeds())
        is_dup, sim = dedup.is_duplicate(
            {"instruction": "Write a poem about the ocean"}
        )
        assert is_dup is True
        assert sim == 1.0

    def test_is_not_duplicate(self):
        dedup = SeedDeduplicator(self._seeds())
        is_dup, sim = dedup.is_duplicate(
            {"instruction": "How do I bake a chocolate cake?"}
        )
        assert is_dup is False
        assert sim < 0.85

    def test_case_insensitive(self):
        dedup = SeedDeduplicator(self._seeds())
        is_dup, _ = dedup.is_duplicate(
            {"instruction": "WRITE A POEM ABOUT THE OCEAN"}
        )
        assert is_dup is True

    def test_case_sensitive(self):
        cfg = DedupConfig(case_sensitive=True)
        dedup = SeedDeduplicator(self._seeds(), config=cfg)
        is_dup, sim = dedup.is_duplicate(
            {"instruction": "WRITE A POEM ABOUT THE OCEAN"}
        )
        # n-gram similarity of different-case strings will be low
        assert is_dup is False

    def test_deduplicate_removes_similar(self):
        dedup = SeedDeduplicator(self._seeds())
        examples = [
            {"instruction": "Write a poem about the ocean"},
            {"instruction": "Translate English to French"},
            {"instruction": "Explain quantum computing basics"},
        ]
        kept, result = dedup.deduplicate(examples)
        # First should be removed (exact match), third might be removed (near match)
        assert result.removed_count >= 1
        assert result.original_count == 3

    def test_deduplicate_all_unique(self):
        dedup = SeedDeduplicator(self._seeds())
        examples = [
            {"instruction": "Bake a cake"},
            {"instruction": "Fix a flat tire"},
        ]
        kept, result = dedup.deduplicate(examples)
        assert result.removed_count == 0
        assert len(kept) == 2

    def test_find_near_duplicates(self):
        dedup = SeedDeduplicator([], config=DedupConfig(similarity_threshold=0.7))
        examples = [
            {"instruction": "Write a poem about the ocean"},
            {"instruction": "Write a poem about the ocean waves"},
            {"instruction": "Explain quantum computing"},
        ]
        pairs = dedup.find_near_duplicates(examples)
        # First two should be near duplicates
        assert any(i == 0 and j == 1 for i, j, _ in pairs)

    def test_add_seeds(self):
        dedup = SeedDeduplicator([])
        assert len(dedup.seeds) == 0
        dedup.add_seeds([{"instruction": "new seed"}])
        assert len(dedup.seeds) == 1

    def test_exact_method(self):
        cfg = DedupConfig(method="exact")
        dedup = SeedDeduplicator(self._seeds(), config=cfg)
        is_dup, sim = dedup.is_duplicate(
            {"instruction": "write a poem about the ocean"}
        )
        # Case-insensitive + exact → matches
        assert is_dup is True

    def test_custom_threshold(self):
        cfg = DedupConfig(similarity_threshold=0.99)
        dedup = SeedDeduplicator(self._seeds(), config=cfg)
        is_dup, _ = dedup.is_duplicate(
            {"instruction": "Write a poem about the sea"}
        )
        # With very high threshold, near-match won't trigger
        assert is_dup is False


# ---------------------------------------------------------------------------
# format_dedup_report
# ---------------------------------------------------------------------------


class TestFormatDedupReport:
    def test_basic_report(self):
        result = DedupResult(
            original_count=10,
            deduplicated_count=8,
            removed_count=2,
            removed_indices=[3, 7],
            similarity_scores=[0.0, 0.0, 0.0, 0.92, 0.0, 0.0, 0.0, 0.88, 0.0, 0.0],
        )
        report = format_dedup_report(result)
        assert "Seed Deduplication Report" in report
        assert "Removed           : 2" in report
        assert "20.0%" in report

    def test_empty_report(self):
        result = DedupResult(original_count=0, deduplicated_count=0, removed_count=0)
        report = format_dedup_report(result)
        assert "0" in report
