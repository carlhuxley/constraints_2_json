"""Tests for scripts/confidence_calibration.py (pure functions; no model)."""
import pytest

from scripts import confidence_calibration as cc


class TestAuroc:

    def test_perfect_separation(self):
        assert cc.auroc([0.9, 0.95, 0.3, 0.4], [True, True, False, False]) == 1.0

    def test_no_signal_when_wrong_answers_score_higher(self):
        assert cc.auroc([0.3, 0.9], [True, False]) == 0.0

    def test_ties_count_half(self):
        assert cc.auroc([0.8, 0.8], [True, False]) == 0.5

    def test_undefined_without_both_classes(self):
        assert cc.auroc([0.9, 0.8], [True, True]) is None


class TestThresholdTable:

    def test_counts_caught_and_needless_fallbacks(self):
        confidences = [0.99, 0.6, 0.4, 0.95]
        correct = [True, True, False, False]

        (row,) = cc.threshold_table(confidences, correct, [0.7])

        assert row["fallback_rate"] == 0.5          # 0.6 and 0.4 flagged
        assert row["errors_caught"] == 1           # 0.4; the 0.95 error slips through
        assert row["errors_caught_rate"] == 0.5
        assert row["correct_sent_to_llm"] == 1     # 0.6
        assert row["accuracy_ceiling"] == 0.75


class TestSplitMatchesTraining:

    def test_split_is_seeded_and_disjoint(self):
        from src.train_adapter_simple import split_examples

        examples = [{"input": str(i), "output": {}} for i in range(100)]
        train_a, val_a = split_examples(examples)
        train_b, val_b = split_examples(examples)

        assert val_a == val_b
        assert len(train_a) == 90 and len(val_a) == 10
        assert not {e["input"] for e in train_a} & {e["input"] for e in val_a}
        assert [e["input"] for e in examples] == [str(i) for i in range(100)]  # not mutated
