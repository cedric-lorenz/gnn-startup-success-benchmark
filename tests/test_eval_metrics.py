"""
Tests for eval.py metric implementations.

These tests import and call the real Evaluator methods.
Each test is designed to FAIL if the implementation has bugs.
"""
import pytest
import torch
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# Shared fixture to create lightweight Evaluator without loading data
@pytest.fixture
def evaluator(monkeypatch):
    """Create a minimal Evaluator for testing without loading downstream data."""
    import src.ml.eval as eval_module

    # Disable downstream analysis by patching the default config
    mock_config = {"analysis": {"enable_downstream_analysis": False}}
    monkeypatch.setattr(eval_module, "_default_config", mock_config)

    from src.ml.eval import Evaluator

    class DummyModel:
        def eval(self): pass
        def state_dict(self): return {}

    return Evaluator(
        model=DummyModel(),
        device='cpu',
        use_wandb=False,
        optimization_metric_type='auc_roc',
        min_amount_of_epochs=0
    )


class TestPrecisionAtK:
    """Test the Evaluator.precision_at_k() method."""

    def test_precision_at_k_perfect_ranking(self, evaluator):
        """
        Perfect ranking: top-K most confident predictions are all correct.
        Should give P@K = 1.0
        """
        # Setup: 5 positives at top, 5 negatives at bottom
        y_true = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0])
        # Probs: positives have higher probs
        y_prob = np.array([0.95, 0.90, 0.85, 0.80, 0.75, 0.25, 0.20, 0.15, 0.10, 0.05])

        result = evaluator.precision_at_k(y_true, y_prob, k=5)

        # Top 5 by confidence should all be correct (probs > 0.5, predictions=1, true=1)
        assert result == 1.0, f"Perfect ranking should give P@5=1.0, got {result}"

    def test_precision_at_k_worst_ranking(self, evaluator):
        """
        Worst ranking: top-K most confident predictions are all wrong.
        """
        # Setup: negatives have high confidence, positives have low
        y_true = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
        y_prob = np.array([0.95, 0.90, 0.85, 0.80, 0.75, 0.25, 0.20, 0.15, 0.10, 0.05])

        result = evaluator.precision_at_k(y_true, y_prob, k=5)

        # Top 5 predict positive (prob>0.5) but true labels are 0
        # So correct = 0, P@5 = 0
        assert result == 0.0, f"Worst ranking should give P@5=0.0, got {result}"

    def test_precision_at_k_random_baseline(self, evaluator):
        """
        Random predictions should give P@K close to base rate.
        """
        np.random.seed(42)
        n = 1000
        y_true = np.random.randint(0, 2, n)
        y_prob = np.random.rand(n)

        base_rate = y_true.mean()
        result = evaluator.precision_at_k(y_true, y_prob, k=100)

        # Should be within reasonable range of base rate
        assert 0.2 < result < 0.8, f"Random P@100 should be near base rate, got {result}"

    def test_precision_at_k_k_larger_than_n(self, evaluator):
        """
        When K > N, should use N samples.
        """
        y_true = np.array([1, 0, 1])
        y_prob = np.array([0.9, 0.8, 0.7])

        result = evaluator.precision_at_k(y_true, y_prob, k=100)

        # Should not crash, should use all 3 samples
        assert 0 <= result <= 1, f"K>N should not crash, got {result}"


class TestRecallAtK:
    """Test the Evaluator.recall_at_k() method."""

    def test_recall_at_k_finds_all_positives(self, evaluator):
        """
        If top-K contains all positives, R@K should be 1.0.
        """
        # 3 positives, all ranked at top
        y_true = np.array([1, 1, 1, 0, 0, 0, 0, 0, 0, 0])
        y_prob = np.array([0.95, 0.90, 0.85, 0.40, 0.35, 0.30, 0.25, 0.20, 0.15, 0.10])

        result = evaluator.recall_at_k(y_true, y_prob, k=5)

        # Top 5 should contain all 3 positives
        # R@K = found_positives / total_positives = 3/3 = 1.0
        assert result == 1.0, f"Should find all positives, got R@5={result}"

    def test_recall_at_k_misses_positives(self, evaluator):
        """
        If positives have very low probability and model confidently predicts them as negative,
        they still appear in top-k by confidence since confidence = 1 - prob for negative predictions.

        This tests a scenario where ALL positives have low confidence (near 0.5 threshold).
        """
        # 5 positives with probs near 0.5 (low confidence), 5 negatives with very low probs (high confidence)
        y_true = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
        # Negatives: prob=0.01 -> pred=0 -> conf=0.99 (very confident)
        # Positives: prob=0.49 -> pred=0 -> conf=0.51 (low confidence)
        y_prob = np.array([0.01, 0.02, 0.03, 0.04, 0.05, 0.49, 0.48, 0.47, 0.46, 0.45])

        result = evaluator.recall_at_k(y_true, y_prob, k=5)

        # Top 5 by confidence are indices 0-4 (conf=0.99,0.98,0.97,0.96,0.95), all y_true=0
        # found_positives = 0, total_positives = 5
        # R@5 = 0/5 = 0
        assert result == 0.0, f"Should miss positives, got R@5={result}"

    def test_recall_at_k_no_positives(self, evaluator):
        """
        If no positives exist, R@K should be 0.
        """
        y_true = np.zeros(10)
        y_prob = np.random.rand(10)

        result = evaluator.recall_at_k(y_true, y_prob, k=5)

        assert result == 0.0, f"No positives should give R@K=0, got {result}"


class TestPrecisionAtKPositivePredicted:
    """Test the Evaluator.precision_at_k_positive_predicted() method."""

    def test_pos_pred_precision_perfect(self, evaluator):
        """
        Top-K by positive probability, all truly positive = P@K=1.0
        """
        y_true = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0])
        y_prob = np.array([0.99, 0.95, 0.90, 0.85, 0.80, 0.20, 0.15, 0.10, 0.05, 0.01])

        result = evaluator.precision_at_k_positive_predicted(y_true, y_prob, k=5)

        # Top 5 by positive prob are indices 0-4, all have y_true=1
        assert result == 1.0, f"Perfect should give 1.0, got {result}"

    def test_pos_pred_precision_worst(self, evaluator):
        """
        Top-K by positive probability, all truly negative = P@K=0.0
        """
        y_true = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
        y_prob = np.array([0.99, 0.95, 0.90, 0.85, 0.80, 0.20, 0.15, 0.10, 0.05, 0.01])

        result = evaluator.precision_at_k_positive_predicted(y_true, y_prob, k=5)

        # Top 5 by positive prob are indices 0-4, all have y_true=0
        assert result == 0.0, f"Worst should give 0.0, got {result}"

    def test_pos_pred_precision_partial(self, evaluator):
        """
        Mixed scenario: some correct, some wrong.
        """
        y_true = np.array([1, 0, 1, 0, 1, 0, 0, 0, 0, 0])
        y_prob = np.array([0.99, 0.95, 0.90, 0.85, 0.80, 0.20, 0.15, 0.10, 0.05, 0.01])

        result = evaluator.precision_at_k_positive_predicted(y_true, y_prob, k=5)

        # Top 5 by prob: indices 0,1,2,3,4 -> y_true = [1,0,1,0,1] -> 3 correct
        # P@5 = 3/5 = 0.6
        assert result == 0.6, f"Expected 0.6, got {result}"


class TestMetricConsistencyWithKnownValues:
    """
    Tests with hand-calculated expected values.
    These tests will FAIL if the implementation differs from expected behavior.
    """

    def test_known_precision_at_k_value(self, evaluator):
        """
        Manually verified scenario with known expected value.

        y_true = [1, 1, 0, 0, 1]
        y_prob = [0.9, 0.7, 0.8, 0.3, 0.6]

        Predictions (threshold 0.5): [1, 1, 1, 0, 1]
        Confidences: [0.9, 0.7, 0.8, 0.7, 0.6]

        Top-3 by confidence: indices [0, 2, 1] (conf: 0.9, 0.8, 0.7)
        Predictions at those: [1, 1, 1]
        True labels at those: [1, 0, 1]
        Correct: pred==true -> [1==1, 1==0, 1==1] -> [T, F, T] -> 2 correct

        P@3 = 2/3 = 0.6667
        """
        y_true = np.array([1, 1, 0, 0, 1])
        y_prob = np.array([0.9, 0.7, 0.8, 0.3, 0.6])

        result = evaluator.precision_at_k(y_true, y_prob, k=3)

        expected = 2/3
        assert abs(result - expected) < 0.01, f"Expected {expected:.4f}, got {result:.4f}"

    def test_known_recall_at_k_positive_predicted_value(self, evaluator):
        """
        Manually verified scenario.

        y_true = [1, 0, 1, 0, 1, 0, 0, 0]
        y_prob = [0.9, 0.85, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2]

        Total positives: 3 (indices 0, 2, 4)
        Top-3 by positive prob: indices [0, 1, 2]
        True labels at those: [1, 0, 1] -> 2 positives found

        R@3 = 2/3 = 0.6667
        """
        y_true = np.array([1, 0, 1, 0, 1, 0, 0, 0])
        y_prob = np.array([0.9, 0.85, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2])

        result = evaluator.recall_at_k_positive_predicted(y_true, y_prob, k=3)

        expected = 2/3
        assert abs(result - expected) < 0.01, f"Expected {expected:.4f}, got {result:.4f}"
