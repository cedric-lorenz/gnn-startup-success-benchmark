"""Tests for Evaluator from src/ml/eval.py — core metric computation and model selection."""
import pytest
import numpy as np
import torch
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.ml.eval import Evaluator


@pytest.fixture
def evaluator():
    """Create an Evaluator with a dummy model."""
    model = MagicMock()
    model.state_dict.return_value = {"dummy": torch.tensor([1.0])}
    with patch("src.ml.eval.DownstreamAnalyzer", return_value=None):
        ev = Evaluator(
            model=model,
            device="cpu",
            use_wandb=False,
            optimization_metric_type="auc_pr",
            min_amount_of_epochs=0,
            config={"analysis": {"enable_downstream_analysis": False}},
        )
    return ev


class TestComputeMetrics:
    """Test the real _compute_metrics method with synthetic predictions."""

    @pytest.fixture
    def large_binary_data(self):
        """Generate 1500 samples (> max k=1000 used in _compute_metrics)."""
        np.random.seed(42)
        n = 1500
        y_true = np.zeros(n, dtype=int)
        y_true[:500] = 1  # 33% positive
        y_score = np.random.rand(n)
        # Make scores correlate with labels for non-random behavior
        y_score[y_true == 1] += 0.3
        y_score = np.clip(y_score, 0, 1)
        y_pred = (y_score > 0.5).astype(int)
        return y_true, y_pred, y_score

    def test_returns_scalar_and_dict(self, evaluator, large_binary_data):
        """_compute_metrics should return (scalar, dict) tuple."""
        y_true, y_pred, y_score = large_binary_data
        result = evaluator._compute_metrics(
            y_true, y_pred, y_score, y_probs=y_score,
            mode="test", suffix="mom",
        )
        assert isinstance(result, tuple)
        assert len(result) == 2
        scalar, metrics = result
        assert isinstance(scalar, float)
        assert isinstance(metrics, dict)

    def test_perfect_predictions_high_auc(self, evaluator):
        """Perfect predictions should yield AUC close to 1.0."""
        n = 1500
        y_true = np.array([0] * 750 + [1] * 750)
        y_score = np.array([0.1] * 750 + [0.9] * 750)
        y_pred = np.array([0] * 750 + [1] * 750)

        scalar, metrics = evaluator._compute_metrics(
            y_true, y_pred, y_score, y_probs=y_score,
            mode="test", suffix="test",
        )
        assert metrics["test_auc_roc_test"] == pytest.approx(1.0)
        assert metrics["test_auc_pr_test"] == pytest.approx(1.0)

    def test_metrics_dict_has_expected_keys(self, evaluator, large_binary_data):
        """Metrics dict should contain AUC-ROC, AUC-PR, F1, precision, recall."""
        y_true, y_pred, y_score = large_binary_data
        _, metrics = evaluator._compute_metrics(
            y_true, y_pred, y_score, y_probs=y_score,
            mode="val", suffix="s",
        )
        assert "val_auc_roc_s" in metrics
        assert "val_auc_pr_s" in metrics
        assert "val_f1_s" in metrics
        assert "val_precision_s" in metrics
        assert "val_recall_s" in metrics

    def test_random_predictions_low_auc(self, evaluator):
        """Random predictions should give AUC around 0.5."""
        np.random.seed(42)
        n = 1500
        y_true = np.random.randint(0, 2, n)
        y_score = np.random.rand(n)
        y_pred = (y_score > 0.5).astype(int)

        _, metrics = evaluator._compute_metrics(
            y_true, y_pred, y_score, y_probs=y_score,
            mode="test", suffix="r",
        )
        assert 0.35 < metrics["test_auc_roc_r"] < 0.65


class TestUpdateBestModel:
    """Test best model tracking logic."""

    def test_updates_on_improvement(self, evaluator):
        evaluator.current_epoch = 5
        evaluator.update_best_model(0.8)
        assert evaluator.best_metric == 0.8
        assert evaluator.best_model_dict is not None

    def test_does_not_update_before_min_epochs(self, evaluator):
        evaluator.min_amount_of_epochs = 10
        evaluator.current_epoch = 5
        evaluator.update_best_model(0.9)
        assert evaluator.best_metric == -1  # unchanged

    def test_does_not_downgrade(self, evaluator):
        evaluator.current_epoch = 5
        evaluator.update_best_model(0.8)
        evaluator.update_best_model(0.6)
        assert evaluator.best_metric == 0.8  # kept the better one

    def test_updates_on_equal_metric(self, evaluator):
        """Should update when metric equals best (>= comparison)."""
        evaluator.current_epoch = 5
        evaluator.update_best_model(0.8)
        evaluator.update_best_model(0.8)
        assert evaluator.best_metric == 0.8


class TestRobustThreshold:
    """Test robust threshold fallback logic."""

    def test_uses_absolute_threshold_when_predictions_exist(self, evaluator):
        evaluator.optimal_threshold = 0.5
        y_prob = np.array([0.1, 0.3, 0.6, 0.8])
        threshold = evaluator.get_robust_threshold(y_prob)
        assert threshold == 0.5

    def test_falls_back_to_percentile_when_no_positives(self, evaluator):
        evaluator.optimal_threshold = 0.99  # too high
        evaluator.optimal_threshold_percentile = 75.0
        y_prob = np.array([0.1, 0.2, 0.3, 0.4])  # none above 0.99
        threshold = evaluator.get_robust_threshold(y_prob)
        assert threshold == pytest.approx(np.percentile(y_prob, 75.0))
