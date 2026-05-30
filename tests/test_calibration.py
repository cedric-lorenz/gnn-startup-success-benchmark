"""
Tests for calibration.py implementations.

These tests import and call the real ModelCalibrationAnalyzer methods.
Each test is designed to FAIL if the implementation has bugs.
"""
import pytest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.ml.calibration import ModelCalibrationAnalyzer


class TestECECalculation:
    """Test the ModelCalibrationAnalyzer._calculate_ece() method."""

    @pytest.fixture
    def analyzer(self):
        return ModelCalibrationAnalyzer(save_plots=False)

    def test_ece_perfectly_calibrated(self, analyzer):
        """
        Perfectly calibrated: confidence = accuracy in each bin.
        ECE should be 0.
        """
        np.random.seed(42)

        # Create perfectly calibrated data
        # For each probability level, actual outcomes match probability
        y_prob = []
        y_true = []

        for prob in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
            n = 100
            y_prob.extend([prob] * n)
            # Generate outcomes with exactly that probability
            outcomes = (np.random.rand(n) < prob).astype(int)
            y_true.extend(outcomes)

        y_prob = np.array(y_prob)
        y_true = np.array(y_true)

        ece = analyzer._calculate_ece(y_true, y_prob, n_bins=10)

        # Should be close to 0 (not exactly due to sampling variance)
        assert ece < 0.05, f"Perfectly calibrated should have ECE≈0, got {ece:.4f}"

    def test_ece_overconfident_high(self, analyzer):
        """
        Overconfident model: predicts high probs but accuracy is low.
        ECE should be high.
        """
        # All predictions are 0.9 confident, but only 30% are correct
        y_prob = np.full(1000, 0.9)
        y_true = np.zeros(1000)
        y_true[:300] = 1  # 30% positive

        ece = analyzer._calculate_ece(y_true, y_prob, n_bins=10)

        # ECE = |0.9 - 0.3| = 0.6 (weighted by 100% in that bin)
        assert ece > 0.5, f"Overconfident should have high ECE, got {ece:.4f}"

    def test_ece_underconfident_medium(self, analyzer):
        """
        Underconfident model: predicts low probs but accuracy is higher.
        """
        # All predictions are 0.3 confident, but 70% are correct
        y_prob = np.full(1000, 0.3)
        y_true = np.zeros(1000)
        y_true[:700] = 1  # 70% positive

        ece = analyzer._calculate_ece(y_true, y_prob, n_bins=10)

        # ECE = |0.3 - 0.7| = 0.4
        assert ece > 0.3, f"Underconfident should have medium-high ECE, got {ece:.4f}"

    def test_ece_known_value(self, analyzer):
        """
        Hand-calculated ECE for verification.

        Bin 0.4-0.5: 50 samples, avg_conf=0.45, accuracy=0.6
        Bin 0.7-0.8: 50 samples, avg_conf=0.75, accuracy=0.5

        ECE = 0.5 * |0.45 - 0.6| + 0.5 * |0.75 - 0.5|
            = 0.5 * 0.15 + 0.5 * 0.25
            = 0.075 + 0.125 = 0.2
        """
        y_prob = np.concatenate([
            np.full(50, 0.45),  # Bin 0.4-0.5
            np.full(50, 0.75),  # Bin 0.7-0.8
        ])
        y_true = np.concatenate([
            np.array([1]*30 + [0]*20),  # 60% accuracy
            np.array([1]*25 + [0]*25),  # 50% accuracy
        ])

        ece = analyzer._calculate_ece(y_true, y_prob, n_bins=10)

        expected = 0.2
        assert abs(ece - expected) < 0.02, f"Expected ECE≈{expected}, got {ece:.4f}"


class TestBrierScore:
    """Test the ModelCalibrationAnalyzer._calculate_brier_score() method."""

    @pytest.fixture
    def analyzer(self):
        return ModelCalibrationAnalyzer(save_plots=False)

    def test_brier_perfect_predictions(self, analyzer):
        """
        Perfect predictions: Brier = 0.
        """
        y_true = np.array([1, 0, 1, 0, 1])
        y_prob = np.array([1.0, 0.0, 1.0, 0.0, 1.0])

        brier = analyzer._calculate_brier_score(y_true, y_prob)

        assert brier == 0.0, f"Perfect predictions should give Brier=0, got {brier}"

    def test_brier_worst_predictions(self, analyzer):
        """
        Completely wrong predictions: Brier = 1.
        """
        y_true = np.array([1, 0, 1, 0, 1])
        y_prob = np.array([0.0, 1.0, 0.0, 1.0, 0.0])

        brier = analyzer._calculate_brier_score(y_true, y_prob)

        assert brier == 1.0, f"Worst predictions should give Brier=1, got {brier}"

    def test_brier_known_value(self, analyzer):
        """
        Hand-calculated Brier score.

        y_true = [1, 0, 1]
        y_prob = [0.8, 0.3, 0.6]

        Brier = mean((p - y)^2)
              = ((0.8-1)^2 + (0.3-0)^2 + (0.6-1)^2) / 3
              = (0.04 + 0.09 + 0.16) / 3
              = 0.29 / 3 = 0.0967
        """
        y_true = np.array([1, 0, 1])
        y_prob = np.array([0.8, 0.3, 0.6])

        brier = analyzer._calculate_brier_score(y_true, y_prob)

        expected = (0.04 + 0.09 + 0.16) / 3
        assert abs(brier - expected) < 0.001, f"Expected {expected:.4f}, got {brier:.4f}"


class TestThresholdOptimization:
    """Test the ModelCalibrationAnalyzer.find_optimal_threshold() method."""

    @pytest.fixture
    def analyzer(self):
        return ModelCalibrationAnalyzer(save_plots=False)

    def test_threshold_maximizes_f1(self, analyzer):
        """
        Optimal threshold should maximize F1.
        """
        np.random.seed(42)

        # Create scenario where optimal threshold is around 0.3
        y_true = np.array([1]*30 + [0]*70)
        y_prob = np.concatenate([
            np.random.uniform(0.2, 0.5, 30),  # Positives: low-medium probs
            np.random.uniform(0.0, 0.3, 70),  # Negatives: low probs
        ])

        optimal_threshold = analyzer.find_optimal_threshold(y_true, y_prob, metric='f1')

        # Verify F1 at optimal is better than at 0.5
        from sklearn.metrics import f1_score

        pred_at_optimal = (y_prob >= optimal_threshold).astype(int)
        pred_at_0_5 = (y_prob >= 0.5).astype(int)

        f1_optimal = f1_score(y_true, pred_at_optimal, zero_division=0)
        f1_0_5 = f1_score(y_true, pred_at_0_5, zero_division=0)

        assert f1_optimal >= f1_0_5, (
            f"Optimal threshold ({optimal_threshold:.2f}) should give F1 >= 0.5 threshold. "
            f"F1@optimal={f1_optimal:.4f}, F1@0.5={f1_0_5:.4f}"
        )

    def test_threshold_stores_percentile(self, analyzer):
        """
        After optimization, analyzer should store percentile-based threshold.
        """
        y_true = np.array([1]*20 + [0]*80)
        y_prob = np.random.rand(100)

        analyzer.find_optimal_threshold(y_true, y_prob, metric='f1')

        assert hasattr(analyzer, 'optimal_threshold_percentile'), (
            "Analyzer should store optimal_threshold_percentile"
        )
        assert 0 <= analyzer.optimal_threshold_percentile <= 100, (
            f"Percentile should be in [0,100], got {analyzer.optimal_threshold_percentile}"
        )


class TestPlattScaling:
    """Test the ModelCalibrationAnalyzer Platt scaling methods."""

    @pytest.fixture
    def analyzer(self):
        return ModelCalibrationAnalyzer(save_plots=False)

    def test_platt_scaling_output_in_range(self, analyzer):
        """
        Calibrated probabilities must be in [0, 1].
        """
        np.random.seed(42)
        y_true = np.random.randint(0, 2, 500)
        y_prob = np.random.rand(500)

        analyzer.fit_platt_scaling(y_true, y_prob)
        calibrated = analyzer.calibrate_probabilities(y_prob)

        assert np.all(calibrated >= 0), "Calibrated probs should be >= 0"
        assert np.all(calibrated <= 1), "Calibrated probs should be <= 1"

    def test_platt_scaling_improves_or_maintains_calibration(self, analyzer):
        """
        Platt scaling should not make calibration significantly worse.
        """
        np.random.seed(42)

        # Overconfident predictions
        y_true = np.random.randint(0, 2, 500)
        y_prob = np.where(y_true == 1,
                         np.random.uniform(0.7, 0.99, 500),
                         np.random.uniform(0.01, 0.3, 500))

        ece_before = analyzer._calculate_ece(y_true, y_prob)

        analyzer.fit_platt_scaling(y_true, y_prob)
        calibrated = analyzer.calibrate_probabilities(y_prob)

        ece_after = analyzer._calculate_ece(y_true, calibrated)

        # Should improve or at worst stay similar
        assert ece_after <= ece_before + 0.05, (
            f"Platt scaling shouldn't make calibration much worse. "
            f"ECE before={ece_before:.4f}, after={ece_after:.4f}"
        )

    def test_calibrate_without_fitting_returns_original(self, analyzer):
        """
        If no calibrator is fitted, calibrate_probabilities returns input unchanged.
        """
        y_prob = np.array([0.1, 0.5, 0.9])

        result = analyzer.calibrate_probabilities(y_prob)

        np.testing.assert_array_equal(result, y_prob,
            err_msg="Without fitting, should return original probs"
        )
