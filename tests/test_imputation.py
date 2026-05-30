"""Tests for split-aware imputation from src/ml/imputation.py."""
import pytest
import numpy as np
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.ml.imputation import (
    MeanImputer,
    MedianImputer,
    ZeroImputer,
    ConstantImputer,
    MostFrequentImputer,
    GraphImputer,
)


class TestMeanImputer:
    """Test MeanImputer fills NaNs with column means."""

    def test_fills_nans_with_mean(self):
        X_train = np.array([[1.0, 2.0], [3.0, np.nan], [5.0, 4.0]])
        imp = MeanImputer()
        imp.fit(X_train)
        result = imp.transform(X_train)
        assert not np.isnan(result).any()
        # Column 1 mean = (2+4)/2 = 3.0
        assert result[1, 1] == pytest.approx(3.0)

    def test_transform_uses_train_statistics(self):
        """Transform on test data should use train-fitted means (leakage prevention)."""
        X_train = np.array([[1.0], [3.0], [5.0]])  # mean = 3.0
        X_test = np.array([[np.nan], [10.0]])
        imp = MeanImputer()
        imp.fit(X_train)
        result = imp.transform(X_test)
        assert result[0, 0] == pytest.approx(3.0), "Should use train mean, not test mean"

    def test_raises_if_not_fitted(self):
        imp = MeanImputer()
        with pytest.raises(ValueError, match="must be fitted"):
            imp.transform(np.array([[1.0]]))


class TestMedianImputer:
    def test_fills_with_median(self):
        X = np.array([[1.0], [np.nan], [3.0], [100.0]])  # median = 3.0
        imp = MedianImputer()
        imp.fit(X)
        result = imp.transform(X)
        assert result[1, 0] == pytest.approx(3.0)


class TestZeroImputer:
    def test_fills_with_zero(self):
        X = np.array([[np.nan, 5.0], [2.0, np.nan]])
        imp = ZeroImputer()
        imp.fit(X)
        result = imp.transform(X)
        assert result[0, 0] == 0.0
        assert result[1, 1] == 0.0

    def test_preserves_existing_values(self):
        X = np.array([[3.0, np.nan]])
        imp = ZeroImputer()
        imp.fit(X)
        result = imp.transform(X)
        assert result[0, 0] == 3.0


class TestConstantImputer:
    def test_fills_with_custom_value(self):
        X = np.array([[np.nan], [2.0]])
        imp = ConstantImputer(fill_value=-1)
        imp.fit(X)
        result = imp.transform(X)
        assert result[0, 0] == -1.0


class TestMostFrequentImputer:
    def test_fills_with_mode(self):
        X = np.array([[1.0], [2.0], [2.0], [np.nan]])
        imp = MostFrequentImputer()
        imp.fit(X)
        result = imp.transform(X)
        assert result[3, 0] == 2.0


class TestGraphImputer:
    """Test the main GraphImputer with registry-based imputer selection."""

    def test_registry_contains_all_methods(self):
        expected = {"mean", "median", "most_frequent", "constant", "knn", "iterative", "zero"}
        assert expected.issubset(set(GraphImputer.IMPUTER_REGISTRY.keys()))

    def test_init_with_config(self):
        """Test initialization with config and feature map."""
        imputation_config = {
            "enabled": True,
            "node_types": {
                "startup": {
                    "numerical_method": "zero",
                    "categorical_method": "most_frequent",
                    "categorical_columns": [],
                },
            },
        }
        feature_map = {"startup": ["feat_a", "feat_b"]}
        gi = GraphImputer(imputation_config, feature_map)
        assert gi.imputation_config == imputation_config

    def test_1d_input_handled(self):
        """BaseImputer should handle 1D input by reshaping."""
        X = np.array([1.0, np.nan, 3.0])
        imp = MeanImputer()
        imp.fit(X)
        result = imp.transform(X)
        assert not np.isnan(result).any()
