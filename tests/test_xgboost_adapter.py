"""Tests for XGBoostAdapter from src/ml/models.py."""
import pytest
import numpy as np
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.ml.models import XGBoostAdapter


class TestXGBoostAdapterInit:
    """Test XGBoostAdapter initialization."""

    def test_default_params(self):
        adapter = XGBoostAdapter()
        assert adapter.params["n_estimators"] == 100
        assert adapter.params["max_depth"] == 6
        assert adapter.params["objective"] == "binary:logistic"
        assert adapter.model is None

    def test_custom_params(self):
        adapter = XGBoostAdapter(n_estimators=50, max_depth=3, learning_rate=0.05)
        assert adapter.params["n_estimators"] == 50
        assert adapter.params["max_depth"] == 3
        assert adapter.params["learning_rate"] == 0.05

    def test_to_device_returns_self(self):
        """to() is a no-op for compatibility — should return self."""
        adapter = XGBoostAdapter()
        result = adapter.to("cuda")
        assert result is adapter


class TestXGBoostAdapterFitPredict:
    """Test XGBoostAdapter fit/predict on synthetic data."""

    @pytest.fixture
    def trained_adapter(self):
        np.random.seed(42)
        X = np.random.randn(200, 5)
        y = (X[:, 0] + X[:, 1] > 0).astype(int)
        adapter = XGBoostAdapter(n_estimators=10, max_depth=2)
        adapter.fit(X, y)
        return adapter, X, y

    def test_fit_creates_model(self, trained_adapter):
        adapter, _, _ = trained_adapter
        assert adapter.model is not None

    def test_predict_returns_correct_shape(self, trained_adapter):
        adapter, X, _ = trained_adapter
        preds = adapter.predict(X)
        assert preds.shape == (200,)

    def test_predict_proba_returns_probabilities(self, trained_adapter):
        adapter, X, _ = trained_adapter
        proba = adapter.predict_proba(X)
        assert proba.shape == (200, 2)
        assert np.allclose(proba.sum(axis=1), 1.0)
        assert (proba >= 0).all() and (proba <= 1).all()

    def test_predictions_better_than_random(self, trained_adapter):
        """Simple linearly separable data — should get >70% accuracy."""
        adapter, X, y = trained_adapter
        preds = adapter.predict(X)
        accuracy = (preds == y).mean()
        assert accuracy > 0.7, f"Accuracy {accuracy:.2f} should be > 0.7 on easy data"
