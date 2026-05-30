"""
Tests for DownstreamAnalyzer from src/ml/downstream_analysis.py.

Tests real methods by mocking the data loading step.
"""
import pytest
import numpy as np
import sys
from pathlib import Path
from unittest.mock import patch

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


@pytest.fixture
def analyzer():
    """Create a DownstreamAnalyzer with mocked data loading."""
    from src.ml.downstream_analysis import DownstreamAnalyzer

    mock_config = {
        "paths": {
            "crunchbase_dir": "/fake/path",
            "graph_dir": "/fake/path"
        },
        "output_dir": "/fake/output"
    }

    with patch.object(DownstreamAnalyzer, '_load_data', return_value=None):
        analyzer = DownstreamAnalyzer(mock_config)

    return analyzer


class TestEstimateValuation:
    """Test the real DownstreamAnalyzer._estimate_valuation() method."""

    def test_valuation_seed_round(self, analyzer):
        """Seed round with $1M raised at 10% dilution should give $10M valuation."""
        result = analyzer._estimate_valuation(1_000_000, 'seed')
        expected = 1_000_000 / 0.10
        assert result == expected

    def test_valuation_series_a(self, analyzer):
        """Series A at 20% dilution."""
        result = analyzer._estimate_valuation(5_000_000, 'series_a')
        expected = 5_000_000 / 0.20
        assert result == expected

    def test_valuation_series_b(self, analyzer):
        """Series B at 16% dilution."""
        result = analyzer._estimate_valuation(10_000_000, 'series_b')
        expected = 10_000_000 / 0.16
        assert result == expected

    def test_valuation_unknown_round_uses_default(self, analyzer):
        """Unknown round type should use default 15% dilution."""
        result = analyzer._estimate_valuation(3_000_000, 'unknown_round')
        expected = 3_000_000 / 0.15
        assert result == expected

    def test_valuation_zero_raised(self, analyzer):
        """Zero raised amount should return 0."""
        result = analyzer._estimate_valuation(0, 'seed')
        assert result == 0.0

    def test_valuation_negative_raised(self, analyzer):
        """Negative raised amount should return 0."""
        result = analyzer._estimate_valuation(-100, 'seed')
        assert result == 0.0

    def test_valuation_nan_raised(self, analyzer):
        """NaN raised amount should return 0."""
        result = analyzer._estimate_valuation(np.nan, 'seed')
        assert result == 0.0

    def test_valuation_nan_round_type(self, analyzer):
        """NaN round type should default to seed (10% dilution)."""
        result = analyzer._estimate_valuation(1_000_000, np.nan)
        expected = 1_000_000 / 0.10
        assert result == expected

    def test_valuation_grant_non_dilutive(self, analyzer):
        """Grant (0% dilution) should return raised amount."""
        result = analyzer._estimate_valuation(500_000, 'grant')
        assert result == 500_000

    def test_valuation_normalizes_round_type(self, analyzer):
        """Round type should be normalized (lowercase, underscores)."""
        result = analyzer._estimate_valuation(5_000_000, 'Series A')
        expected = 5_000_000 / 0.20
        assert result == expected
