"""
Tests for preprocessing.py functions.

Tests real functions from src/ml/preprocessing.py.
"""
import pytest
import torch
import numpy as np
import pandas as pd
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.ml.preprocessing import (
    merge_edge_df,
    edge_preprocessing,
    preprocess_startup_dates,
)


class TestMergeEdgeDF:
    """Test the real merge_edge_df() function."""

    def test_basic_merge(self):
        """Basic edge merge should create correct edge_index."""
        # Create node dataframes
        first_df = pd.DataFrame({
            'uuid': ['a', 'b', 'c'],
            'node_id': [0, 1, 2]
        })
        second_df = pd.DataFrame({
            'uuid': ['x', 'y', 'z'],
            'node_id': [0, 1, 2]
        })
        # Create edge dataframe: a->x, b->y, c->z
        edge_df = pd.DataFrame({
            'src_uuid': ['a', 'b', 'c'],
            'dst_uuid': ['x', 'y', 'z']
        })

        edge_index = merge_edge_df(
            first_df, second_df, edge_df,
            first_id_col='node_id', second_id_col='node_id',
            first_uuid_col='uuid', second_uuid_col='uuid',
            edge_first_col='src_uuid', edge_second_col='dst_uuid'
        )

        assert edge_index.shape == (2, 3)
        # Check edges: 0->0, 1->1, 2->2
        expected = torch.tensor([[0, 1, 2], [0, 1, 2]])
        assert torch.equal(edge_index, expected)

    def test_merge_with_missing_uuid_drops_edge(self):
        """Missing UUIDs in edge_df should result in dropped edges (empty result)."""
        first_df = pd.DataFrame({
            'uuid': ['a', 'b'],
            'node_id': [0, 1]
        })
        second_df = pd.DataFrame({
            'uuid': ['x', 'y'],
            'node_id': [0, 1]
        })
        # Edge references non-existent UUID 'z'
        edge_df = pd.DataFrame({
            'src_uuid': ['a'],
            'dst_uuid': ['z']  # 'z' doesn't exist in second_df
        })

        # Merge drops edges with unmatched UUIDs
        edge_index = merge_edge_df(
            first_df, second_df, edge_df,
            first_id_col='node_id', second_id_col='node_id',
            first_uuid_col='uuid', second_uuid_col='uuid',
            edge_first_col='src_uuid', edge_second_col='dst_uuid'
        )

        # Should return empty tensor since 'z' doesn't match
        assert edge_index.shape == (2, 0)

    def test_merge_preserves_edge_order(self):
        """Edge order should be preserved after merge."""
        first_df = pd.DataFrame({
            'uuid': ['a', 'b', 'c'],
            'node_id': [0, 1, 2]
        })
        second_df = pd.DataFrame({
            'uuid': ['x', 'y'],
            'node_id': [0, 1]
        })
        # Specific edge order
        edge_df = pd.DataFrame({
            'src_uuid': ['c', 'a', 'b'],  # Order: c, a, b
            'dst_uuid': ['y', 'x', 'y']
        })

        edge_index = merge_edge_df(
            first_df, second_df, edge_df,
            first_id_col='node_id', second_id_col='node_id',
            first_uuid_col='uuid', second_uuid_col='uuid',
            edge_first_col='src_uuid', edge_second_col='dst_uuid'
        )

        # Should be: 2->1, 0->0, 1->1
        assert edge_index[0, 0] == 2  # c -> 2
        assert edge_index[0, 1] == 0  # a -> 0
        assert edge_index[0, 2] == 1  # b -> 1


class TestEdgePreprocessing:
    """Test the real edge_preprocessing() function."""

    def test_basic_preprocessing(self):
        """Basic edge preprocessing should return edge_index."""
        first_df = pd.DataFrame({
            'uuid': ['a', 'b'],
            'node_id': [0, 1]
        })
        second_df = pd.DataFrame({
            'uuid': ['x', 'y'],
            'node_id': [0, 1]
        })
        edge_df = pd.DataFrame({
            'src': ['a', 'b'],
            'dst': ['y', 'x']
        })

        edge_index, attrs = edge_preprocessing(
            edge_df, first_df, second_df,
            first_id_col='node_id', second_id_col='node_id',
            first_uuid_col='uuid', second_uuid_col='uuid',
            edge_first_col='src', edge_second_col='dst'
        )

        assert edge_index.shape == (2, 2)
        assert attrs is None  # No attributes specified

    def test_ablation_drops_edges_when_node_df_none(self):
        """When first_df is None (ablation), should return empty edges."""
        second_df = pd.DataFrame({
            'uuid': ['x', 'y'],
            'node_id': [0, 1]
        })
        edge_df = pd.DataFrame({
            'src': ['a'],
            'dst': ['x']
        })

        edge_index, attrs = edge_preprocessing(
            edge_df, None, second_df,  # first_df is None (ablated)
            first_id_col='node_id', second_id_col='node_id',
            first_uuid_col='uuid', second_uuid_col='uuid'
        )

        assert edge_index.shape == (2, 0)  # Empty edge index
        assert attrs is None

    def test_ablation_drops_edges_when_second_df_none(self):
        """When second_df is None (ablation), should return empty edges."""
        first_df = pd.DataFrame({
            'uuid': ['a', 'b'],
            'node_id': [0, 1]
        })
        edge_df = pd.DataFrame({
            'src': ['a'],
            'dst': ['x']
        })

        edge_index, attrs = edge_preprocessing(
            edge_df, first_df, None,  # second_df is None (ablated)
            first_id_col='node_id', second_id_col='node_id',
            first_uuid_col='uuid', second_uuid_col='uuid'
        )

        assert edge_index.shape == (2, 0)

    def test_edge_attributes_returned(self):
        """Edge attributes should be returned when specified."""
        first_df = pd.DataFrame({
            'uuid': ['a'],
            'node_id': [0]
        })
        second_df = pd.DataFrame({
            'uuid': ['x'],
            'node_id': [0]
        })
        edge_df = pd.DataFrame({
            'src': ['a'],
            'dst': ['x'],
            'weight': [0.5],
            'type': ['invested']
        })

        edge_index, attrs = edge_preprocessing(
            edge_df, first_df, second_df,
            first_id_col='node_id', second_id_col='node_id',
            first_uuid_col='uuid', second_uuid_col='uuid',
            edge_first_col='src', edge_second_col='dst',
            edge_attribute_names=['weight', 'type']
        )

        assert attrs is not None
        assert 'weight' in attrs.columns
        assert 'type' in attrs.columns
        assert attrs['weight'].iloc[0] == 0.5


class TestPreprocessStartupDates:
    """Test the real preprocess_startup_dates() function."""

    def test_filters_by_date_range(self):
        """Should filter startups outside date range."""
        df = pd.DataFrame({
            'name': ['old', 'valid', 'future'],
            'founded_on': ['2010-01-01', '2020-06-15', '2030-01-01']
        })
        params = {
            'start_date': '2015-01-01',
            'end_date': '2025-01-01',
            'column': 'founded_on'
        }

        result = preprocess_startup_dates(df, params)

        assert len(result) == 1
        assert result['name'].iloc[0] == 'valid'

    def test_handles_missing_column(self):
        """Should return unchanged df if column doesn't exist."""
        df = pd.DataFrame({
            'name': ['a', 'b'],
            'other_date': ['2020-01-01', '2021-01-01']
        })
        params = {
            'start_date': '2015-01-01',
            'end_date': '2025-01-01',
            'column': 'founded_on'  # This column doesn't exist
        }

        result = preprocess_startup_dates(df, params)

        # Should return unchanged
        assert len(result) == 2

    def test_handles_invalid_dates(self):
        """Should handle invalid date formats gracefully."""
        df = pd.DataFrame({
            'name': ['valid', 'invalid', 'also_valid'],
            'founded_on': ['2020-01-01', 'not-a-date', '2021-06-15']
        })
        params = {
            'start_date': '2015-01-01',
            'end_date': '2025-01-01',
            'column': 'founded_on'
        }

        result = preprocess_startup_dates(df, params)

        # Invalid date becomes NaT and is filtered out
        assert len(result) == 2
        assert 'invalid' not in result['name'].values

    def test_inclusive_date_boundaries(self):
        """Date boundaries should be inclusive."""
        df = pd.DataFrame({
            'name': ['start', 'middle', 'end'],
            'founded_on': ['2020-01-01', '2020-06-15', '2020-12-31']
        })
        params = {
            'start_date': '2020-01-01',
            'end_date': '2020-12-31',
            'column': 'founded_on'
        }

        result = preprocess_startup_dates(df, params)

        # All three should be included (inclusive boundaries)
        assert len(result) == 3

    def test_uses_default_params(self):
        """Should use default params if not specified."""
        df = pd.DataFrame({
            'name': ['recent'],
            'founded_on': ['2020-01-01']
        })
        params = {}  # No params specified

        result = preprocess_startup_dates(df, params)

        # Default start_date is 2014-01-01, end_date is 2100-01-01
        # 2020-01-01 is within range
        assert len(result) == 1


class TestEdgeIndexBounds:
    """Test that edge indices are always within valid bounds."""

    def test_edge_index_non_negative(self):
        """Edge indices should never be negative."""
        first_df = pd.DataFrame({
            'uuid': ['a', 'b', 'c'],
            'node_id': [0, 1, 2]
        })
        second_df = pd.DataFrame({
            'uuid': ['x', 'y', 'z'],
            'node_id': [0, 1, 2]
        })
        edge_df = pd.DataFrame({
            'src': ['a', 'b', 'c', 'a'],
            'dst': ['x', 'y', 'z', 'z']
        })

        edge_index = merge_edge_df(
            first_df, second_df, edge_df,
            first_id_col='node_id', second_id_col='node_id',
            first_uuid_col='uuid', second_uuid_col='uuid',
            edge_first_col='src', edge_second_col='dst'
        )

        assert (edge_index >= 0).all(), "Edge indices should be non-negative"

    def test_edge_index_within_node_count(self):
        """Edge indices should be less than num_nodes."""
        first_df = pd.DataFrame({
            'uuid': ['a', 'b'],
            'node_id': [0, 1]
        })
        second_df = pd.DataFrame({
            'uuid': ['x', 'y', 'z'],
            'node_id': [0, 1, 2]
        })
        edge_df = pd.DataFrame({
            'src': ['a', 'b', 'a'],
            'dst': ['z', 'y', 'x']
        })

        edge_index = merge_edge_df(
            first_df, second_df, edge_df,
            first_id_col='node_id', second_id_col='node_id',
            first_uuid_col='uuid', second_uuid_col='uuid',
            edge_first_col='src', edge_second_col='dst'
        )

        num_src_nodes = len(first_df)
        num_dst_nodes = len(second_df)

        assert (edge_index[0] < num_src_nodes).all(), "Source indices should be < num_src_nodes"
        assert (edge_index[1] < num_dst_nodes).all(), "Dest indices should be < num_dst_nodes"
