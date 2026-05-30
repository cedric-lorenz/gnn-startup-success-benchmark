"""
Tests for graph_assembler.py functions.

Tests real functions from src/ml/graph_assembler.py.
"""
import pytest
import torch
import numpy as np
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.ml.graph_assembler import (
    check_for_mask_overlap,
    normalize_features,
    to_csr,
    random_walk_step,
)


class TestCheckForMaskOverlap:
    """Test the real check_for_mask_overlap() function."""

    def test_no_overlap_passes(self):
        """Non-overlapping masks should pass without assertion error."""
        train_mask = torch.tensor([True, False, False, False, False])
        val_mask = torch.tensor([False, True, False, False, False])
        test_mask = torch.tensor([False, False, True, True, True])

        # Should not raise
        check_for_mask_overlap(train_mask, test_mask, val_mask)

    def test_train_test_overlap_fails(self):
        """Overlapping train and test masks should raise AssertionError."""
        train_mask = torch.tensor([True, True, False])
        test_mask = torch.tensor([False, True, False])  # Overlap at index 1
        val_mask = torch.tensor([False, False, True])

        with pytest.raises(AssertionError):
            check_for_mask_overlap(train_mask, test_mask, val_mask)

    def test_train_val_overlap_fails(self):
        """Overlapping train and val masks should raise AssertionError."""
        train_mask = torch.tensor([True, False, True])
        val_mask = torch.tensor([False, False, True])  # Overlap at index 2
        test_mask = torch.tensor([False, True, False])

        with pytest.raises(AssertionError):
            check_for_mask_overlap(train_mask, test_mask, val_mask)

    def test_val_test_overlap_fails(self):
        """Overlapping val and test masks should raise AssertionError."""
        train_mask = torch.tensor([True, False, False])
        val_mask = torch.tensor([False, True, False])
        test_mask = torch.tensor([False, True, True])  # Overlap at index 1

        with pytest.raises(AssertionError):
            check_for_mask_overlap(train_mask, test_mask, val_mask)

    def test_all_three_overlap_fails(self):
        """All three masks overlapping should raise AssertionError."""
        train_mask = torch.tensor([True, True])
        val_mask = torch.tensor([True, False])  # Overlap at index 0
        test_mask = torch.tensor([True, True])  # Overlap at both

        with pytest.raises(AssertionError):
            check_for_mask_overlap(train_mask, test_mask, val_mask)

    def test_empty_masks_pass(self):
        """Empty masks (all False) should pass."""
        train_mask = torch.tensor([False, False, False])
        val_mask = torch.tensor([False, False, False])
        test_mask = torch.tensor([False, False, False])

        # Should not raise
        check_for_mask_overlap(train_mask, test_mask, val_mask)

    def test_single_element_masks(self):
        """Single element masks should work correctly."""
        # Valid: one element in train only
        train_mask = torch.tensor([True])
        val_mask = torch.tensor([False])
        test_mask = torch.tensor([False])
        check_for_mask_overlap(train_mask, test_mask, val_mask)

        # Invalid: same element in train and test
        train_mask = torch.tensor([True])
        test_mask = torch.tensor([True])
        with pytest.raises(AssertionError):
            check_for_mask_overlap(train_mask, test_mask, val_mask)


class TestNormalizeFeatures:
    """Test the real normalize_features() function."""

    def test_fit_and_transform(self):
        """Fitting should create scaler and transform features."""
        features = torch.tensor([[0.0, 100.0], [50.0, 200.0], [100.0, 300.0]])

        normalized, scaler = normalize_features(features, fit=True)

        assert scaler is not None, "Scaler should be returned when fit=True"
        assert normalized.shape == features.shape
        # MinMaxScaler: min becomes 0, max becomes 1
        assert torch.allclose(normalized.min(dim=0).values, torch.zeros(2), atol=1e-5)
        assert torch.allclose(normalized.max(dim=0).values, torch.ones(2), atol=1e-5)

    def test_transform_with_fitted_scaler(self):
        """Transform with existing scaler should use same parameters."""
        # Fit on training data
        train_features = torch.tensor([[0.0, 0.0], [100.0, 100.0]])
        _, scaler = normalize_features(train_features, fit=True)

        # Transform test data with same scaler
        test_features = torch.tensor([[50.0, 50.0], [200.0, 200.0]])
        normalized, _ = normalize_features(test_features, scaler=scaler, fit=False)

        # 50 should map to 0.5, 200 should map to 2.0 (beyond training range)
        assert torch.isclose(normalized[0, 0], torch.tensor(0.5), atol=1e-5)
        assert torch.isclose(normalized[1, 0], torch.tensor(2.0), atol=1e-5)

    def test_output_is_tensor(self):
        """Output should be a torch tensor."""
        features = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        normalized, _ = normalize_features(features, fit=True)

        assert isinstance(normalized, torch.Tensor)
        assert normalized.dtype == torch.float32

    def test_single_feature_column(self):
        """Should work with single feature column."""
        features = torch.tensor([[10.0], [20.0], [30.0]])
        normalized, scaler = normalize_features(features, fit=True)

        assert normalized.shape == (3, 1)
        assert torch.isclose(normalized[0, 0], torch.tensor(0.0), atol=1e-5)
        assert torch.isclose(normalized[2, 0], torch.tensor(1.0), atol=1e-5)


class TestToCsr:
    """Test the real to_csr() function for edge_index to CSR conversion."""

    def test_simple_graph(self):
        """Convert simple edge_index to CSR format."""
        # Simple graph: 0->1, 0->2, 1->2
        edge_index = torch.tensor([[0, 0, 1], [1, 2, 2]])
        num_nodes = 3

        rowptr, col = to_csr(edge_index, num_nodes)

        # rowptr should have num_nodes+1 elements
        assert rowptr.shape[0] == num_nodes + 1
        # Node 0 has 2 edges, node 1 has 1 edge, node 2 has 0 edges
        assert rowptr[0] == 0
        assert rowptr[1] == 2  # node 0 ends at index 2
        assert rowptr[2] == 3  # node 1 ends at index 3
        assert rowptr[3] == 3  # node 2 has no outgoing edges

    def test_disconnected_nodes(self):
        """Handle nodes with no edges."""
        # Only edge: 0->2 (node 1 has no edges)
        edge_index = torch.tensor([[0], [2]])
        num_nodes = 3

        rowptr, col = to_csr(edge_index, num_nodes)

        assert rowptr.shape[0] == 4
        assert rowptr[1] == 1  # node 0 has 1 edge
        assert rowptr[2] == 1  # node 1 has 0 edges (same as previous)

    def test_empty_graph(self):
        """Handle graph with no edges."""
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        num_nodes = 5

        rowptr, col = to_csr(edge_index, num_nodes)

        assert rowptr.shape[0] == num_nodes + 1
        # All entries should be 0 (no edges)
        assert torch.all(rowptr == 0)
        assert col.shape[0] == 0


class TestRandomWalkStep:
    """Test the real random_walk_step() function."""

    def test_single_step_has_neighbors(self):
        """Random walk step should move to a neighbor."""
        # Graph: 0->1, 0->2 (node 0 has neighbors 1 and 2)
        edge_index = torch.tensor([[0, 0], [1, 2]])
        num_nodes = 3
        rowptr, col = to_csr(edge_index, num_nodes)

        current_nodes = torch.tensor([0])
        next_nodes = random_walk_step(current_nodes, rowptr, col)

        # Should move to either 1 or 2
        assert next_nodes[0] in [1, 2]

    def test_node_with_no_neighbors_returns_minus_one(self):
        """Node with no neighbors should return -1."""
        # Graph: 0->1 (node 1 has no outgoing edges)
        edge_index = torch.tensor([[0], [1]])
        num_nodes = 2
        rowptr, col = to_csr(edge_index, num_nodes)

        current_nodes = torch.tensor([1])  # Start at node 1 (no outgoing edges)
        next_nodes = random_walk_step(current_nodes, rowptr, col)

        assert next_nodes[0] == -1

    def test_invalid_node_stays_invalid(self):
        """Invalid node (-1) should stay -1."""
        edge_index = torch.tensor([[0], [1]])
        num_nodes = 2
        rowptr, col = to_csr(edge_index, num_nodes)

        current_nodes = torch.tensor([-1])
        next_nodes = random_walk_step(current_nodes, rowptr, col)

        assert next_nodes[0] == -1

    def test_batch_random_walk(self):
        """Batch of nodes should all take a step."""
        # Graph: 0->1, 0->2, 1->2, 2->0
        edge_index = torch.tensor([[0, 0, 1, 2], [1, 2, 2, 0]])
        num_nodes = 3
        rowptr, col = to_csr(edge_index, num_nodes)

        current_nodes = torch.tensor([0, 1, 2])
        next_nodes = random_walk_step(current_nodes, rowptr, col)

        # Node 0 should go to 1 or 2
        assert next_nodes[0] in [1, 2]
        # Node 1 should go to 2
        assert next_nodes[1] == 2
        # Node 2 should go to 0
        assert next_nodes[2] == 0


class TestMaskMutualExclusivity:
    """Test mask mutual exclusivity property critical for ML correctness."""

    def test_realistic_split_ratios(self):
        """Test with realistic 70/15/15 split."""
        n = 1000
        indices = torch.randperm(n)

        train_size = int(0.7 * n)
        val_size = int(0.15 * n)

        train_mask = torch.zeros(n, dtype=torch.bool)
        val_mask = torch.zeros(n, dtype=torch.bool)
        test_mask = torch.zeros(n, dtype=torch.bool)

        train_mask[indices[:train_size]] = True
        val_mask[indices[train_size:train_size + val_size]] = True
        test_mask[indices[train_size + val_size:]] = True

        # Should pass - no overlap
        check_for_mask_overlap(train_mask, test_mask, val_mask)

        # Verify completeness
        total_assigned = train_mask.sum() + val_mask.sum() + test_mask.sum()
        assert total_assigned == n, "All nodes should be assigned to exactly one split"

    def test_overlap_detection_catches_single_node(self):
        """Even single node overlap should be caught."""
        n = 100
        train_mask = torch.zeros(n, dtype=torch.bool)
        val_mask = torch.zeros(n, dtype=torch.bool)
        test_mask = torch.zeros(n, dtype=torch.bool)

        train_mask[:70] = True
        val_mask[70:85] = True
        test_mask[85:] = True

        # Valid so far
        check_for_mask_overlap(train_mask, test_mask, val_mask)

        # Add single overlap
        val_mask[50] = True  # Node 50 is now in both train and val

        with pytest.raises(AssertionError):
            check_for_mask_overlap(train_mask, test_mask, val_mask)
