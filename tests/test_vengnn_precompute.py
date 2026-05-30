"""Correctness tests for VenGNN precompute utilities."""
import pytest
import torch

from src.ml.vengnn_precompute import (
    build_union_startup_adjacency,
    sample_random_walks,
)


def test_build_union_startup_adjacency_sums_pathcounts():
    # Two metapaths, overlapping edge (0,1) with weights 2 and 3.
    mp1 = ("startup", "a", "startup")
    mp2 = ("startup", "b", "startup")
    edge_index_dict = {
        mp1: torch.tensor([[0, 1], [1, 0]]),
        mp2: torch.tensor([[0, 2], [1, 1]]),
    }
    edge_weight_dict = {
        mp1: torch.tensor([2.0, 5.0]),
        mp2: torch.tensor([3.0, 7.0]),
    }
    ei, ew = build_union_startup_adjacency(
        edge_index_dict, [mp1, mp2], edge_weight_dict
    )
    # Sort by (src, dst) for stable comparison.
    order = torch.argsort(ei[0] * 10 + ei[1])
    ei_s = ei[:, order]
    ew_s = ew[order]
    expected_ei = torch.tensor([[0, 1, 2], [1, 0, 1]])
    expected_ew = torch.tensor([2.0 + 3.0, 5.0, 7.0])
    assert torch.equal(ei_s, expected_ei)
    assert torch.allclose(ew_s, expected_ew)


def test_build_union_defaults_missing_weights_to_ones():
    mp = ("startup", "c", "startup")
    ei, ew = build_union_startup_adjacency(
        {mp: torch.tensor([[0, 0], [1, 2]])},
        [mp],
        edge_weight_dict=None,
    )
    assert torch.equal(ei, torch.tensor([[0, 0], [1, 2]]))
    assert torch.allclose(ew, torch.tensor([1.0, 1.0]))


def test_random_walk_starts_at_given_nodes():
    ei = torch.tensor([[0, 1, 2], [1, 2, 0]])
    ew = torch.tensor([1.0, 1.0, 1.0])
    walks = sample_random_walks(
        start_nodes=torch.tensor([0, 1, 2]),
        edge_index=ei,
        edge_weight=ew,
        num_nodes=3,
        walks_per_node=4,
        walk_length=3,
    )
    assert walks.shape == (3, 4, 3)
    # First column is always the start node.
    assert torch.equal(walks[:, :, 0], torch.tensor([[0] * 4, [1] * 4, [2] * 4]))


def test_random_walk_on_triangle_follows_edges():
    # Directed triangle 0->1, 1->2, 2->0. Only one possible walk per step.
    ei = torch.tensor([[0, 1, 2], [1, 2, 0]])
    ew = torch.tensor([1.0, 1.0, 1.0])
    walks = sample_random_walks(
        torch.tensor([0]), ei, ew,
        num_nodes=3, walks_per_node=2, walk_length=4,
    )
    for j in range(2):
        assert torch.equal(walks[0, j], torch.tensor([0, 1, 2, 0]))


def test_random_walk_dead_end_repeats_last_node():
    # Node 0 has no outgoing edges.
    ei = torch.tensor([[1, 1], [0, 2]])
    ew = torch.tensor([1.0, 1.0])
    walks = sample_random_walks(
        torch.tensor([0]), ei, ew,
        num_nodes=3, walks_per_node=2, walk_length=5,
    )
    # Every step stays at 0.
    assert torch.equal(walks[0], torch.zeros(2, 5, dtype=torch.long))


def test_random_walk_reproducibility_with_seed():
    ei = torch.tensor([[0, 0, 1, 2], [1, 2, 2, 0]])
    ew = torch.tensor([1.0, 1.0, 1.0, 1.0])
    kw = dict(
        start_nodes=torch.tensor([0, 1, 2]),
        edge_index=ei, edge_weight=ew,
        num_nodes=3, walks_per_node=5, walk_length=6,
    )
    w1 = sample_random_walks(**kw, seed=42)
    w2 = sample_random_walks(**kw, seed=42)
    w3 = sample_random_walks(**kw, seed=43)
    assert torch.equal(w1, w2)
    assert not torch.equal(w1, w3)


def test_build_union_reconciles_mixed_cpu_tensors():
    """Regression: edge_weight entries may live on a different device than
    edge_index (e.g. weights cached on CPU while edges were .to(device)'d).
    All outputs must share the edge_index device; no torch.cat crash."""
    mp1 = ("startup", "a", "startup")
    mp2 = ("startup", "b", "startup")
    ei_dict = {
        mp1: torch.tensor([[0, 1], [1, 0]]),
        mp2: torch.tensor([[0], [1]]),
    }
    # mp1 weights exist, mp2 defaults to ones; all on CPU but different origin.
    ew_dict = {mp1: torch.tensor([2.0, 3.0]), mp2: None}
    ei, ew = build_union_startup_adjacency(ei_dict, [mp1, mp2], ew_dict)
    assert ei.device == ew.device == ei_dict[mp1].device


def test_union_adjacency_edge_count_increases_when_metapaths_added():
    mps = [("startup", f"m{i}", "startup") for i in range(3)]
    edges = {mps[0]: torch.tensor([[0], [1]]),
             mps[1]: torch.tensor([[0], [2]]),
             mps[2]: torch.tensor([[1], [2]])}
    ei, ew = build_union_startup_adjacency(edges, mps)
    assert ei.size(1) == 3
    assert ew.size(0) == 3
