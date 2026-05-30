"""Unit tests for src.ml.graph_variants.apply_graph_variant.

Exercises the filter with fabricated mini-HeteroData to verify the five
variants drop/keep the right edge types without needing the real 1 GB graph.
"""
from __future__ import annotations
import pytest
import torch
from torch_geometric.data import HeteroData

from src.ml.graph_variants import (
    G3_NEAR_BASELINE_METAPATHS,
    G4_HETEROPHILIC_METAPATHS,
    GRAPH_VARIANTS,
    apply_graph_variant,
)


def _make_mini_graph():
    """HeteroData with every edge-type shape we care about for G1-G5.

    Node types: startup, investor, sector.
    Edge types:
      - startup→investor (base)
      - sector→startup (base)
      - startup→sector (base)
      - sector self-loop (base)
      - 3 startup→startup metapaths:
        * sector_peers  (startup-startup path; must be dropped explicitly, not via sector removal)
        * city_peers    (in G3 drop set)
        * late_founder_vc_employment (in G4 keep set)
    """
    g = HeteroData()
    g['startup'].num_nodes = 10
    g['investor'].num_nodes = 5
    g['sector'].num_nodes = 3
    g['startup', 'funded_by', 'investor'].edge_index = torch.tensor([[0, 1], [0, 1]])
    g['sector', 'self_loop', 'sector'].edge_index = torch.tensor([[0, 1, 2], [0, 1, 2]])
    g['sector', 'rev_in_sector', 'startup'].edge_index = torch.tensor([[0, 1], [0, 1]])
    g['startup', 'in_sector', 'sector'].edge_index = torch.tensor([[0, 1], [0, 1]])
    g['startup', 'sector_peers', 'startup'].edge_index = torch.tensor([[0, 1], [1, 2]])
    g['startup', 'city_peers', 'startup'].edge_index = torch.tensor([[0, 1], [1, 2]])
    g['startup', 'late_founder_vc_employment', 'startup'].edge_index = torch.tensor(
        [[0, 1], [1, 2]]
    )
    g['startup', 'early_portfolio_siblings', 'startup'].edge_index = torch.tensor(
        [[0, 1], [1, 2]]
    )
    return g


def test_g1_full_is_noop():
    g = _make_mini_graph()
    before = set(g.edge_types)
    g2 = apply_graph_variant(g, 'g1_full')
    assert set(g2.edge_types) == before
    assert 'sector' in g2.node_types


def test_none_variant_is_noop():
    g = _make_mini_graph()
    before = set(g.edge_types)
    g2 = apply_graph_variant(g, None)
    assert set(g2.edge_types) == before


def test_g2_no_sector_removes_sector_node_and_sector_peers():
    g = _make_mini_graph()
    apply_graph_variant(g, 'g2_no_sector')
    assert 'sector' not in g.node_types
    # All edges touching sector type must be gone
    for et in g.edge_types:
        assert et[0] != 'sector' and et[2] != 'sector'
    # The sector-derived startup-startup metapath is also dropped
    # (G2 = "drop sector and all related paths").
    assert ('startup', 'sector_peers', 'startup') not in g.edge_types
    # Other startup-startup metapaths should remain
    assert ('startup', 'city_peers', 'startup') in g.edge_types
    assert ('startup', 'late_founder_vc_employment', 'startup') in g.edge_types


def test_g3_pruned_drops_near_baseline_metapaths():
    g = _make_mini_graph()
    apply_graph_variant(g, 'g3_pruned')
    # sector gone (G3 builds on G2)
    assert 'sector' not in g.node_types
    # Near-baseline paths gone
    assert ('startup', 'city_peers', 'startup') not in g.edge_types
    # Signal paths survive
    assert ('startup', 'late_founder_vc_employment', 'startup') in g.edge_types
    assert ('startup', 'early_portfolio_siblings', 'startup') in g.edge_types


def test_g4_heterophily_keeps_only_listed_paths():
    g = _make_mini_graph()
    apply_graph_variant(g, 'g4_heterophily')
    # sector gone
    assert 'sector' not in g.node_types
    # late_founder_vc_employment is in the keep list
    assert ('startup', 'late_founder_vc_employment', 'startup') in g.edge_types
    # early_portfolio_siblings is also in the keep list (excess 0.181)
    assert ('startup', 'early_portfolio_siblings', 'startup') in g.edge_types
    # sector_peers and city_peers are not in the keep list
    assert ('startup', 'sector_peers', 'startup') not in g.edge_types
    assert ('startup', 'city_peers', 'startup') not in g.edge_types


def test_g5_base_drops_all_startup_startup_metapaths():
    g = _make_mini_graph()
    apply_graph_variant(g, 'g5_base')
    # sector gone
    assert 'sector' not in g.node_types
    # Every startup-startup edge type is gone
    for et in g.edge_types:
        if et[0] == 'startup' and et[2] == 'startup':
            pytest.fail(f"G5 base-only should not contain startup-startup edge {et}")
    # But base edges remain
    assert ('startup', 'funded_by', 'investor') in g.edge_types


def test_unknown_variant_raises():
    g = _make_mini_graph()
    with pytest.raises(ValueError, match='Unknown graph_variant'):
        apply_graph_variant(g, 'g99_does_not_exist')


def test_all_five_variants_are_defined():
    """The G1-G5 names are part of the paper's Table 3 — lock them."""
    expected = {'g1_full', 'g2_no_sector', 'g3_pruned', 'g4_heterophily', 'g5_base'}
    assert set(GRAPH_VARIANTS.keys()) == expected


def test_g4_keep_set_has_exactly_20_paths():
    """G4's 20-path count is a pre-registered number for the paper."""
    assert len(G4_HETEROPHILIC_METAPATHS) == 20


def test_g3_drops_3_near_baseline_paths():
    """G3's 3-path drop list is pre-registered (excess het < 0.025 on momentum)."""
    assert len(G3_NEAR_BASELINE_METAPATHS) == 3
    assert G3_NEAR_BASELINE_METAPATHS == {
        'city_peers', 'alumni_network', 'serial_founder'
    }
