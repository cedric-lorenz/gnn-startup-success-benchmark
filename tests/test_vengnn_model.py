"""Correctness tests for the VenGNN model class."""
import pytest
import torch
from torch_geometric.data import HeteroData

from src.ml.models import VenGNN


def _toy_graph(num_startups=8, feat_dim=8, num_mps=3, seed=0):
    torch.manual_seed(seed)
    g = HeteroData()
    g["startup"].x = torch.randn(num_startups, feat_dim)
    g["startup"].y = torch.randint(0, 2, (num_startups, 2)).float()
    for i in range(num_mps):
        rel = f"mp_{i}"
        src = torch.randint(0, num_startups, (12,))
        dst = torch.randint(0, num_startups, (12,))
        ei = torch.stack([src, dst], dim=0)
        g[("startup", rel, "startup")].edge_index = ei
        g[("startup", rel, "startup")].edge_weight = torch.rand(12) * 5
    # Distractor non-startup-target metapath — must be filtered out.
    g[("startup", "funded_by", "investor")].edge_index = torch.tensor([[0], [0]])
    return g, feat_dim


def _build(model_kwargs=None, num_mps=3):
    g, d = _toy_graph(num_mps=num_mps)
    kw = dict(
        in_channels=d,
        hidden_channels=16,
        metadata=g.metadata(),
        num_layers=2,
        heads=2,
        dropout=0.0,
        rw_num_walks=3,
        rw_walk_length=4,
    )
    kw.update(model_kwargs or {})
    model = VenGNN(**kw)

    # Register edge weights from HeteroData so GAT sees PathCount.
    weight_dict = {mp: g[mp].edge_weight for mp in model.metapaths}
    model.set_metapath_edge_weights(weight_dict)
    return model, g


def test_only_startup_startup_metapaths_selected():
    model, _ = _build(num_mps=4)
    for mp in model.metapaths:
        assert mp[0] == "startup" and mp[2] == "startup"
    assert len(model.metapaths) == 4


def test_forward_produces_expected_shape_masked_multi_task():
    model, g = _build()
    model.precompute(g.x_dict, g.edge_index_dict)
    out = model(g.x_dict, g.edge_index_dict)
    stacked = out["masked_multi_task_output"]["startup"]
    assert stacked.shape == (g["startup"].num_nodes, 2)
    emb = out["embedding"]["startup"]
    assert emb.shape == (g["startup"].num_nodes, model.hidden_channels)


def test_branch_a_fusion_reduces_from_M_times_hidden_to_hidden():
    model, _ = _build(num_mps=5)
    M = len(model.metapaths)
    assert model.branch_a_fuse.in_features == M * model.hidden_channels
    assert model.branch_a_fuse.out_features == model.hidden_channels


def test_random_walk_cache_shape():
    model, g = _build({"rw_num_walks": 5, "rw_walk_length": 6})
    model.precompute(g.x_dict, g.edge_index_dict)
    assert model._walks.shape == (g["startup"].num_nodes, 5, 6)


def test_position_weights_are_learnable_length_L():
    model, _ = _build({"rw_walk_length": 7})
    assert model.rw_position_weights.requires_grad
    assert model.rw_position_weights.shape == (7,)


def test_forward_without_precompute_raises():
    model, g = _build()
    with pytest.raises(RuntimeError, match="precompute"):
        model(g.x_dict, g.edge_index_dict)


def test_gradient_flows_through_both_branches():
    model, g = _build()
    model.precompute(g.x_dict, g.edge_index_dict)
    out = model(g.x_dict, g.edge_index_dict)
    loss = out["masked_multi_task_output"]["startup"].sum()
    loss.backward()
    # Branch A: at least one GAT layer's lin weight should have grad.
    gat_grads = [
        p.grad for layer in model.gat_layers for mod in layer.values()
        for p in mod.parameters() if p.grad is not None and p.grad.abs().sum() > 0
    ]
    assert gat_grads, "Branch A GAT did not receive gradient"
    # Branch B: rw_attn Q/K/V projections and rw_proj should have grads.
    assert model.rw_proj.weight.grad is not None
    assert model.rw_proj.weight.grad.abs().sum() > 0
    # Position weights (W^2) should have grad too.
    assert model.rw_position_weights.grad is not None
    assert model.rw_position_weights.grad.abs().sum() > 0


def test_paper_attention_branch_runs_and_matches_shape():
    """use_paper_attention=True replaces nn.MultiheadAttention with
    softmax(VV^T / sqrt(d)) V. Output shape must match standard MHA path."""
    model, g = _build({"use_paper_attention": True})
    assert model.rw_attn is None  # no MHA allocated
    model.precompute(g.x_dict, g.edge_index_dict)
    out = model(g.x_dict, g.edge_index_dict)
    emb = out["embedding"]["startup"]
    assert emb.shape == (g["startup"].num_nodes, model.hidden_channels)


def test_branch_mode_a_only_skips_rw_attention():
    model, g = _build({"branch_mode": "a_only"})
    model.precompute(g.x_dict, g.edge_index_dict)
    out = model(g.x_dict, g.edge_index_dict)
    assert out["embedding"]["startup"].shape == (g["startup"].num_nodes, model.hidden_channels)
    # RW position weights shouldn't receive gradient because Branch B is dead.
    loss = out["masked_multi_task_output"]["startup"].sum()
    loss.backward()
    assert model.rw_position_weights.grad is None or model.rw_position_weights.grad.abs().sum() == 0


def test_branch_mode_b_only_skips_gat():
    model, g = _build({"branch_mode": "b_only"})
    model.precompute(g.x_dict, g.edge_index_dict)
    out = model(g.x_dict, g.edge_index_dict)
    assert out["embedding"]["startup"].shape == (g["startup"].num_nodes, model.hidden_channels)
    loss = out["masked_multi_task_output"]["startup"].sum()
    loss.backward()
    # GAT params should have no grad because Branch A is dead.
    gat_grads = [
        p.grad for layer in model.gat_layers for mod in layer.values()
        for p in mod.parameters() if p.grad is not None and p.grad.abs().sum() > 0
    ]
    assert not gat_grads, f"Branch A was not skipped; GAT got gradients: {len(gat_grads)}"


def test_paper_widening_preserves_M_times_hidden():
    """paper_widening=True keeps Branch A at M*hidden dim; Branch B widens to match."""
    model, g = _build({"paper_widening": True})
    M = len(model.metapaths)
    assert model.branch_a_fuse is None  # no reduction layer
    assert model.input_proj is None     # no shared input proj either
    assert model.branch_dim == M * model.hidden_channels
    # post_fuse reduces M*hidden -> hidden
    assert model.post_fuse[1].in_features == M * model.hidden_channels
    assert model.post_fuse[1].out_features == model.hidden_channels
    # rw_proj widens raw features to M*hidden
    assert model.rw_proj.out_features == M * model.hidden_channels
    # Forward works
    model.precompute(g.x_dict, g.edge_index_dict)
    out = model(g.x_dict, g.edge_index_dict)
    assert out["embedding"]["startup"].shape == (g["startup"].num_nodes, model.hidden_channels)


def test_invalid_branch_mode_raises():
    import pytest
    with pytest.raises(ValueError, match="branch_mode"):
        _build({"branch_mode": "bogus"})


def test_invalid_gat_edge_mode_raises():
    import pytest
    with pytest.raises(ValueError, match="gat_edge_mode"):
        _build({"gat_edge_mode": "bogus"})


def test_gat_edge_mode_none_runs_without_edge_attr():
    model, g = _build({"gat_edge_mode": "none"})
    model.precompute(g.x_dict, g.edge_index_dict)
    out = model(g.x_dict, g.edge_index_dict)
    assert out["embedding"]["startup"].shape == (g["startup"].num_nodes, model.hidden_channels)


def test_gat_edge_mode_multiplicative_runs():
    model, g = _build({"gat_edge_mode": "multiplicative"})
    model.precompute(g.x_dict, g.edge_index_dict)
    out = model(g.x_dict, g.edge_index_dict)
    emb = out["embedding"]["startup"]
    assert emb.shape == (g["startup"].num_nodes, model.hidden_channels)
    # Gradient flows through GAT internal params
    loss = out["masked_multi_task_output"]["startup"].sum()
    loss.backward()
    gat_grads = [p.grad for layer in model.gat_layers for mod in layer.values()
                 for p in mod.parameters() if p.grad is not None and p.grad.abs().sum() > 0]
    assert gat_grads, "multiplicative GAT got no gradients"


def test_rw_weighted_biases_toward_heavy_edges():
    """Weighted RW should take the higher-weight edge more often than 50%."""
    import torch
    from src.ml.vengnn_precompute import sample_random_walks
    # Graph: node 0 has two outgoing edges: 0->1 weight 99, 0->2 weight 1.
    ei = torch.tensor([[0, 0], [1, 2]], dtype=torch.long)
    ew = torch.tensor([99.0, 1.0])
    walks = sample_random_walks(
        torch.tensor([0]), ei, ew,
        num_nodes=3, walks_per_node=500, walk_length=2,
        seed=0, weighted=True,
    )
    # Almost all walks should hit node 1.
    frac_to_1 = (walks[0, :, 1] == 1).float().mean().item()
    assert frac_to_1 > 0.9, f"weighted RW should prefer heavy edges; got {frac_to_1:.2f}"


def test_rw_uniform_gives_roughly_equal_split():
    import torch
    from src.ml.vengnn_precompute import sample_random_walks
    ei = torch.tensor([[0, 0], [1, 2]], dtype=torch.long)
    ew = torch.tensor([99.0, 1.0])  # weights ignored in uniform mode
    walks = sample_random_walks(
        torch.tensor([0]), ei, ew,
        num_nodes=3, walks_per_node=1000, walk_length=2,
        seed=0, weighted=False,
    )
    frac_to_1 = (walks[0, :, 1] == 1).float().mean().item()
    assert 0.4 < frac_to_1 < 0.6, f"uniform RW should be ~50/50; got {frac_to_1:.2f}"


def test_pathcount_edge_weights_are_used_by_gat():
    # Sanity: passing different edge_weights yields different outputs.
    model, g = _build()
    # Baseline forward with original weights.
    model.precompute(g.x_dict, g.edge_index_dict)
    out_a = model(g.x_dict, g.edge_index_dict)["embedding"]["startup"].detach()
    # Replace weights with a scaled copy.
    new_weights = {mp: g[mp].edge_weight * 100.0 for mp in model.metapaths}
    model.set_metapath_edge_weights(new_weights)
    out_b = model(g.x_dict, g.edge_index_dict)["embedding"]["startup"].detach()
    assert not torch.allclose(out_a, out_b, atol=1e-5), "GAT output did not change with different edge weights"
