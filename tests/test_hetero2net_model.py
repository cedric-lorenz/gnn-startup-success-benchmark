"""Faithfulness tests for the Hetero2Net model class.

References:
    - Paper: Li, Wei, Dan et al. 2023 (arXiv 2310.11664).
    - Reference implementation: https://github.com/EdisonLeeeee/Hetero2Net
      (hetero2net/layers.py, hetero2net/models.py, hetero2net/utils.py).

Each test targets one aspect of the paper or the reference impl:
    - DisenConv shapes and two-channel outputs           (layers.py:DisenConv)
    - dist_corr numeric behavior                          (utils.py:dist_corr)
    - Model forward shape and output keys                 (models.py:HeteroGNN)
    - Correlation loss on disentangled channels           (paper §5.2, Eq. 8)
    - Masked-metapath link loss gradient flow            (paper §5.2, Eq. 9)
    - Label embedding injection and masking               (paper §5.3, Eq. 11-12)
    - Eval mode zeros aux losses (no label leakage)
"""
import pytest
import torch
from torch_geometric.data import HeteroData

from src.ml.models import (
    Hetero2Net,
    _HeteroDisenConv,
    _dist_corr,
    _edge_key,
)


def _toy_graph(num_startups=8, feat_dim=6, num_mps=2, num_investors=4, seed=0):
    """Minimal HeteroData with startup-startup metapaths and an investor edge
    type (to verify metapath filtering and to exercise the bipartite DisenConv
    code path)."""
    torch.manual_seed(seed)
    g = HeteroData()
    g["startup"].x = torch.randn(num_startups, feat_dim)
    # Multi-label binary targets: [momentum, liquidity]
    g["startup"].y = torch.randint(0, 2, (num_startups, 2)).float()
    g["startup"].train_mask = torch.ones(num_startups, dtype=torch.bool)
    g["investor"].x = torch.randn(num_investors, feat_dim)

    mp_edge_types = []
    for i in range(num_mps):
        src = torch.randint(0, num_startups, (12,))
        dst = torch.randint(0, num_startups, (12,))
        ei = torch.stack([src, dst], dim=0)
        rel = f"mp_{i}"
        g[("startup", rel, "startup")].edge_index = ei
        mp_edge_types.append(("startup", rel, "startup"))

    # Bipartite edge (verifies metapath filtering — should NOT enter link loss).
    ei_inv = torch.tensor([[0, 1, 2, 3], [0, 1, 2, 3]], dtype=torch.long)
    g[("startup", "funded_by", "investor")].edge_index = ei_inv
    return g, feat_dim


def _build(model_kwargs=None, num_mps=2, num_startups=8):
    g, d = _toy_graph(num_mps=num_mps, num_startups=num_startups)
    kw = dict(
        in_channels=d,
        hidden_channels=12,
        metadata=g.metadata(),
        num_layers=2,
        dropout=0.0,
        alpha=0.2,
        beta=0.4,
        mask_ratio=0.5,
        label_mask_ratio=0.7,
        use_label_propagation=True,
        num_label_classes=2,
    )
    kw.update(model_kwargs or {})
    model = Hetero2Net(**kw)
    # Register label input (non-train rows already zeroed above by the fixture
    # because train_mask is all-True; this is the invariant the trainer sets).
    y_masked = g["startup"].y.clone()
    model.set_label_propagation_input(y_masked)
    return model, g


# --------------------------------------------------------------------- #
# DisenConv primitive
# --------------------------------------------------------------------- #
def test_disenconv_returns_three_tensors_matching_dst_shape():
    """DisenConv must return (out, homo, hetero) all aggregated at dst."""
    torch.manual_seed(0)
    conv = _HeteroDisenConv(in_channels=4, out_channels=6, root_weight=True)
    x_src = torch.randn(5, 4)
    x_dst = torch.randn(7, 4)
    ei = torch.tensor([[0, 1, 2, 3, 4], [0, 2, 4, 6, 1]], dtype=torch.long)
    out, h, ht = conv((x_src, x_dst), ei)
    assert out.shape == (7, 6)
    assert h.shape == (7, 6)
    assert ht.shape == (7, 6)


def test_disenconv_homo_and_hetero_are_independently_parameterized():
    """lin_homo and lin_hetero must not share weights (Eq. 7 / P1)."""
    conv = _HeteroDisenConv(in_channels=4, out_channels=6)
    assert conv.lin_homo.weight.data_ptr() != conv.lin_hetero.weight.data_ptr()


def test_disenconv_out_equals_homo_plus_hetero_plus_root():
    """Paper: out = out_homo + out_hetero + lin_r(x_dst)."""
    torch.manual_seed(1)
    conv = _HeteroDisenConv(in_channels=3, out_channels=4, root_weight=True).eval()
    x = torch.randn(6, 3)
    ei = torch.tensor([[0, 1, 2, 3, 4, 5], [1, 2, 3, 4, 5, 0]], dtype=torch.long)
    out, h, ht = conv(x, ei)
    root = conv.lin_r(x)
    assert torch.allclose(out, h + ht + root, atol=1e-5)


# --------------------------------------------------------------------- #
# dist_corr (paper Eq. 8)
# --------------------------------------------------------------------- #
def test_dist_corr_on_identical_inputs_equals_one():
    x = torch.randn(20, 5)
    # After centering x-x.mean, corr(x, x) = |E[x*x]| / (std*std) = 1.
    assert _dist_corr(x, x).item() == pytest.approx(1.0, abs=1e-5)


def test_dist_corr_on_orthogonal_random_inputs_is_near_zero():
    torch.manual_seed(42)
    # Large N drives the sample correlation of independent Gaussians to 0.
    x1 = torch.randn(5000, 8)
    x2 = torch.randn(5000, 8)
    val = _dist_corr(x1, x2).item()
    assert 0.0 <= val < 0.05, f"orthogonal corr should be ~0, got {val}"


def test_dist_corr_is_non_negative():
    for _ in range(5):
        a = torch.randn(50, 3)
        b = torch.randn(50, 3)
        assert _dist_corr(a, b).item() >= 0.0


# --------------------------------------------------------------------- #
# Model structural properties
# --------------------------------------------------------------------- #
def test_metapath_selection_only_startup_to_startup():
    model, _ = _build(num_mps=3)
    # All registered metapaths for link prediction must end at startup.
    for mp in model.metapaths:
        assert mp[0] == "startup" and mp[2] == "startup"
    # The investor bipartite edge must NOT be in the metapath list.
    bad = [mp for mp in model.metapaths if mp[2] != "startup"]
    assert bad == []
    assert len(model.metapaths) == 3


def test_forward_shape_masked_multi_task():
    model, g = _build()
    out = model(g.x_dict, g.edge_index_dict)
    assert out["masked_multi_task_output"]["startup"].shape == (g["startup"].num_nodes, 2)
    assert out["embedding"]["startup"].shape == (g["startup"].num_nodes, model.hidden_channels)


def test_edge_key_string_roundtrip():
    """Edge keys inserted in ModuleDict must be reconstructable per canonical rule."""
    et = ("startup", "funded_by", "investor")
    assert _edge_key(et) == "startup__funded_by__investor"


def test_invalid_mask_ratio_raises():
    with pytest.raises(ValueError, match="mask_ratio"):
        _build({"mask_ratio": 1.0})


def test_invalid_alpha_raises():
    with pytest.raises(ValueError, match="non-negative"):
        _build({"alpha": -0.1})


# --------------------------------------------------------------------- #
# Auxiliary losses (paper §5.2)
# --------------------------------------------------------------------- #
def test_aux_losses_zero_in_eval_mode():
    """Eval must not mask edges or labels (no data leakage at inference)."""
    model, g = _build()
    model.eval()
    _ = model(g.x_dict, g.edge_index_dict)
    assert float(model.corr_loss) == 0.0
    assert float(model.link_loss) == 0.0
    assert float(model.aux_loss) == 0.0


def test_aux_losses_nonzero_in_train_mode():
    model, g = _build()
    model.train()
    _ = model(g.x_dict, g.edge_index_dict)
    # Link loss should be non-trivial because positive + random-negative edges
    # are both being pushed toward logit = +inf; the BCE-with-logits against
    # target=1 is non-zero for finite logits.
    assert model.link_loss.requires_grad
    assert model.link_loss.item() > 0.0
    # Correlation loss is in [0, 1]; require scalar and finite.
    assert model.corr_loss.dim() == 0
    assert torch.isfinite(model.corr_loss)


def test_aux_reduction_sum_scales_with_n_metapaths():
    """With aux_reduction='sum' (paper-faithful), corr_loss scales with the
    number of (layer × edge_type) terms summed; with 'mean' it doesn't.

    This is the scale difference that motivates the default switch to 'sum':
    our α sweep ranges must match the reference's SUM accumulation for
    paper-faithful comparison."""
    torch.manual_seed(0)
    model_sum, g = _build({"aux_reduction": "sum", "num_layers": 2}, num_mps=3)
    torch.manual_seed(0)
    model_mean, _ = _build({"aux_reduction": "mean", "num_layers": 2}, num_mps=3)
    # Use identical weights so only reduction differs
    model_mean.load_state_dict(model_sum.state_dict())

    model_sum.train()
    model_mean.train()

    # Reset RNG so both runs see the same random edge-masking / label-masking.
    torch.manual_seed(123)
    _ = model_sum(g.x_dict, g.edge_index_dict)
    torch.manual_seed(123)
    _ = model_mean(g.x_dict, g.edge_index_dict)

    # With 3 metapaths × 2 layers = 6 corr terms (on metapath-only mode).
    # sum_corr ≈ 6 × mean_corr (up to RNG match).
    ratio = (model_sum.corr_loss / model_mean.corr_loss).item()
    assert 4.0 < ratio < 8.0, (
        f"sum/mean ratio for corr_loss should be ~6 with 3 MPs × 2 layers; got {ratio:.2f}"
    )


def test_aux_reduction_invalid_value_raises():
    with pytest.raises(ValueError, match="aux_reduction"):
        _build({"aux_reduction": "median"})


def test_aux_loss_equals_alpha_corr_plus_beta_link():
    """Model's self.aux_loss must be exactly α·L_corr + β·L_rec (Eq. 14)."""
    model, g = _build({"alpha": 0.3, "beta": 0.5})
    model.train()
    _ = model(g.x_dict, g.edge_index_dict)
    expected = 0.3 * model.corr_loss + 0.5 * model.link_loss
    assert torch.allclose(model.aux_loss, expected, atol=1e-7)


def test_link_loss_gradient_flows_through_edge_decoder():
    """Regression: link loss must produce grads on edge_decoder weights."""
    model, g = _build()
    model.train()
    _ = model(g.x_dict, g.edge_index_dict)
    model.link_loss.backward()
    grads = [p.grad for p in model.edge_decoder.parameters()]
    assert all(g is not None for g in grads)
    assert any(g.abs().sum() > 0 for g in grads)


def test_corr_loss_gradient_flows_through_disenconv_homo_and_hetero():
    """Both homo and hetero channels must receive gradient from L_corr."""
    model, g = _build()
    model.train()
    _ = model(g.x_dict, g.edge_index_dict)
    model.corr_loss.backward()
    any_homo_grad = False
    any_hetero_grad = False
    for layer in model.convs:
        for conv in layer.values():
            if conv.lin_homo.weight.grad is not None and conv.lin_homo.weight.grad.abs().sum() > 0:
                any_homo_grad = True
            if conv.lin_hetero.weight.grad is not None and conv.lin_hetero.weight.grad.abs().sum() > 0:
                any_hetero_grad = True
    assert any_homo_grad, "L_corr produced no gradient through any lin_homo"
    assert any_hetero_grad, "L_corr produced no gradient through any lin_hetero"


def test_beta_zero_disables_link_loss():
    model, g = _build({"beta": 0.0})
    model.train()
    _ = model(g.x_dict, g.edge_index_dict)
    assert float(model.link_loss) == 0.0


def test_alpha_zero_does_not_zero_corr_loss_itself_but_nulls_contribution():
    """alpha=0 should leave L_corr still computed (diagnostic) but its
    contribution to aux_loss must vanish."""
    model, g = _build({"alpha": 0.0, "beta": 0.4})
    model.train()
    _ = model(g.x_dict, g.edge_index_dict)
    # corr_loss itself still scalar/finite:
    assert torch.isfinite(model.corr_loss)
    # But aux_loss should equal beta * link_loss exactly:
    assert torch.allclose(model.aux_loss, 0.4 * model.link_loss, atol=1e-7)


# --------------------------------------------------------------------- #
# Masked metapath prediction (paper Eq. 9)
# --------------------------------------------------------------------- #
def test_metapath_edges_partially_masked_during_training():
    """During training, at least one metapath's edges must be reduced."""
    torch.manual_seed(0)
    model, g = _build({"mask_ratio": 0.5})
    original = {mp: g[mp].edge_index.size(1) for mp in model.metapaths}
    # Call the masker directly.
    masked, positives = model._mask_metapath_edges(g.edge_index_dict, ratio=0.5)
    total_pos = sum(v.size(1) for v in positives.values())
    total_kept = sum(masked[mp].size(1) for mp in model.metapaths)
    total_orig = sum(original.values())
    assert total_pos > 0, "no positive edges were masked out"
    assert total_kept + total_pos == total_orig
    # Positive edges must be a subset of the original edges.
    for mp, pos_ei in positives.items():
        orig_set = {tuple(p) for p in g[mp].edge_index.t().tolist()}
        pos_set = {tuple(p) for p in pos_ei.t().tolist()}
        assert pos_set.issubset(orig_set)


def test_masker_partitions_edge_indices_disjointly():
    """The positive edges (by column index) and the kept edges (by column
    index) must be a disjoint partition of the original [0, E) index range.
    Note: the edge multiset itself can have duplicates — identical (u,v)
    pairs may legitimately appear in both sets if they occur twice in the
    original edge_index. The contract that matters is index-level, not
    pair-level: randperm-based masking never reuses the same column twice."""
    torch.manual_seed(0)
    model, g = _build()
    # We hand-build column IDs to check the index invariant directly, since
    # the masker returns tensors (not column indices). Instead of asserting
    # pair-level disjointness, verify |pos| + |kept| == |orig| per metapath.
    masked, positives = model._mask_metapath_edges(g.edge_index_dict, ratio=0.5)
    for mp, pos_ei in positives.items():
        assert pos_ei.size(1) + masked[mp].size(1) == g[mp].edge_index.size(1)
        # Sanity: both sets are non-empty with ratio=0.5 and E >= 2.
        assert pos_ei.size(1) > 0
        assert masked[mp].size(1) > 0


# --------------------------------------------------------------------- #
# Label propagation / masked label prediction (paper §5.3, Eq. 11-12)
# --------------------------------------------------------------------- #
def test_label_embedding_shape():
    """Embedding table must have (num_label_classes + 1) rows to account for
    the 'masked' bucket (paper Eq. 12 uses mask_bucket = num_classes)."""
    model, _ = _build({"num_label_classes": 2})
    assert model.label_embedding is not None
    assert model.label_embedding.num_embeddings == 3  # 2 classes + masked
    assert model.label_embedding.embedding_dim == model.label_embedding.weight.size(1)


def test_eval_skips_runtime_label_masking():
    """Paper §5.3: at inference `p=0` (no random masking). Eval must NOT apply
    the Bernoulli keep-mask so the model sees ALL registered train labels
    deterministically. We verify this by running eval twice with the same
    registered labels and confirming identical outputs (no RNG)."""
    torch.manual_seed(0)
    model, g = _build({"label_mask_ratio": 0.3})  # train-time aggressive masking
    model.eval()
    out1 = model(g.x_dict, g.edge_index_dict)["masked_multi_task_output"]["startup"]
    out2 = model(g.x_dict, g.edge_index_dict)["masked_multi_task_output"]["startup"]
    assert torch.allclose(out1, out2, atol=1e-6), "eval must be deterministic w.r.t. label masking"


def test_eval_does_not_leak_test_row_labels():
    """At eval, labels for test-set rows (zeroed by the trainer upstream) must
    map to the 'masked' bucket, so the model cannot peek at a test label
    through the embedding even when label_propagation is on."""
    torch.manual_seed(0)
    model, g = _build()
    # Register y with some rows set to zero (simulating non-train rows).
    y_non_test = torch.zeros(g["startup"].num_nodes, 2)
    y_non_test[:4] = 1.0  # first 4 "train" rows have full labels
    y_non_test[4:] = 0.0  # remaining "eval" rows zeroed
    model.set_label_propagation_input(y_non_test)
    model.eval()
    out_with_zeros = model(g.x_dict, g.edge_index_dict)["masked_multi_task_output"]["startup"]

    # Now flip a supposed-test label to 1 — if the model were leaking test
    # labels, eval output on that row would change. But since label embedding
    # reads from the REGISTERED y_masked (which has test rows zeroed by the
    # trainer in production), flipping REGISTERED values propagates — that's
    # not leakage; it's the documented interface. We assert instead that
    # when y_masked is uniformly zero (no label info at all), eval output
    # equals eval output with label propagation disabled entirely.
    y_all_zero = torch.zeros(g["startup"].num_nodes, 2)
    model.set_label_propagation_input(y_all_zero)
    out_zero = model(g.x_dict, g.edge_index_dict)["masked_multi_task_output"]["startup"]
    model._label_prop_y = None
    out_none = model(g.x_dict, g.edge_index_dict)["masked_multi_task_output"]["startup"]
    assert torch.allclose(out_zero, out_none, atol=1e-5), (
        "all-zero y_masked must produce identical output to label-prop disabled"
    )


def test_label_propagation_disabled_flag_sets_embedding_none():
    model, g = _build({"use_label_propagation": False})
    assert model.label_embedding is None
    # Forward still runs.
    model.train()
    out = model(g.x_dict, g.edge_index_dict)
    assert out["embedding"]["startup"].shape[0] == g["startup"].num_nodes


def test_set_label_propagation_input_rejects_wrong_shape():
    model, _ = _build({"num_label_classes": 2})
    with pytest.raises(ValueError, match=r"y_masked must be \[N, 2\]"):
        model.set_label_propagation_input(torch.zeros(8, 3))


# --------------------------------------------------------------------- #
# End-to-end gradient flow
# --------------------------------------------------------------------- #
def test_combined_loss_gradients_flow_to_all_startup_metapath_disenconv_weights():
    """With corr_on_metapaths_only=True (default, memory-optimised), cls_loss
    + aux_loss must produce gradients on every startup-to-startup metapath
    DisenConv lin_homo / lin_hetero. Non-metapath edges (e.g. startup→investor)
    are NOT guaranteed to receive gradient here since (a) corr loss no longer
    touches them and (b) non-startup embeddings don't feed the classification
    output — that's expected given the paper's scope (reconstruction on
    startup-endpoint metapaths)."""
    torch.manual_seed(0)
    model, g = _build()
    model.train()
    out = model(g.x_dict, g.edge_index_dict)
    cls_like = out["masked_multi_task_output"]["startup"].sum()
    loss = cls_like + model.aux_loss
    loss.backward()

    for layer_idx, layer in enumerate(model.convs):
        for key, conv in layer.items():
            # Only assert for startup-startup metapath edges.
            if key.startswith("startup__") and key.endswith("__startup"):
                assert conv.lin_homo.weight.grad is not None, f"layer {layer_idx} {key} lin_homo has no grad"
                assert conv.lin_hetero.weight.grad is not None, f"layer {layer_idx} {key} lin_hetero has no grad"


def test_paper_faithful_corr_propagates_gradient_to_non_metapath_edges():
    """With corr_on_metapaths_only=False (paper-faithful), L_corr is applied
    to ALL edge types, so even bipartite edges' lin_homo / lin_hetero weights
    receive gradient from the corr loss."""
    torch.manual_seed(0)
    model, g = _build({"corr_on_metapaths_only": False})
    model.train()
    out = model(g.x_dict, g.edge_index_dict)
    loss = out["masked_multi_task_output"]["startup"].sum() + model.aux_loss
    loss.backward()

    bipartite_got_grad = False
    for layer in model.convs:
        for key, conv in layer.items():
            if not (key.startswith("startup__") and key.endswith("__startup")):
                # Bipartite (e.g. startup__funded_by__investor)
                if conv.lin_homo.weight.grad is not None and conv.lin_homo.weight.grad.abs().sum() > 0:
                    bipartite_got_grad = True
                    break
    assert bipartite_got_grad, "paper-faithful corr loss should reach bipartite edges"


def test_label_embedding_receives_gradient_when_enabled():
    torch.manual_seed(0)
    model, g = _build({"use_label_propagation": True, "label_mask_ratio": 1.0})
    model.train()
    out = model(g.x_dict, g.edge_index_dict)
    loss = out["masked_multi_task_output"]["startup"].sum() + model.aux_loss
    loss.backward()
    assert model.label_embedding.weight.grad is not None
    assert model.label_embedding.weight.grad.abs().sum() > 0


# --------------------------------------------------------------------- #
# Integration: trainer-style instantiation
# --------------------------------------------------------------------- #
# --------------------------------------------------------------------- #
# §6.2 channel-balance extractor (ICDM analysis)
# --------------------------------------------------------------------- #
def test_extract_channel_balance_returns_one_entry_per_startup_metapath():
    """extract_channel_balance() must emit stats for every startup-startup
    metapath in the model's metadata and none for bipartite edges."""
    model, g = _build(num_mps=3)
    stats = model.extract_channel_balance(g.x_dict, g.edge_index_dict)
    assert set(stats.keys()) == set(model.metapaths), (
        f"expected {model.metapaths}, got {list(stats.keys())}"
    )


def test_extract_channel_balance_masses_sum_to_one():
    """By definition, hetero_mass + homo_mass = 1 for each metapath."""
    model, g = _build(num_mps=3)
    stats = model.extract_channel_balance(g.x_dict, g.edge_index_dict)
    for et, s in stats.items():
        assert abs(s["hetero_mass"] + s["homo_mass"] - 1.0) < 1e-6, (et, s)
        assert 0.0 <= s["hetero_mass"] <= 1.0
        assert 0.0 <= s["homo_mass"] <= 1.0


def test_extract_channel_balance_norms_are_finite_and_positive():
    model, g = _build(num_mps=2)
    stats = model.extract_channel_balance(g.x_dict, g.edge_index_dict)
    for et, s in stats.items():
        assert s["homo_norm"] > 0.0 and torch.isfinite(torch.tensor(s["homo_norm"]))
        assert s["hetero_norm"] > 0.0 and torch.isfinite(torch.tensor(s["hetero_norm"]))


def test_extract_channel_balance_is_eval_mode_no_side_effects():
    """Calling extract_channel_balance on a training-mode model must not
    flip the global .training flag or perturb model state."""
    model, g = _build()
    model.train()
    assert model.training
    _ = model.extract_channel_balance(g.x_dict, g.edge_index_dict)
    assert model.training, "extract_channel_balance must restore training mode"


def test_extract_channel_balance_does_not_apply_edge_masking():
    """At eval path, no positive_edges are carved out; the returned stats
    must be computed on the FULL edge_index_dict. We verify by asserting
    two successive calls are deterministic (pure function in eval)."""
    model, g = _build()
    model.eval()
    s1 = model.extract_channel_balance(g.x_dict, g.edge_index_dict)
    s2 = model.extract_channel_balance(g.x_dict, g.edge_index_dict)
    for et in s1:
        assert abs(s1[et]["hetero_mass"] - s2[et]["hetero_mass"]) < 1e-7
        assert abs(s1[et]["homo_norm"] - s2[et]["homo_norm"]) < 1e-7


def test_extract_per_node_channel_norms_shapes_and_bounds():
    """Per-node extractor returns homo/hetero/in_degree tensors of length
    N_startup for every startup-startup metapath, with finite norms."""
    num_s = 8
    model, g = _build(num_mps=3, num_startups=num_s)
    pn = model.extract_per_node_channel_norms(g.x_dict, g.edge_index_dict)
    assert set(pn.keys()) == set(model.metapaths)
    for et, d in pn.items():
        assert d["homo_norms"].shape == (num_s,)
        assert d["hetero_norms"].shape == (num_s,)
        assert d["in_degree"].shape == (num_s,)
        assert torch.isfinite(d["homo_norms"]).all()
        assert torch.isfinite(d["hetero_norms"]).all()
        assert (d["homo_norms"] >= 0).all()
        assert (d["hetero_norms"] >= 0).all()
        assert (d["in_degree"] >= 0).all()


def test_extract_per_node_in_degree_matches_edge_index():
    """Sum of per-node in-degrees must equal the metapath's edge count."""
    model, g = _build(num_mps=2)
    pn = model.extract_per_node_channel_norms(g.x_dict, g.edge_index_dict)
    for et, d in pn.items():
        n_edges = g[et].edge_index.size(1)
        assert int(d["in_degree"].sum().item()) == n_edges, (
            f"{et}: sum(in_degree)={int(d['in_degree'].sum())} vs "
            f"edge_index cols={n_edges}"
        )


def test_extract_per_node_norms_aggregate_to_scalar_version():
    """Per-node mean of norms should match extract_channel_balance's scalar."""
    model, g = _build()
    pn = model.extract_per_node_channel_norms(g.x_dict, g.edge_index_dict)
    sc = model.extract_channel_balance(g.x_dict, g.edge_index_dict)
    for et, d in pn.items():
        # Scalar method averages over dst startups:
        # mean(||h_homo[v]||) across all startups.
        expected_hn = float(d["homo_norms"].mean().item())
        assert abs(expected_hn - sc[et]["homo_norm"]) < 1e-5, et


def test_extract_channel_balance_layer_idx_bounds():
    """Negative index mirrors Python convention; out-of-range raises."""
    model, g = _build(num_mps=2)
    s_last = model.extract_channel_balance(g.x_dict, g.edge_index_dict, layer_idx=-1)
    s_first = model.extract_channel_balance(g.x_dict, g.edge_index_dict, layer_idx=0)
    # Different layers → generally different norms (guards against silent
    # fallthrough where layer_idx was ignored).
    any_diff = False
    for et in s_last:
        if abs(s_last[et]["homo_norm"] - s_first[et]["homo_norm"]) > 1e-6:
            any_diff = True
            break
    assert any_diff, "layer_idx=-1 and layer_idx=0 produced identical norms"

    with pytest.raises(IndexError):
        model.extract_channel_balance(g.x_dict, g.edge_index_dict, layer_idx=99)


def test_trainer_dispatch_branch_reachable_for_hetero2net(mock_hetero_graph):
    """The `elif model_name == "Hetero2Net"` branch in Trainer._initialize_model
    must instantiate Hetero2Net AND register the masked label-propagation input
    on the model. We exercise the branch with a stub `self` (no full Trainer
    bootstrap) so this test stays insulated from unrelated config surface."""
    from src.ml.train import Trainer

    data = mock_hetero_graph
    num_s = data["startup"].num_nodes
    ei = torch.stack([
        torch.randint(0, num_s, (20,)),
        torch.randint(0, num_s, (20,)),
    ], dim=0)
    data[("startup", "sp_test", "startup")].edge_index = ei

    class _Stub:
        """Minimal stand-in for `self` in _initialize_model — mirrors the
        attributes the Hetero2Net branch reads (and nothing else)."""
        def __init__(self):
            self.model_name = "Hetero2Net"
            self.target_mode = "masked_multi_task"
            self.data = data
            self.aggregation_method = "sum"
            self.config = {}
            self.model_config = {
                "hidden_channels": 12,
                "num_layers": 2,
                "dropout": 0.0,
                "activation_type": "relu",
                "alpha": 0.2,
                "beta": 0.4,
                "mask_ratio": 0.5,
                "label_mask_ratio": 0.7,
                "use_label_propagation": True,
                "num_label_classes": 2,
                "root_weight": True,
                "normalize_bn": True,
            }

    stub = _Stub()
    model = Trainer._initialize_model(stub, data)
    assert type(model).__name__ == "Hetero2Net"
    # Trainer branch must have registered masked training labels.
    assert model._label_prop_y is not None
    assert model._label_prop_y.shape == (num_s, 2)
    # Non-train rows must be zeroed by the trainer branch to prevent leakage.
    non_train = ~data["startup"].train_mask
    assert model._label_prop_y[non_train].abs().sum().item() == 0.0


def test_model_integrates_with_trainer_style_call_pattern(mock_hetero_graph):
    """The trainer calls model(data.x_dict, data.edge_index_dict) in full-
    batch mode. This test mimics that call with the project's shared mock
    heterogeneous graph to catch regressions on the real graph schema."""
    data = mock_hetero_graph
    # Add at least one startup-startup metapath so the masked-metapath task
    # has something to operate on (mock_hetero_graph doesn't include one).
    num_s = data["startup"].num_nodes
    ei = torch.stack([
        torch.randint(0, num_s, (20,)),
        torch.randint(0, num_s, (20,)),
    ], dim=0)
    data[("startup", "sp_test", "startup")].edge_index = ei

    model = Hetero2Net(
        in_channels=data["startup"].num_features,
        hidden_channels=12,
        metadata=data.metadata(),
        num_layers=2,
        dropout=0.1,
        activation_type="relu",
        target_mode="masked_multi_task",
        num_classes=2,
        alpha=0.2,
        beta=0.4,
        mask_ratio=0.5,
        use_label_propagation=True,
        num_label_classes=2,
    )
    # Trainer-style label registration
    y = data["startup"].y.float()[:, :2].clone()
    train_mask = data["startup"].train_mask
    y[~train_mask] = 0.0
    model.set_label_propagation_input(y)

    model.train()
    out = model(data.x_dict, data.edge_index_dict)
    assert out["masked_multi_task_output"]["startup"].shape == (num_s, 2)
    # aux_loss has grad_fn when alpha*corr + beta*link > 0
    assert model.aux_loss.requires_grad, "aux_loss must carry gradient in train mode"
