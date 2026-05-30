"""Faithfulness tests for the Simple-HGN model class.

Reference implementations:
    - Lv et al., "Are we really making much progress? Revisiting, benchmarking,
      and refining heterogeneous graph neural networks" (KDD 2021). Original
      DGL code at https://github.com/THUDM/HGB (NC/benchmark/methods/baseline).
    - Lin et al., H^2GB (KDD 2025) PyG port at
      https://github.com/junhongmit/H2GB/blob/main/H2GB/network/shgn_model.py (MIT).

The key algorithmic ingredients verified below:
    - Edge-type-aware attention logit: leaky_relu(<h_src, a_l> + <h_dst, a_r> + <E[t], a_e>)
    - Softmax per destination node over incoming edges
    - Cross-layer residual attention blend: alpha = alpha * (1-beta) + alpha_prev * beta
    - Per-layer feature residual connection (optional)
    - L2 normalization on the final embedding (paper signature)
    - Output layer: mean across heads (not flatten) with res_attn=None
"""
import pytest
import torch

from src.ml.models import SimpleHGN, SimpleHGNConv


# --------------------------------------------------------------------- #
# Fixtures / helpers
# --------------------------------------------------------------------- #
def _toy_hetero(seed: int = 0):
    """Minimal hetero graph with 3 node types and 5 edge types (forward + reverse
    for the bipartite edges, so every non-startup node type has a path into the
    startup-node loss surface — needed for gradient-flow tests)."""
    torch.manual_seed(seed)
    x_dict = {
        "startup": torch.randn(6, 8),
        "investor": torch.randn(4, 10),
        "city": torch.randn(3, 5),
    }
    edge_index_dict = {
        ("investor", "funds", "startup"): torch.tensor(
            [[0, 1, 2, 3, 0], [0, 1, 2, 3, 4]], dtype=torch.long,
        ),
        ("startup", "rev_funds", "investor"): torch.tensor(
            [[0, 1, 2, 3, 4], [0, 1, 2, 3, 0]], dtype=torch.long,
        ),
        ("startup", "located_in", "city"): torch.tensor(
            [[0, 1, 2, 3, 4, 5], [0, 1, 2, 0, 1, 2]], dtype=torch.long,
        ),
        ("city", "rev_located_in", "startup"): torch.tensor(
            [[0, 1, 2, 0, 1, 2], [0, 1, 2, 3, 4, 5]], dtype=torch.long,
        ),
        ("startup", "similar_to", "startup"): torch.tensor(
            [[0, 1, 2, 3, 4], [1, 2, 3, 4, 5]], dtype=torch.long,
        ),
    }
    metadata = (["startup", "investor", "city"], list(edge_index_dict.keys()))
    in_channels = {"startup": 8, "investor": 10, "city": 5}
    return x_dict, edge_index_dict, metadata, in_channels


# --------------------------------------------------------------------- #
# Conv-level faithfulness
# --------------------------------------------------------------------- #
def test_conv_alpha_one_yields_pure_residual_attention():
    """With blend weight alpha=1.0, the returned attention must equal the
    provided res_attn (paper Eq.: alpha = alpha*(1-beta) + alpha_prev*beta,
    where beta is the constructor's `alpha`)."""
    torch.manual_seed(0)
    conv = SimpleHGNConv(
        in_channels=8, out_channels=4, edge_channels=4,
        num_etypes=2, num_heads=2, alpha=1.0,
        attn_drop=0.0, feat_drop=0.0,
    )
    conv.eval()
    x = torch.randn(4, 8)
    edge_index = torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long)
    edge_type = torch.tensor([0, 1, 0], dtype=torch.long)
    res = torch.full((3, 2, 1), 0.42)
    with torch.no_grad():
        _, alpha_out = conv(x, edge_index, edge_type, res_attn=res)
    assert torch.allclose(alpha_out, res, atol=1e-5)


def test_conv_alpha_zero_ignores_residual_attention():
    """With alpha=0.0 the residual term is ignored regardless of res_attn."""
    torch.manual_seed(0)
    conv = SimpleHGNConv(
        in_channels=8, out_channels=4, edge_channels=4,
        num_etypes=2, num_heads=2, alpha=0.0,
        attn_drop=0.0, feat_drop=0.0,
    )
    conv.eval()
    x = torch.randn(4, 8)
    edge_index = torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long)
    edge_type = torch.tensor([0, 1, 0], dtype=torch.long)
    with torch.no_grad():
        _, alpha_none = conv(x, edge_index, edge_type, res_attn=None)
        _, alpha_fake = conv(x, edge_index, edge_type, res_attn=torch.full((3, 2, 1), 0.7))
    assert torch.allclose(alpha_none, alpha_fake, atol=1e-5)


def test_conv_softmax_per_destination_sums_to_one():
    """Attention weights must sum to 1.0 per destination node, per head —
    this is the PyG-softmax-keyed-on-col invariant of Simple-HGN."""
    torch.manual_seed(0)
    conv = SimpleHGNConv(
        in_channels=8, out_channels=4, edge_channels=4,
        num_etypes=2, num_heads=2, attn_drop=0.0, feat_drop=0.0,
    )
    conv.eval()
    x = torch.randn(4, 8)
    # Destination 1 has 2 incoming edges; others have 1.
    edge_index = torch.tensor([[0, 2, 1], [1, 1, 2]], dtype=torch.long)
    edge_type = torch.tensor([0, 1, 0], dtype=torch.long)
    with torch.no_grad():
        _, alpha = conv(x, edge_index, edge_type, res_attn=None)
    per_head_dst1 = alpha[[0, 1], :, 0].sum(dim=0)  # sum over 2 edges to dst=1
    assert torch.allclose(per_head_dst1, torch.ones(2), atol=1e-5)


def test_conv_edge_type_embedding_differentiates_relations():
    """Two edges with the same src/dst but different edge types must produce
    different attention logits — proves the edge-type embedding enters the
    attention computation (paper's E[t] injection)."""
    torch.manual_seed(0)
    conv = SimpleHGNConv(
        in_channels=8, out_channels=4, edge_channels=4,
        num_etypes=3, num_heads=2, attn_drop=0.0, feat_drop=0.0, alpha=0.0,
    )
    conv.eval()
    x = torch.randn(3, 8)
    ei = torch.tensor([[0, 0], [1, 1]], dtype=torch.long)  # same src/dst, 2 edges
    with torch.no_grad():
        _, a_t0 = conv(x, ei, torch.tensor([0, 0]), res_attn=None)
        _, a_t1 = conv(x, ei, torch.tensor([1, 2]), res_attn=None)
    # At least one head should give a distinct softmax split for t=(0,0) vs t=(1,2).
    assert not torch.allclose(a_t0, a_t1, atol=1e-4), \
        "edge-type embedding does not influence attention"


def test_conv_residual_none_when_disabled_and_identity_when_shape_matches():
    """When residual=False, res_fc is None. When residual=True and in==out*heads,
    res_fc must be nn.Identity (no parameters). These are the two cases that
    show up in the first/subsequent layers of SimpleHGN."""
    c_off = SimpleHGNConv(in_channels=16, out_channels=4, edge_channels=4,
                          num_etypes=1, num_heads=4, residual=False)
    assert c_off.res_fc is None

    c_on_identity = SimpleHGNConv(in_channels=16, out_channels=4, edge_channels=4,
                                  num_etypes=1, num_heads=4, residual=True)
    assert isinstance(c_on_identity.res_fc, torch.nn.Identity)

    c_on_linear = SimpleHGNConv(in_channels=12, out_channels=4, edge_channels=4,
                                num_etypes=1, num_heads=4, residual=True)
    assert isinstance(c_on_linear.res_fc, torch.nn.Linear)


# --------------------------------------------------------------------- #
# SimpleHGN model: shapes + heads
# --------------------------------------------------------------------- #
@pytest.mark.parametrize("num_layers", [1, 2, 3])
def test_model_forward_shapes_masked_multi_task(num_layers):
    x_dict, edge_index_dict, metadata, in_channels = _toy_hetero()
    model = SimpleHGN(
        in_channels=in_channels, hidden_channels=16, metadata=metadata,
        num_layers=num_layers, heads=4, dropout=0.1, attn_dropout=0.1,
        target_mode="masked_multi_task", num_classes=2,
    )
    model.train()
    out = model(x_dict, edge_index_dict)
    assert out["masked_multi_task_output"]["startup"].shape == (6, 2)
    assert out["embedding"]["startup"].shape == (6, 16)
    assert out["out_mom"].shape == (6,)
    assert out["out_liq"].shape == (6,)


def test_model_forward_shape_binary_prediction():
    x_dict, edge_index_dict, metadata, in_channels = _toy_hetero()
    model = SimpleHGN(
        in_channels=in_channels, hidden_channels=16, metadata=metadata,
        num_layers=2, heads=4, target_mode="binary_prediction", num_classes=2,
    )
    out = model(x_dict, edge_index_dict)
    assert out["startup"]["output"].shape == (6, 1)


def test_model_forward_shape_multi_prediction():
    x_dict, edge_index_dict, metadata, in_channels = _toy_hetero()
    model = SimpleHGN(
        in_channels=in_channels, hidden_channels=16, metadata=metadata,
        num_layers=2, heads=4, target_mode="multi_prediction", num_classes=5,
    )
    out = model(x_dict, edge_index_dict)
    assert out["startup"]["output"].shape == (6, 5)


def test_model_layer_count_matches_num_layers():
    """num_layers must be the total conv count (not hidden-layer count)."""
    x_dict, edge_index_dict, metadata, in_channels = _toy_hetero()
    for nl in [1, 2, 4]:
        model = SimpleHGN(
            in_channels=in_channels, hidden_channels=8, metadata=metadata,
            num_layers=nl, heads=2, target_mode="binary_prediction",
        )
        assert len(model.convs) == nl, f"num_layers={nl} produced {len(model.convs)} convs"


def test_model_first_layer_residual_hardcoded_false():
    """DGL original `myGAT.__init__` hardcodes residual=False on the first
    layer regardless of the `residual` flag (line 27 in upstream). The port
    must preserve this."""
    x_dict, edge_index_dict, metadata, in_channels = _toy_hetero()
    model = SimpleHGN(
        in_channels=in_channels, hidden_channels=8, metadata=metadata,
        num_layers=3, heads=2, residual=True,
    )
    # convs[0] is the first; its res_fc should be None because residual=False.
    assert model.convs[0].res_fc is None
    # convs[1] and convs[-1] use the configured residual=True.
    assert model.convs[1].res_fc is not None
    assert model.convs[-1].res_fc is not None


def test_model_output_layer_has_no_activation():
    """The output conv must emit pre-activation features so that L2-norm and
    downstream task heads see clean linear embeddings."""
    x_dict, edge_index_dict, metadata, in_channels = _toy_hetero()
    model = SimpleHGN(
        in_channels=in_channels, hidden_channels=8, metadata=metadata,
        num_layers=3, heads=2,
    )
    assert model.convs[-1].activation is None
    # Hidden layers should have activation.
    assert model.convs[0].activation is not None
    assert model.convs[1].activation is not None


# --------------------------------------------------------------------- #
# L2 normalization (paper signature)
# --------------------------------------------------------------------- #
def test_l2_normalize_produces_unit_norm_embeddings():
    x_dict, edge_index_dict, metadata, in_channels = _toy_hetero()
    model = SimpleHGN(
        in_channels=in_channels, hidden_channels=16, metadata=metadata,
        num_layers=2, heads=4, l2_normalize=True,
        target_mode="binary_prediction",
    )
    model.eval()
    with torch.no_grad():
        out = model(x_dict, edge_index_dict)
    norms = out["startup"]["embedding"].norm(p=2, dim=1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)


def test_l2_normalize_off_preserves_raw_norms():
    x_dict, edge_index_dict, metadata, in_channels = _toy_hetero()
    model = SimpleHGN(
        in_channels=in_channels, hidden_channels=16, metadata=metadata,
        num_layers=2, heads=4, l2_normalize=False,
        target_mode="binary_prediction",
    )
    model.eval()
    with torch.no_grad():
        out = model(x_dict, edge_index_dict)
    norms = out["startup"]["embedding"].norm(p=2, dim=1)
    # Extremely unlikely that every row happens to have norm 1 without the op.
    assert (norms - 1.0).abs().max() > 1e-3


# --------------------------------------------------------------------- #
# Gradient flow
# --------------------------------------------------------------------- #
def test_gradient_flow_reaches_all_linear_weights():
    """Every Linear layer weight (per-type input proj, fc, fc_e, res_fc) AND
    the edge-type embedding must receive non-zero gradient after one backward
    pass on the masked-multi-task head."""
    x_dict, edge_index_dict, metadata, in_channels = _toy_hetero()
    model = SimpleHGN(
        in_channels=in_channels, hidden_channels=16, metadata=metadata,
        num_layers=2, heads=4, dropout=0.0, attn_dropout=0.0, residual=True,
        target_mode="masked_multi_task", num_classes=2,
    )
    model.train()
    out = model(x_dict, edge_index_dict)
    loss = out["masked_multi_task_output"]["startup"].sum()
    loss.backward()

    # Per-type input projections
    for nt, lin in model.input_proj.items():
        assert lin.weight.grad is not None and lin.weight.grad.abs().sum() > 0, \
            f"input_proj[{nt}] did not receive gradient"

    # Conv weights (fc, fc_e, edge_emb) on every layer
    for i, conv in enumerate(model.convs):
        assert conv.fc.weight.grad is not None and conv.fc.weight.grad.abs().sum() > 0, \
            f"conv[{i}].fc did not receive gradient"
        assert conv.fc_e.weight.grad is not None and conv.fc_e.weight.grad.abs().sum() > 0, \
            f"conv[{i}].fc_e did not receive gradient"
        assert conv.edge_emb.weight.grad is not None and conv.edge_emb.weight.grad.abs().sum() > 0, \
            f"conv[{i}].edge_emb did not receive gradient"


def test_previous_layer_attention_params_receive_gradient_via_local_propagation():
    """Layer 0's attention params (attn_l/attn_r/attn_e) must receive gradient
    from their use in layer 0's OWN propagation — even though the residual-
    attention path into layer 1 is a stop-gradient edge (matches DGL's
    `alpha.detach()` on the returned alpha in myGATConv)."""
    x_dict, edge_index_dict, metadata, in_channels = _toy_hetero()
    model = SimpleHGN(
        in_channels=in_channels, hidden_channels=8, metadata=metadata,
        num_layers=2, heads=2, alpha=0.5, dropout=0.0, attn_dropout=0.0,
        target_mode="binary_prediction",
    )
    model.train()
    out = model(x_dict, edge_index_dict)
    out["startup"]["output"].sum().backward()
    assert model.convs[0].attn_l.grad.abs().sum() > 0
    assert model.convs[0].attn_r.grad.abs().sum() > 0
    assert model.convs[0].attn_e.grad.abs().sum() > 0


def test_returned_alpha_is_detached():
    """DGL myGATConv returns `alpha.detach()`. Verify the port does the same:
    the returned alpha must be a leaf with requires_grad=False so the next
    layer's residual-attention blend is a stop-gradient edge."""
    torch.manual_seed(0)
    conv = SimpleHGNConv(
        in_channels=8, out_channels=4, edge_channels=4,
        num_etypes=2, num_heads=2, attn_drop=0.0, feat_drop=0.0,
    )
    x = torch.randn(4, 8, requires_grad=True)
    edge_index = torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long)
    edge_type = torch.tensor([0, 1, 0], dtype=torch.long)
    out, alpha = conv(x, edge_index, edge_type, res_attn=None)
    assert alpha.requires_grad is False, "Returned alpha must be detached (stop-gradient)"
    assert out.requires_grad is True, "Main output must carry gradient through the conv"


# --------------------------------------------------------------------- #
# Robustness
# --------------------------------------------------------------------- #
def test_missing_edge_type_at_runtime_does_not_crash():
    """If an edge type in metadata has zero edges at runtime (e.g. after
    filtering), the model must skip it cleanly rather than erroring."""
    x_dict, edge_index_dict, metadata, in_channels = _toy_hetero()
    # Drop one edge type from the runtime dict but keep it in metadata.
    eid_with_empty = dict(edge_index_dict)
    eid_with_empty[("investor", "funds", "startup")] = torch.empty((2, 0), dtype=torch.long)
    model = SimpleHGN(
        in_channels=in_channels, hidden_channels=8, metadata=metadata,
        num_layers=2, heads=2, target_mode="binary_prediction",
    )
    out = model(x_dict, eid_with_empty)
    assert out["startup"]["output"].shape == (6, 1)


# --------------------------------------------------------------------- #
# Trainer integration
# --------------------------------------------------------------------- #
def test_trainer_dispatch_branch_reachable_for_simple_hgn(mock_hetero_graph):
    """`elif model_name == "SimpleHGN":` in Trainer._initialize_model must
    construct a SimpleHGN from the per-type graph features. Stubs `self` so
    the test is insulated from the full Trainer bootstrap surface."""
    from src.ml.train import Trainer

    data = mock_hetero_graph

    class _Stub:
        def __init__(self):
            self.model_name = "SimpleHGN"
            self.target_mode = "masked_multi_task"
            self.data = data
            self.aggregation_method = "sum"
            self.config = {}
            self.model_config = {
                "hidden_channels": 16,
                "num_layers": 2,
                "heads": 4,
                "edge_dim": None,
                "dropout": 0.2,
                "attn_dropout": 0.2,
                "negative_slope": 0.05,
                "residual": True,
                "alpha": 0.05,
                "l2_normalize": True,
                "bias": False,
                "activation_type": "elu",
            }

    stub = _Stub()
    model = Trainer._initialize_model(stub, data)
    assert type(model).__name__ == "SimpleHGN"
    assert len(model.convs) == 2
    # Input projections must exist for every node type in the graph metadata.
    for nt in data.node_types:
        assert nt in model.input_proj


def test_model_integrates_with_trainer_style_call_pattern(mock_hetero_graph):
    """Trainer invokes `self.model(data.x_dict, data.edge_index_dict)` in
    full-batch mode. Verify the model handles the real mock graph schema."""
    data = mock_hetero_graph
    in_channels = {nt: data[nt].num_features for nt in data.node_types}
    model = SimpleHGN(
        in_channels=in_channels,
        hidden_channels=16,
        metadata=data.metadata(),
        num_layers=2, heads=4,
        target_mode="masked_multi_task", num_classes=2,
    )
    model.train()
    out = model(data.x_dict, data.edge_index_dict)
    n_startups = data["startup"].num_nodes
    assert out["masked_multi_task_output"]["startup"].shape == (n_startups, 2)
    loss = out["masked_multi_task_output"]["startup"].sum()
    loss.backward()  # must not raise
