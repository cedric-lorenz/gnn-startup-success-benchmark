"""Neural network model definitions for startup success prediction (SeHGNN, HAN, GCN, MLP, XGBoost)."""
import torch
from torch.utils.checkpoint import checkpoint
from torch_geometric.nn import SAGEConv, HANConv, to_hetero, GraphConv, RGCNConv, GATConv, MessagePassing
from torch_geometric.utils import softmax as pyg_softmax


class _MultiplicativeEdgeWeightGATConv(GATConv):
    """GATConv with PathCount as a *multiplicative* edge weight on attention.

    Paper literally says 'weight on each edge is PathCount'. This class
    realizes that by adding log(edge_weight) to the pre-softmax attention
    logits, which (after softmax) is equivalent to

        alpha'_ij = (alpha_ij * w_ij) / sum_k alpha_ik * w_ik

    i.e. re-normalized multiplication - but numerically stable. Edges with
    larger PathCount get a higher share of attention, independent of what
    the learnable attention mechanism decides.

    Pass edge_weight via the `edge_weight` kwarg to forward(); DO NOT set
    edge_dim on construction (we ignore the attention-MLP edge-feature path).
    """

    def forward(self, x, edge_index, edge_weight=None, **kwargs):
        # Stash edge_weight on self so edge_update can see it. Safe because
        # PyG MessagePassing doesn't recurse within a single forward.
        self._ew_stash = edge_weight
        try:
            return super().forward(x, edge_index, **kwargs)
        finally:
            self._ew_stash = None

    def edge_update(self, alpha_j, alpha_i, edge_attr, index, ptr, size_i):
        # Recreate GATConv.edge_update but inject a log(edge_weight) bias
        # before the softmax. Mirrors the upstream implementation sans the
        # edge_attr attention-MLP path (we don't use edge_dim here).
        alpha = alpha_j + (alpha_i if alpha_i is not None else 0)
        alpha = F.leaky_relu(alpha, self.negative_slope)
        if getattr(self, "_ew_stash", None) is not None:
            log_ew = torch.log(self._ew_stash.clamp_min(1e-8)).view(-1, 1)
            alpha = alpha + log_ew  # broadcasts across heads
        alpha = pyg_softmax(alpha, index, ptr, size_i)
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        return alpha


class GraphConvEncoder(torch.nn.Module):
    def __init__(
        self,
        in_channels,
        hidden_channels,
        out_channels,
        num_layers=2,
        activation_type="relu",
        normalize=True,
        dropout=0.0,
        aggr="add",
    ):
        super().__init__()
        self.num_layers = num_layers
        self.dropout = dropout

        self.convs = torch.nn.ModuleList()
        self.norms = torch.nn.ModuleList() if normalize else None

        if activation_type == "relu":
            self.activation = torch.nn.ReLU()
        elif activation_type == "prelu":
            self.activation = torch.nn.PReLU()
        else:
            raise ValueError("Unsupported activation type. Choose 'relu' or 'prelu'.")

        # First layer
        self.convs.append(GraphConv(in_channels, hidden_channels, aggr=aggr))
        if normalize:
            self.norms.append(torch_geometric.nn.BatchNorm(hidden_channels))

        # Hidden layers
        for _ in range(num_layers - 2):
            self.convs.append(GraphConv(hidden_channels, hidden_channels, aggr=aggr))
            if normalize:
                self.norms.append(torch_geometric.nn.BatchNorm(hidden_channels))

        # Last layer (if num_layers > 1)
        if num_layers > 1:
            self.convs.append(GraphConv(hidden_channels, out_channels, aggr=aggr))
            if normalize:
                self.norms.append(torch_geometric.nn.BatchNorm(out_channels))

    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if self.norms is not None:
                x = self.norms[i](x)
            
            if i < self.num_layers - 1:
                x = self.activation(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        return x

import math

class ArcFace(torch.nn.Module):
    """
    ArcFace: Additive Angular Margin Loss.
    """
    def __init__(self, in_features, out_features, s=64.0, m=0.5, easy_margin=False):
        super(ArcFace, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        
        # Weights (Class Centers)
        self.weight = torch.nn.Parameter(torch.FloatTensor(out_features, in_features))
        torch.nn.init.xavier_uniform_(self.weight)

        self.easy_margin = easy_margin
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, input, label):
        # --------------------------- cosine(theta) & phi(theta) ---------------------------
        # input: [batch_size, embedding_dim] (already normalized in SeHGNN usually, but safe to re-norm)
        # weight: [num_classes, embedding_dim]
        
        # Normalize input and weights
        # Note: SeHGNN retrieval head already normalizes output, but ArcFace requires strict normalization
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        
        # --------------------------- convert label to one-hot ---------------------------
        # label: [batch_size]
        
        sine = torch.sqrt((1.0 - torch.pow(cosine, 2)).clamp(0, 1))
        
        # phi = cos(theta + m) = cos(theta)cos(m) - sin(theta)sin(m)
        phi = cosine * self.cos_m - sine * self.sin_m
        
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            # When theta > pi - m, cos(theta + m) is not monotonic decreasing
            # Standard implementation uses: where cos(theta) > th, use phi, else use cos(theta) - mm
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)
            
        # --------------------------- convert label to one-hot ---------------------------
        # Clone to avoid in-place errors on the autograd graph.
        output = cosine.clone()

        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine) 
        output *= self.s

        return output

import torch_geometric.nn
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch import long
from typing import Union, Dict, List, Tuple


class BaseGNN(torch.nn.Module):
    def __init__(self, hidden_channels, target_mode, num_classes, activation_type="relu"):
        super().__init__()
        self.target_mode = target_mode
        self.num_classes = num_classes
        self.hidden_channels = hidden_channels
        self._init_activation(activation_type)
        self._init_heads()

    def _init_activation(self, activation_type):
        if activation_type == "relu":
            self.activation = torch.nn.ReLU()
        elif activation_type == "prelu":
            self.activation = torch.nn.PReLU()
        elif activation_type == "leaky_relu":
            self.activation = torch.nn.LeakyReLU()
        elif activation_type == "elu":
            self.activation = torch.nn.ELU()
        else:
            raise ValueError(f"Unsupported activation type: {activation_type}. Choose 'relu', 'prelu', 'leaky_relu', or 'elu'.")

    def _get_activation(self, name):
        """Get activation function by name (helper for retrieval head)"""
        activations = {
            "relu": torch.nn.ReLU(),
            "prelu": torch.nn.PReLU(),
            "gelu": torch.nn.GELU(),
            "tanh": torch.nn.Tanh(),
            "leaky_relu": torch.nn.LeakyReLU()
        }
        return activations.get(name, torch.nn.ReLU())
    
    def _init_retrieval_head(self, config, model_name, input_dim):
        """Initialize retrieval projection head (SimCLR/CLIP pattern)"""
        self.detach_retrieval_head = config["models"][model_name].get("detach_retrieval_head", False)
        if self.detach_retrieval_head:
             print(f"✅ Retrieval Head: Gradient Stop enabled (Backbone Detached)")
             
        proj_config = config["models"][model_name].get("retrieval_projection", {})
        
        hidden_dim = proj_config.get("hidden_dim", 128)
        output_dim = proj_config.get("output_dim", input_dim)
        dropout = proj_config.get("dropout", 0.3)
        use_bn = proj_config.get("use_batch_norm", True)
        activation = proj_config.get("activation", "relu")
        
        # Build projection head (2-layer MLP like SimCLR)
        layers = [
            torch.nn.Linear(input_dim, hidden_dim),
            self._get_activation(activation),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_dim, output_dim)
        ]
        
        if use_bn:
            layers.append(torch.nn.BatchNorm1d(output_dim))
        
        self.retrieval_proj = torch.nn.Sequential(*layers)
        
        print(f"✅ Initialized retrieval projection head: {input_dim}→{hidden_dim}→{output_dim}")

    def _init_heads(self):
        if self.target_mode == "binary_prediction":
            self.output_head = torch.nn.Linear(self.hidden_channels, 1)
        elif self.target_mode == "multi_prediction":
            self.output_head = torch.nn.Linear(self.hidden_channels, self.num_classes)
        elif self.target_mode == "multi_task":
            self.task_binary_encoder = torch.nn.Linear(self.hidden_channels, self.hidden_channels)
            self.task_multi_encoder = torch.nn.Linear(self.hidden_channels, self.hidden_channels)
            self.output_binary = torch.nn.Linear(self.hidden_channels, 1)
            self.output_multi = torch.nn.Linear(self.hidden_channels, self.num_classes)
        elif self.target_mode == "multi_label":
            # 3 Binary Classification Heads w/ optional dedicated encoders
            self.head_fund = torch.nn.Linear(self.hidden_channels, 1)
            self.head_acq = torch.nn.Linear(self.hidden_channels, 1)
            self.head_ipo = torch.nn.Linear(self.hidden_channels, 1)
        elif self.target_mode == "masked_multi_task":
            # Tower 1: Momentum (Funding)
            self.head_momentum = torch.nn.Sequential(
                torch.nn.Linear(self.hidden_channels, self.hidden_channels),
                self.activation,
                torch.nn.Linear(self.hidden_channels, 1)
            )
            # Tower 2: Liquidity (Acq/IPO)
            self.head_liquidity = torch.nn.Sequential(
                torch.nn.Linear(self.hidden_channels, self.hidden_channels),
                self.activation,
                torch.nn.Linear(self.hidden_channels, 1)
            )
        else:
            raise ValueError(f"Unsupported target_mode: {self.target_mode}")

    def _apply_heads(self, startup_x, retrieval_labels=None):
        if self.target_mode == "binary_prediction":
            return {
                "startup": {
                    "output": self.output_head(startup_x),
                    "embedding": startup_x
                }
            }
        elif self.target_mode == "multi_prediction":
            return {
                "startup": {
                    "output": self.output_head(startup_x),
                    "embedding": startup_x
                }
            }
        elif self.target_mode == "multi_task":
            binary_x = self.activation(self.task_binary_encoder(startup_x))
            multi_x = self.activation(self.task_multi_encoder(startup_x))
            return {
                "binary_output": {"startup": self.output_binary(binary_x).squeeze(-1)},
                "multi_class_output": {"startup": self.output_multi(multi_x)},
                "embedding": {"startup": startup_x}
            }
        elif self.target_mode == "multi_label":
             # Apply 3 separate heads
             out_fund = self.head_fund(startup_x).squeeze(-1)
             out_acq = self.head_acq(startup_x).squeeze(-1)
             out_ipo = self.head_ipo(startup_x).squeeze(-1)
             
             out_combined = torch.stack([out_fund, out_acq, out_ipo], dim=1)
             
             return {
                 "multi_label_output": {"startup": out_combined},
                 "embedding": {"startup": startup_x},
                 # Individual outputs if needed for flexibility
                 "out_fund": out_fund,
                 "out_acq": out_acq,
                 "out_ipo": out_ipo
             }
        elif self.target_mode == "masked_multi_task":
             # Apply 2 MLP heads
             out_mom = self.head_momentum(startup_x).squeeze(-1)
             out_liq = self.head_liquidity(startup_x).squeeze(-1)
             
             # Stack for convenient tensor access: [Mom, Liq]
             out_combined = torch.stack([out_mom, out_liq], dim=1)
             
             output = {
                 "masked_multi_task_output": {"startup": out_combined},
                 "embedding": {"startup": startup_x},
                 "out_mom": out_mom, # Momentum Logic
                 "out_liq": out_liq  # Liquidity Logic
             }
             
             # Add retrieval embedding if projection head exists (SimCLR/CLIP pattern)
             if hasattr(self, 'retrieval_proj') and self.retrieval_proj is not None:
                 
                 x_in = startup_x
                 # Gradient Stop: Detach backbone if configured to protect main task
                 if getattr(self, 'detach_retrieval_head', False):
                     x_in = x_in.detach()
                 
                 retrieval_emb = self.retrieval_proj(x_in)  # Project
                 retrieval_emb = F.normalize(retrieval_emb, p=2, dim=1)  # Normalize (like CLIP)
                 output["retrieval_embedding"] = {"startup": retrieval_emb}
                 
                 # ArcFace Logic
                 if hasattr(self, 'arcface_head') and retrieval_labels is not None:
                     # retrieval_labels: [Batch]
                     # arcface_head(emb, labels) -> logits
                     # Note: ArcFace requires labels to compute Margin Loss during Training.
                     # During eval, we usually just want embeddings.
                     if self.training:
                         arcface_logits = self.arcface_head(retrieval_emb, retrieval_labels)
                         output["arcface_logits"] = {"startup": arcface_logits}
             
             return output
        else:
            raise ValueError(f"Unsupported target_mode: {self.target_mode}")


class GAT(BaseGNN):
    def __init__(
        self,
        in_channels,
        hidden_channels,
        out_channels,
        num_layers=2,
        v2=True,
        normalize=True,
        activation="prelu",
        jumping_knowledge="cat",
        add_self_loops=False,
        target_mode="multi_prediction",
        num_classes=2,
        metadata=None,
        aggr="sum",
        dropout=0.0,
        heads=1,
        negative_slope=0.2,
    ):
        super().__init__(hidden_channels, target_mode, num_classes, activation)

        # Initialize GAT Encoder
        # We use the standard PyG GAT implementation
        norm = torch_geometric.nn.BatchNorm(hidden_channels) if normalize else None
        act = torch.nn.PReLU() if activation == "prelu" else torch.nn.ReLU()
        
        self.encoder = torch_geometric.nn.GAT(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            out_channels=hidden_channels, # Encoder outputs hidden_channels
            num_layers=num_layers,
            v2=v2,
            norm=norm,
            act=act,
            jk=jumping_knowledge,
            add_self_loops=add_self_loops,
            dropout=dropout,
            heads=heads,
            negative_slope=negative_slope,
        )

        # Wrap encoder with to_hetero
        if metadata is not None:
            self.encoder = to_hetero(self.encoder, metadata, aggr=aggr)

    def forward(self, x_dict, edge_index_dict):
        # Get embeddings from heterogeneous encoder
        # x_dict will contain embeddings for all node types
        embeddings_dict = self.encoder(x_dict, edge_index_dict)
        
        # Extract startup embeddings
        startup_x = embeddings_dict['startup']
        
        return self._apply_heads(startup_x)


class GCN(BaseGNN):
    def __init__(
        self,
        in_channels,
        hidden_channels,
        out_channels,
        num_layers=2,
        normalize=True,
        activation="prelu",
        jumping_knowledge="cat",
        target_mode="multi_prediction",
        num_classes=2,
        metadata=None,
        aggr="sum",
        dropout=0.0,
        add_self_loops=False, # Kept for interface consistency, but GraphConv handles it differently or implicitly
    ):
        super().__init__(hidden_channels, target_mode, num_classes, activation)

        # Initialize GraphConv Encoder
        self.encoder = GraphConvEncoder(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            out_channels=hidden_channels,
            num_layers=num_layers,
            activation_type=activation,
            normalize=normalize,
            dropout=dropout,
            aggr=aggr, # GraphConv supports 'add', 'mean', 'max'
        )

        # Wrap encoder with to_hetero
        if metadata is not None:
            self.encoder = to_hetero(self.encoder, metadata, aggr=aggr)

    def forward(self, x_dict, edge_index_dict):
        # Get embeddings from heterogeneous encoder
        # x_dict will contain embeddings for all node types
        embeddings_dict = self.encoder(x_dict, edge_index_dict)
        
        # Extract startup embeddings
        startup_x = embeddings_dict['startup']
        
        return self._apply_heads(startup_x)


class RGCN(BaseGNN):
    """Relational GCN (Schlichtkrull et al. 2017) using PyG's RGCNConv.

    Heterogeneous graph is homogenized inside forward: per-node-type features
    are projected to `hidden_channels`, concatenated, and each edge_type becomes
    an integer relation label. Paper Eq. 2's self-loop term W_0 is handled by
    `root_weight=True` on RGCNConv. Basis decomposition (Eq. 3) via `num_bases`,
    block-diagonal (Eq. 4) via `num_blocks`.
    """
    def __init__(
        self,
        in_channels,              # dict: {node_type: in_dim}
        hidden_channels,
        num_layers=2,
        num_bases=30,
        num_blocks=None,
        normalize=True,
        activation="relu",
        target_mode="masked_multi_task",
        num_classes=2,
        metadata=None,
        aggr="mean",
        dropout=0.0,
    ):
        super().__init__(hidden_channels, target_mode, num_classes, activation)

        assert metadata is not None, "RGCN requires heterogeneous metadata"
        node_types, edge_types = metadata
        self.node_types = list(node_types)
        self.edge_types = list(edge_types)
        self.num_relations = len(self.edge_types)
        self._rel_to_idx = {rel: i for i, rel in enumerate(self.edge_types)}
        self.dropout = dropout

        self.input_proj = torch.nn.ModuleDict({
            nt: torch.nn.Linear(in_channels[nt], hidden_channels)
            for nt in self.node_types
        })

        self.convs = torch.nn.ModuleList([
            RGCNConv(
                hidden_channels, hidden_channels,
                num_relations=self.num_relations,
                num_bases=num_bases,
                num_blocks=num_blocks,
                aggr=aggr,
                root_weight=True,
            )
            for _ in range(num_layers)
        ])

        self.norms = torch.nn.ModuleList([
            torch_geometric.nn.BatchNorm(hidden_channels) for _ in range(num_layers)
        ]) if normalize else None

    def forward(self, x_dict, edge_index_dict):
        h_dict = {nt: self.input_proj[nt](x_dict[nt]) for nt in self.node_types}

        node_offsets = {}
        offset = 0
        node_feats = []
        for nt in self.node_types:
            node_offsets[nt] = offset
            node_feats.append(h_dict[nt])
            offset += h_dict[nt].size(0)
        x = torch.cat(node_feats, dim=0)

        edge_indices, edge_type_ids = [], []
        for rel, ei in edge_index_dict.items():
            shifted = ei.clone()
            shifted[0] = shifted[0] + node_offsets[rel[0]]
            shifted[1] = shifted[1] + node_offsets[rel[2]]
            edge_indices.append(shifted)
            edge_type_ids.append(torch.full(
                (ei.size(1),), self._rel_to_idx[rel],
                dtype=torch.long, device=ei.device,
            ))
        edge_index = torch.cat(edge_indices, dim=1)
        edge_type = torch.cat(edge_type_ids, dim=0)

        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index, edge_type)
            if self.norms is not None:
                x = self.norms[i](x)
            if i < len(self.convs) - 1:
                x = self.activation(x)
                x = F.dropout(x, p=self.dropout, training=self.training)

        startup_x = x[node_offsets["startup"]:node_offsets["startup"] + x_dict["startup"].size(0)]
        return self._apply_heads(startup_x)


def _edge_key(edge_type: Tuple[str, str, str]) -> str:
    """Convert (src, rel, dst) tuple to string key usable in ModuleDict."""
    return "__".join(edge_type)


def _dist_corr(x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
    """Pearson-style correlation loss (Hetero2Net paper Eq. 8).

    Port of the reference implementation's `dist_corr` — centers both tensors,
    computes |E[x1*x2]| / (std(x1) * std(x2) + eps). Returns a scalar in [0, 1].
    """
    x1 = x1 - x1.mean(dim=0, keepdim=True)
    x2 = x2 - x2.mean(dim=0, keepdim=True)
    sigma1 = torch.sqrt(torch.mean(x1.pow(2)))
    sigma2 = torch.sqrt(torch.mean(x2.pow(2)))
    return torch.abs(torch.mean(x1 * x2)) / (sigma1 * sigma2 + 1e-8)


class _HeteroDisenConv(torch_geometric.nn.MessagePassing):
    """Two-channel message passing (Hetero2Net's DisenConv).

    Produces (out, out_homo, out_hetero). `lin_homo` and `lin_hetero` are
    independent linear projections of source features; each is mean-aggregated
    at the destination. A root-weight term `lin_r` projects destination's own
    features and is added to `out` only. This mirrors the reference
    implementation at
    https://github.com/EdisonLeeeee/Hetero2Net/blob/main/hetero2net/layers.py
    but uses PyG's default message()/aggregate() path (no SparseTensor spmm)
    so the layer works on CPU without torch-cluster/torch-scatter-extras.
    """

    def __init__(self, in_channels, out_channels, bias: bool = True, root_weight: bool = True):
        super().__init__(aggr="mean")
        # PyG Linear supports lazy init with in_channels=-1.
        from torch_geometric.nn.dense.linear import Linear as PyGLinear
        if isinstance(in_channels, (tuple, list)):
            in_src, in_dst = in_channels
        else:
            in_src = in_dst = in_channels
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.root_weight = root_weight
        self.lin_homo = PyGLinear(in_src, out_channels, bias=bias)
        self.lin_hetero = PyGLinear(in_src, out_channels, bias=bias)
        if root_weight:
            self.lin_r = PyGLinear(in_dst, out_channels, bias=False)
        else:
            self.lin_r = None

    def reset_parameters(self):
        super().reset_parameters()
        self.lin_homo.reset_parameters()
        self.lin_hetero.reset_parameters()
        if self.lin_r is not None:
            self.lin_r.reset_parameters()

    def forward(self, x, edge_index):
        if isinstance(x, torch.Tensor):
            x_src = x_dst = x
        else:
            x_src, x_dst = x

        x_homo = self.lin_homo(x_src)
        x_hetero = self.lin_hetero(x_src)

        # Destination-sized aggregation. Size is inferred from x_dst.
        size = (x_src.size(0), x_dst.size(0))
        out_homo = self.propagate(edge_index, x=(x_homo, x_dst), size=size)
        out_hetero = self.propagate(edge_index, x=(x_hetero, x_dst), size=size)

        out = out_homo + out_hetero
        if self.lin_r is not None and x_dst is not None:
            out = out + self.lin_r(x_dst)

        return out, out_homo, out_hetero

    def message(self, x_j):
        return x_j


class _HeteroEdgeDecoder(torch.nn.Module):
    """2-layer MLP edge decoder for masked-metapath link prediction (Eq. 10).

    Given src/dst embedding tables and an edge_label_index [2, E], computes
    a scalar logit per edge via (src ⊙ dst) → Linear → ReLU → Dropout → Linear.
    """

    def __init__(self, hidden_channels: int, dropout: float = 0.0):
        super().__init__()
        from torch_geometric.nn.dense.linear import Linear as PyGLinear
        self.lin1 = PyGLinear(-1, hidden_channels)
        self.lin2 = PyGLinear(hidden_channels, 1)
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, src: torch.Tensor, dst: torch.Tensor, edge_label_index: torch.Tensor) -> torch.Tensor:
        row, col = edge_label_index
        z = src[row] * dst[col]
        z = self.lin1(z).relu()
        z = self.dropout(z)
        z = self.lin2(z)
        return z.view(-1)


class Hetero2Net(BaseGNN):
    """Hetero2Net (Li, Wei, Dan et al., 2023) — heterophily-aware HGNN.

    Paper: "Hetero2Net: Heterophily-aware Representation Learning on
    Heterogeneous Graphs" (arXiv 2310.11664). Reference implementation:
    https://github.com/EdisonLeeeee/Hetero2Net.

    Core ideas implemented here, with their corresponding paper sections:
      - § 5.1 Multi-view graph fusion: per-edge-type DisenConv message passing,
        sum-fuse across edge types at each destination.
      - § 5.2 Disentangled masked metapath prediction:
          (P1) correlation loss (Eq. 8) — `_dist_corr(h_homo, h_hetero)` per
               (layer, edge_type), averaged. Weight = α.
          (P2) masked metapath reconstruction (Eq. 9): a fraction `mask_ratio`
               of each startup-startup metapath's edges are removed from the
               graph before message passing; the homo channel's multi-layer
               concatenation is rewarded for ranking these as edges and the
               hetero channel for ranking random (likely disconnected) pairs.
               Weight = β.
      - § 5.3 Masked label prediction (Eq. 11-12): if `use_label_propagation`,
        an embedding of partially-observed startup labels is added to the
        startup features BEFORE the first conv. `label_mask_ratio` of the
        training labels are masked each step.

    Differences vs. the reference:
      - Metapath edge drop uses uniform random edge masking rather than the
        `torch_cluster.random_walk`-based path drop; walk_length=1 in the
        reference (for length-1 metapaths the two are equivalent up to the
        sampler's degree-proportional start-node bias, which is uniform here).
      - Multi-label binary targets (momentum + liquidity) are handled by
        element-wise active-vs-masked label embedding sum, adapted from the
        reference's `y.nonzero()` branch for multi-label graphs.

    Auxiliary losses are exposed on the module so the trainer can pick them up
    via the generic `aux_loss` hook:
        self.corr_loss, self.link_loss, self.aux_loss = α·corr + β·link
    """

    def __init__(
        self,
        in_channels,                      # int (startup feature dim)
        hidden_channels: int,
        metadata,                         # (node_types, edge_types) from graph.metadata()
        num_layers: int = 2,
        dropout: float = 0.0,
        activation_type: str = "relu",
        target_mode: str = "masked_multi_task",
        num_classes: int = 2,
        alpha: float = 0.2,               # paper: L_corr weight
        beta: float = 0.4,                # paper: L_rec weight
        mask_ratio: float = 0.5,          # paper 'r': fraction of metapath edges masked per step
        label_mask_ratio: float = 0.7,    # paper 'p': fraction of labels kept (masked = 1 - p)
        use_label_propagation: bool = True,
        num_label_classes: int = 2,       # momentum + liquidity = 2 binary labels
        root_weight: bool = True,
        normalize_bn: bool = True,
        corr_on_metapaths_only: bool = True,  # restrict L_corr to startup-startup metapath edges only (memory)
        aux_reduction: str = "sum",           # "sum" (paper-faithful) or "mean" (scale-invariant)
    ):
        super().__init__(hidden_channels, target_mode, num_classes, activation_type)

        if alpha < 0 or beta < 0:
            raise ValueError(f"Hetero2Net: alpha, beta must be non-negative (got {alpha=}, {beta=})")
        if not 0.0 <= mask_ratio < 1.0:
            raise ValueError(f"Hetero2Net: mask_ratio must be in [0, 1) (got {mask_ratio})")
        if aux_reduction not in ("sum", "mean"):
            raise ValueError(
                f"Hetero2Net: aux_reduction must be 'sum' or 'mean', got {aux_reduction!r}"
            )

        self.metadata = metadata
        self.num_layers = num_layers
        self.dropout_p = dropout
        self.alpha = alpha
        self.beta = beta
        self.mask_ratio = mask_ratio
        self.label_mask_ratio = label_mask_ratio
        self.use_label_propagation = use_label_propagation
        self.num_label_classes = num_label_classes
        self.corr_on_metapaths_only = corr_on_metapaths_only
        self.aux_reduction = aux_reduction

        node_types, edge_types = metadata
        self.node_types: List[str] = list(node_types)
        self.edge_types: List[Tuple[str, str, str]] = list(edge_types)

        # Startup-startup metapaths are the only targets of the masked-metapath
        # reconstruction task (paper: metapaths end where they start).
        self.metapaths: List[Tuple[str, str, str]] = [
            et for et in self.edge_types if et[0] == "startup" and et[2] == "startup"
        ]

        # Per-layer per-edge-type DisenConv + per-node-type BatchNorm.
        self.convs = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()
        bn_cls = torch.nn.BatchNorm1d if normalize_bn else torch.nn.Identity
        for _ in range(num_layers):
            conv = torch.nn.ModuleDict({
                _edge_key(et): _HeteroDisenConv(-1, hidden_channels, root_weight=root_weight)
                for et in self.edge_types
            })
            bn = torch.nn.ModuleDict({
                nt: bn_cls(hidden_channels) if normalize_bn else torch.nn.Identity()
                for nt in self.node_types
            })
            self.convs.append(conv)
            self.bns.append(bn)

        # Per-node-type input projection into `hidden_channels` for nodes with no
        # incoming edges in the graph (keeps dims consistent through the stack).
        # Lazy PyG Linear so we don't need to know each node type's feature dim.
        from torch_geometric.nn.dense.linear import Linear as PyGLinear
        self.input_proj = torch.nn.ModuleDict({
            nt: PyGLinear(-1, hidden_channels) for nt in self.node_types
        })

        # Label embedding, added to startup features BEFORE the first conv.
        # Embedding dim matches startup input feature dim so the sum is valid.
        if self.use_label_propagation:
            if not isinstance(in_channels, int):
                raise ValueError("Hetero2Net: label propagation requires int in_channels (startup dim)")
            # num_label_classes + 1: extra "masked" bucket (per paper Eq. 12).
            self.label_embedding = torch.nn.Embedding(num_label_classes + 1, in_channels)
            torch.nn.init.xavier_normal_(self.label_embedding.weight)
        else:
            self.label_embedding = None

        # Shared edge decoder (works over multi-layer concatenated embeddings).
        self.edge_decoder = _HeteroEdgeDecoder(hidden_channels, dropout=dropout)
        self.feat_dropout = torch.nn.Dropout(dropout)

        # Registered label input set via set_label_propagation_input().
        self._label_prop_y: torch.Tensor = None

        # Aux-loss buffers (written by forward()).
        self.corr_loss: torch.Tensor = torch.zeros(())
        self.link_loss: torch.Tensor = torch.zeros(())
        self.aux_loss: torch.Tensor = torch.zeros(())

    def set_label_propagation_input(self, y_masked: torch.Tensor):
        """Register [N_startup, num_label_classes] multi-label binary targets.

        The caller is expected to zero-out non-training rows so they don't leak
        into the embedding. During forward(), a `label_mask_ratio` fraction of
        remaining active rows are randomly further masked each step.
        """
        if y_masked.dim() != 2 or y_masked.size(1) != self.num_label_classes:
            raise ValueError(
                f"Hetero2Net: y_masked must be [N, {self.num_label_classes}], got {tuple(y_masked.shape)}"
            )
        self._label_prop_y = y_masked

    # ------------------------------------------------------------------ #
    # Forward
    # ------------------------------------------------------------------ #
    def forward(self, x_dict, edge_index_dict, **kwargs):
        device = x_dict["startup"].device

        # Shallow copy — don't mutate caller's dict.
        x_dict = dict(x_dict)

        # 1. Label-embedding injection (paper Eq. 11/12). Paper §5.3: "During
        # the inference stage, we set the masking ratio p to 0, which ensures
        # that the model can leverage the complete label information Y_train
        # to guide its predictions." — so we apply the label embedding in BOTH
        # train AND eval; _compute_label_embedding() skips the Bernoulli
        # masking when not training. Test-set rows have their labels zeroed
        # upstream (trainer registers `y_masked` with `~train_mask` rows set
        # to 0.0), so no leakage happens here.
        if (
            self.use_label_propagation
            and self.label_embedding is not None
            and self._label_prop_y is not None
        ):
            y_emb = self._compute_label_embedding(x_dict["startup"].size(0), device)
            if y_emb is not None:
                x_dict["startup"] = x_dict["startup"] + y_emb

        # 2. Mask metapath edges for link prediction (training only).
        if (
            self.training
            and self.metapaths
            and self.beta > 0.0
            and self.mask_ratio > 0.0
        ):
            masked_edge_index_dict, positive_edges = self._mask_metapath_edges(
                edge_index_dict, ratio=self.mask_ratio
            )
        else:
            masked_edge_index_dict, positive_edges = edge_index_dict, {}

        # 3. Stack of DisenConv layers.
        # First layer takes raw (or label-augmented) features; subsequent layers
        # take projected hidden features. Node types without incoming edges get
        # an input-projection pass-through so their dim stays at hidden_channels.
        #
        # Memory optimisation (vs. reference): rather than storing every edge
        # type's (homo, hetero) tensors from every layer and computing aux
        # losses afterwards, we accumulate L_corr INCREMENTALLY here and only
        # retain the startup-startup metapath tensors needed for the link-loss
        # cross-layer concatenation (paper Eq. 9). Non-metapath edges contribute
        # to L_corr via an inline accumulator and their tensors are released as
        # soon as the loop body ends, preventing the autograd graph from
        # blowing up on graphs with many edge types (we hit 44GB OOM at 128h,
        # 22 metapaths + ~10 bipartite edge types on a full Crunchbase run).
        accum_corr_sum = torch.zeros((), device=device)
        accum_corr_count = 0
        metapath_homo_layers: Dict[Tuple[str, str, str], List[torch.Tensor]] = {
            mp: [] for mp in positive_edges.keys()
        }
        metapath_hetero_layers: Dict[Tuple[str, str, str], List[torch.Tensor]] = {
            mp: [] for mp in positive_edges.keys()
        }

        current_x = x_dict
        for layer_idx, conv in enumerate(self.convs):
            out_dict, homo_dict, hetero_dict = self._hetero_conv(
                conv, current_x, masked_edge_index_dict
            )

            if self.training:
                for et, h in homo_dict.items():
                    ht = hetero_dict.get(et)
                    if h is None or ht is None:
                        continue
                    is_metapath = et[0] == "startup" and et[2] == "startup"
                    # L_corr: either on all edge types (paper-faithful) or
                    # restricted to startup-startup metapaths only (memory).
                    if (not self.corr_on_metapaths_only) or is_metapath:
                        accum_corr_sum = accum_corr_sum + _dist_corr(h, ht)
                        accum_corr_count += 1
                    # Retain tensors for link-loss cross-layer concat — only
                    # for metapaths that were actually masked this step.
                    if et in metapath_homo_layers:
                        metapath_homo_layers[et].append(h)
                        metapath_hetero_layers[et].append(ht)

            # Free the per-layer dicts so non-metapath tensors are not held.
            del homo_dict, hetero_dict

            # BN + activation + dropout for all but the last layer (paper's
            # reference impl follows this pattern; helps oversmoothing control).
            is_last = layer_idx == self.num_layers - 1
            out_dict = {
                nt: (x if is_last else self.feat_dropout(self.activation(self.bns[layer_idx][nt](x))))
                for nt, x in out_dict.items() if x is not None
            }
            current_x = out_dict

        # 4. Finalize auxiliary losses.
        if self.training:
            self._finalize_aux_losses(
                accum_corr_sum, accum_corr_count,
                metapath_homo_layers, metapath_hetero_layers,
                positive_edges, device,
            )
        else:
            self._zero_aux_losses(device)

        startup_x = current_x["startup"]
        return self._apply_heads(startup_x)

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    def _compute_label_embedding(self, n_rows: int, device) -> torch.Tensor:
        """Build label embedding for paper Eq. 11/12.

        Multi-label binary semantics: for each of num_label_classes columns, a
        row is either 'active' (col-value > 0.5) or 'masked'. A further random
        keep-mask (Bernoulli with prob=label_mask_ratio) is applied to 'active'
        entries during training only to prevent leakage. At eval, per paper
        §5.3 ("set masking ratio p to 0 at inference"), NO random masking is
        applied — test-set rows have already been zeroed upstream by the
        trainer, so their entries are naturally in the 'masked' bucket, while
        train-set rows contribute their true labels to the embedding sum.
        """
        y = self._label_prop_y
        if y is None:
            return None
        y = y.to(device)
        if y.size(0) != n_rows:
            # Out of sync (e.g. mini-batch with subsampled seed nodes) — skip.
            return None
        N, C = y.shape
        col_ids = torch.arange(C, device=device).view(1, -1).expand(N, -1)
        active = y > 0.5
        if self.training and self.label_mask_ratio < 1.0:
            keep = torch.rand(N, 1, device=device) < self.label_mask_ratio
            active = active & keep
        masked_idx = torch.full_like(col_ids, self.num_label_classes)
        label_ids = torch.where(active, col_ids, masked_idx)
        # Multiplicative mask zeros out the "masked" embedding contribution.
        mask_factor = (label_ids != self.num_label_classes).float().unsqueeze(-1)
        emb = self.label_embedding(label_ids) * mask_factor  # [N, C, D]
        emb = self.feat_dropout(emb.sum(dim=1))  # [N, D]
        return emb

    def _mask_metapath_edges(self, edge_index_dict, ratio: float):
        """Uniformly sample a `ratio` fraction of edges per startup-startup
        metapath; remove them from the graph and return them as positives for
        the link-prediction task."""
        out_dict = dict(edge_index_dict)
        positive_edges: Dict[Tuple[str, str, str], torch.Tensor] = {}
        for mp in self.metapaths:
            ei = edge_index_dict.get(mp)
            if ei is None or ei.numel() == 0:
                continue
            E = ei.size(1)
            num_drop = max(1, int(ratio * E))
            num_drop = min(num_drop, E - 1)  # keep at least 1 edge for message passing
            if num_drop <= 0:
                continue
            perm = torch.randperm(E, device=ei.device)
            drop_idx = perm[:num_drop]
            keep_idx = perm[num_drop:]
            out_dict[mp] = ei[:, keep_idx]
            positive_edges[mp] = ei[:, drop_idx]
        return out_dict, positive_edges

    def _hetero_conv(self, conv_dict, x_dict, edge_index_dict):
        """Run per-edge-type DisenConv and sum outputs at each destination
        node type (paper §5.1 'sum' fuse). Returns the fused out_dict plus
        raw homo/hetero dicts keyed by edge_type tuple (for aux losses)."""
        out_acc: Dict[str, List[torch.Tensor]] = {nt: [] for nt in self.node_types}
        homo_out: Dict[Tuple[str, str, str], torch.Tensor] = {}
        hetero_out: Dict[Tuple[str, str, str], torch.Tensor] = {}

        for edge_type, ei in edge_index_dict.items():
            key = _edge_key(edge_type)
            if key not in conv_dict:
                continue
            src, rel, dst = edge_type
            if src not in x_dict or dst not in x_dict:
                continue
            if ei.numel() == 0:
                continue
            conv = conv_dict[key]
            if src == dst:
                out, h, ht = conv(x_dict[src], ei)
            else:
                out, h, ht = conv((x_dict[src], x_dict[dst]), ei)
            homo_out[edge_type] = h
            hetero_out[edge_type] = ht
            out_acc[dst].append(out)

        merged: Dict[str, torch.Tensor] = {}
        for nt in self.node_types:
            outs = out_acc.get(nt, [])
            if len(outs) == 0:
                # No incoming edges — fall back to an input projection of the
                # current features so shape stays at hidden_channels for the
                # downstream BN/dropout (and so further layers can still run).
                if nt in x_dict:
                    merged[nt] = self.input_proj[nt](x_dict[nt])
                else:
                    merged[nt] = None
            elif len(outs) == 1:
                merged[nt] = outs[0]
            else:
                merged[nt] = torch.stack(outs, dim=0).sum(dim=0)

        return merged, homo_out, hetero_out

    def _finalize_aux_losses(
        self,
        accum_corr_sum: torch.Tensor,
        accum_corr_count: int,
        metapath_homo_layers: Dict[Tuple[str, str, str], List[torch.Tensor]],
        metapath_hetero_layers: Dict[Tuple[str, str, str], List[torch.Tensor]],
        positive_edges: Dict[Tuple[str, str, str], torch.Tensor],
        device: torch.device,
    ):
        """Turn the incremental corr accumulator and per-metapath layer caches
        into the final corr_loss / link_loss / aux_loss scalars.

        Reduction across (layer × edge_type) for L_corr and across metapaths
        for L_rec is controlled by `self.aux_reduction`:
          - "sum"  (default, paper-faithful): matches the reference impl's
                   `loss += alpha * dist_corr(...)` / `loss += beta * loss_link`
                   inner-loop accumulation. α, β sweep ranges should use the
                   paper's values {0, 0.1, ..., 0.5} directly.
          - "mean": averages instead of summing. Graph-size-invariant; requires
                   scaling α, β up by ~N_terms to match paper's effective
                   regularization strength.
        """
        # (a) Correlation loss (Eq. 8) over edge-type × layer terms.
        if accum_corr_count > 0:
            if self.aux_reduction == "sum":
                self.corr_loss = accum_corr_sum
            else:  # "mean"
                self.corr_loss = accum_corr_sum / accum_corr_count
        else:
            self.corr_loss = torch.zeros((), device=device)

        # (b) Link-prediction loss (Eq. 9) on masked startup-startup edges only.
        if self.beta > 0 and positive_edges:
            link_terms: List[torch.Tensor] = []
            for mp, pos_ei in positive_edges.items():
                homo_layers = metapath_homo_layers.get(mp, [])
                hetero_layers = metapath_hetero_layers.get(mp, [])
                if not homo_layers:
                    continue
                z_h = torch.cat(homo_layers, dim=1)    # [N_startup, L*d]
                z_ht = torch.cat(hetero_layers, dim=1)

                E_pos = pos_ei.size(1)
                N = z_h.size(0)
                neg_src = torch.randint(0, N, (E_pos,), device=device)
                neg_dst = torch.randint(0, N, (E_pos,), device=device)
                neg_ei = torch.stack([neg_src, neg_dst], dim=0)

                # Following the reference: both channels predict '1' on their
                # respective edges (positive for homo, random/likely-neg for
                # hetero). The asymmetry in the aggregation, not the target,
                # drives the disentanglement.
                pred_pos = self.edge_decoder(z_h, z_h, pos_ei)
                pred_neg = self.edge_decoder(z_ht, z_ht, neg_ei)
                pos_loss = F.binary_cross_entropy_with_logits(pred_pos, torch.ones_like(pred_pos))
                neg_loss = F.binary_cross_entropy_with_logits(pred_neg, torch.ones_like(pred_neg))
                link_terms.append(pos_loss + neg_loss)
            if link_terms:
                stacked = torch.stack(link_terms)
                self.link_loss = stacked.sum() if self.aux_reduction == "sum" else stacked.mean()
            else:
                self.link_loss = torch.zeros((), device=device)
        else:
            self.link_loss = torch.zeros((), device=device)

        self.aux_loss = self.alpha * self.corr_loss + self.beta * self.link_loss

    def _zero_aux_losses(self, device):
        self.corr_loss = torch.zeros((), device=device)
        self.link_loss = torch.zeros((), device=device)
        self.aux_loss = torch.zeros((), device=device)

    # ------------------------------------------------------------------ #
    # §6.2 channel-balance analysis hook
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def extract_channel_balance(
        self, x_dict, edge_index_dict, layer_idx: int = -1
    ) -> Dict[Tuple[str, str, str], Dict[str, float]]:
        """Per-startup-startup-metapath channel balance for ICDM §6.2 analysis.

        Runs a single eval-mode forward pass and, at the requested layer
        (default: final layer, index -1), computes for each startup-startup
        metapath `(startup, rel, startup)`:

            homo_norm    = mean over destination startups of ||h_homo||_2
            hetero_norm  = mean over destination startups of ||h_hetero||_2
            hetero_mass  = hetero_norm / (homo_norm + hetero_norm)
                          ∈ [0, 1] — fraction of signal energy in the hetero
                          channel. Parallels the "attention weight per metapath"
                          variable used for the five attention-based HGNNs in
                          §6.2 (Li et al. 2023's heterophily thesis predicts
                          heterophilic metapaths get higher hetero_mass).

        Does NOT apply any edge masking or label masking (pure eval path); the
        label embedding is applied if `use_label_propagation` is on, per paper
        §5.3 semantics. Returns a dict keyed by the canonical edge_type tuple.

        Args:
            x_dict: node feature dict as given to forward().
            edge_index_dict: edge index dict as given to forward().
            layer_idx: which conv layer to extract from. -1 = final layer.

        Returns:
            Dict[(src, rel, dst), {"homo_norm": float, "hetero_norm": float,
                                   "hetero_mass": float, "homo_mass": float}]
            over startup-startup metapaths only; empty for node types without
            incoming metapath edges at the requested layer.
        """
        was_training = self.training
        self.eval()
        device = x_dict["startup"].device

        # Normalize layer index.
        if layer_idx < 0:
            layer_idx = self.num_layers + layer_idx
        if not 0 <= layer_idx < self.num_layers:
            raise IndexError(
                f"Hetero2Net.extract_channel_balance: layer_idx {layer_idx} "
                f"out of range [0, {self.num_layers})"
            )

        # Shallow-copy x_dict; apply label embedding as forward() would.
        x_current = dict(x_dict)
        if (
            self.use_label_propagation
            and self.label_embedding is not None
            and self._label_prop_y is not None
        ):
            y_emb = self._compute_label_embedding(x_current["startup"].size(0), device)
            if y_emb is not None:
                x_current["startup"] = x_current["startup"] + y_emb

        # Run layers up to and including layer_idx, capturing that layer's
        # raw homo/hetero dict before BN/activation/dropout — matching the
        # tensor the aux-loss block operates on during training.
        target_homo_dict: Dict[Tuple[str, str, str], torch.Tensor] = {}
        target_hetero_dict: Dict[Tuple[str, str, str], torch.Tensor] = {}
        for li, conv in enumerate(self.convs):
            out_dict, homo_dict, hetero_dict = self._hetero_conv(
                conv, x_current, edge_index_dict
            )
            if li == layer_idx:
                target_homo_dict = homo_dict
                target_hetero_dict = hetero_dict
                break  # no need to propagate further
            # BN + activation + dropout for the intermediate layer.
            if li < self.num_layers - 1:
                out_dict = {
                    nt: self.feat_dropout(self.activation(self.bns[li][nt](x)))
                    for nt, x in out_dict.items() if x is not None
                }
            x_current = out_dict

        # Compute per-metapath L2 norms averaged over destination startups.
        stats: Dict[Tuple[str, str, str], Dict[str, float]] = {}
        for et in self.metapaths:
            h = target_homo_dict.get(et)
            ht = target_hetero_dict.get(et)
            if h is None or ht is None:
                continue
            hn = float(h.norm(p=2, dim=-1).mean().item())
            htn = float(ht.norm(p=2, dim=-1).mean().item())
            total = hn + htn
            stats[et] = {
                "homo_norm": hn,
                "hetero_norm": htn,
                "homo_mass": hn / total if total > 0 else 0.5,
                "hetero_mass": htn / total if total > 0 else 0.5,
            }

        if was_training:
            self.train()
        return stats

    @torch.no_grad()
    def extract_per_node_channel_norms(
        self, x_dict, edge_index_dict, layer_idx: int = -1
    ) -> Dict[Tuple[str, str, str], Dict[str, torch.Tensor]]:
        """Per-(startup, metapath) channel norms — node-level granularity.

        Unlike `extract_channel_balance()` which returns scalars per metapath,
        this returns per-startup vectors, enabling node-level correlation
        analysis with per-node local MLH (n = N_startups × N_metapaths pairs
        rather than just N_metapaths). This is a probe unique to Hetero2Net:
        attention-based HGNNs expose only per-metapath weights, not per-node
        channel mass.

        Args:
            x_dict: node feature dict.
            edge_index_dict: edge index dict.
            layer_idx: which DisenConv layer (default final, -1).

        Returns:
            Dict[(src, rel, dst), {
                "homo_norms": Tensor[N_startup] — ||h_homo[v]||_2 per startup,
                "hetero_norms": Tensor[N_startup] — ||h_hetero[v]||_2 per startup,
                "in_degree": Tensor[N_startup] — in-degree of each startup
                             via this metapath (useful for filtering /
                             weighting in downstream correlation).
            }] over startup-startup metapaths.
        """
        was_training = self.training
        self.eval()
        device = x_dict["startup"].device

        if layer_idx < 0:
            layer_idx = self.num_layers + layer_idx
        if not 0 <= layer_idx < self.num_layers:
            raise IndexError(
                f"Hetero2Net.extract_per_node_channel_norms: layer_idx "
                f"{layer_idx} out of range [0, {self.num_layers})"
            )

        x_current = dict(x_dict)
        if (
            self.use_label_propagation
            and self.label_embedding is not None
            and self._label_prop_y is not None
        ):
            y_emb = self._compute_label_embedding(x_current["startup"].size(0), device)
            if y_emb is not None:
                x_current["startup"] = x_current["startup"] + y_emb

        target_homo_dict: Dict[Tuple[str, str, str], torch.Tensor] = {}
        target_hetero_dict: Dict[Tuple[str, str, str], torch.Tensor] = {}
        for li, conv in enumerate(self.convs):
            out_dict, homo_dict, hetero_dict = self._hetero_conv(
                conv, x_current, edge_index_dict
            )
            if li == layer_idx:
                target_homo_dict = homo_dict
                target_hetero_dict = hetero_dict
                break
            if li < self.num_layers - 1:
                out_dict = {
                    nt: self.feat_dropout(self.activation(self.bns[li][nt](x)))
                    for nt, x in out_dict.items() if x is not None
                }
            x_current = out_dict

        N_startup = x_dict["startup"].size(0)
        result: Dict[Tuple[str, str, str], Dict[str, torch.Tensor]] = {}
        for et in self.metapaths:
            h = target_homo_dict.get(et)
            ht = target_hetero_dict.get(et)
            if h is None or ht is None:
                continue
            # L2 norm per startup — shape [N_startup].
            hn = h.norm(p=2, dim=-1).detach().cpu()
            htn = ht.norm(p=2, dim=-1).detach().cpu()
            # In-degree per startup via this edge_type.
            ei = edge_index_dict.get(et)
            in_deg = torch.zeros(N_startup, dtype=torch.long)
            if ei is not None and ei.numel() > 0:
                dst = ei[1].detach().cpu()
                in_deg.scatter_add_(0, dst, torch.ones_like(dst))
            result[et] = {
                "homo_norms": hn,
                "hetero_norms": htn,
                "in_degree": in_deg,
            }

        if was_training:
            self.train()
        return result


class VenGNN(BaseGNN):
    """Venture Graph Neural Network (Zhang et al. 2024, HICSS).

    Two branches over startup nodes, summed and fed to a classifier:

    Branch A - Fused Heterogeneous Graph Attention (Eq. 1):
        Per-metapath 2-layer GAT over (startup, rel, startup) edges using
        PathCount as edge weight. K attention heads are averaged within a
        metapath (paper's `concat=False` at head level, `concat=True` at
        metapath level). Concatenation across M metapaths gives a vector of
        dim M*hidden, projected down to `hidden` by a learnable linear.

    Branch B - Sampled Self-Attention (Eq. 2-3):
        Random walks of length L starting at every startup (sampled on the
        union of startup-startup metapath edges, with PathCount weights
        informing the edge set, not the walk sampler for v1). The L-length
        sequence of startup features along each walk is passed through a
        per-position self-attention block; position weights (W2) fuse L
        tokens into a single vector; sum across `b` walks gives g_i.

    Fusion: y = FNN(h + g); then dual masked_multi_task heads via _apply_heads.

    Skipped for v1 (paper is under-specified or not applicable):
    - Centrality encoding (paper shows it in Fig 3 without formula).
    - Transfer learning across funding rounds (we predict directly).
    - Weighted random walks (paper does not prescribe; uniform RW used).
    """

    def __init__(
        self,
        in_channels: int,          # startup feature dim
        hidden_channels: int,
        metadata,
        num_layers: int = 2,
        heads: int = 4,
        dropout: float = 0.2,
        rw_num_walks: int = 8,     # "b" in paper
        rw_walk_length: int = 5,   # "L" in paper
        activation_type: str = "relu",
        target_mode: str = "masked_multi_task",
        num_classes: int = 2,
        max_metapaths: int = 50,
        rw_seed: int = 0,
        use_paper_attention: bool = False,   # if True: paper's softmax(VV^T)V; else: nn.MultiheadAttention
        branch_mode: str = "full",           # "full" | "a_only" (GAT only) | "b_only" (RW self-attn only)
        paper_widening: bool = False,        # if True: keep M*hidden dim through Branch A (paper-faithful)
        gat_edge_mode: str = "feature",      # "feature" (edge_dim=1) | "multiplicative" (log(w) in softmax) | "none" (binary edges)
        rw_weighted: bool = False,           # weight random-walk step by PathCount (else uniform)
    ):
        super().__init__(hidden_channels, target_mode, num_classes, activation_type)
        self.metadata = metadata
        self.num_layers = num_layers
        self.heads = heads
        self.dropout = dropout
        self.rw_num_walks = rw_num_walks
        self.rw_walk_length = rw_walk_length
        self.rw_seed = rw_seed
        if branch_mode not in ("full", "a_only", "b_only"):
            raise ValueError(f"branch_mode must be full/a_only/b_only, got {branch_mode!r}")
        if gat_edge_mode not in ("feature", "multiplicative", "none"):
            raise ValueError(f"gat_edge_mode must be feature/multiplicative/none, got {gat_edge_mode!r}")
        self.branch_mode = branch_mode
        self.paper_widening = paper_widening
        self.gat_edge_mode = gat_edge_mode
        self.rw_weighted = rw_weighted

        # 1. Metapath selection: only (startup, *, startup).
        edge_types = metadata[1]
        self.metapaths: List[Tuple[str, str, str]] = [
            et for et in edge_types if et[0] == "startup" and et[2] == "startup"
        ][:max_metapaths]
        M = len(self.metapaths)
        if M == 0:
            raise ValueError("VenGNN: no (startup, *, startup) metapaths in metadata")
        self.M = M
        print(f"VenGNN: {M} metapaths, heads={heads}, b={rw_num_walks}, L={rw_walk_length}")

        # 2. Branch A: Per-metapath GAT layers.
        # Paper: K heads averaged within a metapath (concat=False at head level),
        # then metapaths concatenated giving dim M*hidden per layer.
        #
        # paper_widening=False (our default): project input to hidden once, each
        #   layer operates hidden -> hidden, a final branch_a_fuse reduces
        #   M*hidden -> hidden so Branch B / post_fuse stay at hidden.
        # paper_widening=True (paper-faithful): no shared input projection,
        #   Layer 1 GATConv takes raw X features, Layer 2 GATConv takes
        #   M*hidden input, branch output stays at M*hidden; Branch B and
        #   post_fuse widen to match.
        branch_dim = M * hidden_channels if paper_widening else hidden_channels
        self.branch_dim = branch_dim

        if paper_widening:
            self.input_proj = None
            layer_in_dims = [in_channels] + [M * hidden_channels] * (num_layers - 1)
        else:
            self.input_proj = torch.nn.Linear(in_channels, hidden_channels)
            layer_in_dims = [hidden_channels] * num_layers

        # GATConv variant depends on gat_edge_mode:
        # - "feature":        edge_dim=1, PathCount fed to attention MLP (current default)
        # - "multiplicative": custom subclass that adds log(PathCount) pre-softmax
        # - "none":           edge_dim=None, PathCount ignored entirely (binary edges)
        if gat_edge_mode == "multiplicative":
            conv_cls = _MultiplicativeEdgeWeightGATConv
            conv_edge_dim = None
        elif gat_edge_mode == "none":
            conv_cls = GATConv
            conv_edge_dim = None
        else:  # "feature"
            conv_cls = GATConv
            conv_edge_dim = 1

        self.gat_layers = torch.nn.ModuleList()
        for layer_idx in range(num_layers):
            layer = torch.nn.ModuleDict()
            for mp in self.metapaths:
                layer[self._mp_key(mp)] = conv_cls(
                    in_channels=layer_in_dims[layer_idx],
                    out_channels=hidden_channels,
                    heads=heads,
                    concat=False,
                    dropout=dropout,
                    edge_dim=conv_edge_dim,
                    add_self_loops=False,
                )
            self.gat_layers.append(layer)

        # Optional reduction after concat: only if NOT paper_widening.
        self.branch_a_fuse = (
            None if paper_widening else torch.nn.Linear(M * hidden_channels, hidden_channels)
        )

        # 3. Branch B: Sampled self-attention.
        # rw_proj goes to branch_dim so Branch A and B outputs can be summed.
        self.rw_proj = torch.nn.Linear(in_channels, branch_dim)       # W^1 (shared)
        self.use_paper_attention = use_paper_attention
        if use_paper_attention:
            # Paper's Eq. 2 exactly: softmax(V V^T / sqrt(d)) V, no Q/K/V projections.
            self.rw_attn = None
        else:
            # Standard multi-head attention (more expressive). embed_dim must be
            # divisible by num_heads; pick head count that divides branch_dim.
            mha_heads = max(1, heads // 2)
            while branch_dim % mha_heads != 0 and mha_heads > 1:
                mha_heads -= 1
            self.rw_attn = torch.nn.MultiheadAttention(
                embed_dim=branch_dim,
                num_heads=mha_heads,
                dropout=dropout,
                batch_first=True,
            )
        # W^2 in R^{L x 1}: learnable position weights that fuse the L tokens of a walk.
        self.rw_position_weights = torch.nn.Parameter(torch.ones(rw_walk_length) / rw_walk_length)

        # 4. Fusion + post-layernorm. Reduces branch_dim (M*hidden if widened,
        # else hidden) to hidden_channels so _apply_heads sees the expected dim.
        self.post_fuse = torch.nn.Sequential(
            torch.nn.Dropout(dropout),
            torch.nn.Linear(branch_dim, hidden_channels),
            torch.nn.LayerNorm(hidden_channels),
        )

        # Caches populated by precompute().
        self._walks: torch.Tensor = None   # [N_startup, b, L]
        self._edge_weight_cache = {}        # per-metapath edge weights on device

    def _mp_key(self, mp) -> str:
        return f"{mp[0]}__{mp[1]}__{mp[2]}"

    def precompute(self, x_dict, edge_index_dict):
        """Sample random walks once for all startups on the union adjacency.

        Uses PathCount-summed union so walks are restricted to startup-startup
        edges. Called by the Trainer before training (same hook as SeHGNN).
        """
        from .vengnn_precompute import build_union_startup_adjacency, sample_random_walks

        x_startup = x_dict["startup"]
        N = x_startup.size(0)
        device = x_startup.device

        # Gather edge weights from HeteroData-carried edge_weight attributes, if any.
        ew_dict = {mp: self._edge_weight_cache.get(mp) for mp in self.metapaths}

        ei_all, ew_all = build_union_startup_adjacency(edge_index_dict, self.metapaths, ew_dict)
        ei_all = ei_all.to(device)
        ew_all = ew_all.to(device)

        self._walks = sample_random_walks(
            start_nodes=torch.arange(N, device=device),
            edge_index=ei_all,
            edge_weight=ew_all,
            num_nodes=N,
            walks_per_node=self.rw_num_walks,
            walk_length=self.rw_walk_length,
            seed=self.rw_seed,
            weighted=self.rw_weighted,
        )
        print(
            f"VenGNN: sampled {self.rw_num_walks}x{self.rw_walk_length} walks "
            f"({'weighted' if self.rw_weighted else 'uniform'}) "
            f"for {N} startups, {ei_all.size(1)} union edges"
        )

    def set_metapath_edge_weights(self, weight_dict):
        """Register per-metapath edge_weight tensors (from HeteroData).

        Called by Trainer before precompute(): `weight_dict[mp] = [E_m] tensor`
        or None if the metapath has no weights.
        """
        self._edge_weight_cache = {mp: (w.detach() if w is not None else None) for mp, w in weight_dict.items()}

    def forward(self, x_dict, edge_index_dict, **kwargs):
        if self._walks is None:
            raise RuntimeError("VenGNN: precompute() must be called before forward.")
        x = x_dict["startup"]
        N = x.size(0)
        device = x.device

        # Branch A - Fused Heterogeneous Graph Attention
        h_a = None
        if self.branch_mode != "b_only":
            # If paper_widening, first-layer GAT takes raw features (no shared proj).
            h_cur = x if self.paper_widening else self.input_proj(x)
            for layer in self.gat_layers:
                per_mp_outputs = []
                for mp in self.metapaths:
                    ei = edge_index_dict[mp].to(device)
                    ew_raw = self._edge_weight_cache.get(mp)
                    ew_flat = None
                    if ew_raw is not None:
                        ew_flat = ew_raw.to(device).float()
                    # Dispatch on gat_edge_mode:
                    if self.gat_edge_mode == "feature":
                        ea = ew_flat.view(-1, 1) if ew_flat is not None else torch.ones(ei.size(1), 1, device=device)
                        out = layer[self._mp_key(mp)](h_cur, ei, edge_attr=ea)
                    elif self.gat_edge_mode == "multiplicative":
                        ew_in = ew_flat if ew_flat is not None else torch.ones(ei.size(1), device=device)
                        out = layer[self._mp_key(mp)](h_cur, ei, edge_weight=ew_in)
                    else:  # "none"
                        out = layer[self._mp_key(mp)](h_cur, ei)
                    per_mp_outputs.append(self.activation(out))
                h_concat = torch.cat(per_mp_outputs, dim=-1)  # [N, M*hidden]
                h_cur = h_concat if self.paper_widening else self.branch_a_fuse(h_concat)
                h_cur = F.dropout(h_cur, p=self.dropout, training=self.training)
            h_a = h_cur

        # Branch B - Sampled Self-Attention
        g_b = None
        if self.branch_mode != "a_only":
            walks = self._walks.to(device)  # [N, b, L]
            N_, b, L = walks.shape
            assert N_ == N and L == self.rw_walk_length
            walk_feats = x[walks.view(-1)].view(N, b, L, -1)  # [N, b, L, in_channels]
            walk_feats = self.rw_proj(walk_feats)               # [N, b, L, branch_dim]

            # Self-attention per walk. Flatten (N, b) into batch dim.
            wf = walk_feats.view(N * b, L, -1)
            if self.use_paper_attention:
                # Paper Eq. 2: softmax(V V^T / sqrt(d)) V, no Q/K/V projections.
                scale = 1.0 / (wf.size(-1) ** 0.5)
                scores = torch.matmul(wf, wf.transpose(-2, -1)) * scale
                attn_weights = torch.softmax(scores, dim=-1)
                attn_out = torch.matmul(attn_weights, wf)
            else:
                from torch.nn.attention import SDPBackend, sdpa_kernel
                with sdpa_kernel([SDPBackend.MATH]):
                    attn_out, _ = self.rw_attn(wf, wf, wf, need_weights=False)
            attn_out = attn_out.view(N, b, L, -1)

            pos_w = self.rw_position_weights.view(1, 1, L, 1)
            per_walk = (attn_out * pos_w).sum(dim=2)
            g_b = per_walk.sum(dim=1)

        # Fusion (one or both branches) and prediction heads.
        if h_a is not None and g_b is not None:
            h = h_a + g_b
        elif h_a is not None:
            h = h_a
        else:
            h = g_b

        h = self.post_fuse(h)
        return self._apply_heads(h)


class HeteroMLP(BaseGNN):
    """
    Heterogeneous MLP Baseline (Startup Features Only).
    
    This model serves as a strict baseline to evaluate the predictive power of intrinsic startup attributes
    without any graph connectivity information. It ignores all neighbor features (investors, founders, etc.)
    and only uses the features of the startup nodes themselves.
    
    Architecture:
    1. Feature Projection: Linearly projects startup features to a hidden dimension.
    2. Prediction: Passes the projected features through a standard MLP to make predictions.
    """
    def __init__(
        self,
        hidden_channels,
        target_mode="multi_prediction",
        num_classes=2,
        activation_type="relu",
        normalize=True,
        dropout=0.0,
        metadata=None, # Kept for interface consistency
    ):
        super().__init__(hidden_channels, target_mode, num_classes, activation_type)
        
        self.dropout = dropout
        
        # 1. Feature Projection: Project startup features to hidden_channels
        # We use torch_geometric.nn.Linear which supports lazy initialization (-1)
        self.startup_projection = torch_geometric.nn.Linear(-1, hidden_channels)

        # 2. MLP for final prediction
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(hidden_channels, hidden_channels),
            self.activation,
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_channels, hidden_channels),
            self.activation,
            torch.nn.Dropout(dropout),
        )
        
        if normalize:
            self.norms = torch.nn.ModuleList([
                torch_geometric.nn.BatchNorm(hidden_channels),
                torch_geometric.nn.BatchNorm(hidden_channels)
            ])
        else:
            self.norms = None

    def forward(self, x_dict, edge_index_dict):
        # 1. Extract and project startup features
        x = x_dict['startup']
        x = self.startup_projection(x)
        
        # 2. Pass through MLP
        if self.norms:
            x = self.norms[0](x)
            
        x = self.mlp[0](x) # Linear
        x = self.mlp[1](x) # Act
        x = self.mlp[2](x) # Dropout
        
        if self.norms:
            x = self.norms[1](x)
            
        x = self.mlp[3](x) # Linear
        x = self.mlp[4](x) # Act
        x = self.mlp[5](x) # Dropout
        
        return self._apply_heads(x)


class SageEncoder(torch.nn.Module):
    def __init__(
        self,
        hidden_channels,
        num_layers=2,
        activation_type="relu",
        normalize=True,
        dropout=0.3,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.dropout = dropout

        self.convs = torch.nn.ModuleList()

        if activation_type == "relu":
            self.activation = torch.nn.ReLU()
        elif activation_type == "prelu":
            self.activation = torch.nn.PReLU()
        else:
            raise ValueError("Unsupported activation type. Choose 'relu' or 'prelu'.")

        # Shared encoder
        for _ in range(num_layers):
            self.convs.append(SAGEConv((-1, -1), hidden_channels, normalize=normalize))

    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < self.num_layers - 1:  # Apply activation and dropout except for last layer
                x = self.activation(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        return x


class XGBoostAdapter:
    def __init__(
        self,
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        gamma=0,
        reg_alpha=0,
        reg_lambda=1,
        min_child_weight=1,
        objective="binary:logistic",
        tree_method="hist",
        target_mode="multi_prediction",
        num_classes=2,
        **kwargs
    ):
        self.params = {
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "learning_rate": learning_rate,
            "subsample": subsample,
            "colsample_bytree": colsample_bytree,
            "gamma": gamma,
            "reg_alpha": reg_alpha,
            "reg_lambda": reg_lambda,
            "min_child_weight": min_child_weight,
            "objective": objective,
            "tree_method": tree_method,
            **kwargs
        }
        self.target_mode = target_mode
        self.num_classes = num_classes
        self.model = None

    def fit(self, X, y, eval_set=None, **kwargs):
        import xgboost as xgb
        
        # Assumes binary classification per task or standard multi-class.
        self.model = xgb.XGBClassifier(**self.params)
        verbose = kwargs.get("verbose", False) # Default to False if not provided, or respect passed val
        self.model.fit(X, y, eval_set=eval_set, verbose=verbose)

    def predict_proba(self, X):
        return self.model.predict_proba(X)
    
    def predict(self, X):
        return self.model.predict(X)
    
    def to(self, device):
        # Dummy method for compatibility with Trainer
        return self
    
    def __call__(self, *args, **kwargs):
        # Dummy forward for compatibility if called inadvertently
        pass


class SageGNN(BaseGNN):
    def __init__(
        self,
        hidden_channels,
        num_layers=2,
        activation_type="relu",
        normalize=True,
        target_mode="multi_prediction",
        num_classes=4,
        dropout=0.3,
        metadata=None,
        aggr="mean",
    ):
        super().__init__(hidden_channels, target_mode, num_classes, activation_type)
        
        # Initialize Encoder
        self.encoder = SageEncoder(
            hidden_channels=hidden_channels,
            num_layers=num_layers,
            activation_type=activation_type,
            normalize=normalize,
            dropout=dropout,
        )
        
        # Wrap encoder with to_hetero
        if metadata is not None:
            self.encoder = to_hetero(self.encoder, metadata, aggr=aggr)

    def forward(self, x_dict, edge_index_dict):
        # Get embeddings from heterogeneous encoder
        # x_dict will contain embeddings for all node types
        embeddings_dict = self.encoder(x_dict, edge_index_dict)
        
        # Extract startup embeddings
        startup_x = embeddings_dict['startup']
        
        return self._apply_heads(startup_x)


class HAN(BaseGNN):
    """
    Heterogeneous Graph Attention Network (HAN) for startup success prediction.
    Implements multi-layer HAN with different output modes for binary, multi-class, or multi-task prediction.
    """
    def __init__(
        self,
        in_channels: Union[int, Dict[str, int]],
        hidden_channels: int,
        metadata,  # Graph metadata (node_types, edge_types)
        num_layers: int = 2,
        heads: int = 8,
        negative_slope: float = 0.2,
        dropout: float = 0.2,
        activation_type: str = "relu",
        target_mode: str = "binary_prediction",  # 'binary_prediction', 'multi_prediction', or 'multi_task'
        num_classes: int = 2,
    ):
        super().__init__(hidden_channels, target_mode, num_classes, activation_type)
        self.num_layers = num_layers
        
        # Multi-layer HAN convolutions
        self.convs = torch.nn.ModuleList()
        
        # First layer
        self.convs.append(
            HANConv(
                in_channels=in_channels,
                out_channels=hidden_channels,
                metadata=metadata,
                heads=heads,
                negative_slope=negative_slope,
                dropout=dropout,
            )
        )
        
        # Additional layers (all with same hidden dimensions)
        for _ in range(num_layers - 1):
            self.convs.append(
                HANConv(
                    in_channels=hidden_channels,
                    out_channels=hidden_channels,
                    metadata=metadata,
                    heads=heads,
                    negative_slope=negative_slope,
                    dropout=dropout,
                )
            )

        # Residual connection
        self.residual = True
        self.residual_proj = torch_geometric.nn.Linear(-1, hidden_channels)
        
        # Gating mechanism
        # If True, learns a gate to balance Residual (Self) vs HAN (Graph)
        self.use_gating = True 
        if self.use_gating:
            # Gate computes a weight z in [0, 1]
            # h_final = z * h_residual + (1-z) * h_graph
            self.gate_linear = torch.nn.Linear(2 * hidden_channels, 1)
            
        # Residual Dropout
        # Dropping out the residual forces the model to rely on the graph path
        self.residual_dropout = 0.5 

    def forward(self, x_dict: Dict[str, torch.Tensor], edge_index_dict: Dict[tuple, torch.Tensor]):
        """
        Forward pass through the HAN layers.
        """
        # Capture original startup features for residual
        x_startup_input = x_dict['startup']

        # Pass through HAN layers
        for i, conv in enumerate(self.convs):
            x_dict = conv(x_dict, edge_index_dict)
            
            # Apply activation AND dropout between layers (except last)
            if i < self.num_layers - 1:
                for node_type in x_dict:
                    if x_dict[node_type] is not None:
                        # 1. Activation
                        x_dict[node_type] = self.activation(x_dict[node_type])
                        
                        # 2. Feature Dropout
                        x_dict[node_type] = torch.nn.functional.dropout(
                            x_dict[node_type], p=0.2, training=self.training
                        )

        # Extract startup node embeddings (Graph Path)
        h_graph = x_dict['startup']
        
        # Calculate Residual Embedding (Self Path)
        h_residual = self.residual_proj(x_startup_input)
        
        # Apply Residual Connection
        if self.residual:
            if self.training and self.residual_dropout > 0:
                # Apply dropout to the residual path to force graph usage
                h_residual = torch.nn.functional.dropout(h_residual, p=self.residual_dropout, training=True)
            
            if self.use_gating:
                # Compute Gate z = sigmoid(Linear([h_residual, h_graph]))
                # Concatenate along feature dimension
                combined = torch.cat([h_residual, h_graph], dim=-1)
                z = torch.sigmoid(self.gate_linear(combined))
                
                # Store mean gate value for debugging
                # z near 1.0 means relying on RESIDUAL (Self)
                # z near 0.0 means relying on GRAPH (Neighbors)
                self.last_gate_mean = z.mean().item()
                
                # Weighted combination
                # z determines how much of the RESIDUAL (Self) to keep
                startup_x = z * h_residual + (1 - z) * h_graph
            else:
                # Simple addition
                startup_x = h_graph + h_residual
        else:
            startup_x = h_graph
        
        return self._apply_heads(startup_x)

    def get_embeddings(self, x_dict, edge_index_dict):
        """Extract startup node embeddings before the final classification head."""
        # Capture original startup features for residual
        x_startup_input = x_dict['startup']

        # Pass through HAN layers
        for i, conv in enumerate(self.convs):
            x_dict = conv(x_dict, edge_index_dict)
            
            # Apply activation AND dropout between layers (except last)
            if i < self.num_layers - 1:
                for node_type in x_dict:
                    if x_dict[node_type] is not None:
                        x_dict[node_type] = self.activation(x_dict[node_type])
                        x_dict[node_type] = torch.nn.functional.dropout(
                            x_dict[node_type], p=0.2, training=self.training
                        )

        # Return startup embeddings with residual
        h_graph = x_dict['startup']
        h_residual = self.residual_proj(x_startup_input)
        
        if self.residual:
            if self.use_gating:
                combined = torch.cat([h_residual, h_graph], dim=-1)
                z = torch.sigmoid(self.gate_linear(combined))
                startup_x = z * h_residual + (1 - z) * h_graph
            else:
                startup_x = h_graph + h_residual
        else:
            startup_x = h_graph
            
        return startup_x

    def get_semantic_attention_weights(self, x_dict, edge_index_dict):
        """
        Extract semantic attention weights (metapath importances) from the HAN layers.
        
        Args:
            x_dict: Dictionary of node features
            edge_index_dict: Dictionary of edge indices
            
        Returns:
            Dictionary containing weights per layer and per destination node type
        """
        weights_dict = {}
        
        # Pass through HAN layers
        for i, conv in enumerate(self.convs):

            out, details = conv(x_dict, edge_index_dict, return_semantic_attention_weights=True)

            layer_weights = {}
            for node_type, weight_tensor in details.items():
                # PyG's HANConv returns None for node types with no incoming edges
                if weight_tensor is None:
                    layer_weights[node_type] = {}
                    continue

                # Map weights to metapath names
                dest_metapaths = [
                    edge_type for edge_type in self.convs[i].metadata[1]
                    if edge_type[2] == node_type
                ]

                # Verify counts match
                if len(dest_metapaths) != len(weight_tensor):
                    print(f"Warning: Mismatch in metapath count for {node_type} in layer {i}")
                    continue
                    
                # Create dict {metapath_name: weight}
                node_weights = {}
                for j, edge_type in enumerate(dest_metapaths):
                    # edge_type is (src, rel, dst)
                    metapath_name = edge_type[1]
                    node_weights[metapath_name] = weight_tensor[j].item()
                    
                layer_weights[node_type] = node_weights
                
            weights_dict[f"layer_{i}"] = layer_weights
            
            # Update x_dict for next layer (same as in forward)
            x_dict = out
            if i < self.num_layers - 1:
                for node_type in x_dict:
                    if x_dict[node_type] is not None:
                        x_dict[node_type] = self.activation(x_dict[node_type])
                        x_dict[node_type] = torch.nn.functional.dropout(
                            x_dict[node_type], p=0.2, training=self.training
                        )
                        
        return weights_dict


class SimpleHGNConv(MessagePassing):
    """Simple-HGN (Lv et al., KDD 2021) attention layer — faithful PyG port.

    Mirrors the DGL original `myGATConv` from THUDM/HGB (NC/benchmark/methods/
    baseline/conv.py). The algorithmic ingredients, in one layer, are:

      - Learned edge-type embedding E[edge_type] (shared across layers).
      - Per-head attention logit: leaky_relu(<h_src, a_l> + <h_dst, a_r> + <E[t], a_e>).
      - Softmax per destination node over incoming edges.
      - Optional cross-layer residual attention blend: α = α·(1-a) + α_prev·a.
      - Optional per-layer feature residual: out += res_fc(x).
      - Optional activation and bias.

    This port is vendored (with adaptation) from the PyG implementation in
    H^2GB (Lin et al., KDD 2025, MIT license). Ref:
    https://github.com/junhongmit/H2GB/blob/main/H2GB/network/shgn_model.py
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        edge_channels: int,
        num_etypes: int,
        num_heads: int,
        feat_drop: float = 0.0,
        attn_drop: float = 0.0,
        negative_slope: float = 0.2,
        residual: bool = False,
        activation=None,
        bias: bool = False,
        alpha: float = 0.0,
    ):
        super().__init__(node_dim=0, aggr="add")
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.edge_channels = edge_channels
        self.num_heads = num_heads
        self.alpha = alpha

        self.edge_emb = nn.Embedding(num_etypes, edge_channels)
        self.fc = nn.Linear(in_channels, out_channels * num_heads, bias=False)
        self.fc_e = nn.Linear(edge_channels, edge_channels * num_heads, bias=False)
        self.attn_l = nn.Parameter(torch.empty(1, num_heads, out_channels))
        self.attn_r = nn.Parameter(torch.empty(1, num_heads, out_channels))
        self.attn_e = nn.Parameter(torch.empty(1, num_heads, edge_channels))
        self.feat_drop = nn.Dropout(feat_drop)
        self.attn_drop_layer = nn.Dropout(attn_drop)
        self.leaky_relu = nn.LeakyReLU(negative_slope)

        if residual:
            if in_channels != out_channels * num_heads:
                self.res_fc = nn.Linear(in_channels, num_heads * out_channels, bias=False)
            else:
                self.res_fc = nn.Identity()
        else:
            self.res_fc = None

        self.activation = activation
        self.bias = nn.Parameter(torch.zeros(1, num_heads, out_channels)) if bias else None

        self.reset_parameters()

    def reset_parameters(self):
        gain = nn.init.calculate_gain("relu")
        nn.init.xavier_normal_(self.fc.weight, gain=gain)
        nn.init.xavier_normal_(self.fc_e.weight, gain=gain)
        nn.init.xavier_normal_(self.attn_l, gain=gain)
        nn.init.xavier_normal_(self.attn_r, gain=gain)
        nn.init.xavier_normal_(self.attn_e, gain=gain)
        if isinstance(self.res_fc, nn.Linear):
            nn.init.xavier_normal_(self.res_fc.weight, gain=gain)
        self.edge_emb.reset_parameters()

    def forward(self, x, edge_index, edge_type, res_attn=None):
        # feat_drop comes BEFORE residual (matches DGL original: res_fc(h_dst) where h_dst = feat_drop(feat))
        x = self.feat_drop(x)
        h = self.fc(x).view(-1, self.num_heads, self.out_channels)

        e_feat = self.edge_emb(edge_type)
        e_feat = self.fc_e(e_feat).view(-1, self.num_heads, self.edge_channels)
        ee = (e_feat * self.attn_e).sum(dim=-1, keepdim=True)  # [E, H, 1]

        row, col = edge_index[0], edge_index[1]
        el = (h * self.attn_l).sum(dim=-1, keepdim=True)  # [N, H, 1]
        er = (h * self.attn_r).sum(dim=-1, keepdim=True)  # [N, H, 1]
        alpha = el[row] + er[col] + ee  # [E, H, 1]
        alpha = self.leaky_relu(alpha)
        alpha = pyg_softmax(alpha, col, num_nodes=x.size(0))
        alpha = self.attn_drop_layer(alpha)

        if res_attn is not None:
            alpha = alpha * (1 - self.alpha) + res_attn * self.alpha

        out = self.propagate(edge_index, x=h, alpha=alpha)  # [N, H, out_channels]

        if self.res_fc is not None:
            out = out + self.res_fc(x).view(-1, self.num_heads, self.out_channels)
        if self.bias is not None:
            out = out + self.bias
        if self.activation is not None:
            out = self.activation(out)

        # Detach the returned alpha so the next layer's residual-attention blend
        # is a stop-gradient edge (matches DGL's `graph.edata.pop('a').detach()`
        # in myGATConv). Grad still flows through `alpha` locally via propagate,
        # so this layer's attn params are trained from its own message passing —
        # only the cross-layer residual path is detached.
        return out, alpha.detach()

    def message(self, x_j, alpha):
        return x_j * alpha


class SimpleHGN(BaseGNN):
    """Simple-HGN (Lv et al., KDD 2021) for heterogeneous graphs.

    Ref: "Are we really making much progress? Revisiting, benchmarking, and
    refining heterogeneous graph neural networks", KDD 2021. Source:
    https://github.com/THUDM/HGB (DGL original). Algorithmic faithfulness:

      1. Per-node-type input projection (Linear -> hidden_channels). Mirrors
         `fc_list` in the original myGAT (one Linear per node-type input dim).
      2. Homogenize the heterogeneous graph: concatenate projected node
         features across types, build a unified edge_index with offset, and
         assign each edge an integer edge-type id into a learned embedding.
      3. Stack of `num_layers` SimpleHGNConv layers with shared edge-type
         embedding, residual attention blending across layers, and an elu
         activation on hidden layers (no activation on the final layer).
      4. Output layer: mean across heads (not flatten) -> [N, hidden_channels];
         no residual attention into the output layer (res_attn=None).
      5. Optional L2 normalization of startup embeddings (paper-faithful) —
         equivalent to tf.math.l2_normalize with an epsilon floor.
      6. Task-specific heads applied via BaseGNN._apply_heads (for
         binary/multi/masked_multi_task compatibility with the rest of the
         training pipeline).
    """

    def __init__(
        self,
        in_channels,              # Union[int, Dict[str, int]]
        hidden_channels: int,
        metadata,
        num_layers: int = 2,
        heads: int = 8,
        edge_dim: int = None,
        dropout: float = 0.0,
        attn_dropout: float = 0.0,
        negative_slope: float = 0.05,
        residual: bool = True,
        alpha: float = 0.05,
        l2_normalize: bool = True,
        bias: bool = False,
        activation_type: str = "elu",
        target_mode: str = "masked_multi_task",
        num_classes: int = 2,
    ):
        super().__init__(hidden_channels, target_mode, num_classes, activation_type)
        assert metadata is not None, "SimpleHGN requires heterogeneous metadata"

        node_types, edge_types = metadata
        self.node_types = list(node_types)
        self.edge_types = list(edge_types)
        self.num_relations = len(self.edge_types)
        self._rel_to_idx = {rel: i for i, rel in enumerate(self.edge_types)}

        self.num_heads = heads
        self.num_layers = num_layers
        self.l2_normalize = l2_normalize
        self.register_buffer("epsilon", torch.tensor(1e-12))

        edge_channels = edge_dim if edge_dim is not None else hidden_channels

        # Per-node-type input projection (myGAT's fc_list)
        if isinstance(in_channels, dict):
            self.input_proj = torch.nn.ModuleDict({
                nt: nn.Linear(in_channels[nt], hidden_channels)
                for nt in self.node_types
            })
        else:
            self.input_proj = torch.nn.ModuleDict({
                nt: nn.Linear(in_channels, hidden_channels)
                for nt in self.node_types
            })
        for lin in self.input_proj.values():
            nn.init.xavier_normal_(lin.weight, gain=1.414)

        # Hidden-layer activation (applied inside conv); None for the output layer
        if activation_type == "elu":
            hidden_act = F.elu
        elif activation_type == "relu":
            hidden_act = F.relu
        elif activation_type == "leaky_relu":
            hidden_act = F.leaky_relu
        elif activation_type == "prelu":
            # PReLU has parameters; use a fresh module shared at conv-level
            hidden_act = torch.nn.PReLU()
        else:
            raise ValueError(f"SimpleHGN: unsupported activation_type {activation_type!r}")

        # Conv stack. num_layers is the total count of SimpleHGNConv layers:
        #   num_layers == 1: single output conv (in=hidden_channels, no activation).
        #   num_layers >= 2: first layer (in=hidden_channels, residual hardcoded False per
        #                    DGL original myGAT) + (num_layers - 2) hidden layers
        #                    (in=hidden*heads) + output layer (in=hidden*heads, no activation).
        assert num_layers >= 1, f"SimpleHGN: num_layers must be >= 1, got {num_layers}"
        self.convs = torch.nn.ModuleList()

        def _make_conv(in_ch, act, res):
            return SimpleHGNConv(
                in_channels=in_ch,
                out_channels=hidden_channels,
                edge_channels=edge_channels,
                num_etypes=self.num_relations,
                num_heads=heads,
                feat_drop=dropout,
                attn_drop=attn_dropout,
                negative_slope=negative_slope,
                residual=res,
                activation=act,
                bias=bias,
                alpha=alpha,
            )

        if num_layers == 1:
            self.convs.append(_make_conv(hidden_channels, None, residual))
        else:
            self.convs.append(_make_conv(hidden_channels, hidden_act, False))
            for _ in range(num_layers - 2):
                self.convs.append(_make_conv(hidden_channels * heads, hidden_act, residual))
            self.convs.append(_make_conv(hidden_channels * heads, None, residual))

    def _homogenize(self, x_dict, edge_index_dict):
        node_offsets = {}
        offset = 0
        feats = []
        for nt in self.node_types:
            node_offsets[nt] = offset
            feats.append(x_dict[nt])
            offset += x_dict[nt].size(0)
        h = torch.cat(feats, dim=0)

        edge_indices, edge_type_ids = [], []
        for rel, ei in edge_index_dict.items():
            if ei.numel() == 0:
                continue
            if rel not in self._rel_to_idx:
                continue
            shifted = ei.clone()
            shifted[0] = shifted[0] + node_offsets[rel[0]]
            shifted[1] = shifted[1] + node_offsets[rel[2]]
            edge_indices.append(shifted)
            edge_type_ids.append(torch.full(
                (ei.size(1),), self._rel_to_idx[rel],
                dtype=torch.long, device=ei.device,
            ))
        if edge_indices:
            edge_index = torch.cat(edge_indices, dim=1)
            edge_type = torch.cat(edge_type_ids, dim=0)
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long, device=h.device)
            edge_type = torch.empty((0,), dtype=torch.long, device=h.device)
        return h, edge_index, edge_type, node_offsets

    def forward(self, x_dict, edge_index_dict, **kwargs):
        # Per-type input projection -> hidden_channels
        h_dict = {nt: self.input_proj[nt](x_dict[nt]) for nt in self.node_types}

        # Homogenize
        h, edge_index, edge_type, node_offsets = self._homogenize(h_dict, edge_index_dict)

        # Conv stack: flatten heads between hidden layers (matches original flatten(1))
        res_attn = None
        for l in range(len(self.convs) - 1):
            h, res_attn = self.convs[l](h, edge_index, edge_type, res_attn=res_attn)
            h = h.reshape(-1, self.num_heads * self.convs[l].out_channels)
        # Output layer: mean across heads (no res_attn fed through to output)
        h, _ = self.convs[-1](h, edge_index, edge_type, res_attn=None)
        h = h.mean(dim=1)  # [N_total, hidden_channels]

        # Extract startup embeddings
        s_off = node_offsets["startup"]
        s_n = x_dict["startup"].size(0)
        startup_x = h[s_off:s_off + s_n]

        # Paper-faithful L2 normalization
        if self.l2_normalize:
            norm = startup_x.norm(p=2, dim=1, keepdim=True).clamp(min=self.epsilon)
            startup_x = startup_x / norm

        return self._apply_heads(startup_x)


class FocalLoss(nn.Module):
    """
    Focal Loss Implementation
    Source: https://github.com/mathiaszinnen/focal_loss_torch
    """

    def __init__(self, gamma=0, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha, (float, int, long)):
            self.alpha = torch.Tensor([alpha, 1 - alpha])
        if isinstance(alpha, list):
            self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target):
        if input.dim() > 2:
            input = input.view(input.size(0), input.size(1), -1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1, 2)  # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1, input.size(2))  # N,H*W,C => N*H*W,C
        target = target.view(-1, 1)

        logpt = F.log_softmax(input, dim=-1)
        logpt = logpt.gather(1, target)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        if self.alpha is not None:
            if self.alpha.type() != input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0, target.data.view(-1))
            logpt = logpt * Variable(at)

        loss = -1 * (1 - pt) ** self.gamma * logpt
        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()

class RandomBaseline(BaseGNN):
    """
    Random Baseline: Generates random predictions.
    """
    def __init__(
        self,
        hidden_channels, # Ignored, kept for interface compatibility
        target_mode="multi_prediction",
        num_classes=2,
        **kwargs
    ):
        super().__init__(hidden_channels, target_mode, num_classes)
        
    def forward(self, x_dict, edge_index_dict, **kwargs):
        # Get batch size from startup features
        batch_size = x_dict['startup'].shape[0]
        device = x_dict['startup'].device
        
        # Generate random outputs
        # We generate logits
        if self.target_mode == "binary_prediction":
            # Output shape: [batch_size, 1]
            out = torch.randn(batch_size, 1, device=device)
            return {
                "startup": {
                    "output": out,
                    "embedding": x_dict['startup'] # Dummy embedding
                }
            }
        elif self.target_mode == "multi_prediction":
            # Output shape: [batch_size, num_classes]
            out = torch.randn(batch_size, self.num_classes, device=device)
            return {
                "startup": {
                    "output": out,
                    "embedding": x_dict['startup']
                }
            }
        elif self.target_mode == "multi_task":
            # Binary and Multi-class
            bin_out = torch.randn(batch_size, 1, device=device)
            multi_out = torch.randn(batch_size, self.num_classes, device=device)
            return {
                "binary_output": {"startup": bin_out.squeeze(-1)},
                "multi_class_output": {"startup": multi_out},
                "embedding": {"startup": x_dict['startup']}
            }
        elif self.target_mode == "masked_multi_task":
             # Tower 1: Momentum (Funding)
             out_mom = torch.randn(batch_size, 1, device=device)
             # Tower 2: Liquidity (Acq/IPO)
             out_liq = torch.randn(batch_size, 1, device=device)
             
             # Stack for convenient tensor access: [Mom, Liq]
             out_combined = torch.stack([out_mom.squeeze(-1), out_liq.squeeze(-1)], dim=1)
             
             return {
                 "masked_multi_task_output": {"startup": out_combined},
                 "embedding": {"startup": x_dict['startup']},
                 "out_mom": out_mom.squeeze(-1),
                 "out_liq": out_liq.squeeze(-1) 
             }
        else:
             raise ValueError(f"Unsupported target_mode: {self.target_mode}")


class Transformer(torch.nn.Module):
    """
    The transformer-based semantic fusion in SeHGNN.
    Adapted from: src/other/SeHGNN/hgb/model.py
    """
    def __init__(self, n_channels, num_heads=1, att_drop=0., act='none', temperature=1.0,
                 gamma_init=0.0, gamma_learnable=True):
        super(Transformer, self).__init__()
        self.n_channels = n_channels
        self.num_heads = num_heads
        self.temperature = temperature
        assert self.n_channels % (self.num_heads * 4) == 0

        self.query = torch.nn.Linear(self.n_channels, self.n_channels//4)
        self.key   = torch.nn.Linear(self.n_channels, self.n_channels//4)
        self.value = torch.nn.Linear(self.n_channels, self.n_channels)

        if gamma_learnable:
            self.gamma = torch.nn.Parameter(torch.tensor([float(gamma_init)]))
        else:
            self.register_buffer('gamma', torch.tensor([float(gamma_init)]))
        self._gamma_learnable = gamma_learnable
        self._gamma_init = gamma_init
        self.att_drop = torch.nn.Dropout(att_drop)
        if act == 'sigmoid':
            self.act = torch.nn.Sigmoid()
        elif act == 'relu':
            self.act = torch.nn.ReLU()
        elif act == 'leaky_relu':
            self.act = torch.nn.LeakyReLU(0.2)
        elif act == 'prelu':
            self.act = torch.nn.PReLU()
        elif act == 'none':
            self.act = lambda x: x
        else:
            raise ValueError(f'Unrecognized activation function {act} for class Transformer')

        self.reset_parameters()

    def reset_parameters(self):
        for k, v in self._modules.items():
            if hasattr(v, 'reset_parameters'):
                v.reset_parameters()
        if self._gamma_learnable:
            self.gamma.data.fill_(self._gamma_init)

    def forward(self, x, mask=None):
        import math
        B, M, C = x.size() # batchsize, num_metapaths, channels
        H = self.num_heads

        f = self.query(x).view(B, M, H, -1).permute(0,2,1,3) # [B, H, M, -1]
        g = self.key(x).view(B, M, H, -1).permute(0,2,3,1)   # [B, H, -1, M]
        h = self.value(x).view(B, M, H, -1).permute(0,2,1,3) # [B, H, M, -1]

        beta = F.softmax(self.act(f @ g / math.sqrt(f.size(-1)) / self.temperature), dim=-1) # [B, H, M, M(normalized)]
        beta = self.att_drop(beta)

        if mask is not None:
            beta = beta * mask.view(B, 1, 1, M)
            beta = beta / (beta.sum(-1, keepdim=True) + 1e-12)

        o = self.gamma * (beta @ h) # [B, H, M, -1]

        # Return output AND attention weights (beta)
        return o.permute(0,2,1,3).reshape((B, M, C)) + x, beta


class SeHGNN(BaseGNN):
    """
    Simple and Efficient Heterogeneous Graph Neural Network (SeHGNN).
    
    Adapts the SeHGNN architecture to work with NeighborLoader by performing
    on-the-fly mean aggregation of neighbors instead of pre-computation.
    """
    def __init__(
        self,
        in_channels: Union[int, Dict[str, int]],
        hidden_channels: int,
        metadata, # Graph metadata (node_types, edge_types)
        num_layers: int = 2, # Used for MLP layers
        heads: int = 1, # Transformer heads
        dropout: float = 0.5,
        input_drop: float = 0.1,
        att_drop: float = 0.0,
        activation_type: str = "relu",
        target_mode: str = "binary_prediction",
        num_classes: int = 2,
        aggregation_method: str = "mean",
        use_residual: bool = True,
        transformer_activation: str = "none",
        use_self_loop: bool = True,
        config: dict = None,  # Config for retrieval head
        model_name: str = "SeHGNN",  # Model name for config lookup
        attention_temperature: float = 1.0, # New param
        num_hops: int = 1, # Number of aggregation hops for startup→startup edges
        gamma_init: float = 0.0,  # Transformer gamma init value
        gamma_learnable: bool = True,  # Whether gamma is learnable
        channel_masking: bool = False,  # Mask empty metapath channels in attention
        use_layer_norm: bool = False,  # LayerNorm before fc_after_concat
        use_discrepancy: bool = False,  # B1: subtract projected self features from each neighbor channel (ego-alter)
        **kwargs
    ):
        super().__init__(hidden_channels, target_mode, num_classes, activation_type)

        self.metadata = metadata
        self.hidden_channels = hidden_channels
        self.dropout = dropout
        self.channel_masking = channel_masking
        self.input_drop = torch.nn.Dropout(input_drop)
        self.aggregation_method = aggregation_method
        self.use_residual = use_residual
        self.num_hops = num_hops
        self.use_self_loop = use_self_loop
        self.use_discrepancy = use_discrepancy
        
        # 1. Identify Metapaths (Edges pointing to 'startup')
        # We treat each edge type (src, rel, dst) where dst='startup' as a "metapath"
        # plus the 'startup' node itself (self-loop equivalent)
        self.metapaths = []
        
        # Add self (startup)
        if self.use_self_loop:
            self.metapaths.append("self")
        
        # Collect all candidate edges (both original and materialized metapaths)
        candidate_edges = []
        edge_types = metadata[1]
        for src, rel, dst in edge_types:
            if dst == 'startup':
                candidate_edges.append((src, rel, dst))
        
        # CAP TOTAL METAPATHS
        # Check config for max_metapaths to prevent OOM
        max_mps = 50  # Default fallback
        drop_list = []

        if config is not None:
            # Standardized config path: metapath_discovery.automatic
            auto_config = config.get('metapath_discovery', {}).get('automatic', {})
            max_mps = auto_config.get('max_metapaths', 50)

            # Standardized ablation path: metapath_discovery.automatic.ablation.drop_edges
            ablation_config = auto_config.get('ablation', {})
            drop_list = ablation_config.get('drop_edges', [])

        if len(candidate_edges) > 0:
            print(f"   Ablation: Dropping edge types: {drop_list}")
            
            # SELECTION LOGIC: 
            # 1. Filter out dropped edges
            # 2. Prioritize Base Edges (Keep ALL non-dropped)
            # 3. Fill remaining with Discovered
            
            discovered = []
            base = []
            
            for mp in candidate_edges:
                src, rel, dst = mp
                
                # Check for drop (Ablation)
                should_drop = False
                for drop_key in drop_list:
                    if drop_key in rel: # Partial match for relation name
                        should_drop = True
                        break
                if should_drop:
                    continue

                if "_via_" in rel or "_to_startup_" in rel:
                    discovered.append(mp)
                else:
                    base.append(mp)
            
            print(f"   Breakdown: {len(discovered)} discovered, {len(base)} base edges (after ablation)")
            
            final_selection = []
            
            # 1. Keep ALL Base Edges (User Request: "dont drop any normal edges")
            if len(base) > max_mps:
                print(f"⚠️ WARNING: Base edges ({len(base)}) exceed max_metapaths ({max_mps}). Capping base edges arbitrarily!")
                final_selection.extend(base[:max_mps])
            else:
                final_selection.extend(base)
            
            # 2. Fill remaining budget with Discovered paths
            remaining_slots = max_mps - len(final_selection)
            if remaining_slots > 0:
                num_discovered = min(len(discovered), remaining_slots)
                final_selection.extend(discovered[:num_discovered])
                print(f"   Selected: {len(base)} base + {num_discovered} discovered paths")
            else:
                print(f"   Selected: All {len(final_selection)} base edges (Discovered paths dropped due to capacity)")
                
            candidate_edges = final_selection
            
        self.metapaths.extend(candidate_edges)
                
        print(f"SeHGNN initialized with {len(self.metapaths)} channels: {self.metapaths}")
        print(f"  Aggregation: {self.aggregation_method}, Residual: {self.use_residual}, Transformer Act: {transformer_activation}, Self Loop: {self.use_self_loop}, Hops: {self.num_hops}")
        
        self.num_channels = len(self.metapaths)
        
        # 2. Feature Projection (Linear per metapath)
        # We need to project each input feature dimension to hidden_channels
        self.projectors = torch.nn.ModuleDict()
        
        # Self projector
        if isinstance(in_channels, dict):
            startup_dim = in_channels['startup']
        else:
            startup_dim = in_channels
            
        self.projectors["self"] = torch.nn.Linear(startup_dim, hidden_channels)
        
        # Neighbor projectors
        for mp in self.metapaths:
            if mp == "self": continue
            src, rel, dst = mp
            
            if isinstance(in_channels, dict):
                src_dim = in_channels[src]
            else:
                src_dim = in_channels
                
            # Key must be string for ModuleDict
            key = f"{src}__{rel}__{dst}"
            self.projectors[key] = torch.nn.Linear(src_dim, hidden_channels)
            
        # 3. Transformer Semantic Fusion
        # Fuses [Batch, Num_Metapaths, Hidden] -> [Batch, Num_Metapaths, Hidden]
        self.semantic_fusion = Transformer(
            hidden_channels,
            num_heads=heads,
            att_drop=att_drop,
            act=transformer_activation,
            temperature=attention_temperature,
            gamma_init=gamma_init,
            gamma_learnable=gamma_learnable
        )
        
        # 4. Optional LayerNorm before projection
        if use_layer_norm:
            self.pre_fc_norm = torch.nn.LayerNorm(self.num_channels * hidden_channels)
        else:
            self.pre_fc_norm = None

        # 5. Aggregation after Transformer
        # The original code does:
        # x = self.fc_after_concat(x.reshape(B, -1))
        # where fc_after_concat reduces (Num_Channels * Hidden) -> Hidden
        self.fc_after_concat = torch.nn.Linear(self.num_channels * hidden_channels, hidden_channels)
        
        # 5. Task MLP (Classifier)
        # Note: BaseGNN._init_heads handles the final output layer, so we just need the intermediate layers
        self.task_mlp = torch.nn.Sequential(
            torch.nn.PReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_channels, hidden_channels),
            torch.nn.BatchNorm1d(hidden_channels),
            torch.nn.PReLU(),
            torch.nn.Dropout(dropout)
        )
        
        # Residual connection (optional in original; added here for stability)
        if self.use_residual:
            self.res_fc = torch.nn.Linear(startup_dim, hidden_channels)
        
        # Initialize retrieval head if enabled (SimCLR/CLIP pattern)
        if config is not None:
            use_retrieval_head = config["models"][model_name].get("use_retrieval_head", False)
            retrieval_loss_type = config["train"]["loss"].get("retrieval_loss_type", "contrastive")
            
            if use_retrieval_head:
                # Initialize Projection Head
                self._init_retrieval_head(config, model_name, hidden_channels)
                
                # Initialize ArcFace Head if selected
                if retrieval_loss_type == "arcface":
                    arc_config = config["train"]["loss"].get("arcface", {})
                    margin = arc_config.get("margin", 0.5)
                    scale = arc_config.get("scale", 64.0)
                    
                    # num_retrieval_classes is passed via kwargs from Trainer
                    # (models have no graph access at init).
                    num_ret_classes = kwargs.get("num_retrieval_classes", None)
                    if num_ret_classes is None:
                        num_ret_classes = kwargs.get("num_classes", 100) # Fallback unlikely to be correct
                        print(f"⚠️ SeHGNN: num_retrieval_classes not provided for ArcFace. Using fallback: {num_ret_classes}")
                    
                    proj_dim = config["models"][model_name].get("retrieval_projection", {}).get("output_dim", 64)
                    
                    self.arcface_head = ArcFace(
                        in_features=proj_dim, 
                        out_features=num_ret_classes, 
                        s=scale, 
                        m=margin
                    )
                    print(f"✅ ArcFace Head Initialized: {proj_dim} -> {num_ret_classes} classes (m={margin}, s={scale})")
                    
            else:
                self.retrieval_proj = None
        else:
            self.retrieval_proj = None

        # Pre-aggregation cache (SeHGNN's key efficiency trick)
        self._cached_agg = None
        self._cached_channel_mask = None

    def precompute(self, x_dict, edge_index_dict):
        """Pre-compute neighbor aggregations once before training.

        This is the core SeHGNN optimization: aggregate raw neighbor features
        once, cache them, then only run projection + transformer during training.
        Since mean aggregation and linear projection commute, the result is
        mathematically identical to project-then-aggregate.
        """
        from torch_geometric.utils import scatter

        batch_size = x_dict['startup'].shape[0]
        device = x_dict['startup'].device
        cache = {}

        for mp in self.metapaths:
            if mp == "self":
                continue
            src, rel, dst = mp
            key = f"{src}__{rel}__{dst}"

            edge_index = edge_index_dict.get(mp)

            if edge_index is None:
                cache[key] = torch.zeros(batch_size, x_dict[src].shape[1],
                                         device=device)
            else:
                x_src = x_dict[src]
                src_idx, dst_idx = edge_index
                num_dst_nodes = x_dict[dst].shape[0]

                n_hops = self.num_hops if (src == dst == 'startup') else 1
                h = x_src
                for _hop in range(n_hops):
                    h_agg = scatter(h[src_idx], dst_idx, dim=0,
                                    dim_size=num_dst_nodes,
                                    reduce=self.aggregation_method)
                    if _hop < n_hops - 1:
                        h = h_agg

                cache[key] = h_agg[:batch_size]

        self._cached_agg = cache

        # Build channel mask: 1 where node has neighbors, 0 where all-zeros
        if self.channel_masking:
            mask = torch.ones(batch_size, len(self.metapaths), device=device)
            for ch_idx, mp in enumerate(self.metapaths):
                if mp == "self":
                    continue  # self channel always active
                key = f"{mp[0]}__{mp[1]}__{mp[2]}"
                agg = cache[key]
                # A node has no neighbors for this channel if its aggregation is all zeros
                is_empty = (agg.abs().sum(dim=-1) == 0).float()
                mask[:, ch_idx] = 1.0 - is_empty
            self._cached_channel_mask = mask
            num_masked = (mask == 0).sum().item()
            total = mask.numel()
            print(f"  SeHGNN: Channel mask built — {num_masked}/{total} entries masked ({100*num_masked/total:.1f}%)")
        else:
            self._cached_channel_mask = None

        print(f"  SeHGNN: Pre-aggregated {len(cache)} metapath channels (cached)")

    def clear_cache(self):
        """Clear pre-aggregation cache (needed for explanation/Captum)."""
        self._cached_agg = None
        self._cached_channel_mask = None

    def forward(self, x_dict, edge_index_dict, retrieval_labels=None, **kwargs):
        from torch_geometric.utils import scatter

        batch_size = x_dict['startup'].shape[0]
        device = x_dict['startup'].device
        use_cache = self._cached_agg is not None

        projected_features = []

        # A. Self Features
        h_self = x_dict['startup']
        h_self = self.input_drop(h_self)
        h_self = self.projectors["self"](h_self)  # [Batch, Hidden]
        projected_features.append(h_self)

        # B. Neighbor Features
        for mp in self.metapaths:
            if mp == "self":
                continue
            src, rel, dst = mp
            key = f"{src}__{rel}__{dst}"

            if use_cache:
                # Fast path: use pre-aggregated raw features, project now
                h_raw = self._cached_agg[key]
                h_agg = self.input_drop(h_raw)
                h_agg = self.projectors[key](h_agg)
            else:
                # Slow path: on-the-fly aggregation (for explanation / mini-batch)
                edge_index = edge_index_dict.get(mp)

                if edge_index is None:
                    h_agg = torch.zeros(batch_size, self.hidden_channels, device=device)
                else:
                    x_src = x_dict[src]
                    src_idx, dst_idx = edge_index
                    num_dst_nodes = x_dict[dst].shape[0]

                    h_src = self.input_drop(x_src)
                    h_src = self.projectors[key](h_src)

                    n_hops = self.num_hops if (src == dst == 'startup') else 1
                    for _hop in range(n_hops):
                        h_agg = scatter(h_src[src_idx], dst_idx, dim=0,
                                        dim_size=num_dst_nodes,
                                        reduce=self.aggregation_method)
                        if _hop < n_hops - 1:
                            h_src = h_agg

                    h_agg = h_agg[:batch_size]

                # Ensure x_dict[src] is in the graph for Captum autograd
                if src in x_dict:
                    dummy = (x_dict[src].sum() * 0.0)
                    h_agg = h_agg + dummy

            projected_features.append(h_agg)
            
        # Stack: [Batch, Num_Channels, Hidden]
        # NeighborLoader returns x_dict containing ALL sampled nodes; the first
        # `batch_size` nodes of `input_type` are the targets, so slice to batch_size.
        projected_features[0] = projected_features[0][:batch_size]

        # B1: ego-alter discrepancy — subtract projected self from each neighbor channel
        # Self channel stays as-is so the ego reference is preserved in the stack.
        if self.use_discrepancy and self.use_self_loop:
            h_self_ref = projected_features[0]
            for ch_idx in range(1, len(projected_features)):
                projected_features[ch_idx] = projected_features[ch_idx] - h_self_ref

        x = torch.stack(projected_features, dim=1)

        # 2. Build channel mask
        channel_mask = None
        if self.channel_masking:
            if self._cached_channel_mask is not None:
                channel_mask = self._cached_channel_mask[:batch_size]
            else:
                # On-the-fly mask for non-cached path (explanation / mini-batch)
                channel_mask = torch.ones(batch_size, len(self.metapaths), device=device)
                for ch_idx in range(len(projected_features)):
                    is_empty = (projected_features[ch_idx].abs().sum(dim=-1) == 0).float()
                    channel_mask[:, ch_idx] = 1.0 - is_empty

        # 3. Semantic Fusion (Transformer)
        x, attn_weights = self.semantic_fusion(x, mask=channel_mask)

        # Zero out masked channels after transformer so they don't leak into fc_after_concat
        if channel_mask is not None:
            x = x * channel_mask.unsqueeze(-1)

        # 4. Flatten/Project
        x = x.reshape(batch_size, -1)
        if self.pre_fc_norm is not None:
            x = self.pre_fc_norm(x)
        x = self.fc_after_concat(x) # [Batch, Hidden]
        
        # 4. Residual
        if self.use_residual:
            # Original SeHGNN: x = x + self.res_fc(features[self.tgt_type])
            h_self_orig = x_dict['startup'][:batch_size]
            x = x + self.res_fc(h_self_orig)
        
        # 5. Task MLP
        x = self.task_mlp(x)
        
        out = self._apply_heads(x, retrieval_labels=retrieval_labels)
        
        # Add attention weights to output
        if isinstance(out, dict):
            out['attention_weights'] = attn_weights
            out['metapath_names'] = self.metapaths
            
        return out


class DegreeCentralityBaseline(BaseGNN):
    """
    Degree Centrality Baseline: Uses normalized node degree as prediction score.
    """
    def __init__(
        self,
        hidden_channels, # Ignored
        degrees, # Tensor of global degrees
        target_mode="multi_prediction",
        num_classes=2,
        **kwargs
    ):
        super().__init__(hidden_channels, target_mode, num_classes)
        # Register degrees as buffer so it moves to device automatically
        self.register_buffer("degrees", degrees.float())
        self.max_degree = self.degrees.max()
        if self.max_degree == 0:
            self.max_degree = 1.0
            
    def forward(self, x_dict, edge_index_dict, batch=None, **kwargs):
        # We need node IDs to look up global degrees
        if batch is not None and hasattr(batch['startup'], 'n_id'):
            n_ids = batch['startup'].n_id
            # n_id maps local batch index -> global index
            batch_degrees = self.degrees[n_ids]
        else:
            batch_size = x_dict['startup'].shape[0]
            if batch_size == self.degrees.shape[0]:
                 batch_degrees = self.degrees
            else:
                batch_degrees = torch.zeros(batch_size, device=x_dict['startup'].device)

        # Normalize degrees to [0, 1]
        scores = batch_degrees / self.max_degree
        
        # Convert scores to logits-like or probability-like
        epsilon = 1e-6
        scores_clipped = torch.clamp(scores, epsilon, 1.0 - epsilon)
        logits = torch.log(scores_clipped / (1.0 - scores_clipped))
        
        out = logits.view(-1, 1)
        
        if self.target_mode == "binary_prediction":
            return {
                "startup": {
                    "output": out,
                    "embedding": x_dict['startup']
                }
            }
        elif self.target_mode == "multi_prediction":
            # Construct logits such that softmax(logits)[1] = score
            # logits = [0, logit]
            zeros = torch.zeros_like(out)
            multi_logits = torch.cat([zeros, out], dim=1) # Class 0: 0, Class 1: logit -> Softmax will give probs
            
            # If more classes, pad with -inf.
            if self.num_classes > 2:
                 padding = torch.full((out.shape[0], self.num_classes - 2), -float('inf'), device=out.device)
                 multi_logits = torch.cat([multi_logits, padding], dim=1)
                 
            return {
                "startup": {
                    "output": multi_logits,
                    "embedding": x_dict['startup']
                }
            }
        elif self.target_mode == "masked_multi_task":
             return {
                 "masked_multi_task_output": {"startup": torch.cat([out, out], dim=1)},
                 "embedding": {"startup": x_dict['startup']},
                 "out_mom": out.squeeze(-1),
                 "out_liq": out.squeeze(-1)
             }
        elif self.target_mode == "multi_task":
             return {
                "binary_output": {"startup": out.squeeze(-1)},
                "multi_class_output": {"startup": torch.cat([torch.zeros_like(out), out], dim=1)}, # Simplified
                "embedding": {"startup": x_dict['startup']}
            }
        else:
             raise ValueError(f"Unsupported target_mode: {self.target_mode}")


class LLMBaseline(BaseGNN):
    """
    LLM-based baseline for startup success prediction.
    Non-trainable - uses HuggingFace Transformers for inference only.

    Supports:
    - binary_prediction: Single task (liquidity by default)
    - masked_multi_task: Both momentum and liquidity predictions
    """

    def __init__(
        self,
        hidden_channels: int,  # Unused, kept for interface
        config: dict,
        raw_features_df,  # pandas DataFrame with startup features
        target_mode: str = "masked_multi_task",
        num_classes: int = 2,
        **kwargs
    ):
        super().__init__(hidden_channels, target_mode, num_classes)

        self.config = config
        self.raw_features_df = raw_features_df
        self.llm_config = config.get("models", {}).get("LLM", {})

        # Initialize predictor lazily
        self._predictor = None

    def _get_predictor(self):
        if self._predictor is None:
            from .llm_predictor import LLMPredictor
            self._predictor = LLMPredictor(
                model_name=self.llm_config.get("model_name", "meta-llama/Meta-Llama-3-8B-Instruct"),
                cache_dir=self.llm_config.get("cache_dir", "outputs/llm_cache"),
                temperature=self.llm_config.get("temperature", 0.0),
                device=self.llm_config.get("device", "auto"),
                torch_dtype=self.llm_config.get("torch_dtype", "auto"),
                load_in_8bit=self.llm_config.get("load_in_8bit", False),
                load_in_4bit=self.llm_config.get("load_in_4bit", False),
                token=self.llm_config.get("huggingface_token"),
                use_calibration=self.llm_config.get("use_calibration", True),
                use_chain_of_thought=self.llm_config.get("use_chain_of_thought", False),
                prompt_features=self.llm_config.get("prompt_features", "full"),
            )
        return self._predictor

    def forward(self, x_dict, edge_index_dict, node_indices=None, eval_mask=None, **kwargs):
        """
        Forward pass - generate predictions via LLM.

        Args:
            x_dict: Node features (used for batch size only)
            edge_index_dict: Ignored (LLM doesn't use graph structure)
            node_indices: Optional indices into raw_features_df
            eval_mask: Optional boolean mask indicating which nodes to predict for
        """
        total_nodes = x_dict['startup'].shape[0]
        device = x_dict['startup'].device
        predictor = self._get_predictor()

        # Determine which node indices to predict for
        if eval_mask is not None:
            # Only predict for masked nodes (efficient for val/test evaluation)
            mask_indices = eval_mask.nonzero(as_tuple=True)[0].cpu().numpy()
            print(f"  LLM: Predicting for {len(mask_indices)} masked nodes (out of {total_nodes} total)")
        elif node_indices is not None:
            mask_indices = node_indices.cpu().numpy()
        else:
            # Try to extract test_mask from batch graph_data passed via kwargs
            batch = kwargs.get('batch')
            if batch is not None and hasattr(batch.get('startup', {}), 'test_mask'):
                eval_mask = batch['startup'].test_mask
                mask_indices = eval_mask.nonzero(as_tuple=True)[0].cpu().numpy()
                print(f"  LLM: eval_mask was None, recovered test_mask from batch ({len(mask_indices)} nodes out of {total_nodes} total)")
            else:
                # Final fallback: predict for all nodes
                print(f"  WARNING: LLM predicting for ALL {total_nodes} nodes (no mask available)")
                mask_indices = range(min(total_nodes, len(self.raw_features_df)))

        # Optional: limit predictions for testing (models.LLM.max_predictions)
        max_preds = self.llm_config.get("max_predictions", 0)
        if max_preds > 0 and len(mask_indices) > max_preds:
            print(f"  LLM: Limiting to {max_preds} predictions (out of {len(mask_indices)})")
            mask_indices = mask_indices[:max_preds]

        # Get feature dicts only for nodes we need to predict
        feature_dicts = [self.raw_features_df.iloc[i].to_dict() for i in mask_indices]

        # Generate predictions based on target_mode
        if self.target_mode == "masked_multi_task":
            # Predict momentum for all eval nodes
            mom_probs = predictor.predict_batch(feature_dicts, "momentum")

            # Predict liquidity only for mature nodes (mask_liq == 1)
            batch = kwargs.get('batch')
            if batch is not None and hasattr(batch['startup'], 'y') and batch['startup'].y.shape[1] >= 4:
                maturity_flags = batch['startup'].y[mask_indices, 3].cpu().numpy()
                mature_local = maturity_flags == 1
                mature_feature_dicts = [fd for fd, m in zip(feature_dicts, mature_local) if m]
                print(f"  LLM: Predicting liquidity for {len(mature_feature_dicts)} mature nodes (skipping {len(feature_dicts) - len(mature_feature_dicts)} immature)")
                mature_liq_probs = predictor.predict_batch(mature_feature_dicts, "liquidity") if mature_feature_dicts else []
                # Expand back to full eval size (non-mature get 0.0 probability)
                liq_probs = []
                j = 0
                for m in mature_local:
                    if m:
                        liq_probs.append(mature_liq_probs[j])
                        j += 1
                    else:
                        liq_probs.append(0.0)
            else:
                liq_probs = predictor.predict_batch(feature_dicts, "liquidity")

            # Create full-size tensors and fill in predictions at masked positions
            mom_logits_full = torch.zeros(total_nodes, device=device)
            liq_logits_full = torch.zeros(total_nodes, device=device)

            mom_logits = self._probs_to_logits(mom_probs, device)
            liq_logits = self._probs_to_logits(liq_probs, device)

            # Fill in predictions at the correct positions
            mask_indices_tensor = torch.tensor(mask_indices, device=device, dtype=torch.long)
            mom_logits_full[mask_indices_tensor] = mom_logits
            liq_logits_full[mask_indices_tensor] = liq_logits

            return {
                "masked_multi_task_output": {
                    "startup": torch.stack([mom_logits_full, liq_logits_full], dim=1)
                },
                "embedding": {"startup": x_dict['startup']},
                "out_mom": mom_logits_full,
                "out_liq": liq_logits_full,
            }
        else:
            # Single task (liquidity)
            liq_probs = predictor.predict_batch(feature_dicts, "liquidity")

            # Create full-size tensor and fill in predictions
            liq_logits_full = torch.zeros(total_nodes, device=device)
            liq_logits = self._probs_to_logits(liq_probs, device)

            mask_indices_tensor = torch.tensor(mask_indices, device=device, dtype=torch.long)
            liq_logits_full[mask_indices_tensor] = liq_logits

            return {
                "startup": {
                    "output": liq_logits_full.unsqueeze(1),
                    "embedding": x_dict['startup'],
                }
            }

    def _probs_to_logits(self, probs: List, device) -> torch.Tensor:
        """Convert probabilities to logits."""
        eps = 1e-6
        probs_t = torch.tensor(probs, device=device, dtype=torch.float32)
        probs_t = torch.clamp(probs_t, eps, 1.0 - eps)
        return torch.log(probs_t / (1.0 - probs_t))
