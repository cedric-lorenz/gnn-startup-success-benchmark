"""Heterophily and homophily metrics for analyzing label agreement and
feature smoothness across typed edges of a heterogeneous graph."""
import torch


def _valid_label_mask(y: torch.Tensor, src: torch.Tensor, dst: torch.Tensor) -> torch.Tensor:
    """Edges whose endpoints carry valid labels (non-NaN, non-(-1))."""
    if torch.is_floating_point(y):
        return (~torch.isnan(y[src])) & (~torch.isnan(y[dst]))
    return (y[src] != -1) & (y[dst] != -1)


def calculate_edge_homophily(edge_index, y, node_mask=None):
    """Edge Homophily Ratio (MSHR / MLH).

        h = |{(u,v) in E : y_u == y_v}| / |E|

    Only considers edges where both endpoints have valid labels. Optionally
    restricts to edges where both endpoints satisfy ``node_mask`` (e.g. the
    mature subset for the Exit target).

    Args:
        edge_index: LongTensor[2, num_edges].
        y: Tensor[num_nodes] of class labels. NaN or -1 means missing.
        node_mask: Optional bool Tensor[num_nodes] selecting the subset over
            which homophily is measured.

    Returns:
        float in [0, 1], or ``None`` if no edges satisfy the constraints.
    """
    if edge_index.numel() == 0:
        return None

    src, dst = edge_index
    valid = _valid_label_mask(y, src, dst)
    if node_mask is not None:
        valid = valid & node_mask[src] & node_mask[dst]
    if valid.sum() == 0:
        return None

    src_v, dst_v = src[valid], dst[valid]
    matches = (y[src_v] == y[dst_v]).float().sum()
    return (matches / valid.float().sum()).item()


def calculate_class_homophily(edge_index, y, node_mask=None):
    """Class Homophily: mean per-class fraction of same-class neighbors.

        h_class = (1/|C|) * sum_c  |{(u,v) : y_u == y_v == c}|
                                     / |{(u,v) : y_u == c}|

    Less sensitive to base-rate imbalance than edge homophily.

    Args:
        edge_index: LongTensor[2, num_edges].
        y: Tensor[num_nodes] class labels.
        node_mask: Optional bool Tensor[num_nodes].

    Returns:
        float in [0, 1], or ``None`` if no valid edges.
    """
    if edge_index.numel() == 0:
        return None

    src, dst = edge_index
    valid = _valid_label_mask(y, src, dst)
    if node_mask is not None:
        valid = valid & node_mask[src] & node_mask[dst]
    if valid.sum() == 0:
        return None

    src, dst = src[valid], dst[valid]
    classes = torch.unique(y[src])
    per_class = []
    for c in classes:
        c_mask = (y[src] == c)
        if c_mask.sum() == 0:
            continue
        matches = (y[dst[c_mask]] == c).float().sum()
        per_class.append((matches / c_mask.float().sum()).item())
    if not per_class:
        return None
    return sum(per_class) / len(per_class)


def calculate_meta_path_dirichlet_energy(edge_index, x, node_mask=None,
                                          per_feature=False):
    """Meta-path Dirichlet energy (MDE), Def. 4 of Li et al. (2023).

        MDE(P) = (1/4) * sum_{(u,v) in E_P}
                   || x_u / sqrt(|N_P(u)|) - x_v / sqrt(|N_P(v)|) ||_2^2

    Measures feature-space heterophily along meta-path edges, normalized by
    the meta-path-induced endpoint degree. Complements MLH's label heterophily:
    a meta-path can be label-heterophilic yet feature-smooth, or vice versa.

    The squared-norm is additive over feature dimensions, so per-feature
    MDE is just the per-dimension summand; ``sum(per_feature=True) == MDE``.

    Args:
        edge_index: LongTensor[2, num_edges] for the meta-path edge set.
        x: FloatTensor[num_nodes, d] node features.
        node_mask: Optional bool Tensor[num_nodes] restricting to a node subset
            (both endpoints must satisfy).
        per_feature: If True, return a FloatTensor[d] of per-feature
            contributions (summing to the total MDE). Default False returns
            the scalar total as a Python float.

    Returns:
        float (default), FloatTensor[d] (per_feature=True), or ``None`` if no
        edges satisfy the constraints.
    """
    if edge_index.numel() == 0:
        return None

    src, dst = edge_index
    if node_mask is not None:
        valid = node_mask[src] & node_mask[dst]
        src, dst = src[valid], dst[valid]
        if src.numel() == 0:
            return None

    # Meta-path-induced degree (each appearance as an endpoint counts once).
    n = x.size(0)
    deg = torch.bincount(torch.cat([src, dst]), minlength=n).float().clamp(min=1.0)
    inv_sqrt_deg = deg.pow(-0.5)

    x_src = x[src] * inv_sqrt_deg[src].unsqueeze(1)
    x_dst = x[dst] * inv_sqrt_deg[dst].unsqueeze(1)
    diff = x_src - x_dst
    sq = 0.25 * (diff * diff)
    if per_feature:
        return sq.sum(dim=0).detach().cpu()
    return sq.sum().item()


def random_pair_baseline(y, node_mask=None):
    """Expected edge homophily under uniformly random edges: sum_c p_c^2.

    MLH below this baseline implies heterophily on the given label; above it
    implies homophily.

    Args:
        y: Tensor[num_nodes] class labels.
        node_mask: Optional bool Tensor[num_nodes].

    Returns:
        float in [0, 1].
    """
    if node_mask is not None:
        y = y[node_mask]
    valid = y[~torch.isnan(y)] if torch.is_floating_point(y) else y[y != -1]
    if valid.numel() == 0:
        return 0.0
    _, counts = torch.unique(valid, return_counts=True)
    p = counts.float() / counts.sum()
    return (p * p).sum().item()
