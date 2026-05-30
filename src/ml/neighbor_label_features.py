"""A4: Per-channel neighbor label statistics as node features.

For each startup-startup edge channel and each target column, compute the mean of
training-neighbor labels for every startup. Concatenate as additional scalar
features on startup.x. Leakage-safe: only training-labeled source nodes are used
(test and val nodes never act as label sources), and self-loops are excluded.

Missing entries (a startup has no qualifying training neighbors in a channel) are
filled with the global training label mean for that target.
"""
import torch
from torch_geometric.utils import scatter


def _compute_channel_label_means(graph_data, target_col, train_mask, y):
    """Return dict {relation_name: tensor [num_startups] of mean labels}."""
    num_startups = graph_data['startup'].num_nodes
    y_target = y[:, target_col].float()
    train_mask_bool = train_mask.bool()

    global_mean = float(y_target[train_mask_bool].mean().item())

    results = {}
    for et in graph_data.edge_types:
        src, rel, dst = et
        if src != 'startup' or dst != 'startup':
            continue
        edge_index = graph_data[et].edge_index
        if edge_index.numel() == 0:
            results[rel] = torch.full((num_startups,), global_mean, dtype=torch.float32)
            continue

        src_idx, dst_idx = edge_index
        valid = (src_idx != dst_idx) & train_mask_bool[src_idx]
        if valid.sum() == 0:
            results[rel] = torch.full((num_startups,), global_mean, dtype=torch.float32)
            continue

        v_src = src_idx[valid]
        v_dst = dst_idx[valid]

        weighted = y_target[v_src]
        ones = torch.ones_like(weighted)
        sum_per_dst = scatter(weighted, v_dst, dim=0, dim_size=num_startups, reduce='sum')
        count_per_dst = scatter(ones, v_dst, dim=0, dim_size=num_startups, reduce='sum')

        mean_per_dst = torch.where(
            count_per_dst > 0,
            sum_per_dst / count_per_dst.clamp(min=1.0),
            torch.full_like(sum_per_dst, global_mean),
        )
        results[rel] = mean_per_dst.float()
    return results


def add_neighbor_label_features(graph_data, config=None):
    """Append per-channel training-neighbor label means to startup.x."""
    startup = graph_data['startup']
    if not hasattr(startup, 'train_mask') or not hasattr(startup, 'y') or startup.y is None:
        print("⚠️ A4: skipping — startup lacks train_mask or y")
        return graph_data

    train_mask = startup.train_mask
    y = startup.y
    if y.ndim != 2 or y.size(1) < 2:
        print(f"⚠️ A4: y shape {tuple(y.shape)} does not have both targets; skipping")
        return graph_data

    target_cols = [0, 1]           # 0 = momentum / Next Funding Round, 1 = liquidity / Exit
    target_names = ['mom', 'liq']

    all_features = []
    first_rel_set = None
    for t_col, t_name in zip(target_cols, target_names):
        channel_means = _compute_channel_label_means(graph_data, t_col, train_mask, y)
        rels = sorted(channel_means.keys())
        if first_rel_set is None:
            first_rel_set = rels
        for rel in rels:
            all_features.append(channel_means[rel].unsqueeze(1))

    if not all_features:
        print("⚠️ A4: no startup-startup channels found; skipping")
        return graph_data

    extra = torch.cat(all_features, dim=1)
    prev_dim = startup.x.size(1)
    startup.x = torch.cat([startup.x, extra], dim=1)

    # The evaluator swaps in x_val_mask / x_test_mask / x_test_mask_original during eval
    # (see src/ml/eval.py). These parallel tensors must carry the same A4 features or
    # the projector will see the wrong input dim at val/test time.
    for split_attr in ("x_val_mask", "x_test_mask", "x_test_mask_original"):
        if hasattr(startup, split_attr):
            split_tensor = getattr(startup, split_attr)
            if split_tensor is not None and split_tensor.dim() == 2 and split_tensor.size(0) == extra.size(0):
                setattr(startup, split_attr, torch.cat([split_tensor, extra], dim=1))

    print(
        f"✅ A4 neighbor-label features: +{extra.size(1)} dims across "
        f"{len(first_rel_set)} channels × {len(target_cols)} targets. "
        f"startup.x: {prev_dim} → {startup.x.size(1)}"
    )
    return graph_data
