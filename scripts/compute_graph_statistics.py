"""
Compute comprehensive graph statistics for thesis experiments 21-24.

Usage:
    python scripts/compute_graph_statistics.py
    python scripts/compute_graph_statistics.py --graph-path outputs/pipeline_state/graph_data.pt
    python scripts/compute_graph_statistics.py --output outputs/graph_statistics.json
"""

import argparse
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import networkx as nx
import numpy as np
import pandas as pd
import torch
from scipy import stats as scipy_stats

from src.ml.heterophily_metrics import calculate_edge_homophily, calculate_class_homophily


def _get_num_nodes(data, ntype):
    """Get num_nodes for a node type, inferring from edges if PyG returns None."""
    nn = data[ntype].num_nodes
    if nn is not None:
        return nn
    # Infer from edge indices
    max_idx = -1
    for etype in data.edge_types:
        src_type, rel, dst_type = etype
        ei = data[etype].edge_index
        if ei.shape[1] == 0:
            continue
        if src_type == ntype:
            max_idx = max(max_idx, ei[0].max().item())
        if dst_type == ntype:
            max_idx = max(max_idx, ei[1].max().item())
    return max_idx + 1 if max_idx >= 0 else 0


# ============================================================
# Section 1: Node Statistics
# ============================================================

def compute_node_stats(data):
    """Per node type: count and feature dimensionality."""
    results = {}
    for ntype in sorted(data.node_types):
        num_nodes = _get_num_nodes(data, ntype)
        num_features = data[ntype].x.shape[1] if hasattr(data[ntype], 'x') and data[ntype].x is not None else 0
        results[ntype] = {"count": num_nodes, "features": num_features}
    return results


# ============================================================
# Section 2: Edge Statistics
# ============================================================

def compute_edge_stats(data):
    """Per edge type: count, density. Separate base vs metapath."""
    metapath_keys = set()
    if hasattr(data, 'metapath_definitions'):
        metapath_keys = set(data.metapath_definitions.keys())

    base_edges = {}
    metapath_edges = {}

    for etype in sorted(data.edge_types, key=str):
        src_type, rel, dst_type = etype
        num_edges = data[etype].edge_index.shape[1]
        num_src = _get_num_nodes(data, src_type)
        num_dst = _get_num_nodes(data, dst_type)
        density = num_edges / (num_src * num_dst) if (num_src * num_dst) > 0 else 0.0

        entry = {
            "src": src_type,
            "rel": rel,
            "dst": dst_type,
            "count": num_edges,
            "density": density,
        }

        # Use composite key to avoid collisions (e.g., multiple 'based_in' edges
        # for startup→city, investor→city, university→city)
        key = f"{src_type}-{rel}-{dst_type}"
        if etype in metapath_keys:
            metapath_edges[key] = entry
        else:
            base_edges[key] = entry

    return {"base": base_edges, "metapath": metapath_edges}


# ============================================================
# Section 3: Degree Distributions
# ============================================================

def compute_degree_distributions(data):
    """Per node type: degree stats across all edge types combined, plus per key edge type."""
    # Accumulate degrees per node type
    degree_accum = defaultdict(lambda: None)

    for etype in data.edge_types:
        src_type, rel, dst_type = etype
        edge_index = data[etype].edge_index
        if edge_index.shape[1] == 0:
            continue
        num_src = _get_num_nodes(data, src_type)
        num_dst = _get_num_nodes(data, dst_type)

        # Source-side degrees
        src_deg = torch.zeros(num_src, dtype=torch.long)
        src_indices = edge_index[0]
        src_deg.scatter_add_(0, src_indices, torch.ones_like(src_indices))

        if degree_accum[src_type] is None:
            degree_accum[src_type] = torch.zeros(num_src, dtype=torch.long)
        degree_accum[src_type] += src_deg

        # Destination-side degrees (skip if same type to avoid double counting on self-loops)
        if src_type != dst_type:
            dst_deg = torch.zeros(num_dst, dtype=torch.long)
            dst_indices = edge_index[1]
            dst_deg.scatter_add_(0, dst_indices, torch.ones_like(dst_indices))

            if degree_accum[dst_type] is None:
                degree_accum[dst_type] = torch.zeros(num_dst, dtype=torch.long)
            degree_accum[dst_type] += dst_deg
        else:
            # For self-type edges, add dst side too
            dst_deg = torch.zeros(num_dst, dtype=torch.long)
            dst_indices = edge_index[1]
            dst_deg.scatter_add_(0, dst_indices, torch.ones_like(dst_indices))
            degree_accum[dst_type] += dst_deg

    # Compute summary stats
    overall = {}
    for ntype in sorted(data.node_types):
        deg = degree_accum.get(ntype)
        if deg is None:
            nn = _get_num_nodes(data, ntype)
            deg = torch.zeros(max(nn, 1), dtype=torch.long)
        deg_np = deg.numpy().astype(float)
        overall[ntype] = {
            "mean": float(np.mean(deg_np)),
            "median": float(np.median(deg_np)),
            "std": float(np.std(deg_np)),
            "min": int(np.min(deg_np)),
            "max": int(np.max(deg_np)),
            "zero_degree": int((deg == 0).sum().item()),
        }

    # Per edge type startup degrees (source side)
    per_edge_type = {}
    for etype in data.edge_types:
        src_type, rel, dst_type = etype
        if src_type != "startup" and dst_type != "startup":
            continue
        edge_index = data[etype].edge_index
        if edge_index.shape[1] == 0:
            continue
        # Compute startup-side degree
        num = _get_num_nodes(data, "startup")
        if src_type == "startup":
            deg = torch.zeros(num, dtype=torch.long)
            deg.scatter_add_(0, edge_index[0], torch.ones_like(edge_index[0]))
        else:
            deg = torch.zeros(num, dtype=torch.long)
            deg.scatter_add_(0, edge_index[1], torch.ones_like(edge_index[1]))
        deg_np = deg.numpy().astype(float)
        per_edge_type[f"{src_type}-{rel}-{dst_type}"] = {
            "mean": float(np.mean(deg_np)),
            "median": float(np.median(deg_np)),
            "std": float(np.std(deg_np)),
            "max": int(np.max(deg_np)),
        }

    return {"overall": overall, "per_edge_type": per_edge_type}


# ============================================================
# Section 4: Connected Components
# ============================================================

def compute_connected_components(data):
    """Build full heterogeneous graph as undirected NX graph, report connectivity."""
    G = nx.Graph()

    # Map (node_type, local_idx) -> global_idx
    offset = {}
    current = 0
    for ntype in sorted(data.node_types):
        offset[ntype] = current
        n = _get_num_nodes(data, ntype)
        G.add_nodes_from(range(current, current + n))
        current += n

    startup_offset = offset["startup"]
    num_startups = _get_num_nodes(data, "startup")
    startup_global_ids = set(range(startup_offset, startup_offset + num_startups))

    for etype in data.edge_types:
        src_type, rel, dst_type = etype
        ei = data[etype].edge_index
        if ei.shape[1] == 0:
            continue
        ei_np = ei.numpy()
        src_global = ei_np[0] + offset[src_type]
        dst_global = ei_np[1] + offset[dst_type]
        edges = list(zip(src_global.tolist(), dst_global.tolist()))
        G.add_edges_from(edges)

    components = list(nx.connected_components(G))
    num_components = len(components)
    largest = max(components, key=len)

    # Startup-specific
    startups_in_largest = len(startup_global_ids & largest)
    isolated_startups = sum(
        1 for s in startup_global_ids if G.degree(s) == 0
    )

    return {
        "total_nodes": G.number_of_nodes(),
        "total_edges_undirected": G.number_of_edges(),
        "num_components": num_components,
        "largest_component_size": len(largest),
        "largest_component_pct": len(largest) / G.number_of_nodes() * 100,
        "startups_in_largest_component": startups_in_largest,
        "startups_in_largest_pct": startups_in_largest / num_startups * 100,
        "isolated_startups": isolated_startups,
        "component_size_distribution": sorted(
            [len(c) for c in components], reverse=True
        )[:20],  # Top 20 largest
    }


# ============================================================
# Section 5: Homophily
# ============================================================

def compute_homophily(data):
    """Edge + class homophily for startup↔startup edge types."""
    y = data["startup"].y
    if y.dim() > 1:
        y = y[:, 0]  # Use first target column

    # Binarize if float
    if torch.is_floating_point(y):
        y = (y > 0.5).long()

    results = {}
    for etype in data.edge_types:
        src_type, rel, dst_type = etype
        # Only meaningful when both endpoints are startups
        if src_type != "startup" or dst_type != "startup":
            continue

        edge_index = data[etype].edge_index
        if edge_index.shape[1] == 0:
            continue

        edge_h = calculate_edge_homophily(edge_index, y)
        class_h = calculate_class_homophily(edge_index, y)

        results[rel] = {
            "edge_homophily": round(edge_h, 4),
            "class_homophily": round(class_h, 4),
            "num_edges": edge_index.shape[1],
        }

    return results


# ============================================================
# Section 6: Class Balance
# ============================================================

def compute_class_balance(data):
    """Positive/negative counts overall and per split."""
    y = data["startup"].y
    results = {}

    if y.dim() == 1:
        targets = {"target": y}
    else:
        targets = {f"target_{i}": y[:, i] for i in range(y.shape[1])}

    for name, col in targets.items():
        valid = ~torch.isnan(col) if torch.is_floating_point(col) else (col != -1)
        col_valid = col[valid]
        pos = (col_valid > 0.5).sum().item() if torch.is_floating_point(col_valid) else (col_valid == 1).sum().item()
        total = valid.sum().item()
        neg = total - pos

        target_result = {
            "overall": {
                "positive": pos,
                "negative": neg,
                "total": total,
                "positive_rate": round(pos / total, 4) if total > 0 else 0,
            }
        }

        # Per split
        for split in ["train_mask", "val_mask", "test_mask"]:
            if not hasattr(data["startup"], split):
                continue
            mask = getattr(data["startup"], split)
            split_valid = valid & mask
            split_col = col[split_valid]
            s_pos = (split_col > 0.5).sum().item() if torch.is_floating_point(split_col) else (split_col == 1).sum().item()
            s_total = split_valid.sum().item()
            s_neg = s_total - s_pos
            target_result[split.replace("_mask", "")] = {
                "positive": s_pos,
                "negative": s_neg,
                "total": s_total,
                "positive_rate": round(s_pos / s_total, 4) if s_total > 0 else 0,
            }

        results[name] = target_result

    return results


# ============================================================
# Section 7: Label-Degree Correlation
# ============================================================

def compute_label_degree_correlation(data):
    """Mean degree by class + point-biserial correlation."""
    y = data["startup"].y
    if y.dim() > 1:
        y = y[:, 0]

    num_startups = _get_num_nodes(data, "startup")
    total_degree = torch.zeros(num_startups, dtype=torch.long)

    for etype in data.edge_types:
        src_type, rel, dst_type = etype
        edge_index = data[etype].edge_index
        if edge_index.shape[1] == 0:
            continue
        if src_type == "startup":
            deg = torch.zeros(num_startups, dtype=torch.long)
            deg.scatter_add_(0, edge_index[0], torch.ones_like(edge_index[0]))
            total_degree += deg
        if dst_type == "startup":
            deg = torch.zeros(num_startups, dtype=torch.long)
            deg.scatter_add_(0, edge_index[1], torch.ones_like(edge_index[1]))
            total_degree += deg

    # Filter valid labels
    if torch.is_floating_point(y):
        valid = ~torch.isnan(y)
    else:
        valid = y != -1

    y_valid = y[valid].float().numpy()
    deg_valid = total_degree[valid].float().numpy()
    labels_binary = (y_valid > 0.5).astype(float)

    pos_mask = labels_binary == 1
    neg_mask = labels_binary == 0

    result = {
        "mean_degree_positive": float(np.mean(deg_valid[pos_mask])) if pos_mask.any() else 0,
        "mean_degree_negative": float(np.mean(deg_valid[neg_mask])) if neg_mask.any() else 0,
        "median_degree_positive": float(np.median(deg_valid[pos_mask])) if pos_mask.any() else 0,
        "median_degree_negative": float(np.median(deg_valid[neg_mask])) if neg_mask.any() else 0,
    }

    # Point-biserial correlation
    if len(np.unique(labels_binary)) == 2 and len(labels_binary) > 2:
        corr, pvalue = scipy_stats.pointbiserialr(labels_binary, deg_valid)
        result["point_biserial_r"] = round(float(corr), 4)
        result["point_biserial_p"] = float(pvalue)
    else:
        result["point_biserial_r"] = None
        result["point_biserial_p"] = None

    return result


# ============================================================
# Section 8: Feature Coverage
# ============================================================

def compute_feature_coverage(data):
    """Per node type: % non-zero, % non-NaN features."""
    results = {}
    for ntype in sorted(data.node_types):
        if not hasattr(data[ntype], 'x') or data[ntype].x is None:
            continue
        x = data[ntype].x.float()
        num_nodes, num_feats = x.shape
        total_cells = num_nodes * num_feats

        nan_count = torch.isnan(x).sum().item()
        zero_count = (x == 0).sum().item()
        non_nan_cells = total_cells - nan_count

        results[ntype] = {
            "num_nodes": num_nodes,
            "num_features": num_feats,
            "pct_non_nan": round((total_cells - nan_count) / total_cells * 100, 2) if total_cells > 0 else 0,
            "pct_non_zero": round((non_nan_cells - zero_count) / total_cells * 100, 2) if total_cells > 0 else 0,
            "pct_zero": round(zero_count / total_cells * 100, 2) if total_cells > 0 else 0,
        }

        # Per-feature coverage (only for startup to keep JSON manageable)
        if ntype == "startup" and hasattr(data, 'feat_labels') and "startup" in data.feat_labels:
            feat_names = data.feat_labels["startup"]
            per_feat = {}
            for i, fname in enumerate(feat_names):
                if i >= num_feats:
                    break
                col = x[:, i]
                col_nan = torch.isnan(col).sum().item()
                col_zero = (col == 0).sum().item()
                per_feat[fname] = {
                    "pct_non_zero": round((num_nodes - col_nan - col_zero) / num_nodes * 100, 2),
                    "pct_non_nan": round((num_nodes - col_nan) / num_nodes * 100, 2),
                }
            results[ntype]["per_feature"] = per_feat

    return results


# ============================================================
# Section 9: Temporal Distribution
# ============================================================

def compute_temporal_distribution(data, csv_path):
    """Founding year distribution from raw CSV."""
    if not os.path.exists(csv_path):
        return {"error": f"CSV not found: {csv_path}"}

    df = pd.read_csv(csv_path, usecols=["founded_on"], low_memory=False)
    df["founded_on"] = pd.to_datetime(df["founded_on"], errors="coerce")
    df["year"] = df["founded_on"].dt.year

    # Match to graph size (preprocessing may filter rows)
    num_startups = _get_num_nodes(data, "startup")
    # The CSV may have more rows than the graph if date filtering was applied.
    # Report CSV-level stats, which describe the dataset.

    valid_years = df["year"].dropna()

    year_counts = Counter(valid_years.astype(int).tolist())
    sorted_years = sorted(year_counts.keys())

    result = {
        "total_rows_csv": len(df),
        "total_with_year": len(valid_years),
        "num_startups_in_graph": num_startups,
        "year_range": [int(sorted_years[0]), int(sorted_years[-1])] if sorted_years else [],
        "median_year": int(valid_years.median()) if len(valid_years) > 0 else None,
    }

    # 5-year bins
    if len(valid_years) > 0:
        min_yr = int(sorted_years[0])
        max_yr = int(sorted_years[-1])
        bins = list(range((min_yr // 5) * 5, max_yr + 6, 5))
        bin_counts, bin_edges = np.histogram(valid_years.values, bins=bins)
        result["bins_5yr"] = {
            f"{int(bin_edges[i])}-{int(bin_edges[i+1]-1)}": int(bin_counts[i])
            for i in range(len(bin_counts))
        }

    # Per-class breakdown (if we can match labels)
    y = data["startup"].y
    if y.dim() > 1:
        y = y[:, 0]
    if len(valid_years) >= num_startups:
        years_graph = valid_years.iloc[:num_startups].values
        labels = y.numpy()
        if np.issubdtype(labels.dtype, np.floating):
            pos_mask = labels > 0.5
        else:
            pos_mask = labels == 1
        valid_label = ~np.isnan(labels) if np.issubdtype(labels.dtype, np.floating) else (labels != -1)
        valid_year_arr = ~np.isnan(years_graph.astype(float))
        both_valid = valid_label & valid_year_arr

        pos_years = years_graph[both_valid & pos_mask]
        neg_years = years_graph[both_valid & ~pos_mask]

        result["median_year_positive"] = int(np.median(pos_years)) if len(pos_years) > 0 else None
        result["median_year_negative"] = int(np.median(neg_years)) if len(neg_years) > 0 else None

    return result


# ============================================================
# Section 10: Feature Dimensionality
# ============================================================

def compute_feature_dimensionality(data):
    """Feature dimensions per node type from the HeteroData object."""
    results = {}
    for ntype in sorted(data.node_types):
        entry = {"num_nodes": _get_num_nodes(data, ntype)}
        if hasattr(data[ntype], 'x') and data[ntype].x is not None:
            entry["total_features"] = data[ntype].x.shape[1]
        else:
            entry["total_features"] = 0

        # Feature names from feat_labels
        if hasattr(data, 'feat_labels') and ntype in data.feat_labels and data.feat_labels[ntype] is not None:
            entry["feature_names"] = data.feat_labels[ntype]
            entry["named_features"] = len(data.feat_labels[ntype])
        else:
            entry["named_features"] = 0

        results[ntype] = entry
    return results


# ============================================================
# Printing helpers
# ============================================================

def _print_header(title):
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")


def _print_table(headers, rows, col_widths=None):
    if col_widths is None:
        col_widths = [max(len(str(h)), max((len(str(r[i])) for r in rows), default=0)) + 2
                      for i, h in enumerate(headers)]
    header_line = "".join(str(h).ljust(w) for h, w in zip(headers, col_widths))
    print(header_line)
    print("-" * sum(col_widths))
    for row in rows:
        print("".join(str(c).ljust(w) for c, w in zip(row, col_widths)))


def print_all(results):
    """Print formatted tables for all sections."""

    # Section 1: Node Stats
    _print_header("1. Node Statistics")
    rows = [(ntype, d["count"], d["features"]) for ntype, d in results["node_stats"].items()]
    _print_table(["Node Type", "Count", "Features"], rows)
    total_nodes = sum(d["count"] for d in results["node_stats"].values())
    print(f"\nTotal nodes: {total_nodes:,}")

    # Section 2: Edge Stats
    _print_header("2. Edge Statistics (Base)")
    rows = [(d["src"], d["rel"], d["dst"], f"{d['count']:,}", f"{d['density']:.6f}")
            for d in results["edge_stats"]["base"].values()]
    _print_table(["Src", "Relation", "Dst", "Count", "Density"], rows)
    base_total = sum(d["count"] for d in results["edge_stats"]["base"].values())
    print(f"\nTotal base edges: {base_total:,}")

    if results["edge_stats"]["metapath"]:
        print(f"\nMetapath-derived edges ({len(results['edge_stats']['metapath'])} types):")
        rows = [(d["rel"], f"{d['count']:,}", f"{d['density']:.6f}")
                for d in sorted(results["edge_stats"]["metapath"].values(), key=lambda x: -x["count"])]
        _print_table(["Metapath", "Count", "Density"], rows)
        mp_total = sum(d["count"] for d in results["edge_stats"]["metapath"].values())
        print(f"Total metapath edges: {mp_total:,}")

    # Section 3: Degree Distributions
    _print_header("3. Degree Distributions (Overall)")
    rows = [(ntype, f"{d['mean']:.1f}", f"{d['median']:.0f}", f"{d['std']:.1f}", d["min"], d["max"], d["zero_degree"])
            for ntype, d in results["degree_distributions"]["overall"].items()]
    _print_table(["Node Type", "Mean", "Median", "Std", "Min", "Max", "Isolated"], rows)

    # Section 4: Connected Components
    _print_header("4. Connected Components")
    cc = results["connected_components"]
    print(f"  Components:          {cc['num_components']:,}")
    print(f"  Largest component:   {cc['largest_component_size']:,} nodes ({cc['largest_component_pct']:.1f}%)")
    print(f"  Startups in largest: {cc['startups_in_largest_component']:,} ({cc['startups_in_largest_pct']:.1f}%)")
    print(f"  Isolated startups:   {cc['isolated_startups']:,}")

    # Section 5: Homophily
    _print_header("5. Homophily (startup-startup edges)")
    if results["homophily"]:
        rows = [(rel, d["num_edges"], d["edge_homophily"], d["class_homophily"])
                for rel, d in sorted(results["homophily"].items(), key=lambda x: -x[1]["edge_homophily"])]
        _print_table(["Edge Type", "Edges", "Edge Homophily", "Class Homophily"], rows)
    else:
        print("  No startup-startup edges found (metapaths not materialized?)")

    # Section 6: Class Balance
    _print_header("6. Class Balance")
    for tname, tdata in results["class_balance"].items():
        print(f"\n  Target: {tname}")
        for split, sd in tdata.items():
            print(f"    {split:8s}: {sd['positive']:5d} pos / {sd['negative']:5d} neg  ({sd['positive_rate']:.1%} positive)")

    # Section 7: Label-Degree Correlation
    _print_header("7. Label-Degree Correlation")
    ld = results["label_degree_correlation"]
    print(f"  Mean degree (positive):  {ld['mean_degree_positive']:.1f}")
    print(f"  Mean degree (negative):  {ld['mean_degree_negative']:.1f}")
    print(f"  Median degree (positive): {ld['median_degree_positive']:.0f}")
    print(f"  Median degree (negative): {ld['median_degree_negative']:.0f}")
    if ld["point_biserial_r"] is not None:
        print(f"  Point-biserial r:        {ld['point_biserial_r']:.4f} (p={ld['point_biserial_p']:.2e})")

    # Section 8: Feature Coverage
    _print_header("8. Feature Coverage")
    rows = [(ntype, d["num_nodes"], d["num_features"], f"{d['pct_non_nan']:.1f}%", f"{d['pct_non_zero']:.1f}%")
            for ntype, d in results["feature_coverage"].items()]
    _print_table(["Node Type", "Nodes", "Features", "Non-NaN", "Non-Zero"], rows)

    # Section 9: Temporal Distribution
    _print_header("9. Temporal Distribution")
    td = results["temporal_distribution"]
    if "error" in td:
        print(f"  {td['error']}")
    else:
        print(f"  Startups in graph: {td['num_startups_in_graph']:,}")
        print(f"  With founding year: {td['total_with_year']:,}")
        if td["year_range"]:
            print(f"  Year range: {td['year_range'][0]} - {td['year_range'][1]}")
        print(f"  Median year: {td['median_year']}")
        if td.get("median_year_positive"):
            print(f"  Median year (positive): {td['median_year_positive']}")
            print(f"  Median year (negative): {td['median_year_negative']}")
        if td.get("bins_5yr"):
            print(f"\n  5-year bins:")
            for bin_label, count in td["bins_5yr"].items():
                bar = "#" * max(1, count // 100)
                print(f"    {bin_label:11s}: {count:5d}  {bar}")

    # Section 10: Feature Dimensionality
    _print_header("10. Feature Dimensionality")
    rows = [(ntype, d["num_nodes"], d["total_features"], d["named_features"])
            for ntype, d in results["feature_dimensionality"].items()]
    _print_table(["Node Type", "Nodes", "Total Feats", "Named Feats"], rows)


# ============================================================
# Main orchestrator
# ============================================================

def _make_json_serializable(obj):
    """Recursively convert numpy/torch types to Python natives."""
    if isinstance(obj, dict):
        return {k: _make_json_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_make_json_serializable(v) for v in obj]
    if isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    if isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (torch.Tensor,)):
        return obj.tolist()
    return obj


def compute_all(graph_path, csv_path, output_path):
    """Run all sections, print tables, save JSON."""
    print(f"Loading graph from {graph_path}...")
    data = torch.load(graph_path, weights_only=False)
    # The content-addressable cache stores (HeteroData, node_names) tuples;
    # the legacy bare-HeteroData path is also supported. Handle both.
    if isinstance(data, tuple):
        data = data[0]
    print(f"  Node types: {data.node_types}")
    print(f"  Edge types: {len(data.edge_types)} types")

    results = {}
    results["node_stats"] = compute_node_stats(data)
    results["edge_stats"] = compute_edge_stats(data)
    results["degree_distributions"] = compute_degree_distributions(data)

    print("Computing connected components (may take a moment)...")
    results["connected_components"] = compute_connected_components(data)

    results["homophily"] = compute_homophily(data)
    results["class_balance"] = compute_class_balance(data)
    results["label_degree_correlation"] = compute_label_degree_correlation(data)
    results["feature_coverage"] = compute_feature_coverage(data)
    results["temporal_distribution"] = compute_temporal_distribution(data, csv_path)
    results["feature_dimensionality"] = compute_feature_dimensionality(data)

    # Print formatted output
    print_all(results)

    # Save JSON
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    serializable = _make_json_serializable(results)
    # Remove per-feature detail from feature_coverage to keep JSON manageable
    # (it's still printed to console)
    with open(output_path, "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"\nResults saved to {output_path}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute graph statistics for thesis")
    parser.add_argument("--graph-path", default="outputs/pipeline_state/graph_data.pt",
                        help="Path to preprocessed HeteroData .pt file")
    parser.add_argument("--csv-path", default="data/graph/startup_nodes.csv",
                        help="Path to startup_nodes.csv for temporal data")
    parser.add_argument("--output", default="outputs/graph_statistics.json",
                        help="Output JSON path")
    args = parser.parse_args()

    compute_all(args.graph_path, args.csv_path, args.output)
