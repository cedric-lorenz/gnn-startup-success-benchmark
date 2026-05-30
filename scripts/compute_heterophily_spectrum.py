#!/usr/bin/env python
"""Compute the per-meta-path heterophily spectrum on the intrinsic-tier graph.

For each of the 32 startup-to-startup meta-paths, computes:
  - MLH on the Next Funding Round target (full population)
  - MLH on the Exit target (full population)
  - MLH on the Exit target (mature subset)
  - MDE (meta-path Dirichlet energy) on the intrinsic feature matrix

Runs preprocessing once with the intrinsic feature tier and all 32 meta-paths
enabled, then computes metrics on the resulting graph. Outputs a seed-suffixed
CSV plus a companion JSON with the per-target baselines and caches the graph
for reuse. Passing --rw-seed K reseeds PyTorch/numpy/random before meta-path
materialization so each seed produces an independent random-walk realization
for robustness analysis.

Usage:
    python scripts/compute_heterophily_spectrum.py              # seed 42
    python scripts/compute_heterophily_spectrum.py --rw-seed 0  # seed 0
"""
import argparse
import json
import random
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import torch

from src.ml.heterophily_metrics import (
    calculate_edge_homophily,
    calculate_meta_path_dirichlet_energy,
    random_pair_baseline,
)
from src.ml.preprocessing import perform_preprocessing
from src.ml.utils import load_config

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "graph_statistics"

# 32 startup-to-startup meta-paths: 9 investor-mediated families x 3 stages
# (= 27) + 4 stage-independent peer paths + 1 serial-founder path
# (Zhang et al., HICSS 2024, M1).
ALL_32_METAPATHS = [
    # Serial founder (1) - Zhang M1: SPS via shared founder
    "serial_founder",
    # Peer similarity (2)
    "sector_peers", "city_peers",
    # Academic (1)
    "alumni_network",
    # Investment co-participation (3)
    "early_portfolio_siblings", "mid_portfolio_siblings", "late_portfolio_siblings",
    # Founder-VC employment (3)
    "early_founder_vc_employment", "mid_founder_vc_employment", "late_founder_vc_employment",
    # Board director (3)
    "early_board_director_network", "mid_board_director_network", "late_board_director_network",
    # Board employment (3) - founder 'worked_at' investor (distinct from 'director_at')
    "early_board_employment_network", "mid_board_employment_network", "late_board_employment_network",
    # Investor alumni (3)
    "early_investor_alumni", "mid_investor_alumni", "late_investor_alumni",
    # Alumni-investor network (3)
    "early_alumni_investor_network", "mid_alumni_investor_network", "late_alumni_investor_network",
    # Co-working network (1)
    "co_working_network",
    # Investor-founder coworking (3)
    "early_investor_founder_coworking", "mid_investor_founder_coworking", "late_investor_founder_coworking",
    # Founder coworking -> investor (3)
    "early_founder_coworking_investor", "mid_founder_coworking_investor", "late_founder_coworking_investor",
    # Founder coworking syndicate (3)
    "early_founder_coworking_syndicate", "mid_founder_coworking_syndicate", "late_founder_coworking_syndicate",
]

# Semantic category per meta-path (used by the figure scripts for coloring).
CATEGORY_OVERRIDES = {
    "serial_founder": "serial_founder",
    "sector_peers": "sector",
    "city_peers": "city",
    "alumni_network": "alumni",
    "co_working_network": "founder_cowork",
}


def categorize(name: str) -> str:
    if name in CATEGORY_OVERRIDES:
        return CATEGORY_OVERRIDES[name]
    if "board_employment" in name:
        return "board_employment"
    if "board" in name:
        return "board_director"
    if "founder_vc" in name or "founder_coworking" in name or "investor_founder" in name:
        return "founder_investor"
    if "alumni" in name:
        return "investor_alumni"
    if "portfolio_siblings" in name or "syndicate" in name:
        return "portfolio_siblings"
    return "other"


def paths_for_seed(rw_seed: int):
    """Return (graph_cache, csv_out, json_out) for a given RW seed."""
    return (
        OUTPUT_DIR / f"graph_data_intrinsic_seed{rw_seed}.pt",
        OUTPUT_DIR / f"heterophily_spectrum_seed{rw_seed}.csv",
        OUTPUT_DIR / f"heterophily_spectrum_meta_seed{rw_seed}.json",
    )


def build_intrinsic_graph(rw_seed: int = 42):
    """Run preprocessing with the intrinsic feature tier and all 32 meta-paths.

    The rw_seed argument reseeds PyTorch/numpy/random before meta-path
    materialization so each call produces an independent random-walk
    realization. The graph is cached at graph_data_intrinsic_seed{K}.pt so
    repeated calls reuse the materialization deterministically.
    """
    graph_cache, _, _ = paths_for_seed(rw_seed)
    if graph_cache.exists():
        print(f"Loading cached intrinsic graph: {graph_cache}")
        return torch.load(graph_cache, weights_only=False)

    print(f"No cached intrinsic graph at {graph_cache}; running preprocessing from scratch (rw_seed={rw_seed}).")
    config = load_config(str(PROJECT_ROOT / "config.yaml"))

    # Seed everything touching randomness before meta-path RW materialization.
    torch.manual_seed(rw_seed)
    np.random.seed(rw_seed)
    random.seed(rw_seed)

    # Feature tier: intrinsic (~29 scalars, no hand-engineered aggregates).
    config["data_processing"]["ablation"]["feature_information_level"] = "intrinsic"

    # All edge types needed for the meta-paths.
    el = config["data_processing"]["edge_loading"]
    el["founder_coworking"] = True
    el["founder_investor_employment"] = True
    el["founder_investor_identity"] = True
    el["founder_role_edges"] = True
    el["founder_co_study"] = False

    # Manual meta-path mode with the full 32.
    config["metapath_discovery"]["mode"] = "manual"
    config["metapath_discovery"]["manual"]["whitelist"] = ALL_32_METAPATHS
    config["data_processing"]["add_metapaths"] = True
    config["features"]["use_non_target_metapaths"] = False

    # Quiet the run.
    config["visualize"]["enabled"] = False
    config["wandb"]["enabled"] = False

    graph, _ = perform_preprocessing(
        startups_filename="startup_nodes.csv",
        investors_filename="investor_nodes.csv",
        founders_filename="founder_nodes.csv",
        cities_filename="city_nodes.csv",
        university_filename="university_nodes.csv",
        sectors_filename="sector_nodes.csv",
        startup_investor_filename="startup_investor_edges.csv",
        startup_city_filename="startup_city_edges.csv",
        startup_founder_filename="startup_founder_edges.csv",
        startup_sector_filename="startup_sector_edges.csv",
        founder_university_filename="founder_university_edges.csv",
        investor_city_filename="investor_city_edges.csv",
        investor_sector_filename="investor_sector_edges.csv",
        university_city_filename="university_city_edges.csv",
        founder_investor_employment_filename="founder_investor_employment_edges.csv",
        founder_coworking_filename="founder_coworking_edges.csv",
        founder_investor_identity_filename="founder_investor_identity_edges.csv",
        founder_co_study_filename="founder_co_study_edges.csv",
        founder_board_filename="founder_board_edges.csv",
        founder_startup_director_filename="founder_startup_director_edges.csv",
        founder_investor_director_filename="founder_investor_director_edges.csv",
        config=config,
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(graph, graph_cache)
    print(f"Saved intrinsic graph cache: {graph_cache}")
    return graph


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--rw-seed", type=int, default=42,
        help="Random-walk materialization seed (default: 42). "
             "Each seed produces an independent RW realization of the 32 meta-paths.",
    )
    args = parser.parse_args()
    rw_seed = args.rw_seed

    _, csv_out, json_out = paths_for_seed(rw_seed)
    graph = build_intrinsic_graph(rw_seed=rw_seed)

    x_raw = graph["startup"].x.float()
    # Standardize per-dim so MDE is not dominated by large-scale features.
    # Heterogeneous intrinsic features (e.g. total_funding_usd) vary across
    # orders of magnitude; standardization is standard before Dirichlet-energy
    # comparison and matches the usual Hetero^2Net convention.
    x = (x_raw - x_raw.mean(dim=0, keepdim=True)) / \
        x_raw.std(dim=0, keepdim=True).clamp(min=1e-6)
    # NaN guard (constant columns → std = 0 → 1e-6 ; still numerically safe).
    x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

    # Per-meta-path MDE baseline: shuffle features and recompute. Averaged
    # over N_shuffles permutations. This is the honest analogue of the MLH
    # random-pair baseline, capturing the expected MDE under the null
    # hypothesis "same topology, independent features."
    n_shuffles = 10
    generator = torch.Generator(device=x.device).manual_seed(42)

    y_all = graph["startup"].y
    y_nfr = y_all[:, 0].float()       # Next Funding Round label
    y_exit = y_all[:, 1].float()      # Exit label
    mature_mask = y_all[:, 3].bool()  # Maturity mask (col 3 per train.py:786)

    baseline_nfr = random_pair_baseline(y_nfr)
    baseline_exit_full = random_pair_baseline(y_exit)
    baseline_exit_mature = random_pair_baseline(y_exit, node_mask=mature_mask)

    print(f"\nIntrinsic feature matrix: shape={tuple(x.shape)} (standardized)")
    print(f"Labels: NFR positive rate = {y_nfr.mean().item():.4f}, "
          f"Exit positive rate = {y_exit.mean().item():.4f}, "
          f"mature fraction = {mature_mask.float().mean().item():.4f}")
    print(f"Random-pair baselines: "
          f"NFR={baseline_nfr:.4f}, Exit-full={baseline_exit_full:.4f}, "
          f"Exit-mature={baseline_exit_mature:.4f}")

    rows = []
    for edge_type in graph.edge_types:
        src, rel, dst = edge_type
        if src != "startup" or dst != "startup":
            continue
        edge_index = graph[edge_type].edge_index
        n_edges = int(edge_index.size(1))
        if n_edges == 0:
            continue

        h_nfr = calculate_edge_homophily(edge_index, y_nfr)
        h_exit_full = calculate_edge_homophily(edge_index, y_exit)
        h_exit_mat = calculate_edge_homophily(edge_index, y_exit, node_mask=mature_mask)
        mde = calculate_meta_path_dirichlet_energy(edge_index, x)
        mde_per_edge = (mde / n_edges) if (mde is not None and n_edges > 0) else None

        # Shuffled-feature baseline: average MDE under random node assignments,
        # same topology. Captures what MDE would be if features were
        # uninformative for this meta-path's edge structure.
        n = x.size(0)
        shuffled = []
        for _ in range(n_shuffles):
            perm = torch.randperm(n, generator=generator, device=x.device)
            shuffled.append(calculate_meta_path_dirichlet_energy(edge_index, x[perm]))
        mde_rand = float(sum(shuffled) / len(shuffled)) if shuffled else None
        mde_rand_per_edge = (mde_rand / n_edges) if (mde_rand is not None and n_edges > 0) else None

        rows.append(dict(
            metapath=rel,
            category=categorize(rel),
            n_edges=n_edges,
            h_nfr=h_nfr,
            delta_nfr=(h_nfr - baseline_nfr) if h_nfr is not None else None,
            h_exit_full=h_exit_full,
            delta_exit_full=(h_exit_full - baseline_exit_full) if h_exit_full is not None else None,
            h_exit_mature=h_exit_mat,
            delta_exit_mature=(h_exit_mat - baseline_exit_mature) if h_exit_mat is not None else None,
            mde=mde,
            mde_per_edge=mde_per_edge,
            mde_rand_per_edge=mde_rand_per_edge,
            delta_mde_per_edge=(mde_per_edge - mde_rand_per_edge)
                if (mde_per_edge is not None and mde_rand_per_edge is not None) else None,
        ))

    df = pd.DataFrame(rows).sort_values("h_nfr", kind="stable").reset_index(drop=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_out, index=False)
    with open(json_out, "w") as f:
        json.dump(dict(
            rw_seed=rw_seed,
            feature_dim=int(x.size(1)),
            n_startups=int(x.size(0)),
            n_mature=int(mature_mask.sum().item()),
            baseline_nfr=baseline_nfr,
            baseline_exit_full=baseline_exit_full,
            baseline_exit_mature=baseline_exit_mature,
            positive_rate_nfr=float(y_nfr.mean().item()),
            positive_rate_exit_full=float(y_exit.mean().item()),
            positive_rate_exit_mature=float(y_exit[mature_mask].mean().item()),
        ), f, indent=2)
    print(f"\nWrote {len(df)} rows to {csv_out}")
    print(f"Wrote baselines to {json_out}")

    # Summary table
    print(f"\n{'='*104}")
    print("  HETEROPHILY SPECTRUM (sorted by MLH-NFR ascending)")
    print(f"{'='*104}")
    hdr = f"  {'metapath':<40} {'#edges':>9} {'MLH-NFR':>9} {'MLH-ExF':>9} {'MLH-ExM':>9} {'MDE/edge':>10}"
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))

    def fmt(v, w, decimals=4):
        return f"{v:>{w}.{decimals}f}" if v is not None and not pd.isna(v) else f"{'—':>{w}}"

    for _, r in df.iterrows():
        print(f"  {r.metapath:<40} {r.n_edges:>9d} "
              f"{fmt(r.h_nfr, 9)} {fmt(r.h_exit_full, 9)} {fmt(r.h_exit_mature, 9)} "
              f"{fmt(r.mde_per_edge, 10)}")

    # Headline
    n_below_nfr = int((df["h_nfr"] < baseline_nfr).sum())
    n_below_exit_mat = int((df["h_exit_mature"] < baseline_exit_mature).sum())
    n_below_exit_full = int((df["h_exit_full"] < baseline_exit_full).sum())
    print("\n  HEADLINE")
    print(f"    NFR:           {n_below_nfr}/{len(df)} meta-paths below baseline ({baseline_nfr:.3f})")
    print(f"    Exit (full):   {n_below_exit_full}/{len(df)} meta-paths below baseline ({baseline_exit_full:.3f})")
    print(f"    Exit (mature): {n_below_exit_mat}/{len(df)} meta-paths below baseline ({baseline_exit_mature:.3f})")


if __name__ == "__main__":
    main()
