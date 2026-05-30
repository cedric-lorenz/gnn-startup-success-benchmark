#!/usr/bin/env python
"""Aggregate per-seed heterophily spectrum CSVs into mean/std multiseed CSV.

Reads every outputs/graph_statistics/heterophily_spectrum_seed{K}.csv produced
by compute_heterophily_spectrum.py --rw-seed K, groups by meta-path, and
reports mean and std of the label-homophily, Dirichlet-energy, and edge-count
columns across random-walk realizations. The resulting
heterophily_spectrum_multiseed.csv is the canonical source for the paper's
appendix inventory table, error bars in the lede figure, and the correlation
claims in section 6.1.

Usage:
    python scripts/aggregate_heterophily_seeds.py
    python scripts/aggregate_heterophily_seeds.py --seeds 42 0 1 2 3
"""
import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "graph_statistics"

# Numeric columns we aggregate across seeds.
AGG_COLS = [
    "n_edges",
    "h_nfr", "delta_nfr",
    "h_exit_full", "delta_exit_full",
    "h_exit_mature", "delta_exit_mature",
    "mde", "mde_per_edge", "mde_rand_per_edge", "delta_mde_per_edge",
]


def load_seed_csv(seed: int) -> pd.DataFrame:
    path = OUTPUT_DIR / f"heterophily_spectrum_seed{seed}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing seed CSV: {path}")
    df = pd.read_csv(path)
    df["seed"] = seed
    return df


def load_seed_meta(seed: int) -> dict:
    path = OUTPUT_DIR / f"heterophily_spectrum_meta_seed{seed}.json"
    if not path.exists():
        return {}
    with open(path) as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--seeds", type=int, nargs="+", default=[42, 0, 1, 2, 3],
        help="Seeds to aggregate (default: 42 0 1 2 3).",
    )
    parser.add_argument(
        "--out", type=Path,
        default=OUTPUT_DIR / "heterophily_spectrum_multiseed.csv",
        help="Output CSV path.",
    )
    args = parser.parse_args()

    frames = []
    for seed in args.seeds:
        df = load_seed_csv(seed)
        print(f"  seed={seed}: {len(df)} rows from heterophily_spectrum_seed{seed}.csv")
        frames.append(df)

    if not frames:
        print("No seed CSVs found; nothing to aggregate.", file=sys.stderr)
        sys.exit(1)

    long_df = pd.concat(frames, ignore_index=True)
    n_seeds = long_df["seed"].nunique()
    print(f"\nLoaded {len(frames)} seed CSVs; {long_df['metapath'].nunique()} unique meta-paths.")

    # Sanity check: every meta-path should appear in every seed.
    counts = long_df.groupby("metapath").size()
    missing = counts[counts < n_seeds]
    if not missing.empty:
        print("WARNING: these meta-paths are missing from some seeds:")
        for name, c in missing.items():
            print(f"  {name}: {c}/{n_seeds} seeds")

    # Aggregate mean + std per meta-path per column.
    agg = {}
    for col in AGG_COLS:
        agg[f"{col}_mean"] = pd.NamedAgg(column=col, aggfunc="mean")
        agg[f"{col}_std"] = pd.NamedAgg(column=col, aggfunc="std")

    # Keep the category from the first-seen row (should be identical across seeds).
    grouped = (
        long_df.groupby("metapath", sort=False)
        .agg(category=pd.NamedAgg(column="category", aggfunc="first"), **agg)
        .reset_index()
    )

    # Sort ascending by mean NFR homophily (most heterophilic first) to match
    # the single-seed CSV convention.
    grouped = grouped.sort_values("h_nfr_mean", kind="stable").reset_index(drop=True)

    # Add the n_seeds column so downstream consumers know the aggregation basis.
    grouped.insert(2, "n_seeds", n_seeds)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    grouped.to_csv(args.out, index=False)
    print(f"\nWrote {len(grouped)} rows to {args.out}")

    # Console summary: most-heterophilic 10 + baseline comparison.
    metas = [load_seed_meta(s) for s in args.seeds]
    baselines = {k: np.mean([m[k] for m in metas if k in m]) for k in
                 ("baseline_nfr", "baseline_exit_full", "baseline_exit_mature")}
    print("\n" + "=" * 120)
    print(f"  HETEROPHILY SPECTRUM — multiseed aggregate ({n_seeds} seeds)")
    print(f"  Baselines (seed-averaged): NFR={baselines['baseline_nfr']:.4f}  "
          f"Exit(full)={baselines['baseline_exit_full']:.4f}  "
          f"Exit(mature)={baselines['baseline_exit_mature']:.4f}")
    print("=" * 120)
    hdr = (f"  {'metapath':<40} {'#edges µ':>10} {'MLH-NFR µ±σ':>18} "
           f"{'MLH-ExM µ±σ':>18} {'MDE/edge µ':>11}")
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for _, r in grouped.iterrows():
        print(f"  {r['metapath']:<40} "
              f"{int(round(r['n_edges_mean'])):>10d} "
              f"{r['h_nfr_mean']:>7.4f} ± {r['h_nfr_std']:>6.4f}    "
              f"{r['h_exit_mature_mean']:>7.4f} ± {r['h_exit_mature_std']:>6.4f}    "
              f"{r['mde_per_edge_mean']:>10.4f}")

    # Headline: how many meta-paths are heterophilic on average?
    n_below_nfr = int((grouped["h_nfr_mean"] < baselines["baseline_nfr"]).sum())
    n_below_exit_full = int((grouped["h_exit_full_mean"] < baselines["baseline_exit_full"]).sum())
    n_below_exit_mat = int((grouped["h_exit_mature_mean"] < baselines["baseline_exit_mature"]).sum())
    total = len(grouped)
    print(f"\n  HEADLINE (seed-averaged)")
    print(f"    NFR:           {n_below_nfr}/{total} meta-paths below baseline "
          f"({baselines['baseline_nfr']:.3f})")
    print(f"    Exit (full):   {n_below_exit_full}/{total} meta-paths below baseline "
          f"({baselines['baseline_exit_full']:.3f})")
    print(f"    Exit (mature): {n_below_exit_mat}/{total} meta-paths below baseline "
          f"({baselines['baseline_exit_mature']:.3f})")

    # Variance diagnostic: which meta-paths are noisy across seeds?
    print(f"\n  TOP 5 HIGHEST MLH-NFR VARIANCE (noisy paths)")
    noisy = grouped.nlargest(5, "h_nfr_std")[
        ["metapath", "n_edges_mean", "h_nfr_mean", "h_nfr_std"]
    ]
    for _, r in noisy.iterrows():
        print(f"    {r['metapath']:<40} n_edges={int(round(r['n_edges_mean'])):>7d}  "
              f"MLH-NFR = {r['h_nfr_mean']:.4f} ± {r['h_nfr_std']:.4f}")


if __name__ == "__main__":
    main()
