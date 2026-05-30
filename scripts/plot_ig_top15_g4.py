#!/usr/bin/env python
"""Render top-15 multi-seed IG feature-importance figures for the paper.

Reuses:
  - scripts/aggregate_ig_features.py::{load_per_seed_csvs, aggregate}
  - src/ml/explain.py::plot_feature_importance_from_df

Inputs (default):
  outputs/explanations/ig_multiseed/g4_heterophily/seed_<0..19>/
    startup_feature_importance_{mom_task,liq_task}_improved_data_full.csv

Outputs (default):
  graph-paper/figures/feature_importance_momentum.pdf   (NFR top-15)
  graph-paper/figures/feature_importance_liquidity.pdf  (Exit top-15)

Excludes `primary_role` per design discussion. Error bars are plus or minus
one std across the 20-seed replication (matches the aggregator convention).

For the Expected-Gradients (EG) figure, point --multiseed-root at the EG
output tree and add --out-suffix _eg so the EG figures land alongside the
zero-baseline ones.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.aggregate_ig_features import load_per_seed_csvs, aggregate
from src.ml.explain import plot_feature_importance_from_df

DEFAULT_ROOT = PROJECT_ROOT / "outputs" / "explanations" / "ig_multiseed"
PAPER_FIGS = PROJECT_ROOT / "graph-paper" / "figures"
VARIANT = "g4_heterophily"
EXCLUDE = ["primary_role"]
TOP_K = 15

TASKS = [
    ("mom_task", "feature_importance_momentum",
     "Feature Attribution for Next Funding Round"),
    ("liq_task", "feature_importance_liquidity",
     "Feature Attribution for Exit"),
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--multiseed-root", default=str(DEFAULT_ROOT),
                   help="Root directory holding <variant>/seed_<N>/ subdirs "
                        "with per-seed feature-importance CSVs.")
    p.add_argument("--out-dir", default=str(PAPER_FIGS),
                   help="Directory to write PDFs into.")
    p.add_argument("--out-suffix", default="",
                   help="Suffix appended before .pdf (for example '_eg' for "
                        "the Expected-Gradients figure).")
    p.add_argument("--title-prefix", default="",
                   help="Optional prefix prepended to the in-figure title "
                        "(for example 'Expected Gradients ').")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    root = Path(args.multiseed_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for task, base_name, base_title in TASKS:
        print(f"\n=== {VARIANT} / {task} ===")
        per_seed = load_per_seed_csvs(root, VARIANT, task)
        n_seeds = per_seed["seed"].nunique()
        agg = aggregate(per_seed)
        print(f"  loaded {n_seeds} seeds, {agg.shape[0]} features")
        print(f"  excluding {EXCLUDE}")

        out_path = out_dir / f"{base_name}{args.out_suffix}.pdf"
        title = f"{args.title_prefix}{base_title}" if args.title_prefix else base_title
        plot_feature_importance_from_df(
            agg, path=str(out_path), top_k=TOP_K, title=title, exclude=EXCLUDE,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
