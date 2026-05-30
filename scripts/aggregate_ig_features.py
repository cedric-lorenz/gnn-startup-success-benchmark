"""Aggregate per-seed Integrated-Gradients attributions across the
SeHGNN g1_full and g4_heterophily 20-seed replications.

Reads the full per-feature CSVs at:
  outputs/explanations/ig_multiseed/{variant}/seed_{N}/startup_feature_importance_{mom,liq}_task_improved_data_full.csv

For each (variant, target), produces:
  - per_seed_attributions.csv  (long: seed × feature × {abs, raw})
  - aggregated_attributions.csv  (per feature: mean_abs, std_abs, mean_raw, std_raw, n_seeds)
  - top_features_summary.json (top-15 by mean_abs with rank stability)

Cross-graph comparison plot (one figure per target) with paired bars and
±std error bars. Sorted by g1 mean_abs to keep the visual comparable.

Usage:
  python scripts/aggregate_ig_features.py \\
    --multiseed-root outputs/explanations/ig_multiseed \\
    --output-dir outputs/explanations/ig_multiseed_aggregated \\
    --top-k 15
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_VARIANTS = ["g1_full", "g4_heterophily"]
TARGETS = {"mom_task": "Next Funding Round", "liq_task": "Exit"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--multiseed-root",
                   default=str(PROJECT_ROOT / "outputs/explanations/ig_multiseed"))
    p.add_argument("--output-dir",
                   default=str(PROJECT_ROOT / "outputs/explanations/ig_multiseed_aggregated"))
    p.add_argument("--top-k", type=int, default=15)
    p.add_argument("--copy-to-paper", action="store_true",
                   help="Copy comparison PDFs into graph-paper/figures/.")
    p.add_argument("--variants", nargs="+", default=None,
                   help="Subset of variant subdirs under --multiseed-root to "
                        "aggregate. Defaults to ['g1_full', 'g4_heterophily']. "
                        "When fewer than 2 variants are supplied the g1-vs-g4 "
                        "paired-bar comparison plot is skipped (used by the EG "
                        "aggregation, which only has g4_heterophily).")
    return p.parse_args()


def load_per_seed_csvs(root: Path, variant: str, target: str) -> pd.DataFrame:
    rows = []
    seed_dirs = sorted((root / variant).glob("seed_*"))
    if not seed_dirs:
        raise FileNotFoundError(f"No seed_* dirs under {root / variant}")
    for sd in seed_dirs:
        seed = int(sd.name.split("_")[-1])
        f = sd / f"startup_feature_importance_{target}_improved_data_full.csv"
        if not f.exists():
            print(f"  WARNING: missing {f}")
            continue
        df = pd.read_csv(f)
        df["seed"] = seed
        rows.append(df)
    return pd.concat(rows, ignore_index=True)


def aggregate(per_seed: pd.DataFrame) -> pd.DataFrame:
    grp = per_seed.groupby("feature").agg(
        mean_abs=("abs_importance", "mean"),
        std_abs=("abs_importance", "std"),
        mean_raw=("raw_importance", "mean"),
        std_raw=("raw_importance", "std"),
        n_seeds=("abs_importance", "count"),
    ).reset_index().sort_values("mean_abs", ascending=False)
    return grp


def rank_stability(per_seed: pd.DataFrame, top_k: int) -> Dict[str, Dict]:
    """For each feature, report fraction of seeds in which it appeared in the
    top-k by abs_importance."""
    out: Dict[str, Dict] = {}
    seeds = sorted(per_seed["seed"].unique())
    per_seed_topk: Dict[int, set] = {}
    for s in seeds:
        sub = per_seed[per_seed["seed"] == s].sort_values(
            "abs_importance", ascending=False).head(top_k)
        per_seed_topk[s] = set(sub["feature"])
    all_features = sorted(set().union(*per_seed_topk.values()))
    for feat in all_features:
        in_count = sum(1 for s in seeds if feat in per_seed_topk[s])
        out[feat] = {"in_top_k_seeds": in_count, "n_seeds": len(seeds)}
    return out


def make_paired_bar_plot(
    g1_agg: pd.DataFrame, g4_agg: pd.DataFrame,
    target: str, target_label: str,
    out_path: Path, top_k: int,
) -> None:
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches

    _RC = {
        "font.family": "serif", "font.size": 11,
        "axes.labelsize": 12, "axes.titlesize": 12,
        "xtick.labelsize": 9, "ytick.labelsize": 9,
        "legend.fontsize": 9, "figure.dpi": 300,
        "savefig.dpi": 300, "savefig.bbox": "tight", "savefig.pad_inches": 0.05,
    }

    # Use top-K of g1 as the canonical ordering; show g4 alongside.
    top_g1 = g1_agg.head(top_k).copy().reset_index(drop=True)
    g4_lookup = g4_agg.set_index("feature")
    rows = []
    for _, r in top_g1.iterrows():
        feat = r["feature"]
        g4_row = g4_lookup.loc[feat] if feat in g4_lookup.index else None
        rows.append({
            "feature": feat,
            "g1_mean": r["mean_abs"], "g1_std": r["std_abs"],
            "g1_raw":  r["mean_raw"],
            "g4_mean": g4_row["mean_abs"] if g4_row is not None else 0.0,
            "g4_std":  g4_row["std_abs"]  if g4_row is not None else 0.0,
            "g4_raw":  g4_row["mean_raw"] if g4_row is not None else 0.0,
        })
    plot_df = pd.DataFrame(rows)

    # Sign annotation: use g1 raw direction.
    sign_lbl = ["(+)" if v >= 0 else "(-)" for v in plot_df["g1_raw"]]
    feat_lbls = [f"{f} {s}" for f, s in zip(plot_df["feature"], sign_lbl)]

    # Normalize so x-axis isn't 1e-7
    max_val = max(plot_df["g1_mean"].max(), plot_df["g4_mean"].max())
    exponent = int(np.floor(np.log10(max_val))) if max_val > 0 else 0
    scale = 10 ** exponent

    with plt.rc_context(_RC):
        fig, ax = plt.subplots(figsize=(7.5, max(5, top_k * 0.32)))
        y = np.arange(len(plot_df))
        bar_h = 0.4
        ax.barh(y - bar_h/2, plot_df["g1_mean"]/scale, bar_h,
                xerr=plot_df["g1_std"]/scale, capsize=2,
                color="#4682B4", edgecolor="white", linewidth=0.3,
                label="G1 (full, 32 metapaths)")
        ax.barh(y + bar_h/2, plot_df["g4_mean"]/scale, bar_h,
                xerr=plot_df["g4_std"]/scale, capsize=2,
                color="#DC143C", edgecolor="white", linewidth=0.3,
                label="G4 (heterophily-pruned, 20 metapaths)")
        ax.set_yticks(y)
        ax.set_yticklabels(feat_lbls)
        ax.invert_yaxis()
        ax.set_xlabel(f"Mean |Attribution| (×10$^{{{exponent}}}$, ±1 std across 20 seeds)")
        ax.set_title(f"{target_label}: SeHGNN IG, G1 vs G4")
        ax.grid(axis='x', alpha=0.2, linewidth=0.5)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.legend(loc='lower right', framealpha=0.85)
        fig.tight_layout()
        fig.savefig(out_path, facecolor='white', edgecolor='none')
        plt.close(fig)
    print(f"  wrote {out_path}")


def main() -> int:
    args = parse_args()
    root = Path(args.multiseed_root)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    variants = args.variants if args.variants else DEFAULT_VARIANTS

    summaries: Dict[str, Dict] = {}

    for target, target_label in TARGETS.items():
        per_target_aggs = {}
        for variant in variants:
            print(f"\n=== {variant} / {target} ===")
            per_seed = load_per_seed_csvs(root, variant, target)
            n_seeds = per_seed["seed"].nunique()
            print(f"  loaded {n_seeds} seeds, {per_seed['feature'].nunique()} unique features")

            agg = aggregate(per_seed)
            agg.to_csv(out_dir / f"aggregated_{variant}_{target}.csv", index=False)
            per_seed.to_csv(out_dir / f"per_seed_{variant}_{target}.csv", index=False)

            stab = rank_stability(per_seed, top_k=args.top_k)
            top_lines = []
            for i, r in agg.head(args.top_k).iterrows():
                f = r["feature"]
                stab_str = f"{stab.get(f,{}).get('in_top_k_seeds','?')}/{n_seeds}"
                rel = (r["std_abs"] / r["mean_abs"]) if r["mean_abs"] > 0 else float("nan")
                sign = "+" if r["mean_raw"] >= 0 else "-"
                top_lines.append({
                    "rank": int(len(top_lines)+1),
                    "feature": f,
                    "sign": sign,
                    "mean_abs": float(r["mean_abs"]),
                    "std_abs": float(r["std_abs"]),
                    "rel_std": float(rel),
                    "in_top_k_seeds": stab_str,
                })
            summaries[f"{variant}__{target}"] = {
                "n_seeds": int(n_seeds),
                "top": top_lines,
            }
            for line in top_lines:
                print(f"  {line['rank']:>2}. {line['sign']} {line['feature']:<42} "
                      f"{line['mean_abs']:.3e} ± {line['std_abs']:.2e}  "
                      f"(stab {line['in_top_k_seeds']})")
            per_target_aggs[variant] = agg

        # Cross-graph comparison plot only when both g1 and g4 are present
        if "g1_full" in per_target_aggs and "g4_heterophily" in per_target_aggs:
            plot_path = out_dir / f"ig_compare_g1_g4_{target}.pdf"
            make_paired_bar_plot(
                per_target_aggs["g1_full"], per_target_aggs["g4_heterophily"],
                target=target, target_label=target_label,
                out_path=plot_path, top_k=args.top_k,
            )

    summary_path = out_dir / "summary.json"
    with summary_path.open("w") as f:
        json.dump(summaries, f, indent=2)
    print(f"\nWrote {summary_path}")

    if args.copy_to_paper:
        figs_dir = PROJECT_ROOT / "graph-paper" / "figures"
        figs_dir.mkdir(parents=True, exist_ok=True)
        import shutil
        for target in TARGETS:
            src = out_dir / f"ig_compare_g1_g4_{target}.pdf"
            if src.exists():
                shutil.copy2(src, figs_dir / src.name)
                print(f"Copied {src} -> {figs_dir / src.name}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
