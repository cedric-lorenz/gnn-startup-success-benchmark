"""Pairwise Wilcoxon signed-rank tests with Holm correction, per metric.

For the ICDM paper replication:
- Selection metric (val_auc_pr_joint) is optimized but NOT the reporting metric.
- Paper reports 4 metrics separately: AUPR-NFR, AUPR-EXIT, AUC-ROC-NFR, AUC-ROC-EXIT.

For each reporting metric, this script runs Wilcoxon signed-rank on paired
seed values (arch_A vs arch_B sharing the same (variant, seed)) and
applies Holm-Bonferroni correction within each "family" of comparisons.

Families:
  - For each variant:  all C(n_archs, 2) arch pairs (n_archs-dependent)
  - For each arch:     all C(n_variants, 2) variant pairs

Inputs: fetched fresh from W&B by group tag (replicate_<Arch>__<variant>).
Outputs:
  experiments/registry/significance/arch_within_<variant>_<metric>.csv
  experiments/registry/significance/variant_within_<arch>_<metric>.csv
  experiments/registry/significance/all_pairwise.csv  (long-form, for LaTeX)

Usage:
    python scripts/pairwise_significance.py
    python scripts/pairwise_significance.py --project <entity>/<project>
"""
from __future__ import annotations
import argparse
import csv
import json
import math
import os
import sys
from collections import defaultdict
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
WINNERS_PATH = REPO_ROOT / "experiments" / "registry" / "winners.json"
OUT_DIR = REPO_ROOT / "experiments" / "registry" / "significance"

# Resolves from $WANDB_PROJECT, else an obvious placeholder a public-repo user
# overrides with their own entity/project.
DEFAULT_PROJECT = os.environ.get("WANDB_PROJECT", "<entity>/<project>")
DEFAULT_METRICS = [
    "val_auc_pr_mom",   # AUPR-NFR
    "val_auc_pr_liq",   # AUPR-EXIT
    "val_auc_roc_mom",  # AUC-ROC-NFR
    "val_auc_roc_liq",  # AUC-ROC-EXIT
    "test_auc_pr_mom",
    "test_auc_pr_liq",
    "test_auc_roc_mom",
    "test_auc_roc_liq",
]

METRIC_DISPLAY = {
    "val_auc_pr_mom": "AUPR-NFR (val)",
    "val_auc_pr_liq": "AUPR-EXIT (val)",
    "val_auc_roc_mom": "AUC-ROC-NFR (val)",
    "val_auc_roc_liq": "AUC-ROC-EXIT (val)",
    "test_auc_pr_mom": "AUPR-NFR (test)",
    "test_auc_pr_liq": "AUPR-EXIT (test)",
    "test_auc_roc_mom": "AUC-ROC-NFR (test)",
    "test_auc_roc_liq": "AUC-ROC-EXIT (test)",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--project", default=DEFAULT_PROJECT)
    p.add_argument("--metrics", nargs="+", default=DEFAULT_METRICS)
    p.add_argument("--min-seeds", type=int, default=5,
                   help="Require at least this many paired seeds (default 5)")
    p.add_argument("--alpha", type=float, default=0.05)
    return p.parse_args()


def fetch_group_per_seed(project: str, group: str) -> Dict[int, Dict[str, float]]:
    """Return {seed: {metric: value}} for finished+ok runs in a W&B group."""
    import wandb
    api = wandb.Api(timeout=30)
    try:
        runs = list(api.runs(project, filters={"group": group}))
    except Exception as e:
        print(f"  [WARN] {group}: {type(e).__name__} {e}")
        return {}
    out: Dict[int, Dict[str, float]] = {}
    for r in runs:
        if r.state != "finished":
            continue
        if r.summary.get("repro/status") not in (None, "ok"):
            continue
        seed = r.config.get("seed")
        if not isinstance(seed, (int, float)):
            continue
        seed = int(seed)
        # If multiple runs share the same seed (re-submit), keep the latest
        out[seed] = {k: float(v) for k, v in r.summary.items()
                     if isinstance(v, (int, float)) and not (isinstance(v, float) and math.isnan(v))}
    return out


def load_winners() -> Dict[str, Any]:
    with open(WINNERS_PATH) as f:
        return json.load(f)["cells"]


def group_name(arch: str, variant: Optional[str]) -> str:
    if variant:
        return f"replicate_{arch}__{variant}"
    return f"replicate_{arch}"


def holm_correction(pvals: List[float]) -> List[float]:
    """Holm-Bonferroni step-down correction. Preserves input order."""
    n = len(pvals)
    if n == 0:
        return []
    ranked = sorted(range(n), key=lambda i: pvals[i])
    corrected = [0.0] * n
    running_max = 0.0
    for rank, idx in enumerate(ranked):
        adj = pvals[idx] * (n - rank)
        running_max = max(running_max, min(adj, 1.0))
        corrected[idx] = running_max
    return corrected


def paired_wilcoxon(a_seeds: Dict[int, float], b_seeds: Dict[int, float]
                    ) -> Tuple[Optional[float], Optional[float], int, float]:
    """Return (W statistic, p-value, n_pairs, mean_diff) or (None, None, 0, 0.0)."""
    from scipy import stats
    common = sorted(set(a_seeds) & set(b_seeds))
    if len(common) < 5:
        return None, None, len(common), 0.0
    a = [a_seeds[s] for s in common]
    b = [b_seeds[s] for s in common]
    diffs = [ai - bi for ai, bi in zip(a, b)]
    try:
        result = stats.wilcoxon(a, b, zero_method="zsplit", alternative="two-sided")
        return float(result.statistic), float(result.pvalue), len(common), sum(diffs) / len(diffs)
    except ValueError:
        # All-zero differences → stats.wilcoxon raises
        return None, None, len(common), 0.0


def run_family(
    families: Dict[Tuple[str, ...], List[Tuple[str, str, Dict[int, float], Dict[int, float]]]],
    metric: str, alpha: float, min_seeds: int,
) -> List[Dict[str, Any]]:
    """For each family of comparisons, run Wilcoxon + Holm. Returns long-form rows."""
    rows = []
    for family_key, comparisons in sorted(families.items()):
        pvals: List[float] = []
        entries: List[Dict[str, Any]] = []
        for a_label, b_label, a_seeds, b_seeds in comparisons:
            stat, p, n, mean_diff = paired_wilcoxon(a_seeds, b_seeds)
            if p is None or n < min_seeds:
                entry = {
                    "family": "/".join(family_key), "metric": metric,
                    "a": a_label, "b": b_label,
                    "n_pairs": n, "mean_diff": mean_diff,
                    "wilcoxon_W": stat, "p_raw": p, "p_holm": None,
                    "significant_0.05": False,
                }
                entries.append(entry)
                continue
            pvals.append(p)
            entries.append({
                "family": "/".join(family_key), "metric": metric,
                "a": a_label, "b": b_label,
                "n_pairs": n, "mean_diff": mean_diff,
                "wilcoxon_W": stat, "p_raw": p, "p_holm": None,
                "significant_0.05": None,  # filled below
            })
        # Apply Holm only over the non-None subset
        non_none_indices = [i for i, e in enumerate(entries) if e["p_raw"] is not None]
        if non_none_indices:
            raw = [entries[i]["p_raw"] for i in non_none_indices]
            adj = holm_correction(raw)
            for i, a in zip(non_none_indices, adj):
                entries[i]["p_holm"] = a
                entries[i]["significant_0.05"] = a < alpha
        rows.extend(entries)
    return rows


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["family", "metric", "a", "b", "n_pairs", "mean_diff",
                  "wilcoxon_W", "p_raw", "p_holm", "significant_0.05"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in fieldnames})


def main() -> int:
    args = parse_args()
    winners = load_winners()

    # Fetch per-seed data once per group (reused across all metrics)
    print("Fetching per-seed metrics from W&B...")
    group_data: Dict[str, Dict[int, Dict[str, float]]] = {}
    for cell_key, w in sorted(winners.items()):
        group = group_name(w["arch"], w.get("variant"))
        if group in group_data:
            continue
        print(f"  {group}")
        group_data[group] = fetch_group_per_seed(args.project, group)

    all_rows: List[Dict[str, Any]] = []

    for metric in args.metrics:
        print(f"\n== metric: {metric} ({METRIC_DISPLAY.get(metric, metric)}) ==")

        # Family 1: archs within a variant
        arch_within_variant: Dict[Tuple[str, ...], List[Tuple[str, str, Dict, Dict]]] = defaultdict(list)
        variants_seen = sorted({w.get("variant") for w in winners.values() if w.get("variant")})
        for variant in variants_seen:
            archs_here = sorted([w["arch"] for w in winners.values() if w.get("variant") == variant])
            for a, b in combinations(archs_here, 2):
                a_data = {s: v.get(metric) for s, v in group_data.get(group_name(a, variant), {}).items() if metric in v}
                b_data = {s: v.get(metric) for s, v in group_data.get(group_name(b, variant), {}).items() if metric in v}
                arch_within_variant[(variant,)].append((a, b, a_data, b_data))
        rows_a = run_family(arch_within_variant, metric, args.alpha, args.min_seeds)
        for (variant,), group_rows in _group_by_family(rows_a):
            out = OUT_DIR / f"arch_within_{variant}_{metric}.csv"
            write_csv(out, group_rows)
            print(f"  wrote {out} ({len(group_rows)} pairs)")
        all_rows.extend(rows_a)

        # Family 2: variants within an arch
        variant_within_arch: Dict[Tuple[str, ...], List[Tuple[str, str, Dict, Dict]]] = defaultdict(list)
        archs_seen = sorted({w["arch"] for w in winners.values()})
        for arch in archs_seen:
            variants_here = sorted([w.get("variant") for w in winners.values()
                                     if w["arch"] == arch and w.get("variant")])
            for a, b in combinations(variants_here, 2):
                a_data = {s: v.get(metric) for s, v in group_data.get(group_name(arch, a), {}).items() if metric in v}
                b_data = {s: v.get(metric) for s, v in group_data.get(group_name(arch, b), {}).items() if metric in v}
                variant_within_arch[(arch,)].append((a, b, a_data, b_data))
        rows_v = run_family(variant_within_arch, metric, args.alpha, args.min_seeds)
        for (arch,), group_rows in _group_by_family(rows_v):
            out = OUT_DIR / f"variant_within_{arch}_{metric}.csv"
            write_csv(out, group_rows)
            print(f"  wrote {out} ({len(group_rows)} pairs)")
        all_rows.extend(rows_v)

    combined = OUT_DIR / "all_pairwise.csv"
    write_csv(combined, all_rows)
    print(f"\nWrote combined {combined} ({len(all_rows)} rows)")
    return 0


def _group_by_family(rows: List[Dict[str, Any]]):
    """Yield (family_key_tuple, rows_in_family)."""
    buckets: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in rows:
        buckets[r["family"]].append(r)
    for k, v in buckets.items():
        yield (tuple(k.split("/")), v)


if __name__ == "__main__":
    sys.exit(main())
