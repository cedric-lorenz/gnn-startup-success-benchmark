"""Statistical significance tests for the paper main table.

For each (target, metric) cell, identifies the leader (highest mean across
20 paired seeds) and tests every other architecture against the leader
using a paired Wilcoxon signed-rank test (two-sided), with Holm-Bonferroni
correction across the k-1 = 9 comparisons within that cell.

Why this method is statistically valid for our setup:
  - The 20 seeds are PAIRED across architectures: every arch was trained on
    the same canonical train/val/test split (seed=0 forks the split RNG; only
    the training-init RNG varies per seed). So for seed s, arch_X and arch_Y
    saw identical data.
  - Wilcoxon signed-rank is the non-parametric equivalent of the paired t-test
    and does not assume normality — appropriate for AUC-PR / F1 distributions
    which can be skewed.
  - Two-sided alternative is the conservative default (we don't pre-specify
    direction). One-sided would double power but invites cherry-picking.
  - Holm-Bonferroni controls family-wise error rate (FWER) within each
    (target, metric) cell. We test 9 vs-leader comparisons per cell; we do
    NOT correct across metrics/targets because each cell is its own claim
    family in the paper (standard ICDM practice).
  - Effect sizes (median pairwise difference, mean pairwise difference) are
    reported alongside p-values, since p-values alone don't quantify
    practical significance.
  - N=20 paired observations gives Wilcoxon enough power to detect
    differences ~0.005 in AUC-PR given typical seed std ~0.003.

Outputs:
  outputs/paper_results/significance_main_table.json
    For each (target, metric):
      - leader: arch with highest mean
      - leader_value: mean ± std
      - per-arch: {p_raw, p_holm, mean_diff, median_diff, sig_at_0.05, n_seeds}

Reproduce:
    python scripts/fetch_paper_metrics.py
    python scripts/significance_paper.py
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = REPO_ROOT / "outputs" / "paper_results" / "main_table_metrics.csv"
DEFAULT_OUTPUT = REPO_ROOT / "outputs" / "paper_results" / "significance_main_table.json"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument("--alpha", type=float, default=0.05)
    return p.parse_args()


def load_long_csv(path: Path) -> Dict[Tuple[str, str, str], Dict[int, float]]:
    """Return cell[(arch, target, metric)] = {seed: value}."""
    cell = defaultdict(dict)
    with open(path) as f:
        for r in csv.DictReader(f):
            key = (r["arch"], r["target"], r["metric"])
            cell[key][int(r["seed"])] = float(r["value"])
    return dict(cell)


def holm_bonferroni(pvals: List[float]) -> List[float]:
    """Apply Holm-Bonferroni step-down adjustment.

    Given m raw p-values, returns m adjusted values such that rejecting at
    alpha controls FWER. Order-preserving in the sense that the smallest raw
    p produces the smallest adjusted p, but adjusted = max(rank * raw, prev).
    """
    m = len(pvals)
    if m == 0:
        return []
    indexed = sorted(enumerate(pvals), key=lambda x: x[1])
    adjusted = [0.0] * m
    running_max = 0.0
    for rank, (orig_idx, p) in enumerate(indexed):
        candidate = (m - rank) * p
        running_max = max(running_max, min(candidate, 1.0))
        adjusted[orig_idx] = running_max
    return adjusted


def run_significance(cell, alpha: float):
    """For each (target, metric), test every non-leader against the leader."""
    from scipy.stats import wilcoxon
    import math

    targets = sorted({k[1] for k in cell.keys()})
    metrics = sorted({k[2] for k in cell.keys()})
    archs   = sorted({k[0] for k in cell.keys()})

    out = {}
    for target in targets:
        for metric in metrics:
            arches_with_data = [a for a in archs
                                if (a, target, metric) in cell
                                and len(cell[(a, target, metric)]) > 0]
            means = {a: statistics.mean(cell[(a, target, metric)].values())
                     for a in arches_with_data}
            leader = max(means, key=means.get)
            leader_seeds = cell[(leader, target, metric)]
            leader_vals = [leader_seeds[s] for s in sorted(leader_seeds.keys())]
            leader_mean = statistics.mean(leader_vals)
            leader_std  = statistics.stdev(leader_vals) if len(leader_vals) > 1 else 0.0

            others = [a for a in arches_with_data if a != leader]
            raw_p = []
            details = []
            for a in others:
                a_vals_dict = cell[(a, target, metric)]
                # Pair on seeds present in BOTH leader and other
                shared_seeds = sorted(set(leader_seeds.keys()) & set(a_vals_dict.keys()))
                if len(shared_seeds) < 6:
                    raw_p.append(1.0)
                    details.append({
                        "arch": a, "n_paired": len(shared_seeds),
                        "mean": statistics.mean(a_vals_dict.values()) if a_vals_dict else None,
                        "std": statistics.stdev(a_vals_dict.values()) if len(a_vals_dict) > 1 else 0.0,
                        "diff_mean": None, "diff_median": None,
                        "p_raw": None, "p_holm": None, "sig": None,
                        "note": "insufficient paired seeds",
                    })
                    continue
                paired_diffs = [leader_seeds[s] - a_vals_dict[s] for s in shared_seeds]
                a_mean = statistics.mean([a_vals_dict[s] for s in shared_seeds])
                a_std  = statistics.stdev([a_vals_dict[s] for s in shared_seeds]) if len(shared_seeds) > 1 else 0.0
                diff_mean = statistics.mean(paired_diffs)
                diff_median = statistics.median(paired_diffs)

                # Wilcoxon: requires non-zero differences
                non_zero = [d for d in paired_diffs if d != 0]
                if len(non_zero) == 0:
                    p_raw = 1.0  # all ties → no evidence of difference
                    note = "all paired diffs are 0"
                else:
                    try:
                        # zero_method='wilcox' drops zeros (standard default)
                        # alternative='two-sided' for FWER claim
                        stat, p_raw = wilcoxon(paired_diffs, alternative='two-sided',
                                               zero_method='wilcox')
                        if math.isnan(p_raw):
                            p_raw = 1.0
                        note = ""
                    except Exception as e:
                        p_raw = 1.0
                        note = f"wilcoxon failed: {e}"

                raw_p.append(p_raw)
                details.append({
                    "arch": a,
                    "n_paired": len(shared_seeds),
                    "mean": round(a_mean, 6),
                    "std": round(a_std, 6),
                    "diff_mean": round(diff_mean, 6),
                    "diff_median": round(diff_median, 6),
                    "p_raw": p_raw,
                    "note": note,
                })

            # Holm-Bonferroni across the (k-1) within-cell comparisons
            adj_p = holm_bonferroni(raw_p)
            for d, ap in zip(details, adj_p):
                d["p_holm"] = ap
                d["sig"] = bool(ap is not None and ap < alpha)

            out[f"{target}__{metric}"] = {
                "target": target,
                "metric": metric,
                "n_archs": len(arches_with_data),
                "leader": leader,
                "leader_mean": round(leader_mean, 6),
                "leader_std": round(leader_std, 6),
                "alpha": alpha,
                "correction": "Holm-Bonferroni",
                "test": "Wilcoxon signed-rank (two-sided)",
                "comparisons": details,
            }

    return out


def main() -> int:
    args = parse_args()
    cell = load_long_csv(args.input)
    results = run_significance(cell, args.alpha)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2, sort_keys=False, default=str)
    print(f"Wrote {args.output}")

    # Also print a readable summary
    print("\nSignificance summary (sig at p_holm < {:.2f}):".format(args.alpha))
    print("=" * 90)
    for cell_key, cdata in results.items():
        print(f"\n{cdata['target']} × {cdata['metric']}  (leader: "
              f"{cdata['leader']} = {cdata['leader_mean']:.4f}±{cdata['leader_std']:.4f})")
        print(f"  {'arch':<12} {'mean':<8} {'diff_med':<9} {'p_raw':<10} {'p_holm':<10} sig")
        for d in cdata["comparisons"]:
            sig_marker = "✗ (worse)" if d.get("sig") else "≈ (n.s.)"
            mean_str = f"{d['mean']:.4f}" if d['mean'] is not None else "  --"
            dm = d.get("diff_median")
            dm_str = f"{dm:+.4f}" if isinstance(dm, (int, float)) else "  --"
            pr = d.get("p_raw")
            pr_str = f"{pr:.2e}" if isinstance(pr, (int, float)) else "  --"
            ph = d.get("p_holm")
            ph_str = f"{ph:.2e}" if isinstance(ph, (int, float)) else "  --"
            print(f"  {d['arch']:<12} {mean_str:<8} {dm_str:<9} {pr_str:<10} {ph_str:<10} {sig_marker}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
