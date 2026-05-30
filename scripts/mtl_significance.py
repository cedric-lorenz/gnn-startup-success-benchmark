"""Wilcoxon-Holm pairwise significance for the MTL ablation (V1/V2/V3/V4).

Reuses `paired_wilcoxon` and `holm_correction` from `pairwise_significance.py`
(the same machinery that powers Table III significance markers).

Pairs tested per metric (only pairs where both variants have a trained head
on the metric in question):

  NFR metrics  (V1, V3, V4 are valid): V1 vs V3, V1 vs V4, V3 vs V4
  Exit metrics (V2, V3, V4 are valid): V2 vs V3, V2 vs V4, V3 vs V4

Holm-Bonferroni correction is applied within each metric family (3 tests
each). Output:

  outputs/mtl_tradeoff/significance.csv   (long-form, all 12 rows)
  outputs/mtl_tradeoff/significance.md    (compact summary for paper prose)

Usage:
  python scripts/mtl_significance.py
"""
from __future__ import annotations
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))
from pairwise_significance import paired_wilcoxon, holm_correction  # noqa: E402

PER_SEED = REPO_ROOT / "outputs" / "mtl_tradeoff" / "per_seed.csv"
OUTDIR = REPO_ROOT / "outputs" / "mtl_tradeoff"

# (label, internal-variant-name)
V1, V2, V3, V4 = ("V1 STL-NFR", "stl_nfr"), ("V2 STL-Exit", "stl_exit"), \
                  ("V3 Joint", "joint"), ("V4 Balanced", "balanced")

# (metric_col_in_per_seed_csv, display_label, pairs_to_test)
PAIRS_NFR = [(V1, V3), (V1, V4), (V3, V4)]
PAIRS_EXIT = [(V2, V3), (V2, V4), (V3, V4)]

FAMILIES = [
    ("nfr_auc_pr",          "NFR AUC-PR",      PAIRS_NFR),
    ("nfr_precision_at_100", "NFR P@100",      PAIRS_NFR),
    ("exit_auc_pr",          "Exit AUC-PR",    PAIRS_EXIT),
    ("exit_precision_at_100","Exit P@100",     PAIRS_EXIT),
]


def main() -> int:
    if not PER_SEED.exists():
        raise SystemExit(f"missing {PER_SEED}; run aggregate_mtl_tradeoff.py first")
    df = pd.read_csv(PER_SEED)

    rows = []
    for metric_col, metric_label, pairs in FAMILIES:
        # First pass: per-pair raw p-value
        raw_results = []
        for (a_label, a_var), (b_label, b_var) in pairs:
            a_seeds = dict(zip(df[df.variant == a_var].seed,
                               df[df.variant == a_var][metric_col]))
            b_seeds = dict(zip(df[df.variant == b_var].seed,
                               df[df.variant == b_var][metric_col]))
            W, p, n, mean_diff = paired_wilcoxon(a_seeds, b_seeds)
            raw_results.append({
                "metric": metric_label, "metric_col": metric_col,
                "a": a_label, "b": b_label, "n_pairs": n,
                "mean_diff": mean_diff, "wilcoxon_W": W, "p_raw": p,
            })
        # Second pass: Holm correction within family
        pvals = [r["p_raw"] for r in raw_results if r["p_raw"] is not None]
        if pvals:
            adj_iter = iter(holm_correction(pvals))
        for r in raw_results:
            r["p_holm"] = next(adj_iter) if r["p_raw"] is not None else None
            r["significant_0.05"] = (r["p_holm"] is not None and r["p_holm"] < 0.05)
        rows.extend(raw_results)

    out_df = pd.DataFrame(rows)
    out_df.to_csv(OUTDIR / "significance.csv", index=False)
    print(f"wrote {(OUTDIR / 'significance.csv').relative_to(REPO_ROOT)} "
          f"({len(out_df)} rows)")

    # Compact markdown summary
    lines = ["# MTL ablation: Wilcoxon-Holm pairwise significance",
             "",
             "Paired by seed, n=20 per pair. Holm-Bonferroni within each metric (3 tests).",
             "Test = two-sided Wilcoxon signed-rank.",
             "",
             "| Metric | Pair | mean diff (a-b) | p_raw | p_Holm | sig (α=0.05) |",
             "|---|---|---|---|---|---|"]
    for r in rows:
        sig = "**yes**" if r["significant_0.05"] else "no"
        p_raw = f"{r['p_raw']:.4f}" if r['p_raw'] is not None else "n/a"
        p_holm = f"{r['p_holm']:.4f}" if r['p_holm'] is not None else "n/a"
        md = r["mean_diff"]
        lines.append(f"| {r['metric']} | {r['a']} vs {r['b']} | {md:+.4f} | {p_raw} | {p_holm} | {sig} |")
    (OUTDIR / "significance.md").write_text("\n".join(lines))
    print(f"wrote {(OUTDIR / 'significance.md').relative_to(REPO_ROOT)}")
    print()
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
