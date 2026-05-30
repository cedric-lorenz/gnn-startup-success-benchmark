"""Assemble the paper-grid metrics into a long-format CSV.

This is the single source of truth for the main paper table: the per-seed
metric values for each (arch, target, metric) cell on the g4_heterophily graph
variant (plus graph-agnostic baselines), deduplicated to one run per seed.

Two sources (``--source``):
  * ``local`` (no W&B account needed): read the result JSONs every run writes
    under outputs/pipeline_state/**/results/. The same numbers W&B stores are
    already on disk; arch + graph variant identify each cell.
  * ``wandb``: pull run summaries from a W&B project, dedup latest-per-seed.

Output: outputs/paper_results/main_table_metrics.csv
        Columns: arch, group, target, metric, seed, value, run_id

This file is consumed by:
- scripts/significance_paper.py  (Wilcoxon signed-rank + Holm-Bonferroni)
- scripts/render_paper_tables.py (LaTeX/markdown table generators)

Reproduce with:
    python scripts/fetch_paper_metrics.py --output outputs/paper_results/main_table_metrics.csv
"""
from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]

# Only used in --source wandb (opt-in). Resolves from $WANDB_PROJECT, else an
# obvious placeholder so a public-repo user knows to set their own entity/project.
DEFAULT_WANDB_PROJECT = os.environ.get("WANDB_PROJECT", "<entity>/<project>")

# (display_name, w&b group)
ARCHES: List[Tuple[str, str]] = [
    # Deterministic graph-only baseline (one ok run = the canonical value;
    # mean±std collapses to value±0 in aggregation).
    ("DegreeCentrality", "replicate_DegreeCentrality__g4_heterophily"),
    ("XGBoost",    "replicate_XGBoost"),
    ("MLP",        "replicate_MLP"),
    ("GCN",        "replicate_GCN__g4_heterophily"),
    ("SageGNN",    "replicate_SageGNN__g4_heterophily"),
    ("HAN",        "replicate_HAN__g4_heterophily"),
    ("RGCN",       "replicate_RGCN__g4_heterophily"),
    ("SeHGNN",     "replicate_SeHGNN__g4_heterophily"),
    ("Hetero2Net", "replicate_Hetero2Net__g4_heterophily"),
    ("SimpleHGN",  "replicate_SimpleHGN__g4_heterophily"),
    ("VenGNN-A",    "replicate_VenGNN_a_only__g4_heterophily"),
    ("VenGNN-B",    "replicate_VenGNN__g4_heterophily"),
    ("VenGNN-Full", "replicate_VenGNN_full__g4_heterophily"),
]

# XGBoost's evaluator logs metrics with a different key convention than the
# GNN evaluator. For F1 we use `f1_binary` for XGBoost (positive-class only)
# to match the GNNs' `test_f1_*` (which is also positive-class only via the
# default classification report path).
def keys_for(arch: str, target_suffix: str) -> Dict[str, str]:
    if arch == "XGBoost":
        return {
            "AUC-PR":  f"test_{target_suffix}_auc_pr",
            "AUC-ROC": f"test_{target_suffix}_auc_roc",
            "F1":      f"test_{target_suffix}_f1_binary",
            "P@100":   f"test_{target_suffix}_precision_at_k_pos_pred_100",
            "P@1000":  f"test_{target_suffix}_precision_at_k_pos_pred_1000",
        }
    return {
        "AUC-PR":  f"test_auc_pr_{target_suffix}",
        "AUC-ROC": f"test_auc_roc_{target_suffix}",
        "F1":      f"test_f1_{target_suffix}",
        "P@100":   f"test_precision_at_k_pos_pred_100_{target_suffix}",
        "P@1000":  f"test_precision_at_k_pos_pred_1000_{target_suffix}",
    }


TARGETS = [("NFR", "mom"), ("Exit", "liq")]
METRIC_NAMES = ["AUC-PR", "AUC-ROC", "F1", "P@100", "P@1000"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", choices=["wandb", "local"], default="local",
                   help="Where to read per-seed metrics from. 'local' (default) "
                        "reads the result JSONs every run writes under "
                        "outputs/pipeline_state/**/results/ (no W&B account "
                        "needed -- the same numbers are on disk). 'wandb' pulls "
                        "the run summaries from the W&B project instead.")
    p.add_argument("--project", default=DEFAULT_WANDB_PROJECT,
                   help="W&B entity/project for --source wandb (default: "
                        "$WANDB_PROJECT, else a placeholder).")
    p.add_argument("--state-dir", type=Path,
                   default=REPO_ROOT / "outputs" / "pipeline_state",
                   help="Root scanned in --source local mode.")
    p.add_argument("--output", type=Path,
                   default=REPO_ROOT / "outputs" / "paper_results" / "main_table_metrics.csv")
    return p.parse_args()


def _local_arch_and_group(d: dict) -> Optional[Tuple[str, str]]:
    """Map a local result JSON to its (display arch, group) using the model name
    and graph variant in the config -- the W&B group string is not stored
    locally, but model + variant identify the same table cell."""
    cfg, meta = d.get("config", {}), d.get("metadata", {})
    model = meta.get("model") or cfg.get("train", {}).get("model")
    if not model:
        return None
    variant = cfg.get("data_processing", {}).get("graph_variant", "g4_heterophily")
    if model == "VenGNN":
        branch = (cfg.get("models", {}).get("VenGNN", {}).get("branch_mode")
                  or cfg.get("train", {}).get("branch_mode") or "b_only")
        disp = {"a_only": "VenGNN-A", "b_only": "VenGNN-B",
                "full": "VenGNN-Full"}.get(branch, "VenGNN-B")
        tag = {"VenGNN-A": "VenGNN_a_only", "VenGNN-B": "VenGNN",
               "VenGNN-Full": "VenGNN_full"}[disp]
        return disp, f"replicate_{tag}__{variant}"
    # graph-agnostic tabular baselines carry no variant in their group
    if model in ("MLP", "XGBoost"):
        return model, f"replicate_{model}"
    known = {"SeHGNN", "HAN", "RGCN", "SimpleHGN", "Hetero2Net", "GCN",
             "SageGNN", "DegreeCentrality"}
    if model in known:
        return model, f"replicate_{model}__{variant}"
    return None


def collect_rows_local(state_dir: Path) -> List[dict]:
    """Build the long-format rows from local result JSONs, no W&B needed.

    Each run writes outputs/pipeline_state/<job>/results/<model>/<mode>/*_test.json
    with config.seed, metadata.model, config.data_processing.graph_variant, and a
    metrics dict whose keys match `keys_for(...)`. Dedup to the latest run per
    (arch, group, seed) by timestamp.
    """
    import glob, json
    best: Dict[Tuple[str, int], Tuple[str, dict, str]] = {}
    for path in glob.glob(str(state_dir / "**" / "results" / "**" / "*_test.json"),
                          recursive=True):
        try:
            d = json.load(open(path))
        except Exception:
            continue
        seed = d.get("config", {}).get("seed")
        ag = _local_arch_and_group(d)
        if seed is None or ag is None:
            continue
        arch, group = ag
        ts = d.get("metadata", {}).get("timestamp", "")
        key = (group, int(seed))
        if key not in best or ts > best[key][0]:
            best[key] = (ts, d, arch)

    rows: List[dict] = []
    for (group, seed), (_ts, d, arch) in best.items():
        metrics = d.get("metrics", {})
        run_id = d.get("metadata", {}).get("wandb_run_id", "local")
        for target_label, target_suffix in TARGETS:
            keys = keys_for(arch, target_suffix)
            for metric_name in METRIC_NAMES:
                val = metrics.get(keys[metric_name])
                if isinstance(val, (int, float)):
                    rows.append({
                        "arch": arch, "group": group, "target": target_label,
                        "metric": metric_name, "seed": int(seed),
                        "value": float(val), "run_id": run_id,
                    })
    return rows


def fetch_runs_dedup(api, project: str, group: str):
    """Fetch finished+ok runs in `group`, keep latest per seed."""
    runs = list(api.runs(project, filters={"group": group, "state": "finished"},
                         per_page=200))
    by_seed = {}
    for r in runs:
        s = r.config.get("seed")
        if s is None or r.summary.get("repro/status") != "ok":
            continue
        prev = by_seed.get(s)
        if prev is None or str(r.created_at) > str(prev.created_at):
            by_seed[s] = r
    return list(by_seed.values())


def main() -> int:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    if args.source == "local":
        rows = collect_rows_local(args.state_dir)
        print(f"  read {len(rows)} rows from local result JSONs under {args.state_dir}")
    else:
        import wandb
        api = wandb.Api()
        rows = []
        for arch, group in ARCHES:
            runs = fetch_runs_dedup(api, args.project, group)
            for r in runs:
                seed = r.config.get("seed")
                for target_label, target_suffix in TARGETS:
                    keys = keys_for(arch, target_suffix)
                    for metric_name in METRIC_NAMES:
                        val = r.summary.get(keys[metric_name])
                        if isinstance(val, (int, float)):
                            rows.append({
                                "arch": arch,
                                "group": group,
                                "target": target_label,
                                "metric": metric_name,
                                "seed": int(seed),
                                "value": float(val),
                                "run_id": r.id,
                            })
            print(f"  {arch:<12} {group:<55} n_seeds_kept={len(runs)}")

    # Write long-format CSV
    with open(args.output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["arch", "group", "target",
                                               "metric", "seed", "value", "run_id"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nWrote {len(rows)} rows to {args.output}")

    # Quick sanity print
    cells = {}
    for r in rows:
        k = (r["arch"], r["target"], r["metric"])
        cells.setdefault(k, []).append(r["value"])
    print("\nCell completeness:")
    for arch, _ in ARCHES:
        for tgt, _ in TARGETS:
            for m in METRIC_NAMES:
                n = len(cells.get((arch, tgt, m), []))
                if n != 20:
                    print(f"  WARN {arch} {tgt} {m}: n={n}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
