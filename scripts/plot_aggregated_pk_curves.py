"""Aggregate N-seed predictions into Precision@k vs VCs plots with CI bands.

For one (arch, variant) cell, this script collects the predictions CSVs that
were exported by ``Evaluator`` (via ``eval.export_predictions: true``) across
all replication seeds, deduplicates to the latest CSV per seed, and hands the
aggregated set to ``DownstreamAnalyzer.perform_downstream_analysis_aggregated``.
The analyzer produces one PDF per sub-category axis (stages / continents /
sectors / countries / funding sources) showing mean cumulative P@k curves
with bootstrap 95% CI bands across seeds and bootstrap-CI'd VC overlays.

The script is fully standalone — no W&B, no training pipeline, just CSVs from
disk plus the master ``config.yaml`` for data paths.

Examples:

    # Default: all SeHGNN g4_heterophily seeds → figures/pk_aggregated/SeHGNN_g4/
    python scripts/plot_aggregated_pk_curves.py \\
        --arch SeHGNN --variant g4_heterophily

    # Custom output dir, custom CI level, fewer bootstrap iterations
    python scripts/plot_aggregated_pk_curves.py \\
        --arch SeHGNN --variant g4_heterophily \\
        --output-dir paper/figures/pk \\
        --alpha 0.10 --n-bootstrap 2000

    # Override the predictions glob (e.g. for an arch that lives outside the
    # standard pipeline_state layout)
    python scripts/plot_aggregated_pk_curves.py \\
        --arch SeHGNN --variant g4_heterophily \\
        --predictions-glob 'outputs/pipeline_state/*/predictions/SeHGNN/masked_multi_task/*_predictions_test.csv'
"""
from __future__ import annotations

import argparse
import ast
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.ml.downstream_analysis import DownstreamAnalyzer  # noqa: E402  (after sys.path mutation)

# Canonical 20-seed replication. Mirrors scripts/plot_pk_by_year.CANONICAL_SEEDS
# so every PK-aggregation script in this repo defaults to the same whitelist;
# ad-hoc seeds (e.g. 42) that share the predictions path are excluded so the
# rendered figures stay consistent with the headline numbers in the main table.
CANONICAL_SEEDS = frozenset(range(20))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Aggregate N-seed predictions into P@k vs VCs plots with CI bands.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--arch", required=True,
                   help="Architecture name as used in pipeline_state paths (e.g. SeHGNN).")
    p.add_argument("--variant", default="g4_heterophily",
                   help="Graph variant tag — informational only; embedded in default "
                        "output subdir. Filtering happens via --predictions-glob.")
    p.add_argument("--target-mode", default="masked_multi_task",
                   help="Target mode subdirectory under predictions/. Must match the "
                        "training config (default: masked_multi_task).")
    p.add_argument("--predictions-glob", default=None,
                   help="Custom glob for predictions CSVs. Default: "
                        "outputs/pipeline_state/*/predictions/<arch>/<target_mode>/*_predictions_test.csv")
    p.add_argument("--config", type=Path, default=REPO_ROOT / "config.yaml",
                   help="Master config.yaml (used only for data paths and maturity-mask params).")
    p.add_argument("--output-dir", type=Path, default=None,
                   help="Output dir for PDFs. Default: outputs/figures/pk_aggregated/<arch>_<variant>/")
    p.add_argument("--max-k", type=int, default=1000)
    p.add_argument("--n-bootstrap", type=int, default=10000,
                   help="Bootstrap resamples for both the seed-CI and per-VC CI.")
    p.add_argument("--alpha", type=float, default=0.05,
                   help="Two-sided significance level (default 0.05 → 95%% CI).")
    p.add_argument("--top-n-sectors", type=int, default=8)
    p.add_argument("--top-n-countries", type=int, default=8)
    p.add_argument("--min-seeds", type=int, default=5,
                   help="Refuse to plot if fewer than this many seed CSVs are found.")
    p.add_argument("--allowed-seeds", type=int, nargs="+",
                   default=sorted(CANONICAL_SEEDS),
                   help="Whitelist of seeds to include. Default: canonical "
                        "20-seed replication (0-19); excludes ad-hoc seeds "
                        "such as 42 that may share the predictions path.")
    return p.parse_args()


def _safe_eval(value):
    """Parse Crunchbase-style serialised dicts/floats from a CSV cell."""
    if isinstance(value, dict):
        return value
    if isinstance(value, (int, float)):
        return float(value)
    if not isinstance(value, str) or not value:
        return 0.0
    try:
        return ast.literal_eval(value)
    except (ValueError, SyntaxError):
        try:
            return float(value)
        except ValueError:
            return 0.0


def _resolve_predictions_glob(args: argparse.Namespace) -> str:
    if args.predictions_glob:
        return args.predictions_glob
    return str(REPO_ROOT
               / "outputs" / "pipeline_state" / "*"
               / "predictions" / args.arch / args.target_mode
               / "*_predictions_test.csv")


def _resolve_output_dir(args: argparse.Namespace) -> Path:
    if args.output_dir:
        return args.output_dir
    return REPO_ROOT / "outputs" / "figures" / "pk_aggregated" / f"{args.arch}_{args.variant}"


def discover_prediction_csvs(glob: str, min_seeds: int) -> Dict[int, Path]:
    """Find prediction CSVs matching ``glob``; dedupe to latest per seed.

    The seed is read from the CSV's ``seed`` column (constant within a CSV).
    When multiple CSVs share a seed (e.g. resubmissions), the file with the
    most recent mtime wins.
    """
    paths = sorted(Path("/").glob(glob.lstrip("/")) if glob.startswith("/")
                   else REPO_ROOT.glob(str(Path(glob).relative_to(REPO_ROOT))))
    if not paths:
        raise FileNotFoundError(f"No prediction CSVs match {glob!r}")

    latest: Dict[int, Path] = {}
    for path in paths:
        try:
            seed_col = pd.read_csv(path, usecols=['seed'], nrows=1)
        except Exception as e:
            print(f"   ⚠️ Skipping unreadable {path.name}: {e}")
            continue
        if seed_col.empty or pd.isna(seed_col.iloc[0]['seed']):
            print(f"   ⚠️ Skipping {path.name}: no 'seed' column / value")
            continue
        seed = int(seed_col.iloc[0]['seed'])
        prev = latest.get(seed)
        if prev is None or path.stat().st_mtime > prev.stat().st_mtime:
            latest[seed] = path

    if len(latest) < min_seeds:
        raise RuntimeError(
            f"Only {len(latest)} unique seeds found (need ≥ {min_seeds}). "
            f"Glob={glob!r}; consider re-running replication."
        )
    return latest


def load_predictions(seed_to_path: Dict[int, Path]
                     ) -> Dict[int, List[Tuple[str, dict, dict]]]:
    """Load each seed's CSV into ``[(uuid, score_dict, label_dict), ...]`` tuples.

    Both ``prediction`` and ``gt_label`` columns are parsed via ``ast.literal_eval``
    to recover the dict structure that the analyzer expects for MTL inputs.
    """
    out: Dict[int, List[Tuple[str, dict, dict]]] = {}
    for seed in sorted(seed_to_path):
        path = seed_to_path[seed]
        df = pd.read_csv(path)
        score_col = 'prediction' if 'prediction' in df.columns else 'score'
        df[score_col] = df[score_col].apply(_safe_eval)
        df['gt_label'] = df['gt_label'].apply(_safe_eval)
        tuples = list(zip(df['org_uuid'].tolist(),
                          df[score_col].tolist(),
                          df['gt_label'].tolist()))
        out[seed] = tuples
        print(f"   seed {seed:>2}: {len(tuples):>6} rows  ({path.name})")
    return out


def main() -> int:
    args = parse_args()

    glob_pattern = _resolve_predictions_glob(args)
    output_dir = _resolve_output_dir(args)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"📂 Discovering prediction CSVs:\n   glob={glob_pattern}")
    seed_to_path = discover_prediction_csvs(glob_pattern, min_seeds=args.min_seeds)
    allowed = set(args.allowed_seeds)
    dropped = sorted(s for s in seed_to_path if s not in allowed)
    if dropped:
        print(f"   excluding non-canonical seeds: {dropped}")
    seed_to_path = {s: p for s, p in seed_to_path.items() if s in allowed}
    if len(seed_to_path) < args.min_seeds:
        raise RuntimeError(f"Only {len(seed_to_path)} canonical seeds remain "
                           f"(need >= {args.min_seeds}).")
    print(f"   found {len(seed_to_path)} unique seeds: {sorted(seed_to_path)}")

    print(f"\n📥 Loading predictions...")
    predictions_per_seed = load_predictions(seed_to_path)

    print(f"\n⚙️  Initialising DownstreamAnalyzer (config={args.config})...")
    with open(args.config) as f:
        config = yaml.safe_load(f)
    config["output_dir"] = str(output_dir)
    analyzer = DownstreamAnalyzer(config)

    print(f"\n🚀 Running aggregated analysis  →  {output_dir}")
    analyzer.perform_downstream_analysis_aggregated(
        predictions_per_seed,
        output_subdir=None,  # output_dir already redirected
        max_k=args.max_k,
        n_bootstrap=args.n_bootstrap,
        alpha=args.alpha,
        top_n_sectors=args.top_n_sectors,
        top_n_countries=args.top_n_countries,
    )
    print(f"\n✅ Done. PDFs in {output_dir}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
