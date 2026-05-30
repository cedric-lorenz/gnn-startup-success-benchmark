"""Extract cohort-stratified Precision@K headlines from the SeHGNN g4 21-seed run.

Pairs with ``scripts/plot_aggregated_pk_curves.py`` and uses the same
``DownstreamAnalyzer`` primitives (``_extract_mtl_task``, ``_build_full_df``,
``analyze_portfolio_by_founded_year``) so the numbers it prints come from
exactly the same data as ``figures/precision_at_k_with_ci_mom_founded_year.pdf``.

Output: a Markdown table of P@K mean and std (across seeds) for each founding
year in 2014-2023 with at least the requested minimum support. By default
reports K in {100, 500, 1000}; override with --ks.

Example::

    python scripts/extract_cohort_pk.py --arch SeHGNN --task mom \
        --ks 100 500 1000

The script does *not* compute bootstrap CI bands; the figure carries those.
For prose-friendly headlines we only need mean +- std across seeds.
"""
from __future__ import annotations

import argparse
import ast
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.ml.downstream_analysis import DownstreamAnalyzer  # noqa: E402

# Canonical 20-seed replication. Mirrors plot_pk_by_year.CANONICAL_SEEDS so
# both scripts share the same default whitelist; ad-hoc seeds (e.g. 42) are
# excluded so the numbers in the paper stay reproducible from disk.
CANONICAL_SEEDS = frozenset(range(20))


def _safe_eval(value):
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


def discover_prediction_csvs(glob_pattern: str, min_seeds: int) -> Dict[int, Path]:
    paths = sorted(REPO_ROOT.glob(str(Path(glob_pattern).relative_to(REPO_ROOT)))
                   if not glob_pattern.startswith("/")
                   else Path("/").glob(glob_pattern.lstrip("/")))
    if not paths:
        raise FileNotFoundError(f"No prediction CSVs match {glob_pattern!r}")
    latest: Dict[int, Path] = {}
    for path in paths:
        try:
            head = pd.read_csv(path, usecols=['seed'], nrows=1)
        except Exception as e:
            print(f"   skipping unreadable {path.name}: {e}", file=sys.stderr)
            continue
        if head.empty or pd.isna(head.iloc[0]['seed']):
            continue
        seed = int(head.iloc[0]['seed'])
        prev = latest.get(seed)
        if prev is None or path.stat().st_mtime > prev.stat().st_mtime:
            latest[seed] = path
    if len(latest) < min_seeds:
        raise RuntimeError(f"Only {len(latest)} unique seeds (need >= {min_seeds}).")
    return latest


def load_predictions(seed_to_path: Dict[int, Path]
                     ) -> Dict[int, List[Tuple[str, dict, dict]]]:
    out: Dict[int, List[Tuple[str, dict, dict]]] = {}
    for seed in sorted(seed_to_path):
        df = pd.read_csv(seed_to_path[seed])
        score_col = 'prediction' if 'prediction' in df.columns else 'score'
        df[score_col] = df[score_col].apply(_safe_eval)
        df['gt_label'] = df['gt_label'].apply(_safe_eval)
        out[seed] = list(zip(df['org_uuid'].tolist(),
                             df[score_col].tolist(),
                             df['gt_label'].tolist()))
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arch", default="SeHGNN")
    parser.add_argument("--target-mode", default="masked_multi_task")
    parser.add_argument("--task", choices=['mom', 'liq'], default='mom')
    parser.add_argument("--ks", type=int, nargs="+", default=[100, 500, 1000])
    parser.add_argument("--min-support", type=int, default=50,
                        help="Drop years with fewer than this many test startups.")
    parser.add_argument("--year-range", type=int, nargs=2, default=[2014, 2023])
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "config.yaml")
    parser.add_argument("--min-seeds", type=int, default=5)
    parser.add_argument("--output-csv", type=Path, default=None,
                        help="Override the per-year CSV output path. Default: "
                             "outputs/figures/pk_aggregated/<arch>_g4_heterophily/"
                             "cohort_pk_<task>.csv")
    parser.add_argument("--allowed-seeds", type=int, nargs="+",
                        default=sorted(CANONICAL_SEEDS),
                        help="Whitelist of seeds. Default: canonical 0-19.")
    args = parser.parse_args()

    glob_pattern = str(REPO_ROOT
                       / "outputs" / "pipeline_state" / "*"
                       / "predictions" / args.arch / args.target_mode
                       / "*_predictions_test.csv")

    print(f"Discovering CSVs: {glob_pattern}", file=sys.stderr)
    seed_to_path = discover_prediction_csvs(glob_pattern, min_seeds=args.min_seeds)
    allowed = set(args.allowed_seeds)
    dropped = sorted(s for s in seed_to_path if s not in allowed)
    if dropped:
        print(f"Excluding non-canonical seeds: {dropped}", file=sys.stderr)
    seed_to_path = {s: p for s, p in seed_to_path.items() if s in allowed}
    if len(seed_to_path) < args.min_seeds:
        raise RuntimeError(f"Only {len(seed_to_path)} canonical seeds remain "
                           f"(need >= {args.min_seeds}).")
    print(f"Seeds: {sorted(seed_to_path)} (n={len(seed_to_path)})", file=sys.stderr)
    predictions_per_seed = load_predictions(seed_to_path)

    with open(args.config) as f:
        config = yaml.safe_load(f)
    analyzer = DownstreamAnalyzer(config)

    # Task projection: mom uses full population (no maturity mask), per
    # _run_aggregated_for_task in downstream_analysis.py.
    use_mature_filter = (args.task == 'liq')
    task_preds_per_seed: Dict[int, List[Tuple[str, float, float]]] = {
        seed: analyzer._extract_mtl_task(preds, args.task, use_mature_filter)
        for seed, preds in predictions_per_seed.items()
    }
    task_preds_per_seed = {s: p for s, p in task_preds_per_seed.items() if p}
    print(f"Task '{args.task}': {len(task_preds_per_seed)} usable seeds", file=sys.stderr)

    # Build full_df + derive founded_year (mirrors _run_aggregated_for_task).
    union_preds = [p for sp in task_preds_per_seed.values() for p in sp]
    full_df = analyzer._build_full_df(union_preds, analyzer.orgs_df)
    full_df['founded_year'] = pd.to_datetime(full_df['founded_on'], errors='coerce').dt.year

    year_counts = full_df['founded_year'].dropna().astype(int).value_counts()
    lo, hi = args.year_range
    years = sorted([int(y) for y in year_counts.index
                    if lo <= int(y) <= hi and year_counts[y] >= args.min_support])

    # Per-year, per-seed P@K computation.
    rows = []
    for year in years:
        year_uuids = set(full_df[full_df['founded_year'] == year]['org_uuid'])
        support = len(year_uuids)
        positives_total = int(full_df[(full_df['founded_year'] == year)
                                      & (full_df['gt_label'] == 1)]['org_uuid'].nunique()) \
            if 'gt_label' in full_df.columns else None
        per_k = {k: [] for k in args.ks}
        for preds in task_preds_per_seed.values():
            filt = [(u, s, l) for (u, s, l) in preds if u in year_uuids]
            filt.sort(key=lambda x: x[1], reverse=True)
            for k in args.ks:
                top = filt[:k]
                if not top:
                    continue
                # Top-K with fewer than K available means K is capped to support.
                effective_k = min(k, len(top))
                pk = sum(l for _, _, l in top) / effective_k
                per_k[k].append(pk)
        row = {'year': year, 'support': support}
        for k in args.ks:
            vals = per_k[k]
            if not vals:
                row[f'P@{k}_mean'] = float('nan')
                row[f'P@{k}_std'] = float('nan')
            else:
                row[f'P@{k}_mean'] = float(np.mean(vals))
                row[f'P@{k}_std'] = float(np.std(vals))
        rows.append(row)

    if not rows:
        print("No qualifying years found.", file=sys.stderr)
        return 1

    df = pd.DataFrame(rows)
    headers = ['year', 'support'] + [f'P@{k}' for k in args.ks]
    print('| ' + ' | '.join(headers) + ' |')
    print('|' + '|'.join(['---'] * len(headers)) + '|')
    for r in rows:
        cells = [str(r['year']), str(r['support'])]
        for k in args.ks:
            m, s = r[f'P@{k}_mean'], r[f'P@{k}_std']
            cells.append(f'{m*100:.1f} +/- {s*100:.1f}')
        print('| ' + ' | '.join(cells) + ' |')

    # Dump per-year CSV. Default lives under outputs/ (gitignored) so a fresh
    # run never pollutes the repo root.
    out_csv = args.output_csv if args.output_csv else (
        REPO_ROOT / "outputs" / "figures" / "pk_aggregated"
        / f"{args.arch}_g4_heterophily"
        / f"cohort_pk_{args.task}.csv"
    )
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"\nCSV: {out_csv}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
