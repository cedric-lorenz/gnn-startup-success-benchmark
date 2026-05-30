"""Plot Precision@K (NFR) vs founding year, one line per K.

Pairs with ``scripts/extract_cohort_pk.py`` and ``scripts/plot_aggregated_pk_curves.py``.
The companion figure (``precision_at_k_with_ci_mom_founded_year.pdf``, x=K
with one curve per year) is good for per-cohort capacity-vs-precision
trade-off; this figure flips axes to put founding year on the x-axis and
draws one line per K. That makes the cohort effect itself the headline:
P@100 climbs monotonically from 45\% (2014) to a peak of 61\% (2020),
then collapses for 2022-2023 cohorts due to right-censoring.

Style matches the rest of the paper (serif font, thesis rcParams,
top/right spines removed, light grid). Color palette is a sequential
navy ramp so the most-selective K (K=100, the headline number) gets
the deepest, paper-anchor navy and matches the ``All Stages`` line
colour in Fig. ``vc_combined``.

Output: a single PDF (column-width). Numbers come from the same 21-seed
SeHGNN-g4 predictions used by the aggregated PK pipeline.
"""
from __future__ import annotations

import argparse
import ast
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.ml.downstream_analysis import (  # noqa: E402
    DownstreamAnalyzer,
    _THESIS_RCPARAMS,
    _apply_thesis_style,
)

# Canonical 20-seed replication used by the paper's main results table.
# Seed 42 and other ad-hoc single-run seeds live on the same predictions
# path and would otherwise be picked up automatically; the filter below
# excludes them by default so figures stay consistent with the headline
# numbers. Override with --allowed-seeds if a different seed set is
# intentionally wanted.
CANONICAL_SEEDS = frozenset(range(20))


# Paper-anchor navy + two lighter shades for the K-ordinal sequence.
# K=100 is the most selective (highest precision, the headline) and gets the
# deepest color, matching the global "All Stages" line in the existing P@K
# figures. K=1000 is the broadest and gets the lightest shade.
K_COLORS = {
    100: '#1f3d7a',   # deep navy (paper global colour)
    500: '#5775b3',   # mid blue
    1000: '#9ab0d4',  # light blue
}
K_MARKERS = {100: 'o', 500: 's', 1000: 'D'}


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


def discover_prediction_csvs(
        glob_pattern: str,
        min_seeds: int,
        job_prefix: str = None) -> Dict[int, Path]:
    """Find one prediction CSV per seed. Picks the latest mtime by default;
    if ``job_prefix`` is supplied, only SLURM job directories whose name
    starts with the prefix are considered (lets us pin to a specific sweep
    rather than the most-recently-written CSV per seed).
    """
    paths = sorted(REPO_ROOT.glob(str(Path(glob_pattern).relative_to(REPO_ROOT)))
                   if not glob_pattern.startswith("/")
                   else Path("/").glob(glob_pattern.lstrip("/")))
    if not paths:
        raise FileNotFoundError(f"No prediction CSVs match {glob_pattern!r}")
    if job_prefix is not None:
        before = len(paths)
        paths = [p for p in paths if p.parts[-5].startswith(job_prefix)]
        print(f"   job-prefix filter '{job_prefix}': {before} -> {len(paths)} CSVs",
              file=sys.stderr)
        if not paths:
            raise FileNotFoundError(
                f"No prediction CSVs match {glob_pattern!r} with "
                f"job-prefix {job_prefix!r}")
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


def compute_per_seed_pk(task_preds_per_seed: Dict[int, List[Tuple[str, float, float]]],
                        full_df: pd.DataFrame,
                        ks: List[int],
                        year_range: Tuple[int, int],
                        min_support: int,
                        ) -> Dict[int, Dict[int, List[float]]]:
    """Returns ``{year: {k: [pk_seed_0, pk_seed_1, ...]}}``."""
    year_counts = full_df['founded_year'].dropna().astype(int).value_counts()
    lo, hi = year_range
    years = sorted([int(y) for y in year_counts.index
                    if lo <= int(y) <= hi and year_counts[y] >= min_support])

    results: Dict[int, Dict[int, List[float]]] = {y: {k: [] for k in ks} for y in years}
    for year in years:
        year_uuids = set(full_df[full_df['founded_year'] == year]['org_uuid'])
        n_cohort = len(year_uuids)
        for preds in task_preds_per_seed.values():
            filt = [(u, s, l) for (u, s, l) in preds if u in year_uuids]
            filt.sort(key=lambda x: x[1], reverse=True)
            for k in ks:
                # Skip degenerate cells: when K > n_cohort, top-K is the
                # whole cohort and P@K collapses to the cohort positive
                # rate, which is not comparable to P@K on cohorts where
                # K < n_cohort.
                if k > n_cohort:
                    continue
                top = filt[:k]
                if not top:
                    continue
                pk = sum(l for _, _, l in top) / k
                results[year][k].append(pk)
    return results


def bootstrap_ci(values: List[float], n_bootstrap: int, alpha: float,
                 rng: np.random.Generator) -> Tuple[float, float, float]:
    """Returns (mean, mean-std, mean+std) so the rendered error bars show
    cross-seed spread rather than the CI of the mean. Bootstrap CI of the
    seed mean shrinks as ~1/sqrt(N) and collapses to a single pixel at
    K=500 and K=1000 in this figure, hiding the seed-to-seed variability
    that the reader actually wants to see. The function name is kept for
    backward compatibility with the rest of the script.
    ``n_bootstrap`` and ``alpha`` are accepted but unused.
    """
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return float('nan'), float('nan'), float('nan')
    if arr.size == 1:
        return float(arr[0]), float(arr[0]), float(arr[0])
    mean = float(arr.mean())
    std = float(arr.std(ddof=1))
    return mean, mean - std, mean + std


def render_figure(per_seed: Dict[int, Dict[int, List[float]]],
                  ks: List[int],
                  output_path: Path,
                  n_bootstrap: int,
                  alpha: float,
                  censored_years: Tuple[int, ...]) -> None:
    rng = np.random.default_rng(seed=20250513)
    years = sorted(per_seed.keys())

    # Pre-compute means + CIs per (year, k).
    table = {k: {'mean': [], 'lo': [], 'hi': [], 'years': []} for k in ks}
    for k in ks:
        for year in years:
            vals = per_seed[year].get(k, [])
            m, lo, hi = bootstrap_ci(vals, n_bootstrap=n_bootstrap, alpha=alpha, rng=rng)
            table[k]['mean'].append(m * 100)
            table[k]['lo'].append((m - lo if not np.isnan(lo) else 0.0) * 100)
            table[k]['hi'].append((hi - m if not np.isnan(hi) else 0.0) * 100)
            table[k]['years'].append(year)

    # This panel is much smaller (3.5"x2.0") than the subgroup precision@K
    # plots (7.5"x4-5"), so the same point sizes look ~2x larger here in
    # print. Shrink relative to the rcParams defaults so labels don't
    # dominate the panel.
    _local_rc = {
        **_THESIS_RCPARAMS,
        "axes.labelsize": 7,
        "xtick.labelsize": 6,
        "ytick.labelsize": 6,
        "legend.fontsize": 6,
    }
    with plt.rc_context(_local_rc):
        # Column-width: ~3.45 in for IEEE conference single column.
        fig, ax = plt.subplots(figsize=(3.5, 2.0))

        # Censored-window backdrop: lightly shade the right-censored years
        # so the reader sees at a glance which cohorts are confound-heavy.
        # The text label is positioned after the y-limits are set, below.
        censored_min = censored_max = None
        if censored_years:
            censored_min, censored_max = min(censored_years), max(censored_years)
            ax.axvspan(censored_min - 0.5, censored_max + 0.5,
                       facecolor='#e0e0e0', alpha=0.45,
                       zorder=0)

        for k in ks:
            color = K_COLORS.get(k, '#444444')
            marker = K_MARKERS.get(k, 'o')
            y = np.asarray(table[k]['mean'])
            lo = np.asarray(table[k]['lo'])
            hi = np.asarray(table[k]['hi'])
            yrs = table[k]['years']
            ax.errorbar(yrs, y,
                        yerr=[lo, hi],
                        fmt=f'-{marker}',
                        color=color,
                        ecolor=color,
                        elinewidth=1.0,
                        capsize=2,
                        markersize=4.5,
                        markerfacecolor=color,
                        markeredgecolor='white',
                        markeredgewidth=0.6,
                        linewidth=1.6,
                        alpha=0.95,
                        label=f'$K={k}$',
                        zorder=3 if k == 100 else 2)

        _apply_thesis_style(ax)
        ax.set_xlabel('Founding year')
        ax.set_ylabel('Precision (%)')
        ax.set_xticks(years)
        ax.set_xticklabels([str(y) for y in years], rotation=35, fontsize=6,
                           ha='right')
        # Pad the top so the legend (above-axes) doesn't crowd the highest data
        # point, and the bottom so the K=1000 baseline doesn't graze the spine.
        # Filter NaN: a K larger than the cohort size produces NaN P@K (e.g.
        # K=1000 in a 45-startup cohort), which would otherwise poison min/max.
        finite_vals = [v for k in ks for v in table[k]['mean']
                       if v is not None and not np.isnan(v)]
        if finite_vals:
            data_min = min(finite_vals)
            data_max = max(finite_vals)
            ax.set_ylim(max(0.0, data_min - 2), data_max + 3)
        # else leave matplotlib defaults — no finite data anywhere on this panel.

        # Legend above the axes so it never overlaps data, including the
        # right-censored region where K=100 plunges past K=500 and K=1000.
        leg = ax.legend(loc='lower center', bbox_to_anchor=(0.5, 1.02),
                        frameon=False, ncol=3, columnspacing=1.4,
                        handlelength=1.5, borderpad=0.3)
        leg.set_zorder(10)

        # Place the "right-censored" annotation now that y-limits are final;
        # tuck it just under the legend, above the K=100 line at year 2022.
        if censored_min is not None:
            ax.text(censored_min - 0.35, ax.get_ylim()[1] - 1.5,
                    'right-censored',
                    fontsize=6, color='#555555', va='top', ha='left',
                    style='italic', zorder=1)

        fig.tight_layout(pad=0.4)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path)
        plt.close(fig)
        print(f"Wrote {output_path}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arch", default="SeHGNN")
    parser.add_argument("--target-mode", default="masked_multi_task")
    parser.add_argument("--task", choices=['mom', 'liq'], default='mom')
    parser.add_argument("--ks", type=int, nargs="+", default=[100, 500, 1000])
    parser.add_argument("--min-support", type=int, default=50,
                        help="Drop years with fewer than this many test startups.")
    parser.add_argument("--year-range", type=int, nargs=2, default=[2014, 2021],
                        help="Inclusive founding-year range to plot. Defaults "
                             "to 2014-2021; 2022 and 2023 are right-censored "
                             "(insufficient label window) and excluded.")
    parser.add_argument("--censored-years", type=int, nargs="*", default=[],
                        help="Years to shade as right-censored backdrop. "
                             "Default empty; the excluded 2022-2023 range "
                             "is documented in the prose, not shaded.")
    parser.add_argument("--n-bootstrap", type=int, default=10000)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "config.yaml")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--min-seeds", type=int, default=5)
    parser.add_argument("--allowed-seeds", type=int, nargs="+",
                        default=sorted(CANONICAL_SEEDS),
                        help="Whitelist of seeds to include. Default: canonical "
                             "20-seed replication (0-19); excludes ad-hoc seeds "
                             "such as 42 that may share the predictions path.")
    parser.add_argument("--job-prefix", type=str, default=None,
                        help="Filter prediction CSVs to SLURM job dirs whose "
                             "name starts with this prefix. Default (None) uses "
                             "all matching CSVs (latest-mtime per seed) — correct "
                             "for a fresh reproduction. Set it to your own sweep's "
                             "run-id/job prefix to pin a specific batch.")
    args = parser.parse_args()

    output = args.output or (REPO_ROOT / "figures" / "pk_aggregated"
                             / f"{args.arch}_g4_heterophily"
                             / f"precision_by_year_{args.task}.pdf")

    glob_pattern = str(REPO_ROOT
                       / "outputs" / "pipeline_state" / "*"
                       / "predictions" / args.arch / args.target_mode
                       / "*_predictions_test.csv")

    print(f"Discovering CSVs: {glob_pattern}", file=sys.stderr)
    seed_to_path = discover_prediction_csvs(
        glob_pattern,
        min_seeds=args.min_seeds,
        job_prefix=args.job_prefix or None,
    )
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

    use_mature_filter = (args.task == 'liq')
    task_preds_per_seed = {
        seed: analyzer._extract_mtl_task(preds, args.task, use_mature_filter)
        for seed, preds in predictions_per_seed.items()
    }
    task_preds_per_seed = {s: p for s, p in task_preds_per_seed.items() if p}

    union_preds = [p for sp in task_preds_per_seed.values() for p in sp]
    full_df = analyzer._build_full_df(union_preds, analyzer.orgs_df)
    full_df['founded_year'] = pd.to_datetime(full_df['founded_on'], errors='coerce').dt.year

    per_seed = compute_per_seed_pk(task_preds_per_seed, full_df,
                                   ks=args.ks,
                                   year_range=tuple(args.year_range),
                                   min_support=args.min_support)

    render_figure(per_seed, ks=args.ks,
                  output_path=output,
                  n_bootstrap=args.n_bootstrap,
                  alpha=args.alpha,
                  censored_years=tuple(args.censored_years))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
