"""Aggregate the MTL-tradeoff sweep (V1 stl_nfr / V2 stl_exit / V3 joint) into
a per-seed metrics CSV, a per-variant summary, and a markdown table.

Reads experiments/registry/mtl_tradeoff_submissions.json to discover the
(variant, seed, job_id) tuples submitted by scripts/submit_mtl_tradeoff_sweep.py,
locates the predictions CSV each run wrote
(outputs/pipeline_state/{job_id}_*/predictions/SeHGNN/masked_multi_task/*_predictions_test.csv),
and computes the VC-deployment metric set:

  - Standard discrimination: AUC-ROC, AUC-PR per target
  - P@K and Recall@K at K in {10, 50, 100, 500, 1000} per target
  - Top-K stability: mean pairwise Jaccard of top-100 uuids across seeds, per variant
  - Top-K diversity (within top-100): Herfindahl index over continent and sector
  - Joint top-K overlap: |top-100 NFR uuids intersect top-100 Exit uuids| per seed

Exit metrics are restricted to the mature subset using the same maturity mask
that eval.py applies internally (paper criteria, src/ml/utils.py:get_maturity_mask).

Usage:
  python scripts/aggregate_mtl_tradeoff.py
  python scripts/aggregate_mtl_tradeoff.py --variants joint
  python scripts/aggregate_mtl_tradeoff.py --top-k 100 --output-dir outputs/mtl_tradeoff/
"""
from __future__ import annotations
import argparse
import ast
import glob
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

SUBMISSIONS_PATH = REPO_ROOT / "experiments" / "registry" / "mtl_tradeoff_submissions.json"
PIPELINE_STATE = REPO_ROOT / "outputs" / "pipeline_state"
GRAPH_CACHE = REPO_ROOT / "outputs" / "graphs"
DEFAULT_OUTDIR = REPO_ROOT / "outputs" / "mtl_tradeoff"

VARIANT_LABEL = {
    "stl_nfr": "V1 (STL-NFR)",
    "stl_exit": "V2 (STL-Exit)",
    "joint": "V3 (Joint, paper)",
    "balanced": "V4 (Balanced 1:1)",
}
VARIANT_ORDER = ["stl_nfr", "stl_exit", "joint", "balanced"]


# ---------------------------------------------------------------------------
# I/O: submissions registry and predictions CSV discovery
# ---------------------------------------------------------------------------

def load_submission_index() -> List[Dict]:
    """Flatten the per-batch submission record into a single list of dicts."""
    if not SUBMISSIONS_PATH.exists():
        raise SystemExit(f"No submissions registry at {SUBMISSIONS_PATH}")
    with open(SUBMISSIONS_PATH) as f:
        data = json.load(f)
    out: List[Dict] = []
    for batch_key, jobs in data.items():
        for j in jobs:
            j = dict(j)
            j["batch_key"] = batch_key
            out.append(j)
    return out


def find_predictions_csv(job_id: int) -> Optional[Path]:
    pattern = str(PIPELINE_STATE / f"{job_id}_*" / "predictions" / "SeHGNN" /
                  "masked_multi_task" / "*_predictions_test.csv")
    matches = glob.glob(pattern)
    if not matches:
        return None
    matches.sort()
    return Path(matches[-1])


def _variant_from_loss(loss: Dict) -> Optional[str]:
    """Classify a run's MTL variant from its train.loss weights.

    The four Table-III variants are fully determined by the (momentum,
    liquidity) loss-weight pair, so a local run can be classified without the
    submission registry:
      * stl_nfr   — liquidity_weight == 0 (NFR head only)
      * stl_exit  — momentum_weight  == 0 (Exit head only)
      * balanced  — momentum_weight == liquidity_weight (both > 0, the 1:1 mix)
      * joint     — both > 0 and unequal (the champion's tuned ratio)
    """
    mw = loss.get("momentum_weight")
    lw = loss.get("liquidity_weight")
    if mw is None or lw is None:
        return None
    if lw == 0 and mw > 0:
        return "stl_nfr"
    if mw == 0 and lw > 0:
        return "stl_exit"
    if mw > 0 and lw > 0:
        return "balanced" if abs(mw - lw) < 1e-9 else "joint"
    return None


def collect_runs_local(state_dir: Path, variants: Sequence[str]) -> List[Dict]:
    """Registry-free discovery of MTL-variant runs from local result JSONs.

    Scans <state_dir>/**/results/SeHGNN/masked_multi_task/*_test.json, reads
    each run's seed + loss weights to classify its variant, and pairs it with
    the predictions CSV in the same job directory. Dedup to the latest run per
    (variant, seed) by timestamp. Mirrors load_submission_index()'s output
    shape so the rest of main() is source-agnostic.
    """
    pattern = str(state_dir / "**" / "results" / "SeHGNN" / "masked_multi_task" /
                  "*_test.json")
    best: Dict[Tuple[str, int], Tuple[str, Dict]] = {}
    for path in glob.glob(pattern, recursive=True):
        try:
            d = json.load(open(path))
        except Exception:
            continue
        cfg = d.get("config", {})
        seed = cfg.get("seed")
        variant = _variant_from_loss(cfg.get("train", {}).get("loss", {}))
        if seed is None or variant is None or variant not in variants:
            continue
        # predictions CSV lives in the sibling predictions/ tree of this job dir
        # path = <job>/results/SeHGNN/masked_multi_task/*_test.json -> parents[3] = <job>
        job_dir = Path(path).parents[3]
        csvs = sorted(glob.glob(str(job_dir / "predictions" / "SeHGNN" /
                                    "masked_multi_task" / "*_predictions_test.csv")))
        if not csvs:
            continue
        ts = d.get("metadata", {}).get("timestamp", "")
        key = (variant, int(seed))
        if key not in best or ts > best[key][0]:
            best[key] = (ts, {"mtl_variant": variant, "seed": int(seed),
                              "predictions_csv": csvs[-1],
                              "graph_version": _graph_version_from_job(job_dir)})
    return [rec for _ts, rec in best.values()]


def _graph_version_from_job(job_dir: Path) -> Optional[str]:
    """Read the graph cache hash this run used from its run_context.json so the
    attribute join can locate the right graph without the user knowing the hash."""
    ctx = job_dir / "results" / "run_context.json"
    if not ctx.exists():
        return None
    try:
        data = json.load(open(ctx))
    except Exception:
        return None
    return data.get("graph_version") or data.get("_run_context", {}).get("graph_version")


def parse_predictions(csv_path: Path) -> pd.DataFrame:
    """Parse a per-startup predictions CSV.

    The prediction and gt_label columns are stringified Python dicts like
    "{'mom': 0.29..., 'liq': 0.50...}". We use ast.literal_eval per cell.
    """
    df = pd.read_csv(csv_path)
    # The single-quote dict format is not JSON; ast.literal_eval handles it.
    preds = df["prediction"].apply(ast.literal_eval)
    labs = df["gt_label"].apply(ast.literal_eval)
    out = pd.DataFrame({
        "org_uuid": df["org_uuid"],
        "p_mom": preds.map(lambda d: float(d["mom"])),
        "p_liq": preds.map(lambda d: float(d["liq"])),
        "y_mom": labs.map(lambda d: float(d["mom"])),
        "y_liq": labs.map(lambda d: float(d["liq"])),
    })
    return out


# ---------------------------------------------------------------------------
# Attribute join: maturity mask + continent + sector (best-effort)
# ---------------------------------------------------------------------------

def _stub_config_for_maturity() -> Dict:
    """Minimal config that turns on the paper's strict_gating maturity rule."""
    return {
        "data_processing": {
            "strict_gating": {
                "enabled": True,
                "late_stage_funding_threshold": 15_000_000,
                "employee_count_threshold": 3,
                "compounder_age_threshold": 5,
                "compounder_funding_threshold": 3_000_000,
            }
        }
    }


def load_attributes(graph_path: Path) -> pd.DataFrame:
    """Build a per-uuid attribute table: is_mature, continent, primary_sector.

    Returns one row per startup_uuid. Missing attributes are NaN (the caller
    handles graceful degradation for diversity metrics).
    """
    from src.ml.utils import get_maturity_mask
    g, meta = torch.load(graph_path, map_location="cpu", weights_only=False)

    # raw_df has founded_on (date), df only has founded_on_year — maturity
    # needs the date column so we use raw_df.
    df = g["startup"].raw_df.copy().reset_index(drop=True)
    df["__node_idx"] = np.arange(len(df))

    # Maturity mask from paper criteria
    try:
        mature = get_maturity_mask(df, _stub_config_for_maturity())
        if mature is None:
            raise RuntimeError("get_maturity_mask returned None (missing column)")
        df["is_mature"] = mature.astype(bool).values
    except Exception as e:
        print(f"  [load_attributes] maturity mask unavailable: {e}")
        df["is_mature"] = False

    # Continent via uuid -> country_code (organizations.csv) -> convert_to_continent
    df["continent"] = pd.NA
    try:
        from src.data_engineering.aux_pipeline import convert_to_continent
        orgs_csv = REPO_ROOT / "data" / "crunchbase" / "2023" / "organizations.csv"
        if not orgs_csv.exists():
            raise FileNotFoundError(orgs_csv)
        orgs = pd.read_csv(
            orgs_csv, usecols=["uuid", "country_code"], low_memory=False
        ).rename(columns={"uuid": "startup_uuid"})
        df = df.merge(orgs, on="startup_uuid", how="left")
        df["continent"] = df["country_code"].apply(
            lambda c: convert_to_continent(c) if pd.notna(c) else pd.NA
        )
    except Exception as e:
        print(f"  [load_attributes] continent unavailable: {e!r}")

    # Primary sector via startup -> sector edges (if present in this variant)
    df["primary_sector"] = pd.NA
    try:
        sector_edge = None
        for et in g.edge_types:
            if et[0] == "startup" and "sector" in et[2]:
                sector_edge = et; break
            if et[2] == "startup" and "sector" in et[0]:
                sector_edge = et; break
        if sector_edge is not None:
            ei = g[sector_edge].edge_index.numpy()
            sector_meta = meta["sector"]  # sector_uuid, sector_name, sector_id
            primary_sector = {}
            if sector_edge[0] == "startup":
                s_arr, sec_arr = ei[0], ei[1]
            else:
                s_arr, sec_arr = ei[1], ei[0]
            for s_idx, sec_idx in zip(s_arr, sec_arr):
                primary_sector.setdefault(int(s_idx), int(sec_idx))
            sector_name = {int(r["sector_id"]): r["sector_name"]
                           for _, r in sector_meta.iterrows()
                           if pd.notna(r.get("sector_id"))}
            df["primary_sector"] = df["__node_idx"].map(
                lambda i: sector_name.get(primary_sector.get(i)) if i in primary_sector else pd.NA
            )
        else:
            print("  [load_attributes] no startup->sector edge in this graph (g4_heterophily prunes it)")
    except Exception as e:
        print(f"  [load_attributes] primary sector unavailable: {e}")

    return df[["startup_uuid", "is_mature", "continent", "primary_sector"]].rename(
        columns={"startup_uuid": "org_uuid"}
    )


# ---------------------------------------------------------------------------
# Per-seed metrics
# ---------------------------------------------------------------------------

def _topk_uuids(df: pd.DataFrame, score_col: str, k: int) -> List[str]:
    if df.empty or k <= 0:
        return []
    return df.nlargest(k, score_col)["org_uuid"].tolist()


def _herfindahl(values: Sequence) -> float:
    """Sum of squared category shares. 1.0 = single category, 1/n = uniform."""
    vals = [v for v in values if pd.notna(v)]
    if not vals:
        return float("nan")
    counts = pd.Series(vals).value_counts(normalize=True)
    return float((counts ** 2).sum())


def _precision_recall_at_k(df: pd.DataFrame, score_col: str, label_col: str,
                           ks: Sequence[int]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    if df.empty:
        for k in ks:
            out[f"precision_at_{k}"] = float("nan")
            out[f"recall_at_{k}"] = float("nan")
        return out
    pos_total = int(df[label_col].sum())
    ordered = df.sort_values(score_col, ascending=False).reset_index(drop=True)
    for k in ks:
        if k > len(ordered):
            out[f"precision_at_{k}"] = float("nan")
            out[f"recall_at_{k}"] = float("nan")
            continue
        top = ordered.iloc[:k]
        hits = int(top[label_col].sum())
        out[f"precision_at_{k}"] = hits / k
        out[f"recall_at_{k}"] = (hits / pos_total) if pos_total > 0 else float("nan")
    return out


def compute_per_seed_metrics(preds: pd.DataFrame, attrs: pd.DataFrame,
                             ks: Sequence[int], k_diversity: int = 100) -> Dict:
    """One row of metrics for a single (variant, seed) run."""
    merged = preds.merge(attrs, on="org_uuid", how="left")
    row: Dict = {}

    # --- NFR: full test population ---
    nfr_df = merged[["org_uuid", "p_mom", "y_mom", "continent", "primary_sector"]].copy()
    nfr_valid = nfr_df.dropna(subset=["y_mom"])
    if len(nfr_valid) >= 2 and nfr_valid["y_mom"].nunique() > 1:
        row["nfr_auc_roc"] = roc_auc_score(nfr_valid["y_mom"], nfr_valid["p_mom"])
        row["nfr_auc_pr"] = average_precision_score(nfr_valid["y_mom"], nfr_valid["p_mom"])
    else:
        row["nfr_auc_roc"] = row["nfr_auc_pr"] = float("nan")
    row.update({f"nfr_{k}": v for k, v in
                _precision_recall_at_k(nfr_valid, "p_mom", "y_mom", ks).items()})
    nfr_topk = _topk_uuids(nfr_valid, "p_mom", k_diversity)
    row["nfr_top100_uuids"] = nfr_topk
    topk_attr = merged.set_index("org_uuid").reindex(nfr_topk)
    row["nfr_top100_hhi_continent"] = _herfindahl(topk_attr["continent"].dropna())
    row["nfr_top100_hhi_sector"] = _herfindahl(topk_attr["primary_sector"].dropna())

    # --- Exit: restrict to mature subset (paper protocol) ---
    exit_df = merged[merged["is_mature"] == True].copy()  # noqa: E712
    exit_valid = exit_df.dropna(subset=["y_liq"])
    if len(exit_valid) >= 2 and exit_valid["y_liq"].nunique() > 1:
        row["exit_auc_roc"] = roc_auc_score(exit_valid["y_liq"], exit_valid["p_liq"])
        row["exit_auc_pr"] = average_precision_score(exit_valid["y_liq"], exit_valid["p_liq"])
    else:
        row["exit_auc_roc"] = row["exit_auc_pr"] = float("nan")
    row.update({f"exit_{k}": v for k, v in
                _precision_recall_at_k(exit_valid, "p_liq", "y_liq", ks).items()})
    exit_topk = _topk_uuids(exit_valid, "p_liq", k_diversity)
    row["exit_top100_uuids"] = exit_topk
    topk_attr_exit = merged.set_index("org_uuid").reindex(exit_topk)
    row["exit_top100_hhi_continent"] = _herfindahl(topk_attr_exit["continent"].dropna())
    row["exit_top100_hhi_sector"] = _herfindahl(topk_attr_exit["primary_sector"].dropna())

    # --- Joint top-K overlap (full population, both heads) ---
    joint_nfr = _topk_uuids(merged.dropna(subset=["y_mom"]), "p_mom", k_diversity)
    joint_exit = _topk_uuids(merged.dropna(subset=["y_liq"]), "p_liq", k_diversity)
    row["joint_top100_overlap"] = len(set(joint_nfr) & set(joint_exit))
    row["nfr_n_eval"] = int(len(nfr_valid))
    row["exit_n_eval"] = int(len(exit_valid))
    row["n_mature"] = int(merged["is_mature"].sum())
    return row


# ---------------------------------------------------------------------------
# Cross-seed aggregation
# ---------------------------------------------------------------------------

def jaccard(a: Sequence, b: Sequence) -> float:
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return float("nan")
    return len(sa & sb) / max(1, len(sa | sb))


def top_k_stability(top_k_lists: List[List[str]]) -> float:
    """Mean pairwise Jaccard across seeds. Defined when >=2 seeds available."""
    if len(top_k_lists) < 2:
        return float("nan")
    sims = []
    for i in range(len(top_k_lists)):
        for j in range(i + 1, len(top_k_lists)):
            sims.append(jaccard(top_k_lists[i], top_k_lists[j]))
    return float(np.nanmean(sims)) if sims else float("nan")


def aggregate_variant(per_seed_rows: List[Dict]) -> Dict:
    """Reduce per-seed rows to mean/std and stability."""
    if not per_seed_rows:
        return {}
    df = pd.DataFrame(per_seed_rows)
    scalar_cols = [c for c in df.columns
                   if c not in ("seed", "nfr_top100_uuids", "exit_top100_uuids")
                   and df[c].dtype.kind in "if"]
    agg: Dict = {"n_seeds": len(df)}
    for c in scalar_cols:
        vals = df[c].dropna()
        agg[f"{c}__mean"] = float(vals.mean()) if len(vals) else float("nan")
        agg[f"{c}__std"] = float(vals.std(ddof=1)) if len(vals) > 1 else float("nan")
    agg["nfr_top100_stability_jaccard"] = top_k_stability(
        df["nfr_top100_uuids"].tolist()
    )
    agg["exit_top100_stability_jaccard"] = top_k_stability(
        df["exit_top100_uuids"].tolist()
    )
    return agg


# ---------------------------------------------------------------------------
# Markdown summary
# ---------------------------------------------------------------------------

def markdown_summary(per_variant: Dict[str, Dict]) -> str:
    def fmt(v, sd=None, pct=False):
        if pd.isna(v): return "—"
        if pct: return f"{v*100:.1f}±{sd*100:.1f}" if sd is not None and not pd.isna(sd) else f"{v*100:.1f}"
        return f"{v:.3f}±{sd:.3f}" if sd is not None and not pd.isna(sd) else f"{v:.3f}"
    lines = []
    lines.append("# MTL Tradeoff Summary (SeHGNN, g4_heterophily)")
    lines.append("")
    lines.append("Variants: V1 = single-task NFR (liquidity_weight=0); V2 = single-task Exit (momentum_weight=0); V3 = joint MTL (champion config).")
    lines.append("")
    cols = [
        ("Variant", lambda v, a: VARIANT_LABEL.get(v, v)),
        ("n_seeds", lambda v, a: str(a.get("n_seeds", 0))),
        ("NFR AUC-ROC", lambda v, a: fmt(a.get("nfr_auc_roc__mean"), a.get("nfr_auc_roc__std"), pct=True)),
        ("NFR AUC-PR", lambda v, a: fmt(a.get("nfr_auc_pr__mean"), a.get("nfr_auc_pr__std"), pct=True)),
        ("NFR P@100", lambda v, a: fmt(a.get("nfr_precision_at_100__mean"), a.get("nfr_precision_at_100__std"), pct=True)),
        ("NFR Recall@100", lambda v, a: fmt(a.get("nfr_recall_at_100__mean"), a.get("nfr_recall_at_100__std"), pct=True)),
        ("NFR top100 stability", lambda v, a: fmt(a.get("nfr_top100_stability_jaccard"))),
        ("NFR top100 HHI(continent)", lambda v, a: fmt(a.get("nfr_top100_hhi_continent__mean"), a.get("nfr_top100_hhi_continent__std"))),
        ("Exit AUC-ROC", lambda v, a: fmt(a.get("exit_auc_roc__mean"), a.get("exit_auc_roc__std"), pct=True)),
        ("Exit AUC-PR", lambda v, a: fmt(a.get("exit_auc_pr__mean"), a.get("exit_auc_pr__std"), pct=True)),
        ("Exit P@100", lambda v, a: fmt(a.get("exit_precision_at_100__mean"), a.get("exit_precision_at_100__std"), pct=True)),
        ("Exit Recall@100", lambda v, a: fmt(a.get("exit_recall_at_100__mean"), a.get("exit_recall_at_100__std"), pct=True)),
        ("Exit top100 stability", lambda v, a: fmt(a.get("exit_top100_stability_jaccard"))),
        ("Joint top100 overlap", lambda v, a: fmt(a.get("joint_top100_overlap__mean"), a.get("joint_top100_overlap__std"))),
    ]
    header = "| " + " | ".join(c[0] for c in cols) + " |"
    sep = "|" + "|".join(["---"] * len(cols)) + "|"
    lines.append(header)
    lines.append(sep)
    for v in VARIANT_ORDER:
        a = per_variant.get(v, {})
        if not a: continue
        lines.append("| " + " | ".join(c[1](v, a) for c in cols) + " |")
    lines.append("")
    lines.append("**Reading notes.**")
    lines.append("- P@K is rank-based (top-K-by-positive-prediction); see paper §5 P@K convention.")
    lines.append("- Top-K stability is mean pairwise Jaccard of top-100 `org_uuid` lists across seeds; higher = more reproducible picks.")
    lines.append("- HHI(continent): 1.0 = top-100 all in one continent; 1/n = uniform. Lower = more geographic diversity.")
    lines.append("- Joint top-100 overlap is the count of startups appearing in both NFR and Exit top-100 lists per seed.")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", choices=["registry", "local"], default="local",
                   help="How to discover the per-(variant,seed) runs. 'local' "
                        "(default) scans the result JSONs under --state-dir and "
                        "classifies each run's variant from its loss weights — no "
                        "submission registry needed. 'registry' reads "
                        "experiments/registry/mtl_tradeoff_submissions.json (written "
                        "by the SLURM sweep submitter; only relevant if you used it).")
    p.add_argument("--state-dir", type=Path, default=PIPELINE_STATE,
                   help="Root scanned in --source local mode.")
    p.add_argument("--variants", nargs="+", choices=VARIANT_ORDER,
                   default=VARIANT_ORDER, help="Variants to aggregate")
    p.add_argument("--graph-version", default=None,
                   help="Graph cache hash for the attribute join. If omitted: in "
                        "--source local it is auto-detected from the runs' "
                        "run_context.json; in --source registry it falls back to "
                        "the applied-paper canonical hash (9af09bd47d2df937).")
    p.add_argument("--top-k", type=int, default=100,
                   help="K for top-K stability/diversity/overlap metrics")
    p.add_argument("--ks", nargs="+", type=int, default=[10, 50, 100, 500, 1000],
                   help="K values for P@K and Recall@K")
    p.add_argument("--output-dir", default=str(DEFAULT_OUTDIR))
    p.add_argument("--skip-attrs", action="store_true",
                   help="Skip the graph load + attribute join (faster sanity check)")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    if args.source == "local":
        print(f"Discovering MTL-variant runs locally under {args.state_dir}")
        submissions = collect_runs_local(args.state_dir, args.variants)
        print(f"  found {len(submissions)} (variant, seed) runs with predictions CSVs")
    else:
        print(f"Loading submission index from {SUBMISSIONS_PATH.resolve().relative_to(REPO_ROOT)}")
        submissions = load_submission_index()

    # Resolve the graph hash for the attribute join. Explicit --graph-version
    # always wins. Otherwise: local mode reads it from the discovered runs'
    # run_context.json (the user need not know the hash); registry mode falls
    # back to the applied-paper canonical hash.
    if args.graph_version is None:
        if args.source == "local":
            gvs = {s.get("graph_version") for s in submissions if s.get("graph_version")}
            if len(gvs) == 1:
                args.graph_version = gvs.pop()
                print(f"  auto-detected graph_version={args.graph_version} from run_context")
            elif len(gvs) > 1:
                print(f"  WARNING: runs span multiple graph_versions {sorted(gvs)}; "
                      "pass --graph-version explicitly. Skipping attribute join.")
                args.skip_attrs = True
            else:
                print("  WARNING: no graph_version in run_context; skipping attribute join.")
                args.skip_attrs = True
        else:
            args.graph_version = "9af09bd47d2df937"

    # Attribute table
    if args.skip_attrs:
        attrs = pd.DataFrame({"org_uuid": [], "is_mature": [], "continent": [], "primary_sector": []})
    else:
        graph_path = GRAPH_CACHE / f"graph_{args.graph_version}.pt"
        if not graph_path.exists():
            print(f"  WARNING: graph cache {graph_path} not found; falling back to no-attribute mode")
            attrs = pd.DataFrame({"org_uuid": [], "is_mature": [], "continent": [], "primary_sector": []})
        else:
            print(f"Loading attributes from {graph_path.resolve().relative_to(REPO_ROOT)}")
            attrs = load_attributes(graph_path)
            print(f"  attrs table: {len(attrs)} rows, mature={int(attrs['is_mature'].sum())}, "
                  f"continent_known={int(attrs['continent'].notna().sum())}, "
                  f"sector_known={int(attrs['primary_sector'].notna().sum())}")

    per_seed_rows: List[Dict] = []
    per_variant_rows: Dict[str, List[Dict]] = defaultdict(list)
    missing: List[Tuple[str, int]] = []

    for sub in submissions:
        variant = sub.get("mtl_variant")
        if variant not in args.variants:
            continue
        seed = int(sub["seed"])
        # local mode resolves the CSV directly; registry mode looks it up by job_id
        if sub.get("predictions_csv"):
            csv = Path(sub["predictions_csv"])
            job_id = sub.get("job_id", "local")
        else:
            job_id = sub.get("job_id")
            if job_id is None:
                missing.append((variant, seed)); continue
            csv = find_predictions_csv(int(job_id))
        if csv is None or not csv.exists():
            missing.append((variant, seed)); continue
        try:
            preds = parse_predictions(csv)
        except Exception as e:
            print(f"  [{variant} seed={seed}] parse failed ({csv}): {e}")
            missing.append((variant, seed)); continue
        metrics = compute_per_seed_metrics(preds, attrs, ks=args.ks, k_diversity=args.top_k)
        metrics.update({"variant": variant, "seed": seed, "job_id": job_id})
        per_seed_rows.append(metrics)
        per_variant_rows[variant].append(metrics)

    if missing:
        print(f"\n  {len(missing)} (variant, seed) pairs are missing predictions CSV:")
        for v, s in missing[:10]:
            print(f"    {v} seed={s}")
        if len(missing) > 10:
            print(f"    (and {len(missing) - 10} more)")

    if not per_seed_rows:
        print("No runs available to aggregate. Exit.")
        return 1

    # Per-seed CSV (drop list columns)
    per_seed_df = pd.DataFrame(per_seed_rows)
    drop_cols = [c for c in per_seed_df.columns if c.endswith("_uuids")]
    per_seed_df.drop(columns=drop_cols).to_csv(outdir / "per_seed.csv", index=False)
    print(f"\nWrote per-seed metrics → {(outdir / 'per_seed.csv').resolve().relative_to(REPO_ROOT)} "
          f"({len(per_seed_df)} rows)")

    # Per-variant summary
    summary = {v: aggregate_variant(rows) for v, rows in per_variant_rows.items()}
    summary_df = pd.DataFrame(summary).T.reset_index().rename(columns={"index": "variant"})
    summary_df.to_csv(outdir / "per_variant_summary.csv", index=False)
    print(f"Wrote per-variant summary → {(outdir / 'per_variant_summary.csv').resolve().relative_to(REPO_ROOT)}")

    # Top-K uuid lists (for reproducing any downstream analysis)
    topk_dump = {
        v: {str(r["seed"]): {"nfr": r["nfr_top100_uuids"], "exit": r["exit_top100_uuids"]}
            for r in rows}
        for v, rows in per_variant_rows.items()
    }
    with open(outdir / "top_k_lists.json", "w") as f:
        json.dump(topk_dump, f, indent=2)
    print(f"Wrote top-K uuid dump → {(outdir / 'top_k_lists.json').resolve().relative_to(REPO_ROOT)}")

    # Markdown summary
    md_path = outdir / "SUMMARY.md"
    md_path.write_text(markdown_summary(summary))
    print(f"Wrote markdown summary → {md_path.resolve().relative_to(REPO_ROOT)}")
    print("\n" + markdown_summary(summary))
    return 0


if __name__ == "__main__":
    sys.exit(main())
