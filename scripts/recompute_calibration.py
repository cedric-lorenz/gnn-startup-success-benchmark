"""Recompute calibration metrics (uncalibrated / Platt / isotonic) for the
applied-paper canonical V3 (joint) SeHGNN runs, without retraining.

Per seed we load best_model.pt, run inference once on the full graph, then
fit Platt and isotonic on validation predictions and apply to test. ECE is
10-bin equal-width to match the paper's stated convention; Brier is the
standard mean-squared error. Exit metrics use the mature subset (paper
maturity mask).

Inputs:
  - experiments/registry/mtl_tradeoff_submissions.json (variant -> run_id)
  - experiments/champion_configs/sehgnn_g4_heterophily.yaml (architecture)
  - outputs/checkpoints/gnn-startup-successs/<run_id>/best_model.pt (per seed)

Outputs (default outputs/calibration_refresh/):
  - per_seed.csv (one row per seed, ECE/Brier per method per task)
  - summary.csv  (20-seed mean +/- std for the paper table)
  - SUMMARY.md   (LaTeX-paste-ready table)

Usage:
  python scripts/recompute_calibration.py
  python scripts/recompute_calibration.py --variant joint --seeds 0 1 2
  python scripts/recompute_calibration.py --n-bins 15      # alt bin count
"""
from __future__ import annotations
import argparse
import glob
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

SUBMISSIONS_PATH = REPO_ROOT / "experiments" / "registry" / "mtl_tradeoff_submissions.json"
CKPT_BASE = REPO_ROOT / "outputs" / "checkpoints" / "gnn-startup-successs"
DEFAULT_OUTDIR = REPO_ROOT / "outputs" / "calibration_refresh"

# Hardcoded label slice in startup.y for masked_multi_task (NFR / Exit).
# These match preprocessing.py's column convention. We verify by name when
# loading config (the column names live in data_processing.binary_column).
NFR_LABEL_IDX = 0   # momentum / Next Funding Round
EXIT_LABEL_IDX = 1  # liquidity / Exit


# ---------------------------------------------------------------------------
# Metric primitives
# ---------------------------------------------------------------------------

def ece_equal_width(p: np.ndarray, y: np.ndarray, n_bins: int) -> float:
    """Expected Calibration Error, equal-width bins, weighted by bin support."""
    edges = np.linspace(0, 1, n_bins + 1)
    out = 0.0
    n = len(p)
    if n == 0:
        return float("nan")
    for lo, hi in zip(edges[:-1], edges[1:]):
        in_bin = (p > lo) & (p <= hi)
        if in_bin.sum() > 0:
            out += abs(p[in_bin].mean() - y[in_bin].mean()) * (in_bin.sum() / n)
    return float(out)


def brier(p: np.ndarray, y: np.ndarray) -> float:
    if len(p) == 0:
        return float("nan")
    return float(np.mean((p - y) ** 2))


# ---------------------------------------------------------------------------
# Per-run inference
# ---------------------------------------------------------------------------

def load_config(champion_yaml: Path) -> Dict:
    from src.ml.utils import load_config as utils_load_config
    # The pipeline uses load_config(path) which merges champion into the base config.yaml.
    # We need the merged config because the slim champion only carries non-default fields.
    base_cfg = utils_load_config(str(REPO_ROOT / "config.yaml"))
    with open(champion_yaml) as f:
        champ = yaml.safe_load(f)

    def deep_merge(base, override):
        out = dict(base)
        for k, v in override.items():
            if isinstance(v, dict) and isinstance(out.get(k), dict):
                out[k] = deep_merge(out[k], v)
            else:
                out[k] = v
        return out

    cfg = deep_merge(base_cfg, champ)
    cfg["wandb"]["enabled"] = False
    cfg["analysis"] = {
        "enable_downstream_analysis": False,
        "enable_homophily_analysis": False,
        "enable_visualization_analysis": False,
    }
    cfg["explain"]["enabled"] = False
    return cfg


def load_graph_from_cache(graph_version: str):
    """Load the exact graph the original run used.

    Each (seed, variant) run logs its graph_version hash to run_context.json.
    We bypass preprocessing entirely and load the cached HeteroData directly,
    which is what create_graph() produced and what the trained model expects.
    """
    cache_path = REPO_ROOT / "outputs" / "graphs" / f"graph_{graph_version}.pt"
    if not cache_path.exists():
        raise FileNotFoundError(f"graph cache not on disk: {cache_path}")
    blob = torch.load(cache_path, map_location="cpu", weights_only=False)
    # The cache file stores a (HeteroData, metadata_dict) tuple.
    if isinstance(blob, tuple) and len(blob) == 2:
        graph_data, _meta = blob
    else:
        graph_data = blob
    return graph_data


def find_run_context(job_id, job_dir: Optional[Path] = None) -> Optional[Path]:
    """Locate the run_context JSON written by main.py for this run.

    In local mode the exact job directory is known and passed directly (the
    SLURM job_id prefix is ambiguous across an array); in registry mode we glob
    by the job_id prefix.
    """
    if job_dir is not None:
        ctx = job_dir / "results" / "run_context.json"
        return ctx if ctx.exists() else None
    matches = glob.glob(str(REPO_ROOT / "outputs" / "pipeline_state" /
                            f"{job_id}_*" / "results" / "run_context.json"))
    return Path(matches[-1]) if matches else None


def graph_version_for_run(job_id, job_dir: Optional[Path] = None) -> Optional[str]:
    ctx_path = find_run_context(job_id, job_dir)
    if ctx_path is None:
        return None
    with open(ctx_path) as f:
        # run_context.json is the run-context block from the final test JSON;
        # depending on layout it may be at top level or nested.
        data = json.load(f)
    # Try a couple of keys
    gv = data.get("graph_version")
    if not gv:
        gv = data.get("_run_context", {}).get("graph_version")
    return gv


def load_model(cfg: Dict, graph_data, ckpt_path: Path, device: torch.device):
    """Build the SeHGNN model with the champion HPs, then load best_model.pt."""
    from src.ml.train import Trainer
    cfg_local = dict(cfg)
    cfg_local["train"] = dict(cfg["train"])
    cfg_local["train"]["device"] = str(device)

    trainer = Trainer(graph_data, cfg_local)
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    # best_model.pt is a training-state dict {epoch, model_state_dict, optimizer_state_dict, ...}.
    # Unwrap to get just the model weights.
    if isinstance(state, dict):
        for key in ("model_state_dict", "state_dict", "model"):
            if key in state and isinstance(state[key], dict):
                state = state[key]
                break
    missing, unexpected = trainer.model.load_state_dict(state, strict=False)
    if missing:
        print(f"    [warn] {len(missing)} missing keys in checkpoint, e.g. {missing[:3]}")
        if len(missing) > 20:
            raise RuntimeError(f"checkpoint load failed: {len(missing)} keys missing — wrong format?")
    if unexpected:
        print(f"    [warn] {len(unexpected)} unexpected keys in checkpoint, e.g. {unexpected[:3]}")
    trainer.model.eval()
    return trainer.model


def _forward_with_split_features(model, graph_data, split_key: str,
                                 eval_mask) -> Tuple[np.ndarray, np.ndarray]:
    """Swap in split-specific imputed features, forward pass, swap back.

    Mirrors eval.py:818-833 — split-specific features are required because
    NaN-imputed startups would otherwise leak train-set statistics into test.
    """
    startup_store = graph_data["startup"]
    original_x = startup_store.x
    if hasattr(startup_store, split_key):
        startup_store.x = getattr(startup_store, split_key)
    try:
        with torch.no_grad():
            try:
                out = model(graph_data.x_dict, graph_data.edge_index_dict,
                            batch=graph_data, eval_mask=eval_mask)
            except TypeError:
                out = model(graph_data.x_dict, graph_data.edge_index_dict)
    finally:
        startup_store.x = original_x
    p_mom = torch.sigmoid(out["out_mom"]).detach().cpu().numpy().reshape(-1)
    p_liq = torch.sigmoid(out["out_liq"]).detach().cpu().numpy().reshape(-1)
    return p_mom, p_liq


def run_inference(model, graph_data, device: torch.device
                  ) -> Dict[str, np.ndarray]:
    """Two forward passes with the appropriate split-specific features each time."""
    graph_data = graph_data.to(device)
    val_mask = graph_data["startup"].val_mask
    test_mask = graph_data["startup"].test_mask
    p_mom_val, p_liq_val = _forward_with_split_features(
        model, graph_data, "x_val_mask", val_mask)
    p_mom_test, p_liq_test = _forward_with_split_features(
        model, graph_data, "x_test_mask", test_mask)
    return {
        "p_mom_val": p_mom_val, "p_liq_val": p_liq_val,
        "p_mom_test": p_mom_test, "p_liq_test": p_liq_test,
    }


def get_labels_and_masks(graph_data, cfg: Dict):
    """Extract gold labels + train/val/test masks + Exit maturity mask (if exposed)."""
    s = graph_data["startup"]
    y = s.y.detach().cpu().numpy()
    y_mom = y[:, NFR_LABEL_IDX].astype(np.int64)
    y_liq = y[:, EXIT_LABEL_IDX].astype(np.int64)
    val_mask = s.val_mask.detach().cpu().numpy().astype(bool)
    test_mask = s.test_mask.detach().cpu().numpy().astype(bool)

    # The Exit head trains/evaluates only on mature startups. The graph store
    # may expose this directly; otherwise compute from raw_df via utils.
    is_mature = None
    for attr in ("m_liq", "mature_mask", "is_mature"):
        if hasattr(s, attr):
            val = getattr(s, attr)
            if hasattr(val, "detach"):
                is_mature = val.detach().cpu().numpy().astype(bool)
            else:
                is_mature = np.asarray(val).astype(bool)
            break
    if is_mature is None:
        from src.ml.utils import get_maturity_mask
        gating = cfg.get("data_processing", {}).get("strict_gating", {})
        if not gating.get("enabled", False):
            cfg["data_processing"]["strict_gating"] = {
                "enabled": True,
                "late_stage_funding_threshold": 15_000_000,
                "employee_count_threshold": 3,
                "compounder_age_threshold": 5,
                "compounder_funding_threshold": 3_000_000,
            }
        raw = s.raw_df
        m = get_maturity_mask(raw, cfg)
        is_mature = (m.astype(bool).values if m is not None
                     else np.ones(len(raw), dtype=bool))
    return y_mom, y_liq, val_mask, test_mask, is_mature


# ---------------------------------------------------------------------------
# Calibration: fit on val, apply to test
# ---------------------------------------------------------------------------

def fit_apply(p_val: np.ndarray, y_val: np.ndarray,
              p_test: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Return (platt_test_probs, isotonic_test_probs)."""
    eps = 1e-7
    p_val_c = np.clip(p_val, eps, 1 - eps)
    p_test_c = np.clip(p_test, eps, 1 - eps)
    platt = LogisticRegression(C=1e6, solver="lbfgs", max_iter=1000)
    platt.fit(p_val_c.reshape(-1, 1), y_val)
    p_test_platt = platt.predict_proba(p_test_c.reshape(-1, 1))[:, 1]

    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    iso.fit(p_val_c, y_val)
    p_test_iso = iso.predict(p_test_c)
    return p_test_platt, p_test_iso


# ---------------------------------------------------------------------------
# Main per-run driver
# ---------------------------------------------------------------------------

def _resolve_checkpoint(run_id: str) -> Optional[Path]:
    """Find best_model.pt under the W&B-named checkpoint root, falling back to
    the WANDB-disabled 'dummy' root that synthetic/offline runs write to."""
    for root in (CKPT_BASE, CKPT_BASE.parent / "dummy"):
        ckpt = root / run_id / "best_model.pt"
        if ckpt.exists():
            return ckpt
    return None


def compute_for_run(seed: int, run_id: str, job_id, champion_yaml: Path,
                    cfg_cache: Dict, graph_cache: Dict,
                    n_bins: int, device: torch.device,
                    job_dir: Optional[Path] = None) -> Dict:
    print(f"\n=== seed={seed}  run_id={run_id}  job_id={job_id} ===")
    if cfg_cache.get("_cfg") is None:
        cfg_cache["_cfg"] = load_config(champion_yaml)
    cfg = cfg_cache["_cfg"]

    gv = graph_version_for_run(job_id, job_dir)
    if gv is None:
        print(f"  [skip] no graph_version found in run_context for job {job_id}")
        return {"seed": seed, "run_id": run_id, "_skip": True}
    if graph_cache.get(gv) is None:
        print(f"  loading cached graph {gv}")
        graph_cache[gv] = load_graph_from_cache(gv)
    graph_data = graph_cache[gv]

    ckpt_path = _resolve_checkpoint(run_id)
    if ckpt_path is None:
        print(f"  [skip] no checkpoint for run_id {run_id} under "
              f"{CKPT_BASE} or {CKPT_BASE.parent / 'dummy'}")
        return {"seed": seed, "run_id": run_id, "_skip": True}

    model = load_model(cfg, graph_data, ckpt_path, device)
    preds = run_inference(model, graph_data, device)
    y_mom, y_liq, val_mask, test_mask, is_mature = get_labels_and_masks(graph_data, cfg)

    # NFR: full population for both val and test
    val_idx_nfr = val_mask
    test_idx_nfr = test_mask
    p_val_mom = preds["p_mom_val"][val_idx_nfr]
    p_test_mom = preds["p_mom_test"][test_idx_nfr]
    y_val_mom = y_mom[val_idx_nfr]
    y_test_mom = y_mom[test_idx_nfr]
    pmom_test_platt, pmom_test_iso = fit_apply(p_val_mom, y_val_mom, p_test_mom)

    # Exit: restrict to mature subset (val and test)
    val_idx_exit = val_mask & is_mature
    test_idx_exit = test_mask & is_mature
    p_val_liq = preds["p_liq_val"][val_idx_exit]
    p_test_liq = preds["p_liq_test"][test_idx_exit]
    y_val_liq = y_liq[val_idx_exit]
    y_test_liq = y_liq[test_idx_exit]
    pliq_test_platt, pliq_test_iso = fit_apply(p_val_liq, y_val_liq, p_test_liq)

    # AUC-PR sanity: should match the test JSON within ~0.01 if the model
    # reproduces exactly. A larger drift means inference is on a slightly
    # different model state — calibration ratios are still meaningful but
    # absolute numbers should be flagged.
    nfr_aucpr = float(average_precision_score(y_test_mom, p_test_mom)) if y_test_mom.sum() > 0 else float("nan")
    exit_aucpr = float(average_precision_score(y_test_liq, p_test_liq)) if y_test_liq.sum() > 0 else float("nan")
    ref_nfr = ref_exit = float("nan")
    if job_dir is not None:
        test_json = glob.glob(str(job_dir / "results" / "SeHGNN" /
                                  "masked_multi_task" / "*_test.json"))
    else:
        test_json = glob.glob(str(REPO_ROOT / "outputs" / "pipeline_state" /
                                  f"{job_id}_*" / "results" / "SeHGNN" /
                                  "masked_multi_task" / "*_test.json"))
    if test_json:
        try:
            tj = json.load(open(test_json[-1]))
            ref_nfr = tj["metrics"].get("test_auc_pr_mom", float("nan"))
            ref_exit = tj["metrics"].get("test_auc_pr_liq", float("nan"))
        except Exception:
            pass
    drift_nfr = nfr_aucpr - ref_nfr if not np.isnan(ref_nfr) else float("nan")
    drift_exit = exit_aucpr - ref_exit if not np.isnan(ref_exit) else float("nan")
    flag_nfr = "⚠️" if abs(drift_nfr) > 0.01 else "✓"
    flag_exit = "⚠️" if abs(drift_exit) > 0.01 else "✓"
    print(f"  sanity: inference NFR AUC-PR={nfr_aucpr:.4f} vs test JSON {ref_nfr:.4f}  "
          f"(drift={drift_nfr:+.4f} {flag_nfr})")
    print(f"          inference Exit AUC-PR={exit_aucpr:.4f} vs test JSON {ref_exit:.4f}  "
          f"(drift={drift_exit:+.4f} {flag_exit})")

    def _aucpr(y, p):
        return float(average_precision_score(y, p)) if y.sum() > 0 else float("nan")

    out = {
        "seed": seed, "run_id": run_id,
        "n_test_nfr": int(test_idx_nfr.sum()),
        "n_test_exit_mature": int(test_idx_exit.sum()),
        "nfr_auc_pr_test_json": ref_nfr,
        "exit_auc_pr_test_json": ref_exit,
        "nfr_auc_pr_drift": drift_nfr,
        "exit_auc_pr_drift": drift_exit,
        # Uncalibrated
        "nfr_ece_uncal": ece_equal_width(p_test_mom, y_test_mom, n_bins),
        "nfr_brier_uncal": brier(p_test_mom, y_test_mom),
        "nfr_auc_pr_uncal": _aucpr(y_test_mom, p_test_mom),
        "exit_ece_uncal": ece_equal_width(p_test_liq, y_test_liq, n_bins),
        "exit_brier_uncal": brier(p_test_liq, y_test_liq),
        "exit_auc_pr_uncal": _aucpr(y_test_liq, p_test_liq),
        # Platt
        "nfr_ece_platt": ece_equal_width(pmom_test_platt, y_test_mom, n_bins),
        "nfr_brier_platt": brier(pmom_test_platt, y_test_mom),
        "nfr_auc_pr_platt": _aucpr(y_test_mom, pmom_test_platt),
        "exit_ece_platt": ece_equal_width(pliq_test_platt, y_test_liq, n_bins),
        "exit_brier_platt": brier(pliq_test_platt, y_test_liq),
        "exit_auc_pr_platt": _aucpr(y_test_liq, pliq_test_platt),
        # Isotonic
        "nfr_ece_iso": ece_equal_width(pmom_test_iso, y_test_mom, n_bins),
        "nfr_brier_iso": brier(pmom_test_iso, y_test_mom),
        "nfr_auc_pr_iso": _aucpr(y_test_mom, pmom_test_iso),
        "exit_ece_iso": ece_equal_width(pliq_test_iso, y_test_liq, n_bins),
        "exit_brier_iso": brier(pliq_test_iso, y_test_liq),
        "exit_auc_pr_iso": _aucpr(y_test_liq, pliq_test_iso),
    }
    print(f"  NFR  ECE   : uncal={out['nfr_ece_uncal']:.4f}  platt={out['nfr_ece_platt']:.4f}  iso={out['nfr_ece_iso']:.4f}")
    print(f"  NFR  Brier : uncal={out['nfr_brier_uncal']:.4f}  platt={out['nfr_brier_platt']:.4f}  iso={out['nfr_brier_iso']:.4f}")
    print(f"  NFR  AUC-PR: uncal={out['nfr_auc_pr_uncal']:.4f}  platt={out['nfr_auc_pr_platt']:.4f}  iso={out['nfr_auc_pr_iso']:.4f}")
    print(f"  Exit ECE   : uncal={out['exit_ece_uncal']:.4f}  platt={out['exit_ece_platt']:.4f}  iso={out['exit_ece_iso']:.4f}")
    print(f"  Exit Brier : uncal={out['exit_brier_uncal']:.4f}  platt={out['exit_brier_platt']:.4f}  iso={out['exit_brier_iso']:.4f}")
    print(f"  Exit AUC-PR: uncal={out['exit_auc_pr_uncal']:.4f}  platt={out['exit_auc_pr_platt']:.4f}  iso={out['exit_auc_pr_iso']:.4f}")
    return out


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def aggregate_rows(rows: List[Dict]) -> pd.DataFrame:
    rows = [r for r in rows if not r.get("_skip")]
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    metric_cols = [c for c in df.columns if c.startswith(("nfr_", "exit_"))]
    summary = {}
    for c in metric_cols:
        v = df[c].dropna()
        summary[f"{c}__mean"] = float(v.mean()) if len(v) else float("nan")
        summary[f"{c}__std"] = float(v.std(ddof=1)) if len(v) > 1 else float("nan")
    summary["n_seeds"] = len(df)
    return pd.DataFrame([summary])


def markdown_table(summary: pd.Series) -> str:
    def fmt(mu, sd):
        if pd.isna(mu): return "—"
        if pd.isna(sd): return f"{mu:.3f}"
        return f"{mu:.3f} \\pm {sd:.3f}"
    rows = [
        ("Uncalibrated", "uncal"),
        ("Platt scaling", "platt"),
        ("Isotonic regression", "iso"),
    ]
    out = ["# Calibration table (applied-rerun, 10-bin ECE)",
           f"n_seeds = {int(summary['n_seeds'])}",
           "",
           "| Method | NFR ECE | NFR Brier | NFR AUC-PR | Exit ECE | Exit Brier | Exit AUC-PR |",
           "|---|---|---|---|---|---|---|"]
    for label, suf in rows:
        ece_n = fmt(summary[f"nfr_ece_{suf}__mean"], summary[f"nfr_ece_{suf}__std"])
        brier_n = fmt(summary[f"nfr_brier_{suf}__mean"], summary[f"nfr_brier_{suf}__std"])
        aucpr_n = fmt(summary[f"nfr_auc_pr_{suf}__mean"], summary[f"nfr_auc_pr_{suf}__std"])
        ece_e = fmt(summary[f"exit_ece_{suf}__mean"], summary[f"exit_ece_{suf}__std"])
        brier_e = fmt(summary[f"exit_brier_{suf}__mean"], summary[f"exit_brier_{suf}__std"])
        aucpr_e = fmt(summary[f"exit_auc_pr_{suf}__mean"], summary[f"exit_auc_pr_{suf}__std"])
        out.append(f"| {label} | {ece_n} | {brier_n} | {aucpr_n} | {ece_e} | {brier_e} | {aucpr_e} |")
    out.append("")
    out.append("**Reading notes.**")
    out.append("- ECE: 10-bin equal-width, weighted by bin support (paper convention).")
    out.append("- Brier: standard mean-squared error of probabilities vs labels.")
    out.append("- AUC-PR included to back the paper's claim that post-hoc recalibration is rank-preserving; any deviation across methods should be ≤0.001 (numerical noise from the monotonic transform).")
    out.append("- Exit metrics are restricted to the mature subset (paper protocol).")
    return "\n".join(out)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def collect_runs(variant: str, seeds: Optional[List[int]]) -> List[Tuple[int, str, int]]:
    """Return [(seed, run_id, job_id), ...] for the variant from submissions registry."""
    if not SUBMISSIONS_PATH.exists():
        raise SystemExit(f"missing {SUBMISSIONS_PATH}")
    with open(SUBMISSIONS_PATH) as f:
        subs = json.load(f)
    items = []
    seen = set()
    for batch_key, jobs in subs.items():
        for j in jobs:
            if j.get("mtl_variant") != variant:
                continue
            seed = int(j["seed"])
            if seeds is not None and seed not in seeds:
                continue
            if seed in seen:
                continue
            jid = j.get("job_id")
            if jid is None:
                continue
            csvs = glob.glob(str(REPO_ROOT / "outputs" / "pipeline_state" /
                                 f"{jid}_*" / "predictions" / "SeHGNN" /
                                 "masked_multi_task" / "*_predictions_test.csv"))
            if not csvs:
                continue
            run_id = Path(csvs[-1]).name.split("_")[-3]
            items.append((seed, run_id, int(jid)))
            seen.add(seed)
    items.sort()
    return items


def _variant_from_loss(loss: Dict) -> Optional[str]:
    """Classify a run's MTL variant from its train.loss weights (same rule as
    aggregate_mtl_tradeoff): liquidity_weight==0 -> stl_nfr; momentum_weight==0
    -> stl_exit; equal positive weights -> balanced; else -> joint."""
    mw, lw = loss.get("momentum_weight"), loss.get("liquidity_weight")
    if mw is None or lw is None:
        return None
    if lw == 0 and mw > 0:
        return "stl_nfr"
    if mw == 0 and lw > 0:
        return "stl_exit"
    if mw > 0 and lw > 0:
        return "balanced" if abs(mw - lw) < 1e-9 else "joint"
    return None


def collect_runs_local(variant: str, seeds: Optional[List[int]],
                       state_dir: Path) -> List[Tuple[int, str, Path]]:
    """Registry-free discovery: scan SeHGNN result JSONs under state_dir, keep
    those whose loss weights classify to `variant`, and return (seed, run_id,
    job_dir) deduped to the latest per seed. job_dir is the exact pipeline_state
    run directory (the SLURM job_id prefix is ambiguous across an array)."""
    pattern = str(state_dir / "**" / "results" / "SeHGNN" / "masked_multi_task" /
                  "*_test.json")
    best: Dict[int, Tuple[str, str, Path]] = {}
    for path in glob.glob(pattern, recursive=True):
        try:
            d = json.load(open(path))
        except Exception:
            continue
        cfg = d.get("config", {})
        seed = cfg.get("seed")
        rid = d.get("metadata", {}).get("wandb_run_id")
        if seed is None or not rid:
            continue
        if _variant_from_loss(cfg.get("train", {}).get("loss", {})) != variant:
            continue
        if seeds is not None and int(seed) not in seeds:
            continue
        job_dir = Path(path).parents[3]
        ts = d.get("metadata", {}).get("timestamp", "")
        if int(seed) not in best or ts > best[int(seed)][0]:
            best[int(seed)] = (ts, rid, job_dir)
    triples = [(s, rid, jd) for s, (_ts, rid, jd) in best.items()]
    triples.sort()
    return triples


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", choices=["registry", "local"], default="local",
                   help="How to discover the variant's runs. 'local' (default) "
                        "scans result JSONs under --state-dir and classifies the "
                        "variant from loss weights — no registry needed. 'registry' "
                        "reads experiments/registry/mtl_tradeoff_submissions.json.")
    p.add_argument("--state-dir", type=Path,
                   default=REPO_ROOT / "outputs" / "pipeline_state",
                   help="Root scanned in --source local mode.")
    p.add_argument("--variant", default="joint",
                   choices=["stl_nfr", "stl_exit", "joint", "balanced"],
                   help="Which MTL-tradeoff variant's checkpoints to use")
    p.add_argument("--seeds", nargs="+", type=int, default=None,
                   help="Subset of seeds (default: all available)")
    p.add_argument("--n-bins", type=int, default=10,
                   help="Bin count for ECE (paper convention = 10)")
    p.add_argument("--champion-config",
                   default=str(REPO_ROOT / "experiments" / "champion_configs" /
                               "sehgnn_g4_heterophily.yaml"),
                   help="Path to the architecture config for instantiation")
    p.add_argument("--output-dir", default=str(DEFAULT_OUTDIR))
    p.add_argument("--device", default=None, help="Override torch device (default: cuda if available)")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device) if args.device else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Variant: {args.variant}  |  n_bins: {args.n_bins}")

    if args.source == "local":
        local_runs = collect_runs_local(args.variant, args.seeds, args.state_dir)
        # normalize to (seed, run_id, job_id=None, job_dir)
        runs = [(s, rid, None, jd) for s, rid, jd in local_runs]
    else:
        runs = [(s, rid, jid, None) for s, rid, jid in
                collect_runs(args.variant, args.seeds)]
    if not runs:
        print("No runs found.")
        return 1
    print(f"Found {len(runs)} (seed, run_id) tuples to score.")

    champion = Path(args.champion_config)
    cfg_cache: Dict = {}
    graph_cache: Dict = {}
    rows: List[Dict] = []

    for seed, run_id, job_id, job_dir in runs:
        try:
            row = compute_for_run(seed, run_id, job_id, champion, cfg_cache, graph_cache,
                                  n_bins=args.n_bins, device=device, job_dir=job_dir)
            rows.append(row)
        except Exception as e:
            print(f"  [error] seed={seed} run_id={run_id}: {e!r}")
            rows.append({"seed": seed, "run_id": run_id, "_skip": True, "_error": repr(e)})

    df = pd.DataFrame([r for r in rows if not r.get("_skip")])
    if df.empty:
        print("No successful runs to aggregate.")
        return 1

    per_seed_path = outdir / "per_seed.csv"
    df.to_csv(per_seed_path, index=False)
    print(f"\nWrote {per_seed_path.resolve().relative_to(REPO_ROOT)}")

    summary_df = aggregate_rows(rows)
    summary_df.to_csv(outdir / "summary.csv", index=False)
    print(f"Wrote {(outdir / 'summary.csv').resolve().relative_to(REPO_ROOT)}")

    md = markdown_table(summary_df.iloc[0])
    (outdir / "SUMMARY.md").write_text(md)
    print(f"Wrote {(outdir / 'SUMMARY.md').resolve().relative_to(REPO_ROOT)}")
    print("\n" + md)

    # LaTeX macros include — overwrites graph-paper/sections/calibration_numbers.tex
    # so the paper auto-picks up new values.
    if args.variant == "joint":
        tex_path = REPO_ROOT / "graph-paper" / "sections" / "calibration_numbers.tex"
        if tex_path.parent.exists():
            write_latex_macros(summary_df.iloc[0], tex_path)
            print(f"Wrote {tex_path.resolve().relative_to(REPO_ROOT)}")
        else:
            print(f"  (skipping LaTeX macros: {tex_path.parent} absent — "
                  "summary.csv/SUMMARY.md hold the numbers)")
    return 0


def write_latex_macros(summary: pd.Series, tex_path: Path) -> None:
    """Emit the paper's calibration_numbers.tex from the aggregated summary.

    Each macro is a math-mode cell content: '0.236' for n=1 (no std),
    '0.236 \\pm 0.012' for n>=2. The table in 05_experiments.tex wraps
    these in $...$.
    """
    n = int(summary["n_seeds"])

    def cell(metric: str, n: int) -> str:
        mu = summary[f"{metric}__mean"]
        sd = summary[f"{metric}__std"]
        if pd.isna(mu):
            return "\\text{n/a}"
        if n >= 2 and not pd.isna(sd):
            return f"{mu:.3f}\\,{{\\scriptstyle\\pm\\,{sd:.3f}}}"
        return f"{mu:.3f}"

    lines = [
        "%% AUTO-GENERATED by scripts/recompute_calibration.py",
        "%% Do not edit by hand. To refresh, run:",
        "%%   python scripts/recompute_calibration.py --variant joint",
        "%% ECE is 10-bin equal-width; Exit metrics use the mature subset.",
        "",
        f"\\providecommand{{\\calNseeds}}{{{n}}}",
        "",
        "%% NFR (full test population)",
    ]
    for tag in ("ECEuncal", "Brieruncal", "AUCPRuncal",
                "ECEplatt", "Brierplatt", "AUCPRplatt",
                "ECEiso", "Brieriso", "AUCPRiso"):
        metric = {
            "ECEuncal": "nfr_ece_uncal", "Brieruncal": "nfr_brier_uncal",
            "AUCPRuncal": "nfr_auc_pr_uncal",
            "ECEplatt": "nfr_ece_platt", "Brierplatt": "nfr_brier_platt",
            "AUCPRplatt": "nfr_auc_pr_platt",
            "ECEiso": "nfr_ece_iso", "Brieriso": "nfr_brier_iso",
            "AUCPRiso": "nfr_auc_pr_iso",
        }[tag]
        lines.append(f"\\providecommand{{\\calNFR{tag}}}{{{cell(metric, n)}}}")
    lines.append("")
    lines.append("%% Exit (mature subset only)")
    for tag in ("ECEuncal", "Brieruncal", "AUCPRuncal",
                "ECEplatt", "Brierplatt", "AUCPRplatt",
                "ECEiso", "Brieriso", "AUCPRiso"):
        metric = {
            "ECEuncal": "exit_ece_uncal", "Brieruncal": "exit_brier_uncal",
            "AUCPRuncal": "exit_auc_pr_uncal",
            "ECEplatt": "exit_ece_platt", "Brierplatt": "exit_brier_platt",
            "AUCPRplatt": "exit_auc_pr_platt",
            "ECEiso": "exit_ece_iso", "Brieriso": "exit_brier_iso",
            "AUCPRiso": "exit_auc_pr_iso",
        }[tag]
        lines.append(f"\\providecommand{{\\calExit{tag}}}{{{cell(metric, n)}}}")
    tex_path.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    sys.exit(main())
