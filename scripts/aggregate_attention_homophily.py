"""Aggregate per-metapath SeHGNN attention across N seeds, with both Pearson
and Spearman correlations against per-metapath edge homophily.

For one (arch, variant) cell — defaulting to ``replicate_SeHGNN__g1_full`` —
loads every ``finished+ok`` checkpoint, builds the model **once**, swaps in each
seed's state dict, runs a single forward pass per seed to extract the 32-d
attention vector, then:

  - Reports per-seed Pearson and Spearman \\rho vs each homophily variant
    (NFR-full, Exit-full, Exit-mature) → mean ± std across seeds.
  - Builds a consensus 32-d attention vector (mean across seeds) and reports the
    consensus \\rho values that the paper figure uses.
  - Writes a tidy CSV (one row per metapath × {mean_attention, std_attention,
    h_*, n_edges_*}).
  - Re-renders the defense-style 2-panel figure with the consensus attention.

Outputs (under ``--output-dir``):
  - per_seed_attention.npy  (n_seeds × n_metapaths)
  - per_seed_correlations.csv  (one row per (seed, variant, method))
  - consensus_attention_homophily.csv  (one row per metapath)
  - correlation_summary.json
  - homophily_vs_attention_defense_style.{pdf,png}

Reproducibility:
  python scripts/aggregate_attention_homophily.py \
    --champion-config experiments/champion_configs/sehgnn_g1_full.yaml \
    --graph outputs/graphs/graph_293036175ef8e7b4.pt \
    --group replicate_SeHGNN__g1_full \
    --output-dir outputs/attention_homophily_g1_full_multiseed
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.ml.utils import load_config  # noqa: E402

from scripts.compute_attention_homophily_correlation import (  # noqa: E402
    _deep_merge,
    _load_graph_any,
    assemble_dataframe,
    build_label_variants,
    compute_homophily_for_labels,
    extract_attention,
    pearson_ignoring_nans,
    plot_defense_style_two_panel,
    spearman_ignoring_nans,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default=str(PROJECT_ROOT / "config.yaml"))
    p.add_argument("--champion-config", required=True,
                   help="Champion YAML deep-merged into --config.")
    p.add_argument("--graph", required=True,
                   help="Cached graph path (HeteroData or (HeteroData, names) tuple).")
    p.add_argument("--source", choices=["wandb", "local"], default="wandb",
                   help="How to discover the per-seed SeHGNN checkpoints. "
                        "'wandb' queries the W&B group; 'local' scans the result "
                        "JSONs under --state-dir (no W&B account needed).")
    p.add_argument("--group", default="replicate_SeHGNN__g1_full",
                   help="W&B group whose ok runs supply the per-seed checkpoints "
                        "(--source wandb).")
    p.add_argument("--project",
                   default=os.environ.get("WANDB_PROJECT", "<entity>/<project>"),
                   help="W&B entity/project for --source wandb (default: "
                        "$WANDB_PROJECT, else a placeholder).")
    p.add_argument("--state-dir", default=str(PROJECT_ROOT / "outputs" / "pipeline_state"),
                   help="Root scanned for SeHGNN result JSONs in --source local.")
    p.add_argument("--checkpoint-root",
                   default=str(PROJECT_ROOT / "outputs" / "checkpoints" / "gnn-startup-successs"),
                   help="Directory holding <run_id>/best_model.pt entries.")
    p.add_argument("--output-dir",
                   default=str(PROJECT_ROOT / "outputs" / "attention_homophily_multiseed"),
                   help="Where to write CSV/JSON/figure artifacts.")
    p.add_argument("--min-edges", type=int, default=50,
                   help="Drop meta-paths with fewer than this many edges on a "
                        "given variant's labeled subset before computing the "
                        "correlation. Matches the same filter applied inside "
                        "the figure-rendering function so the reported numbers "
                        "and the visible scatter agree. Pass 0 to disable.")
    return p.parse_args()


def discover_checkpoints(group: str, checkpoint_root: Path, project: str
                         ) -> List[Tuple[int, str, Path]]:
    """Return [(seed, run_id, checkpoint_path), ...] for every ``finished+ok`` run
    in ``group`` whose checkpoint is present on disk."""
    import wandb
    api = wandb.Api()
    runs = list(api.runs(project,
                         filters={"group": group, "state": "finished"},
                         per_page=200))
    found: List[Tuple[int, str, Path]] = []
    for r in runs:
        if r.summary.get("repro/status") != "ok":
            continue
        seed = r.config.get("seed")
        if not isinstance(seed, int):
            continue
        ckpt = checkpoint_root / r.id / "best_model.pt"
        if ckpt.is_file():
            found.append((seed, r.id, ckpt))
    found.sort(key=lambda t: t[0])
    return found


def discover_checkpoints_local(state_dir: Path, checkpoint_root: Path
                               ) -> List[Tuple[int, str, Path]]:
    """Local equivalent of discover_checkpoints: scan SeHGNN result JSONs under
    `state_dir` for (seed, run_id), and locate each run's checkpoint without
    W&B. Checkpoints may live at <checkpoint_root>/<run_id>/best_model.pt or the
    WANDB-disabled fallback outputs/checkpoints/dummy/<run_id>/best_model.pt.
    Dedup to the latest run per seed by timestamp."""
    import glob, json
    ckpt_dummy = checkpoint_root.parent / "dummy"
    best: Dict[int, Tuple[str, str, Path]] = {}
    for path in glob.glob(str(state_dir / "**" / "results" / "SeHGNN" / "**" / "*_test.json"),
                          recursive=True):
        try:
            d = json.load(open(path))
        except Exception:
            continue
        seed = d.get("config", {}).get("seed")
        rid = d.get("metadata", {}).get("wandb_run_id")
        if not isinstance(seed, int) or not rid:
            continue
        ckpt = checkpoint_root / rid / "best_model.pt"
        if not ckpt.is_file():
            ckpt = ckpt_dummy / rid / "best_model.pt"
        if not ckpt.is_file():
            continue
        ts = d.get("metadata", {}).get("timestamp", "")
        if seed not in best or ts > best[seed][0]:
            best[seed] = (ts, rid, ckpt)
    triples = [(s, rid, ckpt) for s, (_ts, rid, ckpt) in best.items()]
    triples.sort(key=lambda t: t[0])
    return triples


def build_trainer_once(graph, config, device: torch.device, first_ckpt: Path):
    """Build the model once with the first checkpoint loaded.

    Subsequent seeds reuse this Trainer and just call ``load_checkpoint``,
    avoiding the expensive per-seed metapath aggregation.
    """
    from src.ml.train import Trainer
    trainer = Trainer(graph, config)
    trainer.device = device
    trainer.model.to(device)
    if not trainer.load_checkpoint(str(first_ckpt)):
        raise RuntimeError(f"Could not load initial checkpoint: {first_ckpt}")
    trainer.model.eval()
    return trainer


def per_seed_correlations(attention_per_seed: np.ndarray,
                          homophily_by_variant: Dict[str, Dict],
                          metapath_names,
                          min_edges: int = 0) -> pd.DataFrame:
    """Compute Pearson + Spearman per (seed, variant).

    When ``min_edges > 0``, drop meta-paths whose per-variant edge count is
    below the threshold before correlating — keeps the reported ρ aligned with
    the n_edges filter applied inside the figure-rendering function.
    """
    from scripts.compute_attention_homophily_correlation import relation_from_metapath_name
    rels = [relation_from_metapath_name(mp) for mp in metapath_names]
    rows = []
    for seed_idx, attn in enumerate(attention_per_seed):
        for variant_name, hmap in homophily_by_variant.items():
            h_vec = np.asarray([hmap.get(r, (None, 0))[0] for r in rels], dtype=float)
            n_vec = np.asarray([hmap.get(r, (None, 0))[1] for r in rels], dtype=float)
            keep = n_vec >= float(min_edges)
            h_use = np.where(keep, h_vec, np.nan)
            r_p, p_p, n = pearson_ignoring_nans(h_use, attn)
            r_s, p_s, _ = spearman_ignoring_nans(h_use, attn)
            rows.append({
                "seed_idx": seed_idx,
                "variant": variant_name,
                "pearson_rho": r_p, "pearson_p": p_p,
                "spearman_rho": r_s, "spearman_p": p_s,
                "n": n,
            })
    return pd.DataFrame(rows)


def main() -> int:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print(f"Loading graph: {args.graph}")
    graph = _load_graph_any(args.graph)
    ss = [et for et in graph.edge_types if et[0] == "startup" and et[2] == "startup"]
    print(f"  startup→startup edge types: {len(ss)}")

    print(f"Loading config: {args.config}")
    config = load_config(args.config)
    with open(args.champion_config) as f:
        config = _deep_merge(config, yaml.safe_load(f) or {})
    print(f"  Merged champion overrides from {args.champion_config}")

    if args.source == "local":
        print(f"\nDiscovering SeHGNN checkpoints locally under {args.state_dir}…")
        triples = discover_checkpoints_local(Path(args.state_dir), Path(args.checkpoint_root))
        if not triples:
            raise RuntimeError(f"No SeHGNN checkpoints found under {args.state_dir!r}")
    else:
        print(f"\nDiscovering checkpoints in W&B group {args.group}…")
        triples = discover_checkpoints(args.group, Path(args.checkpoint_root), args.project)
        if not triples:
            raise RuntimeError(f"No ok checkpoints found for group {args.group!r}")
    print(f"  found {len(triples)} ok runs with checkpoints on disk: "
          f"seeds={[s for s,_,_ in triples]}")

    print(f"\nBuilding model once (using first checkpoint)…")
    first_seed, first_id, first_ckpt = triples[0]
    trainer = build_trainer_once(graph, config, device, first_ckpt)

    print(f"\nLooping over {len(triples)} seeds — extracting attention…")
    attention_per_seed: List[np.ndarray] = []
    metapath_names_ref = None
    for seed, run_id, ckpt in triples:
        if ckpt != first_ckpt:
            ok = trainer.load_checkpoint(str(ckpt))
            if not ok:
                print(f"  ⚠️ seed {seed} ({run_id}): failed to load {ckpt} — skipping")
                continue
        trainer.model.eval()
        names, attn = extract_attention(trainer.model, graph, device)
        if metapath_names_ref is None:
            metapath_names_ref = names
        elif names != metapath_names_ref:
            raise RuntimeError(f"metapath_names changed between seeds — "
                               f"expected {metapath_names_ref}, got {names}")
        attention_per_seed.append(np.asarray(attn))
        print(f"  seed {seed:>2} ({run_id}): attention vector len={len(attn)}")

    attn_matrix = np.vstack(attention_per_seed)  # (n_seeds, n_metapaths)
    np.save(out_dir / "per_seed_attention.npy", attn_matrix)
    print(f"\nStacked attention matrix shape: {attn_matrix.shape}")

    print("\nComputing per-metapath edge homophily (once — graph is identical)…")
    label_variants = build_label_variants(graph, config)
    homophily_by_variant = {
        name: compute_homophily_for_labels(graph, y) for name, y in label_variants.items()
    }
    for name, hmap in homophily_by_variant.items():
        defined = sum(1 for v in hmap.values() if v[0] is not None)
        print(f"  {name}: {defined}/{len(hmap)} metapaths have defined homophily")

    print(f"\nComputing per-seed correlations (min_edges={args.min_edges})…")
    per_seed_df = per_seed_correlations(
        attn_matrix, homophily_by_variant, metapath_names_ref,
        min_edges=args.min_edges,
    )
    per_seed_df.to_csv(out_dir / "per_seed_correlations.csv", index=False)

    print("\nPer-seed correlation summary (mean ± std across seeds):")
    summary_stats = {}
    for variant, sub in per_seed_df.groupby("variant"):
        for col, label in [("pearson_rho", "Pearson"), ("spearman_rho", "Spearman")]:
            vals = sub[col].dropna().values
            if len(vals) == 0:
                continue
            mean = float(vals.mean())
            std = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0
            summary_stats[f"{variant}__{label}"] = {
                "mean": mean, "std": std, "n_seeds": int(len(vals)),
                "min": float(vals.min()), "max": float(vals.max()),
            }
            print(f"  {label:<8} h_{variant:<14}  mean={mean:+.4f}  "
                  f"std={std:.4f}  n={len(vals)}  min={vals.min():+.4f}  max={vals.max():+.4f}")

    consensus_attention = attn_matrix.mean(axis=0)
    df = assemble_dataframe(metapath_names_ref, consensus_attention, homophily_by_variant)
    df["std_attention"] = attn_matrix.std(axis=0, ddof=1)
    df.rename(columns={"attention_weight": "mean_attention"}, inplace=True)
    df.to_csv(out_dir / "consensus_attention_homophily.csv", index=False)

    print(f"\nConsensus correlations (mean attention vector vs homophily; "
          f"min_edges={args.min_edges}):")
    consensus_correlations: Dict[str, Tuple[float, float, int]] = {}
    consensus_spearman: Dict[str, Tuple[float, float, int]] = {}
    df_for_plot = df.rename(columns={"mean_attention": "attention_weight"})
    for variant_name in label_variants:
        col = f"h_{variant_name}"
        n_col = f"n_edges_{variant_name}"
        # Apply the same n_edges filter used by _scatter_defense_panel so the
        # reported numbers match what the figure visualises (no caption ↔
        # scatter mismatch).
        sub = df[df[n_col].astype(float) >= float(args.min_edges)]
        r_p, p_p, n = pearson_ignoring_nans(sub[col], sub["mean_attention"])
        r_s, p_s, _ = spearman_ignoring_nans(sub[col], sub["mean_attention"])
        consensus_correlations[col] = (r_p, p_p, n)
        consensus_spearman[col] = (r_s, p_s, n)
        print(f"  Pearson (consensus, {col}): rho={r_p:+.4f}  p={p_p:.3g}  n={n}")
        print(f"  Spearman(consensus, {col}): rho={r_s:+.4f}  p={p_s:.3g}  n={n}")

    summary = {
        "champion_config": args.champion_config,
        "graph": args.graph,
        "group": args.group,
        "min_edges": int(args.min_edges),
        "n_seeds": int(attn_matrix.shape[0]),
        "n_metapaths": int(attn_matrix.shape[1]),
        "per_seed_correlations": summary_stats,
        "consensus_correlations": {
            k: {"pearson_rho": v[0], "pearson_p": v[1], "n": v[2],
                "spearman_rho": consensus_spearman[k][0],
                "spearman_p": consensus_spearman[k][1]}
            for k, v in consensus_correlations.items()
        },
    }
    with (out_dir / "correlation_summary.json").open("w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nWrote {out_dir / 'correlation_summary.json'}")

    plot_defense_style_two_panel(
        df_for_plot, consensus_correlations,
        out_dir / "homophily_vs_attention_defense_style.pdf",
        out_dir / "homophily_vs_attention_defense_style.png",
    )
    print(f"Wrote {out_dir / 'homophily_vs_attention_defense_style.pdf'}")
    print("\nDone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
