"""Run Integrated Gradients on a SeHGNN checkpoint with a configurable
baseline (zero or per-feature mean over training startups). Writes per-feature
attribution CSVs and PDFs so the two baselines can be compared side-by-side.

The mean baseline addresses the standard critique that the zero baseline is
artifactual for features where zero is in-distribution. We compute the mean
over training-mask startups (no test leakage) and broadcast it as the per-node
baseline tensor; everything else (graph topology, other-type features) is
left identical to the zero-baseline run.

Reproducibility:
  python scripts/extract_sehgnn_ig_baselines.py \\
    --champion-config experiments/champion_configs/sehgnn_g4_heterophily.yaml \\
    --graph outputs/graphs/graph_9af09bd47d2df937.pt \\
    --checkpoint outputs/checkpoints/gnn-startup-successs/dcdjehww/best_model.pt \\
    --output-dir outputs/explanations/sehgnn_g4_seed15_ig_baselines \\
    --baseline-mode mean

Run twice (once with --baseline-mode zero, once with mean) to get both CSVs
for comparison.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault("WANDB_MODE", "disabled")

import torch
import yaml

from src.ml.utils import load_config
from src.ml.train import Trainer
from src.ml import explain as explain_mod
from scripts.compute_attention_homophily_correlation import _deep_merge, _load_graph_any


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--config", default=str(PROJECT_ROOT / "config.yaml"))
    p.add_argument("--champion-config", required=True)
    p.add_argument("--graph", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--baseline-mode", choices=["zero", "mean", "eg"], default="zero",
                   help="IG reference point. 'zero' = standard (default); "
                        "'mean' = per-feature mean over training startups; "
                        "'eg' = Expected Gradients implemented as N sequential "
                        "IG runs, each against one distinct training startup "
                        "sampled uniformly from the train mask. Per-feature "
                        "attributions are averaged across the N runs. This is "
                        "mathematically equivalent to Captum's GradientShap "
                        "with n_samples=N but stays inside the serial "
                        "internal_batch_size=1 path that the SeHGNN hetero "
                        "scatter ops can handle on an 80GB GPU.")
    p.add_argument("--eg-n", type=int, default=5,
                   help="Number of sampled training startups used as EG "
                        "baselines (Monte Carlo samples over the training "
                        "distribution). With --eg-method gradient_shap each "
                        "iteration costs ~1 gradient pass; N=100 takes ~3 min "
                        "per seed.")
    p.add_argument("--eg-method", choices=["gradient_shap", "ig_serial"],
                   default="gradient_shap",
                   help="EG estimator. 'gradient_shap' (default, recommended) "
                        "uses Captum's native GradientShap with n_samples=1 "
                        "and stdevs=eg_sigma per outer iteration: one random "
                        "alpha and one Gaussian noise sample per call, "
                        "matching Erion et al. 2021. 'ig_serial' is the "
                        "previous no-noise path: full IG with --eg-n-steps "
                        "alpha grid against each sampled training startup. "
                        "Both are mathematically EG, but gradient_shap is "
                        "~50x cheaper per outer iteration and is the form "
                        "implemented in Captum/SHAP.")
    p.add_argument("--eg-sigma", type=float, default=0.1,
                   help="Gaussian noise standard deviation added to the "
                        "input inside Captum's GradientShap. Only used when "
                        "--eg-method gradient_shap. Default 0.1 matches "
                        "Erion et al. 2021 for standardized features.")
    p.add_argument("--eg-n-steps", type=int, default=50,
                   help="IG integration steps inside each EG sub-run. "
                        "Only used when --eg-method ig_serial.")
    p.add_argument("--eg-seed", type=int, default=42,
                   help="RNG seed for deterministic baseline sampling AND "
                        "for the per-iteration torch.manual_seed call that "
                        "fixes GradientShap's alpha and noise draws. "
                        "Independent of the model's training seed.")
    return p.parse_args()


def _sample_eg_training_indices(data, n: int, seed: int = 42):
    """Sample `n` distinct training-startup row indices uniformly from the
    train mask. Returns a 1-D LongTensor. Used to build per-iteration EG
    baselines in the N-sequential-IG-call implementation.
    """
    rng = torch.Generator().manual_seed(seed)
    train_idx = data["startup"].train_mask.nonzero(as_tuple=True)[0]
    if n > len(train_idx):
        raise ValueError(f"n={n} exceeds train-mask size {len(train_idx)}")
    perm = torch.randperm(len(train_idx), generator=rng)
    return train_idx[perm[:n]]


def _eg_single_iteration_baseline_tuple(data, device, sampled_row_idx):
    """Build the IG baseline tuple for ONE EG sub-run: startup baseline is
    the sampled training startup's feature vector broadcast across all
    startup-node rows. Non-startup tensors and edge masks are zero so the
    only signal differing from the mean-baseline run is the choice of
    reference startup.
    """
    sampled_x = data["startup"].x[sampled_row_idx]  # [F]
    n_startup = data["startup"].x.shape[0]
    startup_baseline = (
        sampled_x.unsqueeze(0).expand(n_startup, -1).unsqueeze(0)
        .contiguous().to(device)
    )  # [1, N_startup, F]
    baselines = []
    for nt in data.node_types:
        if nt == "startup":
            baselines.append(startup_baseline)
        else:
            baselines.append(torch.zeros_like(data[nt].x).unsqueeze(0).to(device))
    for et in data.edge_types:
        n_edges = data[et].edge_index.shape[1]
        baselines.append(torch.zeros(1, n_edges, device=device, dtype=torch.float))
    return tuple(baselines)


def _compute_mean_baseline_tuple(data, device):
    """Build a baselines tuple matching PyG `to_captum_input` flattening order
    for `mask_type=node_and_edge`. That order is:
        (x_<node_type_1>.unsqueeze(0), ..., edge_mask_<edge_type_1>.unsqueeze(0), ...)

    Only startup-feature baseline is non-zero: per-feature mean over training
    startups. All other tensors (non-startup features, edge masks) are zero,
    so the only difference between this run and the zero-baseline run is the
    startup-feature reference.
    """
    train_mask = data["startup"].train_mask
    startup_x = data["startup"].x
    startup_mean = startup_x[train_mask].mean(dim=0, keepdim=True)  # [1, F]
    startup_baseline = startup_mean.expand_as(startup_x).unsqueeze(0).to(device)

    baselines = []
    # Node-type tensors (in data.node_types order)
    for nt in data.node_types:
        if nt == "startup":
            baselines.append(startup_baseline)
        else:
            baselines.append(torch.zeros_like(data[nt].x).unsqueeze(0).to(device))
    # Edge-mask tensors (in data.edge_types order, shape [1, n_edges] each)
    for et in data.edge_types:
        n_edges = data[et].edge_index.shape[1]
        baselines.append(torch.zeros(1, n_edges, device=device, dtype=torch.float))
    return tuple(baselines)


def main() -> int:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Baseline mode: {args.baseline_mode}")

    print(f"Loading graph: {args.graph}")
    graph = _load_graph_any(args.graph)
    test_n = int(graph["startup"].test_mask.sum().item())
    print(f"  test startups: {test_n}")

    print(f"Loading config: {args.config}")
    config = load_config(args.config)
    with open(args.champion_config) as f:
        config = _deep_merge(config, yaml.safe_load(f) or {})
    config["train"]["epochs"] = 0
    config.setdefault("explain", {})
    config["explain"]["enabled"] = False
    config["explain"]["sample_size"] = test_n
    config["explain"]["sampling_method"] = "random"
    config.setdefault("wandb", {})["enabled"] = False
    explain_mod.config["wandb"]["enabled"] = False

    print("Building Trainer...")
    trainer = Trainer(graph, config)
    trainer.data = trainer.data.to(device)
    trainer.model.to(device)

    print(f"Loading checkpoint: {args.checkpoint}")
    if not trainer.load_checkpoint(args.checkpoint):
        raise RuntimeError(f"Failed to load checkpoint: {args.checkpoint}")
    trainer.model.eval()

    # Build the IG method dict, optionally injecting a mean baseline.
    target_mode = config["data_processing"]["target_mode"]
    method = {"attribution_method": "IntegratedGradients"}

    if args.baseline_mode == "mean":
        print("Computing per-feature mean baseline over training startups...")
        baselines_tuple = _compute_mean_baseline_tuple(trainer.data, device)
        # to_captum_input flattens hetero(x, edge_index) into a tuple of
        # tensors (one per node type, then one per edge type). Captum's IG
        # `baselines` must match that tuple shape exactly. The helper above
        # builds the tuple in `data.node_types` + `data.edge_types` order;
        # only the startup-feature tensor carries a non-zero baseline
        # (per-feature mean over training startups), everything else is zero
        # so this run differs from the zero-baseline run only on startup
        # features.
        startup_idx = list(trainer.data.node_types).index("startup")
        print(f"  baselines tuple length: {len(baselines_tuple)}  "
              f"(startup at index {startup_idx})")
        print(f"  startup baseline mean (first 5 feats): "
              f"{baselines_tuple[startup_idx][0, 0, :5].tolist()}")
        method["baselines"] = baselines_tuple

    elif args.baseline_mode == "eg":
        if args.eg_method == "gradient_shap":
            print(f"Expected-Gradients via {args.eg_n} native GradientShap "
                  f"calls (n_samples=1, stdevs={args.eg_sigma}), "
                  f"sampling seed={args.eg_seed}.")
        else:
            print(f"Expected-Gradients via {args.eg_n} sequential IG sub-runs "
                  f"(no Gaussian noise), each with n_steps={args.eg_n_steps} "
                  f"integration, sampling seed={args.eg_seed}.")
        sample_idx = _sample_eg_training_indices(
            trainer.data, n=args.eg_n, seed=args.eg_seed)
        print(f"  sampled training-startup row indices: "
              f"{sample_idx.tolist()}")

        import shutil
        import pandas as pd

        per_iter_dirs = []
        for i, row_idx in enumerate(sample_idx.tolist()):
            iter_dir = out_dir / f"eg_iter_{i:02d}_sampleidx_{row_idx}"
            iter_dir.mkdir(parents=True, exist_ok=True)
            print(f"\n--- EG sub-run {i+1}/{args.eg_n} "
                  f"(sample idx {row_idx}, output {iter_dir.name}) ---")
            baselines_tuple = _eg_single_iteration_baseline_tuple(
                trainer.data, device, row_idx)
            # Seed torch globally per iteration so GradientShap's internal
            # alpha and Gaussian noise draws are deterministic from
            # (args.eg_seed, i). The outer-loop baseline sampling is already
            # deterministic via the rng generator in _sample_eg_training_indices.
            torch.manual_seed(args.eg_seed + i)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(args.eg_seed + i)
            if args.eg_method == "gradient_shap":
                iter_method = {
                    "attribution_method": "GradientShap",
                    "baselines": baselines_tuple,
                    "n_samples": 1,
                    "stdevs": args.eg_sigma,
                }
            else:  # ig_serial
                iter_method = {
                    "attribution_method": "IntegratedGradients",
                    "baselines": baselines_tuple,
                    "n_steps": args.eg_n_steps,
                }
            explain_mod.explain_model(
                model=trainer.model,
                data=trainer.data,
                explain_path=str(iter_dir),
                target_mode=target_mode,
                sample_size=test_n,
                method=iter_method,
            )
            per_iter_dirs.append(iter_dir)

        # Aggregate: average per-feature attributions across the N sub-runs.
        print(f"\nAggregating across {len(per_iter_dirs)} sub-runs...")
        for task in ("mom_task", "liq_task"):
            fname = f"startup_feature_importance_{task}_improved_data_full.csv"
            dfs = []
            for d in per_iter_dirs:
                p = d / fname
                if p.exists():
                    dfs.append(pd.read_csv(p).set_index("feature"))
            if not dfs:
                print(f"  {task}: no sub-run CSVs found, skipping aggregation")
                continue
            combined = pd.concat(dfs, axis=1)
            abs_cols = [c for c in combined.columns if c == "abs_importance"]
            raw_cols = [c for c in combined.columns if c == "raw_importance"]
            agg = pd.DataFrame(index=dfs[0].index)
            agg["abs_importance"] = combined[abs_cols].mean(axis=1)
            agg["raw_importance"] = combined[raw_cols].mean(axis=1)
            agg.reset_index().to_csv(out_dir / fname, index=False)
            print(f"  {task}: wrote averaged CSV ({len(agg)} features) to {out_dir / fname}")

        print(f"\nDone. EG outputs (averaged) in {out_dir}; per-iteration in "
              f"{out_dir}/eg_iter_*/")
        return 0

    print(f"target_mode={target_mode}  method.keys={list(method.keys())}  "
          f"sample_size={test_n}")

    explain_mod.explain_model(
        model=trainer.model,
        data=trainer.data,
        explain_path=str(out_dir),
        target_mode=target_mode,
        sample_size=test_n,
        method=method,
    )

    print(f"Done. Baseline mode={args.baseline_mode}; outputs in {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
