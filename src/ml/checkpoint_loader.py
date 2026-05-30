"""Post-hoc checkpoint loading for any architecture.

The sweep writes a complete self-contained bundle per trial at
`outputs/checkpoints/<project>/<wandb_run_id>/best_model.pt`:

    {epoch, model_state_dict, optimizer_state_dict, best_metric, config}

`config` is the fully-merged config as of training time (arch hyperparams,
data_processing, metapath_discovery, eval, explain — every section). This
module rebuilds a `Trainer` from that bundle without reading `config.yaml`,
so post-hoc analyses (IG, attention extraction, per-metapath homophily) are
not sensitive to yaml drift between training and now.

Graph handling: prefers the content-addressable cache file at
`outputs/graphs/graph_<hash>.pt` (discovered via the companion
`run_context_*_<run_id>.json`); falls back to the legacy single-file
`outputs/pipeline_state/graph_data.pt`. Both formats (the newer
`(HeteroData, dict)` tuple and the bare `HeteroData`) are handled
transparently.

Typical use (post-hoc IG on a sweep champion):

    from src.ml.checkpoint_loader import load_trainer_from_checkpoint
    trainer = load_trainer_from_checkpoint(
        'outputs/checkpoints/gnn-startup-successs/<run_id>/best_model.pt',
        config_overrides={'explain': {'enabled': True}},
    )
    # trainer.model is the restored champion on the right device.
"""
from __future__ import annotations

import copy
import glob
import json
import os
from typing import Any, Dict, Optional

import torch

DEFAULT_LEGACY_GRAPH = "outputs/pipeline_state/graph_data.pt"
DEFAULT_CACHE_DIR = "outputs/graphs"
DEFAULT_RUN_CONTEXT_DIR = "outputs/pipeline_state/results"


def load_trainer_from_checkpoint(
    checkpoint_path: str,
    graph_path: Optional[str] = None,
    device: Optional[str] = None,
    config_overrides: Optional[Dict[str, Any]] = None,
):
    """Rebuild a ready-to-use Trainer from a sweep-produced checkpoint.

    Parameters
    ----------
    checkpoint_path:
        Path to a `best_model.pt` containing `{epoch, model_state_dict,
        optimizer_state_dict, best_metric, config}`.
    graph_path:
        Optional explicit path to the graph. If None, discover via the run's
        `graph_version` hash (look up a sibling `run_context_*_<run_id>.json`)
        and fall back to the legacy `outputs/pipeline_state/graph_data.pt`.
    device:
        `'cuda'`, `'cpu'`, or None (honor the embedded config, else autoselect).
    config_overrides:
        Deep-merged AFTER the checkpoint's embedded config. Use this to flip
        post-hoc toggles like `{'explain': {'enabled': True}}` or
        `{'analysis': {'enable_homophily_analysis': True}}`.

    Returns
    -------
    Trainer
        With `model` already populated from the checkpoint's state dict.
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if "config" not in ckpt:
        raise ValueError(
            f"Checkpoint at {checkpoint_path} has no embedded `config` key. "
            "It was likely produced by an older Trainer version. "
            "Fall back to scripts/case_study.load_graph_and_model with "
            "explicit overrides."
        )

    # Checkpoint config is source of truth. Deep-copy so overrides don't
    # leak back into the loaded dict in-place.
    config = copy.deepcopy(ckpt["config"])

    # Inference safety: prevent accidental training / wandb init / heavy
    # analysis phases unless the caller explicitly re-enables via overrides.
    config.setdefault("train", {})["epochs"] = 0
    config.setdefault("wandb", {})["enabled"] = False
    analysis = config.setdefault("analysis", {})
    analysis.setdefault("enable_downstream_analysis", False)
    analysis.setdefault("enable_homophily_analysis", False)
    config.setdefault("explain", {}).setdefault("enabled", False)

    if device is not None:
        config["train"]["device"] = device

    if config_overrides:
        _deep_merge(config, config_overrides)

    # Locate the graph
    resolved_graph_path = graph_path or _autodetect_graph_path(checkpoint_path)
    if not os.path.exists(resolved_graph_path):
        raise FileNotFoundError(
            f"Graph not found at resolved path '{resolved_graph_path}'. "
            f"Pass graph_path= explicitly, or rebuild via preprocessing."
        )
    graph_data = _load_graph(resolved_graph_path)

    # Local import so this module is importable without touching train.py
    # until actually needed (and to avoid circular imports).
    from src.ml.train import Trainer
    trainer = Trainer(graph_data, config)
    trainer.load_checkpoint(checkpoint_path)
    # Trainer moves the model to its device but keeps `data` on CPU (data
    # loaders handle the transfer batch-by-batch during training). For
    # post-hoc full-graph inference we want model + data co-located, so mirror
    # what scripts/case_study.load_graph_and_model does.
    trainer.data = trainer.data.to(trainer.device)
    return trainer


def _autodetect_graph_path(checkpoint_path: str) -> str:
    """Resolve graph_path using the run's run_context graph_version hash,
    falling back to the legacy single-file cache."""
    run_id = os.path.basename(os.path.dirname(checkpoint_path))
    # The run_context file is written with filename `run_context_<pid>_<run_id>.json`
    # under outputs/pipeline_state/results/
    pattern = os.path.join(DEFAULT_RUN_CONTEXT_DIR, f"run_context_*_{run_id}.json")
    matches = sorted(glob.glob(pattern), key=os.path.getmtime, reverse=True)
    for rc_path in matches:
        try:
            with open(rc_path) as f:
                rc = json.load(f)
            gv = rc.get("graph_version")
            if gv:
                cand = os.path.join(DEFAULT_CACHE_DIR, f"graph_{gv}.pt")
                if os.path.exists(cand):
                    return cand
        except (OSError, json.JSONDecodeError):
            continue
    # Fall back to the legacy single-file path. Both training code paths
    # (build_or_load_graph + main.py's explicit torch.save) currently write
    # here, so it's the safest default.
    return DEFAULT_LEGACY_GRAPH


def _load_graph(path: str):
    """Return a HeteroData regardless of which graph-save format was used."""
    loaded = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(loaded, tuple):
        # graph_cache.py format: (graph_data, node_names)
        return loaded[0]
    return loaded


def _deep_merge(base: Dict[str, Any], overrides: Dict[str, Any]) -> None:
    """Recursively merge `overrides` into `base` in place."""
    for k, v in overrides.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _deep_merge(base[k], v)
        else:
            base[k] = v
