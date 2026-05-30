"""Captures per-run reproducibility metadata (Pineau ML Reproducibility Checklist v2.0).

Every metric reported in a paper must trace back to a run whose context is fully logged:
git SHA, dirty flag, environment versions, hardware, seed, CLI command, graph version,
wall-clock time, and peak GPU memory. This module is the single source of truth for
that capture — called once at run start, once at run end.
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

import torch


def _run_cmd(cmd: list[str], cwd: Optional[Path] = None) -> Optional[str]:
    try:
        out = subprocess.check_output(cmd, cwd=cwd, stderr=subprocess.DEVNULL, text=True, timeout=5)
        return out.strip()
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError):
        return None


def _git_info(repo_root: Path) -> Dict[str, Any]:
    commit = _run_cmd(["git", "rev-parse", "HEAD"], cwd=repo_root) or "unknown"
    short = _run_cmd(["git", "rev-parse", "--short", "HEAD"], cwd=repo_root) or "unknown"
    branch = _run_cmd(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=repo_root) or "unknown"
    # Ignore submodule state — HGNN_Thesis (paper-writing repo) always has
    # WIP edits and doesn't affect the code path this run executes.
    status = _run_cmd(["git", "status", "--porcelain", "--ignore-submodules=all"], cwd=repo_root)
    dirty = bool(status)
    dirty_files = [line.split(maxsplit=1)[-1] for line in status.splitlines()] if status else []
    return {
        "commit": commit,
        "commit_short": short,
        "branch": branch,
        "dirty": dirty,
        "dirty_file_count": len(dirty_files),
        "dirty_files": dirty_files[:50],
    }


def _env_info() -> Dict[str, Any]:
    info: Dict[str, Any] = {
        "python_version": sys.version.split()[0],
        "python_implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "hostname": socket.gethostname(),
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None,
    }
    for mod_name, attr_name in [
        ("torch_geometric", "pyg_version"),
        ("numpy", "numpy_version"),
        ("sklearn", "sklearn_version"),
        ("pandas", "pandas_version"),
        ("scipy", "scipy_version"),
        ("wandb", "wandb_version"),
    ]:
        try:
            mod = __import__(mod_name)
            info[attr_name] = getattr(mod, "__version__", "unknown")
        except ImportError:
            info[attr_name] = None
    return info


def _gpu_info() -> Dict[str, Any]:
    if not torch.cuda.is_available():
        return {"gpu_count": 0, "gpu_name": None, "gpu_total_mem_gb": None}
    return {
        "gpu_count": torch.cuda.device_count(),
        "gpu_name": torch.cuda.get_device_name(0),
        "gpu_total_mem_gb": round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 2),
        "gpu_capability": ".".join(str(x) for x in torch.cuda.get_device_capability(0)),
    }


def _slurm_info() -> Dict[str, Any]:
    return {
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_job_name": os.environ.get("SLURM_JOB_NAME"),
        "slurm_node": os.environ.get("SLURMD_NODENAME") or os.environ.get("SLURM_NODELIST"),
        "slurm_partition": os.environ.get("SLURM_JOB_PARTITION"),
        "slurm_ntasks": os.environ.get("SLURM_NTASKS"),
        "slurm_cpus_per_task": os.environ.get("SLURM_CPUS_PER_TASK"),
    }


def _config_hash(config: Dict[str, Any]) -> str:
    """Stable hash of a config dict. Used for graph_version and champion-config keys."""
    canon = json.dumps(config, sort_keys=True, default=str)
    return hashlib.sha256(canon.encode()).hexdigest()[:16]


def _graph_version_subset(config: Dict[str, Any]) -> Dict[str, Any]:
    """Subset of config that affects the prebuilt graph content.

    If any of these change, the graph must be rebuilt. Conversely, if all are unchanged,
    the cached graph artifact is valid for this run.

    `_graph_schema_version` is a build-logic version tag. Bump when the assembler
    emits a structurally different graph for the same config (e.g. edge added/dropped,
    metapath-materialization change). Forces fresh rebuilds without touching the
    user-facing config surface.
    """
    keys = ["data_processing", "graph_assembler", "metapath_discovery", "paths"]
    subset = {k: config.get(k) for k in keys if k in config}
    # Bump tag whenever the assembler output changes for the same nominal config.
    # v2 = worked_with base edge stripped after metapath materialization (2026-04-23).
    # v3 = split RNG pinned to canonical data_processing.split_seed (2026-04-24),
    #      independent of the run's training seed → all runs share one split per variant.
    subset["_graph_schema_version"] = "v3-pinned-split"
    return subset


def capture_run_context(config: Dict[str, Any], cli_command: Optional[list[str]] = None) -> Dict[str, Any]:
    """Gather all reproducibility metadata at run start.

    Returns a flat dict suitable for `wandb.config.update(..., allow_val_change=True)`
    or for writing to a JSON artifact. Pineau checklist fields captured:
    git, env, hardware, seed, hyperparams (via config), CLI command, runtime infra.
    """
    repo_root = Path(__file__).resolve().parents[2]
    ctx = {
        "_run_context_version": "1.0",
        "_captured_at_unix": time.time(),
        "_captured_at_iso": time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime()),
        "cli_command": " ".join(cli_command) if cli_command else " ".join(sys.argv),
        "cli_argv": cli_command or sys.argv,
        "cwd": os.getcwd(),
        "pid": os.getpid(),
        "seed": config.get("seed"),
        "graph_version": _config_hash(_graph_version_subset(config)),
        "config_full_hash": _config_hash(config),
    }
    ctx["git"] = _git_info(repo_root)
    ctx["env"] = _env_info()
    ctx["gpu"] = _gpu_info()
    ctx["slurm"] = _slurm_info()
    return ctx


def finalize_run_context(start_ctx: Dict[str, Any]) -> Dict[str, Any]:
    """Augment a captured context with end-of-run metrics: wall-clock, peak GPU memory."""
    now = time.time()
    runtime = {
        "wall_clock_sec": round(now - start_ctx["_captured_at_unix"], 1),
        "ended_at_unix": now,
        "ended_at_iso": time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime(now)),
    }
    if torch.cuda.is_available():
        runtime["peak_gpu_mem_mb"] = round(torch.cuda.max_memory_allocated() / 1024**2, 1)
        runtime["peak_gpu_mem_reserved_mb"] = round(torch.cuda.max_memory_reserved() / 1024**2, 1)
    return runtime


def log_context_to_wandb(ctx: Dict[str, Any], wandb_run=None) -> None:
    """Push captured context into wandb.config.

    `wandb_run` may be None (wandb disabled); in that case this is a no-op.
    """
    if wandb_run is None:
        return
    flat = _flatten(ctx, prefix="repro")
    try:
        wandb_run.config.update(flat, allow_val_change=True)
    except Exception as e:
        print(f"[run_context] Warning: could not push to wandb.config: {e}")


def write_context_json(ctx: Dict[str, Any], output_dir: str, filename: str = "run_context.json") -> str:
    """Persist the captured context to a local JSON file.

    Also uploaded as a W&B artifact separately so we have an offline record.
    Returns the full path written.
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    path = Path(output_dir) / filename
    with open(path, "w") as f:
        json.dump(ctx, f, indent=2, default=str)
    return str(path)


def _flatten(d: Any, prefix: str = "", sep: str = "/") -> Dict[str, Any]:
    """Flatten a nested dict for wandb.config (which only accepts flat keys well)."""
    out: Dict[str, Any] = {}
    if not isinstance(d, dict):
        out[prefix] = d
        return out
    for k, v in d.items():
        new_key = f"{prefix}{sep}{k}" if prefix else str(k)
        if isinstance(v, dict):
            out.update(_flatten(v, new_key, sep))
        else:
            out[new_key] = v
    return out
