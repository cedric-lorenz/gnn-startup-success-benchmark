"""Content-addressable graph cache.

Preprocessing + graph assembly takes 3-5 minutes per run and produces a deterministic
artifact given the config subset (data_processing, graph_assembler, metapath_discovery,
paths). Across a 1000-run sweep this wastes ~50-80 GPU hours of idle time. This module
caches the built graph on local disk, keyed by a content hash of the relevant config
subset.

Cache location: outputs/graphs/graph_<version>.pt (HPC-local only, NOT uploaded to W&B
because typical graphs are 1-2 GB and the project runs on the 5 GB W&B free tier).

Registry: experiments/registry/graphs.json maps version hashes to human-readable
descriptions and provenance (commit SHA, creation timestamp, file size).

Safety:
- Fresh build on cache miss, bit-equal cached hit.
- Corrupted cache file → rebuild and overwrite.
- Config-incompatible cache → new hash → new file (old cache remains for old runs).
- `--graph.rebuild` flag (or config["graph_cache"]["rebuild"]=true) forces rebuild.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch

from src.ml.run_context import _config_hash, _graph_version_subset


DEFAULT_CACHE_DIR = "outputs/graphs"
DEFAULT_REGISTRY_PATH = "experiments/registry/graphs.json"


def compute_graph_version(config: Dict[str, Any]) -> str:
    """Stable 16-char hash of the config subset that determines graph content."""
    return _config_hash(_graph_version_subset(config))


def _cache_path(version: str, cache_dir: str = DEFAULT_CACHE_DIR) -> Path:
    return Path(cache_dir) / f"graph_{version}.pt"


def _registry_load(registry_path: str = DEFAULT_REGISTRY_PATH) -> Dict[str, Any]:
    p = Path(registry_path)
    if not p.exists():
        return {}
    try:
        with open(p) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}


def _registry_update(
    version: str,
    entry: Dict[str, Any],
    registry_path: str = DEFAULT_REGISTRY_PATH,
) -> None:
    p = Path(registry_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    registry = _registry_load(registry_path)
    registry[version] = entry
    with open(p, "w") as f:
        json.dump(registry, f, indent=2, sort_keys=True, default=str)


def build_or_load_graph(
    config: Dict[str, Any],
    build_fn,
    cache_dir: str = DEFAULT_CACHE_DIR,
    registry_path: str = DEFAULT_REGISTRY_PATH,
    force_rebuild: Optional[bool] = None,
) -> Tuple[Any, Dict[str, Any]]:
    """Load cached graph for this config hash, or build and cache if missing.

    Parameters
    ----------
    config
        Full merged config dict (post-CLI-override).
    build_fn
        Zero-arg callable that returns (graph_data, node_names) — the existing
        preprocessing pipeline. Wrap it as a closure at the call site so this
        module stays decoupled from preprocessing internals.
    cache_dir
        Where to store cached *.pt files on HPC-local disk.
    registry_path
        JSON file mapping version → metadata.
    force_rebuild
        If True, ignore the cache. If None, checks config["graph_cache"]["rebuild"].

    Returns
    -------
    (graph_data, node_names) — same signature as perform_preprocessing().
    """
    version = compute_graph_version(config)
    cache_path = _cache_path(version, cache_dir)

    if force_rebuild is None:
        force_rebuild = bool(config.get("graph_cache", {}).get("rebuild", False))

    if cache_path.exists() and not force_rebuild:
        try:
            print(f"[graph_cache] HIT: loading {cache_path} (version {version})")
            t0 = time.time()
            graph_data, node_names = torch.load(cache_path, weights_only=False)
            print(f"[graph_cache] Loaded in {time.time() - t0:.1f}s")
            _registry_touch(version, registry_path)
            return graph_data, node_names
        except Exception as e:
            print(f"[graph_cache] Cache file at {cache_path} is corrupt ({e}); rebuilding.")

    print(f"[graph_cache] MISS: building graph (version {version})")
    t0 = time.time()
    graph_data, node_names = build_fn()
    build_sec = time.time() - t0
    print(f"[graph_cache] Built in {build_sec:.1f}s; caching to {cache_path}")

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = cache_path.with_suffix(cache_path.suffix + ".tmp")
    torch.save((graph_data, node_names), tmp_path)
    os.replace(tmp_path, cache_path)  # atomic rename; avoids partial writes

    _registry_update(
        version,
        {
            "version": version,
            "path": str(cache_path),
            "size_mb": round(cache_path.stat().st_size / 1024**2, 1),
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime()),
            "build_sec": round(build_sec, 1),
            "git_commit": _git_commit_short(),
            "config_subset": _graph_version_subset(config),
            "last_loaded_at": None,
        },
        registry_path,
    )
    return graph_data, node_names


def _git_commit_short() -> Optional[str]:
    """Best-effort short git SHA for provenance; None if git unavailable."""
    import subprocess
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL, text=True, timeout=3,
        ).strip()
    except Exception:
        return None


def _registry_touch(version: str, registry_path: str) -> None:
    """Update `last_loaded_at` on a cache hit so stale entries can be pruned."""
    registry = _registry_load(registry_path)
    if version in registry:
        registry[version]["last_loaded_at"] = time.strftime(
            "%Y-%m-%dT%H:%M:%S%z", time.localtime()
        )
        p = Path(registry_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w") as f:
            json.dump(registry, f, indent=2, sort_keys=True, default=str)
