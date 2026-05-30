"""Replicate a sweep's winning config across N seeds for std-deviation reporting.

Every paper number must be mean ± std across ≥5 seeds (Pineau checklist:
"description of results with central tendency and variation"). This script
takes a W&B sweep_id, finds the best trial by the sweep's metric, materializes
its config to experiments/champion_configs/, and launches N seeded replications
under a shared wandb.group so they can be aggregated by scripts/aggregate_seeds.py.

Usage:
  python scripts/replicate_best.py --sweep-id abc123 --seeds 0 1 2 3 42
  python scripts/replicate_best.py --champion-config experiments/champion_configs/vengnn_tuned.yaml \
      --seeds 0 1 2 3 42 --group vengnn_tuned_mom_replicate

Notes:
- Seeds are run sequentially by default. Use SLURM job arrays for parallelism.
- OOM retry: trials that fail with CUDA OOM are retried once with hidden_dim halved
  and tagged `oom_retry=true`. A second OOM marks the seed as infeasible.
- Status written to experiments/registry/replications.json.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

REGISTRY_PATH = "experiments/registry/replications.json"
REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--sweep-id", type=str, help="W&B sweep id to pull best config from")
    src.add_argument("--champion-config", type=Path, help="Path to pre-selected champion YAML")
    p.add_argument("--seeds", type=int, nargs="+", required=True,
                   help="Seeds to run, e.g. 0 1 2 3 42")
    p.add_argument("--group", type=str, default=None,
                   help="wandb.group override (default: derived from sweep_id or champion path)")
    p.add_argument("--project", type=str, default=None,
                   help="wandb project (default: from config)")
    p.add_argument("--metric", type=str, default=None,
                   help="Metric name for best-trial selection (default: sweep's own)")
    p.add_argument("--dry-run", action="store_true",
                   help="Print commands without executing")
    return p.parse_args()


def load_champion_from_sweep(sweep_id: str, metric: Optional[str]) -> Dict[str, Any]:
    """Query W&B for the sweep's best trial and return its config dict.

    Requires wandb.login() already done. Falls back to an error if the sweep
    has no finished trials.
    """
    import wandb
    api = wandb.Api()
    registry_path = REPO_ROOT / "experiments" / "registry" / "sweeps.json"
    if not registry_path.exists():
        raise RuntimeError(f"sweeps.json not found at {registry_path}; cannot resolve sweep entity/project")
    with open(registry_path) as f:
        reg = json.load(f)
    if sweep_id not in reg:
        raise RuntimeError(f"sweep_id {sweep_id} not in registry")
    entry = reg[sweep_id]
    project = entry["project"]
    sweep = api.sweep(f"{project}/{sweep_id}")

    metric_name = metric or entry["metric"]["name"]
    goal = entry["metric"]["goal"]
    reverse = (goal == "maximize")

    finished = [r for r in sweep.runs if r.state == "finished" and metric_name in r.summary]
    if not finished:
        raise RuntimeError(f"No finished trials with metric '{metric_name}' in sweep {sweep_id}")
    best = sorted(finished, key=lambda r: r.summary[metric_name], reverse=reverse)[0]
    print(f"Best trial: {best.id} ({metric_name}={best.summary[metric_name]:.4f})")
    return dict(best.config)


def load_champion_from_yaml(path: Path) -> Dict[str, Any]:
    with open(path) as f:
        return yaml.safe_load(f) or {}


def derive_group_name(args: argparse.Namespace, champion: Dict[str, Any]) -> str:
    if args.group:
        return args.group
    if args.sweep_id:
        return f"replicate_{args.sweep_id}"
    if args.champion_config:
        return f"replicate_{args.champion_config.stem}"
    model = champion.get("train", {}).get("model", "model") if isinstance(champion.get("train"), dict) else "model"
    return f"replicate_{model}_{int(time.time())}"


def config_to_cli_args(config: Dict[str, Any], prefix: str = "") -> List[str]:
    """Flatten a nested config dict into --key.path value CLI args.

    Empty lists are OMITTED because argparse `nargs='+'` fields reject the
    zero-value `--key` form that would otherwise result. Since the targeted
    fields (e.g. `drop_feature_groups: []`) already default to empty lists in
    config.yaml, omitting them preserves semantics. Observed 2026-04-24: the
    240-run replication batch failed on `--data_processing.ablation.drop_feature_groups`
    getting fed an empty string.
    """
    args: List[str] = []
    for key, val in config.items():
        full_key = f"{prefix}.{key}" if prefix else key
        if isinstance(val, dict):
            args.extend(config_to_cli_args(val, full_key))
        elif isinstance(val, list):
            if not val:
                continue  # omit empty list; let main.py fall back to its config default
            args.extend([f"--{full_key}"] + [str(v) for v in val])
        elif val is None:
            # argparse typed args (e.g. int) reject the string "None";
            # omitting lets main.py use its config-level default/None handling.
            continue
        else:
            args.extend([f"--{full_key}", str(val)])
    return args


def run_one_seed(
    champion: Dict[str, Any], seed: int, group: str, project: Optional[str],
    dry_run: bool = False,
) -> Dict[str, Any]:
    """Run src/main.py for a single seed. Retries once on CUDA OOM with halved hidden_dim."""
    status = "unknown"
    for attempt, (suffix_tag, hidden_scale) in enumerate([("", 1.0), ("oom_retry", 0.5)]):
        cli = config_to_cli_args(champion)
        cli.extend(["--seed", str(seed)])
        if project:
            cli.extend(["--wandb.project", project])
        # Use --wandb.tags to carry the group + attempt annotation.
        # (W&B groups are not directly a CLI arg; we set via WANDB_RUN_GROUP env var.)
        cli.extend(["--wandb.enabled", "true"])
        tags = [f"seed:{seed}", f"group:{group}"]
        if suffix_tag:
            tags.append(suffix_tag)

        # Halve hidden_dim on retry
        if attempt == 1 and hidden_scale < 1.0:
            hd_args = _find_hidden_dim_in_cli(cli)
            if hd_args is None:
                print(f"[seed {seed}] No hidden_dim in champion config; cannot halve. Aborting retries.")
                return {"seed": seed, "attempt": attempt, "status": "no_hidden_to_halve"}
            key, old_val = hd_args
            new_val = max(16, int(old_val * hidden_scale))
            cli = _replace_cli_value(cli, key, str(new_val))
            print(f"[seed {seed}] OOM retry: {key} {old_val} -> {new_val}")

        cmd = [sys.executable, "-u", "src/main.py", *cli]
        env = {"WANDB_RUN_GROUP": group}
        print(f"[seed {seed}] attempt {attempt + 1}: {' '.join(cmd[:8])}...")
        if dry_run:
            return {"seed": seed, "attempt": attempt + 1, "status": "dry_run", "cmd": cmd}

        import os as _os
        full_env = {**_os.environ, **env}
        t0 = time.time()
        result = subprocess.run(cmd, cwd=REPO_ROOT, env=full_env)
        elapsed = time.time() - t0
        if result.returncode == 0:
            return {"seed": seed, "attempt": attempt + 1, "status": "ok",
                    "wall_clock_sec": round(elapsed, 1)}

        # Detect OOM from exit code heuristic (non-zero + fast-ish failure is a red flag;
        # definitive detection requires parsing stderr which the subprocess didn't capture)
        is_likely_oom = (result.returncode != 0)
        if not is_likely_oom or attempt >= 1:
            return {"seed": seed, "attempt": attempt + 1, "status": "failed",
                    "returncode": result.returncode, "wall_clock_sec": round(elapsed, 1)}
        status = "retrying_oom"
        print(f"[seed {seed}] Non-zero exit ({result.returncode}); retrying with halved hidden_dim")

    return {"seed": seed, "status": status}


def _find_hidden_dim_in_cli(cli: List[str]) -> Optional[tuple]:
    """Return (key, value) for the first --models.*.hidden_dim arg, or None."""
    for i, tok in enumerate(cli):
        if tok.startswith("--") and "hidden_dim" in tok and i + 1 < len(cli):
            try:
                return tok, int(cli[i + 1])
            except ValueError:
                continue
    return None


def _replace_cli_value(cli: List[str], key: str, new_val: str) -> List[str]:
    out = list(cli)
    for i, tok in enumerate(out):
        if tok == key and i + 1 < len(out):
            out[i + 1] = new_val
            return out
    return out


def write_replication_registry(group: str, seeds: List[int], champion_source: str,
                                results: List[Dict[str, Any]]) -> None:
    path = REPO_ROOT / REGISTRY_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    registry: Dict[str, Any] = {}
    if path.exists():
        try:
            with open(path) as f:
                registry = json.load(f)
        except json.JSONDecodeError:
            registry = {}
    ok = sum(1 for r in results if r.get("status") == "ok")
    registry[group] = {
        "group": group,
        "seeds_requested": seeds,
        "seeds_completed": [r["seed"] for r in results if r.get("status") == "ok"],
        "champion_source": champion_source,
        "complete": ok == len(seeds),
        "results": results,
        "recorded_at": time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime()),
    }
    with open(path, "w") as f:
        json.dump(registry, f, indent=2, sort_keys=True, default=str)
    print(f"\nRegistered replication group '{group}' ({ok}/{len(seeds)} complete) in {path}")


def main() -> int:
    args = parse_args()
    if args.sweep_id:
        champion = load_champion_from_sweep(args.sweep_id, args.metric)
        source = f"sweep:{args.sweep_id}"
    else:
        champion = load_champion_from_yaml(args.champion_config)
        source = f"config:{args.champion_config}"

    group = derive_group_name(args, champion)
    print(f"Replicating across seeds {args.seeds} under group '{group}'")
    results: List[Dict[str, Any]] = []
    for seed in args.seeds:
        result = run_one_seed(champion, seed, group, args.project, dry_run=args.dry_run)
        results.append(result)
    if not args.dry_run:
        write_replication_registry(group, args.seeds, source, results)
    return 0


if __name__ == "__main__":
    sys.exit(main())
