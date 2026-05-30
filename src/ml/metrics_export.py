"""
Local JSON metrics export for offline analysis of training runs.
Saves structured metrics files to outputs/results/{model}/{target_mode}/
"""
import json
import os
import traceback
from datetime import datetime
from typing import Dict, Any, Optional

import numpy as np


def export_metrics_json(
    metrics: Dict[str, Any],
    config: Dict[str, Any],
    mode: str,
    epoch: Optional[int],
    model_name: str,
    target_mode: str,
    best_metric: Optional[float] = None,
    best_epoch: Optional[int] = None,
    output_base_dir: str = "outputs/results",
) -> str:
    """
    Save evaluation metrics to a structured JSON file.

    Returns the path to the written file.
    """
    run_metadata = {
        "model": model_name,
        "target_mode": target_mode,
        "mode": mode,
        "epoch": epoch,
        "timestamp": datetime.now().isoformat(),
        "best_metric": best_metric,
        "best_epoch": best_epoch,
        "seed": config.get("seed"),
    }

    # Capture wandb run info if available
    try:
        import wandb

        if wandb.run is not None:
            run_metadata["wandb_run_id"] = wandb.run.id
            run_metadata["wandb_run_name"] = wandb.run.name
            run_metadata["wandb_sweep_id"] = getattr(wandb.run, "sweep_id", None)
    except (ImportError, AttributeError):
        pass

    payload = {
        "metadata": run_metadata,
        "config": _make_json_serializable(config),
        "metrics": _make_json_serializable(metrics),
    }

    wandb_id = run_metadata.get("wandb_run_id", "local")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{timestamp}_{wandb_id}_{mode}.json"

    dir_path = os.path.join(output_base_dir, model_name, target_mode)
    os.makedirs(dir_path, exist_ok=True)

    file_path = os.path.join(dir_path, filename)
    with open(file_path, "w") as f:
        json.dump(payload, f, indent=2, default=str)

    print(f"📁 Saved metrics to {file_path}")
    return file_path


def save_error_report(
    error: Exception,
    config: Dict[str, Any],
    output_base_dir: str = "outputs/results",
) -> str:
    """
    Save error details from a failed run to a structured JSON file.

    Returns the path to the written file.
    """
    model_name = config.get("train", {}).get("model", "unknown")
    target_mode = config.get("data_processing", {}).get("target_mode", "unknown")

    error_metadata = {
        "model": model_name,
        "target_mode": target_mode,
        "status": "failed",
        "timestamp": datetime.now().isoformat(),
        "error_type": type(error).__name__,
        "error_message": str(error),
        "traceback": traceback.format_exception(type(error), error, error.__traceback__),
    }

    # Capture wandb run info if available
    try:
        import wandb

        if wandb.run is not None:
            error_metadata["wandb_run_id"] = wandb.run.id
            error_metadata["wandb_run_name"] = wandb.run.name
            error_metadata["wandb_sweep_id"] = getattr(wandb.run, "sweep_id", None)
    except (ImportError, AttributeError):
        pass

    payload = {
        "metadata": error_metadata,
        "config": _make_json_serializable(config),
    }

    wandb_id = error_metadata.get("wandb_run_id", "local")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{timestamp}_{wandb_id}_error.json"

    dir_path = os.path.join(output_base_dir, model_name, target_mode)
    os.makedirs(dir_path, exist_ok=True)

    file_path = os.path.join(dir_path, filename)
    with open(file_path, "w") as f:
        json.dump(payload, f, indent=2, default=str)

    print(f"📁 Saved error report to {file_path}")
    return file_path


def _make_json_serializable(obj):
    """Recursively convert non-serializable types (numpy, torch) to Python natives."""
    if isinstance(obj, dict):
        return {str(k): _make_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_make_json_serializable(item) for item in obj]
    elif isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating,)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif hasattr(obj, "item"):  # torch scalar
        return obj.item()
    else:
        return obj
