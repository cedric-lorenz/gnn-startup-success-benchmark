"""Tests for src.ml.run_context — reproducibility metadata capture."""
from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from src.ml.run_context import (
    _config_hash,
    _graph_version_subset,
    _flatten,
    capture_run_context,
    finalize_run_context,
    write_context_json,
)


@pytest.fixture
def dummy_config():
    return {
        "seed": 42,
        "train": {"model": "SeHGNN", "lr": 0.01, "epochs": 10},
        "data_processing": {"use_org_description": True, "scaling": "robust"},
        "graph_assembler": {"metapaths": ["SIS", "SFS"]},
        "metapath_discovery": {"mode": "manual"},
        "paths": {"data_dir": "data/"},
        "wandb": {"enabled": False},
    }


def test_config_hash_stable(dummy_config):
    h1 = _config_hash(dummy_config)
    h2 = _config_hash(dummy_config)
    assert h1 == h2
    assert len(h1) == 16


def test_config_hash_changes_on_content(dummy_config):
    h1 = _config_hash(dummy_config)
    modified = {**dummy_config, "train": {**dummy_config["train"], "lr": 0.02}}
    h2 = _config_hash(modified)
    assert h1 != h2


def test_graph_version_only_depends_on_graph_keys(dummy_config):
    subset1 = _graph_version_subset(dummy_config)
    h1 = _config_hash(subset1)

    # Changing train.lr should NOT change graph_version
    modified = {**dummy_config, "train": {**dummy_config["train"], "lr": 0.99}}
    subset2 = _graph_version_subset(modified)
    h2 = _config_hash(subset2)
    assert h1 == h2

    # Changing data_processing SHOULD change graph_version
    modified = {**dummy_config, "data_processing": {**dummy_config["data_processing"], "scaling": "standard"}}
    subset3 = _graph_version_subset(modified)
    h3 = _config_hash(subset3)
    assert h1 != h3


def test_capture_populates_required_fields(dummy_config):
    ctx = capture_run_context(dummy_config)
    # Pineau checklist coverage
    assert ctx["seed"] == 42
    assert "git" in ctx and "commit" in ctx["git"] and "dirty" in ctx["git"]
    assert "env" in ctx and "python_version" in ctx["env"] and "torch_version" in ctx["env"]
    assert "gpu" in ctx and "gpu_count" in ctx["gpu"]
    assert "slurm" in ctx
    assert "cli_command" in ctx
    assert "graph_version" in ctx
    assert "_captured_at_unix" in ctx


def test_finalize_adds_runtime(dummy_config):
    ctx = capture_run_context(dummy_config)
    time.sleep(0.05)
    runtime = finalize_run_context(ctx)
    assert "wall_clock_sec" in runtime
    assert runtime["wall_clock_sec"] >= 0.05
    assert "ended_at_unix" in runtime


def test_write_context_json_round_trip(dummy_config, tmp_path):
    ctx = capture_run_context(dummy_config)
    path = write_context_json(ctx, str(tmp_path))
    assert Path(path).exists()
    with open(path) as f:
        loaded = json.load(f)
    assert loaded["seed"] == 42
    assert loaded["graph_version"] == ctx["graph_version"]


def test_flatten_nested_dict():
    nested = {"a": 1, "b": {"c": 2, "d": {"e": 3}}}
    flat = _flatten(nested, prefix="root")
    assert flat == {"root/a": 1, "root/b/c": 2, "root/b/d/e": 3}


def test_fails_closed_on_missing_seed():
    """A config without a seed should still capture, but seed==None is a red flag."""
    ctx = capture_run_context({"data_processing": {}})
    assert ctx["seed"] is None  # Capture succeeds but this must be flagged upstream
