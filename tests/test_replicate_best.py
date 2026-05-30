"""Tests for scripts/replicate_best.py — config-to-CLI flattening + OOM retry logic."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest


def _load_replicate_best():
    repo_root = Path(__file__).resolve().parents[1]
    spec_path = repo_root / "scripts" / "replicate_best.py"
    spec = importlib.util.spec_from_file_location("replicate_best", spec_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["replicate_best"] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def rb():
    return _load_replicate_best()


def test_config_to_cli_flat(rb):
    cfg = {"train.lr": 0.01, "seed": 42}
    args = rb.config_to_cli_args(cfg)
    # Order matches dict iteration; both keys must appear with values
    assert "--train.lr" in args
    assert "0.01" in args
    assert "--seed" in args
    assert "42" in args


def test_config_to_cli_nested(rb):
    cfg = {"train": {"lr": 0.01, "epochs": 100}, "models": {"VenGNN": {"hidden_dim": 128}}}
    args = rb.config_to_cli_args(cfg)
    assert "--train.lr" in args
    assert "--train.epochs" in args
    assert "--models.VenGNN.hidden_dim" in args


def test_find_hidden_dim_returns_value(rb):
    cli = ["--train.lr", "0.01", "--models.VenGNN.hidden_dim", "128", "--seed", "42"]
    result = rb._find_hidden_dim_in_cli(cli)
    assert result == ("--models.VenGNN.hidden_dim", 128)


def test_find_hidden_dim_none_when_missing(rb):
    cli = ["--train.lr", "0.01", "--seed", "42"]
    assert rb._find_hidden_dim_in_cli(cli) is None


def test_find_hidden_dim_skips_non_integer_values(rb):
    cli = ["--train.lr", "0.01", "--models.X.hidden_dim", "not_an_int"]
    assert rb._find_hidden_dim_in_cli(cli) is None


def test_replace_cli_value(rb):
    cli = ["--a", "1", "--b", "2", "--c", "3"]
    out = rb._replace_cli_value(cli, "--b", "99")
    assert out == ["--a", "1", "--b", "99", "--c", "3"]


def test_replace_cli_value_noop_when_missing(rb):
    cli = ["--a", "1"]
    out = rb._replace_cli_value(cli, "--missing", "99")
    assert out == cli


def test_derive_group_name_respects_override(rb):
    import argparse
    args = argparse.Namespace(group="custom", sweep_id="abc", champion_config=None)
    assert rb.derive_group_name(args, {}) == "custom"


def test_derive_group_name_from_sweep_id(rb):
    import argparse
    args = argparse.Namespace(group=None, sweep_id="abc123", champion_config=None)
    name = rb.derive_group_name(args, {})
    assert name == "replicate_abc123"


def test_derive_group_name_from_champion_path(rb, tmp_path):
    import argparse
    p = tmp_path / "vengnn_tuned.yaml"
    p.write_text("x: 1")
    args = argparse.Namespace(group=None, sweep_id=None, champion_config=p)
    name = rb.derive_group_name(args, {})
    assert name == "replicate_vengnn_tuned"


def test_load_champion_from_yaml(rb, tmp_path):
    p = tmp_path / "champ.yaml"
    p.write_text("train:\n  model: VenGNN\n  lr: 0.01\n")
    cfg = rb.load_champion_from_yaml(p)
    assert cfg["train"]["model"] == "VenGNN"
