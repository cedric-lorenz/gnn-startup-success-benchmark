"""Tests for src.ml.checkpoint_loader.

Two tiers:
- Unit tests: exercise _deep_merge, _load_graph, _autodetect_graph_path with
  fabricated inputs. No real model/graph; always run.
- Integration test: full round-trip on a real sweep checkpoint. Skipped when
  the 1.7 GB graph or a live checkpoint isn't present (so CI and laptop runs
  don't need the HPC artifacts), but runs on the cluster to give us a real
  end-to-end safety net.
"""
from __future__ import annotations

import glob
import json
import os
from pathlib import Path

import pytest
import torch

from src.ml.checkpoint_loader import (
    _autodetect_graph_path,
    _deep_merge,
    _load_graph,
    load_trainer_from_checkpoint,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# Unit tests (fast, always run)
# ---------------------------------------------------------------------------

class TestDeepMerge:
    def test_top_level_override(self):
        base = {"a": 1, "b": 2}
        _deep_merge(base, {"a": 99})
        assert base == {"a": 99, "b": 2}

    def test_nested_override_preserves_siblings(self):
        base = {"train": {"lr": 0.01, "epochs": 100}, "models": {"X": {}}}
        _deep_merge(base, {"train": {"lr": 0.001}})
        assert base == {"train": {"lr": 0.001, "epochs": 100}, "models": {"X": {}}}

    def test_override_can_add_new_key(self):
        base = {"train": {"lr": 0.01}}
        _deep_merge(base, {"train": {"seed": 42}, "wandb": {"enabled": False}})
        assert base["train"] == {"lr": 0.01, "seed": 42}
        assert base["wandb"] == {"enabled": False}

    def test_dict_replaces_scalar(self):
        """If override is a dict but base's value is scalar, dict wins."""
        base = {"a": 1}
        _deep_merge(base, {"a": {"nested": True}})
        assert base == {"a": {"nested": True}}

    def test_scalar_replaces_dict(self):
        """If base is a dict but override is scalar, scalar wins (full replace)."""
        base = {"a": {"nested": True}}
        _deep_merge(base, {"a": 5})
        assert base == {"a": 5}


class TestLoadGraph:
    def test_loads_hetero_data_from_bare_save(self, tmp_path):
        """Legacy single-HeteroData format: torch.save(hetero_data, path)."""
        from torch_geometric.data import HeteroData
        g = HeteroData()
        g["startup"].x = torch.randn(10, 4)
        p = tmp_path / "graph.pt"
        torch.save(g, p)
        loaded = _load_graph(str(p))
        assert loaded["startup"].x.shape == (10, 4)

    def test_loads_hetero_data_from_tuple_save(self, tmp_path):
        """New cache-addressed format: torch.save((g, node_names), path)."""
        from torch_geometric.data import HeteroData
        g = HeteroData()
        g["startup"].x = torch.randn(10, 4)
        node_names = {"startup": ["s1"] * 10}
        p = tmp_path / "graph_abc.pt"
        torch.save((g, node_names), p)
        loaded = _load_graph(str(p))
        # Must unwrap the tuple transparently
        assert loaded["startup"].x.shape == (10, 4)


class TestAutodetectGraphPath:
    def test_falls_back_to_legacy_when_no_run_context(self, tmp_path, monkeypatch):
        """With no matching run_context, return the legacy default."""
        monkeypatch.chdir(tmp_path)
        # Fabricate a checkpoint path whose run_id has no run_context
        ckpt = tmp_path / "outputs" / "checkpoints" / "proj" / "nonesuch123" / "best_model.pt"
        ckpt.parent.mkdir(parents=True)
        ckpt.write_bytes(b"")  # content irrelevant for this helper
        resolved = _autodetect_graph_path(str(ckpt))
        assert resolved == "outputs/pipeline_state/graph_data.pt"

    def test_resolves_from_graph_version_hash(self, tmp_path, monkeypatch):
        """When run_context[graph_version] points at an existing cached graph,
        prefer that over the legacy path."""
        monkeypatch.chdir(tmp_path)
        run_id = "abc12345"

        # Place a run_context json at the expected location
        rc_dir = tmp_path / "outputs" / "pipeline_state" / "results"
        rc_dir.mkdir(parents=True)
        rc = {"graph_version": "deadbeefcafe", "seed": 42}
        (rc_dir / f"run_context_9999_{run_id}.json").write_text(json.dumps(rc))

        # Place the cached graph file at the address the hash maps to
        cache_dir = tmp_path / "outputs" / "graphs"
        cache_dir.mkdir(parents=True)
        (cache_dir / "graph_deadbeefcafe.pt").write_bytes(b"")

        # Fake checkpoint path matching the run_id (dirname basename = run_id)
        ckpt = tmp_path / "outputs" / "checkpoints" / "proj" / run_id / "best_model.pt"
        ckpt.parent.mkdir(parents=True)
        ckpt.write_bytes(b"")

        resolved = _autodetect_graph_path(str(ckpt))
        assert resolved == "outputs/graphs/graph_deadbeefcafe.pt"

    def test_prefers_most_recent_run_context_when_multiple(self, tmp_path, monkeypatch):
        """If a trial was resumed (two run_context files for the same run_id),
        pick the most-recently-modified one."""
        monkeypatch.chdir(tmp_path)
        run_id = "resume99"
        rc_dir = tmp_path / "outputs" / "pipeline_state" / "results"
        rc_dir.mkdir(parents=True)
        cache_dir = tmp_path / "outputs" / "graphs"
        cache_dir.mkdir(parents=True)

        # First run_context: older hash
        old = rc_dir / f"run_context_1111_{run_id}.json"
        old.write_text(json.dumps({"graph_version": "oldhash0000"}))
        (cache_dir / "graph_oldhash0000.pt").write_bytes(b"")

        # Second run_context: newer hash (mtime bumped via os.utime)
        new = rc_dir / f"run_context_2222_{run_id}.json"
        new.write_text(json.dumps({"graph_version": "newhash1111"}))
        (cache_dir / "graph_newhash1111.pt").write_bytes(b"")
        # Force newer mtime on `new`
        os.utime(str(old), (1_700_000_000, 1_700_000_000))
        os.utime(str(new), (1_800_000_000, 1_800_000_000))

        ckpt = tmp_path / "outputs" / "checkpoints" / "proj" / run_id / "best_model.pt"
        ckpt.parent.mkdir(parents=True)
        ckpt.write_bytes(b"")
        resolved = _autodetect_graph_path(str(ckpt))
        assert resolved == "outputs/graphs/graph_newhash1111.pt"


class TestLoadTrainerFromCheckpointErrors:
    def test_raises_on_missing_checkpoint(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="Checkpoint not found"):
            load_trainer_from_checkpoint(str(tmp_path / "nope.pt"))

    def test_raises_on_checkpoint_without_embedded_config(self, tmp_path):
        """Older checkpoints that don't carry `config` must fail loudly
        (not silently fall through to the stale config.yaml)."""
        p = tmp_path / "legacy.pt"
        torch.save({"model_state_dict": {}}, p)
        with pytest.raises(ValueError, match="no embedded `config` key"):
            load_trainer_from_checkpoint(str(p))


# ---------------------------------------------------------------------------
# Integration test (skipped when HPC artifacts absent)
# ---------------------------------------------------------------------------

def _find_live_checkpoint():
    """Return a (checkpoint_path, expected_model) tuple from a finished run
    we have evidence for, or None if nothing suitable exists locally.

    Must work while sweeps are actively writing new checkpoints: skip files
    that were modified in the last 30 seconds (likely mid-write) and
    resiliently skip any that fail to load."""
    import time
    now = time.time()
    candidates = sorted(
        glob.glob("outputs/checkpoints/gnn-startup-successs/*/best_model.pt"),
        key=os.path.getmtime,
        reverse=True,
    )
    if not candidates:
        return None
    for ckpt in candidates[:20]:
        # Skip files actively being written (race with running agents)
        if now - os.path.getmtime(ckpt) < 30:
            continue
        try:
            probe = torch.load(ckpt, weights_only=False, map_location="cpu")
        except Exception:
            continue
        if "config" in probe and "model_state_dict" in probe:
            model_name = probe["config"].get("train", {}).get("model")
            if model_name:
                return ckpt, model_name
    return None


@pytest.mark.skipif(
    _find_live_checkpoint() is None
    or not os.path.exists("outputs/pipeline_state/graph_data.pt"),
    reason="Integration test requires a real sweep checkpoint + graph on disk",
)
class TestCheckpointLoaderIntegration:
    """End-to-end: load a live sweep checkpoint, run a forward pass, sanity-
    check the outputs. Catches regressions that would silently break post-hoc
    analysis without any individual mock being wrong."""

    def test_round_trip_load_and_forward(self):
        found = _find_live_checkpoint()
        assert found is not None
        ckpt_path, expected_model = found

        trainer = load_trainer_from_checkpoint(ckpt_path)

        # Basic structure. Compare by user-facing model name (e.g. "MLP")
        # because Trainer renames some of them internally (MLP -> HeteroMLP,
        # LLM -> LLMBaseline, XGBoost -> XGBoostAdapter).
        assert trainer.model_name == expected_model, (
            f"Reinstantiated model_name ({trainer.model_name}) "
            f"does not match the training-time model ({expected_model})"
        )
        assert trainer.epochs == 0, "Post-hoc load must not allow accidental training"
        assert trainer.use_wandb is False, "Post-hoc load must not open wandb"

        # Forward pass (on whichever device Trainer picked)
        trainer.model.eval()
        with torch.no_grad():
            out = trainer.model(trainer.data.x_dict, trainer.data.edge_index_dict)
        # Outputs should be either a dict (MTL) or a single tensor
        if isinstance(out, dict):
            assert len(out) >= 1
            for k, v in out.items():
                if torch.is_tensor(v):
                    assert torch.isfinite(v).all(), f"NaN/Inf in output '{k}'"
                    # Number of startup nodes matches the graph
                    assert v.shape[0] == trainer.data["startup"].num_nodes
        else:
            assert torch.is_tensor(out)
            assert torch.isfinite(out).all()

    def test_overrides_are_deep_merged(self):
        """Overrides apply cleanly without replacing whole sub-dicts."""
        found = _find_live_checkpoint()
        assert found is not None
        ckpt_path, _ = found

        trainer = load_trainer_from_checkpoint(
            ckpt_path,
            config_overrides={"train": {"epochs": 5}},
        )
        # Override wins
        assert trainer.config["train"]["epochs"] == 5
        # Siblings under train preserved (lr, model, etc.)
        assert trainer.config["train"].get("model") is not None
        assert trainer.config["train"].get("lr") is not None

    def test_explicit_graph_path_is_honored(self):
        """When caller supplies graph_path explicitly, bypass the autodetect
        and use exactly that file."""
        found = _find_live_checkpoint()
        assert found is not None
        ckpt_path, _ = found
        trainer = load_trainer_from_checkpoint(
            ckpt_path,
            graph_path="outputs/pipeline_state/graph_data.pt",
        )
        assert trainer.data is not None
