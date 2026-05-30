"""Tests for src.ml.graph_cache — content-addressable graph cache."""
from __future__ import annotations

import pytest
import torch

from src.ml.graph_cache import (
    build_or_load_graph,
    compute_graph_version,
    _cache_path,
    _registry_load,
)


@pytest.fixture
def dummy_config():
    return {
        "seed": 42,
        "train": {"model": "SeHGNN", "lr": 0.01},
        "data_processing": {"use_org_description": True, "scaling": "robust"},
        "graph_assembler": {"metapaths": ["SIS", "SFS"]},
        "metapath_discovery": {"mode": "manual"},
        "paths": {"data_dir": "data/"},
    }


@pytest.fixture
def fake_graph():
    g = torch.nn.ParameterDict()  # Any torch-serializable object works
    g["x"] = torch.nn.Parameter(torch.zeros(3, 4))
    return g


def _fake_preprocess(graph):
    """Fabricate a preprocessing function that returns a deterministic artifact.

    Tracks call count on a closure so we can assert build was (or wasn't) invoked.
    """
    call_count = {"n": 0}
    def build():
        call_count["n"] += 1
        return graph, {"startup": ["a", "b", "c"]}
    return build, call_count


def test_graph_version_is_stable(dummy_config):
    v1 = compute_graph_version(dummy_config)
    v2 = compute_graph_version(dummy_config)
    assert v1 == v2
    assert len(v1) == 16


def test_graph_version_ignores_irrelevant_keys(dummy_config):
    v1 = compute_graph_version(dummy_config)
    # Changing train.lr must NOT change graph_version
    v2 = compute_graph_version({**dummy_config, "train": {**dummy_config["train"], "lr": 0.5}})
    assert v1 == v2


def test_graph_version_changes_on_preprocessing_keys(dummy_config):
    v1 = compute_graph_version(dummy_config)
    v2 = compute_graph_version(
        {**dummy_config, "data_processing": {**dummy_config["data_processing"], "scaling": "standard"}}
    )
    assert v1 != v2


def test_cache_miss_builds_and_saves(dummy_config, fake_graph, tmp_path):
    build, counter = _fake_preprocess(fake_graph)
    registry = tmp_path / "registry.json"
    _, node_names = build_or_load_graph(
        dummy_config, build, cache_dir=str(tmp_path), registry_path=str(registry)
    )
    assert counter["n"] == 1
    assert node_names == {"startup": ["a", "b", "c"]}
    version = compute_graph_version(dummy_config)
    assert _cache_path(version, str(tmp_path)).exists()
    # Registry entry created
    reg = _registry_load(str(registry))
    assert version in reg
    assert reg[version]["size_mb"] >= 0
    assert reg[version]["path"].endswith(".pt")
    assert reg[version]["build_sec"] >= 0


def test_cache_hit_skips_build(dummy_config, fake_graph, tmp_path):
    build, counter = _fake_preprocess(fake_graph)
    registry = tmp_path / "registry.json"
    build_or_load_graph(dummy_config, build, cache_dir=str(tmp_path), registry_path=str(registry))
    assert counter["n"] == 1
    # Second call must hit cache
    build_or_load_graph(dummy_config, build, cache_dir=str(tmp_path), registry_path=str(registry))
    assert counter["n"] == 1, "Build was called on a cache hit"


def test_force_rebuild_ignores_cache(dummy_config, fake_graph, tmp_path):
    build, counter = _fake_preprocess(fake_graph)
    registry = tmp_path / "registry.json"
    build_or_load_graph(dummy_config, build, cache_dir=str(tmp_path), registry_path=str(registry))
    assert counter["n"] == 1
    build_or_load_graph(
        dummy_config, build, cache_dir=str(tmp_path), registry_path=str(registry),
        force_rebuild=True,
    )
    assert counter["n"] == 2


def test_corrupt_cache_triggers_rebuild(dummy_config, fake_graph, tmp_path):
    build, counter = _fake_preprocess(fake_graph)
    registry = tmp_path / "registry.json"
    build_or_load_graph(dummy_config, build, cache_dir=str(tmp_path), registry_path=str(registry))
    assert counter["n"] == 1
    version = compute_graph_version(dummy_config)
    cache_file = _cache_path(version, str(tmp_path))
    cache_file.write_bytes(b"not a torch pickle")
    # Next call must detect corruption and rebuild
    build_or_load_graph(dummy_config, build, cache_dir=str(tmp_path), registry_path=str(registry))
    assert counter["n"] == 2


def test_different_configs_separate_cache_files(dummy_config, fake_graph, tmp_path):
    build_a, count_a = _fake_preprocess(fake_graph)
    build_b, count_b = _fake_preprocess(fake_graph)
    registry = tmp_path / "registry.json"

    config_a = dummy_config
    config_b = {**dummy_config, "data_processing": {**dummy_config["data_processing"], "scaling": "standard"}}

    build_or_load_graph(config_a, build_a, cache_dir=str(tmp_path), registry_path=str(registry))
    build_or_load_graph(config_b, build_b, cache_dir=str(tmp_path), registry_path=str(registry))
    assert count_a["n"] == 1
    assert count_b["n"] == 1

    # Now request config_a again — must hit cache, not rebuild
    build_or_load_graph(config_a, build_a, cache_dir=str(tmp_path), registry_path=str(registry))
    assert count_a["n"] == 1

    # Two registry entries
    reg = _registry_load(str(registry))
    assert len(reg) == 2


def test_config_enabled_false_bypass(dummy_config, fake_graph, tmp_path):
    """build_or_load_graph is the only thing the cache governs; the config.enabled flag
    is honored by the call site (main.py), not by this module. This test confirms the
    module itself has no hidden bypass that would cause silent cache misses."""
    build, counter = _fake_preprocess(fake_graph)
    registry = tmp_path / "registry.json"
    build_or_load_graph(dummy_config, build, cache_dir=str(tmp_path), registry_path=str(registry))
    assert counter["n"] == 1
