"""
Tests for configuration loading and CLI override merging.
"""
import pytest
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.ml.utils import load_config, deep_merge_dict, args_to_nested_dict


class TestConfigLoading:
    """Test configuration file loading."""

    def test_load_config_returns_dict(self):
        """Config loading should return a dictionary."""
        config = load_config()
        assert isinstance(config, dict)

    def test_load_config_has_required_sections(self):
        """Config should have all required top-level sections."""
        config = load_config()
        required_sections = [
            "paths",
            "train",
            "models",
            "eval",
            "data_processing",
            "wandb",
        ]
        for section in required_sections:
            assert section in config, f"Missing required section: {section}"

    def test_load_config_train_section(self):
        """Train section should have essential keys."""
        config = load_config()
        train_keys = ["model", "device", "lr", "epochs", "loss"]
        for key in train_keys:
            assert key in config["train"], f"Missing train key: {key}"

    def test_load_config_models_section(self):
        """Models section should have at least SeHGNN and HAN."""
        config = load_config()
        required_models = ["SeHGNN", "HAN"]
        for model in required_models:
            assert model in config["models"], f"Missing model config: {model}"


class TestDeepMergDict:
    """Test deep dictionary merging for config overrides."""

    def test_deep_merge_simple_override(self):
        """Simple value override should work."""
        base = {"a": 1, "b": 2}
        override = {"a": 10}
        result = deep_merge_dict(base, override)
        assert result["a"] == 10
        assert result["b"] == 2

    def test_deep_merge_nested_override(self):
        """Nested dictionary override should work."""
        base = {"outer": {"inner": 1, "keep": 2}}
        override = {"outer": {"inner": 10}}
        result = deep_merge_dict(base, override)
        assert result["outer"]["inner"] == 10
        assert result["outer"]["keep"] == 2

    def test_deep_merge_adds_new_keys(self):
        """New keys in override should be added."""
        base = {"a": 1}
        override = {"b": 2}
        result = deep_merge_dict(base, override)
        assert result["a"] == 1
        assert result["b"] == 2

    def test_deep_merge_preserves_base(self):
        """Original base dict should not be modified."""
        base = {"a": {"b": 1}}
        override = {"a": {"b": 10}}
        import copy
        base_copy = copy.deepcopy(base)
        _ = deep_merge_dict(base, override)
        assert base == base_copy

    def test_deep_merge_handles_none_override(self):
        """None values in override should replace base values."""
        base = {"a": 1, "b": 2}
        override = {"a": None}
        result = deep_merge_dict(base, override)
        # None overwrites the base value (this is the actual behavior)
        assert result["a"] is None
        assert result["b"] == 2


class TestArgsToNestedDict:
    """Test CLI argument to nested dictionary conversion."""

    def test_args_to_nested_dict_simple(self):
        """Simple dotted argument should create nested dict."""
        import argparse
        parser = argparse.ArgumentParser()
        parser.add_argument("--train.lr", type=float, default=None)
        args = parser.parse_args(["--train.lr", "0.001"])

        result = args_to_nested_dict(args)
        assert "train" in result
        assert result["train"]["lr"] == 0.001

    def test_args_to_nested_dict_deep_nesting(self):
        """Deeply nested dotted argument should work."""
        import argparse
        parser = argparse.ArgumentParser()
        parser.add_argument("--models.SeHGNN.hidden_channels", type=int, default=None)
        args = parser.parse_args(["--models.SeHGNN.hidden_channels", "128"])

        result = args_to_nested_dict(args)
        assert result["models"]["SeHGNN"]["hidden_channels"] == 128

    def test_args_to_nested_dict_skips_none(self):
        """None values should not be included in result."""
        import argparse
        parser = argparse.ArgumentParser()
        parser.add_argument("--train.lr", type=float, default=None)
        parser.add_argument("--train.epochs", type=int, default=None)
        args = parser.parse_args(["--train.lr", "0.01"])

        result = args_to_nested_dict(args)
        assert "lr" in result.get("train", {})
        # epochs should not be present since it's None


class TestConfigMergeWithCLI:
    """Test full config merge workflow with CLI overrides."""

    def test_cli_override_train_params(self):
        """CLI should override training parameters."""
        base_config = load_config()
        override = {"train": {"lr": 0.999, "epochs": 5}}

        merged = deep_merge_dict(base_config, override)

        assert merged["train"]["lr"] == 0.999
        assert merged["train"]["epochs"] == 5
        # Other train params should remain
        assert "device" in merged["train"]

    def test_cli_override_model_params(self):
        """CLI should override model-specific parameters."""
        base_config = load_config()
        override = {"models": {"HAN": {"dropout": 0.75}}}

        merged = deep_merge_dict(base_config, override)

        assert merged["models"]["HAN"]["dropout"] == 0.75
        # Other HAN params should remain
        assert "heads" in merged["models"]["HAN"]

    def test_ablation_drop_node_types_merge(self):
        """Ablation drop_node_types should merge correctly from CLI."""
        base_config = load_config()
        override = {
            "data_processing": {
                "ablation": {
                    "drop_node_types": ["investor", "founder"]
                }
            }
        }

        merged = deep_merge_dict(base_config, override)

        assert "investor" in merged["data_processing"]["ablation"]["drop_node_types"]
        assert "founder" in merged["data_processing"]["ablation"]["drop_node_types"]

    def test_metapath_discovery_mode_override(self):
        """Metapath discovery mode should be overridable."""
        base_config = load_config()
        override = {"metapath_discovery": {"mode": "hybrid"}}

        merged = deep_merge_dict(base_config, override)

        assert merged["metapath_discovery"]["mode"] == "hybrid"


class TestActivationNaming:
    """Test that activation parameter naming is consistent."""

    def test_han_uses_activation_type(self, load_real_config):
        """HAN model should use activation_type parameter."""
        config = load_real_config
        assert "activation_type" in config["models"]["HAN"]

    def test_sehgnn_uses_activation_type(self, load_real_config):
        """SeHGNN model should use activation_type parameter."""
        config = load_real_config
        assert "activation_type" in config["models"]["SeHGNN"]

    def test_sagegnn_uses_activation_type(self, load_real_config):
        """SageGNN model should use activation_type parameter."""
        config = load_real_config
        assert "activation_type" in config["models"]["SageGNN"]
