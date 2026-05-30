"""Tests for metrics_export.py"""
import pytest
import json
import os

import numpy as np


# =============================================================================
# Tests for metrics_export.py
# =============================================================================


class TestExportMetricsJson:
    def test_creates_json_file(self, tmp_path):
        from src.ml.metrics_export import export_metrics_json

        metrics = {"val_auc_roc_binary": 0.85, "val_auc_pr_binary": 0.72}
        config = {"train": {"model": "HAN", "lr": 0.01}}

        path = export_metrics_json(
            metrics=metrics, config=config, mode="val", epoch=50,
            model_name="HAN", target_mode="binary_prediction",
            output_base_dir=str(tmp_path),
        )

        assert os.path.exists(path)
        with open(path) as f:
            data = json.load(f)
        assert data["metadata"]["model"] == "HAN"
        assert data["metadata"]["mode"] == "val"
        assert data["metadata"]["epoch"] == 50
        assert data["metrics"]["val_auc_roc_binary"] == 0.85

    def test_directory_structure(self, tmp_path):
        from src.ml.metrics_export import export_metrics_json

        path = export_metrics_json(
            metrics={"m": 1.0}, config={}, mode="test", epoch=1,
            model_name="SeHGNN", target_mode="masked_multi_task",
            output_base_dir=str(tmp_path),
        )

        assert "SeHGNN" in path
        assert "masked_multi_task" in path
        assert path.endswith("_test.json")

    def test_handles_numpy_types(self, tmp_path):
        from src.ml.metrics_export import export_metrics_json

        metrics = {
            "val_auc_roc": np.float64(0.85),
            "val_count": np.int64(100),
            "val_array": np.array([1.0, 2.0]),
            "val_bool": np.bool_(True),
        }
        config = {"train": {"model": "MLP"}}

        path = export_metrics_json(
            metrics=metrics, config=config, mode="val", epoch=10,
            model_name="MLP", target_mode="binary_prediction",
            output_base_dir=str(tmp_path),
        )

        with open(path) as f:
            data = json.load(f)  # Should not raise
        assert isinstance(data["metrics"]["val_auc_roc"], float)
        assert isinstance(data["metrics"]["val_count"], int)
        assert isinstance(data["metrics"]["val_array"], list)
        assert isinstance(data["metrics"]["val_bool"], bool)

    def test_stores_best_metric(self, tmp_path):
        from src.ml.metrics_export import export_metrics_json

        path = export_metrics_json(
            metrics={"m": 0.5}, config={}, mode="val", epoch=42,
            model_name="HAN", target_mode="binary_prediction",
            best_metric=0.88, best_epoch=30,
            output_base_dir=str(tmp_path),
        )

        with open(path) as f:
            data = json.load(f)
        assert data["metadata"]["best_metric"] == 0.88
        assert data["metadata"]["best_epoch"] == 30

    def test_config_is_serialized(self, tmp_path):
        from src.ml.metrics_export import export_metrics_json

        config = {
            "train": {"lr": 0.001, "epochs": 100, "model": "SeHGNN"},
            "models": {"SeHGNN": {"num_layers": 3, "hidden_channels": 64}},
            "data_processing": {"target_mode": "masked_multi_task"},
        }

        path = export_metrics_json(
            metrics={"m": 0.5}, config=config, mode="val", epoch=1,
            model_name="SeHGNN", target_mode="masked_multi_task",
            output_base_dir=str(tmp_path),
        )

        with open(path) as f:
            data = json.load(f)
        assert data["config"]["train"]["lr"] == 0.001
        assert data["config"]["models"]["SeHGNN"]["num_layers"] == 3


class TestMakeJsonSerializable:
    def test_nested_numpy(self):
        from src.ml.metrics_export import _make_json_serializable

        obj = {"a": np.float64(1.0), "b": {"c": np.int64(2)}}
        result = _make_json_serializable(obj)

        assert isinstance(result["a"], float)
        assert isinstance(result["b"]["c"], int)

    def test_numpy_array(self):
        from src.ml.metrics_export import _make_json_serializable

        result = _make_json_serializable(np.array([1, 2, 3]))
        assert result == [1, 2, 3]

    def test_plain_types_unchanged(self):
        from src.ml.metrics_export import _make_json_serializable

        obj = {"a": 1, "b": "hello", "c": True, "d": None}
        result = _make_json_serializable(obj)
        assert result == obj


# =============================================================================
# Tests for error capture (metrics_export.save_error_report)
# =============================================================================


class TestSaveErrorReport:
    def test_creates_error_json(self, tmp_path):
        from src.ml.metrics_export import save_error_report

        config = {
            "train": {"model": "HAN", "lr": 0.01},
            "data_processing": {"target_mode": "binary_prediction"},
        }

        try:
            raise ValueError("CUDA out of memory")
        except ValueError as e:
            path = save_error_report(
                error=e, config=config, output_base_dir=str(tmp_path),
            )

        assert os.path.exists(path)
        assert path.endswith("_error.json")
        with open(path) as f:
            data = json.load(f)
        assert data["metadata"]["status"] == "failed"
        assert data["metadata"]["error_type"] == "ValueError"
        assert "CUDA out of memory" in data["metadata"]["error_message"]

    def test_error_directory_structure(self, tmp_path):
        from src.ml.metrics_export import save_error_report

        config = {
            "train": {"model": "SeHGNN"},
            "data_processing": {"target_mode": "masked_multi_task"},
        }

        try:
            raise RuntimeError("test error")
        except RuntimeError as e:
            path = save_error_report(
                error=e, config=config, output_base_dir=str(tmp_path),
            )

        assert "SeHGNN" in path
        assert "masked_multi_task" in path

    def test_traceback_is_captured(self, tmp_path):
        from src.ml.metrics_export import save_error_report

        config = {"train": {"model": "HAN"}, "data_processing": {"target_mode": "binary_prediction"}}

        def inner_function():
            raise KeyError("missing_key")

        try:
            inner_function()
        except KeyError as e:
            path = save_error_report(
                error=e, config=config, output_base_dir=str(tmp_path),
            )

        with open(path) as f:
            data = json.load(f)
        tb = data["metadata"]["traceback"]
        assert isinstance(tb, list)
        assert any("inner_function" in line for line in tb)

    def test_config_is_saved(self, tmp_path):
        from src.ml.metrics_export import save_error_report

        config = {
            "train": {"model": "HAN", "lr": 0.001, "epochs": 100},
            "data_processing": {"target_mode": "binary_prediction"},
        }

        try:
            raise ValueError("test")
        except ValueError as e:
            path = save_error_report(
                error=e, config=config, output_base_dir=str(tmp_path),
            )

        with open(path) as f:
            data = json.load(f)
        assert data["config"]["train"]["lr"] == 0.001
        assert data["config"]["train"]["epochs"] == 100

    def test_missing_config_keys_default_to_unknown(self, tmp_path):
        from src.ml.metrics_export import save_error_report

        try:
            raise ValueError("test")
        except ValueError as e:
            path = save_error_report(
                error=e, config={}, output_base_dir=str(tmp_path),
            )

        assert "unknown" in path  # model defaults to "unknown"
