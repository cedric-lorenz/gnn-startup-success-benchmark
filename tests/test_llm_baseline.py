"""
Unit tests for LLM Baseline model.
"""
import pytest
import torch
import pandas as pd
import numpy as np


class TestParseProbability:
    """Test probability parsing from LLM responses."""

    def test_plain_decimal(self):
        from src.ml.llm_predictor import parse_probability
        assert parse_probability("0.35") == 0.35

    def test_dot_decimal(self):
        from src.ml.llm_predictor import parse_probability
        assert parse_probability(".75") == 0.75

    def test_percentage(self):
        from src.ml.llm_predictor import parse_probability
        assert parse_probability("75%") == 0.75

    def test_percentage_with_word(self):
        from src.ml.llm_predictor import parse_probability
        assert parse_probability("75 percent") == 0.75

    def test_embedded_decimal(self):
        from src.ml.llm_predictor import parse_probability
        assert parse_probability("The probability is 0.42") == 0.42

    def test_fallback_on_garbage(self):
        from src.ml.llm_predictor import parse_probability
        assert parse_probability("garbage text") == 0.5

    def test_clamps_to_zero(self):
        from src.ml.llm_predictor import parse_probability
        # Negative values should be clamped to 0
        assert parse_probability("-0.5") == 0.5  # Falls back since negative

    def test_converts_large_percentage(self):
        from src.ml.llm_predictor import parse_probability
        # Numbers > 1 are treated as percentages
        result = parse_probability("85")
        assert result == 0.85


class TestPromptBuilder:
    """Test prompt template building."""

    def test_momentum_template_contains_name(self):
        from src.ml.llm_predictor import PromptBuilder
        builder = PromptBuilder("momentum")
        features = {
            "name": "TestCorp",
            "description": "AI startup",
            "industries": "Technology",
            "city": "San Francisco",
            "founded_on_year": 2020,
            "total_funding_usd": 1000000,
            "num_funding_rounds": 2,
            "employee_count": 10,
            "founder_count": 2,
        }
        prompt = builder.build(features)
        assert "TestCorp" in prompt
        assert "funding round" in prompt.lower()

    def test_liquidity_template_contains_exit(self):
        from src.ml.llm_predictor import PromptBuilder
        builder = PromptBuilder("liquidity")
        features = {
            "name": "ExitCo",
            "description": "Fintech",
            "industries": "Finance",
            "city": "NYC",
            "founded_on_year": 2019,
            "total_funding_usd": 5000000,
            "num_funding_rounds": 3,
            "employee_count": 25,
            "founder_count": 3,
        }
        prompt = builder.build(features)
        assert "ExitCo" in prompt
        assert "exit" in prompt.lower()

    def test_handles_missing_values(self):
        from src.ml.llm_predictor import PromptBuilder
        builder = PromptBuilder("momentum")
        features = {
            "name": "PartialCo",
            # Missing most fields
        }
        prompt = builder.build(features)
        assert "PartialCo" in prompt
        assert "Unknown" in prompt  # Default value for missing fields


class TestLLMBaseline:
    """Test LLMBaseline model class."""

    def test_llm_baseline_initialization(self):
        from src.ml.models import LLMBaseline

        dummy_df = pd.DataFrame([{
            "name": "Test",
            "org_uuid": "abc-123",
            "description": "Test company",
        }])
        config = {"models": {"LLM": {"model_name": "test-model", "device": "cpu"}}}

        model = LLMBaseline(
            hidden_channels=0,
            config=config,
            raw_features_df=dummy_df,
            target_mode="masked_multi_task",
            num_classes=2,
        )

        assert model.target_mode == "masked_multi_task"
        assert model._predictor is None  # Lazy init

    def test_llm_baseline_probs_to_logits(self):
        from src.ml.models import LLMBaseline

        dummy_df = pd.DataFrame([{"name": "Test"}])
        config = {"models": {"LLM": {}}}

        model = LLMBaseline(
            hidden_channels=0,
            config=config,
            raw_features_df=dummy_df,
        )

        probs = [0.5, 0.9, 0.1]
        logits = model._probs_to_logits(probs, torch.device("cpu"))

        # Logit of 0.5 should be ~0
        assert abs(logits[0].item()) < 0.1
        # Logit of 0.9 should be positive
        assert logits[1].item() > 0
        # Logit of 0.1 should be negative
        assert logits[2].item() < 0

    def test_llm_baseline_binary_target_mode(self):
        from src.ml.models import LLMBaseline

        dummy_df = pd.DataFrame([{"name": "Test"}])
        config = {"models": {"LLM": {}}}

        model = LLMBaseline(
            hidden_channels=0,
            config=config,
            raw_features_df=dummy_df,
            target_mode="binary_prediction",
            num_classes=2,
        )

        assert model.target_mode == "binary_prediction"


class TestPredictionCache:
    """Test disk-based prediction caching."""

    def test_cache_set_and_get(self, tmp_path):
        from src.ml.llm_predictor import PredictionCache

        cache = PredictionCache(str(tmp_path))

        cache.set("model1", "momentum", "startup-123", 0.75)
        result = cache.get("model1", "momentum", "startup-123")

        assert result == 0.75

    def test_cache_returns_none_for_missing(self, tmp_path):
        from src.ml.llm_predictor import PredictionCache

        cache = PredictionCache(str(tmp_path))

        result = cache.get("model1", "liquidity", "nonexistent")
        assert result is None

    def test_cache_persistence(self, tmp_path):
        from src.ml.llm_predictor import PredictionCache

        cache1 = PredictionCache(str(tmp_path))
        cache1.set("model1", "momentum", "startup-456", 0.33)

        # Create new cache instance (simulating restart)
        cache2 = PredictionCache(str(tmp_path))
        result = cache2.get("model1", "momentum", "startup-456")

        assert result == 0.33

    def test_cache_normalizes_model_name(self, tmp_path):
        from src.ml.llm_predictor import PredictionCache

        cache = PredictionCache(str(tmp_path))

        # HuggingFace model names contain slashes
        cache.set("meta-llama/Llama-3.1-8B-Instruct", "momentum", "startup-789", 0.42)
        result = cache.get("meta-llama/Llama-3.1-8B-Instruct", "momentum", "startup-789")

        assert result == 0.42


class TestHuggingFaceClient:
    """Test HuggingFace client initialization."""

    def test_client_initialization(self):
        from src.ml.llm_predictor import HuggingFaceClient

        client = HuggingFaceClient(
            model_name="gpt2",  # Small model for testing
            device="cpu",
            torch_dtype="float32",
        )

        assert client.model_name == "gpt2"
        assert client.device == "cpu"
        assert client._model is None  # Lazy loading

    def test_client_quantization_flags(self):
        from src.ml.llm_predictor import HuggingFaceClient

        client = HuggingFaceClient(
            model_name="gpt2",
            load_in_8bit=True,
            load_in_4bit=False,
        )

        assert client.load_in_8bit is True
        assert client.load_in_4bit is False
