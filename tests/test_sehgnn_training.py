"""
Tests for SeHGNN model initialization and forward pass.
"""
import pytest
import torch
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.ml.models import SeHGNN, HAN, HeteroMLP, BaseGNN


class TestSeHGNNInitialization:
    """Test SeHGNN model initialization."""

    def test_sehgnn_init_basic(self, mock_hetero_graph, mini_config):
        """SeHGNN should initialize with basic configuration."""
        graph = mock_hetero_graph
        metadata = graph.metadata()

        # Get input dimensions from graph
        in_channels = {
            node_type: graph[node_type].x.shape[1]
            for node_type in graph.node_types
        }

        model = SeHGNN(
            in_channels=in_channels,
            hidden_channels=16,
            metadata=metadata,
            num_layers=1,
            heads=2,
            dropout=0.0,
            input_drop=0.0,
            att_drop=0.0,
            activation_type="relu",
            target_mode="masked_multi_task",
            num_classes=2,
            aggregation_method="mean",
            use_residual=True,
            transformer_activation="none",
            use_self_loop=True,
            config=mini_config,
            model_name="SeHGNN",
        )

        assert model is not None
        assert hasattr(model, "projectors")
        assert hasattr(model, "semantic_fusion")
        assert hasattr(model, "task_mlp")

    def test_sehgnn_metapaths_discovery(self, mock_hetero_graph, mini_config):
        """SeHGNN should discover metapaths from graph metadata."""
        graph = mock_hetero_graph
        metadata = graph.metadata()

        in_channels = {
            node_type: graph[node_type].x.shape[1]
            for node_type in graph.node_types
        }

        model = SeHGNN(
            in_channels=in_channels,
            hidden_channels=16,
            metadata=metadata,
            target_mode="masked_multi_task",
            num_classes=2,
            config=mini_config,
            model_name="SeHGNN",
        )

        # Should have discovered metapaths
        assert len(model.metapaths) > 0
        # Should include "self" if use_self_loop=True
        assert "self" in model.metapaths


class TestSeHGNNForward:
    """Test SeHGNN forward pass."""

    def test_sehgnn_forward_shape(self, mock_hetero_graph, mini_config):
        """SeHGNN forward should produce correct output shape."""
        graph = mock_hetero_graph
        metadata = graph.metadata()

        in_channels = {
            node_type: graph[node_type].x.shape[1]
            for node_type in graph.node_types
        }

        model = SeHGNN(
            in_channels=in_channels,
            hidden_channels=16,
            metadata=metadata,
            target_mode="masked_multi_task",
            num_classes=2,
            config=mini_config,
            model_name="SeHGNN",
        )

        # Prepare input
        x_dict = {node_type: graph[node_type].x for node_type in graph.node_types}
        edge_index_dict = {
            edge_type: graph[edge_type].edge_index
            for edge_type in graph.edge_types
        }

        # Forward pass
        model.eval()
        with torch.no_grad():
            output = model(x_dict, edge_index_dict)

        # Check output structure for masked_multi_task
        assert "masked_multi_task_output" in output
        assert "embedding" in output
        assert "startup" in output["masked_multi_task_output"]

        # Check output shape: [num_startups, 2] for (momentum, liquidity)
        num_startups = graph["startup"].x.shape[0]
        output_tensor = output["masked_multi_task_output"]["startup"]
        assert output_tensor.shape == (num_startups, 2)

    def test_sehgnn_forward_attention_weights(self, mock_hetero_graph, mini_config):
        """SeHGNN should return attention weights."""
        graph = mock_hetero_graph
        metadata = graph.metadata()

        in_channels = {
            node_type: graph[node_type].x.shape[1]
            for node_type in graph.node_types
        }

        model = SeHGNN(
            in_channels=in_channels,
            hidden_channels=16,
            metadata=metadata,
            target_mode="masked_multi_task",
            num_classes=2,
            config=mini_config,
            model_name="SeHGNN",
        )

        x_dict = {node_type: graph[node_type].x for node_type in graph.node_types}
        edge_index_dict = {
            edge_type: graph[edge_type].edge_index
            for edge_type in graph.edge_types
        }

        model.eval()
        with torch.no_grad():
            output = model(x_dict, edge_index_dict)

        # Should have attention weights
        assert "attention_weights" in output
        assert "metapath_names" in output

    def test_sehgnn_embedding_dimension(self, mock_hetero_graph, mini_config):
        """SeHGNN embeddings should have correct hidden_channels dimension."""
        graph = mock_hetero_graph
        metadata = graph.metadata()
        hidden_channels = 32

        in_channels = {
            node_type: graph[node_type].x.shape[1]
            for node_type in graph.node_types
        }

        model = SeHGNN(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            metadata=metadata,
            target_mode="masked_multi_task",
            num_classes=2,
            config=mini_config,
            model_name="SeHGNN",
        )

        x_dict = {node_type: graph[node_type].x for node_type in graph.node_types}
        edge_index_dict = {
            edge_type: graph[edge_type].edge_index
            for edge_type in graph.edge_types
        }

        model.eval()
        with torch.no_grad():
            output = model(x_dict, edge_index_dict)

        embedding = output["embedding"]["startup"]
        assert embedding.shape[1] == hidden_channels




class TestSeHGNNTargetModes:
    """Test SeHGNN with different target modes."""

    def test_sehgnn_binary_prediction(self, mock_hetero_graph, mini_config):
        """SeHGNN should work with binary_prediction mode."""
        graph = mock_hetero_graph
        # Adjust target for binary mode
        graph["startup"].y = torch.randint(0, 2, (20,)).float()

        metadata = graph.metadata()
        in_channels = {
            node_type: graph[node_type].x.shape[1]
            for node_type in graph.node_types
        }

        model = SeHGNN(
            in_channels=in_channels,
            hidden_channels=16,
            metadata=metadata,
            target_mode="binary_prediction",
            num_classes=2,
            config=mini_config,
            model_name="SeHGNN",
        )

        x_dict = {node_type: graph[node_type].x for node_type in graph.node_types}
        edge_index_dict = {
            edge_type: graph[edge_type].edge_index
            for edge_type in graph.edge_types
        }

        model.eval()
        with torch.no_grad():
            output = model(x_dict, edge_index_dict)

        assert "startup" in output
        assert "output" in output["startup"]
        # Binary: output should be [N, 1]
        assert output["startup"]["output"].shape[1] == 1

    def test_sehgnn_multi_prediction(self, mock_hetero_graph, mini_config):
        """SeHGNN should work with multi_prediction mode."""
        graph = mock_hetero_graph
        num_classes = 4
        graph["startup"].y = torch.randint(0, num_classes, (20,))

        metadata = graph.metadata()
        in_channels = {
            node_type: graph[node_type].x.shape[1]
            for node_type in graph.node_types
        }

        model = SeHGNN(
            in_channels=in_channels,
            hidden_channels=16,
            metadata=metadata,
            target_mode="multi_prediction",
            num_classes=num_classes,
            config=mini_config,
            model_name="SeHGNN",
        )

        x_dict = {node_type: graph[node_type].x for node_type in graph.node_types}
        edge_index_dict = {
            edge_type: graph[edge_type].edge_index
            for edge_type in graph.edge_types
        }

        model.eval()
        with torch.no_grad():
            output = model(x_dict, edge_index_dict)

        assert "startup" in output
        assert "output" in output["startup"]
        # Multi-class: output should be [N, num_classes]
        assert output["startup"]["output"].shape[1] == num_classes


class TestHANModel:
    """Test HAN model initialization and forward pass."""

    def test_han_init(self, mock_hetero_graph, mini_config):
        """HAN should initialize correctly."""
        graph = mock_hetero_graph
        metadata = graph.metadata()

        model = HAN(
            in_channels=-1,
            hidden_channels=16,
            metadata=metadata,
            num_layers=1,
            heads=2,
            negative_slope=0.2,
            dropout=0.0,
            activation_type="relu",
            target_mode="masked_multi_task",
            num_classes=2,
        )

        assert model is not None
        assert hasattr(model, "convs")
        assert len(model.convs) == 1

    def test_han_forward(self, mock_hetero_graph, mini_config):
        """HAN forward pass should work."""
        graph = mock_hetero_graph
        metadata = graph.metadata()

        model = HAN(
            in_channels=-1,
            hidden_channels=16,
            metadata=metadata,
            num_layers=1,
            heads=2,
            activation_type="relu",
            target_mode="masked_multi_task",
            num_classes=2,
        )

        x_dict = {node_type: graph[node_type].x for node_type in graph.node_types}
        edge_index_dict = {
            edge_type: graph[edge_type].edge_index
            for edge_type in graph.edge_types
        }

        model.eval()
        with torch.no_grad():
            output = model(x_dict, edge_index_dict)

        assert "masked_multi_task_output" in output
        assert output["masked_multi_task_output"]["startup"].shape[0] == 20


class TestHeteroMLP:
    """Test HeteroMLP baseline model."""

    def test_mlp_init(self, mock_hetero_graph):
        """HeteroMLP should initialize correctly."""
        model = HeteroMLP(
            hidden_channels=16,
            target_mode="masked_multi_task",
            num_classes=2,
            activation_type="relu",
            normalize=True,
            dropout=0.0,
        )

        assert model is not None

    def test_mlp_forward(self, mock_hetero_graph):
        """HeteroMLP forward pass should work."""
        graph = mock_hetero_graph

        model = HeteroMLP(
            hidden_channels=16,
            target_mode="masked_multi_task",
            num_classes=2,
            activation_type="relu",
        )

        x_dict = {node_type: graph[node_type].x for node_type in graph.node_types}
        edge_index_dict = {
            edge_type: graph[edge_type].edge_index
            for edge_type in graph.edge_types
        }

        model.eval()
        with torch.no_grad():
            output = model(x_dict, edge_index_dict)

        assert "masked_multi_task_output" in output


class TestBaseGNNHeads:
    """Test BaseGNN head initialization for different target modes."""

    def test_binary_head(self):
        """Binary prediction head should output 1 channel."""

        class TestModel(BaseGNN):
            def forward(self, x):
                return self._apply_heads(x)

        model = TestModel(hidden_channels=16, target_mode="binary_prediction", num_classes=2)
        assert hasattr(model, "output_head")

    def test_multi_prediction_head(self):
        """Multi-prediction head should output num_classes channels."""

        class TestModel(BaseGNN):
            def forward(self, x):
                return self._apply_heads(x)

        model = TestModel(hidden_channels=16, target_mode="multi_prediction", num_classes=4)
        assert hasattr(model, "output_head")

    def test_masked_multi_task_heads(self):
        """Masked multi-task should have momentum and liquidity heads."""

        class TestModel(BaseGNN):
            def forward(self, x):
                return self._apply_heads(x)

        model = TestModel(hidden_channels=16, target_mode="masked_multi_task", num_classes=2)
        assert hasattr(model, "head_momentum")
        assert hasattr(model, "head_liquidity")
