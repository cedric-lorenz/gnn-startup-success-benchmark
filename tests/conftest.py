"""
Pytest fixtures for GNN Startup Success tests.

Provides minimal configurations and mock graph data for fast, isolated tests.
"""
import pytest
import torch
from torch_geometric.data import HeteroData
import sys
from pathlib import Path
import yaml
import copy

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


@pytest.fixture
def mini_config():
    """
    Minimal configuration for quick tests.
    Overrides expensive settings (epochs, wandb, etc.) for fast execution.
    """
    return {
        "seed": 42,
        "paths": {
            "data_dir": "data",
            "crunchbase_dir": "data/crunchbase",
            "graph_dir": "data/graph",
            "other_dir": "data/other",
        },
        "visualize": {"enabled": False},
        "wandb": {"enabled": False, "project": "test", "use_sweep": False},
        "analysis": {
            "enable_homophily_analysis": False,
            "enable_downstream_analysis": False,
        },
        "features": {
            "use_non_target_metapaths": False,
        },
        "metapath_discovery": {
            "mode": "manual",
            "automatic": {
                "max_hops": 2,
                "min_edges": 100,
                "target_node": "startup",
                "prune_self_loops": True,
                "prune_redundant": True,
                "prune_top_k": 5,
                "exclude_node_types": [],
                "max_metapaths": 5,
                "selection_strategy": "least_edges",
                "materialize_method": "composition",
                "ablation": {"drop_edges": []},
            },
            "manual": {"use_existing_definitions": True},
        },
        "train": {
            "use_gnn": True,
            "model": "SeHGNN",
            "device": "cpu",
            "lr": 0.01,
            "weight_decay": 0.0001,
            "epochs": 2,  # Minimal epochs for testing
            "aggregation_method": "sum",
            "gradient_clip_val": 1.0,
            "scheduler": {"type": "None"},
            "loss": {
                "binary_loss_weight": 0.5,
                "momentum_weight": 1.0,
                "liquidity_weight": 1.0,
                "retrieval_weight": 0.0,
                "retrieval_loss_type": "contrastive",
                "use_class_weights": False,
                "use_focal_loss": False,
                "focal_alpha": 0.25,
                "focal_gamma": 2.0,
                "label_smoothing": 0.0,
                "arcface": {"margin": 0.5, "scale": 64.0, "class_column": "industry_groups"},
                "contrastive_positive_source": "label",
                "contrastive_margin": 0.5,
                "contrastive_negatives_per_anchor": 2,
            },
        },
        "models": {
            "SeHGNN": {
                "in_channels": -1,
                "hidden_channels": 16,
                "num_layers": 1,
                "heads": 2,
                "input_drop": 0.0,
                "att_drop": 0.0,
                "activation_type": "relu",
                "use_residual": True,
                "transformer_activation": "none",
                "aggregation_method": "mean",
                "use_self_loop": True,
                "attention_temperature": 1.0,
                "use_retrieval_head": False,
                "detach_retrieval_head": False,
            },
            "HAN": {
                "in_channels": -1,
                "hidden_channels": 16,
                "num_layers": 1,
                "heads": 2,
                "negative_slope": 0.2,
                "dropout": 0.0,
                "activation_type": "relu",
            },
            "MLP": {
                "hidden_channels": 16,
                "num_layers": 1,
                "dropout": 0.0,
                "activation": "relu",
                "normalize": True,
            },
            "GAT": {
                "in_channels": -1,
                "hidden_channels": 16,
                "out_channels": 2,
                "num_layers": 1,
                "v2": True,
                "normalize": True,
                "activation": "relu",
                "jumping_knowledge": "cat",
                "add_self_loops": False,
                "dropout": 0.0,
                "heads": 2,
                "negative_slope": 0.2,
            },
            "GCN": {
                "in_channels": -1,
                "hidden_channels": 16,
                "out_channels": 2,
                "num_layers": 1,
                "normalize": True,
                "activation": "relu",
                "jumping_knowledge": "cat",
                "dropout": 0.0,
                "add_self_loops": False,
            },
        },
        "eval": {
            "export_predictions": False,
            "test_best_model": False,
            "optimization_metric_type": "auc_pr",
            "min_amount_of_epochs": 1,
            "early_stopping": {"enabled": False, "patience": 5, "min_delta": 0.001},
        },
        "explain": {"enabled": False},
        "calibration": {"enabled": False},
        "data_processing": {
            "start_date": "2014-01-01",
            "end_date": "2025-12-31",
            "ablation": {"drop_edges": [], "drop_node_types": [], "drop_feature_groups": []},
            "nan_filtering": {"enabled": False},
            "target_mode": "masked_multi_task",
            "multi_column": "future_status",
            "binary_column": "acq_ipo_funding",
            "remove_operating": False,
            "old_year": 2023,
            "new_year": 2025,
            "keep_dc_status": True,
            "use_org_description": False,
            "use_people_description": False,
            "visualize_embeddings": False,
            "description_embedding_dim": 16,
            "use_centrality_features": False,
            "structural_only": False,
            "resample": {"enabled": False},
            "train": {"ratio": 0.6, "use_batches": False},
            "val": {"ratio": 0.2, "use_batches": False},
            "test": {"ratio": 0.2, "use_batches": False},
            "add_metapaths": False,
            "add_self_loops": True,
            "edge_loading": {
                "founder_investor_employment": False,
                "founder_coworking": False,
                "founder_investor_identity": False,
                "founder_co_study": False,
                "founder_role_edges": False,
            },
        },
        "imputation": {
            "enabled": False,
            "node_types": {
                "startup": {"numerical_method": "zero", "categorical_method": "most_frequent"},
                "default": {"numerical_method": "median", "categorical_method": "most_frequent"},
            },
        },
    }


@pytest.fixture
def mock_hetero_graph():
    """
    Create a minimal heterogeneous graph for testing.

    Graph structure:
    - 20 startup nodes (target)
    - 10 investor nodes
    - 10 founder nodes
    - Edges: startup-investor, startup-founder
    """
    data = HeteroData()

    # Node features (random)
    num_startups = 20
    num_investors = 10
    num_founders = 10
    feature_dim = 8

    data["startup"].x = torch.randn(num_startups, feature_dim)
    data["investor"].x = torch.randn(num_investors, feature_dim)
    data["founder"].x = torch.randn(num_founders, feature_dim)

    # Targets for masked_multi_task: [momentum, liquidity, momentum_mask, liquidity_mask, retrieval_class]
    # Shape: [N, 5]
    targets = torch.zeros(num_startups, 5)
    targets[:10, 0] = 1  # Half have momentum (funding)
    targets[5:15, 1] = 1  # Some have liquidity (exit)
    targets[:, 2] = 1  # All have momentum mask
    targets[:, 3] = 1  # All have liquidity mask
    targets[:, 4] = torch.randint(0, 5, (num_startups,))  # Retrieval classes
    data["startup"].y = targets

    # Masks
    data["startup"].train_mask = torch.zeros(num_startups, dtype=torch.bool)
    data["startup"].train_mask[:12] = True
    data["startup"].val_mask = torch.zeros(num_startups, dtype=torch.bool)
    data["startup"].val_mask[12:16] = True
    data["startup"].test_mask = torch.zeros(num_startups, dtype=torch.bool)
    data["startup"].test_mask[16:] = True

    # Edges: startup -> investor (funded_by)
    src_si = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    dst_si = torch.tensor([0, 1, 2, 3, 4, 0, 1, 2, 3, 4])
    data["startup", "funded_by", "investor"].edge_index = torch.stack([src_si, dst_si])

    # Edges: investor -> startup (reverse)
    data["investor", "rev_funded_by", "startup"].edge_index = torch.stack([dst_si, src_si])

    # Edges: startup -> founder (has_founder)
    src_sf = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])
    dst_sf = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1])
    data["startup", "has_founder", "founder"].edge_index = torch.stack([src_sf, dst_sf])

    # Edges: founder -> startup (reverse)
    data["founder", "rev_has_founder", "startup"].edge_index = torch.stack([dst_sf, src_sf])

    return data


@pytest.fixture
def mock_predictions():
    """
    Mock predictions for downstream analysis testing.

    Returns list of (uuid, score, label) tuples.
    """
    import random
    random.seed(42)

    predictions = []
    for i in range(100):
        uuid = f"mock-uuid-{i:04d}"
        score = random.random()  # Random probability
        label = 1 if score > 0.5 else 0  # Simple threshold-based label
        predictions.append((uuid, score, label))

    return predictions


@pytest.fixture
def mock_multi_task_predictions():
    """
    Mock predictions for masked multi-task downstream analysis.

    Returns list of (uuid, score_dict, label_dict) tuples.
    """
    import random
    random.seed(42)

    predictions = []
    for i in range(100):
        uuid = f"mock-uuid-{i:04d}"
        score_dict = {
            "mom": random.random(),
            "liq": random.random(),
        }
        label_dict = {
            "mom": 1 if score_dict["mom"] > 0.5 else 0,
            "liq": 1 if score_dict["liq"] > 0.5 else 0,
        }
        predictions.append((uuid, score_dict, label_dict))

    return predictions


@pytest.fixture
def base_config_path():
    """Path to the base config.yaml file."""
    return project_root / "config.yaml"


@pytest.fixture
def load_real_config(base_config_path):
    """Load the actual config.yaml for integration tests."""
    with open(base_config_path, "r") as f:
        return yaml.safe_load(f)


@pytest.fixture
def mock_graph_with_proper_masks():
    """
    Create a heterogeneous graph with guaranteed mutually exclusive masks.

    Ensures:
    - train, val, test masks are mutually exclusive (no overlap)
    - masks cover all startup nodes completely
    - Proper split ratios (60/20/20)
    """
    import numpy as np
    data = HeteroData()

    num_startups = 100
    num_investors = 30
    num_founders = 40
    feature_dim = 16

    # Node features
    data["startup"].x = torch.randn(num_startups, feature_dim)
    data["investor"].x = torch.randn(num_investors, feature_dim)
    data["founder"].x = torch.randn(num_founders, feature_dim)

    # Create mutually exclusive masks with deterministic assignment
    np.random.seed(42)
    indices = np.arange(num_startups)
    np.random.shuffle(indices)

    train_size = int(num_startups * 0.6)
    val_size = int(num_startups * 0.2)

    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size + val_size]
    test_indices = indices[train_size + val_size:]

    data["startup"].train_mask = torch.zeros(num_startups, dtype=torch.bool)
    data["startup"].val_mask = torch.zeros(num_startups, dtype=torch.bool)
    data["startup"].test_mask = torch.zeros(num_startups, dtype=torch.bool)

    data["startup"].train_mask[train_indices] = True
    data["startup"].val_mask[val_indices] = True
    data["startup"].test_mask[test_indices] = True

    # Targets for masked_multi_task: [momentum, liquidity, momentum_mask, liquidity_mask, retrieval_class]
    targets = torch.zeros(num_startups, 5)
    targets[:50, 0] = 1  # Half have momentum
    targets[25:75, 1] = 1  # Some have liquidity
    targets[:, 2] = 1  # All have momentum mask
    targets[:, 3] = 1  # All have liquidity mask
    targets[:, 4] = torch.randint(0, 10, (num_startups,))
    data["startup"].y = targets

    # Edges
    src_si = torch.randint(0, num_startups, (50,))
    dst_si = torch.randint(0, num_investors, (50,))
    data["startup", "funded_by", "investor"].edge_index = torch.stack([src_si, dst_si])
    data["investor", "rev_funded_by", "startup"].edge_index = torch.stack([dst_si, src_si])

    src_sf = torch.randint(0, num_startups, (60,))
    dst_sf = torch.randint(0, num_founders, (60,))
    data["startup", "has_founder", "founder"].edge_index = torch.stack([src_sf, dst_sf])
    data["founder", "rev_has_founder", "startup"].edge_index = torch.stack([dst_sf, src_sf])

    return data


@pytest.fixture
def mock_continuous_predictions():
    """
    Mock predictions with continuous probabilities in [0, 1].

    Returns dict with y_true (binary labels) and y_prob (continuous probabilities).
    """
    import numpy as np
    np.random.seed(42)

    n_samples = 200

    # Generate ground truth with ~30% positive rate
    y_true = (np.random.rand(n_samples) < 0.3).astype(int)

    # Generate calibrated-ish probabilities (correlated with labels)
    y_prob = np.zeros(n_samples)
    for i in range(n_samples):
        if y_true[i] == 1:
            y_prob[i] = np.clip(np.random.beta(3, 2), 0, 1)  # Higher probs for positives
        else:
            y_prob[i] = np.clip(np.random.beta(2, 3), 0, 1)  # Lower probs for negatives

    return {
        "y_true": y_true,
        "y_prob": y_prob,
        "n_samples": n_samples
    }


@pytest.fixture
def mock_logits():
    """
    Mock raw model outputs (logits, pre-sigmoid) for loss function testing.

    Returns dict with logits, targets, and expected sigmoid outputs.
    """
    import numpy as np

    n_samples = 100

    # Raw logits (can be any real number)
    logits = torch.randn(n_samples) * 2  # Scale for variety

    # Binary targets
    targets = torch.randint(0, 2, (n_samples,)).float()

    # Expected probabilities after sigmoid
    expected_probs = torch.sigmoid(logits)

    return {
        "logits": logits,
        "targets": targets,
        "expected_probs": expected_probs,
        "n_samples": n_samples
    }


@pytest.fixture
def mock_masked_multi_task_targets():
    """
    Mock targets for masked multi-task learning.

    Shape: [N, 4] with columns [momentum, liquidity, mask_mom, mask_liq]
    Some samples have masks set to 0 (invalid for training).
    """
    n_samples = 100

    # Targets tensor
    targets = torch.zeros(n_samples, 4)

    # Momentum labels (binary)
    targets[:, 0] = torch.randint(0, 2, (n_samples,)).float()

    # Liquidity labels (binary)
    targets[:, 1] = torch.randint(0, 2, (n_samples,)).float()

    # Momentum mask: 80% valid
    targets[:, 2] = (torch.rand(n_samples) < 0.8).float()

    # Liquidity mask: 50% valid (stricter maturity requirement)
    targets[:, 3] = (torch.rand(n_samples) < 0.5).float()

    return {
        "targets": targets,
        "n_samples": n_samples,
        "momentum_labels": targets[:, 0],
        "liquidity_labels": targets[:, 1],
        "momentum_mask": targets[:, 2],
        "liquidity_mask": targets[:, 3],
        "n_valid_momentum": int(targets[:, 2].sum()),
        "n_valid_liquidity": int(targets[:, 3].sum())
    }


@pytest.fixture
def mock_sparse_adjacency_pair():
    """
    Mock sparse adjacency matrices for metapath composition testing.

    Creates two compatible sparse matrices that can be composed via matrix multiplication.
    A: startup -> investor (10 startups, 5 investors)
    B: investor -> founder (5 investors, 8 founders)
    Composed: startup -> founder (10 startups, 8 founders)
    """
    import scipy.sparse as sp
    import numpy as np

    np.random.seed(42)

    n_startups = 10
    n_investors = 5
    n_founders = 8

    # Create random sparse edges
    # A: startup -> investor
    a_edges = [(np.random.randint(0, n_startups), np.random.randint(0, n_investors))
               for _ in range(15)]
    a_rows, a_cols = zip(*a_edges)
    A = sp.csr_matrix(
        (np.ones(len(a_edges)), (a_rows, a_cols)),
        shape=(n_startups, n_investors)
    )

    # B: investor -> founder
    b_edges = [(np.random.randint(0, n_investors), np.random.randint(0, n_founders))
               for _ in range(12)]
    b_rows, b_cols = zip(*b_edges)
    B = sp.csr_matrix(
        (np.ones(len(b_edges)), (b_rows, b_cols)),
        shape=(n_investors, n_founders)
    )

    # Expected composition: A @ B (startup -> founder)
    C = A @ B
    C.data = np.ones_like(C.data)  # Binarize

    return {
        "A": A,
        "B": B,
        "composed": C,
        "n_startups": n_startups,
        "n_investors": n_investors,
        "n_founders": n_founders,
        "a_edge_count": len(a_edges),
        "b_edge_count": len(b_edges),
        "composed_edge_count": C.nnz
    }


@pytest.fixture
def mock_early_stopping_state():
    """
    Mock state for early stopping tests.

    Provides metric history and expected early stopping behavior.
    """
    # Metric history that should trigger early stopping
    # Pattern: improves, plateaus, then no improvement
    metric_history = [
        0.5,   # Epoch 1
        0.55,  # Epoch 2 - improvement
        0.60,  # Epoch 3 - improvement
        0.61,  # Epoch 4 - small improvement
        0.61,  # Epoch 5 - no improvement
        0.60,  # Epoch 6 - worse
        0.61,  # Epoch 7 - no improvement
        0.60,  # Epoch 8 - worse
        0.59,  # Epoch 9 - worse
        0.58,  # Epoch 10 - worse (should stop here with patience=5)
    ]

    return {
        "metric_history": metric_history,
        "patience": 5,
        "min_delta": 0.001,
        "min_epochs": 3,
        "expected_stop_epoch": 10,
        "best_epoch": 4,
        "best_metric": 0.61
    }


@pytest.fixture
def mock_calibration_data():
    """
    Mock data for calibration testing.

    Provides perfectly calibrated and miscalibrated predictions.
    """
    import numpy as np
    np.random.seed(42)

    n_samples = 500

    # Perfectly calibrated: predicted probability matches actual frequency
    bins = np.linspace(0, 1, 11)
    y_prob_perfect = []
    y_true_perfect = []

    for i in range(len(bins) - 1):
        bin_prob = (bins[i] + bins[i+1]) / 2
        n_bin = n_samples // 10
        y_prob_perfect.extend([bin_prob] * n_bin)
        # Actual labels match probability
        y_true_perfect.extend(np.random.binomial(1, bin_prob, n_bin).tolist())

    # Overconfident: predicts high but accuracy is lower
    y_prob_overconfident = np.clip(np.random.beta(5, 2, n_samples), 0, 1)
    y_true_overconfident = (np.random.rand(n_samples) < 0.3).astype(int)

    return {
        "perfect": {
            "y_true": np.array(y_true_perfect),
            "y_prob": np.array(y_prob_perfect)
        },
        "overconfident": {
            "y_true": y_true_overconfident,
            "y_prob": y_prob_overconfident
        },
        "n_samples": n_samples
    }
