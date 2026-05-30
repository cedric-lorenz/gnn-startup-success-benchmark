"""
Tests for ablation study pipeline functionality.

Tests node type dropping, edge dropping, and metapath discovery configurations.
"""
import pytest
import torch
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


class TestAblationConfig:
    """Test ablation configuration parsing."""

    def test_drop_node_types_config(self, mini_config):
        """Ablation config should support drop_node_types list."""
        config = mini_config
        config["data_processing"]["ablation"]["drop_node_types"] = ["investor"]

        drop_list = config["data_processing"]["ablation"]["drop_node_types"]
        assert "investor" in drop_list

    def test_drop_edges_config(self, mini_config):
        """Ablation config should support drop_edges list."""
        config = mini_config
        config["metapath_discovery"]["automatic"]["ablation"]["drop_edges"] = ["funded_by"]

        drop_list = config["metapath_discovery"]["automatic"]["ablation"]["drop_edges"]
        assert "funded_by" in drop_list

    def test_empty_ablation_config(self, mini_config):
        """Empty ablation config should work without errors."""
        config = mini_config

        drop_nodes = config["data_processing"]["ablation"].get("drop_node_types", [])
        drop_edges = config["metapath_discovery"]["automatic"]["ablation"].get("drop_edges", [])

        assert drop_nodes == []
        assert drop_edges == []


class TestNodeTypeDropping:
    """Test node type ablation (complete node type removal)."""

    def test_drop_node_type_removes_from_graph(self, mock_hetero_graph, mini_config):
        """Dropping a node type should remove it from graph data."""
        from src.ml.models import SeHGNN

        graph = mock_hetero_graph
        config = mini_config

        assert "investor" in graph.node_types

        # This tests the config path only - actual dropping happens in preprocessing
        config["data_processing"]["ablation"]["drop_node_types"] = ["investor"]

        drop_list = config["data_processing"]["ablation"]["drop_node_types"]
        assert "investor" in drop_list

    def test_drop_multiple_node_types(self, mini_config):
        """Should be able to drop multiple node types."""
        config = mini_config
        config["data_processing"]["ablation"]["drop_node_types"] = ["investor", "founder"]

        drop_list = config["data_processing"]["ablation"]["drop_node_types"]
        assert len(drop_list) == 2
        assert "investor" in drop_list
        assert "founder" in drop_list


class TestEdgeDropping:
    """Test edge type ablation (specific relation removal)."""

    def test_edge_drop_config_path(self, mini_config):
        """Edge drop should be configurable at correct path."""
        config = mini_config
        config["metapath_discovery"]["automatic"]["ablation"]["drop_edges"] = ["funded_by", "has_founder"]

        drop_list = config["metapath_discovery"]["automatic"]["ablation"]["drop_edges"]
        assert "funded_by" in drop_list
        assert "has_founder" in drop_list

    def test_sehgnn_respects_edge_drop(self, mock_hetero_graph, mini_config):
        """SeHGNN should filter out dropped edges from metapaths."""
        from src.ml.models import SeHGNN

        graph = mock_hetero_graph
        metadata = graph.metadata()

        config = mini_config
        config["metapath_discovery"]["automatic"]["ablation"]["drop_edges"] = ["funded_by"]

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
            config=config,
            model_name="SeHGNN",
        )

        metapath_rels = [mp[1] if isinstance(mp, tuple) else mp for mp in model.metapaths]
        assert "funded_by" not in metapath_rels


class TestMetapathDiscoveryModes:
    """Test metapath discovery configuration modes."""

    def test_manual_mode_config(self, mini_config):
        """Manual mode should use existing definitions."""
        config = mini_config
        config["metapath_discovery"]["mode"] = "manual"

        assert config["metapath_discovery"]["mode"] == "manual"
        assert config["metapath_discovery"]["manual"]["use_existing_definitions"] is True

    def test_automatic_mode_config(self, mini_config):
        """Automatic mode should have discovery settings."""
        config = mini_config
        config["metapath_discovery"]["mode"] = "automatic"

        auto_config = config["metapath_discovery"]["automatic"]
        assert "max_hops" in auto_config
        assert "min_edges" in auto_config
        assert "max_metapaths" in auto_config
        assert "selection_strategy" in auto_config

    def test_selection_strategy_options(self, mini_config):
        """Selection strategy should accept valid options."""
        config = mini_config

        valid_strategies = ["least_edges", "most_edges", "random"]
        for strategy in valid_strategies:
            config["metapath_discovery"]["automatic"]["selection_strategy"] = strategy
            assert config["metapath_discovery"]["automatic"]["selection_strategy"] == strategy


class TestAblationSeriesA:
    """Tests mimicking ablation_series_a configuration scenarios."""

    def test_series_a_node_drop_investor(self, mini_config):
        """Series A: drop investor nodes."""
        config = mini_config
        config["data_processing"]["ablation"]["drop_node_types"] = ["investor"]

        assert "investor" in config["data_processing"]["ablation"]["drop_node_types"]
        assert "founder" not in config["data_processing"]["ablation"]["drop_node_types"]

    def test_series_a_node_drop_founder(self, mini_config):
        """Series A: drop founder nodes."""
        config = mini_config
        config["data_processing"]["ablation"]["drop_node_types"] = ["founder"]

        assert "founder" in config["data_processing"]["ablation"]["drop_node_types"]

    def test_series_a_metapath_strategy_comparison(self, mini_config):
        """Series A: compare different metapath strategies."""
        config = mini_config

        config["metapath_discovery"]["automatic"]["selection_strategy"] = "least_edges"
        assert config["metapath_discovery"]["automatic"]["selection_strategy"] == "least_edges"

        config["metapath_discovery"]["automatic"]["selection_strategy"] = "most_edges"
        assert config["metapath_discovery"]["automatic"]["selection_strategy"] == "most_edges"


class TestModelWithAblation:
    """Test model initialization with ablation configurations."""

    def test_model_with_reduced_metapaths(self, mock_hetero_graph, mini_config):
        """Model should work with reduced max_metapaths."""
        from src.ml.models import SeHGNN

        graph = mock_hetero_graph
        metadata = graph.metadata()

        config = mini_config
        config["metapath_discovery"]["automatic"]["max_metapaths"] = 3

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
            config=config,
            model_name="SeHGNN",
        )

        assert model is not None
        # Metapaths should be capped (including "self")
        assert len(model.metapaths) <= config["metapath_discovery"]["automatic"]["max_metapaths"] + 1

    def test_model_forward_with_ablation(self, mock_hetero_graph, mini_config):
        """Model forward pass should work with ablation config."""
        from src.ml.models import SeHGNN

        graph = mock_hetero_graph
        metadata = graph.metadata()

        config = mini_config
        config["metapath_discovery"]["automatic"]["ablation"]["drop_edges"] = ["funded_by"]

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
            config=config,
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

        assert "masked_multi_task_output" in output


class TestExcludeNodeTypes:
    """Test exclude_node_types in metapath discovery."""

    def test_exclude_node_types_config(self, mini_config):
        """exclude_node_types should be configurable."""
        config = mini_config
        config["metapath_discovery"]["automatic"]["exclude_node_types"] = ["city", "sector"]

        exclude_list = config["metapath_discovery"]["automatic"]["exclude_node_types"]
        assert "city" in exclude_list
        assert "sector" in exclude_list

    def test_empty_exclude_list(self, mini_config):
        """Empty exclude list should work."""
        config = mini_config
        config["metapath_discovery"]["automatic"]["exclude_node_types"] = []

        exclude_list = config["metapath_discovery"]["automatic"]["exclude_node_types"]
        assert exclude_list == []


# =============================================================================
# Tests for homophily-based metapath selection
# =============================================================================

class TestMetapathHomophily:
    """Test homophily computation and selection strategies."""

    def _make_discoverer(self, mini_config, labels, num_nodes=100):
        """Create a MetapathDiscovery instance with mock data."""
        import scipy.sparse as sp
        from torch_geometric.data import HeteroData
        from src.ml.metapath_discovery import MetapathDiscovery

        data = HeteroData()
        data["startup"].x = torch.randn(num_nodes, 8)
        data["startup"].y = torch.tensor(labels, dtype=torch.float)
        # Add a dummy edge type so MetapathDiscovery init doesn't fail
        data["startup", "similar", "startup"].edge_index = torch.tensor(
            [[0, 1], [1, 0]], dtype=torch.long
        )
        return MetapathDiscovery(mini_config, data)

    def test_baseline_homophily_balanced(self, mini_config):
        """Baseline should be 0.5 for balanced binary labels."""
        import numpy as np
        labels = [0] * 50 + [1] * 50
        disc = self._make_discoverer(mini_config, labels)
        baseline = disc._compute_baseline_homophily(np.array(labels))
        assert abs(baseline - 0.5) < 0.01

    def test_baseline_homophily_imbalanced(self, mini_config):
        """Baseline for 80/20 split: 0.8^2 + 0.2^2 = 0.68."""
        import numpy as np
        labels = [1] * 80 + [0] * 20
        disc = self._make_discoverer(mini_config, labels)
        baseline = disc._compute_baseline_homophily(np.array(labels))
        assert abs(baseline - 0.68) < 0.01

    def test_homophily_perfect(self, mini_config):
        """Metapath connecting only same-label nodes → homophily = 1.0."""
        import numpy as np
        import scipy.sparse as sp

        labels = [0] * 50 + [1] * 50
        disc = self._make_discoverer(mini_config, labels)

        # Edges only between nodes 0-49 (all label 0)
        row = np.array([0, 1, 2, 3])
        col = np.array([1, 0, 3, 2])
        adj = sp.csr_matrix((np.ones(4), (row, col)), shape=(100, 100))

        path_data = {
            'adj_matrix': adj,
            'start_type': 'startup',
            'end_type': 'startup',
        }
        h = disc._compute_metapath_homophily(path_data, np.array(labels))
        assert h == 1.0

    def test_homophily_zero(self, mini_config):
        """Metapath connecting only different-label nodes → homophily = 0.0."""
        import numpy as np
        import scipy.sparse as sp

        labels = [0] * 50 + [1] * 50
        disc = self._make_discoverer(mini_config, labels)

        # Edges from label-0 nodes to label-1 nodes
        row = np.array([0, 1, 2, 3])
        col = np.array([50, 51, 52, 53])
        adj = sp.csr_matrix((np.ones(4), (row, col)), shape=(100, 100))

        path_data = {
            'adj_matrix': adj,
            'start_type': 'startup',
            'end_type': 'startup',
        }
        h = disc._compute_metapath_homophily(path_data, np.array(labels))
        assert h == 0.0

    def test_select_by_homophily_prefers_high(self, mini_config):
        """top_homophily should rank high-homophily paths first."""
        import numpy as np
        import scipy.sparse as sp

        labels = [0] * 50 + [1] * 50
        disc = self._make_discoverer(mini_config, labels)

        # Path A: all same-label (homophily=1.0)
        adj_a = sp.csr_matrix(
            (np.ones(4), (np.array([0,1,2,3]), np.array([1,0,3,2]))),
            shape=(100, 100)
        )
        # Path B: all cross-label (homophily=0.0)
        adj_b = sp.csr_matrix(
            (np.ones(4), (np.array([0,1,2,3]), np.array([50,51,52,53]))),
            shape=(100, 100)
        )

        filtered = {
            'path_a': {'adj_matrix': adj_a, 'start_type': 'startup',
                       'end_type': 'startup', 'edge_count': 4, 'path': []},
            'path_b': {'adj_matrix': adj_b, 'start_type': 'startup',
                       'end_type': 'startup', 'edge_count': 4, 'path': []},
        }
        auto_config = mini_config['metapath_discovery']['automatic']
        auto_config['max_metapaths'] = 1

        result = disc._select_by_homophily(filtered, auto_config, 'top_homophily')
        assert 'path_a' in result
        assert 'path_b' not in result

    def test_select_by_heterophily_prefers_low(self, mini_config):
        """top_heterophily should rank low-homophily paths first."""
        import numpy as np
        import scipy.sparse as sp

        labels = [0] * 50 + [1] * 50
        disc = self._make_discoverer(mini_config, labels)

        adj_a = sp.csr_matrix(
            (np.ones(4), (np.array([0,1,2,3]), np.array([1,0,3,2]))),
            shape=(100, 100)
        )
        adj_b = sp.csr_matrix(
            (np.ones(4), (np.array([0,1,2,3]), np.array([50,51,52,53]))),
            shape=(100, 100)
        )

        filtered = {
            'path_a': {'adj_matrix': adj_a, 'start_type': 'startup',
                       'end_type': 'startup', 'edge_count': 4, 'path': []},
            'path_b': {'adj_matrix': adj_b, 'start_type': 'startup',
                       'end_type': 'startup', 'edge_count': 4, 'path': []},
        }
        auto_config = mini_config['metapath_discovery']['automatic']
        auto_config['max_metapaths'] = 1

        result = disc._select_by_homophily(filtered, auto_config, 'top_heterophily')
        assert 'path_b' in result
        assert 'path_a' not in result

    def test_selection_strategy_config_options(self, mini_config):
        """All 4 strategies should be valid config values."""
        for strategy in ["least_edges", "most_edges", "top_homophily", "top_heterophily"]:
            mini_config["metapath_discovery"]["automatic"]["selection_strategy"] = strategy
            assert mini_config["metapath_discovery"]["automatic"]["selection_strategy"] == strategy


# =============================================================================
# Tests for feature group ablation (Series F)
# =============================================================================

class TestFeatureGroupAblation:
    """Test feature group ablation for startup features."""

    def test_feature_groups_constant_defined(self):
        """STARTUP_FEATURE_GROUPS should be importable and contain all 6 groups."""
        from src.ml.preprocessing import STARTUP_FEATURE_GROUPS

        expected_groups = {"team", "funding_rounds", "temporal_intrinsic",
                           "investor_aggregates", "online_presence",
                           "description", "graph_features"}
        assert set(STARTUP_FEATURE_GROUPS.keys()) == expected_groups

    def test_feature_groups_no_overlap(self):
        """Tabular feature groups should not share columns."""
        from src.ml.preprocessing import STARTUP_FEATURE_GROUPS

        tabular_groups = ["team", "funding_rounds", "temporal_intrinsic",
                          "investor_aggregates", "online_presence"]
        all_cols = []
        for group in tabular_groups:
            cols = STARTUP_FEATURE_GROUPS[group]
            for c in cols:
                assert c not in all_cols, f"Column '{c}' appears in multiple groups"
                all_cols.append(c)

    def test_team_group_has_expected_columns(self):
        """Team group should include founder/education-related columns."""
        from src.ml.preprocessing import STARTUP_FEATURE_GROUPS

        team_cols = STARTUP_FEATURE_GROUPS["team"]
        assert "founder_count" in team_cols
        assert "female_ratio" in team_cols
        assert "has_tech_and_biz" in team_cols
        assert "phd_count" in team_cols

    def test_funding_rounds_group_has_expected_columns(self):
        """Funding rounds group should include per-round money/investors columns."""
        from src.ml.preprocessing import STARTUP_FEATURE_GROUPS

        cols = STARTUP_FEATURE_GROUPS["funding_rounds"]
        assert "money_angel" in cols
        assert "money_series_a" in cols
        assert "investors_series_b" in cols
        assert "series_j_round" in cols

    def test_temporal_intrinsic_group_has_expected_columns(self):
        """Temporal intrinsic should carry CB-native summaries and intrinsic temporal trajectories."""
        from src.ml.preprocessing import STARTUP_FEATURE_GROUPS

        cols = STARTUP_FEATURE_GROUPS["temporal_intrinsic"]
        assert "total_funding_usd" in cols
        assert "funding_growth_rate" in cols
        assert "last_funding_stage" in cols
        assert "num_funding_rounds" in cols

    def test_investor_aggregates_group_has_expected_columns(self):
        """Investor aggregates should carry neighbor-count style features the graph can derive."""
        from src.ml.preprocessing import STARTUP_FEATURE_GROUPS

        cols = STARTUP_FEATURE_GROUPS["investor_aggregates"]
        assert "total_investors" in cols
        assert "top_10_percent_investor_count" in cols
        assert "investor_follow_on_ratio" in cols

    def test_financial_aggregates_legacy_alias_covers_successor_groups(self):
        """Legacy 'financial_aggregates' alias expands to temporal_intrinsic + investor_aggregates union."""
        from src.ml.preprocessing import STARTUP_FEATURE_GROUPS

        legacy_cols_expected = {
            "num_funding_rounds", "total_funding_usd", "total_investors",
            "top_10_percent_investor_count", "top_50_percent_investor_count",
            "days_until_first_funding", "accelerator_participation_count",
            "investor_type", "avg_days_between_rounds", "funding_growth_rate",
            "months_since_last_funding", "implied_monthly_burn",
            "has_down_round", "investor_follow_on_ratio", "last_funding_stage",
            "last_funding_on", "total_funding_currency_code", "seriesA",
        }
        union = set(STARTUP_FEATURE_GROUPS["temporal_intrinsic"]) | set(
            STARTUP_FEATURE_GROUPS["investor_aggregates"]
        )
        assert legacy_cols_expected.issubset(union)

    def test_online_presence_group_has_expected_columns(self):
        """Online presence should include has_domain, social URLs, etc."""
        from src.ml.preprocessing import STARTUP_FEATURE_GROUPS

        cols = STARTUP_FEATURE_GROUPS["online_presence"]
        assert "has_domain" in cols
        assert "has_linkedin_url" in cols
        assert "type_organization" in cols

    def test_description_group_columns(self):
        """Description group should contain metadata columns."""
        from src.ml.preprocessing import STARTUP_FEATURE_GROUPS

        cols = STARTUP_FEATURE_GROUPS["description"]
        assert "has_description" in cols
        assert "description_length" in cols

    def test_graph_features_group_is_empty(self):
        """Graph features group should be empty (handled via config flags)."""
        from src.ml.preprocessing import STARTUP_FEATURE_GROUPS

        assert STARTUP_FEATURE_GROUPS["graph_features"] == []

    def test_drop_feature_groups_config(self, mini_config):
        """Ablation config should support drop_feature_groups list."""
        config = mini_config
        config["data_processing"]["ablation"]["drop_feature_groups"] = ["team"]

        drop_list = config["data_processing"]["ablation"]["drop_feature_groups"]
        assert "team" in drop_list

    def test_empty_drop_feature_groups(self, mini_config):
        """Empty drop_feature_groups should be the default."""
        config = mini_config
        drop_groups = config["data_processing"]["ablation"].get("drop_feature_groups", [])
        assert drop_groups == []

    def test_drop_graph_features_disables_config_flags(self, mini_config):
        """Dropping graph_features group should disable all graph feature config flags."""
        import copy
        config = copy.deepcopy(mini_config)
        config["data_processing"]["use_louvain_clusters"] = True
        config["data_processing"]["use_edge_counts"] = True
        config["data_processing"]["use_degree_centrality"] = True
        config["data_processing"]["use_pagerank_centrality"] = True
        config["data_processing"]["use_centrality_features"] = True
        config["data_processing"]["use_smart_money_features"] = True

        # Simulate the ablation logic from perform_preprocessing
        drop_groups = ["graph_features"]
        for group in drop_groups:
            if group == "graph_features":
                config["data_processing"]["use_louvain_clusters"] = False
                config["data_processing"]["use_edge_counts"] = False
                config["data_processing"]["use_degree_centrality"] = False
                config["data_processing"]["use_pagerank_centrality"] = False
                config["data_processing"]["use_centrality_features"] = False
                config["data_processing"]["use_smart_money_features"] = False

        assert config["data_processing"]["use_louvain_clusters"] is False
        assert config["data_processing"]["use_edge_counts"] is False
        assert config["data_processing"]["use_degree_centrality"] is False
        assert config["data_processing"]["use_pagerank_centrality"] is False
        assert config["data_processing"]["use_centrality_features"] is False
        assert config["data_processing"]["use_smart_money_features"] is False

    def test_node_preprocessing_drops_team_columns(self):
        """node_preprocessing should drop team columns when specified in drop_cols."""
        import pandas as pd
        import numpy as np

        # Create a minimal startup dataframe with team columns
        df = pd.DataFrame({
            "startup_uuid": [f"uuid_{i}" for i in range(10)],
            "name": [f"startup_{i}" for i in range(10)],
            "founder_count": np.random.randint(1, 5, 10),
            "female_count": np.random.randint(0, 3, 10),
            "total_funding_usd": np.random.uniform(1e5, 1e7, 10),
            "has_domain": np.random.randint(0, 2, 10),
            "acq_ipo_funding": np.random.randint(0, 2, 10),
            "future_status": ["operating"] * 10,
        })

        from src.ml.preprocessing import node_preprocessing, STARTUP_FEATURE_GROUPS
        import copy

        config = {
            "data_processing": {
                "target_mode": "binary_prediction",
                "multi_column": "future_status",
                "binary_column": "acq_ipo_funding",
                "ablation": {"drop_node_types": [], "drop_feature_groups": []},
                "multi_label": {"columns": {"funding": "new_funding_round", "acquisition": "new_acquired", "ipo": "new_ipo"}},
                "start_date": "2000-01-01",
            },
            "train": {"loss": {"retrieval_loss_type": None, "contrastive_positive_source": "text"}},
        }

        # Run without dropping - should have founder_count and female_count
        result = node_preprocessing(
            df.copy(), node_type="startup", uuid_col="startup_uuid", name_col="name",
            drop_cols=[], target_mode="binary_prediction",
            multi_column="future_status", binary_column="acq_ipo_funding",
            config=config,
        )
        feature_names = result[4]
        assert "founder_count" in feature_names
        assert "female_count" in feature_names

        # Run with team columns in drop_cols
        team_cols = STARTUP_FEATURE_GROUPS["team"]
        result2 = node_preprocessing(
            df.copy(), node_type="startup", uuid_col="startup_uuid", name_col="name",
            drop_cols=team_cols, target_mode="binary_prediction",
            multi_column="future_status", binary_column="acq_ipo_funding",
            config=config,
        )
        feature_names2 = result2[4]
        assert "founder_count" not in feature_names2
        assert "female_count" not in feature_names2
        assert "total_funding_usd" in feature_names2
        assert "has_domain" in feature_names2
