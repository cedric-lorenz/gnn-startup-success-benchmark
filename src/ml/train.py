"""Training loop for graph neural network and baseline models."""
from functools import partial
from torch_geometric.loader import NeighborLoader
from torch.nn import PReLU, ReLU
from torch_geometric.nn import to_hetero, BatchNorm
from torch.optim.adam import Adam
from torch.optim.adamw import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR, LinearLR, SequentialLR
import torch_geometric.nn
from torch import device, from_numpy, cuda
from .models import SageGNN, HAN, FocalLoss, GAT, GCN, RGCN, Hetero2Net, VenGNN, HeteroMLP, XGBoostAdapter, SeHGNN, SimpleHGN, RandomBaseline, DegreeCentralityBaseline, LLMBaseline
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss
import torch
from tqdm import tqdm
import wandb
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
import torch.nn.functional as F
import sys
from .eval import Evaluator
from .explain import explain_model
from .utils import load_config
from pathlib import Path
import os

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
from src.ml.calibration import analyze_model_calibration_from_predictions
from src.ml.visualize import visualize_decision_boundary, visualize_metapath_weights, visualize_metapath_weights_vs_stats

config = load_config()

class Trainer:
    def __init__(self, graph_data, config, calibrators=None):
        self.config = config

        # Retrieve Hyperparameters
        self.device = device(
            config["train"]["device"] if cuda.is_available() else "cpu"
        )
        self.use_wandb = config["wandb"]["enabled"]
        self.lr = config["train"]["lr"]
        self.epochs = config["train"]["epochs"]
        self.aggregation_method = config["train"]["aggregation_method"]
        self.model_name = config["train"]["model"]
        self.model_config = config["models"].get(self.model_name, {})
        self.target_mode = config["data_processing"]["target_mode"]
        self.gradient_clip_val = config["train"].get("gradient_clip_val", None)

        # Batch loader creation
        self.train_loader = self._create_loader(graph_data, "train")
        self.val_loader = self._create_loader(graph_data, "val")
        self.test_loader = self._create_loader(graph_data, "test")

        # Evaluation tracking
        self.current_epoch = -1
        self.best_model_dict = None
        self.test_best_model = config["eval"]["test_best_model"]
        self.min_amount_of_epochs = config["eval"]["min_amount_of_epochs"]
        
        # Early stopping
        self.early_stopping_enabled = config["eval"].get("early_stopping", {}).get("enabled", False)
        self.early_stopping_patience = config["eval"].get("early_stopping", {}).get("patience", 20)
        self.early_stopping_min_delta = config["eval"].get("early_stopping", {}).get("min_delta", 0.001)
        self.early_stopping_counter = 0
        self.aggregation_method = config["train"].get("aggregation_method", "mean")
        
        self.calibrators = calibrators  # Store calibrators (single or dict)
        
        # Model and optimization setup
        self.data = graph_data
        self.model = self._initialize_model(graph_data).to(self.device)
        self.weight_decay = config["train"].get("weight_decay", 0.0)
        
        if not isinstance(self.model, XGBoostAdapter):
            opt_name = config["train"].get("optimizer", "Adam")
            if opt_name == "AdamW":
                self.optimizer = AdamW(
                    self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay
                )
            elif opt_name == "Adam":
                self.optimizer = Adam(
                    self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay
                )
            else:
                raise ValueError(
                    f"train.optimizer must be 'Adam' or 'AdamW' (got {opt_name!r})"
                )
        else:
            self.optimizer = None
        self.scheduler = self._create_scheduler()
        self.loss = self._select_loss_function(graph_data)

        # Evaluator setup
        self.evaluator = Evaluator(
            model=self.model,
            device=self.device,
            use_wandb=self.use_wandb,
            optimization_metric_type=config["eval"]["optimization_metric_type"],
            min_amount_of_epochs=self.min_amount_of_epochs,
            calibrator=self.calibrators,
            config=config,
        )

        self.explain = config["explain"]["enabled"]
        self.explain_path = config["explain"]["path"]
        self.explain_sample_size = config["explain"]["sample_size"]
        self.explain_method = config["explain"]["method"]

    def _create_scheduler(self):
        """Create learning rate scheduler based on config."""
        if self.optimizer is None:
            return None
            
        scheduler_config = self.config["train"].get("scheduler", {})
        scheduler_type = scheduler_config.get("type", None)
        
        if scheduler_type is None or scheduler_type.lower() == "none":
            print("📅 Scheduler: None (constant learning rate)")
            return None
        
        elif scheduler_type.lower() == "reducelronplateau":
            mode = scheduler_config.get("mode", "max")
            factor = float(scheduler_config.get("factor", 0.5))
            patience = int(scheduler_config.get("patience", 10))
            min_lr = float(scheduler_config.get("min_lr", 1e-6))
            verbose = bool(scheduler_config.get("verbose", True))
            threshold = float(scheduler_config.get("threshold", 1e-4))
            threshold_mode = scheduler_config.get("threshold_mode", "rel")
            
            print(f"📅 Scheduler: ReduceLROnPlateau (mode={mode}, factor={factor}, patience={patience}, min_lr={min_lr}, threshold={threshold}, threshold_mode={threshold_mode})")
            return ReduceLROnPlateau(
                self.optimizer, 
                mode=mode, 
                factor=factor, 
                patience=patience, 
                verbose=verbose,
                min_lr=min_lr,
                threshold=threshold,
                threshold_mode=threshold_mode
            )
            
        elif scheduler_type.lower() == "cosineannealinglr":
            T_max = int(scheduler_config.get("T_max", self.epochs))
            eta_min = float(scheduler_config.get("eta_min", 1e-6))
            warmup_epochs = int(scheduler_config.get("warmup_epochs", 0))

            if warmup_epochs > 0:
                warmup_scheduler = LinearLR(
                    self.optimizer,
                    start_factor=1.0 / max(warmup_epochs, 1),
                    end_factor=1.0,
                    total_iters=warmup_epochs
                )
                cosine_scheduler = CosineAnnealingLR(
                    self.optimizer,
                    T_max=T_max - warmup_epochs,
                    eta_min=eta_min
                )
                print(f"📅 Scheduler: LinearWarmup({warmup_epochs} epochs) + CosineAnnealingLR (T_max={T_max - warmup_epochs}, eta_min={eta_min})")
                return SequentialLR(
                    self.optimizer,
                    schedulers=[warmup_scheduler, cosine_scheduler],
                    milestones=[warmup_epochs]
                )
            else:
                print(f"📅 Scheduler: CosineAnnealingLR (T_max={T_max}, eta_min={eta_min})")
                return CosineAnnealingLR(
                    self.optimizer,
                    T_max=T_max,
                    eta_min=eta_min
                )
            
        else:
            raise ValueError(f"Unknown scheduler type: {scheduler_type}. Supported: 'ReduceLROnPlateau', 'CosineAnnealingLR', 'None'")

    def _create_loader(self, graph_data, mode: str):
        assert mode in {"train", "val", "test"}, f"Invalid mode {mode}"
        cfg = self.config["data_processing"][mode]

        if not cfg["use_batches"]:
            return None

        mask = graph_data["startup"][f"{mode}_mask"]
        input_nodes = ("startup", mask)

        return NeighborLoader(
            data=graph_data,
            input_nodes=input_nodes,
            num_neighbors=[cfg["neighbor_loader_nodes"]]
            * cfg["neighbor_loader_iterations"],
            batch_size=cfg["neighbor_loader_batch_size"],
            subgraph_type="bidirectional",
            shuffle=(mode == "train"),
        )

    def _initialize_model(self, graph_data):
        model_name = self.model_name

        # Handle different target formats to get num_classes
        y_data = self.data["startup"].y

        if self.target_mode == "multi_task":
            if isinstance(y_data, tuple):
                # Case: y is a tuple of (multi_class_tensor, binary_tensor)
                # Take the multi-class tensor (first element) to get number of classes
                multi_class_labels = y_data[0]
                num_classes = multi_class_labels.unique().numel()
            elif isinstance(y_data, torch.Tensor) and y_data.dim() == 2:
                # Case: y is a stacked tensor of shape [N, 2]
                # Second column contains multi-class labels
                num_classes = y_data[:, 1].unique().numel()
            else:
                raise ValueError(
                    f"Unsupported y format for multi_task mode: {type(y_data)}"
                )

        elif self.target_mode == "multi_prediction":
            if isinstance(y_data, tuple):
                # Take the first tensor (assuming it's the multi-class labels)
                num_classes = y_data[0].unique().numel()
            else:
                num_classes = y_data.unique().numel()

        elif self.target_mode == "binary_prediction":
            # For binary classification, we have 2 classes (0, 1)
            num_classes = 2

        elif self.target_mode == "multi_label":
            # 3 tasks: Funding, Acquisition, IPO
            num_classes = 3

        elif self.target_mode == "masked_multi_task":
            # 2 predictions: Momentum, Liquidity
            num_classes = 2

        else:
            raise ValueError(f"Unsupported target_mode: {self.target_mode}")

        self.num_classes = num_classes

        # Determine output channels based on target mode
        # For binary prediction with BCEWithLogitsLoss, we need 1 output channel
        # For multi-class prediction, we need as many channels as classes
        if self.target_mode == "binary_prediction":
            output_channels = 1
        elif self.target_mode == "multi_prediction":
            output_channels = num_classes
        elif self.target_mode == "multi_task":
            # Multi-task handled separately in models
            output_channels = self.model_config.get("out_channels", num_classes)
        elif self.target_mode == "multi_label":
            # Multi-label (3 tasks)
            output_channels = 3 # Or hidden if handled by separate heads internal to model
        elif self.target_mode == "masked_multi_task":
            output_channels = 2
        else:
            output_channels = self.model_config.get("out_channels", num_classes)

        if model_name == "GAT":
            # Convert list to tuple for PyTorch Geometric
            in_channels = tuple(self.model_config["in_channels"]) if isinstance(self.model_config["in_channels"], list) else self.model_config["in_channels"]

            model = GAT(
                in_channels=in_channels,
                hidden_channels=self.model_config["hidden_channels"],
                out_channels=output_channels,
                num_layers=self.model_config["num_layers"],
                v2=self.model_config["v2"],
                normalize=self.model_config["normalize"],
                activation=self.model_config["activation_type"],
                jumping_knowledge=self.model_config["jumping_knowledge"],
                add_self_loops=self.model_config["add_self_loops"],
                target_mode=self.target_mode,
                num_classes=num_classes,
                metadata=graph_data.metadata(),
                aggr=self.aggregation_method,
                dropout=self.model_config.get("dropout", 0.0),
                heads=self.model_config.get("heads", 1),
                negative_slope=self.model_config.get("negative_slope", 0.2),
            )
        elif model_name == "GCN":
            in_channels = self.model_config["in_channels"]
            model = GCN(
                in_channels=in_channels,
                hidden_channels=self.model_config["hidden_channels"],
                out_channels=output_channels,
                num_layers=self.model_config["num_layers"],
                normalize=self.model_config["normalize"],
                activation=self.model_config["activation_type"],
                jumping_knowledge=self.model_config["jumping_knowledge"],
                target_mode=self.target_mode,
                num_classes=num_classes,
                metadata=graph_data.metadata(),
                aggr=self.aggregation_method,
                dropout=self.model_config.get("dropout", 0.0),
                add_self_loops=self.model_config.get("add_self_loops", False),
            )
        elif model_name == "RGCN":
            in_channels_dict = {
                node_type: graph_data[node_type].num_features
                for node_type in graph_data.node_types
            }
            # FastRGCNConv only supports {add, sum, mean}; paper uses mean (c_{i,r}=|N_i^r|).
            rgcn_aggr = self.model_config.get("aggr", "mean")
            if rgcn_aggr not in ("add", "sum", "mean"):
                print(f"⚠️  RGCN: global aggregation_method='{self.aggregation_method}' unsupported, falling back to 'mean'")
                rgcn_aggr = "mean"
            model = RGCN(
                in_channels=in_channels_dict,
                hidden_channels=self.model_config["hidden_channels"],
                num_layers=self.model_config.get("num_layers", 2),
                num_bases=self.model_config.get("num_bases", 30),
                num_blocks=self.model_config.get("num_blocks", None),
                normalize=self.model_config.get("normalize", True),
                activation=self.model_config.get("activation_type", "relu"),
                target_mode=self.target_mode,
                num_classes=num_classes,
                metadata=graph_data.metadata(),
                aggr=rgcn_aggr,
                dropout=self.model_config.get("dropout", 0.0),
            )
        elif model_name == "Hetero2Net":
            startup_dim = graph_data["startup"].num_features
            model = Hetero2Net(
                in_channels=startup_dim,
                hidden_channels=self.model_config["hidden_channels"],
                metadata=graph_data.metadata(),
                num_layers=self.model_config.get("num_layers", 2),
                dropout=self.model_config.get("dropout", 0.0),
                activation_type=self.model_config.get("activation_type", "relu"),
                target_mode=self.target_mode,
                num_classes=num_classes,
                alpha=self.model_config.get("alpha", 0.2),
                beta=self.model_config.get("beta", 0.4),
                mask_ratio=self.model_config.get("mask_ratio", 0.5),
                label_mask_ratio=self.model_config.get("label_mask_ratio", 0.7),
                use_label_propagation=self.model_config.get("use_label_propagation", True),
                num_label_classes=self.model_config.get("num_label_classes", 2),
                root_weight=self.model_config.get("root_weight", True),
                normalize_bn=self.model_config.get("normalize_bn", True),
                corr_on_metapaths_only=self.model_config.get("corr_on_metapaths_only", True),
                aux_reduction=self.model_config.get("aux_reduction", "sum"),
            )
            # Register masked training labels (paper §5.3). Non-train rows are
            # zeroed so they don't leak into the label embedding.
            if self.model_config.get("use_label_propagation", True):
                y = graph_data["startup"].y.float()
                num_lc = self.model_config.get("num_label_classes", 2)
                y_masked = y[:, :num_lc].clone()
                train_mask = graph_data["startup"].train_mask
                y_masked[~train_mask] = 0.0
                model.set_label_propagation_input(y_masked)
        elif model_name == "VenGNN":
            startup_dim = graph_data["startup"].num_features
            model = VenGNN(
                in_channels=startup_dim,
                hidden_channels=self.model_config["hidden_channels"],
                metadata=graph_data.metadata(),
                num_layers=self.model_config.get("num_layers", 2),
                heads=self.model_config.get("heads", 4),
                dropout=self.model_config.get("dropout", 0.2),
                rw_num_walks=self.model_config.get("rw_num_walks", 8),
                rw_walk_length=self.model_config.get("rw_walk_length", 5),
                activation_type=self.model_config.get("activation_type", "relu"),
                target_mode=self.target_mode,
                num_classes=num_classes,
                max_metapaths=self.model_config.get("max_metapaths", 50),
                rw_seed=self.config.get("seed", 42),
                use_paper_attention=self.model_config.get("use_paper_attention", False),
                branch_mode=self.model_config.get("branch_mode", "full"),
                paper_widening=self.model_config.get("paper_widening", False),
                gat_edge_mode=self.model_config.get("gat_edge_mode", "feature"),
                rw_weighted=self.model_config.get("rw_weighted", False),
            )
            # Register PathCount edge weights from the HeteroData object.
            weight_dict = {
                mp: getattr(graph_data[mp], "edge_weight", None)
                for mp in model.metapaths
            }
            model.set_metapath_edge_weights(weight_dict)
        elif model_name == "MLP":
            model = HeteroMLP(
                hidden_channels=self.model_config["hidden_channels"],
                target_mode=self.target_mode,
                num_classes=num_classes,
                activation_type=self.model_config["activation_type"],
                normalize=self.model_config["normalize"],
                dropout=self.model_config.get("dropout", 0.0),
                metadata=graph_data.metadata(),
            )
        elif model_name == "XGBoost":
            model = XGBoostAdapter(
                n_estimators=self.model_config.get("n_estimators", 100),
                max_depth=self.model_config.get("max_depth", 6),
                learning_rate=self.model_config.get("learning_rate", 0.1),
                subsample=self.model_config.get("subsample", 0.8),
                colsample_bytree=self.model_config.get("colsample_bytree", 0.8),
                gamma=self.model_config.get("gamma", 0),
                reg_alpha=self.model_config.get("reg_alpha", 0),
                reg_lambda=self.model_config.get("reg_lambda", 1),
                min_child_weight=self.model_config.get("min_child_weight", 1),
                scale_pos_weight=self.model_config.get("scale_pos_weight", 1.0),
                objective=self.model_config.get("objective", "binary:logistic"),
                tree_method=self.model_config.get("tree_method", "hist"),
                target_mode=self.target_mode,
                num_classes=num_classes,
                random_state=self.config.get("seed", 42),
            )
        elif model_name == "SageGNN":
            # Prefer per-model SageGNN.aggr (sweepable) over the global
            # aggregation_method. Paper's defining ablation is mean/max/lstm.
            sage_aggr = self.model_config.get("aggr") or self.aggregation_method
            model = SageGNN(
                hidden_channels=self.model_config["hidden_channels"],
                num_layers=self.model_config["num_layers"],
                activation_type=self.model_config["activation_type"],
                normalize=self.model_config["normalize"],
                target_mode=self.target_mode,
                num_classes=num_classes,
                dropout=self.model_config.get("dropout", 0.0),
                metadata=graph_data.metadata(),
                aggr=sage_aggr,
            )
        elif model_name == "HAN":
            model = HAN(
                in_channels=self.model_config["in_channels"],
                hidden_channels=self.model_config["hidden_channels"],
                metadata=graph_data.metadata(),
                num_layers=self.model_config["num_layers"],
                heads=self.model_config["heads"],
                negative_slope=self.model_config["negative_slope"],
                dropout=self.model_config["dropout"],
                activation_type=self.model_config["activation_type"],
                target_mode=self.target_mode,
                num_classes=num_classes,
            )
        elif model_name == "SimpleHGN":
            # Simple-HGN needs per-type input dims to build the fc_list (one Linear per
            # node-type input dim, paper Section 4).
            in_channels_dict = {
                node_type: graph_data[node_type].num_features
                for node_type in graph_data.node_types
            }
            model = SimpleHGN(
                in_channels=in_channels_dict,
                hidden_channels=self.model_config["hidden_channels"],
                metadata=graph_data.metadata(),
                num_layers=self.model_config.get("num_layers", 2),
                heads=self.model_config.get("heads", 8),
                edge_dim=self.model_config.get("edge_dim", None),
                dropout=self.model_config.get("dropout", 0.5),
                attn_dropout=self.model_config.get("attn_dropout", 0.5),
                negative_slope=self.model_config.get("negative_slope", 0.05),
                residual=self.model_config.get("residual", True),
                alpha=self.model_config.get("alpha", 0.05),
                l2_normalize=self.model_config.get("l2_normalize", True),
                bias=self.model_config.get("bias", False),
                activation_type=self.model_config.get("activation_type", "elu"),
                target_mode=self.target_mode,
                num_classes=num_classes,
            )
        elif model_name == "RandomBaseline":
            model = RandomBaseline(
                hidden_channels=0,
                target_mode=self.target_mode,
                num_classes=num_classes,
            )
        elif model_name == "SeHGNN":
            # SeHGNN requires explicit input dimensions for its Linear layers
            in_channels_dict = {
                node_type: graph_data[node_type].num_features 
                for node_type in graph_data.node_types
            }
            
            # Determine num_retrieval_classes for ArcFace
            num_ret_classes = 100 # Default
            if "retrieval_class_idx" in graph_data["startup"]:
                # If it's a tensor attribute
                 ret_idx = graph_data["startup"]["retrieval_class_idx"]
                 num_ret_classes = int(ret_idx.max().item()) + 1
                 print(f"Trainer found {num_ret_classes} retrieval classes (from graph data).")
            # Otherwise it's encoded in the target tensor y (column 4).
            elif self.target_mode == "masked_multi_task" and graph_data["startup"].y.shape[1] >= 5:
                  # Index 4 is retrieval label
                  y_all = graph_data["startup"].y
                  ret_vals = y_all[:, 4]
                  num_ret_classes = int(ret_vals.max().item()) + 1
                  print(f"Trainer found {num_ret_classes} retrieval classes (from target tensor).")

            model = SeHGNN(
                in_channels=in_channels_dict,
                hidden_channels=self.model_config["hidden_channels"],
                metadata=graph_data.metadata(),
                num_layers=self.model_config.get("num_layers", 2),
                heads=self.model_config.get("heads", 1),
                dropout=self.model_config.get("dropout", 0.5),
                input_drop=self.model_config.get("input_drop", 0.1),
                att_drop=self.model_config.get("att_drop", 0.0),
                activation_type=self.model_config.get("activation_type", "relu"),
                target_mode=self.target_mode,
                num_classes=num_classes,
                aggregation_method=self.aggregation_method, # Use global aggregation setting
                use_residual=self.model_config.get("use_residual", True),
                transformer_activation=self.model_config.get("transformer_activation", "none"),
                use_self_loop=self.model_config.get("use_self_loop", True),
                config=self.config,  # Pass config for retrieval head
                model_name="SeHGNN",  # Pass model name for config lookup
                num_retrieval_classes=num_ret_classes, # Pass finding to model
                num_hops=self.model_config.get("num_hops", 1),
                gamma_init=self.model_config.get("gamma_init", 0.0),
                gamma_learnable=self.model_config.get("gamma_learnable", True),
                channel_masking=self.model_config.get("channel_masking", False),
                use_layer_norm=self.model_config.get("use_layer_norm", False),
                use_discrepancy=self.model_config.get("use_discrepancy", False),
            )
        elif model_name in ("DegreeCentralityBaseline", "DegreeCentrality"):
            from torch_geometric.utils import degree
            
            # Compute global degrees for startups
            # We need to sum degrees from all edge types connected to 'startup'
            print("Computing global degrees for Degree Centrality Baseline...")
            num_startups = graph_data['startup'].num_nodes
            total_degrees = torch.zeros(num_startups, dtype=torch.long)
            
            for edge_type in graph_data.edge_types:
                src, rel, dst = edge_type
                if src == 'startup':
                    # Outgoing edges (index 0)
                    d = degree(graph_data[edge_type].edge_index[0], num_nodes=num_startups)
                    total_degrees += d.long()
                if dst == 'startup':
                    # Incoming edges (index 1)
                    d = degree(graph_data[edge_type].edge_index[1], num_nodes=num_startups)
                    total_degrees += d.long()
            
            model = DegreeCentralityBaseline(
                hidden_channels=0,
                degrees=total_degrees,
                target_mode=self.target_mode,
                num_classes=num_classes,
            )
        elif model_name == "LLM":
            # Use the filtered startup DataFrame from preprocessing (stored on graph_data).
            # This ensures node index i maps to the correct startup, matching the GNN test set.
            raw_features_df = graph_data["startup"].raw_df
            print(f"LLM Baseline: Using {len(raw_features_df)} preprocessed startups for prompting")

            model = LLMBaseline(
                hidden_channels=0,
                config=self.config,
                raw_features_df=raw_features_df,
                target_mode=self.target_mode,
                num_classes=num_classes,
            )
        else:
            print(f"Model {model_name} is not implemented.")
            sys.exit(1)

        print(model)

        # HAN and SageGNN handle heterogeneous graphs internally
        # Explicitly check for DegreeCentralityBaseline to avoid string matching issues
        if model_name in ["HAN", "SageGNN", "GAT", "GCN", "RGCN", "Hetero2Net", "VenGNN", "MLP", "XGBoost", "Random", "DegreeCentrality", "SeHGNN", "SimpleHGN", "RandomBaseline", "LLM"]:
            return model
        elif isinstance(model, (DegreeCentralityBaseline, LLMBaseline)):
            return model
        else:
            return to_hetero(model, graph_data.metadata(), aggr=self.aggregation_method)

    def _select_loss_function(self, graph_data):
        if isinstance(self.model, XGBoostAdapter):
            return None

        # LLMBaseline doesn't need a loss function (no training)
        if isinstance(self.model, LLMBaseline):
            return None

        use_class_weights = self.config["train"]["loss"]["use_class_weights"]
        use_focal_loss = self.config["train"]["loss"].get("use_focal_loss", False)
        resample = self.config["data_processing"]["resample"]["enabled"]
        assert not (
            use_class_weights and resample
        ), "Cannot use class weights with resampling. Choose one method."
        assert not (
            use_class_weights and use_focal_loss
        ), "Cannot use both class weights and focal loss simultaneously. Choose one method."
        device = self.device

        def compute_class_weights(labels):
            classes = np.unique(labels)
            weights = compute_class_weight("balanced", y=labels, classes=classes)
            return torch.tensor(weights, dtype=torch.float32).to(device)

        # Extract labels based on y format
        y_data = graph_data["startup"].y
        train_mask = graph_data["startup"].train_mask
        val_mask = graph_data["startup"].val_mask
        
        # Note: For masked_multi_task, y is [N, 4]. We unpack inside loss function.
        # But for other checks below (like binary_prediction), we extract labels here.
        
        if isinstance(y_data, tuple):
            # y is a tuple: (multi_class_tensor, binary_tensor)
            multi_class_labels = y_data[0][train_mask | val_mask].cpu().numpy()
            binary_labels = y_data[1][train_mask | val_mask].cpu().numpy()
        elif isinstance(y_data, torch.Tensor) and y_data.dim() == 2:
            # y is a stacked tensor
            # Check if likely multi_task (N, 2) or multi_label (N, 3) or masked (N, 4)
            if y_data.shape[1] == 2 and self.target_mode == "multi_task":
                 combined_labels = y_data[train_mask | val_mask].cpu().numpy()
                 binary_labels = combined_labels[:, 0]
                 multi_class_labels = combined_labels[:, 1]
            else:
                 # Default extraction might need adjustment based on target_mode usage
                 pass
        else:
            # y is a single tensor
            labels = y_data[train_mask | val_mask].cpu().numpy()

        if self.target_mode == "binary_prediction":
            if isinstance(y_data, tuple):
                labels = binary_labels

            if use_focal_loss:
                focal_alpha = self.config["train"]["loss"].get("focal_alpha", 0.25)
                focal_gamma = self.config["train"]["loss"].get("focal_gamma", 2.0)
                print(f"📊 Loss Function: Focal Loss (alpha={focal_alpha}, gamma={focal_gamma})")
                
                def focal_bce_loss(input, target):
                    # Convert to probabilities
                    p = torch.sigmoid(input)
                    # Compute BCE loss
                    bce_loss = F.binary_cross_entropy_with_logits(input, target, reduction='none')
                    # Compute focal weight
                    p_t = p * target + (1 - p) * (1 - target)  # pt
                    focal_weight = (1 - p_t) ** focal_gamma
                    if focal_alpha is not None:
                        alpha_t = focal_alpha * target + (1 - focal_alpha) * (1 - target)
                        focal_weight *= alpha_t
                    return (focal_weight * bce_loss).mean()
                
                return focal_bce_loss
                
            elif use_class_weights:
                pos_weight = (labels == 0).sum() / (labels == 1).sum()
                print(f"📊 Loss Function: BCEWithLogitsLoss (with class weights, pos_weight={pos_weight:.4f})")
                return BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight).to(device))
            else:
                print("📊 Loss Function: BCEWithLogitsLoss (no class weights)")
                return BCEWithLogitsLoss()

        elif self.target_mode == "multi_prediction":
            if isinstance(y_data, tuple):
                labels = multi_class_labels

            if use_focal_loss:
                focal_alpha = self.config["train"]["loss"].get("focal_alpha", 0.25)
                focal_gamma = self.config["train"]["loss"].get("focal_gamma", 2.0)
                print(f"📊 Loss Function: Focal CrossEntropy Loss (alpha={focal_alpha}, gamma={focal_gamma})")
                return FocalLoss(gamma=focal_gamma, alpha=focal_alpha)
            else:
                weight = compute_class_weights(labels) if use_class_weights else None
                label_smoothing = self.config["train"]["loss"].get("label_smoothing", 0.0)
                
                if use_class_weights:
                    print(f"📊 Loss Function: Weighted CrossEntropyLoss (class weights: {weight.cpu().numpy() if weight is not None else None}, label_smoothing: {label_smoothing})")
                else:
                    print(f"📊 Loss Function: CrossEntropyLoss (no class weights, label_smoothing: {label_smoothing})")

                def weighted_ce_loss(input, target, sample_weights=None):
                    unweighted_loss = F.cross_entropy(
                        input, target, weight=weight, label_smoothing=label_smoothing, reduction="none"
                    )
                    if sample_weights is not None:
                        return (unweighted_loss * sample_weights).mean()
                    return unweighted_loss.mean()

                return weighted_ce_loss

        elif self.target_mode == "multi_task":
            # Handle both binary and multi-class losses
            if use_class_weights:
                pos_weight = (binary_labels == 0).sum() / (binary_labels == 1).sum()
                bin_loss = BCEWithLogitsLoss(
                    pos_weight=torch.tensor(pos_weight).to(device)
                )

                class_weights = compute_class_weights(multi_class_labels)
                label_smoothing = self.config["train"]["loss"].get("label_smoothing", 0.0)
                multi_loss = CrossEntropyLoss(weight=class_weights, label_smoothing=label_smoothing)
            else:
                bin_loss = BCEWithLogitsLoss()
                label_smoothing = self.config["train"]["loss"].get("label_smoothing", 0.0)
                multi_loss = CrossEntropyLoss(label_smoothing=label_smoothing)

            # Loss weighting
            alpha = self.config["train"]["loss"].get("binary_loss_weight", 0.5)
            alpha = float(alpha)
            assert 0.0 <= alpha <= 1.0, "binary_loss_weight must be between 0 and 1"
            
            loss_type = "with class weights" if use_class_weights else "no class weights"
            print(f"📊 Loss Function: Composite Loss (BCEWithLogitsLoss + CrossEntropyLoss, {loss_type}, α={alpha:.2f})")

            def composite_loss(output, targets):
                bin_l = bin_loss(output["binary"], targets["binary"].float())
                multi_l = multi_loss(
                    output["multi_class"], targets["multi_class"].long()
                )
                print("Binary Loss:", bin_l.item(), "Multi-class Loss:", multi_l.item())
                return alpha * bin_l + (1 - alpha) * multi_l

            return composite_loss
            
        elif self.target_mode == "multi_label":
             # Weighted Multi-Label Loss
             # Targets are [N, 3] -> Fund, Acq, IPO
             # Weights are config-based task weights
             
             w_fund = float(self.config["data_processing"].get("multi_label", {}).get("loss_weights", {}).get("funding", 1.0))
             w_acq = float(self.config["data_processing"].get("multi_label", {}).get("loss_weights", {}).get("acquisition", 5.0))
             w_ipo = float(self.config["data_processing"].get("multi_label", {}).get("loss_weights", {}).get("ipo", 10.0))
             
             print(f"📊 Loss Function: Weighted Multi-Label (Fund={w_fund}, Acq={w_acq}, IPO={w_ipo})")
             
             # Calculate class weights for each task if enabled
             pos_weights = [1.0, 1.0, 1.0] # Default [Fund, Acq, IPO]
             
             if use_class_weights:
                 print("⚖️  Calculating Class Weights (N_neg / N_pos) for Multi-Label Tasks...")
                 try:
                     # Get targets from the startup node
                     # y is [N, 3] in multi_label mode
                     y_all = self.data["startup"].y
                     
                     # Only use training mask for weight calculation to avoid leakage
                     train_mask = self.data["startup"].train_mask
                     y_train = y_all[train_mask]
                     
                     for i, task_name in enumerate(["Funding", "Acquisition", "IPO"]):
                         n_pos = (y_train[:, i] == 1).sum().item()
                         n_neg = (y_train[:, i] == 0).sum().item()
                         
                         if n_pos > 0:
                             pos_weights[i] = n_neg / n_pos
                         else:
                             print(f"   ⚠️ No positive samples for {task_name} in training set! Weight set to 1.0")
                             pos_weights[i] = 1.0
                             
                         print(f"   {task_name}: Pos={n_pos}, Neg={n_neg} -> Weight={pos_weights[i]:.2f}")
                         
                 except Exception as e:
                     print(f"⚠️ Failed to calculate class weights dynamically: {e}")
                     print("   Using default weights (1.0)")

             # Move weights to device
             pos_weight_fund = torch.tensor(pos_weights[0]).to(device)
             pos_weight_acq = torch.tensor(pos_weights[1]).to(device)
             pos_weight_ipo = torch.tensor(pos_weights[2]).to(device)

             # Helper to create weighted loss components
             def multi_label_weighted_loss(output, targets):
                 # Output is { 'multi_label_output': [N, 3], ... }
                 # Targets is [N, 3]
                 
                 preds = output["multi_label_output"]
                 
                 # Task 1: Funding (Index 0)
                 # BCEWithLogitsLoss accepts pos_weight
                 loss_fund = F.binary_cross_entropy_with_logits(
                     preds[:, 0], targets[:, 0], pos_weight=pos_weight_fund
                 )
                 
                 # Task 2: Acquisition (Index 1)
                 loss_acq = F.binary_cross_entropy_with_logits(
                     preds[:, 1], targets[:, 1], pos_weight=pos_weight_acq
                 )
                 
                 # Task 3: IPO (Index 2)
                 loss_ipo = F.binary_cross_entropy_with_logits(
                     preds[:, 2], targets[:, 2], pos_weight=pos_weight_ipo
                 )
                 
                 total_loss = (w_fund * loss_fund) + (w_acq * loss_acq) + (w_ipo * loss_ipo)
                 
                 return total_loss

             return multi_label_weighted_loss

        elif self.target_mode == "masked_multi_task":
            # Masked Dual-Task Loss
            w_mom = float(self.config["train"]["loss"].get("momentum_weight", 1.0))
            w_liq = float(self.config["train"]["loss"].get("liquidity_weight", 1.0))
            w_ret = float(self.config["train"]["loss"].get("retrieval_weight", 0.0))

            normalize_retrieval_loss = self.config["train"]["loss"].get("normalize_retrieval_loss", False)
            ret_loss_ema = [None]  # mutable closure cell for EMA state
            ema_decay = 0.99

            print(f"📊 Loss Function: Masked Dual-Task (Mom={w_mom}, Liq={w_liq}) | Weighted BCE")
            if normalize_retrieval_loss and w_ret > 0:
                print(f"📊 Retrieval loss normalization: ON (EMA decay={ema_decay})")

            # Initialize BCE loss functions (will be updated with class weights if enabled)
            mom_bce_loss_fn = BCEWithLogitsLoss(reduction='none')
            liq_bce_loss_fn = BCEWithLogitsLoss(reduction='none')

            if use_class_weights:
                 try:
                     y_all = self.data["startup"].y
                     train_mask = self.data["startup"].train_mask
                     y_train = y_all[train_mask]

                     # Calculate weight for Momentum (index 0, mask at index 2)
                     mom_target = y_train[:, 0]
                     mom_mask = y_train[:, 2]
                     valid_mom = mom_mask == 1
                     if valid_mom.sum() > 0:
                         valid_targets = mom_target[valid_mom]
                         n_pos = (valid_targets == 1).sum().item()
                         n_neg = (valid_targets == 0).sum().item()
                         if n_pos > 0:
                             mom_pos_weight = torch.tensor(n_neg / n_pos).to(device)
                             mom_bce_loss_fn = BCEWithLogitsLoss(pos_weight=mom_pos_weight, reduction='none')
                             print(f"⚖️  Momentum Class Weight: {mom_pos_weight.item():.2f} (Pos={n_pos}, Neg={n_neg})")

                     # Calculate weight for Liquidity (index 1, mask at index 3)
                     liq_target = y_train[:, 1]
                     liq_mask = y_train[:, 3]
                     valid_liq = liq_mask == 1
                     if valid_liq.sum() > 0:
                         valid_targets = liq_target[valid_liq]
                         n_pos = (valid_targets == 1).sum().item()
                         n_neg = (valid_targets == 0).sum().item()
                         if n_pos > 0:
                             liq_pos_weight = torch.tensor(n_neg / n_pos).to(device)
                             liq_bce_loss_fn = BCEWithLogitsLoss(pos_weight=liq_pos_weight, reduction='none')
                             print(f"⚖️  Liquidity Class Weight: {liq_pos_weight.item():.2f} (Pos={n_pos}, Neg={n_neg})")
                 except Exception as e:
                     print(f"⚠️ Failed to calculate class weights: {e}")

            def masked_dual_loss(output, targets, batch_features=None):
                # Output: dict with 'out_mom', 'out_liq'
                # Targets: [N, 4] -> Mom, Liq, MaskMom, MaskLiq
                
                y_mom = targets[:, 0]
                y_liq = targets[:, 1]
                mask_mom = targets[:, 2]
                mask_liq = targets[:, 3]
                
                pred_mom = output["out_mom"]
                pred_liq = output["out_liq"]
                
                # --- Momentum Loss (Weighted BCE) ---
                valid_mom = mask_mom == 1
                if valid_mom.sum() > 0:
                    loss_mom_vec = mom_bce_loss_fn(pred_mom[valid_mom], y_mom[valid_mom])
                    loss_mom = loss_mom_vec.mean()
                else:
                    loss_mom = torch.tensor(0.0, device=device)

                # --- Liquidity Loss (Weighted BCE) ---
                valid_liq = mask_liq == 1
                if valid_liq.sum() > 0:
                    loss_liq_vec = liq_bce_loss_fn(pred_liq[valid_liq], y_liq[valid_liq])
                    loss_liq = loss_liq_vec.mean()
                else:
                    loss_liq = torch.tensor(0.0, device=device)
                
                total_loss = w_mom * loss_mom + w_liq * loss_liq
                
                # --- Retrieval Loss (Configurable: Distillation or Contrastive or ArcFace) ---
                loss_ret = torch.tensor(0.0, device=device)
                if batch_features is not None and w_ret > 0:
                     retrieval_loss_type = self.config["train"]["loss"].get("retrieval_loss_type", "distillation")
                     
                     # Extract GNN embeddings
                     # Prefer retrieval_embedding if available
                     if "retrieval_embedding" in output and output["retrieval_embedding"] is not None and "startup" in output["retrieval_embedding"]:
                         pred_emb = output["retrieval_embedding"]["startup"]
                     else:
                         pred_emb = output.get("embedding", None)
                         if isinstance(pred_emb, dict) and "startup" in pred_emb:
                             pred_emb = pred_emb["startup"]
                     
                     if pred_emb is not None:
                         # Normalize embeddings
                         pred_norm = F.normalize(pred_emb, p=2, dim=1)
                         
                         if retrieval_loss_type == "arcface":
                             # === ARCFACE LOSS ===
                             if "arcface_logits" in output and output["arcface_logits"] is not None and "startup" in output["arcface_logits"]:
                                 arc_logits = output["arcface_logits"]["startup"]
                                 if targets.shape[1] >= 5:
                                     classes = targets[:, 4].long()
                                     ce_loss = F.cross_entropy(arc_logits, classes)
                                     loss_ret = ce_loss
                             else:
                                 # Fallback if logits missing
                                 pass # loss_ret remains 0.0

                         elif retrieval_loss_type == "contrastive":
                             # === CONTRASTIVE LEARNING (TripletLoss) ===
                             # Check Config for Positive Source
                             positive_source = self.config["train"]["loss"].get("contrastive_positive_source", "text")
                             
                             margin = self.config["train"]["loss"].get("contrastive_margin", 0.5)
                             n_negatives = self.config["train"]["loss"].get("contrastive_negatives_per_anchor", 4)
                             
                             batch_size = pred_norm.shape[0]
                             n_anchors = min(batch_size // 4, 500)
                             anchor_indices = torch.randperm(batch_size, device=device)[:n_anchors]
                             
                             triplet_losses = []
                             
                             if positive_source == "label":
                                 # === SUPERVISED CONTRASTIVE (Implicit Stage-Hardening) ===
                                 if targets.shape[1] >= 5:
                                     labels = targets[:, 4].long()
                                     n_triplets = 0
                                     
                                     for anchor_idx in anchor_indices:
                                         anchor_emb = pred_norm[anchor_idx]
                                         anchor_label = labels[anchor_idx]

                                         
                                         # 1. Select Positive (Same Label)
                                         # Mask of candidates with same label (excluding self)
                                         same_label_mask = (labels == anchor_label)
                                         same_label_mask[anchor_idx] = False
                                         
                                         pos_indices = same_label_mask.nonzero(as_tuple=True)[0]
                                         
                                         if len(pos_indices) == 0:
                                              continue # Skip anchor if no positive pair in batch
                                        
                                         # Pick random positive
                                         pos_idx = pos_indices[torch.randint(0, len(pos_indices), (1,)).item()]
                                         pos_emb = pred_norm[pos_idx]
                                         
                                         # 2. Select Negatives (Hard Negatives)
                                         # Logic: High GNN Similarity BUT Different Label
                                         # Calculate Similarity
                                         anchor_gnn_sim = F.cosine_similarity(anchor_emb.unsqueeze(0), pred_norm, dim=1)
                                         
                                         # Mask out same-label samples (they are valid positives, not negatives)
                                         # We only want negatives from DIFFERENT classes
                                         valid_neg_mask = (labels != anchor_label)
                                         
                                         # Filter similarities
                                         # Set invalid negatives to -1 so they aren't picked
                                         masked_sim = anchor_gnn_sim.clone()
                                         masked_sim[~valid_neg_mask] = -1.0
                                         
                                         # Pick top K hardest negatives
                                         neg_candidates = masked_sim.argsort(descending=True)[:n_negatives]
                                         
                                         for neg_idx in neg_candidates:
                                             if not valid_neg_mask[neg_idx]: continue # Safety check
                                             
                                             neg_emb = pred_norm[neg_idx]
                                             
                                             pos_dist = 1 - F.cosine_similarity(anchor_emb.unsqueeze(0), pos_emb.unsqueeze(0))
                                             neg_dist = 1 - F.cosine_similarity(anchor_emb.unsqueeze(0), neg_emb.unsqueeze(0))
                                             
                                             # Loss
                                             triplet_loss = torch.clamp(margin + pos_dist - neg_dist, min=0.0)
                                             triplet_losses.append(triplet_loss)
                                             n_triplets += 1
                                             
                                     if n_triplets == 0 and torch.rand(1).item() < 0.01:
                                          print(f"⚠️ [Contrastive] No triplets found! Batch size: {batch_size}, Anchors: {len(anchor_indices)}")
                                 else:
                                      if torch.rand(1).item() < 0.01:
                                          print(f"⚠️ [Contrastive] Label source requested but targets shape is {targets.shape} (Need >= 5)")
                                      pass

                             else:
                                 # === UNSUPERVISED / TEXT CONTRASTIVE ===
                                 desc_dim = self.config["data_processing"].get("description_embedding_dim", 64)
                                 text_emb = batch_features[:, -desc_dim:]
                                 text_norm = F.normalize(text_emb, p=2, dim=1)

                                 for anchor_idx in anchor_indices:
                                     anchor_emb = pred_norm[anchor_idx]
                                     anchor_gnn_sim = F.cosine_similarity(anchor_emb.unsqueeze(0), pred_norm, dim=1)
                                     
                                     # Positive: Most similar in TEXT space
                                     anchor_text = text_norm[anchor_idx].unsqueeze(0)
                                     anchor_text_sim = F.cosine_similarity(anchor_text, text_norm, dim=1)
                                     anchor_text_sim[anchor_idx] = -1
                                     pos_idx = anchor_text_sim.argmax()
                                     
                                     # Negatives: Most similar in GNN space (Hard Negatives)
                                     anchor_gnn_sim[anchor_idx] = -1
                                     neg_candidates = anchor_gnn_sim.argsort(descending=True)[:n_negatives]
                                     
                                     pos_emb = pred_norm[pos_idx]
                                     for neg_idx in neg_candidates:
                                         neg_emb = pred_norm[neg_idx]
                                         pos_dist = 1 - F.cosine_similarity(anchor_emb.unsqueeze(0), pos_emb.unsqueeze(0))
                                         neg_dist = 1 - F.cosine_similarity(anchor_emb.unsqueeze(0), neg_emb.unsqueeze(0))
                                         triplet_loss = torch.clamp(margin + pos_dist - neg_dist, min=0.0)
                                         triplet_losses.append(triplet_loss)
                             
                             if triplet_losses:
                                 loss_ret = torch.stack(triplet_losses).mean()
                                 if torch.rand(1).item() < 0.01:
                                     print(f"  ✅ [Contrastive] ({positive_source}) L_Triplet: {loss_ret.item():.6f}")

                         elif retrieval_loss_type == "distillation":
                             # === DISTILLATION ===
                             desc_dim = self.config["data_processing"].get("description_embedding_dim", 64)
                             target_emb = batch_features[:, -desc_dim:]
                             target_norm = F.normalize(target_emb, p=2, dim=1)
                             
                             if pred_norm.shape[1] == target_norm.shape[1]:
                                 ret_loss_fn = torch.nn.CosineEmbeddingLoss()
                                 target_ones = torch.ones(pred_norm.shape[0], device=device)
                                 loss_ret = ret_loss_fn(pred_norm, target_norm, target_ones)
                                 if torch.rand(1).item() < 0.01:
                                     print(f"  ✅ [Distill] L_Ret: {loss_ret.item():.6f}")

                         else:
                             print(f"⚠️  Unknown retrieval_loss_type: {retrieval_loss_type}")

                         # Add weighted retrieval loss to total_loss.
                         # When detach_retrieval_head=True, the backbone is already
                         # protected by x_in.detach() in the model forward pass.
                         # We still add the loss so the retrieval head itself trains.
                         if normalize_retrieval_loss and loss_ret.item() > 0:
                             # EMA-normalize: divide by running average so w_ret
                             # becomes a scale-invariant importance weight.
                             raw = loss_ret.detach().item()
                             if ret_loss_ema[0] is None:
                                 ret_loss_ema[0] = raw  # init on first step
                             else:
                                 ret_loss_ema[0] = ema_decay * ret_loss_ema[0] + (1 - ema_decay) * raw
                             loss_ret_normalized = loss_ret / (ret_loss_ema[0] + 1e-8)
                             total_loss += w_ret * loss_ret_normalized
                         else:
                             total_loss += w_ret * loss_ret
                         
                         # --- Diversity Regularization ---
                         w_div = self.config["train"]["loss"].get("diversity_weight", 0.0)
                         if w_div > 0:
                             batch_std = pred_norm.std(dim=0).mean()
                             loss_diversity = -torch.log(batch_std + 1e-8)
                             total_loss += w_div * loss_diversity
                             if torch.rand(1).item() < 0.01:
                                 print(f"  📊 [Diversity] L_Div: {loss_diversity.item():.6f}")

                     else:
                         pass # pred_emb is None

                # Check if vars exist (safe fallback)
                l_ret = loss_ret if 'loss_ret' in locals() else torch.tensor(0.0, device=device)
                l_div = loss_diversity if 'loss_diversity' in locals() else torch.tensor(0.0, device=device)
                
                # Compute effective retrieval contribution for logging
                if normalize_retrieval_loss and 'loss_ret' in locals() and loss_ret.item() > 0 and ret_loss_ema[0] is not None:
                    ret_contribution = (w_ret * loss_ret.detach() / (ret_loss_ema[0] + 1e-8)).item()
                else:
                    ret_contribution = (w_ret * l_ret).item()

                loss_components = {
                    "Momentum": (w_mom * loss_mom).item(),
                    "Liquidity": (w_liq * loss_liq).item(),
                    "Retrieval": ret_contribution,
                    "Diversity": (w_div * l_div).item() if 'w_div' in locals() and w_div > 0 else 0.0
                }
                
                return total_loss, loss_components
                
            return masked_dual_loss
            
        else:
            raise ValueError(f"Unsupported target_mode: {self.target_mode}")

    def train_xgboost(self):
        print("Training XGBoost model...")
        
        # Extract features and labels
        # Using startup features only as a baseline
        data = self.data
        X = data['startup'].x.cpu().numpy()
        y = data['startup'].y
        # Handle tuple targets if any (multi-task)
        if isinstance(y, tuple):
            if self.target_mode == "multi_prediction":
                y = y[0] # multi-class labels
            elif self.target_mode == "binary_prediction":
                y = y[1] # binary labels
            else:
                y = y[0] # Fallback
        
        tasks_to_run = []
        if self.target_mode == "masked_multi_task":
            # Define tasks: (Name, TargetIndex, MaskIndex)
            tasks_to_run = [
                {"name": "Momentum", "target_idx": 0, "mask_idx": 2},
                {"name": "Liquidity", "target_idx": 1, "mask_idx": 3}
            ]
        else:
            # Default single task
            tasks_to_run = [{"name": "Default", "target_idx": None, "mask_idx": None}]
            
        for task in tasks_to_run:
            print(f"\n{'='*40}")
            print(f"🚀 TRAINING XGBOOST TASK: {task['name']}")
            print(f"{'='*40}")
            
            # Reset data pointers for each iteration
            X_curr = X.copy() 
            y_curr = y.clone() if torch.is_tensor(y) else y.copy()
            data_curr = data['startup'] # lightweight ref
            
            if self.target_mode == "masked_multi_task":
                y_tensor = y_curr if torch.is_tensor(y_curr) else torch.tensor(y_curr)
                y_target = y_tensor[:, task['target_idx']].cpu().numpy()
                mask = y_tensor[:, task['mask_idx']].cpu().numpy()
                
                print(f"Selecting '{task['name']}' (Index {task['target_idx']}) as target and applying Mask (Index {task['mask_idx']})...")
                
                valid_indices = (mask == 1)
                X_curr = X_curr[valid_indices]
                y_curr = y_target[valid_indices]
                
                train_mask = data_curr.train_mask.cpu().numpy()[valid_indices]
                val_mask = data_curr.val_mask.cpu().numpy()[valid_indices]
                test_mask = data_curr.test_mask.cpu().numpy()[valid_indices]
                
                print(f"Filtered dataset to {valid_indices.sum()} samples based on {task['name']} Mask.")
                
                # Filter split features if available
                X_val_full_curr = data_curr.x_val_mask.cpu().numpy()[valid_indices] if hasattr(data_curr, 'x_val_mask') else None
                X_test_full_curr = data_curr.x_test_mask.cpu().numpy()[valid_indices] if hasattr(data_curr, 'x_test_mask') else None
            else:
                y_curr = y_curr.cpu().numpy()
                train_mask = data_curr.train_mask.cpu().numpy()
                val_mask = data_curr.val_mask.cpu().numpy()
                test_mask = data_curr.test_mask.cpu().numpy()
                
                X_val_full_curr = data_curr.x_val_mask.cpu().numpy() if hasattr(data_curr, 'x_val_mask') else None
                X_test_full_curr = data_curr.x_test_mask.cpu().numpy() if hasattr(data_curr, 'x_test_mask') else None

            # Training features
            X_train = X_curr[train_mask]
            y_train = y_curr[train_mask]
            
            # Validation features
            if X_val_full_curr is not None:
                X_val = X_val_full_curr[val_mask]
            else:
                X_val = X_curr[val_mask]
            y_val = y_curr[val_mask]
                
            # Test features
            if X_test_full_curr is not None:
                X_test = X_test_full_curr[test_mask]
            else:
                X_test = X_curr[test_mask]
            y_test = y_curr[test_mask]
            
            # Adapter.fit() re-trains the sklearn model from scratch.
            self.model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        
            # Evaluate
            from sklearn.metrics import (
                accuracy_score, 
                roc_auc_score, 
                f1_score, 
                average_precision_score, 
                recall_score, 
                precision_score, 
                confusion_matrix
            )
            
            # Threshold Optimization
            best_threshold = 0.5
            optimize_threshold = self.config.get("calibration", {}).get("optimize_threshold", False)
            threshold_metric = self.config.get("calibration", {}).get("threshold_metric", "f1")

            if optimize_threshold and self.num_classes == 2:
                print(f"Optimizing threshold based on {threshold_metric}...")
                y_val_prob = self.model.predict_proba(X_val)
                if y_val_prob.shape[1] == 2:
                    y_val_prob = y_val_prob[:, 1]
                
                thresholds = np.arange(0.01, 1.00, 0.01)
                best_score = -1.0
                
                for thresh in thresholds:
                    y_val_pred_thresh = (y_val_prob >= thresh).astype(int)
                    if threshold_metric == "f1":
                        score = f1_score(y_val, y_val_pred_thresh, average='weighted') # Using weighted as per previous metrics
                    elif threshold_metric == "precision":
                        score = precision_score(y_val, y_val_pred_thresh, average='weighted', zero_division=0)
                    elif threshold_metric == "recall":
                        score = recall_score(y_val, y_val_pred_thresh, average='weighted', zero_division=0)
                    elif threshold_metric == "accuracy":
                        score = accuracy_score(y_val, y_val_pred_thresh)
                    else:
                        score = f1_score(y_val, y_val_pred_thresh, average='weighted')
                    
                    if score > best_score:
                        best_score = score
                        best_threshold = thresh
                
                print(f"Best Threshold found: {best_threshold:.4f} (Val {threshold_metric}: {best_score:.4f})")

            # Map task names to metric suffixes
            task_suffix = {"Momentum": "mom", "Liquidity": "liq", "Default": ""}

            def evaluate_set(X, y, name="Test", threshold=0.5):
                y_prob = self.model.predict_proba(X)

                # Handle multi-class vs binary for AUC and Thresholding
                if self.num_classes == 2:
                    # Check if y_prob has 2 columns
                    if y_prob.shape[1] == 2:
                        y_prob_pos = y_prob[:, 1]
                    else:
                        y_prob_pos = y_prob

                    # Apply threshold
                    y_pred = (y_prob_pos >= threshold).astype(int)

                    auc = roc_auc_score(y, y_prob_pos)
                    auc_pr = average_precision_score(y, y_prob_pos)
                else:
                    y_pred = self.model.predict(X) # Multi-class default
                    # For multi-class, we need to handle averaging
                    auc = roc_auc_score(y, y_prob, multi_class='ovr')
                    auc_pr = 0.0 # Placeholder

                acc = accuracy_score(y, y_pred)
                f1 = f1_score(y, y_pred, average='weighted')
                f1_binary = f1_score(y, y_pred) if self.num_classes == 2 else f1
                recall = recall_score(y, y_pred, average='weighted')
                precision = precision_score(y, y_pred, average='weighted')
                cm = confusion_matrix(y, y_pred)

                # Ranking metrics (PP@k, NDCG@k) — same as Evaluator._compute_metrics
                k_values = [5, 10, 20, 50, 100, 1000]
                pp_at_k = {}
                ndcg_at_k = {}
                if self.num_classes == 2:
                    y_np = np.asarray(y)
                    probs_np = np.asarray(y_prob_pos)
                    for k in k_values:
                        # PP@k: precision among k most positively predicted
                        k_eff = min(k, len(y_np))
                        if k_eff > 0:
                            top_k_idx = np.argpartition(probs_np, -k_eff)[-k_eff:]
                            pp_at_k[k] = np.sum(y_np[top_k_idx] == 1) / k_eff
                        else:
                            pp_at_k[k] = 0.0
                        # NDCG@k
                        ranked_idx = np.argsort(probs_np)[::-1][:k_eff]
                        ranked_rel = y_np[ranked_idx].astype(float)
                        discounts = np.log2(np.arange(2, k_eff + 2))
                        dcg = np.sum(ranked_rel / discounts)
                        ideal_rel = np.sort(y_np)[::-1][:k_eff].astype(float)
                        idcg = np.sum(ideal_rel / discounts)
                        ndcg_at_k[k] = dcg / idcg if idcg > 0 else 0.0
                    # NDCG@full
                    n = len(y_np)
                    ranked_idx = np.argsort(probs_np)[::-1]
                    ranked_rel = y_np[ranked_idx].astype(float)
                    discounts = np.log2(np.arange(2, n + 2))
                    dcg = np.sum(ranked_rel / discounts)
                    ideal_rel = np.sort(y_np)[::-1].astype(float)
                    idcg = np.sum(ideal_rel / discounts)
                    ndcg_at_k["full"] = dcg / idcg if idcg > 0 else 0.0

                print(f"XGBoost {name} Metrics (Threshold={threshold:.4f}):")
                print(f"Accuracy: {acc:.4f}")
                print(f"AUC: {auc:.4f}")
                print(f"AUC-PR: {auc_pr:.4f}")
                print(f"F1 Score: {f1:.4f}")
                print(f"F1 (binary): {f1_binary:.4f}")
                print(f"Recall: {recall:.4f}")
                print(f"Precision: {precision:.4f}")
                print(f"Confusion Matrix:\n{cm}")
                if pp_at_k:
                    print("\n--- Ranking Metrics ---")
                    for k in k_values:
                        print(f"PP@{k}: {pp_at_k[k]:.4f}  NDCG@{k}: {ndcg_at_k[k]:.4f}")
                    print(f"NDCG@full ({len(y)}): {ndcg_at_k['full']:.4f}")

                return {"auc": auc, "auc_pr": auc_pr, "acc": acc, "f1": f1,
                        "f1_binary": f1_binary,
                        "recall": recall, "precision": precision,
                        "pp_at_k": pp_at_k, "ndcg_at_k": ndcg_at_k}

            val_metrics = evaluate_set(X_val, y_val, name="Validation", threshold=best_threshold)
            print("-" * 30)
            test_metrics = evaluate_set(X_test, y_test, name="Test", threshold=best_threshold)

            # Log to W&B
            if self.use_wandb:
                import wandb
                suffix = task_suffix.get(task["name"], "")
                prefix_val = f"val_{suffix}" if suffix else "val"
                prefix_test = f"test_{suffix}" if suffix else "test"
                wandb_log = {}
                for prefix, metrics in [(prefix_val, val_metrics), (prefix_test, test_metrics)]:
                    wandb_log[f"{prefix}_auc_roc"] = metrics["auc"]
                    wandb_log[f"{prefix}_auc_pr"] = metrics["auc_pr"]
                    wandb_log[f"{prefix}_accuracy"] = metrics["acc"]
                    wandb_log[f"{prefix}_f1"] = metrics["f1"]
                    wandb_log[f"{prefix}_f1_binary"] = metrics["f1_binary"]
                    # Ranking metrics
                    for k, v in metrics.get("pp_at_k", {}).items():
                        wandb_log[f"{prefix}_precision_at_k_pos_pred_{k}"] = v
                    for k, v in metrics.get("ndcg_at_k", {}).items():
                        wandb_log[f"{prefix}_ndcg_at_{k}"] = v
                wandb.log(wandb_log)
                # Also log with standard metric names for sweep optimization
                if suffix:
                    wandb.log({
                        f"val_auc_pr_{suffix}": val_metrics["auc_pr"],
                        f"test_auc_pr_{suffix}": test_metrics["auc_pr"],
                        f"val_auc_roc_{suffix}": val_metrics["auc"],
                        f"test_auc_roc_{suffix}": test_metrics["auc"],
                    })
                    # Stash per-task values so we can emit a joint metric
                    # after both mom and liq tasks have been evaluated.
                    if not hasattr(self, "_xgb_joint_accum"):
                        self._xgb_joint_accum = {"val": {}, "test": {}}
                    self._xgb_joint_accum["val"][suffix] = val_metrics
                    self._xgb_joint_accum["test"][suffix] = test_metrics

        # After all tasks in this XGBoost run, emit the joint (mom+liq)/2
        # AUC-PR / AUC-ROC for sweep selection consistency with GNN sweeps.
        if self.use_wandb and getattr(self, "_xgb_joint_accum", None):
            import wandb
            for mode in ("val", "test"):
                acc = self._xgb_joint_accum.get(mode, {})
                if "mom" in acc and "liq" in acc:
                    joint_log = {
                        f"{mode}_auc_pr_joint": 0.5 * (acc["mom"]["auc_pr"] + acc["liq"]["auc_pr"]),
                        f"{mode}_auc_roc_joint": 0.5 * (acc["mom"]["auc"] + acc["liq"]["auc"]),
                    }
                    wandb.log(joint_log)

        return
    def train(self):
        if isinstance(self.model, XGBoostAdapter):
            return self.train_xgboost()
            
        # Check for baselines
        from .models import RandomBaseline, DegreeCentralityBaseline, LLMBaseline
        if isinstance(self.model, (RandomBaseline, DegreeCentralityBaseline, LLMBaseline)):
            print(f"🚀 Running Baseline Model: {type(self.model).__name__}")
            print("Skipping training loop...")

            # LLM baseline: skip val and post-hoc calibration (only prompt-level calibration)
            # This avoids running LLM inference twice and the target_mode mismatch in calibration
            if isinstance(self.model, LLMBaseline):
                print("📝 LLM Baseline: Skipping val evaluation and post-hoc calibration")
                print("   (prompt-level calibration is controlled via models.LLM.use_calibration)")
            else:
                if isinstance(self.model, DegreeCentralityBaseline):
                    eval_data = self.val_loader if self.val_loader is not None else self.data
                else:
                    eval_data = self.val_loader if self.val_loader is not None else self.data

                # Evaluate on Validation
                print("\nEvaluating on Validation Set...")
                val_metric, _ = self.evaluator.evaluate(
                    graph_data=eval_data,
                    target_mode=self.target_mode,
                    mode="val",
                    current_epoch=0,
                )

                # Calibration and Threshold Optimization
                should_optimize = True
                if isinstance(self.model, RandomBaseline):
                    should_optimize = False
                    print("⚠️ Skipping threshold optimization for RandomBaseline as requested.")

                if should_optimize and self.config.get("calibration", {}).get("enabled", False) and self.num_classes == 2:
                    print("\n🔄 Running Calibration and Threshold Optimization...")
                    try:
                        from src.ml.calibration import analyze_model_calibration_from_predictions

                        # For masked_multi_task, calibrate momentum task by default
                        calib_task_prefix = None
                        if self.target_mode == "masked_multi_task":
                            calib_task_prefix = "mom"
                        elif self.target_mode == "multi_label":
                            calib_task_prefix = "mom"

                        results, calibrator = analyze_model_calibration_from_predictions(
                            self.evaluator,
                            self.data,
                            modes=['val'],
                            optimize_threshold=self.config["calibration"].get("optimize_threshold", True),
                            threshold_metric=self.config["calibration"].get("threshold_metric", "f1"),
                            calibration_method=self.config.get("calibration", {}).get("method", "platt"),
                            target_mode=self.target_mode,
                            task_prefix=calib_task_prefix,
                        )

                        if calibrator:
                            self.evaluator.set_calibrator(calibrator)
                            self.evaluator.set_optimal_threshold(calibrator.optimal_threshold)
                            if hasattr(calibrator, 'optimal_threshold_percentile'):
                                self.evaluator.set_optimal_threshold_percentile(calibrator.optimal_threshold_percentile)

                            print(f"✅ Evaluator updated with optimal threshold: {calibrator.optimal_threshold:.4f}")
                    except ImportError:
                        print("⚠️ Calibration module not found or failed to import.")
                    except Exception as e:
                        print(f"⚠️ Calibration failed: {e}")
                        import traceback
                        traceback.print_exc()

            # Evaluate on Test
            print("\nEvaluating on Test Set...")
            if isinstance(self.model, LLMBaseline):
                test_eval_data = self.data
            else:
                test_eval_data = self.test_loader if self.test_loader is not None else self.data
            self.evaluator.evaluate(
                graph_data=test_eval_data,
                target_mode=self.target_mode,
                mode="test",
                current_epoch=0,
            )
            return 0.0 # Dummy loss

        data = self.data.to(self.device)

        # SeHGNN optimization: pre-aggregate neighbor features once
        if hasattr(self.model, 'precompute') and not self.train_loader:
            self.model.precompute(data.x_dict, data.edge_index_dict)

        for epoch in range(1, self.epochs + 1):
            self.model.train()
            self.current_epoch = epoch
            total_loss = 0
            total_examples = 0

            if self.train_loader:
                for batch_idx, batch in enumerate(tqdm(self.train_loader, desc=f"Epoch {epoch:03d}")):
                    batch = batch.to(self.device)
                    self.optimizer.zero_grad()
                    batch_size = batch["startup"].batch_size

                    # Extract retrieval labels if needed (for ArcFace)
                    retrieval_labels = None
                    if self.target_mode == "masked_multi_task":
                        y_batch = batch["startup"].y
                        # Check logic similar to full-batch but adapted for PyG loader
                        if y_batch.shape[1] >= 5:
                            # For ArcFace to work on Forward, it should match x.
                            # PyG NeighborLoader usually loads y for all nodes (seeds + neighbors)
                            # so we pass correct size.
                            retrieval_labels = y_batch[:, 4].long()

                    # Get model outputs (only pass retrieval_labels to models that support it)
                    fwd_kwargs = {}
                    if retrieval_labels is not None and hasattr(self.model, 'retrieval_proj'):
                        fwd_kwargs['retrieval_labels'] = retrieval_labels
                    output = self.model(batch.x_dict, batch.edge_index_dict, **fwd_kwargs)

                    if self.target_mode == "multi_task":
                        out = {
                            "binary": output["binary_output"]["startup"][:batch_size].view(-1),
                            "multi_class": output["multi_class_output"]["startup"][
                                :batch_size
                            ],
                        }
                    elif self.target_mode == "binary_prediction":
                        out = output["startup"]["output"][:batch_size].view(-1)
                    elif self.target_mode == "multi_prediction":
                        out = output["startup"]["output"][:batch_size]

                    # Prepare targets
                    y_data = batch["startup"].y
                    if isinstance(y_data, tuple):
                        y_multi = y_data[0][:batch_size]
                        y_binary = y_data[1][:batch_size]
                    else:
                        y = y_data[:batch_size]

                    # Compute loss
                    if self.target_mode == "multi_task":
                        if isinstance(y_data, tuple):
                            targets = {
                                "binary": y_binary.float(),
                                "multi_class": y_multi.long(),
                            }
                        else:
                            targets = {
                                "binary": y[1][:batch_size].float(),
                                "multi_class": y[0][:batch_size].long(),
                            }
                        loss = self.loss(out, targets)

                    elif self.target_mode == "binary_prediction":
                        if isinstance(y_data, tuple):
                            loss = self.loss(out, target=y_binary.float())
                        else:
                            loss = self.loss(out, target=y.float())

                    elif self.target_mode == "multi_prediction":
                        if isinstance(y_data, tuple):
                            labels = y_multi
                        else:
                            labels = y

                        labels = labels[:batch_size].long()
                        status_changed = torch.tensor(
                            batch["startup"].status_changed[:batch_size].values,
                            dtype=torch.float32,
                        ).to(self.device)

                        sample_weights = torch.ones_like(status_changed)
                        sample_weights[status_changed == 1.0] = self.config["train"][
                            "loss"
                        ]["status_change_weight"]

                        loss = self.loss(out, labels, sample_weights=sample_weights)

                    elif self.target_mode == "multi_label":
                        # Multi-label case
                        if isinstance(y_data, tuple):
                             target = y_data[0] 
                        else:
                             target = y_data # [N, 3] tensor

                    # Backpropagation
                    if self.target_mode == "multi_label":
                         loss = self.loss(output, target[:batch_size])

                    elif self.target_mode == "masked_multi_task":
                        # Masked Dual-Task
                        
                        out_mom = output["out_mom"][:batch_size]
                        out_liq = output["out_liq"][:batch_size]
                        
                        # Distillation/ArcFace: Extract embedding and logits
                        out_emb = None
                        if "embedding" in output and "startup" in output["embedding"]:
                            out_emb = output["embedding"]["startup"][:batch_size]
                            
                        arcface_logits = None
                        if "arcface_logits" in output and output["arcface_logits"] is not None and "startup" in output["arcface_logits"]:
                             arcface_logits = output["arcface_logits"]["startup"][:batch_size]
                            
                        targets = y_data[:batch_size] # [Batch, 5]
                        
                        batch_features = batch["startup"].x[:batch_size]
                        
                        masked_output = {
                            "out_mom": out_mom,
                            "out_liq": out_liq,
                            "embedding": {"startup": out_emb} if out_emb is not None else None,
                            "arcface_logits": {"startup": arcface_logits} if arcface_logits is not None else None
                        }
                        
                        ret = self.loss(masked_output, targets, batch_features=batch_features)
                        if isinstance(ret, tuple):
                             loss, components = ret
                        else:
                             loss = ret
                             components = {}
                    
                    loss.backward()
                    
                    # Gradient clipping
                    if self.gradient_clip_val is not None:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip_val)
                    
                    self.optimizer.step()
                    total_loss += loss.item() * batch_size
                    total_examples += batch_size
                    
                    # Debug Gate Values
                    if hasattr(self.model, "last_gate_mean") and batch_idx % 50 == 0:
                        print(f"  Gate Mean (z): {self.model.last_gate_mean:.4f} (1.0=Residual, 0.0=Graph)")
                        if self.use_wandb:
                            wandb.log({"gate_mean": self.model.last_gate_mean})

                avg_loss = total_loss / total_examples

            else:
                # Full-batch training
                self.optimizer.zero_grad()
                
                # Extract retrieval labels if needed (for ArcFace)
                retrieval_labels = None
                if self.target_mode == "masked_multi_task":
                    y_all = data["startup"].y
                    if y_all.shape[1] >= 5:
                        retrieval_labels = y_all[:, 4].long().to(self.device)

                # Only pass retrieval_labels to models that support it (e.g. SeHGNN)
                fwd_kwargs = {}
                if retrieval_labels is not None and hasattr(self.model, 'retrieval_proj'):
                    fwd_kwargs['retrieval_labels'] = retrieval_labels
                output = self.model(data.x_dict, data.edge_index_dict, **fwd_kwargs)

                train_mask = data["startup"].train_mask

                if self.target_mode == "multi_task":
                    out = {
                        "binary": output["binary_output"]["startup"][train_mask].view(-1),
                        "multi_class": output["multi_class_output"]["startup"][train_mask],
                    }
                elif self.target_mode == "binary_prediction":
                    if config['data_processing']['train']['use_batches']:
                        out = output["startup"]["output"][train_mask].view(-1)
                    else:
                        out = output["startup"]["output"][train_mask].view(-1)
                elif self.target_mode == "multi_prediction":
                    if config['data_processing']['train']['use_batches']:
                        out = output["startup"]["output"][train_mask]
                    else:
                        out = output["startup"]["output"][train_mask]
                elif self.target_mode == "multi_label":
                    # Multi-label case for full-batch
                    # output is dict with "multi_label_output"
                    pass # 'out' is not directly used for multi_label loss calculation, the full 'output' dict is.

                y_data = data["startup"].y
                if isinstance(y_data, (tuple, list)):
                    y_multi = y_data[0][train_mask]
                    y_binary = y_data[1][train_mask]
                else:
                    y = y_data[train_mask]

                if self.target_mode == "multi_task":
                    if isinstance(y_data, (tuple, list)):
                        targets = {
                            "binary": y_binary.float(),
                            "multi_class": y_multi.long(),
                        }
                    else:
                        targets = {
                            "binary": y[:, 0].float(),
                            "multi_class": y[:, 1].long(),
                        }
                    loss = self.loss(out, targets)

                elif self.target_mode == "multi_label":
                    # Multi-label full-batch
                    if isinstance(y_data, tuple):
                        target = y_data[0][train_mask]
                    else:
                        target = y_data[train_mask] # [N_train, 3]

                    masked_output = {
                        "multi_label_output": output["multi_label_output"]["startup"][train_mask]
                    }
                    loss = self.loss(masked_output, target)

                elif self.target_mode == "binary_prediction":
                    if isinstance(y_data, tuple):
                        loss = self.loss(out, target=y_binary.float())
                    else:
                        loss = self.loss(out, target=y.float())

                elif self.target_mode == "multi_prediction":
                    if isinstance(y_data, tuple):
                        labels = y_multi
                    else:
                        labels = y

                    labels = labels.long()
                    status_changed = torch.tensor(
                        data["startup"].status_changed.values[train_mask.cpu().numpy()],
                        dtype=torch.float32,
                    ).to(self.device)
                    sample_weights = torch.ones_like(status_changed)
                    sample_weights[status_changed == 1.0] = self.config["train"][
                        "loss"
                    ]["status_change_weight"]

                    loss = self.loss(out, labels, sample_weights=sample_weights)

                elif self.target_mode == "masked_multi_task":
                    # Masked Dual-Task
                    
                    # Extract outputs for training set
                    out_mom_train = output["out_mom"][train_mask]
                    out_liq_train = output["out_liq"][train_mask]
                    
                    # Distillation/ArcFace: Extract embedding and logits
                    out_emb_train = None
                    if "embedding" in output and "startup" in output["embedding"]:
                        out_emb_train = output["embedding"]["startup"][train_mask]
                    
                    # Retrieval: Extract retrieval embedding (separate from main embedding)
                    out_retrieval_emb_train = None
                    if "retrieval_embedding" in output and output["retrieval_embedding"] is not None and "startup" in output["retrieval_embedding"]:
                        out_retrieval_emb_train = output["retrieval_embedding"]["startup"][train_mask]
                        
                    arcface_logits_train = None
                    if "arcface_logits" in output and output["arcface_logits"] is not None and "startup" in output["arcface_logits"]:
                        arcface_logits_train = output["arcface_logits"]["startup"][train_mask]
                        
                    # Extract targets for training set
                    targets_train = y_data[train_mask] # [N_train, 5]
                    
                    # Features for Distillation Target
                    batch_features_train = data["startup"].x[train_mask]
                    
                    masked_output = {
                        "out_mom": out_mom_train,
                        "out_liq": out_liq_train,
                        "embedding": {"startup": out_emb_train} if out_emb_train is not None else None,
                        "retrieval_embedding": {"startup": out_retrieval_emb_train} if out_retrieval_emb_train is not None else None,
                        "arcface_logits": {"startup": arcface_logits_train} if arcface_logits_train is not None else None
                    }
                    
                    ret = self.loss(masked_output, targets_train, batch_features=batch_features_train)
                    if isinstance(ret, tuple):
                        loss, components = ret
                    else:
                        loss = ret
                        components = {}

                # Hetero2Net: add the pre-weighted disentanglement + masked-
                # metapath auxiliary loss (α·L_corr + β·L_rec, paper Eq. 14).
                # Internal alpha/beta already baked into model.aux_loss; the
                # config `aux_loss_weight` is a simple on/off scalar (default 1).
                aux_weight = self.model_config.get("aux_loss_weight", 1.0) if hasattr(self, "model_config") else 1.0
                if aux_weight > 0 and hasattr(self.model, "aux_loss") and torch.is_tensor(self.model.aux_loss) and self.model.aux_loss.requires_grad:
                    aux_term = aux_weight * self.model.aux_loss
                    loss = loss + aux_term
                    components = components or {}
                    components["aux"] = aux_term.item()
                    if hasattr(self.model, "corr_loss") and torch.is_tensor(self.model.corr_loss):
                        components["corr"] = float(self.model.corr_loss.detach())
                    if hasattr(self.model, "link_loss") and torch.is_tensor(self.model.link_loss):
                        components["link"] = float(self.model.link_loss.detach())

                loss.backward()

                # Gradient clipping
                if self.gradient_clip_val is not None:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip_val)
                
                self.optimizer.step()
                total_loss = loss.item() # Full batch has 1 step per epoch
                
                # Debug Gate Values
                if hasattr(self.model, "last_gate_mean") and epoch % 1 == 0:
                    print(f"  Gate Mean (z): {self.model.last_gate_mean:.4f} (1.0=Residual, 0.0=Graph)")
                    if self.use_wandb:
                        wandb.log({"gate_mean": self.model.last_gate_mean})

                avg_loss = total_loss

            loss_str = f"Epoch {epoch:03d}, Total Loss: {avg_loss:.4f}"
            if 'components' in locals() and components:
                loss_str += " | " + " | ".join([f"{k}: {v:.4f}" for k, v in components.items()])
            
            print(loss_str)

            # DEBUG: Print SeHGNN Attention Weights
            if isinstance(output, dict) and "attention_weights" in output and "metapath_names" in output:
                try:
                    attn = output["attention_weights"] # [Batch, Heads, M, M]
                    mp_names = output["metapath_names"]
                    
                    if hasattr(attn, "mean"):
                        # Mean over Batch(0) and Heads(1) -> [M, M] (Average Attention Matrix)
                        # Sum over Rows(0) -> [M] (Total attention RECEIVED by each metapath column)
                        importances = attn.mean(dim=(0, 1)).sum(dim=0)
                        total_mag = importances.sum()
                        
                        mp_weights = []
                        for idx, mp in enumerate(mp_names):
                            name = mp if isinstance(mp, str) else mp[1]
                            val = importances[idx]
                            pct = val / total_mag
                            mp_weights.append((name, pct.item()))
                        
                        mp_weights.sort(key=lambda x: x[1], reverse=True)
                        
                        print(f"  🔍 SeHGNN Weights (Top 5):")
                        for name, pct in mp_weights[:5]:
                            print(f"    {name:<35}: {pct:.4%}")
                        
                        min_pct = min(w for n, w in mp_weights)
                        max_pct = max(w for n, w in mp_weights)
                        print(f"  🔍 Weight Spread: Min={min_pct:.4%}, Max={max_pct:.4%}")
                except Exception as e:
                    print(f"  ⚠️ Error printing weights: {e}")

            if self.use_wandb:
                wandb.log({"epoch": epoch, "loss": avg_loss})

            # Evaluate on validation set
            val_metric, _ = self.evaluator.evaluate(
                graph_data=self.val_loader if self.val_loader is not None else self.data,
                target_mode=self.target_mode,
                mode="val",
                current_epoch=self.current_epoch,
                best_model_callback=self._update_best_model,
            )
            
            # Early stopping check.
            # `evaluator.best_metric` is updated to `val_metric` inside evaluate()
            # whenever this epoch improves, so comparing `val_metric > best + delta`
            # is always false after an improvement (they're equal). Use best_epoch
            # instead: if it equals the current epoch, this was the improving one.
            if self.early_stopping_enabled and epoch >= self.min_amount_of_epochs:
                improved = self.evaluator.best_epoch == epoch
                if improved:
                    self.early_stopping_counter = 0
                else:
                    self.early_stopping_counter += 1
                    
                print(f"🛑 Early stopping counter: {self.early_stopping_counter}/{self.early_stopping_patience}")
                
                if self.early_stopping_counter >= self.early_stopping_patience:
                    print(f"🛑 Early stopping triggered after {epoch} epochs (no improvement for {self.early_stopping_patience} epochs)")
                    break
            
            # Update learning rate scheduler
            if self.scheduler is not None:
                if isinstance(self.scheduler, ReduceLROnPlateau):
                    # ReduceLROnPlateau needs the metric value
                    self.scheduler.step(val_metric)
                    current_lr = self.optimizer.param_groups[0]['lr']
                    print(f"📊 Current Learning Rate: {current_lr:.2e}")
                    if self.use_wandb:
                        wandb.log({"learning_rate": current_lr})
                elif isinstance(self.scheduler, (CosineAnnealingLR, SequentialLR)):
                    self.scheduler.step()
                    current_lr = self.optimizer.param_groups[0]['lr']
                    print(f"📊 Current Learning Rate: {current_lr:.2e}")
                    if self.use_wandb:
                        wandb.log({"learning_rate": current_lr})

            # Log gamma value for SeHGNN
            if hasattr(self.model, 'semantic_fusion') and hasattr(self.model.semantic_fusion, 'gamma'):
                gamma_val = self.model.semantic_fusion.gamma.item()
                print(f"📊 Gamma: {gamma_val:.6f}")
                if self.use_wandb:
                    wandb.log({"gamma": gamma_val})

        # Save last model state for analysis (even if not the best)
        save_dir = self.config.get("output_dir", "outputs")
        suffix = self._get_run_suffix()
        last_model_path = os.path.join(save_dir, "models", f"{suffix}_last.pt")
        self.save_checkpoint(last_model_path)
        print(f"✅ Saved last model state to {last_model_path}")

        return avg_loss

    def _get_run_suffix(self):
        """Build a unique suffix from model name, seed, and W&B run ID."""
        model_name = self.config.get("train", {}).get("model", "unknown")
        seed = self.config.get("seed", 0)
        try:
            import wandb
            wandb_id = wandb.run.id if wandb.run else "local"
        except (ImportError, AttributeError):
            wandb_id = "local"
        return f"{model_name}_seed{seed}_{wandb_id}"

    def _update_best_model(self):
        print(
            f"Updating best model.\nOptimization Metric: {self.evaluator.optimization_metric_type}\nBest Value: {self.evaluator.best_metric}"
        )
        if isinstance(self.model, XGBoostAdapter):
            return
        self.best_model_dict = self.model.state_dict()

        # Save to disk with unique path to prevent overwrites across runs
        save_dir = self.config.get("output_dir", "outputs")
        suffix = self._get_run_suffix()
        model_path = os.path.join(save_dir, "models", f"{suffix}_best.pt")
        self.save_checkpoint(model_path)

    def save_checkpoint(self, path):
        """Save model checkpoint to path."""
        if isinstance(self.model, XGBoostAdapter):
            print(f"⏭️ Skipping checkpoint save for XGBoost (not a PyTorch model)")
            return
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict() if self.optimizer else None,
            'best_metric': self.evaluator.best_metric,
            'config': self.config
        }, path)
        print(f"💾 Saved best model checkpoint to {path}")

    def load_checkpoint(self, path):
        """Load model checkpoint from path."""
        if not os.path.exists(path):
            print(f"⚠️ Checkpoint not found at {path}")
            return False
            
        print(f"🔄 Loading checkpoint from {path}...")
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        if self.optimizer and checkpoint['optimizer_state_dict']:
            try:
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            except ValueError as e:
                print(f"⚠️ Warning: Could not load optimizer state (likely due to architecture change): {e}")
                print("Continuing with model weights only (safe for inference).")
        
        # Restore metrics if needed, but usually we just want the weights for inference
        print(f"✅ Loaded model from epoch {checkpoint.get('epoch', '?')} (Metric: {checkpoint.get('best_metric', '?')})")
        return True


    def evaluate_test(self):
        if isinstance(self.model, XGBoostAdapter):
            print("Skipping evaluate_test for XGBoost (metrics already reported during training).")
            return

        if self.test_best_model and self.current_epoch >= max(
            2, self.min_amount_of_epochs
        ):
            self.model.load_state_dict(self.best_model_dict)
            print(
                f"Loaded best model from epoch {self.evaluator.best_epoch} with {self.evaluator.optimization_metric_type}: {self.evaluator.best_metric}"
            )

        # Calibration Analysis
        if self.config.get("calibration", {}).get("enabled", False):
            self._perform_calibration_analysis()
            
        # Decision Boundary Visualization. Skip for LLM (no embeddings) and
        # when analysis.enable_visualization_analysis is false (sweep yamls
        # disable it to avoid end-of-epoch OOM on memory-constrained allocs).
        if not isinstance(self.model, LLMBaseline) and \
                self.config.get("analysis", {}).get("enable_visualization_analysis", True):
            self._perform_visualization_analysis()
        
        # HAN Metapath Weight Visualization
        if self.model_name == "HAN":
            try:
                print("\n" + "="*60)
                print("HAN METAPATH WEIGHT ANALYSIS")
                print("="*60)
                
                # Get weights using test data (or validation data)
                # We use the full data object which has masks
                weights = self.model.get_semantic_attention_weights(self.data.x_dict, self.data.edge_index_dict)
                
                # Print weights
                for layer, layer_weights in weights.items():
                    print(f"\n[{layer}]")
                    for node_type, node_weights in layer_weights.items():
                        print(f"  Node Type: {node_type}")
                        for metapath, weight in node_weights.items():
                            print(f"    {metapath}: {weight:.4f}")
                            
                # Visualize
                output_dir = self.config.get("output_dir", "outputs")
                visualize_metapath_weights(weights, output_dir)
                visualize_metapath_weights_vs_stats(weights, output_dir=output_dir)
                
                if self.use_wandb:
                    han_images = {}
                    
                    # Iterate to find all generated plots
                    for layer_name, layer_weights in weights.items():
                        for node_type in layer_weights:
                            # 1. Bar charts
                            bar_filename = f"metapath_weights_{layer_name}_{node_type}.pdf"
                            bar_path = os.path.join(output_dir, bar_filename)
                            if os.path.exists(bar_path):
                                han_images[f"han_weights/{layer_name}_{node_type}"] = wandb.Image(bar_path)
                                
                            # 2. Correlation plots (from visualize_metapath_weights_vs_stats)
                            
                            corr_types = ["weight_vs_homophily", "weight_vs_class_homophily", "weight_vs_edge_count"]
                            for ctype in corr_types:
                                fname = f"{ctype}_{layer_name}_{node_type}.pdf"
                                fpath = os.path.join(output_dir, fname)
                                if os.path.exists(fpath):
                                    han_images[f"han_analysis/{ctype}_{layer_name}_{node_type}"] = wandb.Image(fpath)

                    if han_images:
                        wandb.log(han_images)
                
            except Exception as e:
                print(f"⚠️ HAN weight visualization failed: {str(e)}")
                import traceback
                traceback.print_exc()

        self.evaluator.evaluate(
            graph_data=self.data,
            mode="test",
            target_mode=self.target_mode,
        )
        if self.config["data_processing"]["resample"]["test_original"] and self.config["data_processing"]["resample"]["enabled"]:
            self.evaluator.evaluate(
                graph_data=self.data,
                mode="test_original",
                target_mode=self.target_mode,
            )

        if self.explain:
            non_differentiable = (LLMBaseline, RandomBaseline, DegreeCentralityBaseline, XGBoostAdapter)
            if isinstance(self.model, non_differentiable):
                print(f"⏭️ Skipping explanation for {type(self.model).__name__} (no gradient-based attribution)")
            else:
                explain_model(
                    self.model,
                    self.data,
                    self.explain_path,
                    self.target_mode,
                    self.explain_sample_size,
                    self.explain_method,
                )

    def _perform_calibration_analysis(self):
        """
        Perform comprehensive calibration analysis and apply calibration correction.
        """
        print(f"\n{'='*60}")
        print("CALIBRATION ANALYSIS")
        print(f"{'='*60}")
        
        try:
            # Support binary_prediction, multi_task, multi_label, and masked_multi_task
            allowed_modes = ["binary_prediction", "multi_task", "multi_label", "masked_multi_task"]
            if self.target_mode not in allowed_modes:
                print(f"⚠️  Calibration analysis skippped for {self.target_mode} (only supports {allowed_modes})")
                return
                
            # Run calibration analysis
            optimize_threshold = self.config.get("calibration", {}).get("optimize_threshold", True)
            threshold_metric = self.config.get("calibration", {}).get("threshold_metric", "f1")
            calibration_method = self.config.get("calibration", {}).get("method", "platt")

            if self.target_mode == "multi_label":
                 if self.calibrators is None:
                     self.calibrators = {}
                 
                 # Task mapping: config name to prediction prefix
                 tasks = [("funding", "fund"), ("acquisition", "acq"), ("ipo", "ipo")]
                 all_results = {}
                 
                 for task_name, task_prefix in tasks:
                      print(f"\n   ⚙️ Calibrating Task: {task_name.upper()} ({task_prefix})")
                      try:
                          results, calibrator = analyze_model_calibration_from_predictions(
                               evaluator=self.evaluator,
                               graph_data=self.data,
                               modes=['val', 'test'],
                               optimize_threshold=optimize_threshold,
                               threshold_metric=threshold_metric,
                               target_mode="multi_label",
                               task_prefix=task_prefix,
                               calibration_method=calibration_method,
                          )
                          self.calibrators[task_name] = calibrator
                          all_results[task_name] = results
                          print(f"   ✅ Calibrated {task_name}")
                      except Exception as e:
                          print(f"   ⚠️ Failed to calibrate {task_name}: {e}")
                          
                 # Update evaluator with new calibrators
                 if hasattr(self.evaluator, "set_calibrator"):
                     self.evaluator.set_calibrator(self.calibrators) 

                 # Log summary for multi-label
                 print(f"\n📊 CALIBRATION SUMMARY (Multi-Label):")
                 calibration_log = {}
                 for task_name, res in all_results.items():
                     print(f"   --- {task_name.upper()} ---")
                     for mode, data in res.items():
                        if 'metrics' in data:
                            metrics = data['metrics']
                            print(f"   {mode.upper()}:")
                            print(f"      ECE: {metrics['ece']:.4f}")
                            print(f"      MCE: {metrics['mce']:.4f}")
                            print(f"      Brier Score: {metrics['brier_score']:.4f}")
                            
                            prefix = task_name # e.g. funding
                            calibration_log[f"calibration_{prefix}_{mode}_ece"] = metrics['ece']
                            calibration_log[f"calibration_{prefix}_{mode}_mce"] = metrics['mce']
                            calibration_log[f"calibration_{prefix}_{mode}_brier_score"] = metrics['brier_score']

                 if self.use_wandb and calibration_log:
                    wandb.log(calibration_log)

                 self._export_calibration_json(calibration_log)
                 print(f"🔧 Calibration enabled for test predictions")
                 return # Exit function, handled multi-label

            elif self.target_mode == "masked_multi_task":
                 if self.calibrators is None:
                     self.calibrators = {}
                 
                 tasks = [("momentum", "mom"), ("liquidity", "liq")]
                 all_results = {}
                 
                 for task_name, task_prefix in tasks:
                      print(f"\n   ⚙️ Calibrating Task: {task_name.upper()} ({task_prefix})")
                      try:
                          results, calibrator = analyze_model_calibration_from_predictions(
                               evaluator=self.evaluator,
                               graph_data=self.data,
                               modes=['val', 'test'],
                               optimize_threshold=optimize_threshold,
                               threshold_metric=threshold_metric,
                               target_mode="masked_multi_task",
                               task_prefix=task_prefix,
                               calibration_method=calibration_method,
                          )
                          self.calibrators[task_name] = calibrator
                          all_results[task_name] = results
                          print(f"   ✅ Calibrated {task_name}")
                      except Exception as e:
                          print(f"   ⚠️ Failed to calibrate {task_name}: {e}")
                          import traceback
                          traceback.print_exc()

                 # Update evaluator with new calibrators
                 # For masked_multi_task, evaluator needs to know which calibrator is for which task
                 # We pass the dictionary
                 if hasattr(self.evaluator, "set_calibrator"):
                     self.evaluator.set_calibrator(self.calibrators)
                     
                 # Log summary
                 print(f"\n📊 CALIBRATION SUMMARY (Masked Multi-Task):")
                 calibration_log = {}
                 for task_name, res in all_results.items():
                     print(f"   --- {task_name.upper()} ---")
                     for mode, data in res.items():
                        if 'metrics' in data:
                            metrics = data['metrics']
                            print(f"   {mode.upper()}:")
                            print(f"      ECE: {metrics['ece']:.4f}")
                            print(f"      MCE: {metrics['mce']:.4f}")
                            print(f"      Brier Score: {metrics['brier_score']:.4f}")
                            if 'optimal_threshold' in data:
                                print(f"      Optimal Threshold: {data['optimal_threshold']:.4f}")
                            
                            prefix = task_name 
                            calibration_log[f"calibration_{prefix}_{mode}_ece"] = metrics['ece']
                            calibration_log[f"calibration_{prefix}_{mode}_mce"] = metrics['mce']
                            calibration_log[f"calibration_{prefix}_{mode}_brier_score"] = metrics['brier_score']

                 if self.use_wandb and calibration_log:
                    wandb.log(calibration_log)

                 self._export_calibration_json(calibration_log)
                 print(f"🔧 Calibration enabled for test predictions")
                 return

            else:
                results, calibrator = analyze_model_calibration_from_predictions(
                    evaluator=self.evaluator,
                    graph_data=self.data,
                    modes=['val', 'test'],
                    optimize_threshold=optimize_threshold,
                    threshold_metric=threshold_metric,
                    target_mode=self.target_mode,
                    calibration_method=calibration_method,
                )
                self.calibrators = calibrator # Single calibrator (None if method="none")
                if calibrator and hasattr(self.evaluator, "set_calibrator"):
                     self.evaluator.set_calibrator(self.calibrators)
            
                # Store calibrator for later use (e.g. visualization)
                self.calibrator = calibrator
                
                # Print summary (Single Task)
                print(f"\n📊 CALIBRATION SUMMARY:")
                calibration_log = {}
                for mode, data in results.items():
                    if 'metrics' in data:
                        metrics = data['metrics']
                        print(f"   {mode.upper()}:")
                        print(f"      ECE: {metrics['ece']:.4f}")
                        print(f"      MCE: {metrics['mce']:.4f}")
                        print(f"      Brier Score: {metrics['brier_score']:.4f}")
                        
                        # Prepare for wandb logging
                        calibration_log[f"calibration_{mode}_ece"] = metrics['ece']
                        calibration_log[f"calibration_{mode}_mce"] = metrics['mce']
                        calibration_log[f"calibration_{mode}_brier_score"] = metrics['brier_score']
                
                if self.use_wandb and calibration_log:
                    wandb.log(calibration_log)

                self._export_calibration_json(calibration_log)

                # Apply calibration to future predictions if enabled
                if self.config.get("calibration", {}).get("apply_to_test", False) and calibrator:
                    self.evaluator.set_calibrator(calibrator)
                    
                    # Set optimal threshold if available
                    if 'val_calibrated' in results and 'optimal_threshold' in results['val_calibrated']:
                        optimal_threshold = results['val_calibrated']['optimal_threshold']
                        self.evaluator.set_optimal_threshold(optimal_threshold)
                        
                        # Also set percentile threshold for robustness
                        if hasattr(calibrator, 'optimal_threshold_percentile'):
                            self.evaluator.set_optimal_threshold_percentile(calibrator.optimal_threshold_percentile)
                        
                        print(f"🎯 Using optimal threshold: {optimal_threshold:.4f} (instead of 0.5)")
                    
                    print(f"🔧 Calibration enabled for test predictions")
                
        except Exception as e:
            print(f"❌ Calibration analysis failed: {str(e)}")
            print("   Continuing without calibration...")
            import traceback
            traceback.print_exc()

    def _export_calibration_json(self, calibration_log: dict):
        """Save calibration metrics to a JSON file alongside the main metrics export."""
        if not calibration_log:
            return
        try:
            import json
            from datetime import datetime

            save_dir = self.config.get("output_dir", "outputs")
            results_dir = os.path.join(save_dir, "results")
            model_name = self.config.get("train", {}).get("model", "unknown")
            target_mode = self.config.get("data_processing", {}).get("target_mode", "unknown")
            seed = self.config.get("seed", 0)

            try:
                import wandb
                wandb_id = wandb.run.id if wandb.run else "local"
            except (ImportError, AttributeError):
                wandb_id = "local"

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            dir_path = os.path.join(results_dir, model_name, target_mode)
            os.makedirs(dir_path, exist_ok=True)
            file_path = os.path.join(dir_path, f"{timestamp}_{wandb_id}_calibration.json")

            payload = {
                "metadata": {"model": model_name, "target_mode": target_mode, "seed": seed, "wandb_run_id": wandb_id},
                "calibration_metrics": calibration_log,
            }
            with open(file_path, "w") as f:
                json.dump(payload, f, indent=2, default=str)
            print(f"📁 Saved calibration metrics to {file_path}")
        except Exception as e:
            print(f"⚠️ Failed to export calibration JSON: {e}")

    def _perform_visualization_analysis(self):
        """
        Perform decision boundary visualization.
        """
        print("\n" + "="*60)
        print("VISUALIZATION ANALYSIS")
        print("="*60)
        
        try:
            output_dir = self.config.get("output_dir", "outputs")
            os.makedirs(output_dir, exist_ok=True)
            
            # Visualize using PCA (save both PDF and PNG for wandb)
            for ext in ["pdf", "png"]:
                visualize_decision_boundary(
                    self.model,
                    self.data,
                    output_path=os.path.join(output_dir, f"decision_boundary_pca.{ext}"),
                    method='pca',
                    device=self.device
                )
            
            # Visualize calibrated decision boundary if available
            calibrator_to_use = getattr(self, "calibrators", None)
            if calibrator_to_use is None:
                calibrator_to_use = getattr(self, "calibrator", None)

            if calibrator_to_use:
                # 1. Calibrated probabilities, default threshold (0.5)
                for ext in ["pdf", "png"]:
                    visualize_decision_boundary(
                        self.model,
                        self.data,
                        output_path=os.path.join(output_dir, f"decision_boundary_pca_calibrated.{ext}"),
                        method='pca',
                        device=self.device,
                        calibrator=calibrator_to_use
                    )
                
                # 2. Calibrated probabilities, OPTIMAL threshold
                if hasattr(calibrator_to_use, 'optimal_threshold'):
                    optimal_threshold = calibrator_to_use.optimal_threshold
                    print(f"   Generating visualization with optimal threshold: {optimal_threshold:.4f}")
                    
                    for ext in ["pdf", "png"]:
                        visualize_decision_boundary(
                            self.model,
                            self.data,
                            output_path=os.path.join(output_dir, f"decision_boundary_pca_calibrated_threshold.{ext}"),
                            method='pca',
                            device=self.device,
                            calibrator=calibrator_to_use,
                            threshold=optimal_threshold
                        )
                    
            # Log images to wandb
            if self.use_wandb:
                images_to_log = {}
                
                for name in ["decision_boundary_pca", "decision_boundary_pca_calibrated", "decision_boundary_pca_calibrated_threshold"]:
                    pdf_path = os.path.join(output_dir, f"{name}.pdf")
                    png_path = os.path.join(output_dir, f"{name}.png")
                    if os.path.exists(png_path):
                        images_to_log[name] = wandb.Image(png_path)
                    elif os.path.exists(pdf_path):
                        # wandb.Image can't read PDFs; skip
                        pass
                    
                if images_to_log:
                    wandb.log(images_to_log)
            
        except Exception as e:
            print(f"⚠️ Visualization failed: {str(e)}")
            import traceback
            traceback.print_exc()

    def get_all_embeddings(self):
        """
        Run inference on all nodes to extract GNN representations (post-GNN, pre-head).
        """
        self.model.eval()
        
        # Ensure data is on correct device
        data = self.data.to(self.device)
        
        with torch.no_grad():
            # Full graph forward pass
            out = self.model(data.x_dict, data.edge_index_dict)
            
            emb = None
            if isinstance(out, dict):
                if "retrieval_embedding" in out and out["retrieval_embedding"] is not None:
                    emb = out["retrieval_embedding"]["startup"]
                elif "embedding" in out:
                    emb = out["embedding"]
                    if isinstance(emb, dict): emb = emb["startup"]
            
            if emb is None:
                raise ValueError(f"Could not find 'retrieval_embedding' or 'embedding' in model output keys: {out.keys() if isinstance(out, dict) else 'Not a dict'}")

            # Return on CPU to save memory/for saving
            return emb.cpu()
