"""Entry point for GNN startup success prediction: config loading, CLI parsing, and training orchestration."""
import sys
import os
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.ml.preprocessing import perform_preprocessing
from torch_geometric import seed_everything
import torch.backends.cudnn
from src.ml.visualize import visualize_graph
from src.ml.feature_visualization import visualize_graph_features, plot_nan_distribution, visualize_edge_statistics
from src.ml.train import Trainer
from src.ml.utils import load_config, deep_merge_dict, args_to_nested_dict
from src.ml.run_context import (
    capture_run_context,
    finalize_run_context,
    log_context_to_wandb,
    write_context_json,
)
from src.ml.graph_cache import build_or_load_graph
import wandb
import yaml
import argparse


def create_parser():
    """Create argument parser for all config options"""
    parser = argparse.ArgumentParser(description='GNN Startup Success Prediction')
    
    # Paths
    parser.add_argument('--paths.data_dir', type=str, default=None)
    parser.add_argument('--paths.crunchbase_dir', type=str, default=None)
    parser.add_argument('--paths.graph_dir', type=str, default=None)

    # Utility
    parser.add_argument('--preprocess-only', type=str, default=None, metavar='PATH',
                        help='Run preprocessing only, save graph to PATH, then exit (no training)')
    parser.add_argument('--output-dir', type=str, default=None, metavar='PATH',
                        help='Override pipeline_state output directory (default: outputs/pipeline_state)')
    
    # Visualize
    parser.add_argument('--visualize.enabled', type=lambda x: x.lower() == 'true', default=None)
    parser.add_argument('--visualize.output_file', type=str, default=None)
    parser.add_argument('--visualize.show_labels', type=lambda x: x.lower() == 'true', default=None)
    parser.add_argument('--visualize.enable_physics', type=lambda x: x.lower() == 'true', default=None)
    # Seed
    parser.add_argument('--seed', type=int, default=None)
    
    # WandB
    parser.add_argument('--wandb.enabled', type=lambda x: x.lower() == 'true', default=None)
    parser.add_argument('--wandb.project', type=str, default=None)
    parser.add_argument('--wandb.use_sweep', type=lambda x: x.lower() == 'true', default=None)
    parser.add_argument('--wandb.sweep_config_path', type=str, default=None)
    
    # Train
    parser.add_argument('--train.model', type=str, default=None)
    parser.add_argument('--train.device', type=str, default=None)
    parser.add_argument('--train.lr', type=float, default=None)
    parser.add_argument('--train.epochs', type=int, default=None)
    parser.add_argument('--train.aggregation_method', type=str, default=None)
    parser.add_argument('--train.hidden_channels', type=int, default=None)
    parser.add_argument('--train.out_channels', type=int, default=None)
    parser.add_argument('--train.num_layers', type=int, default=None)
    parser.add_argument('--train.normalize', type=lambda x: x.lower() == 'true', default=None)
    parser.add_argument('--train.activation', type=str, default=None)
    parser.add_argument('--train.jumping_knowledge', type=str, default=None)
    
    # Train loss
    parser.add_argument('--train.loss.use_class_weights', type=lambda x: x.lower() == 'true', default=None)
    parser.add_argument('--train.loss.binary_loss_weight', type=float, default=None)
    parser.add_argument('--train.loss.use_focal_loss', type=lambda x: x.lower() == 'true', default=None)
    parser.add_argument('--train.loss.label_smoothing', type=float, default=None)
    parser.add_argument('--train.loss.momentum_weight', type=float, default=None)
    parser.add_argument('--train.loss.liquidity_weight', type=float, default=None)
    parser.add_argument('--train.loss.retrieval_weight', type=float, default=None)
    parser.add_argument('--train.loss.retrieval_loss_type', type=str, default=None)
    parser.add_argument('--train.loss.contrastive_positive_source', type=str, default=None)
    parser.add_argument('--train.loss.normalize_retrieval_loss', type=lambda x: x.lower() == 'true', default=None)
    parser.add_argument('--train.gradient_clip_val', type=float, default=None)
    parser.add_argument('--train.weight_decay', type=float, default=None)
    parser.add_argument('--train.scheduler.type', type=str, default=None)
    parser.add_argument('--train.scheduler.T_max', type=int, default=None)
    parser.add_argument('--train.scheduler.eta_min', type=float, default=None)
    
    # Models (HAN)
    parser.add_argument('--models.HAN.activation_type', type=str, default=None)
    parser.add_argument('--models.HAN.dropout', type=float, default=None)
    parser.add_argument('--models.HAN.heads', type=int, default=None)
    parser.add_argument('--models.HAN.hidden_channels', type=int, default=None)
    parser.add_argument('--models.HAN.negative_slope', type=float, default=None)
    parser.add_argument('--models.HAN.num_layers', type=int, default=None)

    # Models (SimpleHGN)
    parser.add_argument('--models.SimpleHGN.hidden_channels', type=int, default=None)
    parser.add_argument('--models.SimpleHGN.num_layers', type=int, default=None)
    parser.add_argument('--models.SimpleHGN.heads', type=int, default=None)
    parser.add_argument('--models.SimpleHGN.edge_dim', type=int, default=None,
                        help="Edge-type embedding dim; omit or set null for hidden_channels")
    parser.add_argument('--models.SimpleHGN.dropout', type=float, default=None)
    parser.add_argument('--models.SimpleHGN.attn_dropout', type=float, default=None)
    parser.add_argument('--models.SimpleHGN.negative_slope', type=float, default=None)
    parser.add_argument('--models.SimpleHGN.residual', type=lambda x: x.lower() == 'true', default=None)
    parser.add_argument('--models.SimpleHGN.alpha', type=float, default=None,
                        help="Residual-attention blend weight beta in paper (default 0.05)")
    parser.add_argument('--models.SimpleHGN.l2_normalize', type=lambda x: x.lower() == 'true', default=None)
    parser.add_argument('--models.SimpleHGN.bias', type=lambda x: x.lower() == 'true', default=None)
    parser.add_argument('--models.SimpleHGN.activation_type', type=str, default=None)
    
    # Models (SeHGNN)
    parser.add_argument('--models.SeHGNN.heads', type=int, default=None)
    parser.add_argument('--models.SeHGNN.hidden_channels', type=int, default=None)
    parser.add_argument('--models.SeHGNN.dropout', type=float, default=None)
    parser.add_argument('--models.SeHGNN.input_drop', type=float, default=None)
    parser.add_argument('--models.SeHGNN.att_drop', type=float, default=None)
    parser.add_argument('--models.SeHGNN.activation_type', type=str, default=None)
    parser.add_argument('--models.SeHGNN.use_residual', type=lambda x: x.lower() == 'true', default=None)
    parser.add_argument('--models.SeHGNN.transformer_activation', type=str, default=None)
    parser.add_argument('--models.SeHGNN.use_self_loop', type=lambda x: x.lower() == 'true', default=None)
    parser.add_argument('--models.SeHGNN.attention_temperature', type=float, default=None)
    parser.add_argument('--models.SeHGNN.use_retrieval_head', type=lambda x: x.lower() == 'true', default=None)
    parser.add_argument('--models.SeHGNN.detach_retrieval_head', type=lambda x: x.lower() == 'true', default=None)
    parser.add_argument('--models.SeHGNN.num_hops', type=int, default=None)
    parser.add_argument('--models.SeHGNN.num_layers', type=int, default=None)
    parser.add_argument('--models.SeHGNN.gamma_init', type=float, default=None)
    parser.add_argument('--models.SeHGNN.gamma_learnable', type=lambda x: x.lower() == 'true', default=None)
    parser.add_argument('--models.SeHGNN.use_discrepancy', type=lambda x: x.lower() == 'true', default=None)

    # Models (VenGNN)
    parser.add_argument('--models.VenGNN.hidden_channels', type=int, default=None)
    parser.add_argument('--models.VenGNN.num_layers', type=int, default=None)
    parser.add_argument('--models.VenGNN.heads', type=int, default=None)
    parser.add_argument('--models.VenGNN.dropout', type=float, default=None)
    parser.add_argument('--models.VenGNN.rw_num_walks', type=int, default=None)
    parser.add_argument('--models.VenGNN.rw_walk_length', type=int, default=None)
    parser.add_argument('--models.VenGNN.activation_type', type=str, default=None)
    parser.add_argument('--models.VenGNN.max_metapaths', type=int, default=None)
    parser.add_argument('--models.VenGNN.use_paper_attention', type=lambda x: x.lower() == 'true', default=None)
    parser.add_argument('--models.VenGNN.branch_mode', type=str, default=None)
    parser.add_argument('--models.VenGNN.paper_widening', type=lambda x: x.lower() == 'true', default=None)
    parser.add_argument('--models.VenGNN.gat_edge_mode', type=str, default=None)
    parser.add_argument('--models.VenGNN.rw_weighted', type=lambda x: x.lower() == 'true', default=None)

    # Models (Hetero2Net)
    parser.add_argument('--models.Hetero2Net.hidden_channels', type=int, default=None)
    parser.add_argument('--models.Hetero2Net.num_layers', type=int, default=None)
    parser.add_argument('--models.Hetero2Net.dropout', type=float, default=None)
    parser.add_argument('--models.Hetero2Net.activation_type', type=str, default=None)
    parser.add_argument('--models.Hetero2Net.alpha', type=float, default=None,
                        help="Weight on L_corr (Pearson correlation between homo/hetero channels, paper Eq. 8)")
    parser.add_argument('--models.Hetero2Net.beta', type=float, default=None,
                        help="Weight on L_rec (masked-metapath link prediction, paper Eq. 9)")
    parser.add_argument('--models.Hetero2Net.mask_ratio', type=float, default=None,
                        help="Paper 'r': fraction of metapath edges masked per training step")
    parser.add_argument('--models.Hetero2Net.label_mask_ratio', type=float, default=None,
                        help="Paper 'p': fraction of labels KEPT in label embedding (1-p are masked)")
    parser.add_argument('--models.Hetero2Net.use_label_propagation', type=lambda x: x.lower() == 'true', default=None)
    parser.add_argument('--models.Hetero2Net.num_label_classes', type=int, default=None)
    parser.add_argument('--models.Hetero2Net.root_weight', type=lambda x: x.lower() == 'true', default=None)
    parser.add_argument('--models.Hetero2Net.normalize_bn', type=lambda x: x.lower() == 'true', default=None)
    parser.add_argument('--models.Hetero2Net.corr_on_metapaths_only', type=lambda x: x.lower() == 'true', default=None,
                        help="True: L_corr only on startup-startup metapaths (memory-efficient); False: paper-faithful on all edges")
    parser.add_argument('--models.Hetero2Net.aux_reduction', type=str, default=None, choices=[None, 'sum', 'mean'],
                        help="'sum' (paper-faithful, matches reference impl) or 'mean' (graph-size-invariant)")
    parser.add_argument('--models.Hetero2Net.aux_loss_weight', type=float, default=None,
                        help="On/off scalar for the α·corr + β·link aux term (default 1.0)")

    # Models (GCN)
    parser.add_argument('--models.GCN.hidden_channels', type=int, default=None)
    parser.add_argument('--models.GCN.dropout', type=float, default=None)
    parser.add_argument('--models.GCN.normalize', type=lambda x: x.lower() == 'true', default=None)
    parser.add_argument('--models.GCN.add_self_loops', type=lambda x: x.lower() == 'true', default=None)
    parser.add_argument('--models.GCN.activation_type', type=str, default=None)

    # Models (RGCN)
    parser.add_argument('--models.RGCN.hidden_channels', type=int, default=None)
    parser.add_argument('--models.RGCN.num_layers', type=int, default=None)
    parser.add_argument('--models.RGCN.dropout', type=float, default=None)
    parser.add_argument('--models.RGCN.num_bases', type=int, default=None)
    parser.add_argument('--models.RGCN.normalize', type=lambda x: x.lower() == 'true', default=None)
    parser.add_argument('--models.RGCN.activation_type', type=str, default=None)

    # Models (MLP)
    parser.add_argument('--models.MLP.hidden_channels', type=int, default=None)
    parser.add_argument('--models.MLP.dropout', type=float, default=None)
    parser.add_argument('--models.MLP.activation_type', type=str, default=None)

    # Models (SageGNN / GraphSAGE)
    parser.add_argument('--models.SageGNN.hidden_channels', type=int, default=None)
    parser.add_argument('--models.SageGNN.num_layers', type=int, default=None)
    parser.add_argument('--models.SageGNN.dropout', type=float, default=None)
    parser.add_argument('--models.SageGNN.normalize', type=lambda x: x.lower() == 'true', default=None)
    parser.add_argument('--models.SageGNN.activation_type', type=str, default=None)
    parser.add_argument('--models.SageGNN.aggr', type=str, default=None,
                        choices=[None, 'mean', 'max', 'lstm', 'sum'])

    # Models (XGBoost)
    parser.add_argument('--models.XGBoost.n_estimators', type=int, default=None)
    parser.add_argument('--models.XGBoost.max_depth', type=int, default=None)
    parser.add_argument('--models.XGBoost.learning_rate', type=float, default=None)
    parser.add_argument('--models.XGBoost.subsample', type=float, default=None)
    parser.add_argument('--models.XGBoost.colsample_bytree', type=float, default=None)
    parser.add_argument('--models.XGBoost.min_child_weight', type=float, default=None)
    parser.add_argument('--models.XGBoost.gamma', type=float, default=None)
    parser.add_argument('--models.XGBoost.reg_alpha', type=float, default=None)
    parser.add_argument('--models.XGBoost.reg_lambda', type=float, default=None)
    parser.add_argument('--models.XGBoost.scale_pos_weight', type=float, default=None)
    parser.add_argument('--models.XGBoost.objective', type=str, default=None)
    parser.add_argument('--models.XGBoost.tree_method', type=str, default=None)

    # Analysis
    parser.add_argument('--analysis.enable_downstream_analysis', type=lambda x: x.lower() == 'true', default=None)
    parser.add_argument('--analysis.enable_homophily_analysis', type=lambda x: x.lower() == 'true', default=None)
    parser.add_argument('--analysis.enable_visualization_analysis', type=lambda x: x.lower() == 'true', default=None)

    # Optimizer / Scheduler sweep axes. `train.scheduler.type` and
    # `train.scheduler.eta_min` are already registered higher up; only add
    # the new flags here to avoid argparse conflicts.
    parser.add_argument('--train.optimizer', type=str, default=None,
                        choices=[None, 'Adam', 'AdamW'])
    parser.add_argument('--train.scheduler.warmup_epochs', type=int, default=None)

    # Models (LLM)
    parser.add_argument('--models.LLM.model_name', type=str, default=None)
    parser.add_argument('--models.LLM.temperature', type=float, default=None)
    parser.add_argument('--models.LLM.prompt_features', type=str, default=None)
    parser.add_argument('--models.LLM.use_calibration', type=lambda x: x.lower() == 'true', default=None)
    parser.add_argument('--models.LLM.use_chain_of_thought', type=lambda x: x.lower() == 'true', default=None)
    parser.add_argument('--models.LLM.max_predictions', type=int, default=None)

    # Features
    parser.add_argument('--features.description_embedding_dim', type=int, default=None)
    parser.add_argument('--features.use_degree_centrality', type=lambda x: x.lower() == 'true', default=None)
    parser.add_argument('--features.use_edge_counts', type=lambda x: x.lower() == 'true', default=None)
    parser.add_argument('--features.use_org_description', type=lambda x: x.lower() == 'true', default=None)
    parser.add_argument('--features.use_pagerank_centrality', type=lambda x: x.lower() == 'true', default=None)
    parser.add_argument('--features.use_people_description', type=lambda x: x.lower() == 'true', default=None)
    
    # Eval
    parser.add_argument('--eval.test_best_model', type=lambda x: x.lower() == 'true', default=None)
    parser.add_argument('--eval.optimization_metric_type', type=str, default=None)
    parser.add_argument('--eval.min_amount_of_epochs', type=int, default=None)
    parser.add_argument('--eval.early_stopping.enabled', type=lambda x: x.lower() == 'true', default=None)
    parser.add_argument('--eval.early_stopping.patience', type=int, default=None)
    parser.add_argument('--eval.export_metrics_json', type=lambda x: x.lower() == 'true', default=None)
    parser.add_argument('--eval.export_predictions', type=lambda x: x.lower() == 'true', default=None)
    
    # Explain
    parser.add_argument('--explain.enabled', type=lambda x: x.lower() == 'true', default=None)
    parser.add_argument('--explain.path', type=str, default=None)
    parser.add_argument('--explain.sample_size', type=int, default=None)
    
    # Data processing
    parser.add_argument('--data_processing.row_nan_threshold', type=float, default=None)
    parser.add_argument('--data_processing.target_mode', type=str, default=None)
    parser.add_argument('--data_processing.multi_column', type=str, default=None)
    parser.add_argument('--data_processing.binary_column', type=str, default=None)
    parser.add_argument('--data_processing.add_metapaths', type=lambda x: x.lower() == 'true', default=None)
    parser.add_argument('--data_processing.nan_filtering.enabled', type=lambda x: x.lower() == 'true', default=None)

    # Data processing train
    parser.add_argument('--data_processing.train.use_batches', type=lambda x: x.lower() == 'true', default=None)
    parser.add_argument('--data_processing.train.neighbor_loader_nodes', type=int, default=None)
    parser.add_argument('--data_processing.train.neighbor_loader_iterations', type=int, default=None)
    parser.add_argument('--data_processing.train.neighbor_loader_batch_size', type=int, default=None)
    
    # Data processing val
    parser.add_argument('--data_processing.val.use_batches', type=lambda x: x.lower() == 'true', default=None)
    parser.add_argument('--data_processing.val.neighbor_loader_nodes', type=int, default=None)
    parser.add_argument('--data_processing.val.neighbor_loader_iterations', type=int, default=None)
    parser.add_argument('--data_processing.val.neighbor_loader_batch_size', type=int, default=None)
    
    # Data processing test
    parser.add_argument('--data_processing.test.use_batches', type=lambda x: x.lower() == 'true', default=None)
    parser.add_argument('--data_processing.test.neighbor_loader_nodes', type=int, default=None)
    parser.add_argument('--data_processing.test.neighbor_loader_iterations', type=int, default=None)
    parser.add_argument('--data_processing.test.neighbor_loader_batch_size', type=int, default=None)
    
    # Imputation
    parser.add_argument('--imputation.enabled', type=lambda x: x.lower() == 'true', default=None)
    parser.add_argument('--imputation.node_types.startup.numerical_method', type=str, default=None)
    parser.add_argument('--imputation.node_types.startup.categorical_method', type=str, default=None)
    
    # Metapath Discovery & Ablation
    parser.add_argument('--metapath_discovery.automatic.enabled', type=lambda x: x.lower() == 'true', default=None)
    parser.add_argument('--metapath_discovery.automatic.max_metapaths', type=int, default=None)
    parser.add_argument('--metapath_discovery.automatic.selection_strategy', type=str, default=None)
    parser.add_argument('--metapath_discovery.automatic.prune_top_k', type=int, default=None)
    parser.add_argument('--metapath_discovery.automatic.ablation.drop_edges', nargs='+', default=None)
    parser.add_argument('--data_processing.ablation.drop_node_types', nargs='+', default=None)
    parser.add_argument('--data_processing.ablation.drop_feature_groups', nargs='+', default=None)
    parser.add_argument('--data_processing.ablation.feature_information_level', type=str, default=None)
    parser.add_argument('--data_processing.graph_variant', type=str, default=None,
                        choices=[None, 'g1_full', 'g2_no_sector', 'g3_pruned',
                                 'g4_heterophily', 'g5_base'],
                        help='Post-build graph filter for the G1-G5 curation ablation')
    parser.add_argument('--experiment_id', type=str, default=None)
    parser.add_argument('--wandb.tags', nargs='+', default=None)
    parser.add_argument('--metapath_discovery.mode', type=str, default=None)
    parser.add_argument('--metapath_discovery.manual.whitelist', nargs='+', default=None)

    # Edge loading flags (for ablation series B)
    parser.add_argument('--data_processing.edge_loading.enable_only', type=str, default=None)
    parser.add_argument('--data_processing.edge_loading.founder_investor_employment', type=lambda x: x.lower() == 'true', default=None)
    parser.add_argument('--data_processing.edge_loading.founder_coworking', type=lambda x: x.lower() == 'true', default=None)
    parser.add_argument('--data_processing.edge_loading.founder_investor_identity', type=lambda x: x.lower() == 'true', default=None)
    parser.add_argument('--data_processing.edge_loading.founder_co_study', type=lambda x: x.lower() == 'true', default=None)
    parser.add_argument('--data_processing.edge_loading.founder_role_edges', type=lambda x: x.lower() == 'true', default=None)

    # Maturity-mask toggle (Exit-task ablation)
    parser.add_argument('--data_processing.strict_gating.enabled', type=lambda x: x.lower() == 'true', default=None)

    # Graph feature flags (for ablation series F)
    parser.add_argument('--data_processing.use_louvain_clusters', type=lambda x: x.lower() == 'true', default=None)
    parser.add_argument('--data_processing.use_degree_centrality', type=lambda x: x.lower() == 'true', default=None)
    parser.add_argument('--data_processing.use_pagerank_centrality', type=lambda x: x.lower() == 'true', default=None)
    parser.add_argument('--data_processing.use_edge_counts', type=lambda x: x.lower() == 'true', default=None)
    parser.add_argument('--data_processing.use_centrality_features', type=lambda x: x.lower() == 'true', default=None)
    parser.add_argument('--data_processing.use_smart_money_features', type=lambda x: x.lower() == 'true', default=None)
    parser.add_argument('--data_processing.use_neighbor_label_features', type=lambda x: x.lower() == 'true', default=None)

    # Louvain clustering hyperparameters (for Louvain sweep)
    parser.add_argument('--data_processing.louvain_resolution', type=float, default=None)
    parser.add_argument('--data_processing.louvain_projection_types', nargs='+', default=None)

    # Graph feature ablation index (for graph feature sweep)
    parser.add_argument('--data_processing.graph_feature_ablation_index', type=int, default=None)

    # Description embedding dimension
    parser.add_argument('--data_processing.description_embedding_dim', type=int, default=None)
    parser.add_argument('--data_processing.use_org_description', type=lambda x: x.lower() == 'true', default=None)

    # Calibration (for calibration sweep)
    parser.add_argument('--calibration.enabled', type=lambda x: x.lower() == 'true', default=None)
    parser.add_argument('--calibration.method', type=str, default=None)
    parser.add_argument('--calibration.optimize_threshold', type=lambda x: x.lower() == 'true', default=None)
    parser.add_argument('--calibration.threshold_metric', type=str, default=None)

    return parser


def _safe_run(config):
    """Wrapper around run() that captures errors to local JSON for debugging."""
    status = "ok"
    err: Exception | None = None
    try:
        run(config)
    except Exception as e:
        err = e
        status = "error"
        print(f"\n{'='*50}")
        print(f"RUN FAILED: {type(e).__name__}: {e}")
        print(f"{'='*50}")
        try:
            from src.ml.metrics_export import save_error_report
            output_dir = config.get("output_dir", "outputs")
            results_dir = os.path.join(output_dir, "results")
            save_error_report(error=e, config=config, output_base_dir=results_dir)
        except Exception as save_err:
            print(f"Warning: could not save error report: {save_err}")
    finally:
        # Finalize reproducibility context (wall-clock, peak GPU memory) whether run
        # succeeded or failed. Persists as JSON alongside the usual outputs and pushes
        # to wandb.config if a run is active.
        start_ctx = config.get("_run_context")
        if start_ctx is not None:
            runtime = finalize_run_context(start_ctx)
            full_ctx = {**start_ctx, "runtime": runtime, "status": status}
            if err is not None:
                full_ctx["error"] = {"type": type(err).__name__, "message": str(err)[:2000]}
            try:
                output_dir = config.get("output_dir", "outputs/pipeline_state")
                results_dir = os.path.join(output_dir, "results")
                # Include PID and wandb run id in filename so concurrent runs
                # don't clobber each other. Also write the latest as
                # run_context.json for easy inspection.
                run_id_part = wandb.run.id if wandb.run is not None else "local"
                suffixed = f"run_context_{os.getpid()}_{run_id_part}.json"
                written = write_context_json(full_ctx, results_dir, filename=suffixed)
                # Also write a stable alias (may be overwritten by later runs — by design)
                write_context_json(full_ctx, results_dir, filename="run_context.json")
                print(f"[run_context] Wrote {written}")
            except Exception as save_err:
                print(f"[run_context] Warning: could not write run_context.json: {save_err}")
            if wandb.run is not None:
                try:
                    log_context_to_wandb({"runtime": runtime, "status": status}, wandb.run)
                    wandb.run.summary["repro/status"] = status
                    wandb.run.summary["repro/wall_clock_sec"] = runtime.get("wall_clock_sec")
                    if "peak_gpu_mem_mb" in runtime:
                        wandb.run.summary["repro/peak_gpu_mem_mb"] = runtime["peak_gpu_mem_mb"]
                except Exception as wb_err:
                    print(f"[run_context] Warning: could not push runtime to wandb: {wb_err}")
        # Close wandb AFTER repro context is flushed. Owned by _safe_run now,
        # not run(), so summary fields from the finally block reach the UI.
        if wandb.run is not None:
            try:
                wandb.finish()
            except Exception as wb_err:
                print(f"[run_context] Warning: wandb.finish() failed: {wb_err}")
    if err is not None:
        raise err  # Re-raise so wandb/caller still sees the failure


def run(config):
    seed = config["seed"]
    seed_everything(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Capture per-run reproducibility metadata (Pineau checklist).
    # Runs before any heavy work so it's available even if preprocessing crashes.
    _run_ctx = capture_run_context(config)
    config["_run_context"] = _run_ctx
    if wandb.run is not None:
        log_context_to_wandb(_run_ctx, wandb.run)
    print(
        f"[run_context] git={_run_ctx['git']['commit_short']}"
        f"{' (DIRTY)' if _run_ctx['git']['dirty'] else ''} "
        f"seed={_run_ctx['seed']} "
        f"graph_version={_run_ctx['graph_version']} "
        f"gpu={_run_ctx['gpu'].get('gpu_name') or 'cpu'}"
    )
    if _run_ctx["git"]["dirty"]:
        print(
            f"[run_context] WARNING: git is dirty ({_run_ctx['git']['dirty_file_count']} files). "
            "Results from dirty runs are excluded from the paper leaderboard."
        )

    def _preprocess():
        return perform_preprocessing(
            startups_filename="startup_nodes.csv",
            investors_filename="investor_nodes.csv",
            founders_filename="founder_nodes.csv",
            cities_filename="city_nodes.csv",
            university_filename="university_nodes.csv",
            sectors_filename="sector_nodes.csv",
            startup_investor_filename="startup_investor_edges.csv",
            startup_city_filename="startup_city_edges.csv",
            startup_founder_filename="startup_founder_edges.csv",
            startup_sector_filename="startup_sector_edges.csv",
            founder_university_filename="founder_university_edges.csv",
            investor_city_filename="investor_city_edges.csv",
            investor_sector_filename="investor_sector_edges.csv",
            university_city_filename="university_city_edges.csv",
            founder_investor_employment_filename="founder_investor_employment_edges.csv",
            founder_coworking_filename="founder_coworking_edges.csv",
            founder_investor_identity_filename="founder_investor_identity_edges.csv",
            founder_co_study_filename="founder_co_study_edges.csv",
            founder_board_filename="founder_board_edges.csv",
            founder_startup_director_filename="founder_startup_director_edges.csv",
            founder_investor_director_filename="founder_investor_director_edges.csv",
            config=config,
        )

    if config.get("graph_cache", {}).get("enabled", True):
        graph_data, node_names = build_or_load_graph(config, _preprocess)
    else:
        graph_data, node_names = _preprocess()

    if config["visualize"].get("feature_visualization", False):
        plot_nan_distribution(graph_data, node_type="startup")

    if config["visualize"]["enabled"] and not config["wandb"]["use_sweep"]:
        print("\n" + "=" * 50)
        print("VISUALIZATION")
        print("=" * 50)
        
        visualize_graph(
            graph_data=graph_data,
            output_file=config["visualize"]["output_file"],
            visible_node_types=set(config["visualize"]["visible_node_types"]),
            show_labels=config["visualize"]["show_labels"],
            enable_physics=config["visualize"]["enable_physics"],
            max_nodes=config["visualize"].get("max_nodes", 1000),
            sample_method=config["visualize"].get("sample_method", "degree_based"),
            show_features=config["visualize"].get("show_features", False),
            max_features=config["visualize"].get("max_features", 10),
            use_masks=config["visualize"].get("use_masks", False),
            included_masks=config["visualize"].get("included_masks", ["train", "val", "test"]),
        )

    if not config["wandb"]["use_sweep"]: 
        if config["visualize"].get("feature_visualization", False):
            visualize_graph_features(graph_data)
        
        if config["visualize"].get("edge_visualization", False):
            visualize_edge_statistics(graph_data)

    # Handle --preprocess-only: save to custom path and return early
    preprocess_only_path = config.get("_preprocess_only")
    if preprocess_only_path:
        os.makedirs(os.path.dirname(preprocess_only_path) or ".", exist_ok=True)
        print(f"Saving preprocessed graph data to {preprocess_only_path}...")
        torch.save(graph_data, preprocess_only_path)
        print(f"Done. Graph saved ({len(graph_data.node_types)} node types, {len(graph_data.edge_types)} edge types)")
        return

    # Configure output directory for per-run artifacts (metrics JSON, error
    # reports, checkpoints). Made per-run-unique to prevent concurrent sweep
    # agents from stepping on each other.
    _slurm_id = os.environ.get("SLURM_JOB_ID")
    _run_tag = f"{_slurm_id}_{os.getpid()}" if _slurm_id else f"pid_{os.getpid()}"
    output_dir = config.get("_output_dir") or f"outputs/pipeline_state/{_run_tag}"
    os.makedirs(output_dir, exist_ok=True)
    config["output_dir"] = output_dir

    # NOTE: the legacy `outputs/pipeline_state/graph_data.pt` save was removed
    # 2026-04-23. It was a 900 MB full-graph dump written on every run — wasteful
    # I/O and the root cause of the sweep-time NFS race that killed ~670 trials
    # (73% of all failures that day). The graph is already persisted by the
    # content-addressable cache under `outputs/graphs/graph_<hash>.pt` (see
    # src/ml/graph_cache.py). Post-hoc analysis scripts that used to read the
    # legacy path should switch to loading via `build_or_load_graph(config)`.
    # Gated behind a config flag so the legacy path can be re-enabled for
    # one-off runs that feed analyze_embeddings.py / benchmark_quality.py /
    # analyze_graph_features.py (defaults: off).
    if config.get("_persist_legacy_graph_data", False):
        graph_path = os.path.join(output_dir, "graph_data.pt")
        print(f"Saving preprocessed graph data to {graph_path}...")
        torch.save(graph_data, graph_path)

    # ==================================================
    # TRAINING GNN
    # ==================================================
    trainer = Trainer(graph_data, config)
    
    if config["train"].get("use_gnn", True):
        print("\n==================================================")
        print("TRAINING GNN")
        print("==================================================")
        trainer.train()
        
        # Save GNN embeddings if XGBoost needs them (Integrated Flow)
        if config.get("xgboost", {}).get("enabled", False) and config.get("xgboost", {}).get("use_gnn_embeddings", False):
            embedding_path = config.get("xgboost", {}).get("embedding_path", "outputs/gnn_embeddings.pt")
            
            embeddings = trainer.get_all_embeddings()
            os.makedirs(os.path.dirname(embedding_path), exist_ok=True)
            torch.save(embeddings, embedding_path)
            print(f"✅ Saved GNN embeddings to {embedding_path} (Shape: {embeddings.shape})")
            
            # Apply to graph_data for immediate XGBoost usage
            graph_data["startup"].x = embeddings
            
            # ALSO UPDATE SPLIT MASKS TO AVOID DIMENSION MISMATCH (Adapter uses these if present)
            if hasattr(graph_data["startup"], "x_val_mask"):
                 graph_data["startup"].x_val_mask = embeddings.clone()
            if hasattr(graph_data["startup"], "x_test_mask"):
                 graph_data["startup"].x_test_mask = embeddings.clone()
            if hasattr(graph_data["startup"], "x_test_mask_original"):
                 graph_data["startup"].x_test_mask_original = embeddings.clone()

            print("✅ Replaced startup features with GNN embeddings for XGBoost training.")


    # ==================================================
    # TRAINING XGBOOST
    # ==================================================
    if config.get("xgboost", {}).get("enabled", False):
        print("\n==================================================")
        print("TRAINING XGBOOST")
        print("==================================================")
        
        # If GNN training was skipped but we want to use embeddings (Decoupled Flow)
        if not config["train"].get("use_gnn", True) and config.get("xgboost", {}).get("use_gnn_embeddings", False):
            embedding_path = config.get("xgboost", {}).get("embedding_path", "outputs/gnn_embeddings.pt")
            if os.path.exists(embedding_path):
                embeddings = torch.load(embedding_path)
                graph_data["startup"].x = embeddings
                print(f"✅ Loaded GNN embeddings from {embedding_path} (Shape: {embeddings.shape})")
                print("✅ Replaced startup features with GNN embeddings for XGBoost training.")
                
                # ALSO UPDATE SPLIT MASKS TO AVOID DIMENSION MISMATCH (Adapter uses these if present)
                if hasattr(graph_data["startup"], "x_val_mask"):
                     graph_data["startup"].x_val_mask = embeddings.clone()
                if hasattr(graph_data["startup"], "x_test_mask"):
                     graph_data["startup"].x_test_mask = embeddings.clone()
                if hasattr(graph_data["startup"], "x_test_mask_original"):
                     graph_data["startup"].x_test_mask_original = embeddings.clone()

            else:
                print(f"⚠️ Warning: Embedding file {embedding_path} not found. Using default features.")

        # Prepare XGBoost Config
        import copy
        xgb_config = copy.deepcopy(config)
        xgb_config["train"]["model"] = "XGBoost"
        
        if "models" not in xgb_config:
            xgb_config["models"] = {}

        # Map params from 'xgboost' section to 'models.XGBoost'
        xgb_params = config.get("xgboost", {}).get("params", {})
        xgb_config["models"]["XGBoost"] = xgb_params
        # It inherits base config props like target_mode from data_processing.

        print(f"Initializing Trainer with model='XGBoost' and params: {xgb_params}")
        xgb_trainer = Trainer(graph_data, xgb_config)
        xgb_trainer.train()

    if config.get("analysis", {}).get("enable_homophily_analysis", False):
        print("\n" + "=" * 50)
        print("HOMOPHILY ANALYSIS")
        print("=" * 50)
        from src.ml.heterophily_metrics import calculate_edge_homophily, calculate_class_homophily
        
        if 'startup' in graph_data.node_types and hasattr(graph_data['startup'], 'y'):
            y = graph_data['startup'].y
            
            # Identify targets to analyze
            targets_map = {}
            if y is not None:
                if y.ndim > 1 and y.shape[1] >= 2:
                    # Assumes masked_multi_task or similar structure where 0=Mom, 1=Liq
                    targets_map["Momentum"] = y[:, 0]
                    targets_map["Liquidity"] = y[:, 1]
                else:
                    targets_map["Target"] = y if y.ndim == 1 else y[:, 0]

            if targets_map:
                for edge_type in graph_data.edge_types:
                    src_type, rel, dst_type = edge_type
                    if src_type == 'startup' and dst_type == 'startup':
                        edge_index = graph_data[edge_type].edge_index
                        
                        for target_name, target_y in targets_map.items():
                            mshr = calculate_edge_homophily(edge_index, target_y)
                            class_hom = calculate_class_homophily(edge_index, target_y)
                            mshr_s = f"{mshr:.4f}" if mshr is not None else "n/a"
                            ch_s = f"{class_hom:.4f}" if class_hom is not None else "n/a"
                            print(f"{rel} ({target_name}): MSHR={mshr_s}, ClassHom={ch_s}")
            else:
                 print("Skipping homophily: No valid targets found.")
        else:
            print("Skipping homophily analysis: Startup nodes or labels not found.")



    print("\n" + "=" * 50)
    print("TESTING")

    print("=" * 50)

    trainer.evaluate_test()

    # Save persistent per-run checkpoint (keyed by wandb run ID)
    # This prevents sweep agents from overwriting each other's checkpoints.
    if config["wandb"]["enabled"] and wandb.run is not None:
        run_id = wandb.run.id
        run_name = wandb.run.name or run_id
        project = wandb.run.project or "unknown"
        persistent_dir = os.path.join("outputs", "checkpoints", project, run_id)
        os.makedirs(persistent_dir, exist_ok=True)
        persistent_path = os.path.join(persistent_dir, "best_model.pt")
        trainer.save_checkpoint(persistent_path)
        print(f"📦 Saved persistent checkpoint: {persistent_path}")

    # Note: wandb.finish() is intentionally NOT called here. It is called in
    # _safe_run's finally block after the reproducibility context has been
    # finalized and pushed to wandb.summary. Calling it here would mean
    # repro/status, repro/wall_clock_sec, and repro/peak_gpu_mem_mb never
    # reach the W&B UI.


def main():
    # Load base config first
    base_config = load_config()
    
    # Check if we're being called by wandb agent or normal execution
    if len(sys.argv) > 1:
        # Parse command line arguments
        parser = create_parser()
        args = parser.parse_args()
        
        # Convert command line args to nested dict and merge with base config
        config_updates = args_to_nested_dict(args)
        config = deep_merge_dict(base_config, config_updates)

        # Handle --preprocess-only: save graph to custom path and exit (no training, no wandb)
        preprocess_only = getattr(args, 'preprocess_only', None)
        if preprocess_only:
            config["_preprocess_only"] = preprocess_only
            config["wandb"]["enabled"] = False
            run(config)
            sys.exit(0)

        # Handle --output-dir: override pipeline_state directory
        output_dir_override = getattr(args, 'output_dir', None)
        if output_dir_override:
            config["_output_dir"] = output_dir_override

        # Extract experiment_id if provided (for wandb tagging)
        experiment_id = getattr(args, 'experiment_id', None)

        # Initialize wandb for sweep run if we detect wandb-style args
        try:
            # Build tags list: combine config tags + experiment_id
            tags = config.get("wandb", {}).get("tags", []) or []
            if experiment_id:
                tags = list(tags) + [f"exp:{experiment_id}"]
                print(f"Added experiment_id to wandb tags: {experiment_id}")

            wandb.init(
                project=config.get("wandb", {}).get("project"),
                tags=tags if tags else None,
            )
            print("Initialized wandb for sweep run")
            
            # Merge sweep config into main config
            # wandb.config contains the parameters chosen by the sweep controller
            if wandb.run:
                sweep_config = dict(wandb.config)
                print(f"Sweep config: {sweep_config}")
                config = deep_merge_dict(config, sweep_config)
                
                # Update wandb.config with the FULL config so we see all static params too
                # BUT we must exclude keys that are part of the sweep, otherwise W&B warns they are "locked"
                static_config = {k: v for k, v in config.items() if k not in sweep_config}
                # Re-sending locked sweep keys triggers harmless "locked key" warnings; ignore them.
                wandb.config.update(config, allow_val_change=True)
                print("Updated wandb config with full configuration")
            
            # Force wandb enabled in config since we are in a sweep
            config["wandb"]["enabled"] = True
                
        except Exception as e:
            print(f"Running with command line args (no wandb): {e}")

        _safe_run(config)
    else:
        # Normal execution path without arguments
        config = base_config

        if not config["wandb"]["enabled"]:
            _safe_run(config)
        else:
            # Use tags from config
            tags = config["wandb"].get("tags")

            wandb.init(
                project=config["wandb"]["project"],
                name=config["wandb"].get("name"),
                tags=tags,
                notes=config["wandb"].get("notes"),
                config=config
            )
            
            if config["wandb"].get("use_sweep", False):
                # Load sweep config
                with open(config["wandb"]["sweep_config_path"], "r") as f:
                    sweep_config = yaml.safe_load(f)

                sweep_id = wandb.sweep(sweep_config, project=config["wandb"]["project"])

                def sweep_run():
                    base_config = load_config()
                    sweep_config = wandb.config.as_dict()
                    full_config = deep_merge_dict(base_config, sweep_config)
                    _safe_run(full_config)

                wandb.agent(sweep_id, function=sweep_run, count=None)
            else:
                _safe_run(config)


if __name__ == "__main__":
    main()