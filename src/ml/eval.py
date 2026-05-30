"""Evaluation metrics and model selection logic for training and test splits."""
import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_recall_curve,
    roc_curve,
    f1_score,
    recall_score,
    precision_score,
    accuracy_score,
    confusion_matrix,
    classification_report,
)
from .downstream_analysis import DownstreamAnalyzer
from pandas import DataFrame
import wandb
import matplotlib.pyplot as plt
from .utils import load_config
import os

_default_config = load_config()

class Evaluator:
    def __init__(
        self,
        model,
        device,
        use_wandb: bool,
        optimization_metric_type,
        min_amount_of_epochs: int,
        calibrator=None,
        config=None,
    ):
        self.model = model
        self.device = device
        self.use_wandb = use_wandb
        self.optimization_metric_type = optimization_metric_type
        self.min_amount_of_epochs = min_amount_of_epochs
        self.best_metric = -1
        self.best_epoch = -1
        self.calibrator = calibrator
        self.optimal_threshold = 0.5  # Default threshold
        self.best_model_dict = None
        self.current_epoch = -1
        self.config = config if config is not None else _default_config
        
        # Initialize Downstream Analyzer
        if self.config.get("analysis", {}).get("enable_downstream_analysis", True):
            self.analyzer = DownstreamAnalyzer(self.config)
        else:
            self.analyzer = None

    def update_best_model(self, metric_value):
        if (
            metric_value >= self.best_metric
            and self.current_epoch >= self.min_amount_of_epochs
        ):
            self.best_metric = metric_value
            self.best_model_dict = self.model.state_dict()

    def set_calibrator(self, calibrator):
        """Set the calibrator for this evaluator."""
        self.calibrator = calibrator
        print(f"✅ Calibrator set: {type(calibrator).__name__}")

    def set_optimal_threshold(self, threshold: float):
        """Set optimal threshold for binary classification."""
        self.optimal_threshold = threshold
        print(f"🎯 Optimal threshold set: {threshold:.4f}")
        
    def set_optimal_threshold_percentile(self, percentile: float):
        """Set optimal threshold percentile for robust thresholding."""
        self.optimal_threshold_percentile = percentile
        print(f"📊 Optimal threshold percentile set: {percentile:.1f}th percentile")
        
    def get_robust_threshold(self, y_prob: np.ndarray) -> float:
        """Get robust threshold, falling back to percentile-based if absolute fails."""
        # First try absolute threshold
        pos_predictions_abs = np.mean(y_prob > self.optimal_threshold)
        
        # If absolute threshold gives no positive predictions, use percentile
        if pos_predictions_abs == 0 and hasattr(self, 'optimal_threshold_percentile'):
            threshold = np.percentile(y_prob, self.optimal_threshold_percentile)
            print(f"   ⚠️ Absolute threshold {self.optimal_threshold:.4f} gives 0% positive predictions")
            print(f"   🔄 Falling back to percentile threshold: {threshold:.4f} ({self.optimal_threshold_percentile:.1f}th percentile)")
            return threshold
        else:
            return self.optimal_threshold

    def precision_at_k(self, y_true, probs, k):
        """
        Computes precision@k - fraction of correct predictions among top-k most confident samples.
        
        Parameters:
        - y_true: array of shape (n_samples,), true class labels.
        - probs: array of shape (n_samples,) for binary or (n_samples, n_classes) for multi-class
        - k: int, number of top predictions to evaluate.
        
        Returns:
        - Precision at k: float
        """
        y_true = np.asarray(y_true)
        probs = np.asarray(probs)
        
        if probs.ndim == 1:
            # Binary: use confidence of predicted class (not just positive class probability)
            predictions = (probs > 0.5).astype(int)
            # Confidence = probability of the predicted class
            confidences = np.where(predictions == 1, probs, 1 - probs)
        else:
            # Multi-class: use max probability and corresponding class
            confidences = np.max(probs, axis=1)
            predictions = np.argmax(probs, axis=1)
        
        # Get top-k most confident samples
        k = min(k, len(y_true))
        if k == 0:
            return 0.0
            
        top_k_indices = np.argpartition(confidences, -k)[-k:]
        
        # Count correct predictions among top-k
        correct = np.sum(y_true[top_k_indices] == predictions[top_k_indices])
        
        return correct / k
    
    def recall_at_k(self, y_true, probs, k):
        """
        Computes recall@k - fraction of positive samples found among top-k predictions.
        
        Parameters:
        - y_true: array of shape (n_samples,), true class labels.
        - probs: array of shape (n_samples,) for binary or (n_samples, n_classes) for multi-class
        - k: int, number of top predictions to evaluate.
        
        Returns:
        - Recall at k: float (0.0 if no positive samples exist)
        """
        y_true = np.asarray(y_true)
        probs = np.asarray(probs)
        
        # Count total positive samples
        if probs.ndim == 1:
            # Binary case
            total_positives = np.sum(y_true == 1)
            if total_positives == 0:
                return 0.0
            predictions = (probs > 0.5).astype(int)
            # Confidence = probability of the predicted class
            confidences = np.where(predictions == 1, probs, 1 - probs)
        else:
            # Multi-class case - count all samples as they can all be "positive" for their respective class
            total_positives = len(y_true)
            if total_positives == 0:
                return 0.0
            confidences = np.max(probs, axis=1)
            predictions = np.argmax(probs, axis=1)
        
        # Get top-k most confident samples
        k = min(k, len(y_true))
        if k == 0:
            return 0.0
            
        top_k_indices = np.argpartition(confidences, -k)[-k:]
        
        # Count how many positive samples were found in top-k
        if probs.ndim == 1:
            # Binary case - count ALL positive samples in top-k (not just correctly predicted ones)
            found_positives = np.sum(y_true[top_k_indices] == 1)
        else:
            # Multi-class case - count correctly predicted samples in top-k
            found_positives = np.sum(y_true[top_k_indices] == predictions[top_k_indices])
        
        return found_positives / total_positives
    
    def f1_at_k(self, y_true, probs, k):
        """
        Computes F1@k - harmonic mean of Precision@k and Recall@k.
        
        Parameters:
        - y_true: array of shape (n_samples,), true class labels.
        - probs: array of shape (n_samples,) for binary or (n_samples, n_classes) for multi-class
        - k: int, number of top predictions to evaluate.
        
        Returns:
        - F1 at k: float
        """
        precision_k = self.precision_at_k(y_true, probs, k)
        recall_k = self.recall_at_k(y_true, probs, k)
        
        if precision_k + recall_k == 0:
            return 0.0
        
        return 2 * (precision_k * recall_k) / (precision_k + recall_k)
    
    def precision_at_k_positive_predicted(self, y_true, probs, k):
        """
        Computes precision among the k samples with highest positive class prediction.
        Only for binary classification.
        
        Parameters:
        - y_true: array of shape (n_samples,), true class labels (0/1).
        - probs: array of shape (n_samples,) with probabilities for positive class
        - k: int, number of top positive predictions to evaluate.
        
        Returns:
        - Precision among k most positively predicted samples: float
        """
        y_true = np.asarray(y_true)
        probs = np.asarray(probs)
        
        if probs.ndim != 1:
            raise ValueError("This metric is only for binary classification (1D probs)")
        
        # Get top-k samples with highest positive class probability
        k = min(k, len(y_true))
        if k == 0:
            return 0.0
            
        top_k_indices = np.argpartition(probs, -k)[-k:]
        
        # Count how many of these top-k are actually positive
        correct = np.sum(y_true[top_k_indices] == 1)
        
        return correct / k
    
    def recall_at_k_positive_predicted(self, y_true, probs, k):
        """
        Computes recall among the k samples with highest positive class prediction.
        Only for binary classification.
        
        Parameters:
        - y_true: array of shape (n_samples,), true class labels (0/1).
        - probs: array of shape (n_samples,) with probabilities for positive class
        - k: int, number of top positive predictions to evaluate.
        
        Returns:
        - Recall among k most positively predicted samples: float
        """
        y_true = np.asarray(y_true)
        probs = np.asarray(probs)
        
        # Count total positive samples
        total_positives = np.sum(y_true == 1)
        if total_positives == 0:
            return 0.0
        
        if probs.ndim != 1:
            raise ValueError("This metric is only for binary classification (1D probs)")
        
        # Get top-k samples with highest positive class probability
        k = min(k, len(y_true))
        if k == 0:
            return 0.0
            
        top_k_indices = np.argpartition(probs, -k)[-k:]
        
        # Count how many positive samples were found in top-k
        found_positives = np.sum(y_true[top_k_indices] == 1)
        
        return found_positives / total_positives
    
    def f1_at_k_positive_predicted(self, y_true, probs, k):
        """
        Computes F1 score among the k samples with highest positive class prediction.
        Only for binary classification.
        
        Parameters:
        - y_true: array of shape (n_samples,), true class labels (0/1).
        - probs: array of shape (n_samples,) with probabilities for positive class
        - k: int, number of top positive predictions to evaluate.
        
        Returns:
        - F1 score among k most positively predicted samples: float
        """
        precision_k = self.precision_at_k_positive_predicted(y_true, probs, k)
        recall_k = self.recall_at_k_positive_predicted(y_true, probs, k)
        
        if precision_k + recall_k == 0:
            return 0.0
        
        return 2 * (precision_k * recall_k) / (precision_k + recall_k)

    def ndcg_at_k(self, y_true, probs, k):
        """
        Computes NDCG@k - Normalized Discounted Cumulative Gain at k.
        Ranks samples by predicted positive probability and evaluates
        how well the ranking places true positives at the top.

        Parameters:
        - y_true: array of shape (n_samples,), binary true labels (0/1).
        - probs: array of shape (n_samples,), predicted probabilities for positive class.
        - k: int, number of top predictions to evaluate.

        Returns:
        - NDCG at k: float (0.0 to 1.0)
        """
        y_true = np.asarray(y_true)
        probs = np.asarray(probs)

        if probs.ndim != 1:
            return 0.0

        k = min(k, len(y_true))
        if k == 0:
            return 0.0

        # Sort by predicted probability (descending)
        ranked_indices = np.argsort(probs)[::-1][:k]
        ranked_relevance = y_true[ranked_indices].astype(float)

        # DCG@k
        discounts = np.log2(np.arange(2, k + 2))  # log2(2), log2(3), ..., log2(k+1)
        dcg = np.sum(ranked_relevance / discounts)

        # Ideal DCG@k (all positives ranked first)
        ideal_relevance = np.sort(y_true)[::-1][:k].astype(float)
        idcg = np.sum(ideal_relevance / discounts)

        if idcg == 0:
            return 0.0

        return dcg / idcg

    def _compute_metrics(
        self,
        y_true,
        y_pred,
        y_score,
        y_probs=None,
        mode: str = "val",
        suffix: str = "",
        is_multiclass: bool = False,
        update_best: bool = True,
    ):
        # --- Base Metrics ---
        if is_multiclass:
            auc_roc = (
                roc_auc_score(y_true, y_probs, multi_class="ovr")
                if y_probs is not None
                else None
            )
            auc_pr = (
                None  # PR AUC is undefined for multiclass unless binarized per class
            )
            f1 = f1_score(y_true, y_pred, average="weighted")
            recall = recall_score(y_true, y_pred, average="weighted")
            precision = precision_score(y_true, y_pred, average="weighted")
            class_names = [str(cls) for cls in np.unique(y_true)]
        else:
            auc_roc = roc_auc_score(y_true, y_score)
            auc_pr = average_precision_score(y_true, y_score)

            f1 = f1_score(y_true, y_pred)
            recall = recall_score(y_true, y_pred)
            precision = precision_score(y_true, y_pred)
            class_names = ["negative", "positive"]

        accuracy = accuracy_score(y_true, y_pred)

        # --- Precision@K, Recall@K, F1@K ---
        k_values = [5, 10, 20, 50, 100, 1000]
        precision_at_ks = {k: self.precision_at_k(y_true, y_probs, k) for k in k_values}
        recall_at_ks = {k: self.recall_at_k(y_true, y_probs, k) for k in k_values}
        f1_at_ks = {k: self.f1_at_k(y_true, y_probs, k) for k in k_values}
        
        # --- @K Positive Predicted (Binary only) ---
        precision_at_k_pos_pred = {}
        recall_at_k_pos_pred = {}
        f1_at_k_pos_pred = {}
        ndcg_at_ks = {}

        if not is_multiclass and y_probs is not None and y_probs.ndim == 1:
            precision_at_k_pos_pred = {k: self.precision_at_k_positive_predicted(y_true, y_probs, k) for k in k_values}
            recall_at_k_pos_pred = {k: self.recall_at_k_positive_predicted(y_true, y_probs, k) for k in k_values}
            f1_at_k_pos_pred = {k: self.f1_at_k_positive_predicted(y_true, y_probs, k) for k in k_values}
            ndcg_at_ks = {k: self.ndcg_at_k(y_true, y_probs, k) for k in k_values}
            ndcg_at_ks["full"] = self.ndcg_at_k(y_true, y_probs, len(y_true))

        # --- Confusion Matrix ---
        cm = confusion_matrix(y_true, y_pred)
        cm_df = DataFrame(cm, index=class_names, columns=class_names)
        cm_df.index.name = "Actual"
        cm_df.columns.name = "Predicted"

        classification_metrics = classification_report(
            y_true, y_pred, target_names=class_names, output_dict=True
        )

        # --- Print Results ---
        print(f"\n--- Evaluation Results [{mode.upper()} - {suffix}] ---")
        print(f"AUC-ROC: {auc_roc:.4f}" if auc_roc is not None else "AUC-ROC: N/A")
        print(f"AUC-PR: {auc_pr:.4f}" if auc_pr is not None else "AUC-PR: N/A")
        print(
            f"F1: {f1:.4f}, Recall: {recall:.4f}, Precision: {precision:.4f}, Accuracy: {accuracy:.4f}"
        )
        for k in k_values:
            print(f"Precision@{k}: {precision_at_ks[k]:.4f}, Recall@{k}: {recall_at_ks[k]:.4f}, F1@{k}: {f1_at_ks[k]:.4f}")
        
        # Print positive predicted metrics for binary classification
        if precision_at_k_pos_pred:
            print("\n--- Metrics for K Most Positively Predicted Samples ---")
            # Get detailed counts for max K value
            y_true_np = np.asarray(y_true)
            y_probs_np = np.asarray(y_probs)
            
            # Normal @K analysis
            predictions = (y_probs_np > 0.5).astype(int)
            confidences = np.where(predictions == 1, y_probs_np, 1 - y_probs_np)
            max_k = min(max(k_values), len(confidences))

            pos_predictions = np.sum(predictions == 1)
            neg_predictions = np.sum(predictions == 0)
            # Confidence analysis for each prediction type
            pos_mask = predictions == 1
            neg_mask = predictions == 0
            
            if pos_predictions > 0:
                pos_conf_mean = np.mean(confidences[pos_mask])
                pos_conf_min = np.min(confidences[pos_mask])
                pos_conf_max = np.max(confidences[pos_mask])
                print(f"   Positive prediction confidence - Mean: {pos_conf_mean:.4f}, Min: {pos_conf_min:.4f}, Max: {pos_conf_max:.4f}")
            else:
                print(f"   Positive prediction confidence - No positive predictions!")
                
            if neg_predictions > 0:
                neg_conf_mean = np.mean(confidences[neg_mask])
                neg_conf_min = np.min(confidences[neg_mask])
                neg_conf_max = np.max(confidences[neg_mask])
                print(f"   Negative prediction confidence - Mean: {neg_conf_mean:.4f}, Min: {neg_conf_min:.4f}, Max: {neg_conf_max:.4f}")
            else:
                print(f"   Negative prediction confidence - No negative predictions!")
            
            # Top-K confidence analysis
            normal_top_k_indices = np.argpartition(confidences, -max_k)[-max_k:]
            top_k_confidences = confidences[normal_top_k_indices]
            top_k_probs = y_probs_np[normal_top_k_indices]
            top_k_predictions = predictions[normal_top_k_indices]
            
            # Get actual predictions and true labels for top-k samples
            normal_top_k_true = y_true_np[normal_top_k_indices]
            normal_top_k_pred = predictions[normal_top_k_indices]
            
            # Calculate full confusion matrix for top-k
            normal_tp = np.sum((normal_top_k_true == 1) & (normal_top_k_pred == 1))
            normal_fp = np.sum((normal_top_k_true == 0) & (normal_top_k_pred == 1))
            normal_fn = np.sum((normal_top_k_true == 1) & (normal_top_k_pred == 0))
            normal_tn = np.sum((normal_top_k_true == 0) & (normal_top_k_pred == 0))
            
            # PosPred @K analysis  
            pos_top_k_indices = np.argpartition(y_probs_np, -max_k)[-max_k:]
            pos_tp = np.sum(y_true_np[pos_top_k_indices] == 1)
            pos_fp = max_k - pos_tp
            
            print(f"\n📊 RESULTS:")
            print(f"Normal @{max_k}: TP={normal_tp}, FP={normal_fp}, TN={normal_tn}, FN={normal_fn}")
            print(f"Normal @{max_k} Total: {normal_tp + normal_fp + normal_tn + normal_fn} (should be {max_k})")
            print(f"PosPred @{max_k}: TP={pos_tp}, FP={pos_fp}")
            print()
            
            for k in k_values:
                print(f"PosPred-Precision@{k}: {precision_at_k_pos_pred[k]:.4f}, PosPred-Recall@{k}: {recall_at_k_pos_pred[k]:.4f}, PosPred-F1@{k}: {f1_at_k_pos_pred[k]:.4f}")

            if ndcg_at_ks:
                print("\n--- NDCG (Normalized Discounted Cumulative Gain) ---")
                for k in k_values:
                    print(f"NDCG@{k}: {ndcg_at_ks[k]:.4f}")
                print(f"NDCG@full ({len(y_true)}): {ndcg_at_ks['full']:.4f}")

        print(f"Confusion Matrix:\n{cm_df}\n")
        print(
            f"Classification Report:\n{classification_report(y_true, y_pred, target_names=class_names)}"
        )

        # --- Curves (only for binary classification) ---
        if not is_multiclass:
            precision_vals, recall_vals, _ = precision_recall_curve(y_true, y_score)
            fpr, tpr, _ = roc_curve(y_true, y_score)

            # Plot PR curve
            plt.figure()
            plt.plot(recall_vals, precision_vals, label=f"PR AUC = {auc_pr:.4f}")
            plt.xlabel("Recall")
            plt.ylabel("Precision")
            plt.title(f"Precision-Recall Curve ({mode} {suffix})")
            plt.legend(loc="lower left")
            plt.grid()
            if self.use_wandb:
                wandb.log({f"{mode}_{suffix}_pr_curve": wandb.Image(plt)})
            plt.close()

            # Plot ROC curve
            plt.figure()
            plt.plot(fpr, tpr, label=f"ROC AUC = {auc_roc:.4f}")
            plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            plt.title(f"ROC Curve ({mode} {suffix})")
            plt.legend(loc="lower right")
            plt.grid()
            if self.use_wandb:
                wandb.log({f"{mode}_{suffix}_roc_curve": wandb.Image(plt)})
            plt.close()

        # --- Update best model metric ---
        if update_best:
            try:
                metric_v = eval(self.optimization_metric_type)
                if metric_v is not None:
                    self.update_best_model(metric_v)
            except Exception as e:
                print(f"Warning: Failed to update best model in _compute_metrics: {e}")

        # --- Build metrics dict (always, for JSON export + wandb) ---
        metrics_dict = {
            f"{mode}_{key}_{suffix}": val
            for key, val in zip(
                ["auc_roc", "auc_pr", "f1", "recall", "precision", "accuracy"]
                + [f"precision_at_k_{k}" for k in k_values]
                + [f"recall_at_k_{k}" for k in k_values]
                + [f"f1_at_k_{k}" for k in k_values],
                [auc_roc, auc_pr, f1, recall, precision, accuracy]
                + list(precision_at_ks.values())
                + list(recall_at_ks.values())
                + list(f1_at_ks.values()),
            )
        }

        # Add positive predicted metrics for binary classification
        if precision_at_k_pos_pred:
            pos_pred_metrics = {
                f"{mode}_precision_at_k_pos_pred_{k}_{suffix}": v for k, v in precision_at_k_pos_pred.items()
            }
            pos_pred_metrics.update({
                f"{mode}_recall_at_k_pos_pred_{k}_{suffix}": v for k, v in recall_at_k_pos_pred.items()
            })
            pos_pred_metrics.update({
                f"{mode}_f1_at_k_pos_pred_{k}_{suffix}": v for k, v in f1_at_k_pos_pred.items()
            })
            metrics_dict.update(pos_pred_metrics)

        # Add NDCG metrics
        if ndcg_at_ks:
            ndcg_metrics = {
                f"{mode}_ndcg_at_{k}_{suffix}": v for k, v in ndcg_at_ks.items()
            }
            metrics_dict.update(ndcg_metrics)

        if not is_multiclass:
            metrics_dict[f"{mode}_{suffix}_positive_ratio"] = float(
                np.sum(y_pred)
            ) / len(y_pred)

        # --- Log to wandb ---
        if self.use_wandb:
            wandb_metrics = dict(metrics_dict)
            wandb_metrics[f"{mode}_{suffix}_classification_report"] = (
                classification_metrics
            )
            wandb.log(wandb_metrics)

            wandb.log(
                {
                    f"{mode}_{suffix}_confusion_matrix": wandb.plot.confusion_matrix(
                        preds=y_pred,
                        y_true=y_true,
                        class_names=class_names,
                    )
                }
            )

        return eval(self.optimization_metric_type), metrics_dict



    def _get_predictions(self, graph_data, eval_mask, target_mode):
        with torch.no_grad():
            try:
                out = self.model(graph_data.x_dict, graph_data.edge_index_dict, batch=graph_data, eval_mask=eval_mask)
            except TypeError:
                out = self.model(graph_data.x_dict, graph_data.edge_index_dict)
            
        result = {}

        def _process_predictions(output, target, is_binary, calibrator_instance=None):
            # Handle different output structures
            if isinstance(output, torch.Tensor):
                pass
            elif isinstance(output, dict) and "startup" in output and isinstance(output["startup"], dict) and "output" in output["startup"]["startup"]:
                 pass
            
            if is_binary:
                # Sigmoid
                logits = torch.sigmoid(output.view(-1))
                
                # Apply calibration if available
                calib = calibrator_instance if calibrator_instance else self.calibrator
                
                if calib is not None:
                    uncalibrated_probs = logits.detach().cpu().numpy()
                    
                    if hasattr(calib, 'temperature_scaler') and calib.temperature_scaler:
                        epsilon = 1e-7
                        uncalibrated_probs_clipped = np.clip(uncalibrated_probs, epsilon, 1 - epsilon)
                        logits_for_temp = np.log(uncalibrated_probs_clipped / (1 - uncalibrated_probs_clipped))
                        calibrated_probs = calib.calibrate_probabilities(logits_for_temp, from_logits=True)
                    elif hasattr(calib, 'calibrate_probabilities'):
                        calibrated_probs = calib.calibrate_probabilities(uncalibrated_probs, from_logits=False)
                    else:
                        calibrated_probs = uncalibrated_probs
                    
                    calibrated_logits = torch.tensor(calibrated_probs, device=logits.device)
                    
                    robust_threshold = self.get_robust_threshold(calibrated_probs)
                    pred = (calibrated_logits > robust_threshold).long()
                    conf = calibrated_logits
                    
                    if target is not None:
                         y_out = target.detach().cpu().numpy()
                    else:
                         y_out = None
                         
                    return {
                        "y": y_out,
                        "pred": pred.detach().cpu().numpy(),
                        "conf": conf.detach().cpu().numpy(),
                        "probs": calibrated_probs,
                        "probs_uncalibrated": uncalibrated_probs,
                    }
                else:
                    pred = (logits > self.optimal_threshold).long()
                    conf = logits
                    
                    if target is not None:
                         y_out = target.detach().cpu().numpy()
                    else:
                         y_out = None
                         
                    return {
                        "y": y_out,
                        "pred": pred.detach().cpu().numpy(),
                        "conf": conf.detach().cpu().numpy(),
                        "probs": logits.detach().cpu().numpy(),
                    }
            else:
                logits = F.softmax(output, dim=-1)
                pred = logits.argmax(dim=-1)
                conf = logits.max(dim=-1).values
                
                return {
                    "y": target.detach().cpu().numpy() if target is not None else None,
                    "pred": pred.detach().cpu().numpy(),
                    "conf": conf.detach().cpu().numpy(),
                    "probs": logits.detach().cpu().numpy(),
                }  

        if target_mode == "masked_multi_task":
            # Extract raw outputs
            out_mom = out["out_mom"] # [N]
            out_liq = out["out_liq"] # [N]
            
            # Extract Targets and Masks
            # Target Tensor: [N, 4] -> Mom, Liq, MaskMom, MaskLiq
            targets_all = graph_data["startup"].y
            
            # Filter by eval_mask (e.g. val_mask or test_mask)
            # We must apply the eval_mask FIRST to get the subset of nodes we are evaluating
            out_mom_eval = out_mom[eval_mask]
            out_liq_eval = out_liq[eval_mask]
            targets_eval = targets_all[eval_mask]
            
            y_mom = targets_eval[:, 0]
            y_liq = targets_eval[:, 1]
            mask_mom = targets_eval[:, 2]
            mask_liq = targets_eval[:, 3]
            
            valid_mom_idx = mask_mom == 1
            valid_liq_idx = mask_liq == 1 # "Is Mature"
            
            # Momentum Processing
            calib_mom = self.calibrator
            mom_res = _process_predictions(
                out_mom_eval[valid_mom_idx], 
                target=y_mom[valid_mom_idx], 
                is_binary=True, 
                calibrator_instance=calib_mom
            )
            
            # Liquidity Processing
            calib_liq = self.calibrator
            liq_res = _process_predictions(
                out_liq_eval[valid_liq_idx],
                target=y_liq[valid_liq_idx],
                is_binary=True,
                calibrator_instance=calib_liq
            )
            
            result.update({f"mom_{k}": v for k, v in mom_res.items()})
            result.update({f"liq_{k}": v for k, v in liq_res.items()})

            # --- Full Outputs for Export/Backtest (Unfiltered) ---
            # Momentum Full
            mom_res_full = _process_predictions(
                out_mom_eval, 
                target=y_mom, 
                is_binary=True, 
                calibrator_instance=calib_mom
            )
            # Liquidity Full
            liq_res_full = _process_predictions(
                out_liq_eval,
                target=y_liq, 
                is_binary=True,
                calibrator_instance=calib_liq
            )
            
            result.update({f"mom_full_{k}": v for k, v in mom_res_full.items()})
            result.update({f"liq_full_{k}": v for k, v in liq_res_full.items()})

        if target_mode == "multi_label":
            # output is potentially dict {'multi_label_output': ...}
            if isinstance(out, dict) and "multi_label_output" in out:
                out_tensor = out["multi_label_output"]
            elif isinstance(out, dict) and "startup" in out and isinstance(out["startup"], dict):
                # Handle nested dict if models structure differs
                out_tensor = out["startup"].get("multi_label_output", out["startup"])
            else:
                out_tensor = out # Fallback/Simpler tests
                
            if isinstance(out_tensor, dict):
                 out_tensor = out_tensor.get("startup", out_tensor) # Try getting node type
            
            # Now we have [N, 3] logits
            # Filter by eval_mask
            out_tensor = out_tensor[eval_mask]
            target_tensor = graph_data["startup"].y[eval_mask]

            # For each task, process separately
            
            # Funding (0)
            fund_out = out_tensor[:, 0]
            fund_target = target_tensor[:, 0]
            calib_fund = self.calibrator.get("funding") if isinstance(self.calibrator, dict) else self.calibrator
            fund_res = _process_predictions(fund_out, target=fund_target, is_binary=True, calibrator_instance=calib_fund)
            
            # Acquisition (1)
            acq_out = out_tensor[:, 1]
            # Handle list/tuple target if needed, but preprocessing returns [N, 3] tensor
            acq_target = target_tensor[:, 1]
            calib_acq = self.calibrator.get("acquisition") if isinstance(self.calibrator, dict) else self.calibrator
            acq_res = _process_predictions(acq_out, target=acq_target, is_binary=True, calibrator_instance=calib_acq)
            
            # IPO (2)
            ipo_out = out_tensor[:, 2]
            ipo_target = target_tensor[:, 2]
            calib_ipo = self.calibrator.get("ipo") if isinstance(self.calibrator, dict) else self.calibrator
            ipo_res = _process_predictions(ipo_out, target=ipo_target, is_binary=True, calibrator_instance=calib_ipo)

            result.update({f"fund_{k}": v for k, v in fund_res.items()})
            result.update({f"acq_{k}": v for k, v in acq_res.items()})
            result.update({f"ipo_{k}": v for k, v in ipo_res.items()})

        elif target_mode == "binary_prediction":
            result.update(
                _process_predictions(out[eval_mask].view(-1), graph_data["startup"].y[eval_mask], is_binary=True)
            )

        elif target_mode == "multi_prediction":
            result.update(
                _process_predictions(out[eval_mask], graph_data["startup"].y[eval_mask], is_binary=False)
            )

        elif target_mode == "multi_task":
            binary = _process_predictions(
                out["binary_output"][eval_mask].view(-1), target=graph_data["startup"].y[1][eval_mask], is_binary=True
            )
            multi = _process_predictions(
                out["multi_class_output"][eval_mask], target=graph_data["startup"].y[0][eval_mask], is_binary=False
            )

            # Prefix keys to avoid collisions
            result.update({f"binary_{k}": v for k, v in binary.items()})
            result.update({f"multi_{k}": v for k, v in multi.items()})
            
        # Capture SeHGNN Attention Weights if available
        if isinstance(out, dict) and "attention_weights" in out:
            result["attention_weights"] = out["attention_weights"]
            result["metapath_names"] = out["metapath_names"]

        return result

    def evaluate(
        self,
        graph_data,
        mode: str,
        target_mode,
        current_epoch=None,
        best_model_callback=None,
    ):
        assert mode in {"val", "test", "test_original"}, f"Unsupported mode: {mode}"
        self.model.eval()
        graph_data = graph_data.to(self.device)
        eval_mask = (
            graph_data["startup"].val_mask
            if mode == "val"
            else (
                graph_data["startup"].test_mask_original
                if mode == "test_original"
                else graph_data["startup"].test_mask
            )
        )

        # Determine split key for feature swapping
        split_key = None
        if mode == "val":
            split_key = "x_val_mask"
        elif mode == "test":
            split_key = "x_test_mask"
        elif mode == "test_original":
            split_key = "x_test_mask_original"
            
        # Swap features if available
        original_x = None
        if split_key and hasattr(graph_data["startup"], split_key):
            print(f"🔄 Swapping features to {split_key} for evaluation")
            original_x = graph_data["startup"].x
            graph_data["startup"].x = getattr(graph_data["startup"], split_key)

        # Run prediction and extract y, preds, conf
        metric_value = 0.0
        try:
            preds = self._get_predictions(graph_data, eval_mask, target_mode)
        finally:
            # Always swap back to original features
            if original_x is not None:
                graph_data["startup"].x = original_x
                
        # Visualize SeHGNN Weights if available and in test mode
        if mode in ["test", "test_original"] and "attention_weights" in preds:
            print("\n📊 Visualizing SeHGNN Attention Weights...")
            from .visualize import visualize_metapath_weights
            
            attn_weights = preds["attention_weights"]
            metapath_names = preds["metapath_names"]
            
            # Compute importance: Mean over Batch(0), Heads(1), TargetMetapaths(2)
            # Result: [M]
            importance = attn_weights.mean(dim=(0, 1, 2)).detach().cpu().numpy()
            
            sehgnn_weights = {
                mp: float(imp) 
                for mp, imp in zip(metapath_names, importance)
            }
            
            # Format for visualizer: {layer: {node_type: {metapath: weight}}}
            weights_dict = {
                "SeHGNN_Transformer": {
                    "startup": sehgnn_weights
                }
            }
            
            # Print Top Weights
            print("\n🔥 Top 10 SeHGNN Attention Weights:")
            sorted_weights = sorted(sehgnn_weights.items(), key=lambda x: x[1], reverse=True)
            for mp, w in sorted_weights[:10]:
                print(f"   {mp}: {w:.4f}")
            
            output_dir = self.config.get("output_dir", "outputs")
            # Use same structure as HAN: outputs/metapath_weights
            viz_dir = os.path.join(output_dir, "metapath_weights")
            visualize_metapath_weights(weights_dict, output_dir=viz_dir)
            
            if self.use_wandb:
                # Convert tuple keys to strings for JSON serialization
                sehgnn_weights_str = {str(k): v for k, v in sehgnn_weights.items()}
                wandb.log({"sehgnn_weights": sehgnn_weights_str})
                
                # Log the visualization image
                img_path = os.path.join(viz_dir, "metapath_weights_SeHGNN_Transformer_startup.png")
                if os.path.exists(img_path):
                    wandb.log({"sehgnn_weights_plot": wandb.Image(img_path)})

        all_metrics = {}  # Accumulate metrics from all _compute_metrics calls

        if target_mode == "binary_prediction":
            metric_value, m_dict = self._compute_metrics(
                preds["y"],
                preds["pred"],
                preds["conf"],
                preds["probs"],
                mode,
                suffix="binary",
            )
            all_metrics.update(m_dict)

        elif target_mode == "multi_prediction":
            metric_value, m_dict = self._compute_metrics(
                preds["y"],
                preds["pred"],
                preds["conf"],
                preds["probs"],
                mode,
                suffix="multi",
                is_multiclass=True,
            )
            all_metrics.update(m_dict)

            # Evaluation only on status change samples
            if self.config["data_processing"]["test"]["test_change"]:
                status_changed_mask = graph_data["startup"].status_changed.to_numpy()[
                    eval_mask.cpu().numpy()
                ]
                if np.any(status_changed_mask):
                    print("\n--- Evaluating on status change samples ---")
                    _, change_dict = self._compute_metrics(
                        preds["y"][status_changed_mask],
                        preds["pred"][status_changed_mask],
                        preds["conf"][status_changed_mask],
                        preds["probs"][status_changed_mask],
                        mode,
                        suffix="multi_change",
                        is_multiclass=True,
                    )
                    all_metrics.update(change_dict)
                else:
                    print(f"\n--- No status changes found in {mode} set ---")

        elif target_mode == "masked_multi_task":
             # 1. Momentum Metrics
             mom_res, mom_dict = self._compute_metrics(
                 y_true=preds["mom_y"],
                 y_pred=preds["mom_pred"],
                 y_score=preds["mom_conf"],
                 y_probs=preds["mom_probs"],
                 mode=mode,
                 suffix="mom",
                 update_best=False
             )
             all_metrics.update(mom_dict)

             # 2. Liquidity Metrics (mature subset only)
             if len(preds["liq_y"]) > 0:
                 liq_res, liq_dict = self._compute_metrics(
                     y_true=preds["liq_y"],
                     y_pred=preds["liq_pred"],
                     y_score=preds["liq_conf"],
                     y_probs=preds["liq_probs"],
                     mode=mode,
                     suffix="liq",
                     update_best=False
                 )
                 all_metrics.update(liq_dict)
             else:
                 print(f"⚠️ No valid mature samples for Liquidity evaluation in {mode} set.")
                 liq_res = 0.0

             # 3. Liquidity Metrics (full test set, including non-mature)
             if len(preds.get("liq_full_y", [])) > 0:
                 _, liq_full_dict = self._compute_metrics(
                     y_true=preds["liq_full_y"],
                     y_pred=preds["liq_full_pred"],
                     y_score=preds["liq_full_conf"],
                     y_probs=preds["liq_full_probs"],
                     mode=mode,
                     suffix="liq_full",
                     update_best=False
                 )
                 all_metrics.update(liq_full_dict)

             w_mom = 0.5
             w_liq = 0.5
             metric_value = (w_mom * mom_res) + (w_liq * liq_res)

             # Joint AUC-PR / AUC-ROC: equal-weight average of the two masked
             # tasks. These are the sweep-selection metrics for the ICDM
             # leaderboard — single scalar that reflects performance on both
             # momentum (NFR) and liquidity (Exit) simultaneously, rather than
             # optimizing either task in isolation.
             for key in ("auc_pr", "auc_roc"):
                 mom_k = f"{mode}_{key}_mom"
                 liq_k = f"{mode}_{key}_liq"
                 if mom_k in all_metrics and liq_k in all_metrics:
                     all_metrics[f"{mode}_{key}_joint"] = 0.5 * (
                         all_metrics[mom_k] + all_metrics[liq_k]
                     )
             # Also push joint scalars to wandb for sweep-optimizer access
             if self.use_wandb:
                 joint_log = {
                     k: v for k, v in all_metrics.items()
                     if k.endswith("_joint")
                 }
                 if joint_log:
                     wandb.log(joint_log)

        elif target_mode == "multi_task":
            # Compute Binary Metrics
            binary_metric_val, bin_dict = self._compute_metrics(
                preds["binary_y"],
                preds["binary_pred"],
                preds["binary_conf"],
                preds["binary_probs"],
                mode,
                suffix="binary",
                update_best=False,
            )
            all_metrics.update(bin_dict)

            # Compute Multi-class Metrics
            multi_metric_val, multi_dict = self._compute_metrics(
                preds["multi_y"],
                preds["multi_pred"],
                preds["multi_conf"],
                preds["multi_probs"],
                mode,
                suffix="multi",
                is_multiclass=True,
                update_best=False,
            )
            all_metrics.update(multi_dict)

            # Hybrid Metric Selection
            if "auc" in self.optimization_metric_type.lower():
                if binary_metric_val is not None:
                    print("⚠️ Multi-Task Optimization: Using Binary Task metric (Multi-class AUC often invalid)")
                    metric_value = binary_metric_val
                else:
                    metric_value = multi_metric_val if multi_metric_val is not None else 0.0
            else:
                v1 = binary_metric_val if binary_metric_val is not None else 0.0
                v2 = multi_metric_val if multi_metric_val is not None else 0.0
                metric_value = (v1 + v2) / 2.0
                print(f"⚠️ Multi-Task Optimization: Using Average of Binary ({v1:.4f}) and Multi ({v2:.4f})")

        elif target_mode == "multi_label":
            # Compute Metrics for each task
            fund_res, fund_dict = self._compute_metrics(
                preds["fund_y"], preds["fund_pred"], preds["fund_conf"], preds["fund_probs"],
                mode, suffix="fund", update_best=False
            )
            all_metrics.update(fund_dict)
            acq_res, acq_dict = self._compute_metrics(
                preds["acq_y"], preds["acq_pred"], preds["acq_conf"], preds["acq_probs"],
                mode, suffix="acq", update_best=False
            )
            all_metrics.update(acq_dict)
            ipo_res, ipo_dict = self._compute_metrics(
                preds["ipo_y"], preds["ipo_pred"], preds["ipo_conf"], preds["ipo_probs"],
                mode, suffix="ipo", update_best=False
            )
            all_metrics.update(ipo_dict)

            # Weighted average
            weights_config = self.config["data_processing"].get("multi_label", {}).get("combined_metric_weights", {})
            w_fund = float(weights_config.get("funding", 0.2))
            w_acq = float(weights_config.get("acquisition", 0.3))
            w_ipo = float(weights_config.get("ipo", 0.5))

            fund_score = fund_res if fund_res is not None else 0.0
            acq_score = acq_res if acq_res is not None else 0.0
            ipo_score = ipo_res if ipo_res is not None else 0.0

            metric_value = (w_fund * fund_score) + (w_acq * acq_score) + (w_ipo * ipo_score)
            print(f"Multi-Label Combined Score ({self.optimization_metric_type}): {metric_value:.4f} "
                  f"(Fund: {fund_score:.4f}, Acq: {acq_score:.4f}, IPO: {ipo_score:.4f})")

        # --- JSON Metrics Export ---
        export_enabled = self.config.get("eval", {}).get("export_metrics_json", True)
        if export_enabled and all_metrics:
            from .metrics_export import export_metrics_json
            output_base = self.config.get("output_dir", "outputs")
            results_dir = os.path.join(output_base, "results")
            model_name = self.config.get("train", {}).get("model", "unknown")

            export_metrics_json(
                metrics=all_metrics,
                config=self.config,
                mode=mode,
                epoch=current_epoch if current_epoch is not None else self.current_epoch,
                model_name=model_name,
                target_mode=target_mode,
                best_metric=self.best_metric,
                best_epoch=self.best_epoch,
                output_base_dir=results_dir,
            )

        if (
            mode == "val"
            and current_epoch is not None
            and current_epoch >= self.min_amount_of_epochs
        ):
            if metric_value is None: metric_value = 0.0
            
            if metric_value > self.best_metric:
                self.best_metric = metric_value
                self.best_epoch = current_epoch
                if best_model_callback:
                    best_model_callback()
                            
        # --- Post-Processing (Export & Analysis) ---
        if mode in ["test", "test_original"]:
            # 1. Prepare Prediction Tuples (UUID, Score, Label)
            prediction_tuples = []
            
            if hasattr(graph_data['startup'], 'df'):
                import pandas as pd
                startup_df = graph_data['startup'].df
                target_indices = eval_mask.nonzero(as_tuple=True)[0].cpu().numpy()
                
                for i, idx in enumerate(target_indices):
                    if idx < len(startup_df):
                        uuid = startup_df.iloc[idx]['startup_uuid']
                        # Handle outputs for tuples
                        score = 0
                        label = 0
                        
                        if target_mode == "multi_task":
                            score = float(preds["binary_probs"][i])
                            label = float(preds["binary_y"][i])
                        elif target_mode == "masked_multi_task":
                             # Use FULL outputs to ensure alignment with eval_mask indices
                             # Convert to Python float for full float64 precision in CSV export
                             score = {
                                 "mom": float(preds.get("mom_full_probs", [])[i]) if len(preds.get("mom_full_probs", [])) > i else 0.0,
                                 "liq": float(preds.get("liq_full_probs", [])[i]) if len(preds.get("liq_full_probs", [])) > i else 0.0
                             }
                             label = {
                                 "mom": float(preds.get("mom_full_y", [])[i]) if len(preds.get("mom_full_y", [])) > i else 0.0,
                                 "liq": float(preds.get("liq_full_y", [])[i]) if len(preds.get("liq_full_y", [])) > i else 0.0
                             }
                        elif target_mode == "multi_label":
                            score = {
                                "fund": float(preds.get("fund_probs", [])[i]),
                                "acq": float(preds.get("acq_probs", [])[i]),
                                "ipo": float(preds.get("ipo_probs", [])[i])
                            }
                            label = {
                                "fund": float(preds.get("fund_y", [])[i]),
                                "acq": float(preds.get("acq_y", [])[i]),
                                "ipo": float(preds.get("ipo_y", [])[i])
                            }
                        else:
                            score = float(preds["probs"][i])
                            label = float(preds["y"][i])
                            
                        prediction_tuples.append((uuid, score, label))
            else:
                print("⚠️ startup_df not found in graph_data. Cannot export predictions with UUIDs.")

            # 2. Export Predictions to CSV (unique path per run)
            if self.config.get("eval", {}).get("export_predictions", False) and prediction_tuples:
                print("📤 Starting prediction export...")
                seed = self.config.get("seed")
                export_data = [
                        {"org_uuid": p[0], "prediction": p[1], "gt_label": p[2], "seed": seed}
                        for p in prediction_tuples
                    ]
                export_df = pd.DataFrame(export_data)
                # Build unique path to prevent overwrites across runs
                model_name = self.config.get("train", {}).get("model", "unknown")
                target_mode_str = self.config.get("data_processing", {}).get("target_mode", "unknown")
                try:
                    wandb_id = wandb.run.id if wandb.run else "local"
                except (NameError, AttributeError):
                    wandb_id = "local"
                from datetime import datetime
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_dir = self.config.get("output_dir", "outputs")
                pred_dir = os.path.join(output_dir, "predictions", model_name, target_mode_str)
                os.makedirs(pred_dir, exist_ok=True)
                export_path = os.path.join(pred_dir, f"{timestamp}_{wandb_id}_predictions_{mode}.csv")
                export_df.to_csv(export_path, index=False)
                print(f"✅ Exported {len(export_df)} predictions to {export_path}")
                if self.use_wandb:
                    wandb.save(export_path)

            # 3. Downstream Analysis
            if self.analyzer and prediction_tuples:
                self.analyzer.perform_downstream_analysis(prediction_tuples)

        return metric_value, all_metrics
