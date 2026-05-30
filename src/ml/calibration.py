"""
Model Calibration Analysis and Improvement Tools

This module provides comprehensive analysis of model calibration and implements
the most basic calibration improvement method: Platt Scaling.

Calibration measures how well predicted probabilities match actual frequencies.
A well-calibrated model's confidence should match its accuracy.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.calibration import CalibrationDisplay, calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import cross_val_score
from typing import Dict, Tuple, Optional
import warnings
warnings.filterwarnings("ignore")



class ModelCalibrationAnalyzer:
    """
    Analyzes and improves model calibration using various techniques.
    
    Primary focus on Platt Scaling as the most basic and effective method.
    """
    
    def __init__(self, save_plots: bool = True, plot_dir: str = "outputs"):
        self.save_plots = save_plots
        self.plot_dir = plot_dir
        self.calibrator = None
        self.calibration_method = None
        self.optimal_threshold = 0.5  # Default threshold
        self.optimal_threshold_percentile = 50.0  # Default percentile

        

        
    def get_percentile_threshold(self, y_prob: np.ndarray) -> float:
        """
        Calculate threshold based on stored percentile from validation optimization.
        
        This provides robustness against domain shift between validation and test.
        """
        if hasattr(self, 'optimal_threshold_percentile'):
            threshold = np.percentile(y_prob, self.optimal_threshold_percentile)
            # print(f"   🎯 Using percentile-based threshold: {threshold:.4f} ({self.optimal_threshold_percentile:.1f}th percentile)")
            return threshold
        else:
            return self.optimal_threshold
        
    def find_optimal_threshold(self, y_true: np.ndarray, y_prob: np.ndarray, 
                              metric: str = "f1", title_suffix: str = "") -> float:
        """
        Find optimal classification threshold based on validation data.
        
        Args:
            y_true: True binary labels
            y_prob: Predicted probabilities (can be calibrated)
            metric: Optimization metric ('f1', 'precision', 'recall', 'accuracy')
            
        Returns:
            Optimal threshold value
        """
        from sklearn.metrics import precision_recall_curve, f1_score, precision_score, recall_score, accuracy_score
        
        # print(f"\n🎯 FINDING OPTIMAL THRESHOLD (optimizing {metric.upper()})...")
        
        # Try a range of thresholds
        thresholds = np.arange(0.01, max(0.99, np.max(y_prob) + 0.1), 0.01)
        
        best_score = 0
        best_threshold = 0.5
        scores = []
        
        for threshold in thresholds:
            y_pred = (y_prob >= threshold).astype(int)
            
            # Skip if no positive predictions
            if np.sum(y_pred) == 0:
                score = 0
            else:
                if metric == "f1":
                    score = f1_score(y_true, y_pred)
                elif metric == "precision":
                    score = precision_score(y_true, y_pred, zero_division=0)
                elif metric == "recall":
                    score = recall_score(y_true, y_pred)
                elif metric == "accuracy":
                    score = accuracy_score(y_true, y_pred)
                else:
                    raise ValueError(f"Unknown metric: {metric}")
            
            scores.append(score)
            
            if score > best_score:
                best_score = score
                best_threshold = threshold
        
        self.optimal_threshold = best_threshold
        
        # Calculate statistics with optimal threshold
        y_pred_optimal = (y_prob >= best_threshold).astype(int)
        optimal_f1 = f1_score(y_true, y_pred_optimal)
        optimal_precision = precision_score(y_true, y_pred_optimal, zero_division=0)
        optimal_recall = recall_score(y_true, y_pred_optimal)
        optimal_accuracy = accuracy_score(y_true, y_pred_optimal)
        pos_predictions = np.sum(y_pred_optimal) / len(y_pred_optimal)
        
        # print(f"   ✅ Optimal threshold found: {best_threshold:.4f}")
        # print(f"   📊 Performance at optimal threshold:")
        # print(f"      F1: {optimal_f1:.4f}")
        # print(f"      Precision: {optimal_precision:.4f}")
        # print(f"      Recall: {optimal_recall:.4f}")
        # print(f"      Accuracy: {optimal_accuracy:.4f}")
        # print(f"      Positive predictions: {pos_predictions:.1%}")
        
        # Calculate percentile-based threshold as backup
        percentile_threshold = np.percentile(y_prob, 100 - (pos_predictions * 100))
        # print(f"   📊 Equivalent percentile threshold: {percentile_threshold:.4f} ({100-pos_predictions*100:.1f}th percentile)")
        
        # Store both absolute and percentile information
        self.optimal_threshold = best_threshold
        self.optimal_threshold_percentile = 100 - (pos_predictions * 100)
        
        # Create threshold optimization plot
        self._plot_threshold_optimization(thresholds, scores, best_threshold, metric, title_suffix)
        
        return best_threshold
        
    def analyze_calibration(self, y_true: np.ndarray, y_prob: np.ndarray, 
                          title_suffix: str = "", n_bins: int = 20, threshold: float = 0.5) -> Dict:
        """
        Comprehensive calibration analysis with multiple metrics and visualizations.
        
        Args:
            y_true: True binary labels (0/1)
            y_prob: Predicted probabilities [0,1]
            title_suffix: Suffix for plot titles and filenames
            n_bins: Number of bins for reliability diagram
            
        Returns:
            Dictionary with calibration metrics
        """
        # print(f"\n{'='*60}")
        # print(f"CALIBRATION ANALYSIS{' - ' + title_suffix if title_suffix else ''}")
        # print(f"{'='*60}")
        
        # 1. Expected Calibration Error (ECE)
        ece = self._calculate_ece(y_true, y_prob, n_bins)
        
        # 2. Maximum Calibration Error (MCE)
        mce = self._calculate_mce(y_true, y_prob, n_bins)
        
        # 3. Brier Score
        brier_score = self._calculate_brier_score(y_true, y_prob)
        
        # 4. Confidence distribution analysis
        confidence_stats = self._analyze_confidence_distribution(y_prob)
        
        # 5. Create visualizations
        self._create_calibration_plots(y_true, y_prob, title_suffix, n_bins, threshold)
        
        # Print results
        print(f"📊 Calibration Metrics ({title_suffix}): ECE={ece:.4f}, MCE={mce:.4f}, Brier={brier_score:.4f} (n={len(y_true)})")
        # print(f"   Expected Calibration Error (ECE): {ece:.4f}")
        # print(f"   Maximum Calibration Error (MCE): {mce:.4f}") 
        # print(f"   Brier Score: {brier_score:.4f}")
        # print(f"   Samples: {len(y_true)}")
        
        # print(f"\n📈 CONFIDENCE DISTRIBUTION:")
        # for key, value in confidence_stats.items():
        #    print(f"   {key}: {value:.4f}")
            
        return {
            'ece': ece,
            'mce': mce,
            'brier_score': brier_score,
            'confidence_stats': confidence_stats,
            'n_samples': len(y_true)
        }
    
    def fit_platt_scaling(self, y_true: np.ndarray, y_prob: np.ndarray) -> 'ModelCalibrationAnalyzer':
        """
        Fit Platt Scaling calibrator on validation data.
        
        If temperature scaling is already fitted, this will be applied on top of it.
        
        Args:
            y_true: True binary labels for validation set
            y_prob: Probabilities from validation set (potentially after temperature scaling)
            
        Returns:
            Self for method chaining
        """
        # print(f"\n🔧 FITTING PLATT SCALING CALIBRATOR...")
        
        # Reshape for sklearn
        y_prob_reshaped = y_prob.reshape(-1, 1)
        
        # Fit logistic regression
        self.calibrator = LogisticRegression()
        self.calibrator.fit(y_prob_reshaped, y_true)
        self.calibration_method = "platt"
        
        # Get fitted parameters
        a, b = self.calibrator.coef_[0][0], self.calibrator.intercept_[0]
        
        pipeline_info = "Platt only"
        print(f"   ✅ Platt Scaling fitted successfully! (Pipeline: {pipeline_info})")
        print(f"   📐 Parameters: a={a:.4f}, b={b:.4f}")
        print(f"   📋 Formula: p_cal = sigmoid({a:.4f} * p_input + {b:.4f})")
        
        # Validate calibrator performance
        score = self.calibrator.score(y_prob_reshaped, y_true)
        print(f"   🎯 Calibrator accuracy on validation: {score:.4f}")
        
        return self
    
    def fit_isotonic_regression(self, y_true: np.ndarray, y_prob: np.ndarray) -> 'ModelCalibrationAnalyzer':
        """
        Fit Isotonic Regression calibrator (alternative to Platt Scaling).
        
        More flexible but requires more data. Good for non-sigmoid calibration curves.
        """
        print(f"\n🔧 FITTING ISOTONIC REGRESSION CALIBRATOR...")
        
        self.calibrator = IsotonicRegression(out_of_bounds='clip')
        self.calibrator.fit(y_prob, y_true)
        self.calibration_method = "isotonic"
        
        print(f"   ✅ Isotonic Regression fitted successfully!")
        
        return self
    
    def calibrate_probabilities(self, y_prob: np.ndarray, from_logits: bool = False) -> np.ndarray:
        """
        Apply fitted calibration pipeline to new probabilities.
        
        Pipeline: Platt scaling → final probabilities
        
        Args:
            y_prob: Input probabilities or logits
            from_logits: Whether input is logits (True) or probabilities (False)
            
        Returns:
            Calibrated probabilities
        """
        
        if self.calibrator is None:
            return y_prob  # No Platt scaling fitted
        
        if self.calibration_method == "platt":
            return self.calibrator.predict_proba(y_prob.reshape(-1, 1))[:, 1]
        else:  # isotonic
            return self.calibrator.predict(y_prob)
    
    def _calculate_ece(self, y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
        """Calculate Expected Calibration Error."""
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        ece = 0.0
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (y_prob > bin_lower) & (y_prob <= bin_upper)
            prop_in_bin = in_bin.mean()
            
            if prop_in_bin > 0:
                accuracy_in_bin = y_true[in_bin].mean()
                avg_confidence_in_bin = y_prob[in_bin].mean()
                ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
                
        return ece
    
    def _calculate_mce(self, y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
        """Calculate Maximum Calibration Error."""
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        max_ce = 0.0
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (y_prob > bin_lower) & (y_prob <= bin_upper)
            
            if in_bin.sum() > 0:
                accuracy_in_bin = y_true[in_bin].mean()
                avg_confidence_in_bin = y_prob[in_bin].mean()
                ce = np.abs(avg_confidence_in_bin - accuracy_in_bin)
                max_ce = max(max_ce, ce)
                
        return max_ce
    
    def _calculate_brier_score(self, y_true: np.ndarray, y_prob: np.ndarray) -> float:
        """Calculate Brier Score (lower is better)."""
        return np.mean((y_prob - y_true) ** 2)
    
    def _analyze_confidence_distribution(self, y_prob: np.ndarray) -> Dict:
        """Analyze the distribution of predicted probabilities."""
        return {
            'mean_confidence': np.mean(y_prob),
            'std_confidence': np.std(y_prob),
            'min_confidence': np.min(y_prob),
            'max_confidence': np.max(y_prob),
            'median_confidence': np.median(y_prob),
            'confident_positive_ratio': np.mean(y_prob > 0.7),
            'confident_negative_ratio': np.mean(y_prob < 0.3),
            'uncertain_ratio': np.mean((y_prob >= 0.4) & (y_prob <= 0.6))
        }
    
    def _create_calibration_plots(self, y_true: np.ndarray, y_prob: np.ndarray, 
                                title_suffix: str, n_bins: int, threshold: float = 0.5):
        """Create comprehensive calibration visualization plots."""
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'Model Calibration Analysis{" - " + title_suffix if title_suffix else ""}', 
                    fontsize=16, fontweight='bold')
        
        # 1. Reliability Diagram (Calibration Plot)
        ax1 = axes[0, 0]
        fraction_of_positives, mean_predicted_value = calibration_curve(
            y_true, y_prob, n_bins=n_bins, strategy='uniform'
        )
        
        ax1.plot(mean_predicted_value, fraction_of_positives, "s-", 
                label=f"Model (n_bins={n_bins})", linewidth=2, markersize=8)
        ax1.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated", linewidth=2)
        ax1.set_xlabel('Mean Predicted Probability')
        ax1.set_ylabel('Fraction of Positives')
        ax1.set_title('Reliability Diagram')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Confidence Histogram
        ax2 = axes[0, 1]
        ax2.hist(y_prob, bins=50, alpha=0.7, density=True, color='skyblue', edgecolor='black')
        ax2.set_xlabel('Predicted Probability')
        ax2.set_ylabel('Density')
        ax2.set_title('Confidence Distribution')
        ax2.axvline(np.mean(y_prob), color='red', linestyle='--', linewidth=2, 
                   label=f'Mean: {np.mean(y_prob):.3f}')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Calibration Error per Bin
        ax3 = axes[1, 0]
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        bin_centers = (bin_lowers + bin_uppers) / 2
        
        calibration_errors = []
        bin_counts = []
        
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (y_prob > bin_lower) & (y_prob <= bin_upper)
            bin_count = in_bin.sum()
            
            if bin_count > 0:
                accuracy_in_bin = y_true[in_bin].mean()
                avg_confidence_in_bin = y_prob[in_bin].mean()
                cal_error = avg_confidence_in_bin - accuracy_in_bin
            else:
                cal_error = 0
                
            calibration_errors.append(cal_error)
            bin_counts.append(bin_count)
        
        bars = ax3.bar(bin_centers, calibration_errors, width=0.08, alpha=0.7, 
                      color=['red' if x > 0 else 'blue' for x in calibration_errors])
        ax3.set_xlabel('Confidence Bin')
        ax3.set_ylabel('Calibration Error (Confidence - Accuracy)')
        ax3.set_title('Calibration Error per Bin')
        ax3.axhline(y=0, color='black', linestyle='-', linewidth=1)
        ax3.grid(True, alpha=0.3)
        
        # Add bin count annotations
        for i, (bar, count) in enumerate(zip(bars, bin_counts)):
            height = bar.get_height()
            ax3.annotate(f'{count}', xy=(bar.get_x() + bar.get_width()/2, height),
                        xytext=(0, 3 if height >= 0 else -15), textcoords="offset points",
                        ha='center', va='bottom' if height >= 0 else 'top', fontsize=8)
        
        # 4. Prediction vs Ground Truth Scatter
        ax4 = axes[1, 1]
        jitter_strength = 0.05
        y_true_jittered = y_true + np.random.normal(0, jitter_strength, len(y_true))
        
        scatter = ax4.scatter(y_prob, y_true_jittered, alpha=0.6, s=20, c=y_true, 
                            cmap='RdYlBu_r')
        ax4.set_xlabel('Predicted Probability')
        ax4.set_ylabel('True Label (jittered)')
        ax4.set_title('Predictions vs Ground Truth')
        ax4.grid(True, alpha=0.3)
        
        # Add threshold line
        ax4.axvline(x=threshold, color='black', linestyle='--', linewidth=2, label=f'Decision Threshold ({threshold:.2f})')
        ax4.legend()
        
        plt.tight_layout()
        
        if self.save_plots:
            filename = f"calibration_analysis{'_' + title_suffix if title_suffix else ''}.png"
            plt.savefig(f"{self.plot_dir}/{filename}", dpi=300, bbox_inches='tight')
            print(f"   💾 Calibration plots saved: {self.plot_dir}/{filename}")
        
        plt.show()

    def _plot_threshold_optimization(self, thresholds: np.ndarray, scores: list, 
                                   optimal_threshold: float, metric: str, title_suffix: str = ""):
        """Plot threshold optimization curve."""
        plt.figure(figsize=(10, 6))
        plt.plot(thresholds, scores, 'b-', linewidth=2, label=f'{metric.upper()} Score')
        plt.axvline(x=optimal_threshold, color='red', linestyle='--', linewidth=2, 
                   label=f'Optimal Threshold: {optimal_threshold:.4f}')
        plt.axvline(x=0.5, color='gray', linestyle=':', linewidth=1, 
                   label='Default Threshold: 0.5')
        
        plt.xlabel('Threshold')
        plt.ylabel(f'{metric.upper()} Score')
        plt.title(f'Threshold Optimization - {metric.upper()} Curve')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Annotate optimal point
        max_score = max(scores)
        plt.annotate(f'Best {metric.upper()}: {max_score:.4f}', 
                    xy=(optimal_threshold, max_score),
                    xytext=(optimal_threshold + 0.1, max_score + 0.02),
                    arrowprops=dict(arrowstyle='->', color='red'),
                    fontsize=12, fontweight='bold')
        
        if self.save_plots:
            filename = f"threshold_optimization_{metric}{'_' + title_suffix if title_suffix else ''}.png"
            plt.savefig(f"{self.plot_dir}/{filename}", dpi=300, bbox_inches='tight')
            print(f"   💾 Threshold optimization plot saved: {self.plot_dir}/{filename}")
        
        plt.show()


def analyze_model_calibration_from_predictions(evaluator, graph_data, modes=['val', 'test'],
                                               optimize_threshold=True, threshold_metric="f1",
                                               target_mode="binary_prediction", task_prefix=None,
                                               calibration_method="platt"):
    """
    Complete calibration analysis workflow.

    Pipeline:
    1. Fit calibration (platt or isotonic) on validation set
    2. Threshold optimization - finds optimal decision boundary

    Args:
        evaluator: Trained Evaluator instance
        graph_data: Graph data with train/val/test splits
        modes: List of modes to analyze ['val', 'test']
        optimize_threshold: Whether to optimize decision threshold
        threshold_metric: Metric to optimize threshold for
        target_mode: Target mode for prediction extraction
        task_prefix: Task prefix for multi-label/multi-task modes
        calibration_method: 'platt' or 'isotonic'

    Returns:
        Dictionary with calibration results and fitted calibrator
    """
    analyzer = ModelCalibrationAnalyzer()
    results = {}
    calibrator = None
    
    for mode in modes:
        print(f"\n🔍 Analyzing calibration for {mode.upper()} set...")
        
        # Get predictions using existing evaluator
        if mode == "val":
            eval_mask = graph_data["startup"].val_mask
        else:
            eval_mask = graph_data["startup"].test_mask
        
        # Get predictions based on mode
        if target_mode == "multi_task":
             preds = evaluator._get_predictions(graph_data, eval_mask, "multi_task")
             # Extract binary components for calibration
             y_true = preds["binary_y"]
             y_prob = preds["binary_probs"]
        elif target_mode == "multi_label" and task_prefix:
             preds = evaluator._get_predictions(graph_data, eval_mask, "multi_label")
             y_true = preds[f"{task_prefix}_y"]
             y_prob = preds[f"{task_prefix}_probs"]
        elif target_mode == "masked_multi_task" and task_prefix:
             preds = evaluator._get_predictions(graph_data, eval_mask, "masked_multi_task")
             # _get_predictions returns {mom_y, mom_probs, liq_y, liq_probs, ...}
             y_true = preds[f"{task_prefix}_y"]
             y_prob = preds[f"{task_prefix}_probs"]
        else:
             preds = evaluator._get_predictions(graph_data, eval_mask, "binary_prediction")
             y_true = preds["y"]
             y_prob = preds["probs"]
        
        # Analyze uncalibrated model
        # Construct proper suffix
        current_suffix = mode.upper()
        if task_prefix:
            current_suffix = f"{task_prefix.upper()}_{current_suffix}"
            
        calibration_metrics = analyzer.analyze_calibration(
            y_true, y_prob, title_suffix=current_suffix
        )
        
        results[mode] = {
            'metrics': calibration_metrics,
            'y_true': y_true,
            'y_prob': y_prob
        }
        
        # Fit calibration on validation data
        if mode == 'val':
            # Step 1: Fit calibration model (skip for "none")
            if calibration_method == "none":
                print(f"  [Calibration] method='none' — skipping calibration fitting")
                y_prob_calibrated = y_prob
            elif calibration_method == "isotonic":
                analyzer.fit_isotonic_regression(y_true, y_prob)
                calibrator = analyzer
                y_prob_calibrated = analyzer.calibrate_probabilities(y_prob)
            else:
                analyzer.fit_platt_scaling(y_true, y_prob)
                calibrator = analyzer
                y_prob_calibrated = analyzer.calibrate_probabilities(y_prob)

            # Step 2: Find optimal threshold
            if optimize_threshold:
                label = "CALIBRATED" if calibration_method != "none" else "UNCALIBRATED"
                print(f"\n🎯 FINDING OPTIMAL THRESHOLD ON {label} PROBABILITIES...")
                optimal_threshold = analyzer.find_optimal_threshold(
                    y_true, y_prob_calibrated, metric=threshold_metric, title_suffix=current_suffix
                )
                print(f"   📊 Threshold optimized ({threshold_metric})")
            else:
                optimal_threshold = 0.5

            # Step 3: Analyze with the correct threshold
            suffix = f"{current_suffix}_CALIBRATED" if calibration_method != "none" else f"{current_suffix}_UNCALIBRATED"
            calibration_metrics_cal = analyzer.analyze_calibration(
                y_true, y_prob_calibrated, title_suffix=suffix,
                threshold=optimal_threshold
            )

            results[f'{mode}_calibrated'] = {
                'metrics': calibration_metrics_cal,
                'y_true': y_true,
                'y_prob': y_prob_calibrated,
                'optimal_threshold': optimal_threshold
            }
    
    # Apply calibration to test set if we have it
    if 'test' in results and calibrator is not None:
        print(f"\n🔧 Applying calibration to TEST set...")
        
        test_eval_mask = graph_data["startup"].test_mask
        
        # Apply calibration to probabilities
        test_y_prob_calibrated = calibrator.calibrate_probabilities(
            results['test']['y_prob']
        )
        
        test_y_true = results['test']['y_true']
        
        # Get optimal threshold from calibrator if available
        test_threshold = calibrator.optimal_threshold if hasattr(calibrator, 'optimal_threshold') else 0.5
        
        # Check if threshold is too strict (0 positive predictions)
        test_pred = (test_y_prob_calibrated >= test_threshold).astype(int)
        if np.sum(test_pred) == 0 and hasattr(calibrator, 'optimal_threshold_percentile'):
             print(f"   ⚠️ Absolute threshold {test_threshold:.4f} gives 0% positive predictions")
             test_threshold = np.percentile(test_y_prob_calibrated, calibrator.optimal_threshold_percentile)
             print(f"   🔄 Falling back to percentile threshold: {test_threshold:.4f} ({calibrator.optimal_threshold_percentile:.1f}th percentile)")
        
        if task_prefix:
            test_suffix = f"{task_prefix.upper()}_TEST_CALIBRATED"
        else:
            test_suffix = "TEST_CALIBRATED"

        calibration_metrics_test_cal = analyzer.analyze_calibration(
            test_y_true, test_y_prob_calibrated, title_suffix=test_suffix,
            threshold=test_threshold
        )
        
        results['test_calibrated'] = {
            'metrics': calibration_metrics_test_cal,
            'y_true': test_y_true,
            'y_prob': test_y_prob_calibrated
        }
    
    return results, calibrator


# Integration points for existing code

def get_calibrated_predictions_integration_points():
    """
    Information about where to integrate calibration in the existing codebase.
    
    Returns:
        Dictionary with integration information
    """
    return {
        'eval.py_changes': {
            'file': 'src/ml/eval.py',
            'locations': [
                {
                    'function': '_get_predictions',
                    'line_approx': 286,
                    'change': 'Add calibration option to _process_predictions function'
                },
                {
                    'function': '__init__',
                    'line_approx': 24,
                    'change': 'Add calibrator parameter to Evaluator init'
                }
            ]
        },
        'train.py_changes': {
            'file': 'src/ml/train.py', 
            'change': 'Add calibration fitting after validation in training loop'
        },
        'config.yaml_changes': {
            'file': 'config.yaml',
            'change': 'Add calibration section to configuration'
        }
    }


if __name__ == "__main__":
    print("📊 Model Calibration Analysis Module")
    print("=====================================")
    print()
    print("This module provides:")
    print("• Expected Calibration Error (ECE) calculation")
    print("• Maximum Calibration Error (MCE) calculation") 
    print("• Brier Score calculation")
    print("• Reliability diagrams and calibration plots")
    print("• Platt Scaling calibration improvement")
    print("• Isotonic Regression alternative")
    print()
    print("Usage:")
    print("1. Import: from calibration import analyze_model_calibration_from_predictions")
    print("2. Run: results, calibrator = analyze_model_calibration_from_predictions(evaluator, graph_data)")
    print("3. For new predictions: Use calibrator.calibrate_probabilities(new_probs)")
    print()
    print("Integration points:", get_calibrated_predictions_integration_points())