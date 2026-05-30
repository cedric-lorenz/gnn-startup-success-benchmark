"""
Feature and Graph Structure Visualization for Heterogeneous Graph Neural Networks.

This module provides visualization capabilities for analyzing feature distributions
and graph structural properties (density, degree distribution, homophily) 
to diagnose GNN performance issues.
"""

from pathlib import Path
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from torch_geometric.utils import degree, homophily

# Set style for better-looking plots
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Color scheme for splits
SPLIT_COLORS = {
    'train': '#1f77b4',    # blue
    'val': '#ff7f0e',      # orange
    'test': '#2ca02c',     # green
    'all': '#d62728'       # red
}


def extract_features_by_split(data, node_type='startup'):
    """
    Extract features separated by train/val/test splits.
    
    Args:
        data: HeteroData object
        node_type: Type of node to extract features from
        
    Returns:
        dict: Dictionary with keys 'train', 'val', 'test' containing feature arrays
        list: List of feature names
    """
    print(f"\nExtracting features for node type: {node_type}")
    
    # Prefer pre-normalization features (saved during preprocessing) so
    # visualizations show original value ranges. Fall back to `x` otherwise.
    if hasattr(data[node_type], 'x_pre_norm'):
        features = data[node_type].x_pre_norm.cpu().numpy()
    else:
        features = data[node_type].x.cpu().numpy()

    feature_names = data.feat_labels.get(node_type, [f"feature_{i}" for i in range(features.shape[1])])
    
    print(f"  Total features: {len(feature_names)}")
    print(f"  Feature matrix shape: {features.shape}")
    
    if node_type == 'startup':
        # Extract masks
        train_mask = data['startup'].train_mask.cpu().numpy()
        val_mask = data['startup'].val_mask.cpu().numpy()
        test_mask = data['startup'].test_mask.cpu().numpy()

        # Helper to get features for a split
        def get_split_features(split_name, mask, default_features):
            # Try pre-norm split specific features first
            key_pre = f"x_{split_name}_mask_pre_norm"
            if hasattr(data['startup'], key_pre):
                return getattr(data['startup'], key_pre).cpu().numpy()[mask]
            
            # Try normalized split specific features
            key = f"x_{split_name}_mask"
            if hasattr(data['startup'], key):
                return getattr(data['startup'], key).cpu().numpy()[mask]
                
            # Fallback to slicing the default features
            return default_features[mask]

        # Split features by mask using appropriate source tensors
        splits = {
            'train': features[train_mask], # Train is always in .x (or .x_pre_norm)
            'val': get_split_features('val', val_mask, features),
            'test': get_split_features('test', test_mask, features)
        }
        
        print(f"  Train samples: {splits['train'].shape[0]}")
        print(f"  Val samples:   {splits['val'].shape[0]}")
        print(f"  Test samples:  {splits['test'].shape[0]}")
    else:
        # For other node types, return all features (no split)
        splits = {
            'all': features
        }
        print(f"  All samples: {features.shape[0]}")
    
    return splits, feature_names


def classify_feature_type(values, feature_name):
    """
    Classify feature into binary, integer, or float based on actual data content.
    Since categorical features are already encoded as numbers, we distinguish
    between integer and float data types.
    """
    # Remove NaN values for analysis
    clean_vals = values[~np.isnan(values)]
    
    if len(clean_vals) == 0:
        return 'float'  # Default for empty data
    
    # Check if binary (only 0 and 1, regardless of dtype)
    unique_vals = set(clean_vals)
    if len(unique_vals) <= 2 and unique_vals <= {0, 1, 0.0, 1.0}:
        return 'binary'
    
    # Check if values are actually integers (even if stored as float)
    # This handles the case where integer columns are stored as float64 due to NaN
    try:
        # Check if all non-NaN values are whole numbers
        is_whole = np.allclose(clean_vals, np.round(clean_vals))
        if is_whole:
            return 'integer'
        else:
            return 'float'
    except (ValueError, OverflowError):
        # If conversion fails, treat as float
        return 'float'


def smart_axis_limits(values, feature_type='float'):
    """
    Calculate smart axis limits that focus on the data-dense regions.
    
    Args:
        values: Array of feature values
        feature_type: Type of feature ('float', 'binary', 'integer')
        
    Returns:
        tuple: (min_limit, max_limit) for better axis scaling
    """
    if feature_type == 'binary':
        return None, None  # Use default limits for binary
    
    clean_vals = values[~np.isnan(values)]
    if len(clean_vals) == 0:
        return None, None
    
    # Calculate percentiles to focus on main data distribution
    p5, p95 = np.percentile(clean_vals, [5, 95])
    p25, p75 = np.percentile(clean_vals, [25, 75])
    
    # Calculate IQR for outlier detection
    iqr = p75 - p25
    
    # If the range is very large compared to IQR, use percentile-based limits
    full_range = np.max(clean_vals) - np.min(clean_vals)
    
    if iqr > 0 and full_range / iqr > 20:  # Data is heavily skewed
        # Use extended IQR bounds
        lower_bound = p25 - 1.5 * iqr
        upper_bound = p75 + 1.5 * iqr
        
        # But don't go beyond reasonable percentiles
        lower_bound = max(lower_bound, p5)
        upper_bound = min(upper_bound, p95)
        
        return lower_bound, upper_bound
    else:
        # Use full range with small padding
        padding = 0.05 * full_range
        return np.min(clean_vals) - padding, np.max(clean_vals) + padding


def plot_feature_distribution(feature_idx, feature_name, splits, node_type='startup'):
    """
    Create a distribution plot for a single feature across splits.
    
    Args:
        feature_idx: Index of the feature
        feature_name: Name of the feature
        splits: Dictionary with feature arrays per split
        node_type: Type of node
        
    Returns:
        matplotlib figure object and feature type
    """
    # Determine number of subplots
    n_splits = len(splits)
    fig, axes = plt.subplots(1, n_splits, figsize=(5*n_splits, 4))
    
    # Handle single subplot case
    if n_splits == 1:
        axes = [axes]
    
    all_values = []
    
    # Collect all values across splits to determine feature type and limits
    for split_name, features in splits.items():
        feat_vals = features[:, feature_idx]
        clean_vals = feat_vals[~np.isnan(feat_vals)]
        if len(clean_vals) > 0:
            all_values.extend(clean_vals)
    
    # Determine feature type from all values
    if len(all_values) > 0:
        all_values_array = np.array(all_values)
        feature_type = classify_feature_type(all_values_array, feature_name)
    else:
        feature_type = 'float'
    
    # Calculate smart axis limits for non-binary features
    xlim = smart_axis_limits(np.array(all_values), feature_type) if all_values and feature_type != 'binary' else (None, None)
    
    for ax, (split_name, features) in zip(axes, splits.items()):
        feat_vals = features[:, feature_idx]
        clean_vals = feat_vals[~np.isnan(feat_vals)]
        
        if len(clean_vals) == 0:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f'{split_name.upper()} - No Data')
            continue
        
        color = SPLIT_COLORS.get(split_name, '#888888')
        
        # Plot according to feature type
        if feature_type == 'binary':
            counts = np.bincount(clean_vals.astype(int))
            ax.bar(['0', '1'], counts, color=color, alpha=0.7, edgecolor='black')
            ax.set_ylabel('Count')
            ax.set_title(f'{split_name.upper()} (Binary)\nn=0: {counts[0]}, n=1: {counts[1] if len(counts) > 1 else 0}')
            
        elif feature_type == 'integer':
            # Integer feature with integer bins
            min_val, max_val = int(clean_vals.min()), int(clean_vals.max())
            
            # Use integer bins for reasonable ranges, otherwise use regular binning
            if max_val - min_val <= 50:
                bins = np.arange(min_val, max_val + 2) - 0.5  # Center integers in bins
                ax.hist(clean_vals, bins=bins, color=color, alpha=0.7, edgecolor='black')
                if max_val - min_val <= 20:
                    ax.set_xticks(range(min_val, max_val + 1))
            else:
                n_bins = min(50, max(10, int(np.sqrt(len(clean_vals)))))
                ax.hist(clean_vals, bins=n_bins, color=color, alpha=0.7, edgecolor='black')
            
            ax.set_ylabel('Count')
            ax.set_title(f'{split_name.upper()} (Integer)\nn={len(clean_vals)}')
            
        else:
            # Float feature with improved binning and axis limits
            if xlim[0] is not None and xlim[1] is not None:
                # Filter values to the display range for better bin calculation
                display_vals = clean_vals[(clean_vals >= xlim[0]) & (clean_vals <= xlim[1])]
                if len(display_vals) > 0:
                    clean_vals_for_hist = display_vals
                else:
                    clean_vals_for_hist = clean_vals
            else:
                clean_vals_for_hist = clean_vals
            
            # Calculate number of bins based on data size and range
            n_bins = min(50, max(10, int(np.sqrt(len(clean_vals_for_hist)))))
            
            ax.hist(clean_vals_for_hist, bins=n_bins, color=color, alpha=0.7, edgecolor='black')
            
            # Add mean line and statistics
            mean_val = np.mean(clean_vals)
            median_val = np.median(clean_vals)
            std_val = np.std(clean_vals)
            
            ax.axvline(mean_val, color='darkred', linestyle='--', linewidth=2, label=f'μ={mean_val:.2g}')
            ax.axvline(median_val, color='darkgreen', linestyle=':', linewidth=2, label=f'med={median_val:.2g}')
            
            # Set axis limits for better focus
            if xlim[0] is not None and xlim[1] is not None:
                ax.set_xlim(xlim)
            
            ax.set_ylabel('Count')
            ax.set_title(f'{split_name.upper()} (Float)\nμ={mean_val:.2g}, σ={std_val:.2g}')
            ax.legend(loc='upper left')
            
            # Format x-axis labels for readability
            if xlim[0] is not None and xlim[1] is not None:
                range_size = xlim[1] - xlim[0]
                if range_size > 1000:
                    ax.ticklabel_format(style='scientific', axis='x', scilimits=(0,0))
                elif range_size < 0.01:
                    ax.ticklabel_format(style='scientific', axis='x', scilimits=(0,0))
    
    fig.suptitle(f'{node_type.upper()} - Feature: {feature_name} ({feature_type})', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    return fig, feature_type


def compute_feature_statistics(splits, feature_names):
    """
    Compute comprehensive statistics for all features.
    
    Args:
        splits: Dictionary with feature arrays per split
        feature_names: List of feature names
        
    Returns:
        pandas DataFrame with statistics
    """
    stats_list = []
    
    for feat_idx, feat_name in enumerate(feature_names):
        for split_name, features in splits.items():
            feat_vals = features[:, feat_idx]
            clean_vals = feat_vals[~np.isnan(feat_vals)]
            
            if len(clean_vals) == 0:
                continue
            
            stats_dict = {
                'feature': feat_name,
                'split': split_name,
                'count': len(clean_vals),
                'missing': np.sum(np.isnan(feat_vals)),
                'mean': np.mean(clean_vals),
                'std': np.std(clean_vals),
                'min': np.min(clean_vals),
                '25%': np.percentile(clean_vals, 25),
                'median': np.median(clean_vals),
                '75%': np.percentile(clean_vals, 75),
                'max': np.max(clean_vals),
                'skewness': stats.skew(clean_vals),
                'kurtosis': stats.kurtosis(clean_vals),
            }
            stats_list.append(stats_dict)
    
    return pd.DataFrame(stats_list)


def visualize_all_features(data, node_type='startup', output_dir=None):
    """
    Create and save distribution plots for all features of a node type.
    
    Args:
        data: HeteroData object
        node_type: Type of node to visualize
        output_dir: Output directory path (defaults to outputs/feature_distributions)
        
    Returns:
        pandas DataFrame with statistics
    """
    if output_dir is None:
        # Get project root (assuming this file is in src/ml/)
        project_root = Path(__file__).parent.parent.parent
        output_dir = project_root / "outputs" / "feature_distributions"
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'=' * 70}")
    print(f"VISUALIZING FEATURES FOR {node_type.upper()}")
    print('=' * 70)
    
    # Extract features
    splits, feature_names = extract_features_by_split(data, node_type=node_type)
    
    # Create output subdirectory
    node_output_dir = output_dir / node_type
    node_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Track feature types
    feature_types = {}
    
    # Create plots for each feature
    print(f"\nGenerating {len(feature_names)} distribution plots...")
    for feat_idx, feat_name in enumerate(feature_names):
        fig, feat_type = plot_feature_distribution(feat_idx, feat_name, splits, node_type=node_type)
        feature_types[feat_name] = feat_type
        
        # Save figure
        safe_name = feat_name.replace('/', '_').replace('\\', '_')
        output_file = node_output_dir / f"{feat_idx:03d}_{safe_name}.png"
        fig.savefig(output_file, dpi=100, bbox_inches='tight')
        plt.close(fig)
        
        if (feat_idx + 1) % 10 == 0:
            print(f"  Saved {feat_idx + 1}/{len(feature_names)} plots")
    
    print(f"  Saved {len(feature_names)}/{len(feature_names)} plots")
    
    # Compute and save statistics
    stats_df = compute_feature_statistics(splits, feature_names)
    stats_file = node_output_dir / "statistics.csv"
    stats_df.to_csv(stats_file, index=False)
    print(f"\nSaved statistics to: {stats_file}")
    
    # Print summary
    print(f"\nFeature type breakdown for {node_type}:")
    type_counts = pd.Series(feature_types).value_counts()
    for feat_type, count in type_counts.items():
        print(f"  {feat_type}: {count}")
    
    return stats_df


def visualize_graph_features(graph_data, output_dir=None):
    """
    Main function to visualize features for all node types in a heterogeneous graph.
    
    Args:
        graph_data: HeteroData object containing the graph
        output_dir: Output directory path (optional)
        
    Returns:
        dict: Dictionary mapping node types to their statistics DataFrames
    """
    if output_dir is None:
        # Get project root (assuming this file is in src/ml/)
        project_root = Path(__file__).parent.parent.parent
        output_dir = project_root / "outputs" / "feature_distributions"
    
    output_dir = Path(output_dir)

    # Visualize all node types
    node_types = ['startup', 'investor', 'founder', 'city', 'university', 'sector']
    all_stats = {}
    combined_stats_list = []
    
    for node_type in node_types:
        if node_type in graph_data.node_types:
            stats_df = visualize_all_features(graph_data, node_type=node_type, output_dir=output_dir)
            all_stats[node_type] = stats_df
            combined_stats_list.append(stats_df)
    
    # Combine all statistics
    if combined_stats_list:
        combined_stats = pd.concat(combined_stats_list, ignore_index=True)
        combined_stats_file = output_dir / "all_statistics.csv"
        combined_stats.to_csv(combined_stats_file, index=False)
        
        print("\n" + "=" * 70)
        print("FEATURE DISTRIBUTION ANALYSIS COMPLETE")
        print("=" * 70)
        print(f"\nOutput saved to: {output_dir}")
        print(f"  - Individual feature plots organized by node type")
        print(f"  - Statistics file: {combined_stats_file}")
        print(f"\nVisualization summary:")
        for node_type in node_types:
            if node_type in graph_data.node_types:
                n_features = len(graph_data.feat_labels.get(node_type, []))
                print(f"  {node_type}: {n_features} features")
    
    return all_stats


def plot_nan_distribution(graph_data, node_type='startup', output_dir=None):
    """
    Plot the distribution of NaN counts per class.
    Assumes 'nan_count' is a feature in the graph data.
    
    Args:
        graph_data: HeteroData object
        node_type: Type of node to visualize
        output_dir: Output directory path
    """
    if output_dir is None:
        project_root = Path(__file__).parent.parent.parent
        output_dir = project_root / "outputs" / "nan_distribution"
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nGenerating NaN distribution plot for {node_type}...")
    
    # Check if node type exists
    if node_type not in graph_data.node_types:
        print(f"Node type {node_type} not found in graph.")
        return

    # Get features and labels
    if hasattr(graph_data[node_type], 'x_pre_norm'):
        features = graph_data[node_type].x_pre_norm.cpu().numpy()
    else:
        features = graph_data[node_type].x.cpu().numpy()
        
    # Find nan_count index
    feat_labels = graph_data.feat_labels.get(node_type, [])
    try:
        nan_idx = feat_labels.index("nan_count")
    except ValueError:
        print(f"'nan_count' feature not found for {node_type}. Available features: {len(feat_labels)}")
        return

    nan_counts = features[:, nan_idx]
    
    # Get targets/classes
    targets_to_plot = []
    if hasattr(graph_data[node_type], 'y') and graph_data[node_type].y is not None:
        y = graph_data[node_type].y.cpu().numpy()
        
        # Handle flattened 1D array
        if len(y.shape) == 1:
            targets_to_plot.append(("Class", y))
        # Handle [N, 1]
        elif y.shape[1] == 1:
            targets_to_plot.append(("Class", y.flatten()))
        # Handle Multi-Task columns
        else:
            print(f"Multi-dimensional target {y.shape} detected.")
            # Assume first two columns are main targets (Momentum, Liquidity)
            # If standard 4-col (Mom, Liq, MaskMom, MaskLiq) or just 2-col
            if y.shape[1] >= 2:
                targets_to_plot.append(("Momentum", y[:, 0]))
                targets_to_plot.append(("Liquidity", y[:, 1]))
            else:
                # Fallback loops
                for i in range(y.shape[1]):
                    targets_to_plot.append((f"Target {i}", y[:, i]))
    else:
        print(f"No targets found for {node_type}, plotting overall distribution.")
        targets_to_plot.append(("Overall", np.zeros_like(nan_counts)))
        
    # Create Layout
    n_plots = len(targets_to_plot)
    fig, axes = plt.subplots(1, n_plots, figsize=(8 * n_plots, 6))
    if n_plots == 1: axes = [axes] # Ensure iterable
    
    for i, (name, target) in enumerate(targets_to_plot):
        ax = axes[i]
        
        # Create DataFrame for plotting
        df_plot = pd.DataFrame({
            "nan_count": nan_counts,
            "class": target
        })
        
        # Identify unique classes for reporting
        unique_classes = np.unique(target)
        
        sns.histplot(data=df_plot, x="nan_count", hue="class", 
                     element="step", stat="density", common_norm=False, bins=30, ax=ax,
                     palette="tab10") # Use distinct colors
        
        ax.set_title(f"NaN Count Distribution - {name}")
        ax.set_xlabel("Number of Missing Values")
        ax.set_ylabel("Density")
        
        # Print stats for this target
        print(f"\nNaN Statistics for {name}:")
        for cls in unique_classes:
            cls_data = df_plot[df_plot["class"] == cls]["nan_count"]
            print(f"  Class {cls}: Mean={cls_data.mean():.2f}, Std={cls_data.std():.2f}, Count={len(cls_data)}")

    output_path = output_dir / f"{node_type}_nan_distribution.png"
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"\nNaN distribution plots saved to {output_path}")


# =============================================================================
# NEW: Graph Structure Visualization Functions (Density, Homophily, etc.)
# =============================================================================

def visualize_edge_statistics(graph_data, output_dir=None):
    """
    Calculate and visualize statistics about all edges and metapaths.
    
    Includes:
    1. Average Degree (Source & Destination)
    2. Edge Count vs Node Count
    3. Degree Distribution (Violin Plots)
    4. Sparsity (% Isolated Nodes)
    5. Homophily (Simple Match Ratio)
    6. Edge Homophily Ratio (MSHR) - for metapaths
    7. Class Homophily - for metapaths
    
    Generates:
    - edge_statistics.csv: Raw metrics for all edge types.
    - homophily_spectrum.png: Bar chart of MSHR. Use to diagnose graph structure (Safe/Ambiguous/Heterophilic).
    - signal_vs_noise.png: Scatter of MSHR vs Edge Volume. Use to identify high-volume but noisy edges (potential pitfalls).
    - avg_degree_comparison.png: Bar chart of average degrees. Use to find dominant relation types.
    - edge_count_vs_nodes.png: Scatter of edges vs nodes. Use to check for super-linear scaling (network effects).
    - degree_distribution_violin.png: Violin plots of degrees. Use to detect hub nodes or scale-free properties.
    - sparsity_comparison.png: Bar chart of isolated node %. Use to check data quality/completeness.
    - homophily_comparison.png: Bar chart of simple homophily. Use as a baseline check for label smoothness.
    
    Args:
        graph_data: HeteroData object
        output_dir: Output directory path
    """
    if output_dir is None:
        project_root = Path(__file__).parent.parent.parent
        output_dir = project_root / "outputs" / "graph_statistics"
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'=' * 70}")
    print("VISUALIZING GRAPH EDGE STATISTICS")
    print('=' * 70)
    
    stats_data = []
    
    # Iterate over all edge types
    for edge_type in graph_data.edge_types:
        src_type, rel_type, dst_type = edge_type
        edge_index = graph_data[edge_type].edge_index
        
        num_edges = edge_index.shape[1]
        num_src_nodes = graph_data[src_type].num_nodes
        num_dst_nodes = graph_data[dst_type].num_nodes
        
        # Check if metapath
        is_metapath = 'metapath' in rel_type
        if hasattr(graph_data, 'metapath_definitions'):
            if edge_type in graph_data.metapath_definitions:
                is_metapath = True
        
        # 1. Basic Counts & Density
        avg_degree_src = num_edges / num_src_nodes if num_src_nodes > 0 else 0
        avg_degree_dst = num_edges / num_dst_nodes if num_dst_nodes > 0 else 0
        
        # 2. Sparsity (Isolated Nodes)
        # Count unique nodes involved in edges
        src_active = torch.unique(edge_index[0]).numel()
        dst_active = torch.unique(edge_index[1]).numel()
        
        src_sparsity = 1.0 - (src_active / num_src_nodes) if num_src_nodes > 0 else 0
        dst_sparsity = 1.0 - (dst_active / num_dst_nodes) if num_dst_nodes > 0 else 0
        
        # 3. Homophily (Only if both src and dst have 'y')
        homophily_score = np.nan
        class_homophily_score = np.nan
        
        # Import metrics locally to avoid circular imports if any
        from src.ml.heterophily_metrics import calculate_edge_homophily, calculate_class_homophily
        
        if hasattr(graph_data[src_type], 'y') and hasattr(graph_data[dst_type], 'y'):
            # Only calculate if both are 'startup' or same type with labels
            # Currently our metrics support any same-type edges
            if src_type == dst_type:
                y = graph_data[src_type].y
                
                # Handle multi-task/multi-dimensional targets
                if y is not None and y.ndim > 1 and y.shape[1] > 1:
                    # Use the first target column for homophily statistics
                    # This prevents the "too many indices" error
                    y = y[:, 0]
                    
                homophily_score = calculate_edge_homophily(edge_index, y)
                class_homophily_score = calculate_class_homophily(edge_index, y)

        # Store stats
        stats_data.append({
            'edge_type': f"{src_type}-{rel_type}-{dst_type}",
            'relation': rel_type,
            'is_metapath': is_metapath,
            'num_edges': num_edges,
            'avg_degree_src': avg_degree_src,
            'avg_degree_dst': avg_degree_dst,
            'src_sparsity': src_sparsity,
            'dst_sparsity': dst_sparsity,
            'homophily': homophily_score,
            'class_homophily': class_homophily_score
        })
        
    df_stats = pd.DataFrame(stats_data)
    
    # Save CSV
    csv_path = output_dir / "edge_statistics.csv"
    df_stats.to_csv(csv_path, index=False)
    print(f"Saved edge statistics to {csv_path}")
    
    # --- VISUALIZATIONS ---
    
    # Filter for metapaths (startup-startup) for homophily plots
    df_meta = df_stats[df_stats['is_metapath'] & df_stats['homophily'].notna()].copy()
    
    if not df_meta.empty:
        # Sort by homophily
        df_meta = df_meta.sort_values('homophily', ascending=True)
        
        # 1. Homophily Spectrum Plot
        plt.figure(figsize=(12, max(6, len(df_meta) * 0.4)))
        
        # Color coding
        # > 0.75: Safe (Green)
        # 0.55 - 0.75: Ambiguous (Orange)
        # < 0.55: Heterophilic (Red)
        colors = []
        for h in df_meta['homophily']:
            if h > 0.75: colors.append('#2ca02c') # Green
            elif h > 0.55: colors.append('#ff7f0e') # Orange
            else: colors.append('#d62728') # Red
            
        bars = plt.barh(df_meta['relation'], df_meta['homophily'], color=colors, alpha=0.8)
        
        # Add vertical lines for zones
        plt.axvline(0.55, color='gray', linestyle='--', alpha=0.5)
        plt.axvline(0.75, color='gray', linestyle='--', alpha=0.5)
        plt.text(0.50, -1, 'Heterophilic', ha='center', color='#d62728', fontweight='bold')
        plt.text(0.65, -1, 'Ambiguous', ha='center', color='#ff7f0e', fontweight='bold')
        plt.text(0.85, -1, 'Homophilic', ha='center', color='#2ca02c', fontweight='bold')
        
        plt.title('The Heterophily Spectrum: Metapath Diagnosis', fontsize=14)
        plt.xlabel('MSHR (Edge Homophily Ratio)')
        plt.xlim(0, 1.0)
        
        # Add value labels
        for bar in bars:
            width = bar.get_width()
            plt.text(width + 0.01, bar.get_y() + bar.get_height()/2, f'{width:.2f}', 
                     va='center', fontsize=9)
            
        plt.tight_layout()
        plt.savefig(output_dir / "homophily_spectrum.png")
        plt.close()
        
        # 2. Signal vs Noise (Homophily vs Edge Count)
        plt.figure(figsize=(10, 8))
        
        # Continuous styling
        # High Homophily (1.0) -> Green, Low Homophily (0.0) -> Red
        cmap = plt.cm.RdYlGn 
        
        # Log edge counts for regression
        # Filter for valid data points (non-zero edges, finite homophily)
        valid_mask = (df_meta['num_edges'] > 0) & np.isfinite(df_meta['homophily'])
        valid_df = df_meta[valid_mask]
        
        if len(valid_df) > 1:
            log_counts = np.log10(valid_df['num_edges'])
            try:
                m, c = np.polyfit(log_counts, valid_df['homophily'], 1)
                
                # Plot Trend Line (Slightly visible)
                x_fit = np.linspace(log_counts.min(), log_counts.max(), 100)
                y_fit = m * x_fit + c
                plt.plot(10**x_fit, y_fit, color='gray', linestyle='--', alpha=0.4, label='Trend')
            except np.linalg.LinAlgError:
                print("Could not fit trend line due to LinAlgError")
        else:
            print("Skipping trend line: Not enough valid data points")

        # Dynamic Color Scaling
        h_min = df_meta['homophily'].min()
        h_max = df_meta['homophily'].max()
        
        # Scatter Plot with Gradient
        scatter = plt.scatter(df_meta['num_edges'], df_meta['homophily'], 
                             s=300, c=df_meta['homophily'], cmap=cmap, 
                             vmin=h_min, vmax=h_max, alpha=0.8, edgecolors='black')
        
        plt.xscale('log')
        
        # Add labels
        from adjustText import adjust_text
        texts = []
        for i, row in df_meta.iterrows():
            texts.append(plt.text(row['num_edges'], row['homophily'], row['relation'], fontsize=9))
            
        try:
            adjust_text(texts, arrowprops=dict(arrowstyle='-', color='gray', lw=0.5))
        except ImportError:
            pass
            
        plt.title('Signal vs Noise: Homophily vs Edge Volume', fontsize=14)
        plt.xlabel('Number of Edges (Log Scale)')
        plt.ylabel('MSHR (Edge Homophily)')
        plt.grid(True, which="both", ls="-", alpha=0.2)
        # plt.ylim(-0.05, 1.05) # Removed to allow adaptive Y-axis
        
        # Add Colorbar Legend
        cbar = plt.colorbar(scatter)
        cbar.set_label('Homophily Score (Red=Min, Green=Max)')
        
        # Add simpler Legend for Trend
        plt.legend(loc='upper left')
        
        plt.tight_layout()
        plt.savefig(output_dir / "signal_vs_noise.png")
        plt.close()
        
        print("Saved homophily visualizations.")
    
    # 1. Average Degree Comparison (Bar Chart)
    plt.figure(figsize=(14, 8))
    # Sort by avg degree src
    df_sorted = df_stats.sort_values('avg_degree_src', ascending=True)
    
    colors = ['#d62728' if x else '#1f77b4' for x in df_sorted['is_metapath']]
    
    plt.barh(df_sorted['edge_type'], df_sorted['avg_degree_src'], color=colors)
    plt.xlabel('Average Degree (Source)')
    plt.title('Average Degree by Edge Type (Red = Metapath)')
    plt.xlim(0, 100) # Cap at 100 to visualize smaller degrees better
    plt.tight_layout()
    plt.savefig(output_dir / "avg_degree_source.png")
    plt.close()
    
    # 2. Edge Count vs Sparsity (Scatter)
    plt.figure(figsize=(12, 8)) # Increased size for labels
    
    # Convert sparsity to percentage
    df_stats['src_sparsity_pct'] = df_stats['src_sparsity'] * 100
    
    sns.scatterplot(data=df_stats, x='num_edges', y='src_sparsity_pct', hue='is_metapath', size='avg_degree_src', sizes=(20, 500))
    plt.xscale('log')
    plt.title('Edge Count vs Sparsity (Size = Avg Degree)')
    plt.xlabel('Number of Edges (Log Scale)')
    plt.ylabel('Source Node Sparsity (% Isolated)')
    plt.ylim(-5, 105) # Set limits to 0-100 with padding
    
    # Add labels for metapaths
    if hasattr(graph_data, 'metapath_definitions'):
        print("\nMetapath Legend:")
        mp_dict = graph_data.metapath_definitions
        for idx, row in df_stats.iterrows():
            if row['is_metapath']:
                # Try to find the definition
                # The edge_type in df is a string "src-rel-dst"
                # We need to match it to the tuple key in mp_dict
                for key, val in mp_dict.items():
                    # key is (src, rel, dst)
                    key_str = f"{key[0]}-{key[1]}-{key[2]}"
                    if key_str == row['edge_type']:
                        # val is list of tuples [('startup', 'funded_by', 'investor'), ...]
                        # Show the path: "funded_by -> rev_funded_by"
                        path_str = "->".join([x[1] for x in val])
                        df_stats.at[idx, 'relation'] = f"{row['relation']} ({path_str})"
                        print(f"  {row['relation']}: {path_str}")
                        break
            
    # Now iterate again to plot labels, using potentially updated 'relation'
    for i, row in df_stats.iterrows():
        if row['is_metapath']:
            plt.text(row['num_edges'], row['src_sparsity_pct'], 
                     row['relation'], # Use the potentially updated relation
                     horizontalalignment='left', 
                     size='small', 
                     color='black', 
                     weight='semibold')
                     
    plt.tight_layout()
    plt.savefig(output_dir / "sparsity_vs_count.png")
    plt.close()
    
    # 3. Degree Distribution (Violin Plots for Top 15 Densest Edges)
    # We need to compute degrees per node for this
    top_edges = df_stats.sort_values('avg_degree_src', ascending=False).head(15)['edge_type'].tolist()
    
    degree_data = []
    for edge_type_str in top_edges:
        # Parse string back to tuple
        parts = edge_type_str.split('-')
        # edge_type_str is constructed as f"{src}-{rel}-{dst}"
        row = df_stats[df_stats['edge_type'] == edge_type_str].iloc[0]
        
        # Reconstruct tuple from graph_data.edge_types
        # This is safer than parsing string
        target_edge_type = None
        for et in graph_data.edge_types:
            if f"{et[0]}-{et[1]}-{et[2]}" == edge_type_str:
                target_edge_type = et
                break
        
        if target_edge_type:
            edge_index = graph_data[target_edge_type].edge_index
            # Compute source degrees
            src_degrees = degree(edge_index[0], num_nodes=graph_data[target_edge_type[0]].num_nodes)
            
            # Sample if too large
            deg_np = src_degrees.cpu().numpy()
            if len(deg_np) > 10000:
                deg_np = np.random.choice(deg_np, 10000, replace=False)
                
            for d in deg_np:
                degree_data.append({
                    'edge_type': edge_type_str,
                    'degree': d
                })
    
    if degree_data:
        df_degrees = pd.DataFrame(degree_data)
        
        # Clean data for log scale plotting
        # Remove NaNs and Infs
        df_degrees = df_degrees.replace([np.inf, -np.inf], np.nan).dropna(subset=['degree'])
        
        # For log scale, add epsilon instead of filtering zeros: isolated nodes (degree 0)
        # are important structural info.
        df_degrees['degree_log'] = df_degrees['degree'] + 0.1
        
        plt.figure(figsize=(14, 6))
        try:
            # Filter zeros for the distribution shape (sparsity is reported separately) to avoid skewing.
            df_nonzero = df_degrees[df_degrees['degree'] > 0].copy()
            
            if not df_nonzero.empty:
                sns.violinplot(data=df_nonzero, x='edge_type', y='degree', log_scale=True)
                plt.xticks(rotation=45, ha='right')
                plt.title('Degree Distribution of Connected Nodes (Log Scale)')
                plt.tight_layout()
                plt.savefig(output_dir / "degree_distributions.png")
            else:
                print("Warning: No non-zero degrees found for top edges, skipping violin plot.")
                
        except Exception as e:
            print(f"Warning: Failed to generate violin plot: {e}")
        finally:
            plt.close()

    print(f"Visualizations saved to {output_dir}")
    
    # 4. Homophily Analysis (Bar Chart)
    # Filter edges that have homophily score
    df_homophily = df_stats.dropna(subset=['homophily']).sort_values('homophily', ascending=False)
    
    if not df_homophily.empty:
        plt.figure(figsize=(12, 6))
        colors = ['#d62728' if x else '#1f77b4' for x in df_homophily['is_metapath']]
        sns.barplot(data=df_homophily, x='homophily', y='edge_type', palette=colors)
        plt.title('Homophily Score by Edge Type (Red = Metapath)')
        plt.xlabel('Homophily (Fraction of edges connecting same-class nodes)')
        plt.tight_layout()
        plt.savefig(output_dir / "homophily_scores.png")
        plt.close()
        
    # 5. Redundancy Analysis (Jaccard Similarity Heatmap)
    # Compute Jaccard similarity between edge sets for the SAME source node type
    # Focus on 'startup' as source
    
    source_node_type = 'startup'
    relevant_edges = []
    
    for edge_type in graph_data.edge_types:
        if edge_type[0] == source_node_type:
            relevant_edges.append(edge_type)
            
    if len(relevant_edges) > 1:
        n_rels = len(relevant_edges)
        jaccard_matrix = np.zeros((n_rels, n_rels))
        labels = [f"{et[1]}" for et in relevant_edges]
        
        # Pre-compute adjacency sets for efficiency
        adj_sets = []
        for et in relevant_edges:
            edge_index = graph_data[et].edge_index
            # Jaccard similarity of edge sets: |A intersect B| / |A union B|.
            # Redundancy is checked for metapaths connecting the same (src, dst) pair
            # (different destination types have strictly 0 overlap), so group by (src, dst).
            pass
            
        # Group by (src, dst) pair
        from collections import defaultdict
        grouped_edges = defaultdict(list)
        for et in relevant_edges:
            grouped_edges[(et[0], et[2])].append(et)
            
        for (src, dst), edge_types_group in grouped_edges.items():
            if len(edge_types_group) < 2:
                continue
                
            # Compute Jaccard for this group
            group_labels = [et[1] for et in edge_types_group]
            n_group = len(edge_types_group)
            group_matrix = np.zeros((n_group, n_group))
            
            # Convert to sets of edges (src_idx, dst_idx)
            edge_sets = []
            for et in edge_types_group:
                ei = graph_data[et].edge_index
                # Create set of tuples
                e_set = set(zip(ei[0].tolist(), ei[1].tolist()))
                edge_sets.append(e_set)
                
            for i in range(n_group):
                for j in range(i, n_group):
                    if i == j:
                        sim = 1.0
                    else:
                        set_i = edge_sets[i]
                        set_j = edge_sets[j]
                        intersection = len(set_i.intersection(set_j))
                        union = len(set_i.union(set_j))
                        sim = intersection / union if union > 0 else 0

                    group_matrix[i, j] = sim
                    group_matrix[j, i] = sim
            
            sns.heatmap(group_matrix, annot=True, fmt=".2f", cmap="YlGnBu", 
                        xticklabels=group_labels, yticklabels=group_labels)
            plt.title(f'Jaccard Similarity: {src} -> {dst}')
            plt.tight_layout()
            plt.savefig(output_dir / f"redundancy_{src}_{dst}.png")
            plt.close()


def plot_embeddings(embeddings, title, output_path, labels=None):
    """
    Plot embeddings in 2D using PCA.
    
    Args:
        embeddings: numpy array of shape (n_samples, n_features)
        title: Title of the plot
        output_path: Path to save the plot
        labels: Optional array of class labels for coloring
    """
    from sklearn.decomposition import PCA
    
    # Handle NaNs
    valid_mask = ~np.isnan(embeddings).any(axis=1)
    X = embeddings[valid_mask]
    
    if labels is not None:
        y = np.array(labels)[valid_mask]
    
    if len(X) == 0:
        print(f"No valid embeddings to plot for {title}")
        return

    # Reduce to 2D
    pca = PCA(n_components=2)
    X_2d = pca.fit_transform(X)
    
    explained_var = pca.explained_variance_ratio_.sum()
    
    # Plot 1: Density (Hexbin or Scatter)
    plt.figure(figsize=(10, 8))
    
    if len(X) > 10000:
        plt.hexbin(X_2d[:, 0], X_2d[:, 1], gridsize=50, cmap='viridis', mincnt=1)
        plt.colorbar(label='Count')
    else:
        plt.scatter(X_2d[:, 0], X_2d[:, 1], alpha=0.5, s=5)
        
    plt.title(f"{title}\n2D Projection Explained Variance: {explained_var:.2%}")
    plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.2%})")
    plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.2%})")
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path)
    plt.close()
    print(f"Saved embedding plot to {output_path}")
    
    # Plot 2: Classes (if labels provided)
    if labels is not None:
        plt.figure(figsize=(12, 8))
        
        # Convert labels to strings if they aren't already, to get discrete colors
        # Handle NaN labels
        y_clean = []
        for val in y:
            if pd.isna(val):
                y_clean.append("Unknown")
            else:
                y_clean.append(str(val))
        y_clean = np.array(y_clean)
        
        unique_labels = np.unique(y_clean)
        
        # Use seaborn for easy categorical plotting
        plot_df = pd.DataFrame({
            'PC1': X_2d[:, 0],
            'PC2': X_2d[:, 1],
            'Class': y_clean
        })
        
        # Sample if too large to avoid overplotting/slow rendering
        if len(plot_df) > 20000:
            plot_df = plot_df.sample(20000, random_state=42)
            
        sns.scatterplot(data=plot_df, x='PC1', y='PC2', hue='Class', alpha=0.6, s=15, palette='tab10')
        
        plt.title(f"{title} (By Class)\n2D Projection Explained Variance: {explained_var:.2%}")
        plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.2%})")
        plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.2%})")
        
        # Construct class plot path
        path_obj = Path(output_path)
        class_output_path = path_obj.parent / f"{path_obj.stem}_classes{path_obj.suffix}"
        
        plt.savefig(class_output_path)
        plt.close()
        print(f"Saved class embedding plot to {class_output_path}")