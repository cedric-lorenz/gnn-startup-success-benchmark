"""Visualization utilities for graph structure, embeddings, decision boundaries, and metapath weights."""
import torch
from torch_geometric.loader import NeighborLoader
from pyvis.network import Network
import numpy as np
import random
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd


def visualize_graph(
    graph_data,
    output_file="graph.html",
    visible_node_types=None,
    show_labels=True,
    enable_physics=False,
    max_nodes=1000,
    sample_method="degree_based",  # "random", "degree_based", or "connected_component"
    show_features=False,
    max_features=10,
    use_masks=False,
    included_masks=None,
):
    """
    Create an efficient and correct graph visualization with proper node sampling.
    
    Args:
        graph_data: HeteroData object
        output_file: HTML output file path
        visible_node_types: List of node types to show (None = all)
        show_labels: Whether to show node names
        enable_physics: Enable stable physics simulation with clustering (False = static layout)
        max_nodes: Maximum number of nodes to display
        sample_method: How to sample nodes ("random", "degree_based", "connected_component")
        show_features: Whether to display node features in tooltips
        max_features: Maximum number of features to show per node
    """
    print(f"\n🔍 Creating graph visualization...")
    
    if visible_node_types is None:
        visible_node_types = list(graph_data.node_types)
    
    if included_masks is None:
        included_masks = ["train", "val", "test"]
    
    # Filter data to only include masked nodes if requested
    mask_mapping = None
    if use_masks:
        print(f"Filtering to nodes with masks: {included_masks}")
        graph_data, mask_mapping = filter_graph_by_masks(graph_data, included_masks, visible_node_types)
    
    # Count total nodes and edges for all visible types
    total_nodes = sum(graph_data[nt].x.shape[0] for nt in visible_node_types if nt in graph_data.node_types)
    total_edges = sum(
        graph_data[et].edge_index.shape[1] 
        for et in graph_data.edge_types 
        if et[0] in visible_node_types and et[2] in visible_node_types
    )
    
    print(f"Total nodes: {total_nodes}, Total edges: {total_edges}")
    if show_features:
        print(f"Feature display enabled: showing up to {max_features} features per node")
    
    if total_nodes <= max_nodes:
        print("Using full graph (within node limit)")
        sampled_data = graph_data
        node_mapping = None
    else:
        print(f"Sampling graph to {max_nodes} nodes using {sample_method} method")
        sampled_data, node_mapping = sample_graph(graph_data, max_nodes, sample_method, visible_node_types)
    
    # Create visualization
    net = create_pyvis_network(sampled_data, visible_node_types, show_labels, enable_physics, 
                              node_mapping, show_features, max_features, mask_mapping)
    
    # Save and report
    net.save_graph(output_file)
    
    final_nodes = len(net.nodes)
    final_edges = len(net.edges) 
    feature_status = " (with features)" if show_features else ""
    print(f"✅ Saved visualization: {final_nodes} nodes, {final_edges} edges{feature_status} → {output_file}")


def sample_graph(graph_data, max_nodes, method, visible_node_types):
    """Sample a subgraph while maintaining connectivity and proper indexing."""
    if method == "connected_component":
        return sample_connected_component(graph_data, max_nodes, visible_node_types)
    elif method == "degree_based":
        return sample_by_degree(graph_data, max_nodes, visible_node_types)
    else:  # random
        return sample_random(graph_data, max_nodes, visible_node_types)


def sample_connected_component(graph_data, max_nodes, visible_node_types):
    """
    Sample using NeighborLoader to get a connected subgraph starting from high-degree startup nodes.
    Only selects startups with connections and ensures complete connectivity.
    """
    # Find startup nodes with highest degree (most connections) - only connected ones
    startup_degrees = torch.zeros(graph_data["startup"].x.shape[0])
    
    for edge_type in graph_data.edge_types:
        src_type, rel, dst_type = edge_type
        edge_index = graph_data[edge_type].edge_index
        
        if src_type == "startup":
            startup_degrees.index_add_(0, edge_index[0], torch.ones(edge_index.shape[1]))
        if dst_type == "startup":
            startup_degrees.index_add_(0, edge_index[1], torch.ones(edge_index.shape[1]))
    
    # Only consider connected startups (degree > 0)
    connected_startups = startup_degrees > 0
    if connected_startups.sum() == 0:
        print("Warning: No connected startup nodes found!")
        return graph_data, {}
    
    connected_indices = torch.where(connected_startups)[0]
    connected_degrees = startup_degrees[connected_indices]
    
    # Get top-degree startups as seeds - treat max_nodes as max startup count
    num_seeds = min(max_nodes, len(connected_indices))
    _, top_relative_indices = torch.topk(connected_degrees, num_seeds)
    top_startup_indices = connected_indices[top_relative_indices]
    
    # Calculate neighbor sampling parameters
    estimated_nodes_per_hop = max(10, max_nodes // (num_seeds * 2))  # 2 hops
    num_neighbors = [estimated_nodes_per_hop, estimated_nodes_per_hop]
    
    loader = NeighborLoader(
        data=graph_data,
        input_nodes=("startup", top_startup_indices),
        num_neighbors=num_neighbors,
        batch_size=num_seeds,
        shuffle=False,
    )
    
    sampled_data = next(iter(loader))
    
    # Create node mapping from sampled indices back to original
    node_mapping = {}
    for node_type in sampled_data.node_types:
        if hasattr(sampled_data[node_type], 'n_id'):
            # n_id contains the original node indices for the sampled nodes
            node_mapping[node_type] = sampled_data[node_type].n_id
        else:
            # Fallback: assume identity mapping
            node_mapping[node_type] = torch.arange(sampled_data[node_type].x.shape[0])
    
    total_selected = sum(sampled_data[nt].x.shape[0] for nt in sampled_data.node_types)
    print(f"Sampled connected component with {total_selected} nodes ({len(top_startup_indices)} startup seeds)")
    
    # Copy x_pre_norm features if they exist in original data
    for node_type in sampled_data.node_types:
        if hasattr(graph_data[node_type], 'x_pre_norm'):
            # Map sampled indices back to original indices
            if hasattr(sampled_data[node_type], 'n_id'):
                original_indices = sampled_data[node_type].n_id
            else:
                original_indices = torch.arange(sampled_data[node_type].x.shape[0])
            
            sampled_data[node_type].x_pre_norm = graph_data[node_type].x_pre_norm[original_indices]
    
    return sampled_data, node_mapping


def sample_by_degree(graph_data, max_nodes, visible_node_types):
    """
    Sample nodes based on their degree, prioritizing startups and ensuring complete connectivity.
    Max_nodes is treated as max startup count, all connected nodes are included.
    """
    selected_nodes = {}
    node_mapping = {}
    
    # Step 1: Sample startup nodes with connections only
    startup_degrees = torch.zeros(graph_data["startup"].x.shape[0])
    
    # Count edges for startup nodes
    for edge_type in graph_data.edge_types:
        src_type, rel, dst_type = edge_type
        edge_index = graph_data[edge_type].edge_index
        
        if src_type == "startup":
            startup_degrees.index_add_(0, edge_index[0], torch.ones(edge_index.shape[1]))
        if dst_type == "startup":
            startup_degrees.index_add_(0, edge_index[1], torch.ones(edge_index.shape[1]))
    
    # Only select startups that have connections (degree > 0)
    connected_startups = startup_degrees > 0
    if connected_startups.sum() == 0:
        print("Warning: No connected startup nodes found!")
        return graph_data, {}
    
    # Select top startups by degree from connected ones only
    connected_indices = torch.where(connected_startups)[0]
    connected_degrees = startup_degrees[connected_indices]
    
    max_startup_count = min(max_nodes, len(connected_indices))
    _, top_relative_indices = torch.topk(connected_degrees, max_startup_count)
    selected_startup_indices = connected_indices[top_relative_indices]
    
    selected_nodes["startup"] = selected_startup_indices
    node_mapping["startup"] = selected_startup_indices
    
    print(f"Selected {len(selected_startup_indices)} startups out of {connected_startups.sum()} connected startups")
    
    # Step 2: Include ALL nodes connected to selected startups
    all_connected_nodes = {"startup": set(selected_startup_indices.tolist())}
    
    for edge_type in graph_data.edge_types:
        src_type, rel, dst_type = edge_type
        edge_index = graph_data[edge_type].edge_index
        
        # Find all nodes connected to our selected startups
        if src_type == "startup" and dst_type != "startup":
            # Outgoing edges from selected startups
            startup_mask = torch.isin(edge_index[0], selected_startup_indices)
            connected_dst_nodes = edge_index[1, startup_mask].unique()
            
            if dst_type not in all_connected_nodes:
                all_connected_nodes[dst_type] = set()
            all_connected_nodes[dst_type].update(connected_dst_nodes.tolist())
            
        elif dst_type == "startup" and src_type != "startup":
            # Incoming edges to selected startups
            startup_mask = torch.isin(edge_index[1], selected_startup_indices)
            connected_src_nodes = edge_index[0, startup_mask].unique()
            
            if src_type not in all_connected_nodes:
                all_connected_nodes[src_type] = set()
            all_connected_nodes[src_type].update(connected_src_nodes.tolist())
    
    # Convert sets to tensors and update selected_nodes/node_mapping
    for node_type, node_set in all_connected_nodes.items():
        if node_type in visible_node_types and node_set:
            indices_tensor = torch.tensor(list(node_set), dtype=torch.long)
            selected_nodes[node_type] = indices_tensor
            # node_mapping should be empty here - it gets created in create_subgraph_from_selection
    
    total_selected = sum(len(indices) for indices in selected_nodes.values())
    print(f"Total nodes selected: {total_selected} (including all connections)")
    
    # Create subgraph with selected nodes
    return create_subgraph_from_selection(graph_data, selected_nodes, {})


def sample_random(graph_data, max_nodes, visible_node_types):
    """
    Random sampling prioritizing connected startup nodes.
    Max_nodes is treated as max startup count.
    """
    selected_nodes = {}
    node_mapping = {}
    
    # Step 1: Get connected startup nodes only
    startup_degrees = torch.zeros(graph_data["startup"].x.shape[0])
    
    for edge_type in graph_data.edge_types:
        src_type, rel, dst_type = edge_type
        edge_index = graph_data[edge_type].edge_index
        
        if src_type == "startup":
            startup_degrees.index_add_(0, edge_index[0], torch.ones(edge_index.shape[1]))
        if dst_type == "startup":
            startup_degrees.index_add_(0, edge_index[1], torch.ones(edge_index.shape[1]))
    
    # Only consider connected startups
    connected_startups = startup_degrees > 0
    if connected_startups.sum() == 0:
        print("Warning: No connected startup nodes found!")
        return graph_data, {}
    
    connected_indices = torch.where(connected_startups)[0]
    
    # Random sample from connected startups
    num_startups = min(max_nodes, len(connected_indices))
    random_indices = torch.randperm(len(connected_indices))[:num_startups]
    selected_startup_indices = connected_indices[random_indices]
    
    selected_nodes["startup"] = selected_startup_indices
    node_mapping["startup"] = selected_startup_indices
    
    # Step 2: Include all connected nodes for these startups
    all_connected_nodes = {"startup": set(selected_startup_indices.tolist())}
    
    for edge_type in graph_data.edge_types:
        src_type, rel, dst_type = edge_type
        edge_index = graph_data[edge_type].edge_index
        
        if src_type == "startup" and dst_type != "startup":
            startup_mask = torch.isin(edge_index[0], selected_startup_indices)
            connected_dst_nodes = edge_index[1, startup_mask].unique()
            
            if dst_type not in all_connected_nodes:
                all_connected_nodes[dst_type] = set()
            all_connected_nodes[dst_type].update(connected_dst_nodes.tolist())
            
        elif dst_type == "startup" and src_type != "startup":
            startup_mask = torch.isin(edge_index[1], selected_startup_indices)
            connected_src_nodes = edge_index[0, startup_mask].unique()
            
            if src_type not in all_connected_nodes:
                all_connected_nodes[src_type] = set()
            all_connected_nodes[src_type].update(connected_src_nodes.tolist())
    
    # Convert to tensors
    for node_type, node_set in all_connected_nodes.items():
        if node_type in visible_node_types and node_set:
            indices_tensor = torch.tensor(list(node_set), dtype=torch.long)
            selected_nodes[node_type] = indices_tensor
            # node_mapping will be created in create_subgraph_from_selection
    
    total_selected = sum(len(indices) for indices in selected_nodes.values())
    print(f"Random sample: {total_selected} total nodes ({num_startups} startup seeds)")
    
    # Create subgraph with selected nodes
    return create_subgraph_from_selection(graph_data, selected_nodes, {})


def create_subgraph_from_selection(graph_data, selected_nodes, old_node_mapping):
    """Create a subgraph containing only the selected nodes and their edges."""
    from torch_geometric.data import HeteroData
    
    subgraph = HeteroData()
    
    # Create correct node_mapping: local_index -> original_index
    node_mapping = {}
    
    # Add selected node features
    for node_type, indices in selected_nodes.items():
        subgraph[node_type].x = graph_data[node_type].x[indices]
        if hasattr(graph_data[node_type], 'y'):
            subgraph[node_type].y = graph_data[node_type].y[indices]
        # Copy pre-normalization features if they exist
        if hasattr(graph_data[node_type], 'x_pre_norm'):
            subgraph[node_type].x_pre_norm = graph_data[node_type].x_pre_norm[indices]
        
        # Create mapping: local_index -> original_index
        node_mapping[node_type] = indices
    
    # Create index mappings for edge filtering
    index_maps = {}
    for node_type, indices in selected_nodes.items():
        index_map = torch.full((graph_data[node_type].x.shape[0],), -1, dtype=torch.long)
        index_map[indices] = torch.arange(len(indices))
        index_maps[node_type] = index_map
    
    # Add edges between selected nodes
    for edge_type in graph_data.edge_types:
        src_type, rel, dst_type = edge_type
        
        if src_type not in selected_nodes or dst_type not in selected_nodes:
            continue
            
        edge_index = graph_data[edge_type].edge_index
        
        # Map edge indices to new node indices
        src_mapped = index_maps[src_type][edge_index[0]]
        dst_mapped = index_maps[dst_type][edge_index[1]]
        
        # Keep only edges where both nodes are selected
        valid_mask = (src_mapped >= 0) & (dst_mapped >= 0)
        
        if valid_mask.any():
            new_edge_index = torch.stack([src_mapped[valid_mask], dst_mapped[valid_mask]])
            subgraph[edge_type].edge_index = new_edge_index
            
            # Copy edge attributes if they exist
            if hasattr(graph_data[edge_type], 'edge_attr') and graph_data[edge_type].edge_attr is not None:
                subgraph[edge_type].edge_attr = graph_data[edge_type].edge_attr[valid_mask]
    
    # Copy metadata
    if hasattr(graph_data, 'node_names'):
        subgraph.node_names = graph_data.node_names
    if hasattr(graph_data, 'feat_labels'):
        subgraph.feat_labels = graph_data.feat_labels
    
    return subgraph, node_mapping


def create_pyvis_network(graph_data, visible_node_types, show_labels, enable_physics, 
                         node_mapping=None, show_features=False, max_features=10, mask_mapping=None,
                         node_scores=None, edge_scores=None):
    """Create PyVis network directly from graph data with correct node-edge mapping.
       node_scores: dict[node_type] -> Tensor/Array of scores for sizing (aligned with graph_data[node_type])
       edge_scores: dict[edge_type] -> float (Score for the entire edge type)
    """
    # Calculate global top 5 nodes for labeling
    top_nodes_set = set()
    if node_scores:
        all_scores = []
        for ntype, scores in node_scores.items():
            if scores.numel() > 0:
                # Store (score, ntype, local_idx)
                for idx, s in enumerate(scores):
                    all_scores.append((s.item(), ntype, idx))
        
        # Sort descending by absolute score
        all_scores.sort(key=lambda x: abs(x[0]), reverse=True)
        top_5 = all_scores[:5]
        top_nodes_set = {(t[1], t[2]) for t in top_5} # (ntype, local_idx)
        print("   Top 5 scoring nodes marked for labeling.")

    net = Network(
        notebook=False, 
        height="750px", 
        width="100%", 
        directed=False,
        bgcolor="#ffffff",
        font_color="black"
    )
    
    if not enable_physics:
        net.set_options('''
        {
            "physics": {
                "enabled": false
            },
            "layout": {
                "improvedLayout": true,
                "hierarchical": {
                    "enabled": false
                }
            }
        }
        ''')
    else:
        # Use stable physics with reduced repulsion and damping
        net.set_options('''
        {
            "physics": {
                "enabled": true,
                "stabilization": {
                    "enabled": true,
                    "iterations": 100,
                    "updateInterval": 25
                },
                "barnesHut": {
                    "gravitationalConstant": -8000,
                    "centralGravity": 0.3,
                    "springLength": 95,
                    "springConstant": 0.04,
                    "damping": 0.09,
                    "avoidOverlap": 0.1
                },
                "maxVelocity": 15,
                "minVelocity": 0.1,
                "timestep": 0.35
            },
            "layout": {
                "improvedLayout": true
            }
        }
        ''')
    
    # Color scheme for different node types
    colors = {
        "startup": "#e74c3c",      # red
        "investor": "#3498db",     # blue  
        "founder": "#f39c12",      # orange
        "city": "#2ecc71",         # green
        "university": "#9b59b6",   # purple
        "sector": "#95a5a6",       # gray
    }
    
    # Add nodes with correct indexing
    node_counter = 0
    node_id_map = {}  # Maps (node_type, local_index) to unique node_id
    
    # Debug: Print sampling information
    if node_mapping:
        print("🔍 Node mapping detected - using mapped indices for names")
        for nt, mapping in node_mapping.items():
            print(f"  {nt}: {len(mapping)} sampled nodes")
    else:
        print("🔍 No node mapping - using direct indices")
        
    # Calculate global max score for normalization
    max_score = 0.0
    if node_scores:
        for ntype, scores in node_scores.items():
             if scores.numel() > 0:
                max_score = max(max_score, scores.abs().max().item())
    
    if max_score == 0: max_score = 1.0 # Prevent div by zero

    
    for node_type in visible_node_types:
        if node_type not in graph_data.node_types:
            continue
            
        num_nodes = graph_data[node_type].x.shape[0]
        color = colors.get(node_type, "#95a5a6")
        
        # Get node names and features
        name_df = getattr(graph_data, 'node_names', {}).get(node_type)
        feature_info = get_node_feature_info(graph_data, node_type, max_features) if show_features else None
        
        for local_idx in range(num_nodes):
            # Create unique node ID
            node_id = f"{node_type}_{node_counter}"
            node_counter += 1
            
            # Store mapping: local_idx in sampled graph -> unique node_id
            node_id_map[(node_type, local_idx)] = node_id
            
            # Get node name using BOTH mask_mapping and sampling node_mapping
            if show_labels and name_df is not None:
                node_name = get_node_name(name_df, local_idx, node_mapping, node_type, mask_mapping)
            else:
                # For debugging: show the full mapping chain
                final_original_idx = get_final_original_index(local_idx, node_mapping, node_type, mask_mapping)
                node_name = f"{node_type}_{final_original_idx}"
            
            # Debug: Print mapping for first few nodes
            if local_idx < 2:
                final_original_idx = get_final_original_index(local_idx, node_mapping, node_type, mask_mapping)
                print(f"🔍 Node {node_type}[local:{local_idx}] -> final_original[{final_original_idx}] -> name: {node_name}")
            
            # Create tooltip with features - use local_idx for sampled data
            tooltip = create_node_tooltip(graph_data, node_type, local_idx, node_name, 
                                        node_mapping, show_features, max_features)
            
            # Determine node size
            base_size = 25
            size = base_size
            score_val = 0.0
            
            if node_scores is not None and node_type in node_scores:
                # Get score for this node
                raw_score = node_scores[node_type][local_idx].item()
                abs_score = abs(raw_score)
                score_val = raw_score # for tooltip 
                
                # --- SIZING ---
                # Normalize relative to global max score in this subgraph
                # Range 0.0 - 1.0
                norm_score = abs_score / max_score
                
                # Non-linear boost for size
                # small scores stay small, high scores get big
                # Power 1.5 keeps mid-range visible but emphasizes top
                size = base_size + (norm_score ** 1.5 * 100) 
                size = min(size, 80)
                
                # --- COLOR SATURATION ---
                import matplotlib.colors as mcolors
                
                try:
                    rgb = mcolors.to_rgb(color)
                    hsv = mcolors.rgb_to_hsv(rgb)
                    
                    # hsv[1] is saturation
                    base_sat = hsv[1]
                    
                    # Force low scores to be very gray (desaturated)
                    # Using power 2 to punish low scores
                    target_sat = 1.0 # High scores get full saturation
                    
                    # New Saturation: 
                    # If norm_score is low, saturation -> 0
                    new_sat = target_sat * (norm_score ** 2)
                    
                    # Value/Brightness
                    # Make high scores fully bright/distinct
                    base_val = hsv[2]
                    target_val = 1.0 
                    # Linear interp for brightness
                    new_val = base_val + (target_val - base_val) * (norm_score * 0.5)
                    
                    new_hsv = (hsv[0], new_sat, new_val)
                    new_rgb = mcolors.hsv_to_rgb(new_hsv)
                    color = mcolors.to_hex(new_rgb)
                    
                except Exception as e:
                    pass
            
            # Construct final label
            final_label = node_name if show_labels else ""
            if (node_type, local_idx) in top_nodes_set:
                final_label += f"\nScore: {score_val:.4f}"
                
            # Add to visualization
            net.add_node(
                node_id,
                label=final_label,
                title=f"{tooltip}<br><b>Score:</b> {score_val:.4f}" if node_scores and node_type in node_scores else tooltip,
                color=color,
                size=size
            )
    
    # Add edges with correct node references
    edge_count = 0
    skipped_edges = 0
    
    for edge_type in graph_data.edge_types:
        src_type, rel, dst_type = edge_type
        
        # Skip reverse edges (those starting with 'rev_')
        if rel.startswith('rev_'):
            continue
            
        if (src_type not in visible_node_types or 
            dst_type not in visible_node_types or 
            edge_type not in graph_data.edge_types):
            continue
            
        edge_index = graph_data[edge_type].edge_index
        
        # Validate edge indices are within bounds
        src_max = graph_data[src_type].x.shape[0]
        dst_max = graph_data[dst_type].x.shape[0]
        
        # Determine edge style (Uniform, Thin, No Arrows)
        width = 1.0
        color = "#bdc3c7"
        title = f"Relation: {rel}"
        
        # Ignore edge_scores for styling as per user request
        
        for i in range(edge_index.shape[1]):
            src_local_idx = edge_index[0, i].item()
            dst_local_idx = edge_index[1, i].item()
            
            if src_local_idx >= src_max or dst_local_idx >= dst_max:
                skipped_edges += 1
                continue
                
            src_node_id = node_id_map.get((src_type, src_local_idx))
            dst_node_id = node_id_map.get((dst_type, dst_local_idx))
            
            if src_node_id and dst_node_id:
                try:
                    net.add_edge(
                        src_node_id, 
                        dst_node_id, 
                        label=rel,
                        title=title,
                        color=color,
                        width=width,
                        arrowStrikethrough=False
                    )
                    edge_count += 1
                except AssertionError as e:
                    print(f"⚠️ PyVis error adding edge {src_node_id} -> {dst_node_id}: {e}")
                    skipped_edges += 1
            else:
                print(f"⚠️ Missing node mapping: {src_type}[{src_local_idx}] -> {dst_type}[{dst_local_idx}]")
                skipped_edges += 1
    
    print(f"📊 Added {edge_count} edges, skipped {skipped_edges} edges")
    return net


def get_final_original_index(local_idx, node_mapping, node_type, mask_mapping):
    """Get the final original index by chaining mask_mapping and node_mapping."""
    # Step 1: sampling mapping (if exists): local_idx -> post_mask_idx
    if node_mapping and node_type in node_mapping:
        post_mask_idx = node_mapping[node_type][local_idx].item()
    else:
        post_mask_idx = local_idx
    
    # Step 2: mask mapping (if exists): post_mask_idx -> original_idx
    if mask_mapping and node_type in mask_mapping:
        original_idx = mask_mapping[node_type][post_mask_idx].item()
    else:
        original_idx = post_mask_idx
    
    return original_idx


def get_node_name(name_df, local_idx, node_mapping, node_type, mask_mapping=None):
    """Get the correct node name using both mask mapping and sampling mapping."""
    try:
        # Chain the mappings: local_idx -> post_mask_idx -> original_idx
        original_idx = get_final_original_index(local_idx, node_mapping, node_type, mask_mapping)
        
        # Try different name columns
        if "name" in name_df.columns:
            return str(name_df.iloc[original_idx]["name"])
        
        # Look for columns ending with "_name"
        name_cols = [col for col in name_df.columns if col.endswith("_name")]
        if name_cols:
            return str(name_df.iloc[original_idx][name_cols[0]])
        
        # Special case for city
        if "city" in name_df.columns:
            return str(name_df.iloc[original_idx]["city"])
        
        # Fallback
        return f"{node_type}_{original_idx}"
        
    except (IndexError, KeyError):
        return f"{node_type}_{local_idx}"


def filter_graph_by_masks(graph_data, included_masks, visible_node_types):
    """Filter graph to only include nodes that have masks in included_masks.
    For node types without masks, include all nodes to maintain connectivity.
    
    Returns:
        filtered_data: HeteroData with filtered nodes
        index_mapping: Dict mapping filtered_index -> original_index for each node type
    """
    from torch_geometric.data import HeteroData
    
    filtered_data = HeteroData()
    index_mapping = {}  # Maps filtered index to original index
    
    # For each node type, collect indices of nodes that have the specified masks
    selected_indices = {}
    
    for node_type in visible_node_types:
        if node_type not in graph_data.node_types:
            continue
            
        # Check if this node type has any masks
        has_masks = False
        mask_indices = []
        
        for mask_name in included_masks:
            mask_attr = f"{mask_name}_mask"
            if hasattr(graph_data[node_type], mask_attr):
                mask = getattr(graph_data[node_type], mask_attr)
                if mask is not None:
                    has_masks = True
                    mask_indices.extend(mask.nonzero(as_tuple=True)[0].tolist())
        
        if has_masks and mask_indices:
            # Remove duplicates and convert to tensor
            unique_indices = torch.tensor(sorted(set(mask_indices)), dtype=torch.long)
            selected_indices[node_type] = unique_indices
            index_mapping[node_type] = unique_indices  # Store original indices
            
            # Copy node features for selected indices
            filtered_data[node_type].x = graph_data[node_type].x[unique_indices]
            
            # Copy pre-normalization features if they exist
            if hasattr(graph_data[node_type], 'x_pre_norm'):
                filtered_data[node_type].x_pre_norm = graph_data[node_type].x_pre_norm[unique_indices]
            
            # Copy target labels if they exist
            if hasattr(graph_data[node_type], 'y'):
                filtered_data[node_type].y = graph_data[node_type].y[unique_indices]
            
            # Copy masks
            for mask_name in included_masks:
                mask_attr = f"{mask_name}_mask"
                if hasattr(graph_data[node_type], mask_attr):
                    original_mask = getattr(graph_data[node_type], mask_attr)
                    if original_mask is not None:
                        setattr(filtered_data[node_type], mask_attr, original_mask[unique_indices])
            
            print(f"  {node_type}: {len(mask_indices)} nodes with masks (filtered)")
        elif not has_masks:
            # If no masks exist for this node type, include ALL nodes to maintain connectivity
            num_nodes = graph_data[node_type].x.shape[0]
            all_indices = torch.arange(num_nodes)
            selected_indices[node_type] = all_indices
            index_mapping[node_type] = all_indices  # Identity mapping
            
            # Copy all node features
            filtered_data[node_type].x = graph_data[node_type].x
            
            # Copy pre-normalization features if they exist
            if hasattr(graph_data[node_type], 'x_pre_norm'):
                filtered_data[node_type].x_pre_norm = graph_data[node_type].x_pre_norm
            
            # Copy target labels if they exist
            if hasattr(graph_data[node_type], 'y'):
                filtered_data[node_type].y = graph_data[node_type].y
            
            print(f"  {node_type}: {num_nodes} nodes (no masks - included all)")
        else:
            print(f"  {node_type}: 0 nodes with masks (excluded)")
    
    # Create index mappings for edge filtering
    index_maps = {}
    for node_type, indices in selected_indices.items():
        index_map = torch.full((graph_data[node_type].x.shape[0],), -1, dtype=torch.long)
        index_map[indices] = torch.arange(len(indices))
        index_maps[node_type] = index_map
    
    # Filter edges to only include those between selected nodes
    for edge_type in graph_data.edge_types:
        src_type, rel, dst_type = edge_type
        
        if (src_type not in selected_indices or dst_type not in selected_indices or
            src_type not in visible_node_types or dst_type not in visible_node_types):
            continue
            
        edge_index = graph_data[edge_type].edge_index
        
        # Map edge indices to new node indices
        src_mapped = index_maps[src_type][edge_index[0]]
        dst_mapped = index_maps[dst_type][edge_index[1]]
        
        # Keep only edges where both nodes are selected
        valid_mask = (src_mapped >= 0) & (dst_mapped >= 0)
        
        if valid_mask.any():
            new_edge_index = torch.stack([src_mapped[valid_mask], dst_mapped[valid_mask]])
            filtered_data[edge_type].edge_index = new_edge_index
            
            # Copy edge attributes if they exist
            if hasattr(graph_data[edge_type], 'edge_attr') and graph_data[edge_type].edge_attr is not None:
                filtered_data[edge_type].edge_attr = graph_data[edge_type].edge_attr[valid_mask]
    
    # Copy metadata
    if hasattr(graph_data, 'node_names'):
        filtered_data.node_names = graph_data.node_names
    if hasattr(graph_data, 'feat_labels'):
        filtered_data.feat_labels = graph_data.feat_labels
    
    return filtered_data, index_mapping


def create_node_tooltip(graph_data, node_type, local_idx, node_name, node_mapping, show_features, max_features):
    """Create tooltip text for a node including features if requested.
    
    Important: local_idx refers to the index in the sampled graph_data,
    node_mapping is used only for getting original node names.
    """
    tooltip = f"{node_type}: {node_name}"
    
    if not show_features:
        return tooltip
    
    try:
        # Use local_idx directly for sampled graph features
        # (the graph_data passed here is already the sampled subgraph)
        
        # Get pre-normalization features if available, otherwise use normalized
        if hasattr(graph_data[node_type], 'x_pre_norm'):
            features = graph_data[node_type].x_pre_norm[local_idx]
            feature_type = "Pre-norm"
        else:
            features = graph_data[node_type].x[local_idx]
            feature_type = "Normalized"
        
        # Get feature labels if available
        feat_labels = None
        if hasattr(graph_data, 'feat_labels') and node_type in graph_data.feat_labels:
            feat_labels = graph_data.feat_labels[node_type]
        
        # Add feature information to tooltip
        tooltip += f"\n\n{feature_type} Features:"
        
        num_features = min(max_features, features.shape[0])
        for i in range(num_features):
            value = features[i].item()
            
            # Use feature label if available
            if feat_labels and i < len(feat_labels):
                feat_name = feat_labels[i]
            else:
                feat_name = f"Feature_{i}"
            
            # Format value
            if abs(value) < 1e-3 and value != 0:
                value_str = f"{value:.2e}"
            else:
                value_str = f"{value:.3f}"
            
            tooltip += f"\n{feat_name}: {value_str}"
        
        if features.shape[0] > max_features:
            tooltip += f"\n... and {features.shape[0] - max_features} more features"
        
        # Debug info for index mapping
        if node_mapping and node_type in node_mapping and local_idx < len(node_mapping[node_type]):
            original_idx = node_mapping[node_type][local_idx].item()
            tooltip += f"\n\n🔍 Debug: local_idx={local_idx}, original_idx={original_idx}"
            
    except Exception as e:
        tooltip += f"\n(Feature display error: {str(e)})"
    
    return tooltip


def get_node_feature_info(graph_data, node_type, max_features=10):
    """Extract feature names and values for a node type (pre-normalization when available)."""
    try:
        # Get feature names
        feature_names = getattr(graph_data, 'feat_labels', {}).get(node_type, [])
        
        # Get pre-normalization features if available, otherwise use normalized
        if hasattr(graph_data[node_type], 'x_pre_norm'):
            features = graph_data[node_type].x_pre_norm
        else:
            features = graph_data[node_type].x
        
        return {
            'names': feature_names[:max_features],  # Limit to max_features
            'values': features[:, :max_features] if features.shape[1] > 0 else None
        }
    except Exception:
        return None
        
def visualize_metapath_weights(weights_dict, output_dir="outputs"):
    """
    Visualize learned metapath weights as bar charts.
    
    Args:
        weights_dict: Dictionary of weights per layer and node type
        output_dir: Directory to save plots
    """
    import matplotlib.pyplot as plt
    import os
    import pandas as pd
    import seaborn as sns
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Iterate over layers
    for layer_name, layer_weights in weights_dict.items():
        # Iterate over destination node types (usually just 'startup')
        for node_type, metapath_weights in layer_weights.items():
            if not metapath_weights:
                continue
                
            # Create DataFrame for plotting
            df = pd.DataFrame(list(metapath_weights.items()), columns=['Metapath', 'Weight'])
            df = df.sort_values('Weight', ascending=False)
            
            # Plot
            plt.figure(figsize=(10, 6))
            sns.barplot(x='Weight', y='Metapath', data=df, palette='viridis')
            plt.title(f'Learned Metapath Weights - {layer_name} ({node_type})')
            plt.xlabel('Attention Weight')
            plt.tight_layout()
            
            # Save
            filename = f"metapath_weights_{layer_name}_{node_type}.pdf"
            filepath = os.path.join(output_dir, filename)
            plt.savefig(filepath)
            plt.close()
            
            print(f"📊 Saved metapath weights plot to {filepath}")


def visualize_metapath_weights_vs_stats(weights_dict, stats_dir="outputs/graph_statistics", output_dir="outputs/metapath_analysis"):
    """
    Visualize learned metapath weights against graph statistics (Homophily, Edge Count).
    
    Args:
        weights_dict: Dictionary of weights per layer and node type
        stats_dir: Directory containing edge_statistics.csv
        output_dir: Directory to save plots
    """
    import matplotlib.pyplot as plt
    import os
    import pandas as pd
    import seaborn as sns
    from pathlib import Path
    import numpy as np
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    stats_path = Path(stats_dir) / "edge_statistics.csv"
    if not stats_path.exists():
        print(f"⚠️ Statistics file not found at {stats_path}. Skipping correlation plots.")
        return
        
    # Load statistics
    stats_df = pd.read_csv(stats_path)
    # Don't filter by is_metapath - include all edges as HAN might use direct edges as 1-hop metapaths
    
    # Iterate over layers
    for layer_name, layer_weights in weights_dict.items():
        # Iterate over destination node types
        for node_type, metapath_weights in layer_weights.items():
            if not metapath_weights:
                continue
                
            # Create Weights DataFrame
            weights_df = pd.DataFrame(list(metapath_weights.items()), columns=['relation', 'weight'])
            
            # Merge with statistics
            merged_df = pd.merge(weights_df, stats_df, on='relation', how='inner')
            
            if merged_df.empty:
                print(f"⚠️ No matching metapaths found between weights and statistics for {node_type}.")
                continue
                
            print(f"Generating correlation plots for {layer_name} - {node_type} ({len(merged_df)} metapaths)...")
            
            # Assign IDs for legend
            merged_df['plot_id'] = range(1, len(merged_df) + 1)
            
            # Helper to create legend text
            legend_text = "\n".join([f"{row['plot_id']}: {row['relation']}" for _, row in merged_df.iterrows()])
            
            # Helper function for common plotting logic
            def create_plot(x_col, y_col, title, filename, log_x=False, size_col='num_edges'):
                plt.figure(figsize=(16, 10))
                
                # Main scatter plot
                # Use a grid spec to put legend on the right
                gs = plt.GridSpec(1, 2, width_ratios=[3, 1])
                ax = plt.subplot(gs[0])
                
                # Plot points
                # Let seaborn handle sizing and legend
                if size_col in merged_df.columns:
                    sns.scatterplot(data=merged_df, x=x_col, y=y_col, size=size_col, sizes=(400, 1000), 
                                    alpha=0.7, ax=ax, legend=True, hue='weight', palette='viridis')
                else:
                    sns.scatterplot(data=merged_df, x=x_col, y=y_col, s=500, 
                                    alpha=0.7, ax=ax, legend=True, hue='weight', palette='viridis')
                
                # Add numbered labels
                for _, row in merged_df.iterrows():
                    ax.text(row[x_col], row[y_col], str(row['plot_id']), 
                           horizontalalignment='center', verticalalignment='center',
                           fontweight='bold', color='black', fontsize=8)
                
                if log_x:
                    ax.set_xscale('log')
                    # Explicitly set limits to ensure points aren't cut off
                    x_min = merged_df[x_col].min()
                    x_max = merged_df[x_col].max()
                    if x_min > 0:
                        ax.set_xlim(x_min * 0.5, x_max * 2.0)
                else:
                    # Add some padding for linear scale too
                    x_range = merged_df[x_col].max() - merged_df[x_col].min()
                    if x_range == 0: x_range = 1
                    ax.set_xlim(merged_df[x_col].min() - x_range*0.1, merged_df[x_col].max() + x_range*0.1)

                ax.set_title(f'{title}\n{layer_name} - {node_type}')
                ax.set_xlabel(x_col)
                ax.set_ylabel(y_col)
                ax.grid(True, alpha=0.3)
                
                # Add legend in the second column
                ax_legend = plt.subplot(gs[1])
                ax_legend.axis('off')
                # Split legend text if too long
                ax_legend.text(0, 1, "Metapath Legend:\n\n" + legend_text, 
                              verticalalignment='top', fontsize=9, wrap=True)
                
                plt.tight_layout()
                plt.savefig(output_dir / filename, dpi=300)
                plt.close()

            # 1. Weight vs Homophily (MSHR)
            if 'homophily' in merged_df.columns and merged_df['homophily'].notna().any():
                create_plot('homophily', 'weight', 'Attention Weight vs. Edge Homophily (MSHR)', 
                           f"weight_vs_homophily_{layer_name}_{node_type}.pdf")
                
            # 2. Weight vs Class Homophily
            if 'class_homophily' in merged_df.columns and merged_df['class_homophily'].notna().any():
                create_plot('class_homophily', 'weight', 'Attention Weight vs. Class Homophily', 
                           f"weight_vs_class_homophily_{layer_name}_{node_type}.pdf")

            # 3. Weight vs Edge Count
            create_plot('num_edges', 'weight', 'Attention Weight vs. Edge Count', 
                       f"weight_vs_edge_count_{layer_name}_{node_type}.pdf", log_x=True, size_col=None)
            
    print(f"✅ Saved metapath correlation plots to {output_dir}")

def visualize_decision_boundary(model, graph_data, output_path="decision_boundary.pdf", method='pca', device='cpu', calibrator=None, threshold=0.5):
    """
    Visualize the decision boundary of the model in 2D space.
    
    Args:
        model: The trained GNN model
        graph_data: The graph data object
        output_path: Path to save the plot
        method: Dimensionality reduction method ('pca' or 'tsne')
        device: Device to run the model on
        calibrator: Optional calibrator object to calibrate probabilities
        threshold: Decision threshold for the boundary line
    """
    title_suffix = " (Calibrated)" if calibrator else ""
    if threshold != 0.5:
        title_suffix += f" (Thresh={threshold:.2f})"
        
    print(f"\n🔍 Generating decision boundary visualization using {method.upper()}{title_suffix}...")
    
    model.eval()
    with torch.no_grad():
        # 1. Extract Embeddings
        if hasattr(graph_data, 'edge_index_dict'):    # Get embeddings for all nodes
            out = model(graph_data.x_dict, graph_data.edge_index_dict)
            if "embedding" in out and isinstance(out["embedding"], dict) and "startup" in out["embedding"]:
                # Multi-task structure
                embeddings = out["embedding"]["startup"]
            elif "startup" in out and isinstance(out["startup"], dict) and "embedding" in out["startup"]:
                # Single-task structure
                embeddings = out["startup"]["embedding"]
            else:
                 raise KeyError("Could not find 'startup' embeddings in model output")
            # Get ground truth for startup nodes
            y_true = graph_data['startup'].y
            if isinstance(y_true, (tuple, list)):
                # In multi-task mode, y[1] is the binary target
                y_true = y_true[1]
            # Get mask for test set
            mask = graph_data['startup'].test_mask
        else: # Homogeneous
            embeddings = model.get_embeddings(graph_data.x, graph_data.edge_index)
            y_true = graph_data.y
            mask = graph_data.test_mask
            
        # Filter to test set only
        X = embeddings[mask].cpu().numpy()
        y = y_true[mask].cpu().numpy()
        
        # Get model predictions for coloring the background
        if hasattr(graph_data, 'edge_index_dict'):
            out_dict = model(graph_data.x_dict, graph_data.edge_index_dict)
            if isinstance(out_dict, dict):
                if "startup" in out_dict and isinstance(out_dict["startup"], dict) and "output" in out_dict["startup"]:
                    out = out_dict["startup"]["output"]
                elif "binary_output" in out_dict:
                    out = out_dict["binary_output"]["startup"]
                else:
                    # Fallback
                    out = out_dict["startup"] if "startup" in out_dict else out_dict
            
    
    # 3. Handle Multi-Label Output (Dict) -> Subplots
    if isinstance(out, dict) and "multi_label_output" in out: 
        # Model output from apply_heads: { "embedding": ..., "multi_label_output": tensor, "fund_output": ..., "acq_output": ..., "ipo_output": ...}

        # We need to process 3 tasks: Fund(0), Acq(1), IPO(2)
        tasks = [("Funding", 0), ("Acquisition", 1), ("IPO", 2)]
        
        fig, axes = plt.subplots(1, 3, figsize=(24, 8))
        fig.suptitle(f"Decision Boundaries ({method.upper()}) - Multi-Label Tasks", fontsize=16)
        
        # Extract embeddings for dimensionality reduction (shared)
        if "embedding" in out and "startup" in out["embedding"]:
             embeddings = out["embedding"]["startup"]
        elif "startup" in out and isinstance(out["startup"], dict) and "embedding" in out["startup"]:
             embeddings = out["startup"]["embedding"]
        else:
             embeddings = model.get_embeddings(graph_data.x_dict, graph_data.edge_index_dict)
        
        # Filter to test set
        X = embeddings[mask].cpu().numpy()
        
        # 2. Reduce Dimensionality (Once for all plots to keep geometry consistent)
        print(f"   Reducing dimensionality from {X.shape[1]} to 2...")
        if method.lower() == 'tsne':
            reducer = TSNE(n_components=2, random_state=42)
        else:
            reducer = PCA(n_components=2)
        X_2d = reducer.fit_transform(X)
        
        # Grid for Background
        x_min, x_max = X_2d[:, 0].min() - 1, X_2d[:, 0].max() + 1
        y_min, y_max = X_2d[:, 1].min() - 1, X_2d[:, 1].max() + 1
        grid_resolution = 200
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, grid_resolution),
                             np.linspace(y_min, y_max, grid_resolution))
        
        for i, (task_name, task_idx) in enumerate(tasks):
             ax = axes[i]
             print(f"   Visualizing Task: {task_name}")
             
             # Get predictions/probs for this task
             # We need to extract the specific output head or slice the combined one
             # out["multi_label_output"] is [N, 3] usually
             if "multi_label_output" in out:
                  task_out = out["multi_label_output"] # [N, 3]
                  if isinstance(task_out, dict): task_out = task_out["startup"] # Handle heterogeneous dict wrapper if present
                  task_logits = task_out[mask, task_idx] # Slice [N_test]
             else:
                  # Fallback or error
                  print(f"⚠️ Could not find multi_label_output for {task_name}")
                  continue
             
             probs = torch.sigmoid(task_logits).cpu().detach().numpy().flatten()
             
             # Get Calibrator for this task if available
             task_calibrator = None
             if isinstance(calibrator, dict):
                  # Map task index/name to key
                  key_map = {0: "funding", 1: "acquisition", 2: "ipo"}
                  task_calibrator = calibrator.get(key_map.get(task_idx))
             else:
                  task_calibrator = calibrator # Single or None
             
             if task_calibrator:
                  probs_calib = task_calibrator.calibrate_probabilities(probs)
                  threshold_val = task_calibrator.optimal_threshold if hasattr(task_calibrator, "optimal_threshold") else threshold
                  preds = (probs_calib >= threshold_val).astype(int)
             else:
                  threshold_val = threshold
                  preds = (probs >= threshold_val).astype(int)

             # Ground Truth
             # y[task_idx] from [N, 3]
             y_true_task = graph_data['startup'].y[mask, task_idx].cpu().numpy()
             
             # Train Proxy (KNN)
             clf = KNeighborsClassifier(n_neighbors=15)
             clf.fit(X_2d, preds)
             
             # Predict Grid
             Z_proba = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])
             if Z_proba.shape[1] == 1:
                  Z_proba = np.ones(Z_proba.shape[0]) if clf.classes_[0] == 1 else np.zeros(Z_proba.shape[0])
             else:
                  Z_proba = Z_proba[:, 1]
             
             if task_calibrator:
                  Z_proba = task_calibrator.calibrate_probabilities(Z_proba)
             
             Z = Z_proba.reshape(xx.shape)
             
             # Plot contours
             contour = ax.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.RdBu_r, levels=np.linspace(0, 1, 11))

             # Plot Points
             # Scatter plot of test points colored by ground truth
             scatter = ax.scatter(X_2d[:, 0], X_2d[:, 1], c=y_true_task, cmap=plt.cm.RdBu_r, 
                                  edgecolors='k', s=20, alpha=0.7)
             
             ax.set_title(f"{task_name} (Thresh: {threshold_val:.2f})")
             ax.set_xlabel('Component 1')
             ax.set_ylabel('Component 2')

             # Add Legend
             legend_elements = [
                 plt.Line2D([0], [0], marker='o', color='w', label='Negative',
                            markerfacecolor=plt.cm.RdBu_r(0.0), markersize=8, markeredgecolor='k'),
                 plt.Line2D([0], [0], marker='o', color='w', label='Positive',
                            markerfacecolor=plt.cm.RdBu_r(1.0), markersize=8, markeredgecolor='k')
             ]
             ax.legend(handles=legend_elements, loc='upper right')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300)
        print(f"✅ Saved multi-label decision boundary plot: {output_path}")
        return
        
    # 4. Handle Masked Multi-Task Output (Dict) -> Subplots
    if isinstance(out, dict) and "out_mom" in out and "out_liq" in out: 
        # Tasks: Momentum(0), Liquidity(1)
        tasks = [("Momentum", 0), ("Liquidity", 1)]
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 8))
        fig.suptitle(f"Decision Boundaries ({method.upper()}) - Masked Multi-Task", fontsize=16)
        
        # Extract embeddings for dimensionality reduction (shared)
        if "embedding" in out and isinstance(out["embedding"], dict) and "startup" in out["embedding"]:
             embeddings = out["embedding"]["startup"]
        elif "startup" in out and isinstance(out["startup"], dict) and "embedding" in out["startup"]:
             embeddings = out["startup"]["embedding"] # Fallback
        else:
             # Try to get from model if not in output
             embeddings = model.get_embeddings(graph_data.x_dict, graph_data.edge_index_dict)
        
        # Filter to test set
        X = embeddings[mask].cpu().numpy()
        
        # 2. Reduce Dimensionality (Once for all plots to keep geometry consistent)
        print(f"   Reducing dimensionality from {X.shape[1]} to 2...")
        if method.lower() == 'tsne':
            reducer = TSNE(n_components=2, random_state=42)
        else:
            reducer = PCA(n_components=2)
        X_2d = reducer.fit_transform(X)
        
        # Grid for Background
        x_min, x_max = X_2d[:, 0].min() - 1, X_2d[:, 0].max() + 1
        y_min, y_max = X_2d[:, 1].min() - 1, X_2d[:, 1].max() + 1
        grid_resolution = 200
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, grid_resolution),
                             np.linspace(y_min, y_max, grid_resolution))
        
        for i, (task_name, task_idx) in enumerate(tasks):
             ax = axes[i]
             print(f"   Visualizing Task: {task_name}")
             
             # Get predictions/probs for this task
             if task_name == "Momentum":
                 task_logits = out["out_mom"][mask].view(-1)
                 task_key = "momentum"
             else:
                 task_logits = out["out_liq"][mask].view(-1)
                 task_key = "liquidity"
             
             probs = torch.sigmoid(task_logits).cpu().detach().numpy().flatten()
             
             # Get Calibrator for this task if available
             task_calibrator = None
             if isinstance(calibrator, dict):
                  task_calibrator = calibrator.get(task_key)
             else:
                  task_calibrator = calibrator # Single or None
             
             if task_calibrator:
                  probs_calib = task_calibrator.calibrate_probabilities(probs)
                  threshold_val = task_calibrator.optimal_threshold if hasattr(task_calibrator, "optimal_threshold") else threshold
                  preds = (probs_calib >= threshold_val).astype(int)
             else:
                  threshold_val = threshold
                  preds = (probs >= threshold_val).astype(int)

             # Ground Truth
             # y[task_idx] from [N, 4] for masked_multi_task
             # targets: [mom, liq, mom_mask, liq_mask]
             y_true_task = graph_data['startup'].y[mask, task_idx].cpu().numpy()

             # Train Proxy (KNN)
             clf = KNeighborsClassifier(n_neighbors=15)
             clf.fit(X_2d, preds)
             
             # Predict Grid
             Z_proba = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])
             if Z_proba.shape[1] == 1:
                  Z_proba = np.ones(Z_proba.shape[0]) if clf.classes_[0] == 1 else np.zeros(Z_proba.shape[0])
             else:
                  Z_proba = Z_proba[:, 1]
             
             if task_calibrator:
                  Z_proba = task_calibrator.calibrate_probabilities(Z_proba)
             
             Z = Z_proba.reshape(xx.shape)
             
             # Plot contours
             contour = ax.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.RdBu_r, levels=np.linspace(0, 1, 11))
             
             # Plot Points
             # Scatter plot of test points colored by ground truth
             scatter = ax.scatter(X_2d[:, 0], X_2d[:, 1], c=y_true_task, cmap=plt.cm.RdBu_r, 
                                  edgecolors='k', s=20, alpha=0.7)
             
             ax.set_title(f"{task_name} (Thresh: {threshold_val:.2f})")
             ax.set_xlabel('Component 1')
             ax.set_ylabel('Component 2')

             # Add Legend
             legend_elements = [
                 plt.Line2D([0], [0], marker='o', color='w', label='Negative',
                            markerfacecolor=plt.cm.RdBu_r(0.0), markersize=8, markeredgecolor='k'),
                 plt.Line2D([0], [0], marker='o', color='w', label='Positive',
                            markerfacecolor=plt.cm.RdBu_r(1.0), markersize=8, markeredgecolor='k')
             ]
             ax.legend(handles=legend_elements, loc='upper right')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300)
        print(f"✅ Saved masked multi-task decision boundary plot: {output_path}")
        return
    # Get predictions based on output shape
    if hasattr(out, 'shape') and len(out.shape) > 1 and out.shape[1] > 1:
        # Multi-class case
        probs = torch.softmax(out[mask], dim=1).cpu().numpy()
        preds = np.argmax(probs, axis=1)
    else:
        # Binary case
        probs = torch.sigmoid(out[mask]).cpu().numpy().flatten()
        
        # Apply calibration if provided so preds match the threshold
        if calibrator:
            probs = calibrator.calibrate_probabilities(probs)
            
        preds = (probs >= threshold).astype(int)
        
    # 2. Reduce Dimensionality
    print(f"   Reducing dimensionality from {X.shape[1]} to 2...")
    if method.lower() == 'tsne':
        reducer = TSNE(n_components=2, random_state=42)
    else:
        reducer = PCA(n_components=2)
        
    X_2d = reducer.fit_transform(X)
    
    # 3. Train Proxy Classifier (KNN) to approximate decision boundary
    print("   Training proxy classifier for decision surface...")
    clf = KNeighborsClassifier(n_neighbors=15)
    clf.fit(X_2d, preds)
    
    # Check if classifier learned both classes
    unique_classes = clf.classes_
    print(f"   Proxy classifier learned classes: {unique_classes}")
    
    # 4. Generate Meshgrid
    x_min, x_max = X_2d[:, 0].min() - 1, X_2d[:, 0].max() + 1
    y_min, y_max = X_2d[:, 1].min() - 1, X_2d[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                         np.arange(y_min, y_max, 0.1))
    
    # Predict probabilities for meshgrid to show gradients
    if len(unique_classes) > 2:
        # Multi-class: plot the predicted class
        Z = clf.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
        print(f"   Multi-class mode: plotting decision regions for {len(unique_classes)} classes")
    else:
        # Binary case: plot probability of positive class
        proba = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])
        if proba.shape[1] == 1:
            # Only one class - create uniform probability
            print("   ⚠️ Only one class in predictions, using uniform probability")
            if unique_classes[0] == 1:
                Z_proba = np.ones(proba.shape[0])  # All positive
            else:
                Z_proba = np.zeros(proba.shape[0])  # All negative
        else:
            Z_proba = proba[:, 1]  # Probability of positive class
        
        # Apply calibration if provided
        if calibrator:
            print("   Applying calibration to decision boundary...")
            Z_proba = calibrator.calibrate_probabilities(Z_proba)
            
        Z = Z_proba.reshape(xx.shape)
    
    # 5. Plot
    plt.figure(figsize=(12, 10))
    
    # Check if we have both classes
    unique_preds = np.unique(preds)
    print(f"   Unique predicted classes: {unique_preds}")
    
    if len(unique_preds) < 2:
        print("⚠️  Warning: Model predicts only one class. Decision boundary will not be visible.")
        plt.text(0.5, 0.5, f"Model predicts only class {unique_preds[0]}\\nNo decision boundary", 
                 horizontalalignment='center', verticalalignment='center', transform=plt.gca().transAxes,
                 fontsize=14, bbox=dict(facecolor='white', alpha=0.8))
    
    # Plot decision boundary (background) with gradients
    if len(unique_classes) > 2:
        # Multi-class: use discrete colormap
        cmap = plt.cm.get_cmap('tab10', len(unique_classes))
        contour = plt.contourf(xx, yy, Z, alpha=0.4, cmap=cmap)
        plt.colorbar(contour, ticks=range(len(unique_classes)), label='Predicted Class')
    else:
        # Binary: use continuous colormap
        z_min, z_max = Z.min(), Z.max()
        print(f"   Probability range for plot: [{z_min:.4f}, {z_max:.4f}]")
        
        # Ensure we have some range to avoid division by zero or single level
        if z_max - z_min < 1e-4:
            z_max += 1e-4
            z_min -= 1e-4
            
        levels = np.linspace(z_min, z_max, 21)
        
        # Use a sequential colormap if range is small and far from 0.5, otherwise diverging
        cmap = plt.cm.RdBu_r
            
        contour = plt.contourf(xx, yy, Z, alpha=0.4, cmap=cmap, levels=levels)
        plt.colorbar(contour, label='Predicted Probability (Proxy)')
        
        # Add explicit boundary line at threshold
        if len(unique_preds) > 1:
            plt.contour(xx, yy, Z, levels=[threshold], colors='k', linestyles='--', linewidths=2)
    
    if len(unique_classes) <= 2:
        # Binary Classification Logic
        tp_mask = (y == 1) & (preds == 1)
        tn_mask = (y == 0) & (preds == 0)
        fp_mask = (y == 0) & (preds == 1)
        fn_mask = (y == 1) & (preds == 0)
        
        # Plot each category
        # TP: Green (Correct Positive)
        plt.scatter(X_2d[tp_mask, 0], X_2d[tp_mask, 1], c='green', label=f'TP (Correct Pos): {tp_mask.sum()}', 
                    alpha=0.6, edgecolors='white', s=40)
        
        # TN: Blue (Correct Negative)
        plt.scatter(X_2d[tn_mask, 0], X_2d[tn_mask, 1], c='blue', label=f'TN (Correct Neg): {tn_mask.sum()}', 
                    alpha=0.6, edgecolors='white', s=40)
        
        # FP: Orange (False Positive - Type I Error)
        plt.scatter(X_2d[fp_mask, 0], X_2d[fp_mask, 1], c='orange', label=f'FP (False Pos): {fp_mask.sum()}', 
                    alpha=0.8, edgecolors='black', s=50, marker='^')
        
        # FN: Red (False Negative - Type II Error)
        plt.scatter(X_2d[fn_mask, 0], X_2d[fn_mask, 1], c='red', label=f'FN (False Neg): {fn_mask.sum()}', 
                    alpha=0.8, edgecolors='black', s=50, marker='v')
                    
    else:
        # Multi-class fallback
        unique_y = np.unique(y)
        colors = ['red', 'blue', 'green', 'orange', 'purple', 'cyan', 'magenta', 'yellow']
        
        for i, cls in enumerate(unique_y):
            color = colors[i % len(colors)]
            label = f'Actual Class {cls}'
            plt.scatter(X_2d[y == cls, 0], X_2d[y == cls, 1], c=color, label=label, 
                        alpha=0.6, edgecolors='white', s=40)
    
    plt.title(f"Decision Boundary Visualization ({method.upper()}){title_suffix}\\nTest Set Embeddings (Projected to 2D)", fontsize=15)
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.legend(loc='upper right')
    plt.grid(True, alpha=0.3)
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved decision boundary plot: {output_path}")

def visualize_embedding_neighborhood(
    embeddings: np.ndarray,
    node_indices: list, # Local indices in the embeddings array
    center_node_idx: int, # Local index of center node
    metadata: dict, # {local_idx: {'name': str, 'sector': str, 'status': str}}
    output_path: str,
    perplexity: int = 30,
    n_iter: int = 1000,
    title: str = "Embedding Neighborhood"
):
    """
    Visualize a subset of embeddings using t-SNE, highlighting a center node.
    """
    print(f"   Visualizing embedding neighborhood ({len(embeddings)} nodes)...")
    
    # Run t-SNE
    from sklearn.manifold import TSNE
    tsne = TSNE(n_components=2, perplexity=min(perplexity, len(embeddings)-1), n_iter=n_iter, random_state=42)
    embeddings_2d = tsne.fit_transform(embeddings)
    
    # Plot
    plt.figure(figsize=(12, 10))
    import seaborn as sns
    
    # Prepare DataFrame for plotting
    plot_data = []
    for local_idx in range(len(embeddings)):
        x, y = embeddings_2d[local_idx]
        is_center = (local_idx == center_node_idx)
        
        meta = metadata.get(local_idx, {})
        sector = meta.get('sector', 'Unknown')
        status = meta.get('status', 'Unknown')
        name = meta.get('name', f'Node {local_idx}')
        
        # Simplify status
        if str(status) == '1': status = 'Success'
        elif str(status) == '0': status = 'Fail/Operating'
        
        plot_data.append({
            'x': x, 'y': y, 
            'sector': sector, 
            'status': status, 
            'name': name,
            'is_center': is_center,
            'size': 200 if is_center else 50,
            'style': 'Center' if is_center else 'Neighbor'
        })
        
    df_plot = pd.DataFrame(plot_data)
    
    # Plot scatter
    # Use 'sector' for color
    sns.scatterplot(
        data=df_plot, 
        x='x', y='y', 
        hue='sector', 
        style='style',
        size='size',
        sizes=(50, 300),
        palette='tab10',
        alpha=0.8,
        legend=False # Disable legend to save space
    )
    
    # Annotate center node

    center_point = df_plot[df_plot['is_center']].iloc[0]
    plt.annotate(
        center_point['name'], 
        (center_point['x'], center_point['y']),
        xytext=(5, 5), textcoords='offset points',
        fontweight='bold', fontsize=12,
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", alpha=0.8)
    )
    
    # Annotate a few neighbors
    neighbors = df_plot[~df_plot['is_center']].sample(min(5, len(df_plot)-1))
    for _, row in neighbors.iterrows():
        plt.annotate(
            row['name'], 
            (row['x'], row['y']),
            xytext=(3, 3), textcoords='offset points',
            fontsize=8, alpha=0.7
        )
        
    plt.title(title, fontsize=16)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    print(f"   Saved embedding visualization to {output_path}")
    plt.close()
