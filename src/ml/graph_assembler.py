"""Heterogeneous graph construction, metapath materialization, train/val/test splitting, and graph imputation."""
import os
from torch_geometric.data import HeteroData
import torch_geometric.transforms as T
from torch_geometric.utils import to_undirected
import networkx as nx
import community.community_louvain as community_louvain  # python-louvain
from .imputation import GraphImputer
import torch
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from .utils import load_config, success_ratio
from torch_geometric.transforms import RandomNodeSplit
from sklearn.utils import resample
from scipy import sparse
import time as _time

config = load_config()

# Intermediate node types to project through for Louvain clustering.
# For each type, list the edge types connecting startups to that intermediate,
# and the max degree (intermediates with more connections are excluded as mega-hubs).
_LOUVAIN_PROJECTION_REGISTRY = {
    "investor": {
        "edge_types": [
            ("startup", "early_stage_funded_by", "investor"),
            ("startup", "mid_stage_funded_by", "investor"),
            ("startup", "late_stage_funded_by", "investor"),
            ("startup", "other_funded_by", "investor"),
        ],
        "max_degree": 500,
    },
    "city": {
        "edge_types": [
            ("startup", "based_in", "city"),
        ],
        "max_degree": 500,
    },
    "founder": {
        "edge_types": [
            ("startup", "founded_by", "founder"),
            ("startup", "rev_director_at", "founder"),
            ("startup", "rev_on_board_of", "founder"),
        ],
        "max_degree": 500,
    },
    "sector": {
        "edge_types": [
            ("startup", "in_sector", "sector"),
        ],
        "max_degree": 500,
    },
}


def add_endogenous_clusters(data, feature_names, config=None):
    """
    Assign Louvain cluster IDs to startups via metapath-projected graph.

    Projects the heterogeneous graph onto a weighted startup-startup graph
    via shared intermediate nodes (investors, cities, founders). Each shared
    intermediate node k contributes 1/d_k to the edge weight, where d_k is
    the number of startups connected to k. This ensures niche intermediates
    (specialist investors, small cities) receive more weight than mega-hubs.

    Runs Louvain community detection on the resulting projection.
    """
    if config is None:
        config = load_config()
    t_start = _time.time()
    resolution = config["data_processing"].get("louvain_resolution", 0.5)
    projection_types = config["data_processing"].get(
        "louvain_projection_types", ["investor", "city", "founder"]
    )
    print(f"Computing Endogenous Clusters via Louvain "
          f"(projection={projection_types}, res={resolution})...")

    num_startups = data["startup"].num_nodes
    all_rows, all_cols, all_weights = [], [], []

    for ptype in projection_types:
        pconfig = _LOUVAIN_PROJECTION_REGISTRY[ptype]
        edge_types = pconfig["edge_types"]
        max_deg = pconfig["max_degree"]

        # Collect all (startup, intermediate) pairs across edge types
        pairs_list = []
        for rel in edge_types:
            if rel not in data.edge_types:
                continue
            ei = data[rel].edge_index.cpu().numpy()
            pairs_list.append(np.stack([ei[0], ei[1]], axis=1))

        if not pairs_list:
            print(f"  {ptype}: no edges found, skipping")
            continue

        all_pairs = np.unique(np.concatenate(pairs_list, axis=0), axis=0)

        # Group startups by intermediate node (sort by intermediate ID)
        order = np.argsort(all_pairs[:, 1])
        sorted_startups = all_pairs[order, 0]
        sorted_intermediates = all_pairs[order, 1]

        change_idx = np.where(np.diff(sorted_intermediates) != 0)[0] + 1
        groups = np.split(sorted_startups, change_idx)

        n_used, n_skipped, n_edges = 0, 0, 0
        for group in groups:
            d = len(group)
            if d < 2:
                continue
            if d > max_deg:
                n_skipped += 1
                continue
            n_used += 1
            w = 1.0 / d
            idx_i, idx_j = np.triu_indices(d, k=1)
            all_rows.append(group[idx_i])
            all_cols.append(group[idx_j])
            all_weights.append(np.full(len(idx_i), w, dtype=np.float32))
            n_edges += len(idx_i)

        print(f"  {ptype}: {n_used:,} intermediates, {n_edges:,} projected edges"
              + (f" ({n_skipped} skipped, d>{max_deg})" if n_skipped else ""))

    if not all_rows:
        print("  No projected edges found, skipping Louvain")
        return data, feature_names

    # Assemble sparse matrix, symmetrize, deduplicate
    rows = np.concatenate(all_rows)
    cols = np.concatenate(all_cols)
    weights = np.concatenate(all_weights)

    W = sparse.coo_matrix(
        (weights, (rows, cols)), shape=(num_startups, num_startups)
    ).tocsr()
    W = W + W.T
    W = sparse.triu(W, k=1, format="coo")
    print(f"  Projected graph: {W.nnz:,} unique edges")

    # Build NetworkX graph for Louvain
    G = nx.Graph()
    G.add_nodes_from(range(num_startups))
    G.add_weighted_edges_from(
        zip(W.row.tolist(), W.col.tolist(), W.data.tolist())
    )

    connected = sum(1 for n in G.nodes() if G.degree(n) > 0)
    print(f"  Connected: {connected:,}/{num_startups:,} ({connected/num_startups*100:.1f}%)")

    # Run Louvain
    t0 = _time.time()
    try:
        partition = community_louvain.best_partition(
            G, weight="weight", resolution=resolution, random_state=42
        )
    except Exception as e:
        print(f"  Louvain clustering failed: {e}")
        return data, feature_names
    print(f"  Louvain completed in {_time.time()-t0:.1f}s")

    n_clusters = len(set(partition.values()))

    # Convert to tensor feature (raw cluster ID; normalized later by pipeline)
    cluster_ids = torch.zeros((num_startups, 1), dtype=torch.float32)
    for node_idx, comm_id in partition.items():
        cluster_ids[node_idx] = comm_id

    # Append to existing startup features
    if hasattr(data["startup"], "x"):
        data["startup"].x = torch.cat([data["startup"].x, cluster_ids], dim=1)
    else:
        data["startup"].x = cluster_ids

    if "startup" in feature_names:
        feature_names["startup"].append("endogenous_cluster_id")

    print(f"  {n_clusters:,} clusters identified ({_time.time()-t_start:.1f}s total)")
    return data, feature_names


def define_hetero_data(params):
    data = HeteroData()

    # Nodes
    if params["startup_node_features"] is not None:
        data["startup"].x = params["startup_node_features"]
    if params["investor_node_features"] is not None:
        data["investor"].x = params["investor_node_features"]
    if params["founder_node_features"] is not None:
        data["founder"].x = params["founder_node_features"]
    if params["city_node_features"] is not None:
        data["city"].x = params["city_node_features"]
    if params["university_node_features"] is not None:
        data["university"].x = params["university_node_features"]
    if params["sector_node_features"] is not None:
        data["sector"].x = params["sector_node_features"]

    # Edges
    # Helper to safely add edge index
    def add_edge_if_present(src, rel, dst, key):
        val = params.get(key)
        if val is not None and val.numel() > 0: # Ensure not empty/None
            # Skip edges whose src/dst node type was dropped (params is None for those).
            if f"{src}_node_features" in params and params[f"{src}_node_features"] is None: return
            if f"{dst}_node_features" in params and params[f"{dst}_node_features"] is None: return
            
            data[src, rel, dst].edge_index = val

    # Split Funded By Edges
    add_edge_if_present("startup", "early_stage_funded_by", "investor", "startup_early_funded_edge_index")
    add_edge_if_present("startup", "mid_stage_funded_by", "investor", "startup_mid_funded_edge_index")
    add_edge_if_present("startup", "late_stage_funded_by", "investor", "startup_late_funded_edge_index")
    add_edge_if_present("startup", "other_funded_by", "investor", "startup_other_funded_edge_index")

    add_edge_if_present("startup", "based_in", "city", "startup_city_edge_index")
    add_edge_if_present("startup", "founded_by", "founder", "startup_founder_edge_index")
    add_edge_if_present("startup", "in_sector", "sector", "startup_sector_edge_index")
    add_edge_if_present("founder", "studied_at", "university", "founder_university_edge_index")
    add_edge_if_present("investor", "based_in", "city", "investor_city_edge_index")
    add_edge_if_present("university", "based_in", "city", "university_city_edge_index")
    
    # New Professional Edges
    add_edge_if_present("founder", "worked_at", "investor", "founder_investor_employment_edge_index")
        
    if params.get("founder_coworking_edge_index") is not None:
        # Undirected handling needs care if founder is dropped
        if params["founder_node_features"] is not None:
            data["founder", "worked_with", "founder"].edge_index = to_undirected(params["founder_coworking_edge_index"])
    
    # Identity Edges
    if params.get("founder_investor_identity_edge_index") is not None and params["founder_node_features"] is not None and params["investor_node_features"] is not None:
        # Founder -> Investor
        data["founder", "same_as", "investor"].edge_index = params["founder_investor_identity_edge_index"]
        # Investor -> Founder (Reverse)
        rev_edge_index = torch.stack([
            params["founder_investor_identity_edge_index"][1], 
            params["founder_investor_identity_edge_index"][0]
        ], dim=0)
        data["investor", "same_as", "founder"].edge_index = rev_edge_index

    if params.get("founder_co_study_edge_index") is not None:
         if params["founder_node_features"] is not None:
            data["founder", "studied_with", "founder"].edge_index = to_undirected(params["founder_co_study_edge_index"])

    add_edge_if_present("founder", "on_board_of", "startup", "founder_board_edge_index")
    add_edge_if_present("founder", "director_at", "startup", "founder_startup_director_edge_index")
    add_edge_if_present("founder", "director_at", "investor", "founder_investor_director_edge_index")

    if params.get("use_edge_attributes", False):
        # Only add attributes if the edge type exists
        if ("startup", "based_in", "city") in data.edge_types:
             data["startup", "based_in", "city"].edge_attr = params["startup_city_edge_attributes"]
        if ("startup", "founded_by", "founder") in data.edge_types:
             data["startup", "founded_by", "founder"].edge_attr = params["startup_founder_edge_attributes"]
        if ("startup", "in_sector", "sector") in data.edge_types:
             data["startup", "in_sector", "sector"].edge_attr = params["startup_sector_edge_attributes"]
        if ("founder", "studied_at", "university") in data.edge_types:
             data["founder", "studied_at", "university"].edge_attr = params["founder_university_edge_attributes"]
        if ("investor", "based_in", "city") in data.edge_types:
             data["investor", "based_in", "city"].edge_attr = params["investor_city_edge_attributes"]
        if ("university", "based_in", "city") in data.edge_types:
             data["university", "based_in", "city"].edge_attr = params["university_city_edge_attributes"]

    # Target Variable
    data["startup"].y = params["target"]

    # Status Changed
    data["startup"].status_changed = params["status_changed"]

    # Column Names
    data.feat_labels = params["feature_names"]

    # Node Names
    data.node_names = params["node_names"]

    return data


def to_csr(edge_index, num_nodes):
    """Converts edge_index to CSR format (rowptr, col) for fast sampling."""
    from torch_geometric.utils import sort_edge_index
    edge_index = sort_edge_index(edge_index, num_nodes=num_nodes)
    row, col = edge_index
    # Manual index_to_ptr
    # row is sorted, so we can use bincount to get degrees
    degree = torch.bincount(row, minlength=num_nodes)
    rowptr = torch.cat([torch.tensor([0], device=row.device), degree.cumsum(0)])
    return rowptr, col

def random_walk_step(current_nodes, rowptr, col):
    """
    Performs one step of random walk for a batch of nodes.
    Args:
        current_nodes: Tensor of node indices [Batch]
        rowptr, col: CSR representation of the graph
    Returns:
        next_nodes: Tensor of next node indices [Batch]. -1 if no neighbors.
    """
    # Filter out invalid nodes (-1)
    valid_mask = current_nodes != -1
    valid_nodes = current_nodes[valid_mask]
    
    if valid_nodes.numel() == 0:
        return current_nodes
    
    # Get degrees
    row_start = rowptr[valid_nodes]
    row_end = rowptr[valid_nodes + 1]
    degrees = row_end - row_start
    
    # Identify nodes with neighbors
    has_neighbors = degrees > 0
    nodes_with_neighbors = valid_nodes[has_neighbors]
    
    if nodes_with_neighbors.numel() == 0:
        # No one has neighbors, all become -1
        next_nodes = current_nodes.clone()
        next_nodes[valid_mask] = -1
        return next_nodes

    # Sample neighbors
    # random_index = start + rand(0, degree)
    starts = row_start[has_neighbors]
    degs = degrees[has_neighbors]
    offsets = (torch.rand(degs.size(0), device=degs.device) * degs).long()
    neighbor_indices = starts + offsets
    sampled_neighbors = col[neighbor_indices]
    
    # Construct result
    next_nodes = current_nodes.clone()
    # Default to -1 for valid nodes
    next_nodes[valid_mask] = -1

    # Create a full mask of nodes that will get a neighbor
    update_mask = torch.zeros_like(current_nodes, dtype=torch.bool)
    # Indices in 'valid_nodes' that have neighbors
    valid_indices = torch.nonzero(valid_mask).squeeze()
    if valid_indices.dim() == 0: valid_indices = valid_indices.unsqueeze(0)
    
    indices_with_neighbors = valid_indices[has_neighbors]
    next_nodes[indices_with_neighbors] = sampled_neighbors
    
    return next_nodes

def _add_metapath_random_walk(data, mp_name, mp_def, walks=10, weighted=True, drop_self_loops=True):
    """
    Adds a metapath edge type using random walks.
    Supports arbitrary path lengths and avoids OOM by sampling.
    
    Args:
        data: HeteroData object
        mp_name: Name of the new metapath edge type
        mp_def: List of tuples defining the path [(src, rel, dst), (dst, rel2, dst2)...]
        walks: Number of random walks to start per source node.
        weighted: Whether to count path occurrences as weights.
        drop_self_loops: Whether to remove edges where src == dst.
    """
    import torch
    from torch_geometric.utils import coalesce

    src_type = mp_def[0][0]
    num_src = data[src_type].num_nodes
    
    print(f"  Processing {mp_name} (Random Walk, walks={walks})...")
    
    # Initialize walkers: [0, 0, 0, 1, 1, 1, ...] if walks=3

    # Determine device safely
    device = 'cpu'
    if hasattr(data[src_type], 'x') and isinstance(data[src_type].x, torch.Tensor):
        device = data[src_type].x.device
        
    start_nodes = torch.arange(num_src, device=device).repeat_interleave(walks)
    current_nodes = start_nodes.clone()
    
    # Traverse path
    for step_idx, (u, rel, v) in enumerate(mp_def):
        if (u, rel, v) not in data.edge_types:
            print(f"    Skipping {mp_name}: Edge type {(u, rel, v)} not found.")
            return data
            
        # Prepare CSR for this step
        edge_index = data[(u, rel, v)].edge_index
        
        # Ensure edge_index is a tensor
        if not isinstance(edge_index, torch.Tensor):
            edge_index = torch.tensor(edge_index, dtype=torch.long, device=device)
        else:
            edge_index = edge_index.to(device)
            
        num_u = data[u].num_nodes
        rowptr, col = to_csr(edge_index, num_u)
        
        # Step
        current_nodes = random_walk_step(current_nodes, rowptr, col)
        
        # Check if all died
        if torch.all(current_nodes == -1):
            print(f"    Path died at step {step_idx} ({u}-{rel}-{v})")
            return data

    # Filter valid paths
    dst_type = mp_def[-1][-1]
    
    # Filter valid paths
    valid_mask = current_nodes != -1
    
    if drop_self_loops and src_type == dst_type:
        # Remove self-loops for symmetric paths
        non_loop_mask = start_nodes != current_nodes
        valid_mask = valid_mask & non_loop_mask
        
    final_src = start_nodes[valid_mask]
    final_dst = current_nodes[valid_mask]
    
    if final_src.numel() == 0:
        print(f"    No valid paths found for {mp_name}")
        return data
        
    # --- 1. Store Forward Edge (Src -> Dst) ---
    # This represents the walk direction (e.g. Startup -> University)
    new_edge_index = torch.stack([final_src, final_dst], dim=0)
    
    if weighted:
        edge_index_coalesced, edge_weight = coalesce(
            new_edge_index, 
            torch.ones(new_edge_index.shape[1], dtype=torch.float), 
            reduce='add'
        )
        data[(src_type, mp_name, dst_type)].edge_index = edge_index_coalesced
        data[(src_type, mp_name, dst_type)].edge_weight = edge_weight
    else:
        edge_index_coalesced = coalesce(
            new_edge_index, 
            None, 
            reduce='mean'
        )[0]
        data[(src_type, mp_name, dst_type)].edge_index = edge_index_coalesced

    print(f"    Created {mp_name}: {data[(src_type, mp_name, dst_type)].edge_index.shape[1]} edges ({src_type}->{dst_type})")

    # --- 2. Store Reverse Edge (Dst -> Src) ---
    # Crucial for SeHGNN: If we walked Startup -> Uni, SeHGNN needs Uni -> Startup
    # to aggregate University features into the Startup node.
    if src_type == 'startup' and dst_type != 'startup':
        rev_name = f"rev_{mp_name}"
        print(f"    Creating reverse edge for SeHGNN: {dst_type} -> {rev_name} -> {src_type}")
        
        # Flip indices: [Dst, Src]
        rev_edge_index = torch.stack([final_dst, final_src], dim=0)
        
        if weighted:
             rev_idx_coalesced, rev_weight = coalesce(
                rev_edge_index,
                torch.ones(rev_edge_index.shape[1], dtype=torch.float),
                reduce='add'
            )
             data[(dst_type, rev_name, src_type)].edge_index = rev_idx_coalesced
             data[(dst_type, rev_name, src_type)].edge_weight = rev_weight
        else:
             rev_idx_coalesced = coalesce(
                rev_edge_index,
                None,
                reduce='mean'
            )[0]
             data[(dst_type, rev_name, src_type)].edge_index = rev_idx_coalesced

    # Add to metapath_definitions
    if not hasattr(data, 'metapath_definitions'):
        data.metapath_definitions = {}
    data.metapath_definitions[(src_type, mp_name, dst_type)] = mp_def
    
    return data

def add_metapaths(data, config):
    """
    Adds metapaths to the HeteroData object.
    Supports three modes via config:
    - manual: Use predefined metapaths (existing behavior)
    - automatic: Brute-force enumeration via matrix composition
    - hybrid: Combine manual + automatic
    """
    mode = config['metapath_discovery']['mode']
    
    print(f"\n{'='*60}")
    print(f"METAPATH DISCOVERY MODE: {mode.upper()}")
    print(f"{'='*60}\n")
    
    if mode == 'none':
        # No metapath materialization — use only base edges already in the graph
        print("Skipping metapath materialization (mode=none). Using base edges only.")
        return data

    elif mode == 'manual':
        # Use existing manual implementation
        return _add_metapaths_manual(data, config)

    elif mode == 'automatic':
        # Use automatic discovery
        from .metapath_discovery import MetapathDiscovery
        discoverer = MetapathDiscovery(config, data)
        metapaths_config = discoverer.discover()
        return _materialize_metapaths(data, metapaths_config, config)
    
    elif mode == 'hybrid':
        # Combine manual + automatic
        from .metapath_discovery import MetapathDiscovery
        discoverer = MetapathDiscovery(config, data)
        discovered = discoverer.discover()
        
        # Hybrid mode returns both manual and auto paths
        return _materialize_metapaths(data, discovered, config)
    
    else:
        raise ValueError(f"Unknown metapath discovery mode: {mode}")

def _materialize_metapaths(data, metapaths_config, config):
    """
    Materialize discovered metapaths into the graph.
    Supports both random walk and pre-computed composition methods.
    """
    
    print(f"\nMaterializing {len(metapaths_config)} metapaths...")

    # Materialize each metapath
    for mp_name, mp_config in metapaths_config.items():
        # Check if pre-computed edge_index exists (from automatic discovery)
        if 'edge_index' in mp_config and mp_config.get('materialize_method') == 'composition':
            # Use pre-computed adjacency from matrix composition
            path = mp_config['def']
            edge_index = mp_config['edge_index']
            start_type = path[0][0]
            end_type = path[-1][2]
            data[start_type, mp_name, end_type].edge_index = edge_index
            print(f"  ✓ {mp_name}: {edge_index.shape[1]} edges (pre-computed)")
        else:
            # Use random walk materialization
            data = _add_metapath_random_walk(
                data, 
                mp_name, 
                mp_config['def'], 
                walks=mp_config['walks'], 
                weighted=True,
                drop_self_loops=True
            )
    
    print("✅ Metapath materialization complete!\n")
    
    print(f"--- Final Edge Types ({len(data.edge_types)} total) ---")
    for edge_type in data.edge_types:
        if edge_type[1] in metapaths_config:
            print(f"  [METAPATH] {edge_type}")
        else:
            print(f"  [ORIGINAL] {edge_type}")
    
    return data

def _add_metapaths_manual(data, config):
    """
    Original manual metapath definition implementation.
    Adds metapaths to the HeteroData object based on random walks.
    """
    # Define all metapaths in a single dictionary with config
    # Format: name: {'def': [...], 'walks': int}
    metapaths_config = {
        # Serial founder (Zhang et al., HICSS 2024, M1): SPS via shared founder.
        # walks=10 matches portfolio_siblings; prolific founders (up to ~80 startups)
        # benefit from the larger budget to avoid under-sampling dense neighborhoods.
        'serial_founder': {
            'def': [
                ('startup', 'founded_by', 'founder'),
                ('founder', 'rev_founded_by', 'startup')
            ],
            'walks': 10
        },
        'early_portfolio_siblings': {
            'def': [('startup', 'early_stage_funded_by', 'investor'), ('investor', 'rev_early_stage_funded_by', 'startup')],
            'walks': 10
        },
        'late_portfolio_siblings': {
            'def': [('startup', 'late_stage_funded_by', 'investor'), ('investor', 'rev_late_stage_funded_by', 'startup')],
            'walks': 10
        },
        'sector_peers': {
            'def': [('startup', 'in_sector', 'sector'), ('sector', 'rev_in_sector', 'startup')],
            'walks': 5
        },
        'city_peers': {
            'def': [('startup', 'based_in', 'city'), ('city', 'rev_based_in', 'startup')],
            'walks': 5
        },
        'alumni_network': {
            'def': [
                ('startup', 'founded_by', 'founder'), 
                ('founder', 'studied_at', 'university'),
                ('university', 'rev_studied_at', 'founder'),
                ('founder', 'rev_founded_by', 'startup')
            ],
            'walks': 5
        },
        'early_investor_alumni': {
            'def': [
                ('startup', 'early_stage_funded_by', 'investor'),
                ('investor', 'same_as', 'founder'),
                ('founder', 'studied_at', 'university'),
                ('university', 'rev_studied_at', 'founder'),
                ('founder', 'rev_founded_by', 'startup')
            ],
            'walks': 5
        },
        'late_investor_alumni': {
            'def': [
                ('startup', 'late_stage_funded_by', 'investor'),
                ('investor', 'same_as', 'founder'),
                ('founder', 'studied_at', 'university'),
                ('university', 'rev_studied_at', 'founder'),
                ('founder', 'rev_founded_by', 'startup')
            ],
            'walks': 5
        },
        'early_alumni_investor_network': {
            'def': [
                ('startup', 'founded_by', 'founder'),
                ('founder', 'studied_at', 'university'),
                ('university', 'rev_studied_at', 'founder'),
                ('founder', 'same_as', 'investor'),
                ('investor', 'rev_early_stage_funded_by', 'startup')
            ],
            'walks': 5
        },
        'late_alumni_investor_network': {
            'def': [
                ('startup', 'founded_by', 'founder'),
                ('founder', 'studied_at', 'university'),
                ('university', 'rev_studied_at', 'founder'),
                ('founder', 'same_as', 'investor'),
                ('investor', 'rev_late_stage_funded_by', 'startup')
            ],
            'walks': 5
        },
        # Founder VC-employment metapaths
        'early_founder_vc_employment': {
            'def': [
                ('startup', 'founded_by', 'founder'),
                ('founder', 'worked_at', 'investor'),
                ('investor', 'rev_early_stage_funded_by', 'startup')
            ],
            'walks': 5
        },
        'late_founder_vc_employment': {
            'def': [
                ('startup', 'founded_by', 'founder'),
                ('founder', 'worked_at', 'investor'),
                ('investor', 'rev_late_stage_funded_by', 'startup')
            ],
            'walks': 5
        },
        'co_working_network': {
            'def': [
                ('startup', 'founded_by', 'founder'),
                ('founder', 'worked_with', 'founder'),
                ('founder', 'rev_founded_by', 'startup')
            ],
            'walks': 5
        },
        'early_investor_founder_coworking': {
            'def': [
                ('startup', 'early_stage_funded_by', 'investor'),
                ('investor', 'same_as', 'founder'),
                ('founder', 'worked_with', 'founder'),
                ('founder', 'rev_founded_by', 'startup')
            ],
            'walks': 5
        },
        'late_investor_founder_coworking': {
            'def': [
                ('startup', 'late_stage_funded_by', 'investor'),
                ('investor', 'same_as', 'founder'),
                ('founder', 'worked_with', 'founder'),
                ('founder', 'rev_founded_by', 'startup')
            ],
            'walks': 5
        },
        'early_founder_coworking_investor': {
            'def': [
                ('startup', 'founded_by', 'founder'),
                ('founder', 'worked_with', 'founder'),
                ('founder', 'same_as', 'investor'),
                ('investor', 'rev_early_stage_funded_by', 'startup')
            ],
            'walks': 5
        },
        'mid_founder_coworking_investor': {
            'def': [
                ('startup', 'founded_by', 'founder'),
                ('founder', 'worked_with', 'founder'),
                ('founder', 'same_as', 'investor'),
                ('investor', 'rev_mid_stage_funded_by', 'startup')
            ],
            'walks': 5
        },
        'late_founder_coworking_investor': {
            'def': [
                ('startup', 'founded_by', 'founder'),
                ('founder', 'worked_with', 'founder'),
                ('founder', 'same_as', 'investor'),
                ('investor', 'rev_late_stage_funded_by', 'startup')
            ],
            'walks': 5
        },
        # Syndicate Metapaths (Startup -> Investor -> Founder -> Coworker -> Investor -> Startup)
        'early_founder_coworking_syndicate': {
            'def': [
                ('startup', 'early_stage_funded_by', 'investor'),
                ('investor', 'same_as', 'founder'),
                ('founder', 'worked_with', 'founder'),
                ('founder', 'same_as', 'investor'),
                ('investor', 'rev_early_stage_funded_by', 'startup')
            ],
            'walks': 5
        },
        'mid_founder_coworking_syndicate': {
            'def': [
                ('startup', 'mid_stage_funded_by', 'investor'),
                ('investor', 'same_as', 'founder'),
                ('founder', 'worked_with', 'founder'),
                ('founder', 'same_as', 'investor'),
                ('investor', 'rev_mid_stage_funded_by', 'startup')
            ],
            'walks': 5
        },
        'late_founder_coworking_syndicate': {
            'def': [
                ('startup', 'late_stage_funded_by', 'investor'),
                ('investor', 'same_as', 'founder'),
                ('founder', 'worked_with', 'founder'),
                ('founder', 'same_as', 'investor'),
                ('investor', 'rev_late_stage_funded_by', 'startup')
            ],
            'walks': 5
        },
        # Mid variants for existing metapaths
        'mid_portfolio_siblings': {
            'def': [('startup', 'mid_stage_funded_by', 'investor'), ('investor', 'rev_mid_stage_funded_by', 'startup')],
            'walks': 10
        },
        'mid_investor_alumni': {
            'def': [
                ('startup', 'mid_stage_funded_by', 'investor'),
                ('investor', 'same_as', 'founder'),
                ('founder', 'studied_at', 'university'),
                ('university', 'rev_studied_at', 'founder'),
                ('founder', 'rev_founded_by', 'startup')
            ],
            'walks': 5
        },
        'mid_alumni_investor_network': {
            'def': [
                ('startup', 'founded_by', 'founder'),
                ('founder', 'studied_at', 'university'),
                ('university', 'rev_studied_at', 'founder'),
                ('founder', 'same_as', 'investor'),
                ('investor', 'rev_mid_stage_funded_by', 'startup')
            ],
            'walks': 5
        },
        'mid_founder_vc_employment': {
            'def': [
                ('startup', 'founded_by', 'founder'),
                ('founder', 'worked_at', 'investor'),
                ('investor', 'rev_mid_stage_funded_by', 'startup')
            ],
            'walks': 5
        },
        'mid_investor_founder_coworking': {
            'def': [
                ('startup', 'mid_stage_funded_by', 'investor'),
                ('investor', 'same_as', 'founder'),
                ('founder', 'worked_with', 'founder'),
                ('founder', 'rev_founded_by', 'startup')
            ],
            'walks': 5
        },
        # Board Director Metapaths (Startup -> Board Member -> Investor -> Startup)
        'early_board_director_network': {
            'def': [
                ('startup', 'rev_on_board_of', 'founder'),
                ('founder', 'director_at', 'investor'),
                ('investor', 'rev_early_stage_funded_by', 'startup')
            ],
            'walks': 5
        },
        'mid_board_director_network': {
            'def': [
                ('startup', 'rev_on_board_of', 'founder'),
                ('founder', 'director_at', 'investor'),
                ('investor', 'rev_mid_stage_funded_by', 'startup')
            ],
            'walks': 5
        },
        'late_board_director_network': {
            'def': [
                ('startup', 'rev_on_board_of', 'founder'),
                ('founder', 'director_at', 'investor'),
                ('investor', 'rev_late_stage_funded_by', 'startup')
            ],
            'walks': 5
        },
        # Board Employment Metapaths (Startup -> Board Member -> Works/Worked At -> Investor -> Startup)
        'early_board_employment_network': {
            'def': [
                ('startup', 'rev_on_board_of', 'founder'),
                ('founder', 'worked_at', 'investor'),
                ('investor', 'rev_early_stage_funded_by', 'startup')
            ],
            'walks': 5
        },
        'mid_board_employment_network': {
            'def': [
                ('startup', 'rev_on_board_of', 'founder'),
                ('founder', 'worked_at', 'investor'),
                ('investor', 'rev_mid_stage_funded_by', 'startup')
            ],
            'walks': 5
        },
        'late_board_employment_network': {
            'def': [
                ('startup', 'rev_on_board_of', 'founder'),
                ('founder', 'worked_at', 'investor'),
                ('investor', 'rev_late_stage_funded_by', 'startup')
            ],
            'walks': 5
        }
    }

    # --- NEW CONTEXT METAPATHS (Target-to-Non-Target) ---
    if config["features"].get("use_non_target_metapaths", False):
        print("✅ Enabling Context Metapaths (Target-to-Non-Target)...")
        context_metapaths = {
            'startup_founder_uni': {
                'def': [
                    ('startup', 'founded_by', 'founder'),
                    ('founder', 'studied_at', 'university')
                ],
                'walks': 5
            },
            'startup_investor_city': {
                'def': [
                    ('startup', 'early_stage_funded_by', 'investor'),
                    ('investor', 'based_in', 'city')
                ],
                'walks': 5
            },
            # 'startup_local_uni': {
            #     'def': [
            #         ('startup', 'based_in', 'city'),
            #         ('city', 'rev_based_in', 'university')
            #     ],
            #     'walks': 5
            # },
            'startup_board_investor': {
                'def': [
                    ('startup', 'rev_on_board_of', 'founder'),
                    ('founder', 'director_at', 'investor')
                ],
                'walks': 5
            },
            'startup_network_investor': {
                'def': [
                    ('startup', 'founded_by', 'founder'),
                    ('founder', 'worked_with', 'founder'),
                    ('founder', 'worked_at', 'investor')
                ],
                'walks': 5
            }
        }
        metapaths_config.update(context_metapaths)

    # --- WHITELIST FILTER ---
    # If manual.whitelist is set in config, only keep the listed metapaths.
    # This allows controlling channel count without deleting code definitions.
    manual_config = config.get('metapath_discovery', {}).get('manual', {})
    whitelist = manual_config.get('whitelist', None)
    if whitelist:
        before = len(metapaths_config)
        metapaths_config = {k: v for k, v in metapaths_config.items() if k in whitelist}
        print(f"  Whitelist filter: {before} → {len(metapaths_config)} metapaths (kept: {list(metapaths_config.keys())})")

    print(f"Adding {len(metapaths_config)} metapaths using Random Walks...")

    for mp_name, mp_config in metapaths_config.items():
        data = _add_metapath_random_walk(
            data, 
            mp_name, 
            mp_config['def'], 
            walks=mp_config['walks'], 
            weighted=True,
            drop_self_loops=True
        )

    print("Transformation successful!\n")
    
    print(f"--- Final Edge Types ({len(data.edge_types)} total) ---")
    for edge_type in data.edge_types:
        if edge_type[1] in metapaths_config:
            print(f"  [METAPATH] {edge_type}")
        elif 'metapath' in edge_type[1]:
            print(f"  [METAPATH-UNK] {edge_type}")
        else:
            print(f"  [ORIGINAL] {edge_type}")
    
    return data


def create_graph(startup_df, params, config=None):
    if config is None:
        config = load_config()

    data = define_hetero_data(params)
    
    # Attach filtered dataframe to graph object for later use (e.g. export)
    data['startup'].df = startup_df

    # Make graph undirected
    data = T.ToUndirected()(data)
    
    # Add explicit self-loops for node types EXCEPT 'startup'
    # We exclude 'startup' because we want the Graph Branch of the HAN to be forced
    # to aggregate NEIGHBOR information (Investors, etc.).
    # The 'startup' self-features are already provided via the Residual Connection.
    # If we add a self-loop here, the model learns to set Attention(Self-Loop) = 1.0
    # to bypass the Residual Dropout penalty, effectively ignoring the graph structure.
    if config["data_processing"].get("add_self_loops", False):
        print("Adding explicit self-loops for auxiliary node types (excluding startup)...")
        for node_type in data.node_types:
            if node_type == "startup":
                continue
                
            num_nodes = data[node_type].num_nodes
            loop_index = torch.arange(num_nodes, dtype=torch.long)
            loop_edge_index = torch.stack([loop_index, loop_index], dim=0)
            data[node_type, 'self_loop', node_type].edge_index = loop_edge_index
    else:
        print("Skipping self-loops (disabled in config).")
    
    # --- ABLATION VERIFICATION ---
    # Print proof that nodes/edges are removed
    drop_node_types = config["data_processing"].get("ablation", {}).get("drop_node_types", [])
    if drop_node_types:
        print("\n" + "=" * 50)
        print("VERIFYING ABLATION (DROP NODE TYPES)")
        print("=" * 50)
        print(f"Targeted for removal: {drop_node_types}")
        
        # Check Node Types
        all_node_types = data.node_types
        print(f"Current Graph Node Types: {all_node_types}")
        for dropped_node in drop_node_types:
            if dropped_node not in all_node_types:
                print(f"  ✅ Node type '{dropped_node}' is ABSENT from graph keys (Successfully dropped).")
            else:
                count = data[dropped_node].num_nodes
                if count == 0:
                     print(f"  ✅ Node type '{dropped_node}' is present but EMPTY (Count: 0).")
                else:
                     print(f"  ❌ FAILURE: Node type '{dropped_node}' is present with {count} nodes!")

        # Check Edge Types (Metapaths)
        print("\nChecking for residual edges involving dropped nodes:")
        found_residual = False
        for edge_type in data.edge_types:
            src, rel, dst = edge_type
            if src in drop_node_types or dst in drop_node_types:
                print(f"  ❌ FAILURE: Found edge involving dropped node: {edge_type}")
                found_residual = True
        
        if not found_residual:
            print("  ✅ No edges found involving dropped node types.")
        print("=" * 50 + "\n")

    # Add metapaths for HAN model
    if config["data_processing"]["add_metapaths"]:
        print("\n" + "=" * 50)
        print("ADDING METAPATHS FOR HAN")
        print("=" * 50)
        data = add_metapaths(data, config)

    # Drop the `(founder, worked_with, founder)` base edge after metapath
    # materialization. The coworking relation is a clique expansion (~3.5M edges
    # for ~514k founders) that dominates the graph; the signal it carries is
    # already captured by the coworking-based startup→startup metapaths
    # (early/mid/late_founder_coworking_*, *_investor_founder_coworking,
    # co_working_network, *_founder_coworking_syndicate), which were just
    # materialized above. Keeping both double-counts coworking signal and
    # inflates message-passing cost. Done unconditionally so every graph build
    # is consistent.
    ww = ("founder", "worked_with", "founder")
    if ww in data.edge_types:
        n_dropped = data[ww].edge_index.shape[1]
        del data[ww]
        print(f"🧹 Dropped base edge {ww}: {n_dropped:,} edges "
              f"(signal preserved via coworking metapaths).")

    # Remove isolated nodes (nodes with no connections)
    print("\n" + "=" * 50)
    print("REMOVING ISOLATED NODES")
    print("=" * 50)
    
    total_nodes_before = sum([data[node_type].x.shape[0] for node_type in data.node_types])
    print(f"Total nodes before: {total_nodes_before}")
    
    data = data # T.RemoveIsolatedNodes()(data) - DISABLED to preserve isolated 'startup' nodes during ablation
    
    total_nodes_after = sum([data[node_type].x.shape[0] for node_type in data.node_types])
    total_removed = total_nodes_before - total_nodes_after
    print(f"Total nodes after: {total_nodes_after}")
    print(f"Removed {total_removed} isolated nodes ({total_removed/total_nodes_before:.1%})")

    # Prepare features
    data = prepare_node_features(data)

    # Add Endogenous Clusters (Louvain)
    if config["data_processing"].get("use_louvain_clusters", False):
        data, params["feature_names"] = add_endogenous_clusters(data, params["feature_names"], config=config)

    # Split data into train/test/val
    print("\n" + "=" * 50)
    print("SPLITING DATA")
    print("=" * 50)

    data = split_data_in_train_test_val(
        startup_df=startup_df,
        graph_data=data,
        config=config,
    )
    startup_df.drop(columns=["founded_on"], inplace=True)
    
    data, params = add_graph_features_per_split(data, params, config=config)

    # Perform imputation BEFORE normalization to prevent data leakage
    print("\n" + "=" * 50)
    print("PERFORMING IMPUTATION")
    print("=" * 50)

    if config["imputation"].get("enabled", True):
        imputer = GraphImputer(
            imputation_config=config["imputation"]["node_types"],
            feature_column_map=params["feature_names"],
        )
        data = imputer.fit_transform(data, train_mask_key="train_mask")
    else:
        print("Imputation is disabled.")

    print("\n" + "=" * 50)
    print("PERFORMING NORMALIZATION")
    print("=" * 50)

    # Normalize features after imputation
    data = normalize_split_features(
        graph_data=data,
        normalization=MinMaxScaler,
    )

    # Validate splits
    check_for_mask_overlap(
        data["startup"].train_mask,
        data["startup"].test_mask,
        data["startup"].val_mask,
    )

    # Final tensor conversion
    for node_type, x in data.x_dict.items():
        if isinstance(x, np.ndarray):
            data.x_dict[node_type] = torch.tensor(x, dtype=torch.float32)
        elif isinstance(x, torch.Tensor):
            data.x_dict[node_type] = x.to(dtype=torch.float32)
        else:
            raise TypeError(f"Unexpected type for x_dict[{node_type}]: {type(x)}")

    # Export startup degree dict for post-hoc degree-stratified evaluation
    try:
        num_startups = data['startup'].x.shape[0]
        degree_counts = np.zeros(num_startups, dtype=int)
        for edge_type in data.edge_types:
            src_type, _, dst_type = edge_type
            ei = data[edge_type].edge_index
            if src_type == 'startup':
                for idx in ei[0].cpu().numpy():
                    if idx < num_startups:
                        degree_counts[idx] += 1
            if dst_type == 'startup':
                for idx in ei[1].cpu().numpy():
                    if idx < num_startups:
                        degree_counts[idx] += 1

        # Map to UUIDs
        degree_dict = {}
        if hasattr(data['startup'], 'df') and data['startup'].df is not None:
            for i, row in enumerate(data['startup'].df.itertuples()):
                if i < num_startups:
                    degree_dict[row.startup_uuid] = int(degree_counts[i])

        if degree_dict:
            import json as _json
            seed = config.get("seed", "unknown")
            output_dir = config.get("output_dir", "outputs")
            dd_dir = os.path.join(output_dir, "degree_dicts")
            os.makedirs(dd_dir, exist_ok=True)
            dd_path = os.path.join(dd_dir, f"startup_degrees_seed{seed}.json")
            with open(dd_path, 'w') as _f:
                _json.dump(degree_dict, _f)
            print(f"📊 Saved startup degree dict ({len(degree_dict)} nodes) to {dd_path}")
    except Exception as e:
        print(f"⚠️ Failed to export degree dict: {e}")

    # Apply post-build graph-variant filter for the ICDM graph-curation
    # ablation. No-op when graph_variant is unset or "g1_full".
    variant = config["data_processing"].get("graph_variant")
    if variant and variant != "g1_full":
        from src.ml.graph_variants import apply_graph_variant
        data = apply_graph_variant(data, variant)

    return data, params


def normalize_features(features, scaler=None, fit=False, normalization=MinMaxScaler):
    if fit:
        scaler = normalization()
        normalized_features = scaler.fit_transform(features)
    else:
        normalized_features = scaler.transform(features)
    return torch.tensor(normalized_features, dtype=torch.float32), scaler


def check_for_mask_overlap(startup_train_mask, startup_test_mask, startup_val_mask):
    for train, test, val in zip(
        startup_train_mask, startup_test_mask, startup_val_mask
    ):
        if train:
            assert not test and not val
        if test:
            assert not val and not train
        if val:
            assert not test and not train


def normalize_split_features(graph_data, normalization):
    """
    Normalize features with train-test split awareness.
    Only fit normalizers on training data to prevent data leakage.
    """
    train_mask = graph_data["startup"].train_mask
    val_mask = graph_data["startup"].val_mask
    test_mask = graph_data["startup"].test_mask
    test_original_mask = graph_data["startup"].test_mask_original

    # Save pre-normalization copies (post-imputation) for visualization
    for node_type in graph_data.node_types:
        if not hasattr(graph_data[node_type], 'x'):
            continue
        node_x = graph_data[node_type].x
        if isinstance(node_x, torch.Tensor):
            graph_data[node_type].x_pre_norm = node_x.clone()
        else:
            # convert numpy arrays to torch then clone
            graph_data[node_type].x_pre_norm = torch.tensor(np.array(node_x), dtype=torch.float32)

    # Iterate over all node types to normalize features
    for node_type in graph_data.node_types:
        # Skip if no features
        if not hasattr(graph_data[node_type], 'x'):
            continue
        if graph_data[node_type].x.shape[1] == 0:
            continue

        # For 'startup' node, we respect the train/val/test splits
        if node_type == "startup":
            # Normalize training node features and fit the scaler
            # Handle NaNs before normalization
            graph_data["startup"].x[train_mask] = torch.nan_to_num(graph_data["startup"].x[train_mask], nan=0.0)
            
            graph_data["startup"].x[train_mask], scaler_startup = normalize_features(
                graph_data["startup"].x[train_mask], fit=True, normalization=normalization
            )
            
            # Normalize validation and test node features using the fitted scaler
            # Note: graph_data["startup"].x currently holds x_train_mask.
            # We also need to normalize the separate split tensors: x_val_mask, x_test_mask, etc.
            
            # 1. Normalize the rest of the base .x (which is x_train_mask)
            # Although validation/test nodes in x_train_mask are zero/imputed, we normalize them for consistency
            if any(val_mask):
                graph_data["startup"].x[val_mask] = torch.nan_to_num(graph_data["startup"].x[val_mask], nan=0.0)
                graph_data["startup"].x[val_mask], _ = normalize_features(
                    graph_data["startup"].x[val_mask], scaler=scaler_startup
                )
            if any(test_mask):
                graph_data["startup"].x[test_mask] = torch.nan_to_num(graph_data["startup"].x[test_mask], nan=0.0)
                graph_data["startup"].x[test_mask], _ = normalize_features(
                    graph_data["startup"].x[test_mask], scaler=scaler_startup
                )
            if any(test_original_mask):
                graph_data["startup"].x[test_original_mask] = torch.nan_to_num(graph_data["startup"].x[test_original_mask], nan=0.0)
                graph_data["startup"].x[test_original_mask], _ = normalize_features(
                    graph_data["startup"].x[test_original_mask], scaler=scaler_startup
                )
                
            # 2. Normalize the specific split tensors (x_val_mask, x_test_mask, etc.)
            # These are the ones we swap to during evaluation!
            for split in ['val_mask', 'test_mask', 'test_mask_original']:
                split_key = f"x_{split}"
                if hasattr(graph_data["startup"], split_key):
                    print(f"Normalizing split feature tensor: {split_key}")
                    split_tensor = getattr(graph_data["startup"], split_key)
                    
                    # Handle NaNs
                    split_tensor = torch.nan_to_num(split_tensor, nan=0.0)
                    
                    # Save pre-norm copy for visualization
                    setattr(graph_data["startup"], f"{split_key}_pre_norm", split_tensor.clone())
                    
                    # Normalize using the scaler fitted on training data
                    # We normalize the WHOLE tensor because in this tensor, all nodes have features computed from that split's perspective
                    split_tensor, _ = normalize_features(
                        split_tensor, scaler=scaler_startup
                    )
                    
                    # Update the tensor in the data object
                    setattr(graph_data["startup"], split_key, split_tensor)
        
        # Non-target node types have no train/val/test split here, so we normalize the
        # full feature set (their features are available at training time in the
        # transductive setting).
        else:
            graph_data[node_type].x = torch.nan_to_num(graph_data[node_type].x, nan=0.0)
            graph_data[node_type].x, _ = normalize_features(
                graph_data[node_type].x, fit=True, normalization=normalization
            )

    return graph_data


def resample_split_by_class(df, mask, label_col, class_distribution):
    """
    Resample a dataset split to match a target class distribution via downsampling.

    Args:
        df (pd.DataFrame): The full dataset.
        mask (torch.Tensor): Boolean mask selecting the split.
        label_col (str): Name of the column containing class labels.
        class_distribution (list of float): Desired class distribution, summing to 1.

    Returns:
        torch.Tensor: Updated boolean mask after resampling.
    """
    df_split = df[mask.numpy()].copy()
    if df_split.empty:
        return mask  # no need to resample

    class_counts = df_split[label_col].value_counts()
    all_classes = sorted(class_counts.index)
    if len(all_classes) != len(class_distribution):
        raise ValueError(
            "Length of class_distribution must match number of classes in the split."
        )

    # Compute the total number of samples to keep (limited by the most constrained class)
    available_counts = {cls: class_counts.get(cls, 0) for cls in all_classes}
    max_total = min(
        int(available_counts[cls] / class_distribution[i])
        for i, cls in enumerate(all_classes)
        if class_distribution[i] > 0
    )

    target_counts = {
        cls: int(class_distribution[i] * max_total) for i, cls in enumerate(all_classes)
    }

    resampled_parts = []
    for cls in all_classes:
        cls_df = df_split[df_split[label_col] == cls]
        count = target_counts[cls]
        if len(cls_df) < count:
            continue  # skip class if not enough samples; this avoids oversampling
        resampled_cls = resample(
            cls_df, replace=False, n_samples=count, random_state=42
        )
        resampled_parts.append(resampled_cls)

    if not resampled_parts:
        return mask  # fallback in case no class met threshold

    resampled_df = pd.concat(resampled_parts)
    resampled_indices = resampled_df.index.values

    # Create new mask
    new_mask = torch.zeros_like(mask, dtype=torch.bool)
    new_mask[resampled_indices] = True
    return new_mask


def add_graph_features_per_split(data, params, config=None):
    """
    Add graph-derived features to nodes.

    Two phases:
      Phase 1 — Structural features (edge counts, degree centrality, PageRank,
                per-edge-type centrality) are computed ONCE on the full graph
                (transductive).  They use only topology, so no label leakage.
      Phase 2 — Label-derived features (smart money) are computed per split
                using training labels only, to prevent leakage.
    """
    import torch
    import networkx as nx
    if config is None:
        config = load_config()

    splits = ['train_mask', 'val_mask', 'test_mask', 'test_mask_original']

    # Get config switches
    dp_config = config["data_processing"]
    structural_only = dp_config.get("structural_only", False)
    use_edge_counts = dp_config.get("use_edge_counts", True)
    use_degree_centrality = dp_config.get("use_degree_centrality", True)
    use_pagerank_centrality = dp_config.get("use_pagerank_centrality", True)
    use_centrality_features = dp_config.get("use_centrality_features", False)
    use_smart_money_features = dp_config.get("use_smart_money_features", False)

    if structural_only:
        print("[Experimental] 'structural_only' enabled: Will drop all tabular features and keep only computed graph features.")

    edge_info = {
        'startup': {
            'outgoing': {
                'funded_by': 'investor',
                'based_in': 'city',
                'founded_by': 'founder',
                'in_sector': 'sector',
            }
        },
        'investor': {
            'incoming': {
                'funded_by': 'startup',
                'worked_at': 'founder',
                'same_as': 'founder'
            },
            'outgoing': {
                'based_in': 'city',
                'same_as': 'founder'
            }
        },
        'founder': {
            'incoming': {
                'founded_by': 'startup',
                'worked_with': 'founder',
                'same_as': 'investor',
            },
            'outgoing': {
                'studied_at': 'university',
                'worked_at': 'investor',
                'worked_with': 'founder',
                'same_as': 'investor',
            }
        },
        'city': {
            'incoming': {'based_in': ['startup', 'investor', 'university']}
        },
        'sector': {
            'incoming': {'in_sector': 'startup'}
        },
        'university': {
            'incoming': {'studied_at': 'founder'},
            'outgoing': {'based_in': 'city'}
        }
    }

    # Pre-calculate node offsets for global graph construction (used in PageRank)
    node_offsets = {}
    cum_sum = 0
    for nt in data.node_types:
        node_offsets[nt] = cum_sum
        if hasattr(data[nt], 'num_nodes') and data[nt].num_nodes is not None:
             cum_sum += data[nt].num_nodes
        elif hasattr(data[nt], 'x') and data[nt].x is not None:
             cum_sum += data[nt].x.shape[0]

    def _safe_bincount(indices, num_nodes):
        """bincount with guaranteed output size == num_nodes."""
        degs = torch.bincount(indices, minlength=num_nodes).float()
        if len(degs) > num_nodes:
            degs = degs[:num_nodes]
        elif len(degs) < num_nodes:
            degs = torch.cat([degs, torch.zeros(num_nodes - len(degs))])
        return degs

    # =========================================================================
    # PHASE 1: Structural features — computed ONCE on the full graph
    # These use only graph topology (no labels), consistent with transductive
    # GNN message passing which also sees the full graph.
    # =========================================================================
    global_features = {}  # {node_type: (tensor[N, F], [feat_name, ...])}

    if use_edge_counts or use_degree_centrality or use_pagerank_centrality:
        print("Computing structural graph features on full graph (transductive)...")

        for node_type in edge_info.keys():
            if hasattr(data[node_type], 'num_nodes') and data[node_type].num_nodes is not None:
                num_nodes = data[node_type].num_nodes
            elif hasattr(data[node_type], 'x') and data[node_type].x is not None:
                num_nodes = data[node_type].x.shape[0]
            else:
                num_nodes = 0
            if num_nodes == 0:
                continue

            all_feats, feat_names = [], []

            # --- Edge Counts (full graph) ---
            if use_edge_counts:
                for edge_type, target_type in edge_info[node_type].get('outgoing', {}).items():
                    if (node_type, edge_type, target_type) not in data.edge_types:
                        continue
                    edge_index = data[node_type, edge_type, target_type].edge_index
                    edge_counts = torch.zeros(num_nodes, dtype=torch.float32)
                    if edge_index.shape[1] > 0:
                        unique_nodes, counts = torch.unique(edge_index[0], return_counts=True)
                        edge_counts[unique_nodes] = counts.float()
                    all_feats.append(torch.nan_to_num(edge_counts, nan=0.0).unsqueeze(1))
                    feat_names.append(f"{node_type}_to_{target_type}_count")

                for edge_type, source_types in edge_info[node_type].get('incoming', {}).items():
                    source_types = [source_types] if isinstance(source_types, str) else source_types
                    for source_type in source_types:
                        if (source_type, edge_type, node_type) not in data.edge_types:
                            continue
                        edge_index = data[source_type, edge_type, node_type].edge_index
                        edge_counts = torch.zeros(num_nodes, dtype=torch.float32)
                        if edge_index.shape[1] > 0:
                            unique_nodes, counts = torch.unique(edge_index[1], return_counts=True)
                            edge_counts[unique_nodes] = counts.float()
                        all_feats.append(torch.nan_to_num(edge_counts, nan=0.0).unsqueeze(1))
                        feat_names.append(f"{source_type}_to_{node_type}_count")

            # --- Centrality (full graph) ---
            if use_degree_centrality or use_pagerank_centrality:
                global_rels = [rel for rel in data.edge_types
                               if rel[0] == node_type or rel[2] == node_type]

                if use_degree_centrality:
                    deg_tensor = torch.zeros(num_nodes, dtype=torch.float32)
                    for rel in global_rels:
                        src, etype, dst = rel
                        edge_index = data[rel].edge_index
                        if edge_index.shape[1] > 0:
                            if src == node_type:
                                deg_tensor += _safe_bincount(edge_index[0], num_nodes)
                            if dst == node_type and src != dst:
                                deg_tensor += _safe_bincount(edge_index[1], num_nodes)
                    if num_nodes > 1:
                        deg_tensor = deg_tensor / (num_nodes - 1)
                    all_feats.append(deg_tensor.unsqueeze(1))
                    feat_names.append(f"{node_type}_global_degree_centrality")

                if use_pagerank_centrality:
                    G_global = nx.Graph()
                    for rel in global_rels:
                        if rel not in data.edge_types:
                            continue
                        src, etype, dst = rel
                        edge_index = data[rel].edge_index
                        if edge_index.shape[1] > 0:
                            u = (edge_index[0] + node_offsets[src]).cpu().numpy()
                            v = (edge_index[1] + node_offsets[dst]).cpu().numpy()
                            G_global.add_edges_from(zip(u, v))

                    if G_global.number_of_nodes() > 0:
                        try:
                            pagerank = nx.pagerank(G_global, alpha=0.85, max_iter=100)
                            pr_tensor = torch.tensor(
                                [pagerank.get(i + node_offsets[node_type], 0.0) for i in range(num_nodes)],
                                dtype=torch.float32)
                        except (nx.NetworkXError, nx.PowerIterationFailedConvergence, KeyError) as e:
                            print(f"   Warning: PageRank failed for {node_type}: {e}")
                            pr_tensor = torch.zeros(num_nodes, dtype=torch.float32)
                    else:
                        pr_tensor = torch.zeros(num_nodes, dtype=torch.float32)
                    all_feats.append(torch.nan_to_num(pr_tensor, nan=0.0).unsqueeze(1))
                    feat_names.append(f"{node_type}_global_pagerank_centrality")

                # Per-edge-type centrality
                if use_centrality_features and use_degree_centrality:
                    for rel in global_rels:
                        src, etype, dst = rel
                        if src != node_type and dst != node_type:
                            continue
                        if src == dst:
                            feat_name = f"{src}_{etype}_degree_centrality"
                        else:
                            other = dst if src == node_type else src
                            feat_name = f"{node_type}_{etype}_{other}_degree_centrality"

                        deg_tensor = torch.zeros(num_nodes, dtype=torch.float32)
                        edge_index = data[rel].edge_index
                        if edge_index.shape[1] > 0:
                            if src == node_type:
                                deg_tensor += _safe_bincount(edge_index[0], num_nodes)
                            if dst == node_type and src != dst:
                                deg_tensor += _safe_bincount(edge_index[1], num_nodes)
                        if num_nodes > 1:
                            deg_tensor = deg_tensor / (num_nodes - 1)
                        all_feats.append(torch.nan_to_num(deg_tensor, nan=0.0).unsqueeze(1))
                        feat_names.append(feat_name)

            if all_feats:
                global_features[node_type] = (torch.cat(all_feats, dim=1), feat_names)
                t = global_features[node_type][0]
                nz = torch.count_nonzero(t)
                print(f"  {node_type}: {len(feat_names)} features, "
                      f"Non-zeros={nz}/{t.numel()} ({nz/t.numel()*100:.1f}%)")

    # =========================================================================
    # PHASE 2: Per-split assembly
    # Global structural features are reused identically for every split.
    # Smart money (label-derived) is computed per-split using training labels.
    # =========================================================================
    for split in splits:
        if not hasattr(data['startup'], split):
            continue

        print(f"\nAssembling features for split '{split}'...")
        split_mask = getattr(data['startup'], split)

        # Visible nodes (cumulative: train ⊂ val ⊂ test)
        if split == 'train_mask':
             visible_mask = data['startup'].train_mask
        elif split == 'val_mask':
             visible_mask = data['startup'].train_mask | data['startup'].val_mask
        elif split == 'test_mask':
             visible_mask = data['startup'].train_mask | data['startup'].val_mask | data['startup'].test_mask
        elif split == 'test_mask_original':
             visible_mask = data['startup'].train_mask | data['startup'].val_mask | data['startup'].test_mask_original
        else:
             visible_mask = split_mask
        startup_nodes_in_split = torch.where(visible_mask)[0]

        for node_type in edge_info.keys():
            if hasattr(data[node_type], 'num_nodes') and data[node_type].num_nodes is not None:
                num_nodes = data[node_type].num_nodes
            elif hasattr(data[node_type], 'x') and data[node_type].x is not None:
                num_nodes = data[node_type].x.shape[0]
            else:
                num_nodes = 0
            if num_nodes == 0:
                continue

            all_edge_features, feature_names = [], []

            # Add precomputed global structural features (identical for every split)
            if node_type in global_features:
                gf_tensor, gf_names = global_features[node_type]
                all_edge_features.append(gf_tensor)
                feature_names.extend(gf_names)

            # --- Smart Money Features (per-split, uses training labels) ---
            if use_smart_money_features and node_type == 'startup':
                # 1. Compute investor scores from TRAINING labels (once, cached on data)
                if not hasattr(data, 'investor_scores'):
                    print("  Computing Global Investor Scores (Smart Money) from Training Data...")
                    train_mask = data['startup'].train_mask
                    train_idx = torch.where(train_mask)[0]

                    if train_idx.numel() > 0:
                        y_train = data['startup'].y[train_idx]
                        target_col_idx = 1  # Liquidity for masked_multi_task
                        if y_train.dim() == 1:
                            target_col_idx = 0

                        investor_rels = [
                            ('startup', 'early_stage_funded_by', 'investor'),
                            ('startup', 'mid_stage_funded_by', 'investor'),
                            ('startup', 'late_stage_funded_by', 'investor'),
                            ('startup', 'other_funded_by', 'investor'),
                            ('startup', 'funded_by', 'investor')
                        ]

                        num_investors = data['investor'].num_nodes
                        investor_success_sum = torch.zeros(num_investors, device=y_train.device)
                        investor_counts = torch.zeros(num_investors, device=y_train.device)

                        global_y = torch.zeros(data['startup'].num_nodes, device=y_train.device)
                        if data['startup'].y.dim() > 1:
                            global_y[train_idx] = data['startup'].y[train_idx, target_col_idx]
                        else:
                            global_y[train_idx] = data['startup'].y[train_idx]

                        for rel in investor_rels:
                            if rel in data.edge_types:
                                edge_index = data[rel].edge_index
                                mask = torch.isin(edge_index[0], train_idx)
                                filtered_edges = edge_index[:, mask]
                                if filtered_edges.shape[1] > 0:
                                    s_idx = filtered_edges[0]
                                    i_idx = filtered_edges[1]
                                    investor_success_sum.index_add_(0, i_idx, global_y[s_idx])
                                    investor_counts.index_add_(0, i_idx, torch.ones_like(global_y[s_idx]))

                        investor_counts = investor_counts.clamp(min=1.0)
                        data.investor_scores = investor_success_sum / investor_counts
                        print(f"  Calculated scores for {num_investors} investors. "
                              f"Mean: {data.investor_scores.mean():.4f}, Max: {data.investor_scores.max():.4f}")
                    else:
                        print("  No training data found for Smart Money calculation!")
                        data.investor_scores = torch.zeros(data['investor'].num_nodes)

                # 2. Aggregate investor scores for startups visible in this split
                startups_in_split = startup_nodes_in_split
                agg_sum = torch.zeros(data['startup'].num_nodes, device=data.investor_scores.device)
                agg_max = torch.zeros(data['startup'].num_nodes, device=data.investor_scores.device)
                agg_count = torch.zeros(data['startup'].num_nodes, device=data.investor_scores.device)

                investor_rels = [
                    ('startup', 'early_stage_funded_by', 'investor'),
                    ('startup', 'mid_stage_funded_by', 'investor'),
                    ('startup', 'late_stage_funded_by', 'investor'),
                    ('startup', 'other_funded_by', 'investor'),
                    ('startup', 'funded_by', 'investor')
                ]

                mapped_any = False
                for rel in investor_rels:
                    if rel in data.edge_types:
                        edge_index = data[rel].edge_index
                        mask = torch.isin(edge_index[0], startups_in_split)
                        filtered_edges = edge_index[:, mask]
                        if filtered_edges.shape[1] > 0:
                            s_idx = filtered_edges[0]
                            i_idx = filtered_edges[1]
                            scores = data.investor_scores[i_idx]
                            agg_sum.index_add_(0, s_idx, scores)
                            agg_count.index_add_(0, s_idx, torch.ones_like(scores))
                            try:
                                from torch_scatter import scatter_max
                                max_vals, _ = scatter_max(scores, s_idx, dim=0, dim_size=data['startup'].num_nodes)
                                agg_max = torch.maximum(agg_max, max_vals)
                            except ImportError:
                                pass
                            mapped_any = True

                if mapped_any:
                    mean_score = agg_sum / agg_count.clamp(min=1.0)
                    all_edge_features.append(mean_score.unsqueeze(1))
                    feature_names.append("smart_money_mean")
                    all_edge_features.append(agg_max.unsqueeze(1))
                    feature_names.append("smart_money_max")
                    print(f"  [Smart Money] Added mean/max investor success scores.")

            # --- Concatenate and store ---
            if all_edge_features:
                split_features = torch.cat(all_edge_features, dim=1)
                if split_features.shape[0] > 0:
                    non_zeros = torch.count_nonzero(split_features)
                    total_elements = split_features.numel()
                    print(f"  [{split}] {node_type}: Shape={split_features.shape}, "
                          f"Non-zeros={non_zeros}/{total_elements} ({non_zeros/total_elements*100:.1f}%)")
            else:
                split_features = torch.empty(num_nodes, 0, dtype=torch.float32)

            split_features_key = f"x_{split}"
            current_features = getattr(data[node_type], split_features_key, data[node_type].x)

            if structural_only:
                final_features = split_features if split_features.shape[1] > 0 else split_features
            else:
                final_features = torch.cat([current_features, split_features], dim=1)

            setattr(data[node_type], split_features_key, final_features)

            # Update feature labels for this split
            if not hasattr(data, f"feat_labels_{split}"):
                setattr(data, f"feat_labels_{split}", {k: v.copy() for k, v in getattr(data, "feat_labels", {}).items() if v is not None})
            split_labels = getattr(data, f"feat_labels_{split}")
            if node_type not in split_labels:
                split_labels[node_type] = []
            split_labels[node_type].extend(feature_names)

    # Update global feature registry and set base .x from first split
    for node_type in edge_info.keys():
        first_split = splits[0]
        if hasattr(data, f"feat_labels_{first_split}") and node_type in getattr(data, f"feat_labels_{first_split}"):
            original_count = len(params["feature_names"].get(node_type, []))
            all_features = getattr(data, f"feat_labels_{first_split}")[node_type]
            new_features = all_features[original_count:]

            if node_type not in params["feature_names"]:
                params["feature_names"][node_type] = []
            params["feature_names"][node_type].extend(new_features)

            if not hasattr(data, "feat_labels"):
                data.feat_labels = {}
            data.feat_labels[node_type] = params["feature_names"][node_type]

        # Set base .x to train split
        for split in splits:
            split_key = f"x_{split}"
            if hasattr(data[node_type], split_key):
                data[node_type].x = getattr(data[node_type], split_key)
                print(f"  [Final] {node_type} .x updated from {split_key}. Shape: {data[node_type].x.shape}")
                break

    return data, params


def split_data_in_train_test_val(graph_data, config, startup_df):
    split = RandomNodeSplit(
        num_val=config["data_processing"]["val"]["ratio"],
        num_test=config["data_processing"]["test"]["ratio"],
    )
    # Pin the split RNG to a hard-coded canonical value (0), independent of
    # the run's training seed. Every replication run therefore evaluates on
    # the SAME train/val/test startups (OGB convention). Not configurable.
    with torch.random.fork_rng():
        torch.manual_seed(0)
        graph_data = split(graph_data)

    graph_data["startup"].test_mask_original = graph_data["startup"].test_mask.clone()
    # Resample (if enabled)
    if config["data_processing"]["resample"]["enabled"]:
        label_col = config["data_processing"]["multi_column"] if config["data_processing"]["target_mode"] == "multi_prediction" else config["data_processing"]["binary_column"]
        print("Performing class-balanced resampling...")

        class_distribution = config["data_processing"]["resample"]["class_distribution_multi"] if config["data_processing"]["target_mode"] == "multi_prediction" else config["data_processing"]["resample"]["class_distribution_binary"]
        
        train_mask = resample_split_by_class(
            startup_df, graph_data["startup"].train_mask, label_col, class_distribution
        )
        val_mask = resample_split_by_class(
            startup_df, graph_data["startup"].val_mask, label_col, class_distribution
        )
        test_mask = resample_split_by_class(
            startup_df, graph_data["startup"].test_mask, label_col, class_distribution
        )

        # Assign updated masks
        graph_data["startup"].train_mask = train_mask
        graph_data["startup"].val_mask = val_mask
        graph_data["startup"].test_mask = test_mask

    # Force case study startup into test set
    case_study_uuid = config.get("eval", {}).get("case_study_uuid")
    if case_study_uuid:
        if "startup_uuid" in startup_df.columns:
             matches = np.where(startup_df["startup_uuid"] == case_study_uuid)[0]
             if len(matches) > 0:
                 node_idx = matches[0]
                 print(f"✅ Forcing Case Study Startup ({case_study_uuid}) at index {node_idx} into TEST set.")
                 
                 graph_data["startup"].train_mask[node_idx] = False
                 graph_data["startup"].val_mask[node_idx] = False
                 graph_data["startup"].test_mask[node_idx] = True
                 # Ensure it's in the original test mask too if that's used for reference
                 if hasattr(graph_data["startup"], "test_mask_original"):
                     graph_data["startup"].test_mask_original[node_idx] = True
             else:
                 print(f"⚠️ Case Study Startup ({case_study_uuid}) not found in startup_df during splitting.")
        else:
             print("⚠️ 'startup_uuid' column not found in startup_df. Cannot force case study split.")

    print(f"  Train: {graph_data['startup'].train_mask.sum().item()} nodes")
    print(f"  Val:   {graph_data['startup'].val_mask.sum().item()} nodes")
    print(
        f"  Test Original:  {graph_data['startup'].test_mask_original.sum().item()} nodes"
    )
    print(f"  Test:  {graph_data['startup'].test_mask.sum().item()} nodes\n")

    # Convert masks to numpy for indexing
    train_df = startup_df[graph_data["startup"].train_mask.numpy()]
    val_df = startup_df[graph_data["startup"].val_mask.numpy()]
    test_original_df = startup_df[graph_data["startup"].test_mask_original.numpy()]
    test_df = startup_df[graph_data["startup"].test_mask.numpy()]

    # Call success_ratio with DataFrame
    print(f"Success ratio (Train): {success_ratio(train_df, config)}")
    print(f"Success ratio (Val):   {success_ratio(val_df, config)}")
    print(f"Success ratio (Test Original): {success_ratio(test_original_df, config)}")
    print(f"Success ratio (Test):  {success_ratio(test_df, config)}\n")

    return graph_data


def prepare_node_features(graph_data):
    """
    Prepare node features for imputation and normalization.
    - Adds dummy features to empty node types.
    - Encodes categorical string columns.
    - Ensures all features are torch.FloatTensors.
    """
    print("Preparing node features...")

    for node_type in graph_data.node_types:
        node_data = graph_data[node_type]
        x = node_data.x

        # 1. Handle Node Types with No Features
        # if x.shape[1] == 0:
        #     print(f"Node type '{node_type}' has no features. Adding a dummy feature.")
        #     graph_data[node_type].x = torch.zeros(
        #         (node_data.num_nodes, 1), dtype=torch.float32
        #     )
        #     # Add a dummy feature name as well
        #     graph_data.feat_labels[node_type] = [f"{node_type}_dummy_feature"]
        #     continue

        # 2. Convert x to a DataFrame
        if isinstance(x, torch.Tensor):
            x_np = x.cpu().numpy()
        elif isinstance(x, np.ndarray):
            x_np = x
        else:
            raise TypeError(
                f"Unsupported feature type for node_type '{node_type}': {type(x)}"
            )

        x_df = pd.DataFrame(x_np)

        # 3. Replace inf values with NaN
        x_df.replace([np.inf, -np.inf], np.nan, inplace=True)

        # 4. Convert back to float tensor
        graph_data[node_type].x = torch.tensor(x_df.to_numpy(), dtype=torch.float32)

    return graph_data
