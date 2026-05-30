"""
Metapath Discovery Module

Implements automatic and manual metapath discovery for heterogeneous graphs.
Supports three modes:
- Manual: Use predefined metapaths (existing behavior)
- Automatic: Brute-force enumeration via matrix composition
- Hybrid: Combine manual + automatic

Based on SeHGNN paper: https://arxiv.org/abs/2207.02547
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from torch_geometric.data import HeteroData
from torch_geometric.utils import to_scipy_sparse_matrix, from_scipy_sparse_matrix
import scipy.sparse as sp
from collections import defaultdict


class MetapathDiscovery:
    """
    Handles metapath discovery for heterogeneous graphs.
    
    Attributes:
        config: Configuration dictionary
        data: HeteroData object
        mode: Discovery mode ('manual', 'automatic', 'hybrid')
    """
    
    def __init__(self, config: dict, data: HeteroData):
        """
        Initialize metapath discovery.
        
        Args:
            config: Configuration dictionary with metapath_discovery section
            data: HeteroData object containing graph structure
        """
        self.config = config
        self.data = data
        self.mode = config['metapath_discovery']['mode']
        self.target_node = config['metapath_discovery']['automatic']['target_node']
        
        print(f"\n{'='*60}")
        print(f"METAPATH DISCOVERY MODE: {self.mode.upper()}")
        print(f"{'='*60}")
    
    def discover(self) -> Dict[str, dict]:
        """
        Main entry point for metapath discovery.
        
        Returns:
            Dictionary of metapath configurations:
            {
                'metapath_name': {
                    'def': [(src, rel, dst), ...],
                    'walks': int,
                    'edge_index': Optional[torch.Tensor]
                }
            }
        """
        if self.mode == 'manual':
            return self._discover_manual()
        elif self.mode == 'automatic':
            return self._discover_automatic()
        elif self.mode == 'hybrid':
            return self._discover_hybrid()
        else:
            raise ValueError(f"Unknown metapath discovery mode: {self.mode}")
    
    def _discover_manual(self) -> Dict[str, dict]:
        """
        Manual metapath discovery (existing behavior).
        Returns empty dict - actual definitions are in graph_assembler.py
        """
        print("Using manually defined metapaths from graph_assembler.py")
        return {}  # Signal to use existing manual definitions
    
    def _discover_automatic(self) -> Dict[str, dict]:
        """
        Automatic metapath discovery via brute-force enumeration.
        
        Algorithm:
        1. Start with base edges (1-hop)
        2. Iteratively compose adjacency matrices for k=2..max_hops
        3. Filter to paths ending at target_node
        4. Apply pruning rules
        5. Convert to metapath config format
        """
        auto_config = self.config['metapath_discovery']['automatic']
        max_hops = auto_config['max_hops']
        
        print(f"Enumerating ALL metapaths up to {max_hops} hops...")
        
        # Step 1: Enumerate all metapaths
        metapaths = self._enumerate_metapaths(max_hops)
        print(f"  Generated {len(metapaths)} raw metapaths")
        
        # Step 2: Filter and prune
        filtered = self._filter_metapaths(metapaths)
        print(f"  After filtering: {len(filtered)} metapaths")
        
        # Step 3: Convert to config format
        metapaths_config = self._convert_to_config_format(filtered)
        
        print(f"✅ Automatic discovery complete: {len(metapaths_config)} metapaths")
        return metapaths_config
    
    def _discover_hybrid(self) -> Dict[str, dict]:
        """
        Hybrid discovery: manual + automatic.
        """
        hybrid_config = self.config['metapath_discovery']['hybrid']
        
        metapaths_config = {}
        
        # Add manual metapaths
        if hybrid_config['use_manual']:
            print("Including manual metapaths...")
            # Signal to include manual definitions
            metapaths_config['__use_manual__'] = True
        
        # Add automatic metapaths (with shorter max_hops)
        if hybrid_config['add_automatic']:
            print(f"Adding automatic metapaths (max_hops={hybrid_config['automatic_max_hops']})...")
            
            # Temporarily override max_hops
            original_max_hops = self.config['metapath_discovery']['automatic']['max_hops']
            self.config['metapath_discovery']['automatic']['max_hops'] = hybrid_config['automatic_max_hops']
            
            auto_paths = self._discover_automatic()
            
            # Restore original
            self.config['metapath_discovery']['automatic']['max_hops'] = original_max_hops
            
            # Prefix automatic paths to avoid name collisions
            for name, config in auto_paths.items():
                metapaths_config[f"auto_{name}"] = config
        
        print(f"✅ Hybrid discovery complete")
        return metapaths_config
    
    def _enumerate_metapaths(self, max_hops: int) -> Dict[str, dict]:
        """
        Enumerate all metapaths via iterative composition with reachability pruning.
        """
        edge_types = self.data.edge_types
        min_edges = self.config['metapath_discovery']['automatic']['min_edges']
        top_k = self.config['metapath_discovery']['automatic'].get('prune_top_k', 0)
        
        exclude_types = set(self.config['metapath_discovery']['automatic'].get('exclude_node_types', []))
        if exclude_types:
            print(f"  🚫 Excluding paths through hub nodes: {exclude_types}")
        if top_k > 0:
            print(f"  ✂️ Top-K Pruning Active: keeping max {top_k} paths per node")
        
        # Pre-compute reachability map: valid_end_types[k] = set of types that can reach target in <= k hops
        valid_end_types = self._compute_reachability_map(max_hops)
        
        # Initialize with 1-hop paths (base edges)
        metapaths = {}
        
        print(f"  Hop 1: Collecting base edges...")
        for src, rel, dst in edge_types:
            hops_remaining = max_hops - 1
            if dst not in valid_end_types[hops_remaining]:
                continue
            if dst in exclude_types and dst != self.target_node:
                continue

            path_id = f"{src}_{rel}_{dst}"
            edge_index = self.data[src, rel, dst].edge_index
            if edge_index.shape[1] < min_edges:
                continue

            num_src = self.data[src].num_nodes
            num_dst = self.data[dst].num_nodes
            adj_matrix = self._edge_index_to_sparse(edge_index, num_src, num_dst)
            
            # Prune Top-K immediately for base edges if needed (e.g. dense base edges)
            if top_k > 0 and adj_matrix.nnz > 0:
                 adj_matrix = self._prune_top_k_neighbors(adj_matrix, top_k)

            metapaths[path_id] = {
                'path': [(src, rel, dst)],
                'length': 1,
                'edge_count': adj_matrix.nnz, # Update count after prune
                'adj_matrix': adj_matrix,
                'start_type': src,
                'end_type': dst
            }
        
        print(f"    Found {len(metapaths)} candidate 1-hop paths capable of reaching target")
        
        # Iteratively compose
        for hop in range(2, max_hops + 1):
            print(f"  Hop {hop}: Composing metapaths...")
            new_paths = {}
            hops_remaining = max_hops - hop
            
            for path_id, path_data in metapaths.items():
                if path_data['length'] != hop - 1:
                    continue
                
                path_end = path_data['end_type']
                
                for src, rel, dst in edge_types:
                    if src != path_end: continue
                    if dst not in valid_end_types[hops_remaining]: continue
                    if dst in exclude_types and dst != self.target_node: continue

                    edge_index = self.data[src, rel, dst].edge_index
                    if edge_index.shape[1] < min_edges: continue

                    num_src = self.data[src].num_nodes
                    num_dst = self.data[dst].num_nodes
                    adj_new = self._edge_index_to_sparse(edge_index, num_src, num_dst)
                    
                    try:
                        new_adj = adj_new @ path_data['adj_matrix']
                    except Exception as e:
                        print(f"    ⚠️ Skipping path {path_id} -> {dst} due to error: {e}")
                        continue
                    
                    # Apply Top-K Pruning to result
                    if top_k > 0 and new_adj.nnz > 0:
                        new_adj = self._prune_top_k_neighbors(new_adj, top_k)
                    
                    if new_adj.nnz < min_edges:
                        continue
        
                    new_path_id = f"{path_id}_{rel}_{dst}"
                    new_path = path_data['path'] + [(src, rel, dst)]
                    path_str = " -> ".join([f"{p[0]}({p[1]})" for p in new_path] + [new_path[-1][2]])
                    print(f"    ✓ Found: {path_str} ({new_adj.nnz} edges)")
                    
                    new_paths[new_path_id] = {
                        'path': new_path,
                        'length': hop,
                        'edge_count': new_adj.nnz,
                        'adj_matrix': new_adj,
                        'start_type': path_data['start_type'],
                        'end_type': dst
                    }
            
            metapaths.update(new_paths)
            print(f"    Generated {len(new_paths)} new {hop}-hop paths")
        
        final_metapaths = {
            pid: pdata for pid, pdata in metapaths.items() 
            if pdata['end_type'] == self.target_node
        }
        return final_metapaths

    def _prune_top_k_neighbors(self, adj: sp.csr_matrix, k: int) -> sp.csr_matrix:
        """
        Keep only top K largest elements per row (destination node).
        For matrix A[dest, src], this keeps the top K contributors from src for each dest.
        """
        if k <= 0: return adj

        data = adj.data
        indices = adj.indices
        indptr = adj.indptr
        
        new_data = []
        new_indices = []
        new_indptr = [0]
        
        # Row-wise iteration
        for i in range(adj.shape[0]): # For each destination node
            row_start = indptr[i]
            row_end = indptr[i+1]
            
            if row_start == row_end:
                new_indptr.append(new_indptr[-1])
                continue
                
            row_data = data[row_start:row_end]
            row_indices = indices[row_start:row_end]
            
            if len(row_data) > k:
                # Find indices of top k elements
                # argpartition puts top k at the end
                top_k_idx = np.argpartition(row_data, -k)[-k:]
                
                new_data.extend(row_data[top_k_idx])
                new_indices.extend(row_indices[top_k_idx])
                new_indptr.append(new_indptr[-1] + k)
            else:
                new_data.extend(row_data)
                new_indices.extend(row_indices)
                new_indptr.append(new_indptr[-1] + len(row_data))
                
        # Reconstruct CSR
        return sp.csr_matrix((new_data, new_indices, new_indptr), shape=adj.shape)

    def _compute_reachability_map(self, max_hops: int) -> Dict[int, set]:
        """
        Compute map of node types to valid remaining hops.

        valid_end_types[k] = {types that can reach target in <= k hops}
        """
        # reachable[k] = exact k hops from target (reverse)
        reachable =  defaultdict(set)
        reachable[0] = {self.target_node}
        
        for k in range(1, max_hops + 1):
            for src, rel, dst in self.data.edge_types:
                # Reverse direction: if dst CAN reach target in k-1 steps,
                # then src CAN reach target in k steps.
                if dst in reachable[k-1]:
                    reachable[k].add(src)
        
        # Consolidate: <= k hops
        valid_end_types = {}
        for k in range(max_hops):
            # For remaining hops = k, we need to reach target in j steps where j <= k,
            # so take the union of reachable[0]...reachable[k].
            
            valid = set()
            for j in range(k + 1):
                valid.update(reachable[j])
            valid_end_types[k] = valid
            
        return valid_end_types

    
    def _edge_index_to_sparse(self, edge_index: torch.Tensor, 
                              num_src: int, num_dst: int) -> sp.csr_matrix:
        """
        Convert edge_index to scipy sparse matrix.
        
        Args:
            edge_index: [2, num_edges] tensor where edge_index[0] = src, edge_index[1] = dst
            num_src: Number of source nodes
            num_dst: Number of destination nodes
            
        Returns:
            Sparse matrix of shape [num_dst, num_src] for left multiplication
            (rows = destination, cols = source)
        """
        if edge_index.shape[1] == 0:
            return sp.csr_matrix((num_dst, num_src))
        
        # edge_index[0] = source, edge_index[1] = destination
        src = edge_index[0].cpu().numpy()
        dst = edge_index[1].cpu().numpy()
        data = np.ones(len(src))
        
        # Create sparse matrix: rows=dst, cols=src
        # This allows: result[i,j] = 1 if there's an edge from j to i
        adj = sp.csr_matrix((data, (dst, src)), shape=(num_dst, num_src))
        
        return adj
    
    def _filter_metapaths(self, metapaths: Dict[str, dict]) -> Dict[str, dict]:
        """
        Apply filtering rules to reduce metapath count.
        """
        auto_config = self.config['metapath_discovery']['automatic']
        filtered = {}
        
        for path_id, path_data in metapaths.items():
            # Filter 1: Minimum edges
            if path_data['edge_count'] < auto_config['min_edges']:
                continue
            
            # Filter 2: Self-loops (A->B->A when A->A exists)
            if auto_config['prune_self_loops']:
                if self._is_self_loop_redundant(path_data):
                    continue
            
            # Filter 3: Redundancy
            if auto_config['prune_redundant']:
                if self._is_redundant(path_data, filtered):
                    continue
            
            filtered[path_id] = path_data
        
        # Filter 4: Cap total count
        if len(filtered) > auto_config['max_metapaths']:
            strategy = auto_config.get('selection_strategy', 'most_edges')
            print(f"  Selection Strategy: {strategy}")

            if strategy in ('top_homophily', 'top_heterophily'):
                filtered = self._select_by_homophily(filtered, auto_config, strategy)
            else:
                reverse_sort = (strategy == 'most_edges')
                sorted_paths = sorted(
                    filtered.items(),
                    key=lambda x: x[1]['edge_count'],
                    reverse=reverse_sort
                )
                filtered = dict(sorted_paths[:auto_config['max_metapaths']])

            print(f"  Capped to {len(filtered)} paths based on {strategy}")

        return filtered
    
    def _compute_metapath_homophily(self, path_data: dict, labels: np.ndarray) -> float:
        """
        Compute label homophily ratio for a metapath's adjacency matrix.

        Homophily = fraction of connected pairs that share the same label.
        Only considers startup→...→startup metapaths (same-type endpoints).

        Returns:
            Homophily ratio in [0, 1], or NaN if no valid pairs.
        """
        adj = path_data['adj_matrix']
        if adj.nnz == 0:
            return float('nan')

        # Only works for startup→startup paths (both endpoints indexed into labels)
        if path_data['start_type'] != self.target_node or path_data['end_type'] != self.target_node:
            return float('nan')

        # For masked_multi_task, labels may be [N, 4] — use first column (momentum)
        if labels.ndim > 1:
            labels = labels[:, 0]

        coo = adj.tocoo()
        src_labels = labels[coo.row]
        dst_labels = labels[coo.col]

        # Only count pairs where both labels are valid (not NaN/-1)
        valid = (src_labels >= 0) & (dst_labels >= 0)
        if valid.sum() == 0:
            return float('nan')

        same_label = (src_labels[valid] == dst_labels[valid]).sum()
        return float(same_label / valid.sum())

    def _compute_baseline_homophily(self, labels: np.ndarray) -> float:
        """
        Compute expected homophily under random connection (class prior).

        For binary labels, baseline = p^2 + (1-p)^2 where p = positive rate.
        E.g., with 80% positive: 0.8^2 + 0.2^2 = 0.68
        """
        if labels.ndim > 1:
            labels = labels[:, 0]
        valid = labels[labels >= 0]
        if len(valid) == 0:
            return 0.5
        p = (valid > 0).mean()
        return float(p ** 2 + (1 - p) ** 2)

    def _select_by_homophily(self, filtered: dict, auto_config: dict, strategy: str) -> dict:
        """
        Select metapaths by homophily deviation from baseline.

        top_homophily: keep paths with highest homophily above baseline
        top_heterophily: keep paths with lowest homophily below baseline
        """
        labels = self.data[self.target_node].y
        if isinstance(labels, torch.Tensor):
            labels = labels.cpu().numpy()

        baseline = self._compute_baseline_homophily(labels)
        print(f"  Label baseline homophily: {baseline:.4f}")

        # Compute homophily for each metapath
        scored = []
        for path_id, path_data in filtered.items():
            h = self._compute_metapath_homophily(path_data, labels)
            deviation = h - baseline if not np.isnan(h) else 0.0
            path_data['homophily'] = h
            path_data['homophily_deviation'] = deviation
            scored.append((path_id, path_data, deviation))
            print(f"    {path_id}: homophily={h:.4f}, deviation={deviation:+.4f}")

        max_paths = auto_config['max_metapaths']

        if strategy == 'top_homophily':
            # Sort by deviation descending — most homophilic first
            scored.sort(key=lambda x: x[2], reverse=True)
        else:  # top_heterophily
            # Sort by deviation ascending — most heterophilic first
            scored.sort(key=lambda x: x[2])

        return {pid: pdata for pid, pdata, _ in scored[:max_paths]}

    def _is_self_loop_redundant(self, path_data: dict) -> bool:
        """
        Check if path is A->...->A and direct A->A edge exists.
        """
        path = path_data['path']
        if len(path) <= 1:
            return False
        
        start_type = path[0][0]
        end_type = path[-1][2]
        
        if start_type != end_type:
            return False
        
        # Check if direct edge exists
        for src, rel, dst in self.data.edge_types:
            if src == start_type and dst == end_type and len(path) > 1:
                # Direct edge exists, this is redundant
                return True
        
        return False
    
    def _is_redundant(self, path_data: dict, existing: Dict[str, dict]) -> bool:
        """
        Check if path is redundant with existing paths.
        Simple heuristic: same start/end types and similar edge count.
        """
        for existing_id, existing_data in existing.items():
            if (path_data['start_type'] == existing_data['start_type'] and
                path_data['end_type'] == existing_data['end_type'] and
                path_data['length'] == existing_data['length']):
                
                # Check edge count similarity (within 10%)
                ratio = path_data['edge_count'] / max(existing_data['edge_count'], 1)
                if 0.9 <= ratio <= 1.1:
                    return True
        
        return False
    
    def _convert_to_config_format(self, metapaths: Dict[str, dict]) -> Dict[str, dict]:
        """
        Convert discovered metapaths to config format compatible with graph_assembler.
        """
        auto_config = self.config['metapath_discovery']['automatic']
        metapaths_config = {}
        
        for path_id, path_data in metapaths.items():
            # Create readable name
            path_str = "_".join([rel for _, rel, _ in path_data['path']])
            name = f"{path_data['start_type']}_to_{self.target_node}_via_{path_str}"
            
            # Truncate long names
            if len(name) > 80:
                name = f"{path_data['start_type']}_to_{self.target_node}_h{path_data['length']}_{hash(path_id) % 10000}"
            
            config = {
                'def': path_data['path'],
                'walks': 10,  # Default for random walk materialization
            }
            
            # If using composition, include pre-computed adjacency
            if auto_config['materialize_method'] == 'composition':
                # Convert scipy sparse back to edge_index
                adj = path_data['adj_matrix']
                edge_index = self._sparse_to_edge_index(adj)
                config['edge_index'] = edge_index
                config['materialize_method'] = 'composition'
            
            metapaths_config[name] = config
        
        return metapaths_config
    
    def _sparse_to_edge_index(self, adj: sp.csr_matrix) -> torch.Tensor:
        """
        Convert scipy sparse matrix to PyG edge_index.
        """
        adj_coo = adj.tocoo()
        
        # adj is [dst, src], so we need to flip to [src, dst]
        row = torch.from_numpy(adj_coo.col).long()
        col = torch.from_numpy(adj_coo.row).long()
        
        edge_index = torch.stack([row, col], dim=0)
        return edge_index
