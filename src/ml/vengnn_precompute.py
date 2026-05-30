"""Pre-computation helpers for VenGNN (Zhang et al. 2024).

Two artefacts:
- The union adjacency over all startup-startup metapaths (for Branch B's
  random walker). PathCount weights are summed.
- Random-walk samples of length L starting at every startup (Branch B).

The per-metapath adjacency + PathCount edge weights live on the
HeteroData object already (see `_add_metapath_random_walk` in
`graph_assembler.py`); Branch A reads them directly, no precompute needed.
"""
from typing import List, Tuple

import torch
from torch_geometric.utils import coalesce


def build_union_startup_adjacency(
    edge_index_dict,
    metapaths: List[Tuple[str, str, str]],
    edge_weight_dict=None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Concatenate all (startup, rel, startup) edges into one weighted graph.

    Duplicate (src, dst) pairs across metapaths are summed. Returns
    (edge_index [2, E], edge_weight [E]) coalesced. `edge_weight_dict` maps
    metapath key -> [E_m] tensor; any missing key defaults to all-ones.
    """
    # Device taken from the first edge_index; all weights are moved to match.
    first_ei = edge_index_dict[metapaths[0]]
    device = first_ei.device

    edge_indices = []
    edge_weights = []
    for mp in metapaths:
        ei = edge_index_dict[mp].to(device)
        edge_indices.append(ei)
        if edge_weight_dict is not None and mp in edge_weight_dict and edge_weight_dict[mp] is not None:
            edge_weights.append(edge_weight_dict[mp].to(device=device, dtype=torch.float))
        else:
            edge_weights.append(torch.ones(ei.size(1), dtype=torch.float, device=device))

    ei_all = torch.cat(edge_indices, dim=1)
    ew_all = torch.cat(edge_weights, dim=0)
    ei_coalesced, ew_coalesced = coalesce(ei_all, ew_all, reduce="add")
    return ei_coalesced, ew_coalesced


def _to_csr_weighted(edge_index: torch.Tensor, edge_weight: torch.Tensor, num_nodes: int):
    """Convert (edge_index, edge_weight) to weighted CSR (rowptr, col, weight)."""
    from torch_geometric.utils import sort_edge_index
    edge_index, edge_weight = sort_edge_index(edge_index, edge_weight, num_nodes=num_nodes)
    row, col = edge_index
    deg = torch.bincount(row, minlength=num_nodes)
    rowptr = torch.cat([torch.tensor([0], device=row.device), deg.cumsum(0)])
    return rowptr, col, edge_weight


def sample_random_walks(
    start_nodes: torch.Tensor,
    edge_index: torch.Tensor,
    edge_weight: torch.Tensor,
    num_nodes: int,
    walks_per_node: int,
    walk_length: int,
    seed: int = 0,
    weighted: bool = False,
) -> torch.Tensor:
    """Sample `walks_per_node` random walks of `walk_length` startup nodes.

    Returns a tensor of shape [N_start, walks_per_node, walk_length] where
    entry [i, j, k] is the k-th node of the j-th walk starting at
    start_nodes[i]. A "dead" walk (no outgoing edges) keeps the last node.

    When `weighted=False`, each neighbor is equally likely at every step
    (uniform random walk). When `weighted=True`, neighbors are sampled with
    probability proportional to `edge_weight` (PathCount), so walks are
    biased toward strongly-connected startup pairs. The paper doesn't
    explicitly prescribe either; weighted better matches the notion that
    PathCount represents relational proximity strength.
    """
    device = edge_index.device
    rowptr, col, ew = _to_csr_weighted(edge_index, edge_weight, num_nodes)

    N_start = start_nodes.size(0)
    total = N_start * walks_per_node
    out = torch.zeros(total, walk_length, dtype=torch.long, device=device)
    current = start_nodes.repeat_interleave(walks_per_node)
    out[:, 0] = current

    gen = torch.Generator(device=device)
    gen.manual_seed(seed)

    # Pre-compute cumulative edge weights per row for weighted sampling.
    # Inverse-CDF via `searchsorted` then lets us draw a neighbor for every
    # walker in parallel without per-walker loops.
    if weighted:
        cum_ew = torch.cumsum(ew, dim=0)
        # Per-row cumulative maximum = sum of that row's weights. Use
        # (cum at row_end - cum at row_start) for the draw range.

    for step in range(1, walk_length):
        row_start = rowptr[current]
        row_end = rowptr[current + 1]
        degs = row_end - row_start
        has_nbr = degs > 0

        next_nodes = current.clone()
        if has_nbr.any():
            starts = row_start[has_nbr]
            ends = row_end[has_nbr]
            counts = (ends - starts).long()
            if weighted:
                # Weighted inverse-CDF sampling, vectorized:
                #   draw u ~ Uniform(0, row_sum) per walker
                #   row cumsum slice = cum_ew[starts:ends] - cum_ew[starts-1]
                #   neighbor = searchsorted over that slice
                # We use the global cum_ew and offset by cum_ew[starts] to get the row's local CDF range.
                cum_start = torch.where(starts > 0, cum_ew[starts - 1], torch.zeros_like(cum_ew[:1]))
                cum_end = cum_ew[ends - 1]  # ends > starts since has_nbr
                row_sum = cum_end - cum_start
                u = torch.rand(counts.size(0), generator=gen, device=device) * row_sum + cum_start
                # searchsorted returns global index into cum_ew ; that IS the edge offset.
                neighbor_edge = torch.searchsorted(cum_ew, u).clamp_(max=cum_ew.size(0) - 1)
                next_nodes[has_nbr] = col[neighbor_edge]
            else:
                offsets = (
                    torch.rand(counts.size(0), generator=gen, device=device) * counts.float()
                ).long().clamp_(max=counts - 1)
                next_nodes[has_nbr] = col[starts + offsets]

        out[:, step] = next_nodes
        current = next_nodes

    return out.view(N_start, walks_per_node, walk_length)
