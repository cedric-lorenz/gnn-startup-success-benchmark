"""Graph-curation variants for the ICDM graph-sensitivity ablation.

Five variants applied as a post-build filter on the fully-materialized graph.
Each variant gets its own content-addressable cache file automatically
(graph_cache.py hashes `data_processing` which contains `graph_variant`), so
all five coexist without rebuilding.

Rationale per variant:
  G1 Full          — baseline; full heterogeneous signal including all 32
                     startup-to-startup meta-paths.
  G2 No-Sector     — drop the sector super-hub (max in-degree 34,160 on 743
                     nodes). Tests whether the near-baseline `sector_peers`
                     meta-path and the dense startup→sector messages are load-
                     bearing or noise.
  G3 Pruned        — G2 + drop the three near-baseline meta-paths on Next
                     Funding Round (`city_peers`, `alumni_network`,
                     `serial_founder` — all with |excess heterophily| < 0.025).
                     Conservative noise pruning.
  G4 Heterophily   — G2 + keep only the 20 startup-to-startup meta-paths whose
                     excess heterophily on momentum is ≥ 0.15 AND which have
                     ≥ 1,000 edges (for statistical reliability). Aggressive
                     heterophily-driven curation.
  G5 Base-Only     — G2 + drop all 32 startup-to-startup meta-paths. Tests
                     whether meta-path materialization adds value over raw
                     heterogeneous 1-hop connectivity.
"""
from __future__ import annotations
from typing import Any, Dict

# G4 — 20 meta-paths with momentum excess heterophily ≥ 0.15 AND |E| ≥ 1000.
# Measured on the full-32 graph with coworking=true, co_study=false,
# worked_with base edges stripped (2026-04-22).
G4_HETEROPHILIC_METAPATHS = frozenset({
    "late_founder_vc_employment",         # +0.261
    "late_board_employment_network",      # +0.254
    "late_board_director_network",        # +0.254
    "mid_investor_founder_coworking",     # +0.238
    "late_founder_coworking_investor",    # +0.237
    "late_portfolio_siblings",            # +0.236
    "mid_board_employment_network",       # +0.228
    "mid_board_director_network",         # +0.227
    "mid_portfolio_siblings",             # +0.224
    "mid_alumni_investor_network",        # +0.212
    "mid_founder_coworking_investor",     # +0.210
    "early_board_employment_network",     # +0.210
    "early_founder_coworking_syndicate",  # +0.210
    "early_board_director_network",       # +0.205
    "mid_investor_alumni",                # +0.201
    "mid_founder_vc_employment",          # +0.192
    "early_portfolio_siblings",           # +0.181
    "early_founder_coworking_investor",   # +0.159
    "early_investor_founder_coworking",   # +0.158
    "early_founder_vc_employment",        # +0.156
})

# G3 — near-baseline meta-paths on momentum (excess heterophily < 0.025).
# These join `sector_peers` (itself a near-baseline path, dropped by G2 as
# a sector-related path) for the conservative noise-pruning set.
G3_NEAR_BASELINE_METAPATHS = frozenset({
    "city_peers",       # +0.0001
    "alumni_network",   # +0.0220
    "serial_founder",   # −0.0004
})

# Dropping the sector NODE removes 1-hop startup↔sector edges but NOT the
# pre-materialized `sector_peers` startup↔startup metapath (a startup-startup
# edge whose endpoints are both "startup" — neither touches sector type).
# G2 semantically means "drop sector and all related paths", so we drop
# sector_peers explicitly alongside the node removal.
_SECTOR_RELATED_METAPATHS = frozenset({"sector_peers"})


GRAPH_VARIANTS: Dict[str, Dict[str, Any]] = {
    "g1_full": {
        # No post-build filters.
    },
    "g2_no_sector": {
        "drop_node_types": ["sector"],
        "drop_startup_startup_metapaths": _SECTOR_RELATED_METAPATHS,
    },
    "g3_pruned": {
        "drop_node_types": ["sector"],
        "drop_startup_startup_metapaths": (
            _SECTOR_RELATED_METAPATHS | G3_NEAR_BASELINE_METAPATHS
        ),
    },
    "g4_heterophily": {
        # keep_only already excludes sector_peers (not in G4_HETEROPHILIC_METAPATHS)
        "drop_node_types": ["sector"],
        "keep_only_startup_startup_metapaths": G4_HETEROPHILIC_METAPATHS,
    },
    "g5_base": {
        # All startup-startup metapaths removed (including sector_peers)
        "drop_node_types": ["sector"],
        "drop_all_startup_startup_metapaths": True,
    },
}


def apply_graph_variant(data, variant: str):
    """Apply the named variant's filters to a fully-materialized HeteroData.

    Called once at the end of `create_graph` (graph_assembler). Mutates `data`
    in-place and returns it. A variant of `g1_full` or `None` is a no-op.
    """
    if variant is None or variant == "g1_full":
        return data

    if variant not in GRAPH_VARIANTS:
        raise ValueError(
            f"Unknown graph_variant '{variant}'. "
            f"Choices: {sorted(GRAPH_VARIANTS.keys())}"
        )

    spec = GRAPH_VARIANTS[variant]

    # 1) Drop entire node types and all incident edges.
    for nt in spec.get("drop_node_types", []):
        if nt in data.node_types:
            # First drop edge types that touch this node type on either side.
            for et in list(data.edge_types):
                if et[0] == nt or et[2] == nt:
                    del data[et]
            del data[nt]

    # 2) Drop specific startup→startup meta-paths by name.
    for mp_name in spec.get("drop_startup_startup_metapaths", ()):
        et = ("startup", mp_name, "startup")
        if et in data.edge_types:
            del data[et]

    # 3) Keep only a specified set of startup→startup meta-paths.
    keep = spec.get("keep_only_startup_startup_metapaths")
    if keep is not None:
        for et in list(data.edge_types):
            if et[0] == "startup" and et[2] == "startup" and et[1] not in keep:
                del data[et]

    # 4) Drop all startup→startup meta-paths.
    if spec.get("drop_all_startup_startup_metapaths"):
        for et in list(data.edge_types):
            if et[0] == "startup" and et[2] == "startup":
                del data[et]

    print(f"  [graph_variant] Applied '{variant}': "
          f"{len(data.node_types)} node types, {len(data.edge_types)} edge types remaining")
    return data
