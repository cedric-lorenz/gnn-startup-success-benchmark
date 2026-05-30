"""Case study utilities for loading trained models and performing inference on individual startups."""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn.functional as F
from src.ml.train import Trainer
from src.ml.utils import load_config
from src.ml.preprocessing import perform_preprocessing
from src.ml.graph_assembler import create_graph
from src.ml.explain import explain_single_node, make_explainer, get_binary_logits, WrappedModel
from src.ml.visualize import create_pyvis_network, visualize_embedding_neighborhood
import os
import argparse
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd

def load_graph_and_model(model_path=None, graph_path=None, state_dir=None, config_overrides=None):
    # 1. Load Data and Model
    config = load_config()
    # Force settings for inference
    config["train"]["epochs"] = 0
    config["explain"]["enabled"] = False # Disable auto-explain

    # Apply config overrides (e.g. max_metapaths to match checkpoint architecture)
    if config_overrides:
        for key_path, value in config_overrides.items():
            parts = key_path.split(".")
            d = config
            for p in parts[:-1]:
                d = d.setdefault(p, {})
            d[parts[-1]] = value

    # Load persistence paths
    if state_dir is None:
        state_dir = "outputs/pipeline_state"
    if graph_path is None:
        graph_path = os.path.join(state_dir, "graph_data.pt")
    if model_path is None:
        # Prefer last_model.pt (more stable, not overwritten by sweeps)
        last_path = os.path.join(state_dir, "models", "last_model.pt")
        best_path = os.path.join(state_dir, "models", "best_model.pt")
        model_path = last_path if os.path.exists(last_path) else best_path
    
    if not os.path.exists(graph_path):
        raise FileNotFoundError(f"❌ Graph data not found at {graph_path}! Please run 'python src/main.py' first.")

    print(f"📚 Loading graph data from {graph_path}...")
    graph_data = torch.load(graph_path, weights_only=False)
    # The content-addressable cache stores (HeteroData, node_names) tuples; the
    # legacy outputs/pipeline_state/graph_data.pt is a bare HeteroData. Handle both.
    if isinstance(graph_data, tuple):
        graph_data = graph_data[0]

    print("🤖 Initializing model...")
    trainer = Trainer(graph_data=graph_data, config=config)
    
    # Move data to correct device (Model is on device, Data must be too for full-batch inference)
    print(f"   Moving data to {trainer.device}...")
    trainer.data = trainer.data.to(trainer.device)
    
    if os.path.exists(model_path):
        print(f"🔄 Found checkpoint at {model_path}, loading...")
        trainer.load_checkpoint(model_path)
    else:
        print("⚠️ No checkpoint found. Model is using random initialization!")
        
    return trainer

def collect_neighbors(data, center_idx, n_hops=1, max_per_type=50):
    """Collect n-hop neighbors of a startup node from the heterogeneous graph.

    For n_hops > 1, the per-type cap is applied with 1-hop priority: every
    1-hop neighbor that fits within the cap is kept before any 2-hop (or
    later) neighbor is added. The previous behaviour sorted all hops
    together by global index and capped from the front, which silently
    dropped high-index 1-hop nodes in favour of low-index 2-hop nodes
    and made the 2-hop ego subgraph a strict subset of the 1-hop subgraph
    on some startups (e.g. Lacework).
    """
    # neighbors_by_hop[h][ntype] is the set of nodes first discovered at hop h.
    neighbors_by_hop = {0: {"startup": {center_idx}}}
    discovered_per_type = {"startup": {center_idx}}  # union across hops, for dedup
    frontier = {"startup": {center_idx}}  # nodes to expand from at the next step

    for hop in range(n_hops):
        next_frontier = {}
        for edge_type in data.edge_types:
            src, rel, dst = edge_type
            edge_index = data[edge_type].edge_index

            # Expand from frontier nodes of type src
            if src in frontier and frontier[src]:
                src_set = frontier[src]
                for node_idx in src_set:
                    mask = edge_index[0] == node_idx
                    connected = edge_index[1][mask].unique().tolist()
                    for c in connected:
                        if c not in discovered_per_type.get(dst, set()):
                            discovered_per_type.setdefault(dst, set()).add(c)
                            next_frontier.setdefault(dst, set()).add(c)

            # Reverse direction
            if dst in frontier and frontier[dst]:
                dst_set = frontier[dst]
                for node_idx in dst_set:
                    mask = edge_index[1] == node_idx
                    connected = edge_index[0][mask].unique().tolist()
                    for c in connected:
                        if c not in discovered_per_type.get(src, set()):
                            discovered_per_type.setdefault(src, set()).add(c)
                            next_frontier.setdefault(src, set()).add(c)

        neighbors_by_hop[hop + 1] = {t: set(s) for t, s in next_frontier.items()}
        frontier = next_frontier

    # Build per-type lists with 1-hop priority, then 2-hop, etc.
    all_types = set()
    for h_dict in neighbors_by_hop.values():
        all_types.update(h_dict.keys())

    result = {}
    for ntype in all_types:
        merged = []
        for hop in sorted(neighbors_by_hop.keys()):
            hop_set = neighbors_by_hop[hop].get(ntype, set())
            # Exclude center from startup so it never consumes a slot.
            if ntype == "startup":
                hop_set = hop_set - {center_idx}
            for idx in sorted(hop_set):
                if idx not in merged:
                    merged.append(idx)
                if len(merged) >= max_per_type:
                    break
            if len(merged) >= max_per_type:
                break
        if merged:
            result[ntype] = torch.tensor(merged, dtype=torch.long, device='cpu')

    return result


def _get_node_label(node_names_df, ntype, gidx, type_labels):
    """Look up a human-readable label for a node."""
    lbl = f"{type_labels.get(ntype, ntype)} {gidx}"
    if ntype in node_names_df:
        ndf = node_names_df[ntype]
        try:
            # Try exact "name" column first
            if "name" in ndf.columns:
                lbl = str(ndf.iloc[gidx]["name"])
            # Type-specific name columns (e.g. sector_name, city)
            else:
                name_cols = [c for c in ndf.columns if "name" in c.lower()]
                if name_cols:
                    lbl = str(ndf.iloc[gidx][name_cols[0]])
                elif "city" in ndf.columns:
                    lbl = str(ndf.iloc[gidx]["city"])
        except (IndexError, KeyError):
            pass
    return lbl


def plot_static_ego_graph(data, neighbors, node_scores_full, edge_scores_map,
                          best_node_idx, name, explain_path, metapath_names=None,
                          original_data=None, suffix="", layout="shell",
                          min_imp_threshold=1e-12):
    """Render a publication-quality static ego graph using matplotlib + networkx."""
    import matplotlib.colors as mcolors

    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 10,
        "axes.labelsize": 11,
        "axes.titlesize": 12,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
    })

    # Color scheme — matches the graph-schema TikZ (§3) and the feature-importance
    # plot palette in src/ml/explain.py exactly. Previous value of "#95a5a6"
    # for sector disagreed with the rest of the paper (grey vs brown).
    colors = {
        "startup": "#4682B4",     # steelblue
        "investor": "#FF8C00",    # darkorange
        "founder": "#228B22",     # forestgreen
        "city": "#DC143C",        # crimson
        "university": "#800080",  # purple
        "sector": "#A52A2A",      # brown
    }
    type_labels = {
        "startup": "Startup",
        "investor": "Investor",
        "founder": "Founder",
        "city": "City",
        "university": "University",
        "sector": "Sector",
    }

    # Build networkx graph
    G = nx.Graph()
    # Use original_data for node_names (clone() doesn't preserve custom attrs)
    names_source = original_data if original_data is not None else data
    node_names_df = getattr(names_source, 'node_names', {})

    # Center node
    center_id = f"startup_{best_node_idx}"
    G.add_node(center_id, ntype="startup", label=name, is_center=True)

    # Add neighbor nodes
    node_id_map = {}  # (ntype, global_idx) -> node_id
    node_id_map[("startup", best_node_idx)] = center_id

    for ntype, indices in neighbors.items():
        idx_list = indices.tolist() if torch.is_tensor(indices) else list(indices)
        for gidx in idx_list:
            if ntype == "startup" and gidx == best_node_idx:
                continue
            nid = f"{ntype}_{gidx}"
            lbl = _get_node_label(node_names_df, ntype, gidx, type_labels)
            G.add_node(nid, ntype=ntype, label=lbl, is_center=False)
            node_id_map[(ntype, gidx)] = nid

    # Add edges from the actual graph data
    for edge_type in data.edge_types:
        src_type, rel, dst_type = edge_type
        edge_index = data[edge_type].edge_index

        for s, d in zip(edge_index[0].tolist(), edge_index[1].tolist()):
            src_id = node_id_map.get((src_type, s))
            dst_id = node_id_map.get((dst_type, d))
            if src_id and dst_id and src_id != dst_id and G.has_node(src_id) and G.has_node(dst_id):
                score = edge_scores_map.get(edge_type, 0.5)
                G.add_edge(src_id, dst_id, rel=rel, score=score)

    # Remove isolated nodes (no edges to center's subgraph)
    isolates = list(nx.isolates(G))
    G.remove_nodes_from(isolates)

    print(f"   Graph has {len(G.nodes())} nodes, {len(G.edges())} edges")

    # Compute node importance scores
    max_score = 1e-10
    node_importance = {}
    for nid in G.nodes():
        ntype = G.nodes[nid]["ntype"]
        gidx = int(nid.split("_")[-1])
        if ntype in node_scores_full:
            scores = node_scores_full[ntype]
            if gidx < len(scores):
                imp = abs(scores[gidx].item())
            else:
                imp = 0.0
        else:
            imp = 0.0
        node_importance[nid] = imp
        if imp > max_score:
            max_score = imp

    # Filter strategy depends on hop depth:
    #   * 1-hop is dense; the most-important neighbors carry the signal and
    #     the rest just add clutter. Keep top-K (= labeling budget) so
    #     every visible node also gets a label.
    #   * 2-hop is sparse on raw IG magnitude (signal dilutes through more
    #     relations); the user-relevant question is "any 2-hop node with
    #     non-zero IG", so drop only exact-zero attributions.
    is_two_hop = "_2hop" in suffix
    if is_two_hop:
        drop_low = [
            n for n, imp in node_importance.items()
            if not G.nodes[n].get("is_center")
            and imp < min_imp_threshold
        ]
        filter_label = f"imp<{min_imp_threshold:g}"
    else:
        TOP_K = 15
        sorted_by_importance = sorted(
            [(n, i) for n, i in node_importance.items()
             if not G.nodes[n].get("is_center")],
            key=lambda x: x[1], reverse=True,
        )
        keep_neighbors = {n for n, imp in sorted_by_importance[:TOP_K]
                          if imp >= min_imp_threshold}
        keep_set = {center_id} | keep_neighbors
        drop_low = [n for n in list(G.nodes()) if n not in keep_set]
        filter_label = f"top-{TOP_K}"
    if drop_low:
        G.remove_nodes_from(drop_low)
        for nid in drop_low:
            node_importance.pop(nid, None)
        # Re-prune any neighbors left dangling after the removal.
        new_isolates = list(nx.isolates(G))
        G.remove_nodes_from(new_isolates)
        for nid in new_isolates:
            node_importance.pop(nid, None)
        print(f"   Dropped {len(drop_low)} low-importance neighbors"
              f" ({filter_label} filter; +{len(new_isolates)} newly-isolated)")
        print(f"   Graph now has {len(G.nodes())} nodes, {len(G.edges())} edges")

    # Two layouts available, controlled by the ``layout`` argument:
    #   * "shell" — concentric rings (center, 1-hop, 2-hop). Deterministic,
    #     no node overlap, easy to compare across panels.
    #   * "spring" — force-directed Fruchterman-Reingold. Reflects edge
    #     topology better, can position related nodes near each other,
    #     but occasionally collides on dense subgraphs.
    n_nodes = len(G.nodes())
    if layout == "shell":
        G_undirected = G.to_undirected()
        try:
            hop_distance = nx.single_source_shortest_path_length(
                G_undirected, center_id)
        except nx.NodeNotFound:
            hop_distance = {center_id: 0}
        shell_inner = [center_id]
        shell_1hop = [n for n, d in hop_distance.items() if d == 1]
        shell_2hop = [n for n, d in hop_distance.items() if d >= 2]
        shells = [shell_inner, shell_1hop]
        if shell_2hop:
            shells.append(shell_2hop)
        pos = nx.shell_layout(G, nlist=shells, scale=1.5)
    elif layout == "spring":
        k_spacing = 1.2 / (n_nodes ** 0.4 + 1) if n_nodes > 10 else 0.8
        pos = nx.spring_layout(
            G, k=k_spacing, iterations=300, seed=7,
            fixed=[center_id], pos={center_id: (0, 0)},
            scale=0.6,
        )
    else:
        raise ValueError(f"Unknown layout '{layout}', expected 'shell' or 'spring'")

    fig, ax = plt.subplots(figsize=(14, 12))

    # Draw edges. Uniform width and alpha for all edges; per-edge weighting
    # by edge IG score added more visual noise than signal and made it
    # hard to compare connectivity across panels of the case-study figure.
    nx.draw_networkx_edges(G, pos, ax=ax, width=2.0,
                           edge_color=(0.5, 0.5, 0.5, 0.5))

    # Draw nodes by type — bigger sizes
    for ntype in set(nx.get_node_attributes(G, "ntype").values()):
        nodes_of_type = [n for n in G.nodes() if G.nodes[n]["ntype"] == ntype]
        if not nodes_of_type:
            continue

        sizes = []
        node_colors_list = []
        for n in nodes_of_type:
            imp = node_importance[n] / max_score if max_score > 1e-10 else 0.0

            if G.nodes[n].get("is_center"):
                sizes.append(24000)
            else:
                # Base 6000, scale up to 18000 by importance
                sizes.append(6000 + imp ** 0.8 * 12000)

            node_colors_list.append(colors.get(ntype, "#95a5a6"))

        nx.draw_networkx_nodes(G, pos, nodelist=nodes_of_type, node_size=sizes,
                              node_color=node_colors_list, ax=ax,
                              edgecolors='white', linewidths=1.5, alpha=1.0)

    # Every visible node gets a name label and an IG score annotation. The
    # filter above already capped the number of nodes shown, so we don't
    # need a separate labeling-budget rule. If a node has no human-readable
    # name in the metadata, fall back to "<type> <index>" so the node is
    # never left blank in the figure.
    label_nodes = set(G.nodes())

    labels = {}
    for nid in label_nodes:
        lbl = G.nodes[nid].get("label") or ""
        if not lbl:
            ntype = G.nodes[nid].get("ntype", "node")
            idx = nid.split("_")[-1]
            lbl = f"{ntype.capitalize()} {idx}"
        if len(lbl) > 22:
            lbl = lbl[:20] + "..."
        labels[nid] = lbl

    nx.draw_networkx_labels(G, pos, labels, font_size=15, font_family="serif",
                           ax=ax, font_weight="bold")

    # Draw importance score annotations below each node
    for nid in label_nodes:
        imp = node_importance.get(nid, 0.0)
        x, y = pos[nid]
        # Format as plain decimal, not scientific notation
        if imp >= 0.01:
            score_str = f"{imp:.3f}"
        elif imp > 0:
            score_str = f"{imp:.4f}"
        else:
            score_str = "0"
        ax.annotate(score_str, (x, y), textcoords="offset points",
                   xytext=(0, -10), ha="center", va="top",
                   fontsize=16, fontstyle="italic", color="black",
                   fontfamily="serif")

    # Legend
    legend_handles = []
    present_types = set(nx.get_node_attributes(G, "ntype").values())
    for ntype in ["startup", "investor", "founder", "city", "university", "sector"]:
        if ntype in present_types:
            legend_handles.append(mpatches.Patch(
                color=colors.get(ntype, "#95a5a6"),
                label=type_labels.get(ntype, ntype)
            ))
    ax.legend(handles=legend_handles, loc="upper left", frameon=True, framealpha=0.9,
              edgecolor='#cccccc', fontsize=13)

    ax.axis("off")

    # Expand axis limits so labels don't clip. Guard against the edge case where
    # importance filtering left the graph empty (no nodes / no positions) — on
    # weakly-trained synthetic models the attribution threshold can drop every
    # neighbor. Skip rather than crash on max() of an empty sequence.
    x_vals = [p[0] for p in pos.values()]
    y_vals = [p[1] for p in pos.values()]
    if not x_vals or not y_vals:
        plt.close(fig)
        print(f"   ⚠️ skipping ego graph{suffix}: no nodes survived "
              f"importance filtering (graph empty)")
        return None
    x_margin = (max(x_vals) - min(x_vals)) * 0.25
    y_margin = (max(y_vals) - min(y_vals)) * 0.15
    ax.set_xlim(min(x_vals) - x_margin, max(x_vals) + x_margin)
    ax.set_ylim(min(y_vals) - y_margin, max(y_vals) + y_margin)

    for spine in ax.spines.values():
        spine.set_visible(False)

    out_path = f"{explain_path}/startup_{best_node_idx}_ego_graph{suffix}.pdf"
    fig.savefig(out_path, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"   Saved static ego graph to {out_path}")
    return out_path


def main():
    parser = argparse.ArgumentParser(description="Run Voize case study")
    parser.add_argument("--model-path", type=str, default=None,
                        help="Path to model checkpoint")
    parser.add_argument("--graph-path", type=str, default=None,
                        help="Path to graph data .pt file")
    parser.add_argument("--max-metapaths", type=int, default=None,
                        help="Override max_metapaths to match checkpoint architecture")
    parser.add_argument("--num-hops", type=int, default=None,
                        help="Override SeHGNN num_hops to match checkpoint")
    parser.add_argument("--target-uuid", type=str, default=None,
                        help="Override target startup UUID (defaults to Voize)")
    parser.add_argument("--champion-config", type=str, default=None,
                        help="Path to a champion YAML (e.g. "
                             "experiments/champion_configs/sehgnn_g1_full.yaml). "
                             "Every leaf is converted to a dot-notation override "
                             "so the rebuilt model matches the checkpoint's "
                             "architecture exactly.")
    parser.add_argument("--ig-n-steps", type=int, default=None,
                        help="Override the number of IG interpolation steps. "
                             "Default is Captum's 50; reduce (e.g. 20) when "
                             "the IG forward pass OOMs on dense ego graphs.")
    parser.add_argument("--attribution-method",
                        choices=["integrated_gradients", "gradient_shap"],
                        default="integrated_gradients",
                        help="Per-instance attribution method. "
                             "'integrated_gradients' (default) preserves the "
                             "original case-study pipeline; 'gradient_shap' "
                             "matches the §VI.A Expected-Gradients population "
                             "figure by averaging --eg-n native GradientShap "
                             "sub-runs with --eg-sigma noise against baselines "
                             "sampled from the train mask via --eg-seed.")
    parser.add_argument("--eg-n", type=int, default=100,
                        help="Monte Carlo samples for Expected Gradients. "
                             "Only used when --attribution-method gradient_shap.")
    parser.add_argument("--eg-sigma", type=float, default=0.1,
                        help="Gaussian input noise std for Expected Gradients. "
                             "Only used when --attribution-method gradient_shap.")
    parser.add_argument("--eg-seed", type=int, default=42,
                        help="RNG seed for EG baseline sampling and the "
                             "per-iter torch.manual_seed used by GradientShap.")
    args = parser.parse_args()

    print("🚀 Starting Single Startup Case Study...")

    def _flatten(d, prefix=""):
        out = {}
        for k, v in d.items():
            key = f"{prefix}.{k}" if prefix else k
            if isinstance(v, dict):
                out.update(_flatten(v, key))
            else:
                out[key] = v
        return out

    config_overrides = {}
    if args.champion_config:
        import yaml
        with open(args.champion_config) as f:
            config_overrides.update(_flatten(yaml.safe_load(f) or {}))
    if args.max_metapaths is not None:
        config_overrides["metapath_discovery.automatic.max_metapaths"] = args.max_metapaths
    if args.num_hops is not None:
        config_overrides["models.SeHGNN.num_hops"] = args.num_hops

    try:
        trainer = load_graph_and_model(
            model_path=args.model_path,
            graph_path=args.graph_path,
            config_overrides=config_overrides or None,
        )
    except Exception as e:
        print(e)
        return

    config = trainer.config

    # Run test evaluation to get predictions on test set
    print("📊 Generating predictions...")
    trainer.model.eval()

    # We need predictions for ALL startups to find a candidate
    # Use the test set (masked)
    data = trainer.data

    # Match Trainer.train() flow exactly so the case-study prediction equals
    # the value the Evaluator wrote into the prediction CSV:
    #   1. precompute() the SeHGNN cache on TRAIN-view features (data.x).
    #   2. swap data['startup'].x -> x_test_mask (eval.py line 830).
    #   3. forward — neighbor channels read the train-view cache; self+residual
    #      read the post-swap test-view x. This mixed regime is what training
    #      saw at eval time and what `test_*` metrics in W&B/CSVs reflect.
    if hasattr(trainer.model, "precompute"):
        print("🧮 Pre-aggregating SeHGNN neighbor cache on train-view features")
        trainer.model.precompute(data.x_dict, data.edge_index_dict)

    if hasattr(data["startup"], "x_test_mask"):
        print("🔄 Swapping features to x_test_mask for test-time evaluation")
        data["startup"].x = data["startup"].x_test_mask

    with torch.no_grad():
        out = trainer.model(data.x_dict, data.edge_index_dict)
        
        # Determine output format
        if isinstance(out, dict) and "out_mom" in out:
             # SeHGNN Masked Multi Task format
             mom_out = out["out_mom"] # [N] (logits)
             liq_out = out["out_liq"] # [N] (logits)
             
             mom_probs = torch.sigmoid(mom_out)
             liq_probs = torch.sigmoid(liq_out)
             
             # Combined score for candidate selection
             pred_probs = (mom_probs + liq_probs) / 2
        
        elif isinstance(out, dict) and "startup" in out:
             # Standard PyG Hetero format
             pred_probs = out["startup"]["output"]
             if pred_probs.size(1) > 1:
                 pred_probs = F.softmax(pred_probs, dim=1)[:, 1]
             else:
                 pred_probs = torch.sigmoid(pred_probs).squeeze()
             
             mom_probs = pred_probs
             liq_probs = pred_probs
        else:
             print("⚠️ Unknown output format, check case_study.py")
             return

    # Find interesting candidate
    print("🔎 selecting candidate startup...")
    
    # Target UUID (defaults to Voize, override via --target-uuid)
    target_uuid = args.target_uuid or "5b8450df-dfb5-4d47-9168-4918c1aba3ac"
    
    # Try to find the target startup
    if hasattr(data, 'node_names') and 'startup' in data.node_names:
        startup_names = data.node_names['startup']
        # node_names is indexed by UUID (standard pipeline behavior); check an explicit
        # uuid column first, then fall back to the index.
        
        # Try to find target UUID in index or column
        target_indices = []
        
        # Check explicit column first (safer)
        if 'startup_uuid' in startup_names.columns:
            matches = startup_names.index[startup_names['startup_uuid'] == target_uuid].tolist()
            target_indices.extend(matches)
        
        # Check index if no matches found yet
        if not target_indices and target_uuid in startup_names.index:
             loc = startup_names.index.get_loc(target_uuid)
             if isinstance(loc, int):
                 target_indices.append(loc)
             elif isinstance(loc, slice):
                 target_indices.extend(range(loc.start, loc.stop, loc.step or 1))
             else:
                 # Boolean array or similar
                 target_indices.extend(np.where(loc)[0].tolist())

        if target_indices:
             best_node_idx = target_indices[0] # Take first match
             best_prob = pred_probs[best_node_idx].item()
             name = startup_names.iloc[best_node_idx]['name'] if 'name' in startup_names.columns else "Unknown"
             print(f"🎯 Found target startup '{name}' (UUID: {target_uuid}) at index {best_node_idx}")
             print(f"   pred_mom = {mom_probs[best_node_idx].item():.6f}")
             print(f"   pred_liq = {liq_probs[best_node_idx].item():.6f}")
             print(f"   joint    = {best_prob:.6f}")
        else:
             print(f"⚠️ Target UUID {target_uuid} not found in dataset (Index or startup_uuid column). Falling back to top prediction.")
             if "startup" in data and hasattr(data["startup"], "test_mask"):
                 mask = data["startup"].test_mask
             else:
                 mask = torch.ones_like(pred_probs, dtype=torch.bool)

             candidates = torch.where(mask)[0]
             candidate_probs = pred_probs[mask]

             sorted_indices = torch.argsort(candidate_probs, descending=True)

             top_relative_idx = sorted_indices[0]
             best_node_idx = candidates[top_relative_idx].item()
             best_prob = candidate_probs[top_relative_idx].item()
             
             print(f"🚀 Top Candidate: Node {best_node_idx} with Prob: {best_prob:.4f}")

    else:
        print("⚠️ No node names available for lookup. Falling back to top prediction.")
        if "startup" in data and hasattr(data["startup"], "test_mask"):
             mask = data["startup"].test_mask
        else:
             mask = torch.ones_like(pred_probs, dtype=torch.bool)

        candidates = torch.where(mask)[0]
        candidate_probs = pred_probs[mask]

        sorted_indices = torch.argsort(candidate_probs, descending=True)

        top_relative_idx = sorted_indices[0]
        best_node_idx = candidates[top_relative_idx].item()
        best_prob = candidate_probs[top_relative_idx].item()

        print(f"🚀 Top Candidate: Node {best_node_idx} with Prob: {best_prob:.4f}")

    print(f"� Top Candidate: Node {best_node_idx} with Prob: {best_prob:.4f}")
    
    node_idx = best_node_idx
    
    task_to_explain = "momentum" # or "liquidity"
    
    if hasattr(data, 'node_names') and 'startup' in data.node_names:
        name = data.node_names['startup'].iloc[best_node_idx]['name']
        print(f"   Name: {name}")

    # 3. Explain Feature Importance
    print("\n🔍 Generating Feature Importance...")
    explain_path = "outputs/case_study"
    os.makedirs(explain_path, exist_ok=True)
    
    # Reproduce `explain_model` logic for this single node.
    explain_config = config.get("explain", {})
    method = explain_config.get("method", "integrated_gradients")
    if isinstance(method, dict):
        # Fallback if config structure is unexpected
        print(f"⚠️ Warning: config['explain']['method'] is a dict: {method}. Using 'integrated_gradients'.")
        method = "integrated_gradients"
        
    method_params = explain_config.get(method, {})
    if "attribution_method" not in method_params and method == "integrated_gradients":
         method_params["attribution_method"] = "IntegratedGradients"
    if args.ig_n_steps is not None:
        method_params["n_steps"] = args.ig_n_steps
        print(f"⚙️  IG n_steps overridden to {args.ig_n_steps}")


    model_config = dict(
        mode="binary_classification",
        task_level="node",
        return_type="probs",
    )
    
    # Captum IG/GradientShap perturbs x_dict to get gradients. The cached
    # neighbor aggregations would shadow those perturbations and zero out
    # gradients for upstream node features, so clear the cache before
    # attribution.
    if hasattr(trainer.model, "clear_cache"):
        print("🧹 Clearing SeHGNN cache so Captum sees live x_dict gradients")
        trainer.model.clear_cache()

    # Wrap model for Captum
    wrapped_model = WrappedModel(trainer.model, get_binary_logits)

    if args.attribution_method == "gradient_shap":
        # Expected Gradients via N native GradientShap sub-runs, mirroring
        # the §VI.A population pipeline (extract_sehgnn_ig_baselines.py).
        # Each sub-run uses one sampled training startup as baseline with
        # Gaussian input noise stdevs=args.eg_sigma. The per-iter node and
        # edge masks are accumulated and averaged into a synthetic
        # HeteroExplanation that the downstream ego-graph + feature CSV
        # writers consume unchanged.
        print(f"   Running Expected Gradients via {args.eg_n} native "
              f"GradientShap sub-runs (sigma={args.eg_sigma}, "
              f"seed={args.eg_seed})...")
        sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
        from extract_sehgnn_ig_baselines import (
            _sample_eg_training_indices,
            _eg_single_iteration_baseline_tuple,
        )
        from torch_geometric.explain import HeteroExplanation

        sample_idx = _sample_eg_training_indices(
            data, n=args.eg_n, seed=args.eg_seed)
        device = data["startup"].x.device
        target_index = torch.tensor([best_node_idx], device=device,
                                    dtype=torch.long)

        accum_node_masks = None
        accum_edge_masks = None
        for i, row_idx in enumerate(sample_idx.tolist()):
            torch.manual_seed(args.eg_seed + i)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(args.eg_seed + i)

            baselines_tuple = _eg_single_iteration_baseline_tuple(
                data, device, row_idx)
            iter_method_params = {
                "attribution_method": "GradientShap",
                "baselines": baselines_tuple,
                "n_samples": 1,
                "stdevs": args.eg_sigma,
            }
            iter_explainer = make_explainer(
                wrapped_model, iter_method_params, model_config)
            iter_expl = iter_explainer(
                x=data.x_dict,
                edge_index=data.edge_index_dict,
                index=target_index,
            )

            if accum_node_masks is None:
                accum_node_masks = {
                    k: v.detach().clone()
                    for k, v in iter_expl.node_mask_dict.items()
                }
                accum_edge_masks = {
                    k: v.detach().clone()
                    for k, v in iter_expl.edge_mask_dict.items()
                }
            else:
                for k in accum_node_masks:
                    accum_node_masks[k] += iter_expl.node_mask_dict[k].detach()
                for k in accum_edge_masks:
                    accum_edge_masks[k] += iter_expl.edge_mask_dict[k].detach()
            if (i + 1) % 20 == 0 or i + 1 == args.eg_n:
                print(f"     completed {i + 1}/{args.eg_n} EG sub-runs")

        for k in accum_node_masks:
            accum_node_masks[k] /= float(args.eg_n)
        for k in accum_edge_masks:
            accum_edge_masks[k] /= float(args.eg_n)

        explanation = HeteroExplanation()
        explanation.set_value_dict('node_mask', accum_node_masks)
        explanation.set_value_dict('edge_mask', accum_edge_masks)

        improved_path = (f"{explain_path}/startup_{best_node_idx}"
                         f"_feature_importance_case_study_improved.pdf")
        print(f"Saving single node feature importance plot to {improved_path}")
        from src.ml.explain import create_improved_feature_importance_plot
        create_improved_feature_importance_plot(
            explanation,
            data.feat_labels,
            improved_path,
            top_k=20,
            title=f"Expected-Gradients Attribution for {name}",
        )
    else:
        explainer = make_explainer(wrapped_model, method_params, model_config)
        print("   Running Integrated Gradients...")
        explanation = explain_single_node(
            best_node_idx,
            data,
            explainer,
            explain_path,
            mode="case_study",
            model=trainer.model,
            title=f"Feature Attribution for {name}",
        )
    
    # 4. Extract Attention Weights (SeHGNN specific)
    print("\n🧠 Extracting Attention Weights...")
    # Re-run forward pass to get internal state
    trainer.model.eval()
    with torch.no_grad():
        out_dict = trainer.model(data.x_dict, data.edge_index_dict)
    
    if "attention_weights" in out_dict:
        attn_weights = out_dict["attention_weights"] # [B, Heads, M, M] or [B, Heads, M] depending on impl
        metapath_names = out_dict.get("metapath_names", [])

        node_attn = attn_weights[best_node_idx] # [Heads, M, M]

        # SeHGNN fuses M metapath channels; attention[i, j] = how much channel i attends to j.
        # As a metapath-importance proxy, visualize the raw (head-averaged) attention matrix.

        node_attn_avg = node_attn.mean(dim=0).cpu().numpy() # [M, M]

        plt.figure(figsize=(12, 10))
        im = plt.imshow(node_attn_avg, cmap='viridis', interpolation='nearest')
        plt.colorbar(im, label="Attention Weight")
        
        plt.title(f"Metapath Interaction Matrix for {name}\n(Row i attends to Column j)", fontsize=14)
        
        # Center ticks on pixels
        tick_locs = np.arange(len(metapath_names))
        plt.xticks(tick_locs, metapath_names, rotation=90, fontsize=10)
        plt.yticks(tick_locs, metapath_names, fontsize=10)
        
        plt.tight_layout()
        plt.savefig(f"{explain_path}/startup_{best_node_idx}_attention.pdf", dpi=300)
        print(f"   Saved attention plot to {explain_path}/startup_{best_node_idx}_attention.pdf")
        
        print("\n🤔 Attention Interpretation:")
        print("   The matrix shows how different relational views (metapaths) interact.")
        print("   - High Diagonal: The metapath reinforces its own signal (strong independent feature).")
        print("   - High Off-Diagonal (Row i, Col j): Metapath i relies on context from Metapath j.")
        
        # Rows are a softmax distribution (Row i = attention FROM channel i TO others),
        # so a high column sum means many channels attend TO that metapath: column
        # sum is the "received attention" / centrality of each metapath.
        agg_importance = node_attn_avg.sum(axis=0) # Column Sum
        sorted_imp_idx = np.argsort(agg_importance)[::-1]
        
        print("\n🏆 Top Metapaths by Received Attention (Centrality):")
        for i in range(min(5, len(metapath_names))):
            idx = sorted_imp_idx[i]
            print(f"   {i+1}. {metapath_names[idx]} (Score: {agg_importance[idx]:.4f})")
            
        # Plot Bar Chart of Importance
        plt.figure(figsize=(12, 6))
        plt.bar(range(len(metapath_names)), agg_importance, color='teal')
        plt.title(f"Metapath Centrality (Received Attention) for {name}")
        plt.xticks(range(len(metapath_names)), metapath_names, rotation=90)
        plt.ylabel("Total Attention Received")
        plt.tight_layout()
        plt.savefig(f"{explain_path}/startup_{best_node_idx}_metapath_importance.pdf")
        print(f"   Saved Metapath Importance Bar Chart.")
        
    else:
        print("   (Model does not return attention weights, skipping)")
    
    # 5. Calculate Node Importance from HeteroExplanation
    # Sum feature importance for each node to get scalar node importance
    print("\n🧮 Calculating Node Importance Scores...")
    node_scores_full = {}
    
    if hasattr(explanation, 'node_mask_dict'):
        masks = explanation.node_mask_dict
        for ntype, mask_tensor in masks.items():
            # mask_tensor: [Num_Nodes, Num_Features]; sum over features for a
            # per-node scalar. Kept as raw sums (no normalization).
            scores = mask_tensor.sum(dim=1).detach().cpu()

            node_scores_full[ntype] = scores

            # Top 3 important nodes globally
            if ntype == 'investor':
                top_k = torch.topk(scores, min(3, len(scores)))
                print(f"   Top investors globally: indices={top_k.indices.tolist()}, values={top_k.values.tolist()}")

    # 5b. Prepare Edge Scores from Attention
    # We map "Metapath Importance" to "Edge Type Importance"
    # Metapaths are tuples: (src, rel, dst)
    # Edge types in PyG are also tuples: (src, rel, dst)
    # This mapping is direct for 1-hop metapaths.
    print("\n🔗 Preparing Edge Attention Scores...")
    edge_scores_map = {}
    if "attention_weights" in out_dict:
        # agg_importance has shape [Num_Metapaths], aligned with metapath_names
        for mp_idx, mp_name in enumerate(metapath_names):
            # mp_name is a tuple like ('startup', 'early_stage_funded_by', 'investor')
            # (models.py uses tuples), which doubles as the PyG edge_type key.
            score = agg_importance[mp_idx]

            # The "self" metapath is not an edge.
            if mp_name == "self": continue
            
            edge_scores_map[mp_name] = score.item() if hasattr(score, 'item') else score
            
    
    # 6. Ego Graph Visualization
    print("\n🕸️ Generating Ego Graph...")

    print("   Moving data to CPU for visualization...")
    viz_data = data.clone().to('cpu')

    # 6a/b. 1-hop and 2-hop ego graphs, each rendered in both shell and
    # spring layouts so we can compare side by side. 2-hop uses
    # max_per_type=100 (with 1-hop priority inside collect_neighbors) so
    # every 1-hop neighbour is preserved before the 2-hop expansion fills
    # remaining slots.
    neighbors_1hop = collect_neighbors(viz_data, best_node_idx, n_hops=1)
    neighbors_2hop = collect_neighbors(viz_data, best_node_idx, n_hops=2, max_per_type=100)

    # 1-hop uses 1e-4 floor (keeps top single-digit-rank neighbors,
    # drops the long tail of zero-attribution startups). 2-hop uses the
    # stricter 1e-3 floor since 2-hop has 300+ candidates and crowds.
    for hop_label, neighbors, imp_threshold in [
        ("_1hop", neighbors_1hop, 1e-4),
        ("_2hop", neighbors_2hop, 1e-3),
    ]:
        for layout in ("shell", "spring"):
            print(f"\n📄 Generating {hop_label[1:]} ego graph (PDF, {layout} layout)...")
            plot_static_ego_graph(
                data=viz_data,
                neighbors=neighbors,
                node_scores_full=node_scores_full,
                edge_scores_map=edge_scores_map,
                best_node_idx=best_node_idx,
                name=name,
                explain_path=explain_path,
                original_data=data,
                suffix=f"{hop_label}_{layout}",
                layout=layout,
                min_imp_threshold=imp_threshold,
            )

    # 7. Embedding Neighborhood Visualization (optional diagnostic, not a
    # paper figure; requires the competitor_retrieval helper, which is not
    # bundled in the paper distribution).
    try:
        from scripts.competitor_retrieval import CompetitorRetriever
    except ImportError:
        print("\n⏭  Skipping embedding-neighborhood visualization "
              "(competitor_retrieval not bundled).")
        print("\n✅ Case Study Complete!")
        return
    print("\n🌌 Generating Embedding Neighborhood Visualization...")

    # Initialize Retriever
    retriever = CompetitorRetriever(trainer)
    
    # Get Top Neighbors (Competitors)
    # top_k=200 for good density
    print("   Retrieving neighbors...")
    neighbor_indices, neighbor_scores = retriever.retrieve(best_node_idx, method='gnn', top_k=200)
    
    print("\n   🧐 Neighbor Analysis:")
    print(f"   Top 5 Neighbor Scores: {neighbor_scores[:5]}")
    print(f"   Bottom 5 Neighbor Scores: {neighbor_scores[-5:]}")
    
    if retriever.gnn_embeddings is None:
        retriever._extract_gnn_embeddings()

    sub_embs = retriever.gnn_embeddings[neighbor_indices]
    center_emb = retriever.gnn_embeddings[best_node_idx]
    
    # Calculate distance from center manually to verify
    from sklearn.metrics.pairwise import cosine_similarity
    sims = cosine_similarity(center_emb.reshape(1, -1), sub_embs)[0]
    print(f"   Re-calculated Top 5 Similarities: {sims[:5]}")
    
    # Variance check
    variance = np.var(sub_embs, axis=0).mean()
    print(f"   Neighbor Embedding Variance: {variance:.6f} (Low variance = Collapse)")
    
    # Add some random context nodes for contrast (100 random)
    all_indices = list(range(len(retriever.df)))
    random_indices = np.random.choice(all_indices, 100, replace=False)
    
    # Combine: Center + Neighbors + Random
    viz_indices = [best_node_idx] + list(neighbor_indices) + list(random_indices)
    viz_indices = list(set(viz_indices)) # Unique
    
    if retriever.gnn_embeddings is None:
        retriever._extract_gnn_embeddings()

    subset_embeddings = retriever.gnn_embeddings[viz_indices]
    
    # Build Metadata
    viz_metadata = {}
    for i, global_idx in enumerate(viz_indices):
        row = retriever.df.iloc[global_idx]
        curr_name = row.get('name', str(global_idx))
        
        # Try raw df for better metadata if available
        meta_sector = "Unknown"
        meta_status = "Unknown"
        
        if retriever.raw_df is not None:
             uid = row.get('startup_uuid')
             if not uid: uid = row.get('items_id')
             if uid and uid in retriever.raw_df.index:
                 raw_row = retriever.raw_df.loc[uid]
                 meta_sector = str(raw_row.get('industry_groups', 'Unknown'))
                 if len(meta_sector) > 20: meta_sector = meta_sector[:17] + "..."
                 meta_status = str(raw_row.get('status', 'Unknown'))
        
        viz_metadata[i] = {
            'name': curr_name,
            'sector': meta_sector,
            'status': meta_status
        }

    center_local_idx = viz_indices.index(best_node_idx)
    
    visualize_embedding_neighborhood(
        embeddings=subset_embeddings,
        node_indices=viz_indices,
        center_node_idx=center_local_idx,
        metadata=viz_metadata,
        output_path=f"{explain_path}/startup_{best_node_idx}_embedding_tsne.pdf",
        title=f"Embedding Landscape for {name}"
    )

    print("\n✅ Case Study Complete!")

if __name__ == "__main__":
    main()
