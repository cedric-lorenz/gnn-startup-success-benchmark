#!/usr/bin/env python
"""
Compute attention-homophily correlation for NFR and Exit targets.

Loads the trained 33-metapath SeHGNN checkpoint, runs a single inference pass
to extract per-metapath semantic-fusion attention weights, then computes per-
metapath edge homophily under three label variants:

  (1) h_nfr_full        — Next-Funding-Round labels on all labeled startups
  (2) h_exit_full       — Exit labels on all labeled startups
  (3) h_exit_mature     — Exit labels restricted to the mature subset

For each variant, reports Pearson correlation between the 33-dim attention
vector and the 33-dim homophily vector, and plots a 3-panel scatter.

Outputs:
  outputs/attention_homophily/attention_homophily.csv
  outputs/attention_homophily/correlation_summary.json
  outputs/attention_homophily/homophily_vs_attention_3panel.pdf
  outputs/attention_homophily/homophily_vs_attention_3panel.png
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr, spearmanr

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.ml.heterophily_metrics import calculate_edge_homophily
from src.ml.utils import get_maturity_mask, load_config


DEFAULT_CHECKPOINT = (
    PROJECT_ROOT
    / "outputs"
    / "pipeline_state"
    / "models"
    / "SeHGNN_seed42_sbgicu8c_best.pt"
)
DEFAULT_GRAPH = PROJECT_ROOT / "outputs" / "pipeline_state" / "graph_data.pt"
DEFAULT_CONFIG = PROJECT_ROOT / "config.yaml"
DEFAULT_OUTDIR = PROJECT_ROOT / "outputs" / "attention_homophily"


def _deep_merge(base: dict, overrides: dict) -> dict:
    """Recursively merge ``overrides`` into ``base`` (override wins on leaves)."""
    out = dict(base)
    for k, v in (overrides or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def _load_graph_any(path):
    """Load a graph that might be a HeteroData OR a (HeteroData, node_names) tuple."""
    obj = torch.load(path, weights_only=False, map_location="cpu")
    return obj[0] if isinstance(obj, tuple) else obj


# -------- category mapping for plot coloring ------------------------------
# Defense-deck scheme: Geographic / Industry / Academic / Professional /
# Early-/Mid-/Late-Stage investor. Mirrors scripts/generate_thesis_figures.py.

METAPATH_CATEGORIES = {
    "city_peers": "Geographic",
    "sector_peers": "Industry",
    "alumni_network": "Academic",
    "co_working_network": "Professional",
}

CATEGORY_COLORS = {
    "Geographic": "#9467bd",
    "Industry": "#7f7f7f",
    "Academic": "#17becf",
    "Professional": "#bcbd22",
    "Early-Stage": "#2ca02c",
    "Mid-Stage": "#ff7f0e",
    "Late-Stage": "#d62728",
    "Other": "#333333",
}

CATEGORY_ORDER = [
    "Geographic", "Industry", "Academic", "Professional",
    "Early-Stage", "Mid-Stage", "Late-Stage",
]

METAPATH_PRETTY = {
    "city_peers": "city_peers",
    "sector_peers": "sector_peers",
    "alumni_network": "alumni_network",
    "co_working_network": "co_working_network",
    "early_portfolio_siblings": "early_portfolio_siblings",
    "mid_portfolio_siblings": "mid_portfolio_siblings",
    "late_portfolio_siblings": "late_portfolio_siblings",
    "early_founder_vc_employment": "early_founder_vc_employment",
    "mid_founder_vc_employment": "mid_founder_vc_employment",
    "late_founder_vc_employment": "late_founder_vc_employment",
}


def categorize(rel: str) -> str:
    if rel in METAPATH_CATEGORIES:
        return METAPATH_CATEGORIES[rel]
    if rel.startswith("early_"):
        return "Early-Stage"
    if rel.startswith("mid_"):
        return "Mid-Stage"
    if rel.startswith("late_"):
        return "Late-Stage"
    return "Other"


# -------- core routines ----------------------------------------------------


def load_model(graph, config, checkpoint_path: Path, device: torch.device):
    """Instantiate the SeHGNN model and load the trained state dict."""
    from src.ml.train import Trainer

    trainer = Trainer(graph, config)
    trainer.device = device
    trainer.model.to(device)
    loaded = trainer.load_checkpoint(str(checkpoint_path))
    if not loaded:
        raise RuntimeError(f"Could not load checkpoint: {checkpoint_path}")
    trainer.model.eval()
    return trainer.model


@torch.no_grad()
def extract_attention(model, graph, device):
    """Run one forward pass and return (metapath_names, importance_vector)."""
    graph = graph.to(device)

    out = model(graph.x_dict, graph.edge_index_dict)
    if not isinstance(out, dict) or "attention_weights" not in out:
        raise RuntimeError(
            "Model forward did not expose 'attention_weights' — ensure SeHGNN "
            "build and that the forward path is the semantic-fusion branch."
        )

    attn = out["attention_weights"]  # [B, H, M, M]
    metapath_names = out["metapath_names"]  # list of (src, rel, dst) tuples
    # Mean over batch (0), heads (1), source-metapath (2) → [M]
    importance = attn.mean(dim=(0, 1, 2)).detach().cpu().numpy()
    return list(metapath_names), importance


def relation_from_metapath_name(mp) -> str:
    """SeHGNN stores metapaths as (src, relation, dst). Return relation."""
    if isinstance(mp, tuple) and len(mp) == 3:
        return mp[1]
    return str(mp)


def compute_homophily_for_labels(graph, y: torch.Tensor):
    """Return {relation: (h, num_edges)} over startup-startup edges."""
    results = {}
    for edge_type in graph.edge_types:
        src, rel, dst = edge_type
        if src != "startup" or dst != "startup":
            continue
        edge_index = graph[edge_type].edge_index
        n_edges = int(edge_index.shape[1])
        if n_edges == 0:
            results[rel] = (None, 0)
            continue
        h = calculate_edge_homophily(edge_index, y)
        # calculate_edge_homophily returns None when no edge has two valid
        # labels (possible on sparse/synthetic graphs even with n_edges > 0).
        if h is None:
            results[rel] = (None, n_edges)
            continue
        # may return a torch scalar or a python float
        if torch.is_tensor(h):
            h = h.item()
        results[rel] = (float(h), n_edges)
    return results


def build_label_variants(graph, config):
    """Return dict {variant_name: y_tensor} for the three analysis targets."""
    y = graph["startup"].y  # [N, 2]
    y_nfr = y[:, 0].clone().float()
    y_exit = y[:, 1].clone().float()

    variants = {
        "nfr_full": y_nfr,
        "exit_full": y_exit,
    }

    # Mature-subset Exit: nan-mask the non-mature rows
    mature_mask = None
    raw_df = getattr(graph["startup"], "raw_df", None)
    if raw_df is not None:
        is_mature = get_maturity_mask(raw_df, config)
        if is_mature is not None:
            mature_mask = torch.as_tensor(is_mature).bool()

    if mature_mask is None:
        print(
            "⚠️  Could not derive mature mask from graph['startup'].raw_df; "
            "skipping h_exit_mature."
        )
    else:
        y_exit_mature = y_exit.clone()
        y_exit_mature[~mature_mask] = float("nan")
        variants["exit_mature"] = y_exit_mature

    return variants


def assemble_dataframe(metapath_names, attention_vec, homophily_by_variant):
    """Join attention + homophily-by-variant into a single tidy dataframe."""
    relations = [relation_from_metapath_name(mp) for mp in metapath_names]
    df = pd.DataFrame(
        {
            "metapath": relations,
            "attention_weight": attention_vec,
            "category": [categorize(r) for r in relations],
        }
    )
    for variant_name, hmap in homophily_by_variant.items():
        df[f"h_{variant_name}"] = [
            hmap.get(r, (None, 0))[0] for r in relations
        ]
        df[f"n_edges_{variant_name}"] = [
            hmap.get(r, (None, 0))[1] for r in relations
        ]
    return df


def pearson_ignoring_nans(x, y):
    a = np.asarray(x, dtype=float)
    b = np.asarray(y, dtype=float)
    mask = ~(np.isnan(a) | np.isnan(b))
    if mask.sum() < 3:
        return float("nan"), float("nan"), int(mask.sum())
    r, p = pearsonr(a[mask], b[mask])
    return float(r), float(p), int(mask.sum())


def spearman_ignoring_nans(x, y):
    """Rank-based correlation — sanity check against Pearson when relationship
    is non-linear or sensitive to outliers. Same NaN handling."""
    a = np.asarray(x, dtype=float)
    b = np.asarray(y, dtype=float)
    mask = ~(np.isnan(a) | np.isnan(b))
    if mask.sum() < 3:
        return float("nan"), float("nan"), int(mask.sum())
    r, p = spearmanr(a[mask], b[mask])
    return float(r), float(p), int(mask.sum())


def plot_three_panel(df, correlations, output_path_pdf, output_path_png):
    variants = [
        ("h_nfr_full", "Next Funding Round (full)"),
        ("h_exit_full", "Exit (full)"),
        ("h_exit_mature", "Exit (mature subset)"),
    ]
    variants = [v for v in variants if v[0] in df.columns]

    categories = sorted(df["category"].unique())
    palette = plt.get_cmap("tab10").colors
    color_by_cat = {cat: palette[i % len(palette)] for i, cat in enumerate(categories)}

    n_panels = len(variants)
    fig, axes = plt.subplots(1, n_panels, figsize=(4.2 * n_panels, 4.2), sharey=True)
    if n_panels == 1:
        axes = [axes]

    for ax, (col, title) in zip(axes, variants):
        sub = df.dropna(subset=[col, "attention_weight"])
        for cat in categories:
            mask = sub["category"] == cat
            if not mask.any():
                continue
            ax.scatter(
                sub.loc[mask, col],
                sub.loc[mask, "attention_weight"],
                s=60,
                alpha=0.85,
                color=color_by_cat[cat],
                edgecolor="black",
                linewidth=0.4,
                label=cat,
            )
        # Fit line
        if len(sub) >= 2:
            coef = np.polyfit(sub[col], sub["attention_weight"], 1)
            xs = np.linspace(sub[col].min(), sub[col].max(), 100)
            ax.plot(xs, np.polyval(coef, xs), color="grey", linewidth=1.2, linestyle="--")
        r, p, n = correlations.get(col, (float("nan"), float("nan"), 0))
        ax.set_title(f"{title}\n$\\rho$={r:.2f}, p={p:.1g}, n={n}", fontsize=10)
        ax.set_xlabel("Edge homophily h")
        ax.grid(True, alpha=0.25)
    axes[0].set_ylabel("Mean metapath attention weight")
    axes[-1].legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8, frameon=False)
    plt.tight_layout()
    plt.savefig(output_path_pdf, bbox_inches="tight")
    plt.savefig(output_path_png, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _scatter_defense_panel(ax, sub, x_col, baseline_h, rho, x_min, x_max,
                           label_top_k=7, show_legend=False, attention_unit_pct=True,
                           baseline_label_side="left", min_edges=50,
                           label_whitelist=None):
    """One panel in defense-deck style: shaded heterophilic/homophilic zones,
    baseline line, log-sized points by edge count, category colors, annotations
    on top-k highest-attention points, rho box.

    Drawing order is deliberately:
      1. filter low-evidence metapaths (``n_edges < min_edges``)
      2. backdrop (zones + baseline line + scatter + axis bounds)
      3. static text (baseline-h, hetero/homo arrows, rho box, optional legend)
      4. high-attention metapath labels (then ``adjust_text`` reflows them
         while avoiding all the static artists captured in step 3).
    Without this ordering ``adjust_text`` cannot see the legend / rho box and
    the labels will collide with them.

    Args:
        baseline_label_side: "left" places the ``baseline h=…`` annotation just
            left of the dashed line (heterophilic side); "right" places it just
            right of the line (homophilic side). Useful when the homophilic zone
            is too narrow to fit the label, or vice versa.
        min_edges: drop metapaths whose ``n_edges_<variant>`` is below this. The
            default 50 culls statistical noise (a metapath with 4 edges that
            happens to land at h=1.0 is uninformative and crowds the plot).
    """
    # 1. Filter low-evidence metapaths --------------------------------------
    n_col = f"n_edges_{x_col.replace('h_', '')}"
    sub = sub.loc[sub[n_col].astype(float) >= float(min_edges)].copy()
    if sub.empty:
        return

    # 2. Backdrop -----------------------------------------------------------
    ax.axvspan(x_min, baseline_h, alpha=0.07, color="#d62728")  # heterophilic
    ax.axvspan(baseline_h, x_max, alpha=0.07, color="#2ca02c")  # homophilic
    ax.axvline(x=baseline_h, color="black", linestyle="--", linewidth=1.0, alpha=0.7)

    # Points: size ∝ log(n_edges), color by category
    n_edges = sub[n_col].astype(float).clip(lower=1)
    log_e = np.log10(n_edges)
    sizes = 30 + (log_e - log_e.min()) / max(log_e.max() - log_e.min(), 1e-9) * 450

    y_attn = sub["attention_weight"].astype(float).values
    y_plot = y_attn * 100.0 if attention_unit_pct else y_attn

    for cat in CATEGORY_ORDER:
        mask = sub["category"].values == cat
        if not mask.any():
            continue
        ax.scatter(
            sub[x_col].values[mask],
            y_plot[mask],
            s=sizes.values[mask],
            color=CATEGORY_COLORS[cat],
            alpha=0.85,
            edgecolor="white",
            linewidth=0.7,
            label=cat,
            zorder=5,
        )

    # Lock axis bounds before adjust_text runs.
    y_max = max(y_plot.max(), 1e-3)
    y_top = y_max * 1.20
    ax.set_ylim(bottom=-y_top * 0.22, top=y_top)
    ax.set_xlim(x_min, x_max)
    ax.grid(True, alpha=0.18)

    # Hide negative y-tick labels (we keep the bottom margin for the
    # heterophilic/homophilic arrow labels but don't want a "-2.5" on the axis).
    ax.yaxis.set_major_formatter(plt.FuncFormatter(
        lambda v, _pos: "" if v < 0 else f"{v:.0f}"
    ))

    # Per-category metapath count → drives 1-of-a-kind detection for both the
    # legend (replace category name with the metapath name) and the labeling
    # loop (skip metapaths that get a unique legend entry).
    unique_per_cat = sub.groupby("category")["metapath"].nunique().to_dict()

    # 3. Static text (baseline label + hetero/homo arrows + rho box + legend).
    static_artists = []
    if baseline_label_side == "right":
        static_artists.append(ax.text(
            baseline_h + (x_max - x_min) * 0.005, -y_top * 0.02,
            f"baseline $h$={baseline_h:.3f}",
            fontsize=7, va="bottom", ha="left", style="italic", color="#333",
        ))
    else:
        static_artists.append(ax.text(
            baseline_h - (x_max - x_min) * 0.005, -y_top * 0.02,
            f"baseline $h$={baseline_h:.3f}",
            fontsize=7, va="bottom", ha="right", style="italic", color="#333",
        ))
    y_arrow = -y_top * 0.18
    static_artists.append(ax.text(
        (x_min + baseline_h) / 2, y_arrow,
        "← Heterophilic", fontsize=8, color="#b22222",
        ha="center", style="italic",
    ))
    static_artists.append(ax.text(
        (baseline_h + x_max) / 2, y_arrow,
        "Homophilic →", fontsize=8, color="#228b22",
        ha="center", style="italic",
    ))
    x_rho = baseline_h + (x_max - baseline_h) * 0.62
    y_rho = y_top * 0.45
    static_artists.append(ax.text(
        x_rho, y_rho,
        f"Pearson $\\rho = {rho:.2f}$",
        fontsize=10, ha="center", va="center", fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.45", fc="white", ec="#444", lw=0.9),
    ))

    if show_legend:
        # Build proxy handles so the legend swatches are uniform-sized circles
        # regardless of the data marker size, and so we can swap labels to the
        # metapath name when a category has exactly one metapath in this panel.
        from matplotlib.lines import Line2D
        proxy_handles, proxy_labels = [], []
        cats_present = [c for c in CATEGORY_ORDER if c in sub["category"].values]
        for cat in cats_present:
            if unique_per_cat.get(cat, 0) == 1:
                label = sub.loc[sub["category"] == cat, "metapath"].iloc[0]
                label = METAPATH_PRETTY.get(label, label)
            else:
                label = cat
            proxy_handles.append(Line2D(
                [0], [0], marker="o", linestyle="",
                markerfacecolor=CATEGORY_COLORS[cat], markeredgecolor="white",
                markersize=7, markeredgewidth=0.7, label=label,
            ))
            proxy_labels.append(label)
        legend = ax.legend(
            proxy_handles, proxy_labels,
            fontsize=7, loc="upper left", bbox_to_anchor=(0.015, 0.985),
            frameon=True, framealpha=0.95, facecolor="white", edgecolor="#888",
            title="Meta-path group", title_fontsize=7,
            borderpad=0.45, handletextpad=0.6, labelspacing=0.4,
        )
        legend.set_zorder(20)
        static_artists.append(legend)

    # 4. High-attention labels. If ``label_whitelist`` is provided, only
    # metapaths whose name is in the set are labeled (used by the
    # single-panel NFR figure to call out one outlier explicitly). Otherwise
    # fall back to the old threshold-based selection that skips metapaths
    # whose category is uniquely represented in this panel.
    label_texts = []
    if label_whitelist is not None:
        for i in range(len(sub)):
            name = sub["metapath"].values[i]
            if name not in label_whitelist:
                continue
            label = METAPATH_PRETTY.get(name, name)
            label_texts.append(ax.text(
                sub[x_col].values[i], y_plot[i], label,
                fontsize=7, color="#222", zorder=6,
            ))
    else:
        thresh = y_max * 0.18
        for i in range(len(sub)):
            if y_plot[i] < thresh:
                continue
            cat = sub["category"].values[i]
            if unique_per_cat.get(cat, 0) == 1:
                continue
            name = sub["metapath"].values[i]
            label = METAPATH_PRETTY.get(name, name)
            label_texts.append(ax.text(
                sub[x_col].values[i], y_plot[i], label,
                fontsize=7, color="#222", zorder=6,
            ))
    if label_texts:
        from adjustText import adjust_text
        adjust_text(
            label_texts, ax=ax,
            objects=static_artists,
            expand=(1.30, 1.80),
            force_text=(0.6, 1.4),
            force_static=(0.5, 0.8),
            arrowprops=dict(arrowstyle="-", color="#888", lw=0.5, alpha=0.75),
            min_arrow_len=4.0,
            iter_lim=400,
        )


def plot_defense_style_nfr_only(df, correlations, output_path_pdf,
                                output_path_png=None,
                                label_whitelist=("early_portfolio_siblings",)):
    """Single-panel NFR-only variant for the paper figure.

    Drops the duplicate "Next Funding Round" title (the LaTeX caption already
    names the panel), forces the serif font family to match the rest of the
    paper, and labels only the metapaths in ``label_whitelist`` (default: the
    high-attention high-homophily corner outlier early_portfolio_siblings).
    Writes directly to the final paper filename so the brittle
    ``crop_attention_pdf.py`` step is not needed.
    """
    col = "h_nfr_full"
    title_unused = "Next Funding Round"  # noqa: F841 (kept for documentation)
    baseline_h = 0.778
    side = "left"
    sub = df.dropna(subset=[col, "attention_weight"]).copy()
    if sub.empty:
        return
    x_vals = sub[col].values
    x_min = max(0.0, x_vals.min() - 0.03)
    x_max = min(1.0, x_vals.max() + 0.03)
    if baseline_h > x_max:
        x_max = baseline_h + 0.02
    if baseline_h < x_min:
        x_min = baseline_h - 0.02
    rho, _, _ = correlations.get(col, (float("nan"), float("nan"), 0))

    with plt.rc_context({"font.family": "serif"}):
        fig, ax = plt.subplots(figsize=(5.6, 3.1))
        _scatter_defense_panel(
            ax, sub, col, baseline_h, rho, x_min, x_max,
            label_top_k=0,
            show_legend=True,
            baseline_label_side=side,
            label_whitelist=set(label_whitelist) if label_whitelist else None,
        )
        ax.set_xlabel("Edge homophily ratio $h$ (label-aware)", fontsize=9)
        ax.set_ylabel("Mean learned attention (%)", fontsize=9)
        plt.tight_layout()
        plt.savefig(output_path_pdf, bbox_inches="tight")
        if output_path_png is not None:
            plt.savefig(output_path_png, dpi=300, bbox_inches="tight")
        plt.close(fig)


def plot_defense_style_two_panel(df, correlations, output_path_pdf, output_path_png):
    """Fig 1 paper: two side-by-side panels in defense-deck visual style.

    The first panel carries the meta-path-group legend (anchored upper-left,
    inside the axes, with a white block background); the second panel is
    legend-free for visual cleanliness. ``_scatter_defense_panel`` handles the
    legend internally and passes its bbox to ``adjust_text`` so high-attention
    labels do not collide with it.
    """
    variants = [
        # (column, title, baseline_h, baseline_label_side)
        ("h_nfr_full", "Next Funding Round", 0.778, "left"),
        # Exit baseline (0.920) is too close to the right edge to fit the label
        # on the homophilic side using the default left anchor; flip to "right".
        ("h_exit_mature", "Exit (mature subset)", 0.920, "right"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.6))
    for ax, (col, title, baseline_h, side) in zip(axes, variants):
        sub = df.dropna(subset=[col, "attention_weight"]).copy()
        if sub.empty:
            continue
        x_vals = sub[col].values
        x_min = max(0.0, x_vals.min() - 0.03)
        x_max = min(1.0, x_vals.max() + 0.03)
        if baseline_h > x_max:
            x_max = baseline_h + 0.02
        if baseline_h < x_min:
            x_min = baseline_h - 0.02
        rho, _, _ = correlations.get(col, (float("nan"), float("nan"), 0))
        _scatter_defense_panel(
            ax, sub, col, baseline_h, rho, x_min, x_max,
            label_top_k=7,
            show_legend=(ax is axes[0]),
            baseline_label_side=side,
        )
        ax.set_title(title, fontsize=11, pad=6)
        ax.set_xlabel("Edge homophily ratio $h$ (label-aware)", fontsize=9)
    axes[0].set_ylabel("Mean learned attention (%)", fontsize=9)
    axes[1].set_ylabel("")
    plt.tight_layout()
    plt.savefig(output_path_pdf, bbox_inches="tight")
    plt.savefig(output_path_png, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_two_panel(df, correlations, output_path_pdf, output_path_png):
    """Polished 2-panel figure for paper Fig 1: NFR (left) + Exit-mature (right)."""
    variants = [
        ("h_nfr_full", "Next Funding Round"),
        ("h_exit_mature", "Exit (mature subset)"),
    ]
    variants = [v for v in variants if v[0] in df.columns]

    categories = sorted(df["category"].unique())
    palette = plt.get_cmap("tab10").colors
    color_by_cat = {cat: palette[i % len(palette)] for i, cat in enumerate(categories)}

    fig, axes = plt.subplots(1, 2, figsize=(9.0, 4.2), sharey=True)

    for ax, (col, title) in zip(axes, variants):
        sub = df.dropna(subset=[col, "attention_weight"])
        for cat in categories:
            mask = sub["category"] == cat
            if not mask.any():
                continue
            ax.scatter(
                sub.loc[mask, col],
                sub.loc[mask, "attention_weight"],
                s=75,
                alpha=0.88,
                color=color_by_cat[cat],
                edgecolor="black",
                linewidth=0.5,
                label=cat,
            )
        if len(sub) >= 2:
            coef = np.polyfit(sub[col], sub["attention_weight"], 1)
            xs = np.linspace(sub[col].min(), sub[col].max(), 100)
            ax.plot(xs, np.polyval(coef, xs), color="grey", linewidth=1.4, linestyle="--")
        r, _, _ = correlations.get(col, (float("nan"), float("nan"), 0))
        ax.set_title(f"{title}   $\\rho$={r:.2f}", fontsize=11)
        ax.set_xlabel("Edge homophily $h$", fontsize=10)
        ax.grid(True, alpha=0.25)
    axes[0].set_ylabel("Mean metapath attention weight", fontsize=10)
    axes[1].legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8, frameon=False, title="Metapath category")
    plt.tight_layout()
    plt.savefig(output_path_pdf, bbox_inches="tight")
    plt.savefig(output_path_png, dpi=300, bbox_inches="tight")
    plt.close(fig)


# -------- main -------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--graph", default=str(DEFAULT_GRAPH))
    parser.add_argument("--config", default=str(DEFAULT_CONFIG),
                        help="Master config.yaml (provides paths and base defaults).")
    parser.add_argument("--champion-config", default=None,
                        help="Optional param-trimmed champion YAML "
                             "(experiments/champion_configs/*.yaml). Deep-merged "
                             "into --config so the Trainer instantiates the model "
                             "with the same HPs the checkpoint was trained under.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTDIR))
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print(f"Loading graph: {args.graph}")
    graph = _load_graph_any(args.graph)
    print(f"  node types: {graph.node_types}")
    ss_edge_types = [et for et in graph.edge_types if et[0] == "startup" and et[2] == "startup"]
    print(f"  startup→startup edge types: {len(ss_edge_types)}")

    print(f"Loading config: {args.config}")
    config = load_config(args.config)
    if args.champion_config:
        import yaml
        with open(args.champion_config) as f:
            champion_overrides = yaml.safe_load(f) or {}
        config = _deep_merge(config, champion_overrides)
        print(f"  Merged champion overrides from {args.champion_config}")

    print(f"Loading model: {args.checkpoint}")
    model = load_model(graph, config, Path(args.checkpoint), device)

    print("Running inference to extract attention weights...")
    metapath_names, attention_vec = extract_attention(model, graph, device)
    print(f"  extracted {len(metapath_names)} metapath attention weights")

    print("Computing homophily per label variant...")
    label_variants = build_label_variants(graph, config)
    homophily_by_variant = {
        name: compute_homophily_for_labels(graph, y)
        for name, y in label_variants.items()
    }
    for name, hmap in homophily_by_variant.items():
        defined = sum(1 for v in hmap.values() if v[0] is not None)
        print(f"  {name}: {defined}/{len(hmap)} metapaths have defined homophily")

    df = assemble_dataframe(metapath_names, attention_vec, homophily_by_variant)
    csv_path = out_dir / "attention_homophily.csv"
    df.to_csv(csv_path, index=False)
    print(f"Saved {csv_path}")

    correlations = {}
    spearman_correlations = {}
    for variant_name in label_variants.keys():
        col = f"h_{variant_name}"
        r_p, p_p, n = pearson_ignoring_nans(df[col], df["attention_weight"])
        r_s, p_s, _ = spearman_ignoring_nans(df[col], df["attention_weight"])
        correlations[col] = (r_p, p_p, n)
        spearman_correlations[col] = (r_s, p_s, n)
        print(f"  Pearson (attention, {col}): rho={r_p:.4f}  p={p_p:.3g}  n={n}")
        print(f"  Spearman(attention, {col}): rho={r_s:.4f}  p={p_s:.3g}  n={n}")

    summary = {
        "checkpoint": str(args.checkpoint),
        "graph": str(args.graph),
        "n_metapaths": len(metapath_names),
        "correlations": {
            k: {"pearson_rho": v[0], "pearson_p": v[1], "n": v[2],
                "spearman_rho": spearman_correlations[k][0],
                "spearman_p": spearman_correlations[k][1]}
            for k, v in correlations.items()
        },
    }
    with (out_dir / "correlation_summary.json").open("w") as f:
        json.dump(summary, f, indent=2)

    plot_three_panel(
        df,
        correlations,
        out_dir / "homophily_vs_attention_3panel.pdf",
        out_dir / "homophily_vs_attention_3panel.png",
    )
    print(f"Saved {out_dir / 'homophily_vs_attention_3panel.pdf'}")

    plot_two_panel(
        df,
        correlations,
        out_dir / "homophily_vs_attention_fig1.pdf",
        out_dir / "homophily_vs_attention_fig1.png",
    )
    print(f"Saved {out_dir / 'homophily_vs_attention_fig1.pdf'}")

    plot_defense_style_two_panel(
        df,
        correlations,
        out_dir / "homophily_vs_attention_defense_style.pdf",
        out_dir / "homophily_vs_attention_defense_style.png",
    )
    print(f"Saved {out_dir / 'homophily_vs_attention_defense_style.pdf'}")

    print("\nDone.")


if __name__ == "__main__":
    main()
