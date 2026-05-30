#!/usr/bin/env python
"""Generate LaTeX tables for the graph-paper appendix from committed data.

Reads:
  outputs/graph_statistics/heterophily_spectrum.csv
  outputs/graph_statistics/heterophily_spectrum_meta.json
  outputs/graph_statistics/graph_statistics_full.json
  outputs/graph_statistics/edge_statistics.csv

Writes:
  graph-paper/tables/metapath_inventory.tex
  graph-paper/tables/graph_statistics.tex

Also prints a verification block recomputing every quantitative claim
in sections 1 and 6.1 so the user can spot-check numbers against the
paper prose before committing.
"""

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
# Prefer 5-seed multiseed aggregate (canonical); fall back to seed-42 single-seed.
HETERO_MULTISEED_CSV = REPO / "outputs/graph_statistics/heterophily_spectrum_multiseed.csv"
HETERO_SINGLE_CSV = REPO / "outputs/graph_statistics/heterophily_spectrum_seed42.csv"
HETERO_META = REPO / "outputs/graph_statistics/heterophily_spectrum_meta_seed42.json"
GRAPH_STATS = REPO / "outputs/graph_statistics/graph_statistics_full.json"
EDGE_STATS = REPO / "outputs/graph_statistics/edge_statistics.csv"
OUT_DIR = REPO / "graph-paper/tables"

# -- Metapath display names ---------------------------------------------------
# Manual slug -> human label. Keeps table readable.
METAPATH_LABEL = {
    "late_founder_vc_employment":      "Late: founder--VC employment",
    "mid_founder_vc_employment":       "Mid: founder--VC employment",
    "early_founder_vc_employment":     "Early: founder--VC employment",
    "late_board_director_network":     "Late: board-director network",
    "mid_board_director_network":      "Mid: board-director network",
    "early_board_director_network":    "Early: board-director network",
    "late_alumni_investor_network":    "Late: alumni-investor network",
    "mid_alumni_investor_network":     "Mid: alumni-investor network",
    "early_alumni_investor_network":   "Early: alumni-investor network",
    "late_portfolio_siblings":         "Late: portfolio siblings",
    "mid_portfolio_siblings":          "Mid: portfolio siblings",
    "early_portfolio_siblings":        "Early: portfolio siblings",
    "late_founder_coworking_syndicate":  "Late: founder coworking syndicate",
    "mid_founder_coworking_syndicate":   "Mid: founder coworking syndicate",
    "early_founder_coworking_syndicate": "Early: founder coworking syndicate",
    "late_founder_coworking_investor":   "Late: founder-coworking investor",
    "mid_founder_coworking_investor":    "Mid: founder-coworking investor",
    "early_founder_coworking_investor":  "Early: founder-coworking investor",
    "late_investor_founder_coworking":   "Late: investor-founder coworking",
    "mid_investor_founder_coworking":    "Mid: investor-founder coworking",
    "early_investor_founder_coworking":  "Early: investor-founder coworking",
    "late_investor_alumni":            "Late: investor alumni",
    "mid_investor_alumni":             "Mid: investor alumni",
    "early_investor_alumni":           "Early: investor alumni",
    "late_board_employment_network":   "Late: board-employment network",
    "mid_board_employment_network":    "Mid: board-employment network",
    "early_board_employment_network":  "Early: board-employment network",
    "co_working_network":              "Founder co-work",
    "alumni_network":                  "Alumni",
    "sector_peers":                    "Sector peers",
    "city_peers":                      "City peers",
    "serial_founder":                  "Serial founder",
}

# Non-stage categories map to themselves; stage-prefixed slugs all collapse
# to "Investor-mediated" to match the five-category aggregation in 6.1.
PEER_CATEGORY = {
    "co_working_network": "Founder co-work",
    "alumni_network":     "Alumni",
    "sector_peers":       "Sector",
    "city_peers":         "City",
    "serial_founder":     "Serial founder",
}


# -- Formatting helpers -------------------------------------------------------
def fmt_int(n):
    return f"{n:,}"


def _is_missing(x):
    if x is None or x == "":
        return True
    try:
        return math.isnan(float(x))
    except (TypeError, ValueError):
        return True


def fmt3(x):
    if _is_missing(x):
        return "---"
    return f"{float(x):.3f}"


def fmt_delta(x):
    if _is_missing(x):
        return "---"
    xf = float(x)
    if abs(xf) < 5e-4:
        return "0.000"
    return f"{xf:+.3f}"


def fmt2(x):
    return f"{x:.2f}"


def pearson_weighted(xs, ys, ws):
    """Weighted Pearson correlation (used by 6.1 for log-edge weighting)."""
    total_w = sum(ws)
    mx = sum(w * x for w, x in zip(ws, xs)) / total_w
    my = sum(w * y for w, y in zip(ws, ys)) / total_w
    cov = sum(w * (x - mx) * (y - my) for w, x, y in zip(ws, xs, ys)) / total_w
    vx = sum(w * (x - mx) ** 2 for w, x in zip(ws, xs)) / total_w
    vy = sum(w * (y - my) ** 2 for w, y in zip(ws, ys)) / total_w
    return cov / math.sqrt(vx * vy)


# -- Loaders ------------------------------------------------------------------
def load_spectrum():
    """Prefer the multiseed aggregate; fall back to seed-42 single-seed.

    When multiseed is loaded, `_mean` suffixes are stripped so downstream code
    can access h_nfr, h_exit_mature, mde_per_edge, n_edges uniformly. Standard
    deviations remain accessible under `h_nfr_std`, etc. and feed the caption
    footnote about RW robustness.
    """
    rows = []
    if HETERO_MULTISEED_CSV.exists():
        csv_path = HETERO_MULTISEED_CSV
        multiseed = True
    elif HETERO_SINGLE_CSV.exists():
        csv_path = HETERO_SINGLE_CSV
        multiseed = False
    else:
        raise FileNotFoundError(
            f"Neither {HETERO_MULTISEED_CSV} nor {HETERO_SINGLE_CSV} exists. "
            "Run scripts/compute_heterophily_spectrum.py (plus aggregate for multiseed)."
        )

    rename_on_multiseed = {
        "n_edges_mean": "n_edges",
        "h_nfr_mean": "h_nfr",
        "delta_nfr_mean": "delta_nfr",
        "h_exit_full_mean": "h_exit_full",
        "delta_exit_full_mean": "delta_exit_full",
        "h_exit_mature_mean": "h_exit_mature",
        "delta_exit_mature_mean": "delta_exit_mature",
        "mde_mean": "mde",
        "mde_per_edge_mean": "mde_per_edge",
        "mde_rand_per_edge_mean": "mde_rand_per_edge",
        "delta_mde_per_edge_mean": "delta_mde_per_edge",
    }

    with csv_path.open() as f:
        for r in csv.DictReader(f):
            # Canonicalize mean columns to unsuffixed names.
            if multiseed:
                for old, new in rename_on_multiseed.items():
                    if old in r:
                        r[new] = r.pop(old)
            for k in list(r.keys()):
                if k in ("metapath", "category"):
                    continue
                # Numeric columns: coerce blanks/non-numeric to NaN so downstream
                # arithmetic doesn't crash (rare on real data; happens on synthetic
                # when a metapath has zero labeled mature edges, etc.).
                try:
                    r[k] = float(r[k])
                except (TypeError, ValueError):
                    r[k] = float("nan")
            r["n_edges"] = int(round(r["n_edges"])) if not math.isnan(r["n_edges"]) else 0
            r["stage"] = r["metapath"].split("_")[0] if r["metapath"].split("_")[0] in ("early", "mid", "late") else None
            rows.append(r)
    return rows


def load_edges():
    rows = []
    with EDGE_STATS.open() as f:
        for r in csv.DictReader(f):
            if r["is_metapath"].lower() == "true":
                continue
            r["num_edges"] = int(r["num_edges"])
            if r["num_edges"] == 0:
                continue
            rows.append(r)
    return rows


# -- Table 1: metapath inventory ---------------------------------------------
def build_metapath_table(rows, baseline_nfr, baseline_exit):
    rows = sorted(rows, key=lambda r: r["h_nfr"])
    body = []
    prev_was_investor = True  # first rows are investor-mediated
    for r in rows:
        is_peer = r["metapath"] in PEER_CATEGORY
        if prev_was_investor and is_peer:
            body.append(r"\midrule  % investor-mediated (above) vs peer paths (below)")
        prev_was_investor = not is_peer
        label = METAPATH_LABEL[r["metapath"]]
        body.append(
            f"  {label:<38} & {fmt_int(r['n_edges']):>7} & "
            f"{fmt3(r['h_nfr'])} & {fmt_delta(r['delta_nfr'])} & "
            f"{fmt3(r['h_exit_mature'])} & {fmt_delta(r['delta_exit_mature'])} & "
            f"{fmt2(r['mde_per_edge']):>6} & {fmt2(r['mde_rand_per_edge']):>6} \\\\"
        )

    n_investor = sum(1 for r in rows if r["metapath"] not in PEER_CATEGORY)
    n_peer = sum(1 for r in rows if r["metapath"] in PEER_CATEGORY)
    return r"""% Auto-generated by scripts/generate_paper_tables.py -- do not edit by hand.
\begin{table*}[t]
\centering
\footnotesize
\setlength{\tabcolsep}{4pt}
\caption{\textbf{Per-meta-path inventory.} All $""" + str(len(rows)) + r"""$ startup-to-startup
meta-paths sorted by NFR MLH (most heterophilic first). Top: $""" + str(n_investor) + r"""$
investor-mediated paths; bottom: $""" + str(n_peer) + r"""$ peer-category paths. $\Delta$ =
change vs random-pair baseline ($h^{(\text{NFR})}_{\mathrm{rand}}=""" + fmt3(baseline_nfr) + r"""$,
$h^{(\text{Exit})}_{\mathrm{rand}}=""" + fmt3(baseline_exit) + r"""$). MDE per edge on standardized
$33$-dim features; rightmost column is the shuffled-pair baseline.
Means over $5$ random-walk materializations.}
\label{tab:metapath_inventory}
\begin{tabular}{@{}lrcccccc@{}}
\toprule
\textbf{Meta-path} & \textbf{$|E|$} &
\textbf{MLH} & \textbf{$\Delta$} &
\textbf{MLH} & \textbf{$\Delta$} &
\multicolumn{2}{c}{\textbf{MDE/edge}} \\
\cmidrule(lr){3-4} \cmidrule(lr){5-6} \cmidrule(lr){7-8}
 & & \multicolumn{2}{c}{NFR} & \multicolumn{2}{c}{Exit (mature)} &
\textbf{obs.} & \textbf{rand.} \\
\midrule
""" + "\n".join(body) + r"""
\bottomrule
\end{tabular}
\end{table*}
"""


# -- Table 2: graph statistics -----------------------------------------------
# Semantic grouping of base edges into seven relation types.
EDGE_GROUPS = [
    ("Funding (startup $\\to$ investor)", [
        "startup-early_stage_funded_by-investor",
        "startup-mid_stage_funded_by-investor",
        "startup-late_stage_funded_by-investor",
        "startup-other_funded_by-investor",
    ]),
    ("Founding (startup $\\to$ founder)", ["startup-founded_by-founder"]),
    ("Board/director (founder $\\to$ startup, investor)", [
        "founder-on_board_of-startup",
        "founder-director_at-startup",
        "founder-director_at-investor",
    ]),
    ("Sector (startup $\\to$ sector)", ["startup-in_sector-sector"]),
    ("Geography ($\\star \\to$ city)", [
        "startup-based_in-city",
        "investor-based_in-city",
        "university-based_in-city",
    ]),
    ("Education (founder $\\to$ university)", ["founder-studied_at-university"]),
    ("Founder--Investor links", [
        "founder-worked_at-investor",
        "founder-same_as-investor",
    ]),
]


def build_graph_stats_table(stats, edges, meta):
    nodes = stats["node_stats"]
    edge_by_key = {e["edge_type"]: e for e in edges}
    cc = stats["connected_components"]

    # --- Nodes block (feature counts live in Fig graph_schema + Table
    # feature_inventory; repeating them here would duplicate the schema
    # figure's intrinsic-tier counts with the JSON's raw-tier counts.)
    node_order = ["startup", "founder", "investor", "city", "sector", "university"]
    node_lines = []
    for t in node_order:
        d = nodes[t]
        node_lines.append(f"{t.capitalize():<10} & {fmt_int(d['count']):>9} \\\\")

    # --- Edges block
    edge_lines = []
    total_edges = 0
    for label, keys in EDGE_GROUPS:
        # Skip anything not present or zero
        subcounts = []
        for k in keys:
            if k in edge_by_key:
                subcounts.append(edge_by_key[k]["num_edges"])
        if not subcounts:
            continue
        n = sum(subcounts)
        total_edges += n
        n_rels = len(subcounts)
        edge_lines.append(f"{label} & {n_rels} & {fmt_int(n):>9} \\\\")

    # --- Connectivity block (three numbers)
    conn_lines = [
        f"Largest weakly-connected component   & {cc['largest_component_pct']:.1f}\\% "
        f"({fmt_int(cc['largest_component_size'])} of {fmt_int(cc['total_nodes'])} nodes) \\\\",
        f"Total connected components           & {fmt_int(cc['num_components'])} \\\\",
        f"Isolated startups                    & {cc['isolated_startups']} (all {fmt_int(cc['startups_in_largest_component'])} startups in the giant component) \\\\",
    ]

    # --- Class balance (three rows): source from graph_statistics_full.json
    # class_balance. target_0=NFR, target_1=Exit(full), target_4=(Exit masked
    # by maturity); the negative class of target_4 encodes "Exit occurred
    # within the mature window" and its rate over the full population is the
    # "effective Exit positive rate" quoted in 1.
    cb = stats["class_balance"]
    nfr_rate = cb["target_0"]["overall"]["positive"] / cb["target_0"]["overall"]["total"]
    exit_full_rate = cb["target_1"]["overall"]["positive"] / cb["target_1"]["overall"]["total"]
    exit_mature_rate = cb["target_4"]["overall"]["negative"] / cb["target_4"]["overall"]["total"]

    balance_lines = [
        f"\\NFR\\ (full population)        & {nfr_rate*100:.2f}\\% positive "
        f"(baseline $h^{{(\\text{{NFR}})}}_{{\\mathrm{{rand}}}} = {meta['baseline_nfr']:.3f}$) \\\\",
        f"\\EXIT\\ (full population)       & {exit_full_rate*100:.2f}\\% positive "
        f"(baseline $h^{{(\\text{{Exit}})}}_{{\\mathrm{{rand}}}} = {meta['baseline_exit_full']:.3f}$) \\\\",
        f"\\EXIT\\ (effective, mature-masked) & {exit_mature_rate*100:.2f}\\% positive "
        f"(baseline $h^{{(\\text{{Exit,mature}})}}_{{\\mathrm{{rand}}}} = {meta['baseline_exit_mature']:.3f}$) \\\\",
    ]

    total_nodes = sum(nodes[t]["count"] for t in node_order)

    return r"""% Auto-generated by scripts/generate_paper_tables.py -- do not edit by hand.
\begin{table}[t]
\centering
\footnotesize
\setlength{\tabcolsep}{4pt}
\caption{\textbf{Graph statistics.} Node and edge cardinalities,
connectivity, and target class balance. Base edges ($15$ relations)
carry the heterogeneous $1$-hop signal; on top we materialize $32$
semantic meta-paths (Table~\ref{tab:metapath_inventory}). Forward
direction only; reverse edges stored but counted once.}
\label{tab:graph_statistics}
\begin{tabular}{@{}lr@{}}
\toprule
\multicolumn{2}{@{}l}{\textbf{Nodes}} \\
\textbf{Type} & \textbf{Count} \\
\midrule
""" + "\n".join(node_lines) + r"""
\midrule
\textbf{Total} & """ + fmt_int(total_nodes) + r""" \\
\bottomrule
\end{tabular}

\vspace{4pt}
\begin{tabular}{@{}lrr@{}}
\toprule
\multicolumn{3}{@{}l}{\textbf{Edges by semantic group}} \\
\textbf{Group} & \textbf{\#rel.} & \textbf{Count} \\
\midrule
""" + "\n".join(edge_lines) + r"""
\midrule
\textbf{Total (forward direction only)} & """ + str(sum(len(k) for _, k in EDGE_GROUPS)) + r""" & """ + fmt_int(total_edges) + r""" \\
\bottomrule
\end{tabular}

\vspace{4pt}
\begin{tabular}{@{}lp{0.62\columnwidth}@{}}
\toprule
\multicolumn{2}{@{}l}{\textbf{Connectivity}} \\
\midrule
""" + "\n".join(conn_lines) + r"""
\bottomrule
\end{tabular}

\vspace{4pt}
\begin{tabular}{@{}lp{0.62\columnwidth}@{}}
\toprule
\multicolumn{2}{@{}l}{\textbf{Target class balance}} \\
\midrule
""" + "\n".join(balance_lines) + r"""
\bottomrule
\end{tabular}
\end{table}
"""


# -- Verification -------------------------------------------------------------
def verify(rows, meta, stats):
    """Recompute every quantitative claim that appears in 1, 5.1, 6.1 prose."""
    print("=" * 72)
    print("VERIFICATION: recomputing paper claims from committed data")
    print("=" * 72)

    def check(label, got, expected, tol=5e-3):
        status = "PASS" if abs(got - expected) <= tol else "MISMATCH"
        print(f"  [{status}] {label}: computed={got:.4f}, paper claims={expected}")

    # --- meta.json vs paper
    print("\n-- Baselines (paper 1 para 3 and 6.1) --")
    check("h^(NFR)_rand",         meta["baseline_nfr"],         0.778)
    check("h^(Exit,mature)_rand", meta["baseline_exit_mature"], 0.924)

    # --- Category means on NFR (6.1 para 2)
    print("\n-- Category means on NFR (6.1) --")
    by_cat = defaultdict(list)
    for r in rows:
        cat = PEER_CATEGORY.get(r["metapath"], "Investor-mediated")
        by_cat[cat].append(r)

    # Claims updated for 32-path, 5-seed multiseed aggregate (2026-04-21).
    # Investor-mediated now spans 27 paths (9 families x 3 stages) including
    # the 3 board_employment_* variants previously excluded from the whitelist.
    claims = {
        "Investor-mediated": (0.572, -0.206, 27),
        "Sector":            (0.772, None, 1),
        "City":              (0.777, None, 1),
        "Alumni":            (0.755, None, 1),
        "Founder co-work":   (0.732, None, 1),
        "Serial founder":    (0.779, None, 1),
    }
    for cat, (mlh_claim, delta_claim, n_claim) in claims.items():
        sub = by_cat[cat]
        n = len(sub)
        mlh = sum(r["h_nfr"] for r in sub) / n
        delta = sum(r["delta_nfr"] for r in sub) / n
        check(f"  {cat} MLH (n={n}, paper claims n={n_claim})", mlh, mlh_claim)
        if delta_claim is not None:
            check(f"  {cat} delta", delta, delta_claim)

    # --- Stage gradient on investor-mediated (6.1 para 3)
    # Updated for 32-path, 5-seed multiseed aggregate (2026-04-21):
    # early 0.605, mid 0.560, late 0.555 — direction unchanged from 28-path.
    print("\n-- Stage gradient within investor-mediated (6.1) --")
    stage_claims = {"early": 0.61, "mid": 0.56, "late": 0.55}
    for stage, claim in stage_claims.items():
        sub = [r for r in rows if r["stage"] == stage
               and r["metapath"] not in PEER_CATEGORY]
        if not sub:
            continue
        mlh = sum(r["h_nfr"] for r in sub) / len(sub)
        check(f"  {stage}-stage mean MLH (n={len(sub)})", mlh, claim, tol=0.02)

    # Most heterophilic: late_founder_vc_employment, MLH = 0.514 on multiseed
    # (was 0.516 at seed 42 — barely shifts after 5-seed averaging).
    # next(..., None) so synthetic spectra without this specific metapath skip
    # the check instead of raising StopIteration.
    late_fvc = next((r for r in rows if r["metapath"] == "late_founder_vc_employment"), None)
    if late_fvc is not None:
        check("Most heterophilic path (late founder-VC employment)",
              late_fvc["h_nfr"], 0.514, tol=0.01)

    # --- MLH-MDE Pearson correlations (6.1 para 4)
    # Paper prose says "log-edge weighting"; the MDE axis in Fig 1 is log-scale,
    # so the reported correlation is on log10(MDE). 32-path multiseed values:
    # r(NFR) = -0.57 (was -0.61 on 28), r(Exit-mature) = -0.49 (was -0.28).
    # Drop rows with mde_per_edge <= 0 or NaN — log10 would crash / produce NaN
    # (rare on real data, possible on synthetic spectra with degenerate paths).
    print("\n-- MLH vs log10(MDE) Pearson correlations (log-edge weighted, 6.1) --")
    valid = [r for r in rows
             if isinstance(r["mde_per_edge"], (int, float))
             and r["mde_per_edge"] > 0 and not math.isnan(r["mde_per_edge"])]
    ws = [math.log(r["n_edges"] + 1) for r in valid]
    mlh_nfr = [r["h_nfr"] for r in valid]
    mlh_exit = [r["h_exit_mature"] for r in valid]
    log_mde = [math.log10(r["mde_per_edge"]) for r in valid]
    r_nfr = pearson_weighted(mlh_nfr, log_mde, ws) if len(valid) >= 2 else float("nan")
    r_exit = pearson_weighted(mlh_exit, log_mde, ws) if len(valid) >= 2 else float("nan")
    check("r(MLH, log10 MDE) on NFR",         r_nfr,  -0.57, tol=0.03)
    check("r(MLH, log10 MDE) on Exit mature", r_exit, -0.49, tol=0.03)

    # --- 31/32 heterophilic on NFR, 30/32 on Exit mature (1, 6.1, 32-path 5-seed)
    print("\n-- Heterophily counts (1, 6.1) --")
    n_nfr_below = sum(1 for r in rows if r["delta_nfr"] <= 0)
    n_exit_below = sum(1 for r in rows if r["delta_exit_mature"] < 0)
    check("#paths at or below NFR baseline", n_nfr_below, 31, tol=0)
    check("#paths below Exit-mature baseline", n_exit_below, 30, tol=0)

    # --- Graph structure claims (5.1 and appendix)
    print("\n-- Graph structure (5.1, appendix) --")
    check("#startups in graph",
          stats["node_stats"]["startup"]["count"], 163531, tol=0)
    check("largest component %",
          stats["connected_components"]["largest_component_pct"], 81.1, tol=0.2)

    # --- Class balance (1): source from graph_statistics_full.json class_balance
    print("\n-- Class balance from graph_statistics_full.json vs paper 1 --")
    cb = stats["class_balance"]
    nfr_rate = cb["target_0"]["overall"]["positive"] / cb["target_0"]["overall"]["total"]
    exit_full_rate = cb["target_1"]["overall"]["positive"] / cb["target_1"]["overall"]["total"]
    exit_mature_rate = cb["target_4"]["overall"]["negative"] / cb["target_4"]["overall"]["total"]
    check("NFR positive rate (overall, target_0)",
          nfr_rate, 0.128, tol=0.002)
    check("Exit positive rate (full, target_1)",
          exit_full_rate, 0.017, tol=0.002)
    check("Exit effective rate (mature-masked, target_4 neg)",
          exit_mature_rate, 0.042, tol=0.002)

    print("\n" + "=" * 72)


# -- Main --------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out-dir", type=Path, default=OUT_DIR,
                   help="Where to write metapath_inventory.tex and "
                        "graph_statistics.tex. Defaults to the LaTeX tree "
                        "(graph-paper/tables); redirect to a local path such "
                        "as outputs/tables when that tree is absent.")
    args = p.parse_args()
    out_dir = args.out_dir

    rows = load_spectrum()
    if HETERO_META.exists():
        meta = json.loads(HETERO_META.read_text())
    else:
        # Fall back to averaging per-seed metas when the seed42 file is absent.
        import glob
        seed_meta_paths = sorted(glob.glob(str(REPO / "outputs/graph_statistics/heterophily_spectrum_meta_seed*.json")))
        if not seed_meta_paths:
            raise FileNotFoundError(
                f"Neither {HETERO_META} nor any heterophily_spectrum_meta_seed*.json exists."
            )
        metas = [json.loads(Path(p).read_text()) for p in seed_meta_paths]
        meta = {k: sum(m[k] for m in metas) / len(metas)
                for k in ("baseline_nfr", "baseline_exit_full",
                          "baseline_exit_mature",
                          "positive_rate_nfr", "positive_rate_exit_full",
                          "positive_rate_exit_mature")}
        meta.update({k: metas[0][k] for k in ("feature_dim", "n_startups", "n_mature")})
    stats = json.loads(GRAPH_STATS.read_text())
    edges = load_edges()

    out_dir.mkdir(parents=True, exist_ok=True)

    (out_dir / "metapath_inventory.tex").write_text(
        build_metapath_table(rows, meta["baseline_nfr"], meta["baseline_exit_mature"])
    )
    (out_dir / "graph_statistics.tex").write_text(
        build_graph_stats_table(stats, edges, meta)
    )

    print(f"Wrote {out_dir / 'metapath_inventory.tex'}")
    print(f"Wrote {out_dir / 'graph_statistics.tex'}")
    print()
    verify(rows, meta, stats)


if __name__ == "__main__":
    main()
