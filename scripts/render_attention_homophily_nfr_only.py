"""Re-render the NFR-only attention-vs-MLH figure from cached aggregate output.

Reads
  outputs/attention_homophily_g1_full_multiseed/consensus_attention_homophily.csv
  outputs/attention_homophily_g1_full_multiseed/correlation_summary.json

Writes
  graph-paper/figures/homophily_vs_attention_nfr_only.pdf

The aggregate driver (scripts/aggregate_attention_homophily.py) is the source
of truth for the consensus attention and homophily values; this script only
re-renders the figure with the paper-style adjustments (serif font, no title,
single labeled outlier) without re-running the GPU forward pass.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.compute_attention_homophily_correlation import (  # noqa: E402
    plot_defense_style_nfr_only,
)

CACHE_DIR = PROJECT_ROOT / "outputs" / "attention_homophily_g1_full_multiseed"
CSV_PATH = CACHE_DIR / "consensus_attention_homophily.csv"
JSON_PATH = CACHE_DIR / "correlation_summary.json"
OUT_PDF = PROJECT_ROOT / "graph-paper" / "figures" / "homophily_vs_attention_nfr_only.pdf"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input-dir", type=Path, default=CACHE_DIR,
                    help="Directory holding consensus_attention_homophily.csv + "
                         "correlation_summary.json from aggregate_attention_homophily.py. "
                         "Defaults to outputs/attention_homophily_g1_full_multiseed.")
    ap.add_argument("--output", type=Path, default=OUT_PDF,
                    help="Output PDF path. Defaults to the LaTeX tree "
                         "(graph-paper/figures); redirect to a local path such "
                         "as outputs/figures/ when that tree is absent.")
    args = ap.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    csv_path = args.input_dir / "consensus_attention_homophily.csv"
    json_path = args.input_dir / "correlation_summary.json"

    df = pd.read_csv(csv_path).rename(columns={"mean_attention": "attention_weight"})
    summary = json.loads(json_path.read_text())
    consensus = summary["consensus_correlations"]
    correlations = {
        col: (vals["pearson_rho"], vals["pearson_p"], vals["n"])
        for col, vals in consensus.items()
    }
    plot_defense_style_nfr_only(df, correlations, str(args.output))
    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
