# Heterogeneous Graph Neural Networks for Startup Success Prediction: A Multi-Task Benchmark

Code for the ICDM 2026 paper above. A leakage-free, two-snapshot benchmark of
heterogeneous GNNs (HAN, SeHGNN, R-GCN, Simple-HGN, Hetero²Net), a VenGNN
comparator, and tabular / homogeneous-GNN / zero-shot-LLM baselines on a
163,531-startup Crunchbase graph, jointly predicting Next Funding Round (NFR)
and Exit. SeHGNN reaches Precision@100 = 72.9% on NFR, a 5.7× lift over the
12.8% base rate; Exit is architecture-invariant (a label ceiling, not a model
bottleneck).

## Main results

Mean ± std over 20 seeds on the heterophily-curated graph variant (20 metapaths). All
models consume the same 32 Crunchbase-native scalar features. NFR (Next Funding
Round) covers all 32,707 test startups; Exit covers the mature subset (n=6,154).
**Bold** = best per column, _italic_ = second-best, † = ties the leader
(Wilcoxon signed-rank, Holm p > 0.05).

| Group | Model | NFR AUC-ROC | NFR AUC-PR | NFR F1 | NFR P@100 | Exit AUC-ROC | Exit AUC-PR | Exit F1 | Exit P@100 |
|---|---|--:|--:|--:|--:|--:|--:|--:|--:|
| Baselines | Random | 50.0±0.1 | 12.8±0.1 | 20.7±2.3 | 12.8±0.9 | 49.5±0.4 | 4.1±0.3 | 6.8±0.9 | 3.5±2.3 |
| | DegreeCentrality | 73.3±0.2 | 30.8±0.4 | — | 45.2±2.6 | 61.9±0.3 | 5.8±0.3 | — | 8.7±2.3 |
| LLM | Llama-3 8B | 73.5±0.5 | 24.9±0.6 | — | 37.0±3.1 | 64.9±0.4 | 6.4±0.3 | — | 6.0±2.3 |
| Tabular | MLP | 77.6±0.2 | 35.2±0.3 | 37.9±1.4 | 64.7±1.7 | 63.8±0.7 | 5.9±0.3 | 9.4±0.3 | 8.0±2.2 |
| | XGBoost | 79.4±0.0 | 37.5±0.1 | 39.9±1.1 | _70.5±1.5_ | 65.7±0.2 | 7.0±0.2 | 11.8±0.2 | **10.6±0.9** |
| Homogeneous GNN | GCN | 80.6±0.2 | 37.9±0.4 | 42.0±0.5 | 64.8±4.0 | 63.8±1.4 | 6.5±0.5 | 11.0±0.6 | 8.2±2.7 |
| | GraphSAGE | 81.0±0.1 | _38.6±0.3_ | 42.2±1.8† | 68.2±3.0 | 64.4±2.1 | 6.6±0.3 | 11.4±0.6 | 7.8±2.0 |
| Graph SSP | VenGNN-A | 70.9±0.1 | 26.9±0.2 | 38.0±0.2 | 44.2±4.5 | 58.1±1.0 | 5.3±0.2 | 10.1±0.4 | 5.3±2.1 |
| | VenGNN-B | 79.4±0.1 | 35.6±0.4 | 40.1±0.6 | 58.0±3.4 | 66.7±0.6 | 7.1±0.2 | 10.8±0.3 | 8.5±1.5 |
| | VenGNN-Full | 78.2±0.1 | 34.0±0.3 | 36.0±0.4 | 47.9±2.8 | 65.1±0.4 | 6.6±0.2 | 11.0±0.3 | 8.7±0.8 |
| Heterogeneous GNN | HAN | 79.5±0.4 | 36.2±0.5 | _42.5±0.6_ | 62.1±2.9 | 64.1±1.2 | 6.4±0.2 | 10.7±0.4 | 8.6±2.3 |
| | SeHGNN | **81.1±0.0** | **39.2±0.1** | 41.9±0.5 | **72.9±2.5** | 62.9±0.5 | 6.9±0.2 | 10.7±0.2 | 8.2±1.6 |
| | R-GCN | 80.7±0.1 | 37.5±0.3 | **43.0±0.3** | 62.5±3.7 | **68.2±0.4** | _7.1±0.2_ | **12.1±0.3** | 7.6±1.9 |
| | Simple-HGN | _81.1±0.3_† | 38.3±0.8 | 41.9±0.9 | 65.1±4.4 | 64.4±1.1 | 6.2±0.4 | 10.5±0.3 | 7.1±2.4 |
| | Hetero²Net | 80.5±0.1 | 37.8±0.4 | 35.8±1.3 | 66.2±3.4 | _67.7±0.6_ | **7.2±0.2** | _12.1±0.6_† | _8.9±2.8_ |

Reproduce this table (paper Table I): train the 20-seed grid, then aggregate and
test for significance:

```bash
for cfg in experiments/champion_configs/*_g4_heterophily.yaml \
           experiments/champion_configs/mlp.yaml \
           experiments/champion_configs/xgboost.yaml; do
    python scripts/replicate_best.py --champion-config "$cfg" \
        --seeds $(seq 0 19) --group "replicate_$(basename "$cfg" .yaml)"
done
python scripts/fetch_paper_metrics.py --source local \
    --output outputs/paper_results/main_table_metrics.csv
python scripts/significance_paper.py \
    --input outputs/paper_results/main_table_metrics.csv \
    --output outputs/paper_results/significance_main_table.json
```

Every other figure, table, and reported number maps to an explicit command in
[`REPRODUCE.md`](REPRODUCE.md), with the seed group and graph cache each step needs.

## Install

```bash
python -m venv .venv && source .venv/bin/activate
# Install torch + torch-geometric matching your CUDA build first:
#   https://pytorch.org/get-started/locally/
#   https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html
pip install -r requirements.txt
```

All results use the exact pins in `requirements.txt` (PyTorch 2.6.0, PyTorch
Geometric 2.6.1, Python 3.11).

## Data

The graph comes from a Crunchbase academic-license export; the license forbids
redistribution, so no data, UUIDs, or trained weights ship here. The 60/20/20
split is deterministic (seed 0 in `torch.random.fork_rng()`), so the same
export rebuilds the identical partition.

To run the pipeline without a license, generate a synthetic graph of the same
schema (`scripts/generate_synthetic_data.py`); see `REPRODUCE.md`.

## Repository layout

```
src/data_engineering/   raw Crunchbase -> typed graph CSVs
src/ml/                 preprocessing, graph assembly, models, training, eval,
                        calibration, explainability, downstream analysis
scripts/                paper-figure + reproduction scripts
experiments/            champion_configs (winning HPs) + sweep_configs (search spaces)
tests/                  unit + integration tests
```

## License

Code under the MIT License (see `LICENSE`). Crunchbase data is not included and
is subject to Crunchbase's own license.
