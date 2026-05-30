# Reproducing the paper

One entry per paper figure, table, and reported number, each mapped to the
exact command(s) that produce it. Every script referenced here ships in this
repository (`scripts/`). Training is deterministic up to GPU non-determinism;
the CPU path is exactly reproducible.

## Prerequisites

1. **Environment** — install per `README.md` (`python -m venv` + `pip install
   -r requirements.txt`, with PyTorch / PyTorch Geometric matched to your CUDA
   build). All numbers were produced with the exact pins in `requirements.txt`.

2. **Data** — the real results need a Crunchbase academic-license export at
   `data/crunchbase/{2023,2025}/*.csv`. No Crunchbase data or identifiers are
   shipped (see `README.md`). The 60/20/20 split is deterministic (seed 0 in
   `torch.random.fork_rng()`), so the same export reconstructs the same split.
   To run the pipeline without a license, generate a synthetic graph (last
   section).

3. **Seeds** — all multi-seed results use the canonical set `{0, 1, …, 19}`.
   The CLI seed varies only training-init RNG (weight init, dropout, batch
   order); the data split is seed-independent.

4. **Graph variants** — built on demand from each champion config's
   `data_processing.graph_variant`; the cache (`outputs/graphs/`) is keyed on
   config and rebuilt every run (`graph_cache.rebuild: true`):
   - **g4_heterophily** — heterophily-pruned 20-meta-path graph; all model
     training and the main results (`*_g4_heterophily.yaml`).
   - **g1_full** — full 32-meta-path graph; the SeHGNN attention–MLH analysis
     only (`sehgnn_g1_full.yaml`).

5. **Weights & Biases is optional.** Every run writes its metrics to a local
   JSON under `outputs/pipeline_state/**/results/`. Every aggregation script
   below accepts `--source local` (passed explicitly in the commands): it
   discovers runs by scanning those JSONs on disk and classifies each by its
   loss weights, so no W&B account or submission registry is needed. `--source
   wandb` (pass `--group`/`--project`) and `--source registry` (reads the
   `*_submissions.json` files) remain available.

6. **Output paths** — the figure/table scripts default their output to
   `graph-paper/figures` / `graph-paper/tables` (the LaTeX tree, not shipped
   here). Pass `--out-dir`/`--output` to a local path such as `outputs/figures`.

---

## Tables

### Table I — Main performance (all architectures, both targets)

```bash
# 1. Train the 20-seed grid for each architecture (one champion config per row).
#    replicate_best.py runs src/main.py once per seed (sequential; use a SLURM
#    array for parallelism) under a shared W&B group.
for cfg in experiments/champion_configs/*_g4_heterophily.yaml \
           experiments/champion_configs/mlp.yaml \
           experiments/champion_configs/xgboost.yaml; do
    arch=$(basename "$cfg" .yaml)
    python scripts/replicate_best.py \
        --champion-config "$cfg" \
        --seeds $(seq 0 19) \
        --group "replicate_${arch}"
done

# 2. Assemble the long-format table CSV from the local result JSONs (no W&B).
python scripts/fetch_paper_metrics.py --source local \
    --output outputs/paper_results/main_table_metrics.csv

# 3. Wilcoxon signed-rank + Holm-Bonferroni significance (the daggers).
python scripts/significance_paper.py \
    --input outputs/paper_results/main_table_metrics.csv \
    --output outputs/paper_results/significance_main_table.json
```

Each table cell is the per-`(arch, target, metric)` mean ± std over the 20
seeds in `main_table_metrics.csv`; bold/underline/dagger placement comes from
`significance_main_table.json`.

### Table II — Selected hyperparameters

The reported values are the sweep-selected winners stored verbatim in
`experiments/champion_configs/<arch>_g4_heterophily.yaml` (plus `mlp.yaml`,
`xgboost.yaml`). No script needed; read the YAMLs.

### Table III — Multi-task ablation on SeHGNN

```bash
# 1. Train SeHGNN in the four MTL modes (NFR-only, Exit-only, joint 1:1,
#    joint tuned) across the 20 seeds, e.g. joint-tuned:
python scripts/replicate_best.py \
    --champion-config experiments/champion_configs/sehgnn_g4_heterophily.yaml \
    --seeds $(seq 0 19) \
    --group replicate_SeHGNN__g4_heterophily
#    (NFR-only / Exit-only / 1:1 use the same config with
#     --train.loss.momentum_weight / liquidity_weight overrides; see section V.)

# 2. Aggregate the per-mode prediction CSVs into the ablation table
#    (writes to outputs/mtl_tradeoff/, where step 3 reads per_seed.csv).
python scripts/aggregate_mtl_tradeoff.py --source local

# 3. Wilcoxon + Holm daggers for the ablation.
python scripts/mtl_significance.py
```

### Supplementary — Graph statistics + meta-path inventory

```bash
# Node/edge cardinalities, degrees, class balance (--output is a JSON file).
python scripts/compute_graph_statistics.py \
    --graph-path <g4_graph.pt> \
    --output outputs/graph_statistics/graph_statistics_full.json

# Per-meta-path label homophily (MLH) / Dirichlet energy spectrum,
# averaged over 5 random-walk materializations.
for s in 42 0 1 2 3; do
    python scripts/compute_heterophily_spectrum.py --rw-seed $s
done
python scripts/aggregate_heterophily_seeds.py --seeds 42 0 1 2 3 \
    --out outputs/graph_statistics/heterophily_spectrum_multiseed.csv

# Render the LaTeX stat/inventory tables.
python scripts/generate_paper_tables.py --out-dir outputs/tables
```

---

## Figures

### Feature-importance bars — feature_importance_{momentum,liquidity}.pdf

Top-15 Expected-Gradients attributions per target, 20-seed mean +/- std.

Needs per-seed EG attribution CSVs laid out as
`outputs/explanations/ig_multiseed_eg/g4_heterophily/seed_<N>/startup_feature_importance_{mom,liq}_task_improved_data_full.csv`
(produced by running `src/main.py` with `--explain.enabled true` per seed).

```bash
python scripts/plot_ig_top15_g4.py \
    --multiseed-root outputs/explanations/ig_multiseed_eg \
    --out-dir outputs/figures
```

### Attention vs MLH — homophily_vs_attention_nfr_only.pdf

Per-meta-path SeHGNN attention vs label homophily on the full 32-path graph
(Pearson rho = 0.76 NFR / 0.68 Exit).

```bash
# 1. Average attention across the 20 SeHGNN g1_full seeds, join with per-meta-
#    path MLH, and compute correlations.
python scripts/aggregate_attention_homophily.py \
    --champion-config experiments/champion_configs/sehgnn_g1_full.yaml \
    --graph <g1_full_graph.pt> \
    --source local \
    --output-dir outputs/attention_homophily_g1_full_multiseed

# 2. Render the single-panel NFR figure.
python scripts/render_attention_homophily_nfr_only.py \
    --output outputs/figures/homophily_vs_attention_nfr_only.pdf
```

### Draftwise ego graph — ego_graph_draftwise.pdf

```bash
python scripts/case_study.py \
    --model-path <SeHGNN_seed15_g4_checkpoint.pt> \
    --graph-path <g4_full_graph.pt> \
    --champion-config experiments/champion_configs/sehgnn_g4_heterophily.yaml \
    --target-uuid <your-target-uuid> \
    --attribution-method gradient_shap --eg-n 100 --eg-sigma 0.1 --eg-seed 42
```

Writes `outputs/case_study/startup_<idx>_ego_graph_1hop_spring.pdf`; rename to
`ego_graph_draftwise.pdf`.

### Precision@K by stage / continent

`precision_at_k_with_ci_mom_{stages,continents}.pdf` — multi-seed P@K curves
with bootstrap CI bands, aggregated over the 20 SeHGNN g4 prediction CSVs:

```bash
python scripts/plot_aggregated_pk_curves.py \
    --arch SeHGNN --variant g4_heterophily --max-k 1100 \
    --output-dir outputs/figures/pk_aggregated
# writes precision_at_k_with_ci_mom_{stages,continents,...}.pdf into the output dir
```

### Precision@K by founding year — precision_by_year_mom.pdf

Both scripts read the SeHGNN prediction CSVs under `outputs/pipeline_state/`.
`plot_pk_by_year.py` defaults to all matching CSVs (latest-mtime per seed); pass
`--job-prefix <prefix>` only if you want to pin a specific sweep's run-id/job
prefix.

```bash
python scripts/extract_cohort_pk.py --arch SeHGNN --task mom --ks 100 500 1000
python scripts/plot_pk_by_year.py  --arch SeHGNN --task mom --ks 100 500 1000 \
    --output outputs/figures/precision_by_year_mom.pdf
```

---

## Reported numbers

### Calibration (ECE, sections/calibration_numbers.tex)

`--variant` is the MTL mode (`joint`, `balanced`, `stl_nfr`, `stl_exit`), not a
graph variant. Run the Table III training first so the per-seed checkpoints
exist.

```bash
python scripts/recompute_calibration.py \
    --source local --variant joint \
    --champion-config experiments/champion_configs/sehgnn_g4_heterophily.yaml \
    --seeds $(seq 0 19) \
    --output-dir outputs/calibration
```

### Schema graph statistics (section III / Fig. 1)

Node/edge counts and mean degrees come from `compute_graph_statistics.py` (see
the supplementary-tables block above).

---

## Statistical methodology and output schemas

`significance_paper.py` (Table I) and `mtl_significance.py` (Table III) both use
a paired, two-sided **Wilcoxon signed-rank test** with **Holm-Bonferroni**
correction applied within each (target, metric) cell; no correction is made
across cells. For Table I the comparisons are vs-leader — each non-leader
architecture against the cell's best, not all-pairs. For Table III they are
pairwise among the MTL variants (3 tests per metric).

Why this holds: the 20 seeds are paired (every architecture saw the same
train/val/test split per seed, so pairing removes between-split variance);
Wilcoxon is non-parametric (no normality assumption for skewed AUC-PR / P@k);
Holm-Bonferroni controls FWER at α=0.05 within each cell. Mean and median
pairwise differences are reported alongside p-values as effect sizes. Edge cases
handled in code: all-zero paired diffs → `p_raw=1.0` (never significant); scipy
NaN from ties → `p=1.0`; fewer than 6 paired seeds → excluded from the correction.

**`main_table_metrics.csv`** — long format, one row per (arch, target, metric,
seed): columns `arch, group, target, metric, seed, value, run_id`.

**`significance_main_table.json`** — one key per `<target>__<metric>` (e.g.
`NFR__AUC-PR`), each with `leader`, `leader_mean`/`leader_std`, `test`,
`correction`, `alpha`, and a `comparisons` list (`arch, n_paired, mean, std,
diff_mean, diff_median, p_raw, p_holm, sig, note`). `sig: true` means the leader
beats that architecture at `p_holm < 0.05`.

---

## Running without Crunchbase (synthetic smoke test)

```bash
python scripts/generate_synthetic_data.py --n-startups 20000   # clears stale cache
python src/main.py --train.model SeHGNN
```

The generator plants feature signal (a latent quality score) and structural
signal (quality-assortative investor/founder/city edges, tunable via
`--homophily` and `--feature-noise`), so feature-based and graph-based models
both train to AUC-ROC ~ 0.77-0.84. Absolute numbers are not comparable to the
real-data results.
