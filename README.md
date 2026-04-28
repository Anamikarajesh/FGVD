# FGVD Vehicle Classification — Multi-method Benchmarks

Implementation suite extending the SGCN paper *Graph-Based Two-Three Wheeler Classification in Unconstrained Indian Roads* (ITSC 2024) to **all vehicle classes** (2W + 3W + 4W) at three label levels:

- **L1** — vehicle type (motorcycle / scooter / autorickshaw / car / bus / mini-bus / truck) — 7 classes
- **L2** — manufacturer (Honda / Hero / Maruti / …) — 39 classes
- **L3** — model (Activa / Splendor / City / …) — 200 classes

The paper is replicated with raw RGB+Gabor+Sobel features. We additionally evaluate Random Forest, GAT, an MLP CNN baseline, and a deep-feature variant of SGCN, all sharing the same data pipeline via `fgvd_utils.py`.

---

## Repository layout

```
fgvd_utils.py       # all shared logic (data, models, losses, training, metrics)
sgcn.ipynb          # paper SGCN — raw features, dynamic Gaussian edge weights
sgcn_df.ipynb       # SGCN — multilevel deep features
rf.ipynb            # Random Forest — raw (PCA->256) and deep features
gat.ipynb           # GAT — deep features, dynamic edge weights
cnn.ipynb           # MLP over mean-pooled deep features (sanity baseline)
results.ipynb       # cross-method comparison table + plots

FGVD_Graph_Handover/
    metadata.csv
    raw_features/{rgb,gabor,sobel}/{train,val,test}/<vehicle_id>.npy
    deep_features/multilevel/{train,val,test}/<vehicle_id>.npy
    master_grid_skeleton.npz   # 8-neighbour connectivity (preferred)
    # master_grid_adj.npz       # legacy shared adjacency (NOT used as weights)
    per_sample_adj/             # legacy per-sample edge weights (NOT used)
    README.md                   # data-format reference

checkpoints/{method}/{features}/{level}_{case}/   # best.pt, last.pt, metrics.json
plots/{method}/{features}/{level}_{case}_curves.png
results/                                         # comparison_table.csv + summary plots
```

`{method}` ∈ `{sgcn, gat, rf, cnn}` · `{features}` ∈ `{raw, deep}` · `{level}` ∈ `{L1, L2, L3}` · `{case}` ∈ `{all, tw_vs_all, ...}`.

---

## Why this departs from the paper

| Aspect | Paper | This repo |
|---|---|---|
| Vehicle scope | 2W + 3W only | 2W + 3W + 4W (full FGVD) |
| L1 classes | 3 | 7 |
| L2 classes | ~10 | 39 |
| L3 classes | ~30 | 200 |
| SGCN edge weights | Dynamic Gaussian per crop | Same (skeleton + computed weights) |
| Long-tail handling | None reported | **Tail-merge + masked-softmax hierarchy + class-balanced loss** for L2/L3 |
| Reported metrics | Top-1 acc | Top-1 + top-3 + top-5 + macro/weighted F1 + coverage |

L2 has 13 classes with <20 train samples; L3 has 112. Those are merged into `<parent>::other` buckets at threshold `n_min=20` (configurable) and **coverage** is reported alongside accuracy so the comparison stays honest.

---

## Setup

### Environment

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

PyG companion wheels (`torch-scatter`, `torch-sparse`, `torch-cluster`, `torch-spline-conv`) must match your local torch + CUDA build — install from <https://data.pyg.org/whl/>:

```bash
pip install torch-scatter torch-sparse torch-cluster torch-spline-conv \
  -f https://data.pyg.org/whl/torch-2.5.0+cu118.html
```

Replace the URL suffix to match `torch.__version__` and CUDA version.

### Data

The processed FGVD data must be at `FGVD_Graph_Handover/` — see [its README](FGVD_Graph_Handover/README.md) for the file layout. You need at least `metadata.csv`, `raw_features/`, and `deep_features/multilevel/`. The skeleton adjacency falls back to `master_grid_adj.npz` if `master_grid_skeleton.npz` is absent (only the connectivity is read; weights are recomputed dynamically per batch).

---

## Running experiments

Every notebook follows the same pattern:

```python
from fgvd_utils import ExperimentConfig, run_experiment

cfg = ExperimentConfig(method='gat', level='L2', feature_source='deep', epochs=60)
out = run_experiment(cfg)
```

`run_experiment` will:
1. Load metadata, build the level's label space, filter rows whose feature files exist.
2. Auto-enable hierarchy + tail-merge for L2/L3 (defaults: `tail_merge_min=20`).
3. Train with class-balanced loss (or logit-adjusted CE for flat L1) and resume-capable checkpointing.
4. Save `best.pt` / `last.pt` / `metrics.json` under `checkpoints/{method}/{features}/{level}_all/`.
5. Save the loss/accuracy plot under `plots/{method}/{features}/{level}_all_curves.png`.
6. Return a dict with metrics, history, predictions.

### Suggested run order

| Day | Notebook | Approx. wall time |
|---|---|---|
| 1 | `cnn.ipynb`, `rf.ipynb` (deep cells) | <1 hour |
| 1 | `rf.ipynb` (raw cells with PCA→256) | 1–2 hours |
| 2 | `gat.ipynb` | 4–6 hours |
| 3 | `sgcn.ipynb` | 4–6 hours |
| 4 | `sgcn_df.ipynb` | 4–6 hours |
| 5 | `results.ipynb` (comparison table + plots) | 5 minutes |

All four training notebooks support resume — re-running a level's cell continues from the last epoch.

---

## What `fgvd_utils.py` contains

- **Data:** `load_metadata`, `build_level_labels`, `apply_case`, `tail_merge`, `check_group_leakage`
- **Features:** `load_raw_stacked`, `load_deep_features`, `pool_deep`
- **Graphs:** `load_skeleton_edge_index`, `compute_gaussian_edge_weights`
- **Datasets:** `FGVDGraphDataset` (PyG), `FGVDPooledDataset` (CNN/MLP)
- **Models:** `SGCNModel`, `GATModel`, `DeepMLP`
- **Hierarchy:** `MaskedHierarchicalCE`, `build_parent_index`
- **Long-tail losses:** `LogitAdjustedCE`, `FocalLoss`, `compute_class_balanced_weights`
- **Training:** `fit_with_resume`, `run_epoch`, `evaluate` (with top-k)
- **Metrics:** `stratified_accuracy_by_support`, `per_parent_macro_f1`
- **Entry point:** `run_experiment(ExperimentConfig)`

---

## Methodology notes

### Adjacency / edge weights
The paper computes per-crop Gaussian edge weights `exp(-||RGB_u - RGB_v||² / 2σ²)` over the 8-neighbour grid. This repo computes those weights **dynamically on the GPU per batch** (`compute_gaussian_edge_weights`) for both SGCN variants and GAT. The pre-baked `master_grid_adj.npz` and `per_sample_adj/` files are therefore not consumed at training time. This fixes a bug in the original `sgcn.ipynb` which was reusing one shared edge-weight matrix across all 24,450 vehicles.

### Hierarchy via masked softmax
For L2/L3, training uses `MaskedHierarchicalCE`: for each sample, logits whose L1 (or L2) parent doesn't match the true parent are set to `-∞` before the cross-entropy. This forces the model to compete only inside the correct parent group during training — equivalent to a cascaded prediction head, but trained end-to-end.

### Tail merging
Classes with `train_count < n_min` (default 20) are merged into `"<parent>::other"`. Coverage (% of test samples whose true class survived merging) is reported in `metrics.json` and the comparison table.

### Class imbalance
- All training uses `compute_class_balanced_weights` (Cui et al., CVPR'19): `weight_c = (1-β)/(1-β^n_c)` with `β=0.999`.
- L1 can additionally use `LogitAdjustedCE` (Menon et al., ICLR'21).

---

## Troubleshooting

- **PyG warns about missing `torch_scatter` / `torch_sparse`**: usually a CUDA-version mismatch. PyG falls back to pure Python implementations — slower, but training still runs.
- **OOM on raw RF**: raw L1 features are `(4096, 8)` per sample. PCA→256 is applied automatically (`rf_pca_dim=256`). Lower if needed.
- **`master_grid_skeleton.npz` not found**: code falls back to `master_grid_adj.npz` and uses only its connectivity. No action required.
- **Resume logic refuses to load checkpoint**: the saved `label_classes` must match the current label space. If you change `tail_merge_min` or `case`, delete the old `last.pt` first.

---

## Reference

Original paper: *Graph-Based Two-Three Wheeler Classification in Unconstrained Indian Roads* 
