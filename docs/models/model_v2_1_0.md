# Model v2.1.0 — Forest Type Classification

## Overview

| Item | Value |
|---|---|
| Version | 2.1.0 |
| Training notebook | `notebook/03_training_ftype-model_v2_1_0.ipynb` |
| Prediction script | *(none released — training-stage version, succeeded by v2.1.1)* |
| Model file | `model/mfl_v2_1_0/hybrid_lstm_cnn_model-20250825.keras` |
| Best checkpoint | `model/mfl_v2_1_0/best_model.h5` |
| Scaling params | Computed in memory (**not persisted** to `.npz`) |
| Task | Multi-class forest type classification |
| Classes | 5 classes: `ddf`, `def`, `delh`, `mdf`, `mix` |
| Total parameters | **70,309** (-45% vs v2.0.0) |

## Why v2.1.0 (vs v2.0.0)

Focus on training-stability + imbalanced-class handling:
- Smaller model (~45% parameter reduction) to curb overfitting on minority classes.
- **Focal loss** (`CategoricalFocalCrossentropy`, γ=2.0) replaces sparse categorical loss.
- **Stratified split** + explicit **hold-out test set** introduced.

## Model Architecture

**Hybrid LSTM-CNN** — same 3-stream topology as v2.0.0, downsized

```
Input 1: Biophysical TS (60, 3) ──► LSTM(32) ───────────┐
Input 2: Indices TS    (4, 6)  ──► Conv1D(32)→Conv1D(64)┼──► Concatenate ──► Dense ──► Softmax (5)
Input 3: Static        (2,)    ──► Dense                ─┘
```

| Branch | Layer | Units/Filters (v2.0.0 → v2.1.0) |
|---|---|---|
| Biophys | LSTM | 64 → **32** |
| Indices | Conv1D × 2 | 64→128 → **32→64** |
| Head | Dense → Softmax | 5 |

## Features

Same as v2.0.0:
- Input 1 — Biophys TS `(N, 60, 3)`: LAI, FAPAR, FCOVER (sliced `[:, 31:-15, :]`).
- Input 2 — Indices TS `(N, 4, 6)`: VV, VH, GVMI, NDVI, SLAVI, VV/VH ratio.
- Input 3 — Static `(N, 2)`: elevation, slope.

## Data Paths

| Data | Path |
|---|---|
| Training dataset | `/fs2/ftype_mflf/shared_data/dataset/2024/1.1.0/dataset/` (separate `.npy` files) |

## Training Configuration

| Item | Value |
|---|---|
| Dataset version | `2024/1.1.0/dataset` (same as v2.0.0) |
| Samples | **341,630** |
| Class mapping | `ddf` (0), `def` (1), `delh` (2), `mdf` (3), `mix` (4) |
| Train/val split | **Stratified 70/30** + in-training val 10% via `StratifiedShuffleSplit` |
| Hold-out test set | **102,489 samples** (stratified) |
| Loss | **`CategoricalFocalCrossentropy`** (γ = 2.0, from_logits=False) |
| Optimizer | Adam, LR = `1e-3` |
| Epochs | 60 |
| Batch size | 32 |
| Class imbalance | `class_weight='balanced'` (minority `delh` ≈ 0.5% → weight ≈ 19.56) + focal loss |
| Scaling | Z-score on training set (mean/std held in memory) |
| Callbacks | EarlyStopping(patience=10), ReduceLROnPlateau(factor=0.5, patience=5, min_lr=1e-6), ModelCheckpoint(best_only), **`DistributionMonitor`** (new) |
| Reported metrics | Accuracy ≈ 63%, macro F1 ≈ 0.67 on hold-out |

### DistributionMonitor

Custom callback — logs predicted-class distribution each epoch to catch collapse to majority class early.

## Output Artifacts

| File | Description |
|---|---|
| `hybrid_lstm_cnn_model-20250825.keras` | Final trained model |
| `best_model.h5` | Best-val checkpoint |
| *(no `.npz` scaling)* | Scaling params **not written** — inference must recompute (risk of drift) |

## Key Changes vs v2.0.0

| Area | v2.0.0 | v2.1.0 |
|---|---|---|
| Loss | sparse_categorical | **CategoricalFocalCrossentropy (γ=2)** |
| Parameters | 126,789 | **70,309** (-45%) |
| Train/val split | Simple 70/30 | **Stratified + 102,489 hold-out** |
| In-training val | ❌ | **StratifiedShuffleSplit 10%** |
| Callbacks | Standard | + `DistributionMonitor` |
| Class labels | def/mdf×3/mixed | **ddf, def, delh, mdf, mix** (cleaned) |

## Limitations

- Scaling params still not persisted — fixed in v2.1.1.
- 5 classes retained (`delh` minority still unstable) — collapsed to 4 in v2.1.1.
- Dataset still `1.1.0` — superseded by cleaner `1.1.1` in v2.1.1.
- No outlier filtering — introduced in v2.2.0.
