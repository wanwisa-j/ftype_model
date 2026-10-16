# Model v2.1.1 — Forest Type Classification

## Overview

| Item | Value |
|---|---|
| Version | 2.1.1 |
| Training notebook | `notebook/03_training_ftype-model_v2_1_1.ipynb` |
| Prediction script | *(none dedicated — used via generic v2.x runner)* |
| Model file | `model/mfl_v2_1_1/hybrid_lstm_cnn_model-20250827.keras` |
| Best checkpoint | `model/mfl_v2_1_1/best_model.h5` |
| Scaling params | `model/mfl_v2_1_1/config/scaling_params_from_training.npz` ✓ |
| Task | Multi-class forest type classification |
| Classes | **4 classes**: `DDF`, `DEF`, `MDF`, `MIX` (dropped `delh`) |
| Total parameters | **70,309** (same as v2.1.0) |

## Why v2.1.1 (vs v2.1.0)

Data quality + reproducibility fixes:
- Upgraded to cleaner dataset `1.1.1` (hand-curated — 341K → **6,836 samples**, ≈50× fewer but higher quality).
- Dropped unreliable `delh` class (down to 4 classes).
- **Doubled training budget** (60 → 120 epochs) since smaller dataset converges slower per-epoch under focal loss.
- **Persist scaling params** to `.npz` — inference can now reuse training statistics (fixes inference drift seen in v2.0.0/v2.1.0).

## Model Architecture

Identical to v2.1.0 — no architectural change:

```
Input 1: Biophysical TS (60, 3) ──► LSTM(32) ───────────┐
Input 2: Indices TS    (4, 6)  ──► Conv1D(32)→Conv1D(64)┼──► Concatenate ──► Dense ──► Softmax (4)
Input 3: Static        (2,)    ──► Dense                ─┘
```

Output head reduced 5 → 4 units.

## Features

Same feature construction as v2.0.0/v2.1.0:
- Input 1 — Biophys TS `(N, 60, 3)`: LAI, FAPAR, FCOVER.
- Input 2 — Indices TS `(N, 4, 6)`: VV, VH, GVMI, NDVI, SLAVI, VV/VH ratio.
- Input 3 — Static `(N, 2)`: elevation, slope.

## Data Paths

| Data | Path |
|---|---|
| Training dataset | `/fs2/ftype_mflf/shared_data/dataset/2024/1.1.1/dataset/` ← upgraded |

## Training Configuration

| Item | Value |
|---|---|
| Dataset version | `2024/1.1.1/dataset` (cleaner, curated) |
| Samples | **6,836** (↓ 50× from v2.1.0) |
| Class mapping | `DDF` (0), `DEF` (1), `MDF` (2), `MIX` (3) — `delh` **removed** |
| Train/val split | **Stratified 70/30** + in-training val **20%** via `StratifiedShuffleSplit` |
| Hold-out test set | **2,051 samples** (stratified) |
| Loss | `CategoricalFocalCrossentropy` (γ = 2.0) |
| Optimizer | Adam, LR = `1e-3` |
| Epochs | **120** (↑ from 60) |
| Batch size | 32 |
| Class imbalance | `class_weight='balanced'` + focal loss |
| Scaling | Z-score — **persisted** to `scaling_params_from_training.npz` |
| Callbacks | EarlyStopping(patience=10), ReduceLROnPlateau(factor=0.5, patience=5, min_lr=1e-6), ModelCheckpoint(best_only), `DistributionMonitor` |
| Reported metrics | Accuracy ≈ 61% on hold-out |

### Scaling `.npz` format

Keys follow v2.0.0-era convention (still separate per-input):
```
biophys_mean, biophys_std
indices_mean, indices_std
static_mean,  static_std
```

v2.2.0 later consolidates to `timeseries_mean/std` + `static_mean/std`.

## Output Artifacts

| File | Description |
|---|---|
| `hybrid_lstm_cnn_model-20250827.keras` | Final trained model |
| `best_model.h5` | Best-val checkpoint |
| `config/scaling_params_from_training.npz` | **New** — training-set Z-score params |

## Key Changes vs v2.1.0

| Area | v2.1.0 | v2.1.1 |
|---|---|---|
| Dataset version | `1.1.0` | **`1.1.1`** (cleaner) |
| Samples | 341,630 | **6,836** (↓ 50×) |
| Classes | 5 (with `delh`) | **4** (`delh` removed) |
| Epochs | 60 | **120** |
| In-training val fraction | 10% | **20%** |
| Scaling params persisted | ❌ | ✓ `.npz` |
| Hold-out size | 102,489 | 2,051 |

## Limitations

- Very small dataset (6,836) → high variance in metrics across runs.
- Still uses separate-input architecture — inconsistent with unified 21-week datacubes used by the v2.2.x prediction pipeline.
- `MIX` class retained — dropped in v2.2.0 (3-class focus).
- `.npz` keys (`biophys_*`, `indices_*`) incompatible with v2.2.x loader (which expects `timeseries_*`).
