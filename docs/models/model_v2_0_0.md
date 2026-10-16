# Model v2.0.0 — Forest Type Classification

## Overview

| Item | Value |
|---|---|
| Version | 2.0.0 |
| Training notebook | `notebook/03_training_ftype-model_v2_0_0.ipynb` |
| Script | `src/prediciton_ftype_2_0_0.py` |
| Model file | `model/mfl_v2_0_0/hybrid_lstm_cnn_model-20250806.keras` |
| Best checkpoint | `model/mfl_v2_0_0/best_model.h5` |
| Task | Multi-class forest type classification |
| Classes | 5 classes (indices 0–4): `def`, `mdf` (×3 variants), `mixed` |
| Total parameters | **126,789** |

## Model Architecture

**Hybrid LSTM-CNN** — multi-input, 3 separate input streams

```
Input 1: Biophysical TS (60, 3) ──► LSTM(64) ────────────┐
Input 2: Indices TS    (4, 6)  ──► Conv1D(64)→Conv1D(128)┼──► Concatenate ──► Dense ──► Softmax (5 classes)
Input 3: Static        (2,)    ──► Dense                 ─┘
```

| Branch | Layer | Units/Filters |
|---|---|---|
| Biophys | LSTM | 64 |
| Indices | Conv1D × 2 | 64 → 128 |
| Static | Dense | — |
| Head | Dense → Softmax | 5 |

## Features

### Input 1 — Biophysical Time Series: `(samples, 60, 3)`

| Channel | Feature | Description |
|---|---|---|
| 0 | LAI | Leaf Area Index |
| 1 | FAPAR | Fraction of Absorbed Photosynthetically Active Radiation |
| 2 | FCOVER | Fraction of Vegetation Cover |

- Raw shape from file: `(samples, 105, 3)` → sliced `[:, 31:-15, :]` → 60 timesteps
- Source file: `{zone}/biophys_data_{year}.npy`

### Input 2 — Indices Time Series: `(samples, 4, 6)`

| Channel | Feature | Description |
|---|---|---|
| 0 | VV | SAR backscatter, vertical-vertical |
| 1 | VH | SAR backscatter, vertical-horizontal |
| 2 | GVMI | Green Vegetation Moisture Index |
| 3 | NDVI | Normalized Difference Vegetation Index |
| 4 | SLAVI | Specific Leaf Area Vegetation Index |
| 5 | VV/VH ratio | Computed: `VV / VH` (added during preprocessing) |

- Source file: `{zone}/indices_data_{year}.npy` (shape `(samples, 4, 5)` before ratio added)
- Division-by-zero guard: denominator clipped to `1e-6`

### Input 3 — Static Features: `(samples, 2)`

| Channel | Feature | Description |
|---|---|---|
| 0 | Elevation | meters |
| 1 | Slope | degrees |

- Source files: `{zone}/elev_data_all.npy`, `{zone}/slope_data_all.npy`

## Preprocessing

```python
# 1. Biophysical: slice temporal axis
biophys = biophys[:, 31:-15, :]          # 105 → 60 timesteps

# 2. Indices: add VV/VH ratio as 6th channel
vv_vh_ratio = VV / max(VH, 1e-6)
indices = concat([indices, vv_vh_ratio], axis=-1)   # (N, 4, 5) → (N, 4, 6)

# 3. Static: stack elevation + slope
static = np.vstack([elev, slope]).T      # (N, 2)
```

> **Note**: Scaling parameters loaded but applied — see model artifacts in `mfl_v2_0_0/config/`.

## Spatial Masking

- `mask_processing_{year}.tif` defines valid pixels (bool)
- Invalid pixels excluded before prediction, restored as `-999` nodata after

## Inference Workflow

```
biophys_data_{year}.npy ──► slice [31:-15] ──────────────────────────────────┐
indices_data_{year}.npy ──► add VV/VH ratio ─────────────────────────────────┼──► model.predict()
elev_data_all.npy       ──┐                                                   │
slope_data_all.npy      ──┴──► stack ────────────────────────────────────────┘
                                                                               │
                                                                        argmax / softmax
                                                                               │
                                                                     pred_{year}_2.0.0-class.tif
                                                                     pred_{year}_2.0.0-proba.tif (5 bands)
```

## Output

| File | Format | Description |
|---|---|---|
| `pred_{year}_2.0.0-class.tif` | int32, nodata=-999 | Per-year class map |
| `pred_{year}_2.0.0-proba.tif` | float32, 5 bands, nodata=-999 | Per-class probabilities |

> Majority vote output (`pred_mode_2.0.0.tif`) present in code but commented out — tuple collection disabled.

## Data Paths

| Data | Path pattern |
|---|---|
| Training dataset | `/fs2/ftype_mflf/shared_data/dataset/2024/1.1.0/dataset/` (separate `.npy` files) |
| Datacube | `/fs2/ftype_mflf/shared_data/datacube/2025/1.0.0/chm_project_2025/{zone}/` |
| Output | `/fs2/ftype_mflf/result/2025/1.0.0/chm_project_2025/{zone}/model_version_2.0.0/` |

## Training Configuration

| Item | Value |
|---|---|
| Dataset version | `2024/1.1.0/dataset` (separate biophys / indices / static `.npy`) |
| Samples | **341,630** |
| Class mapping | `def`, `mdf` (×3 variants), `mixed` — 5 indices |
| Train/val split | Simple 70/30 split (**no hold-out test set**) |
| Loss | `sparse_categorical_crossentropy` |
| Optimizer | Adam, LR = `1e-3` |
| Epochs | 60 |
| Batch size | 32 |
| Class imbalance | `class_weight='balanced'` (minority class ≈ 0.3% → weight ≈ 19.13). SMOTE disabled. |
| Scaling | Z-score (params **not persisted** to disk) |
| Callbacks | EarlyStopping(patience=10), ReduceLROnPlateau(factor=0.5, patience=5, min_lr=1e-6), ModelCheckpoint(best_only) |
| Reported accuracy | ~64% (val split, no hold-out) |

### Training vs Inference Feature Slicing

- Biophys raw `(N, 105, 3)` → sliced `[:, 31:-15, :]` → `(N, 60, 3)` in both training and inference.
- Indices raw `(N, 4, 5)` → VV/VH ratio added → `(N, 4, 6)`.

## Limitations

- Hardcoded to `zone_3` (not parameterized)
- Numpy binary format — requires pre-extracted `.npy` files
- Majority vote disabled in this version
- 4 monthly timesteps for indices (coarse, not ISO-week standardized)
- **No hold-out test set** — reported metrics likely optimistic
- **Scaling params not persisted** → inference recomputes stats, risking train/inference drift
- Sparse categorical loss under heavy class imbalance — later replaced by focal loss in v2.1.0
