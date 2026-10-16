# Model v2.2.0 — Forest Type Classification

## Overview

| Item | Value |
|---|---|
| Version | 2.2.0 |
| Training notebook | `notebook/03_training_ftype-model_v2_2_0.ipynb` |
| Prediction script | `src/prediction_ftype_2_2_0.py` |
| Inference model | `model/mfl_v2_2_0/hybrid_lstm_cnn_model-20250910.keras` (5 classes) |
| Training-notebook model | `model/mfl_v2_2_0/hybrid_lstm_cnn_model-20250903.keras` (3 classes: DDF, DEF, MDF) |
| Additional checkpoint | `model/mfl_v2_2_0/hybrid_lstm_cnn_model-20250904.keras` |
| Scaling params (inference) | `model/mfl_v2_2_0/config/scaling_params_from_training_hybrid_lstm_cnn_model-20250910.npz` |
| Scaling params (notebook) | `model/mfl_v2_2_0/config/scaling_params_from_training.npz` |
| Task | Multi-class forest type classification |
| Classes (inference) | 5 (indices 0–4) |
| Classes (training notebook) | 3 (DDF, DEF, MDF) |
| Total parameters | **~70,000** |

> ⚠️ **Checkpoint provenance**: the v2.2.0 training notebook committed to the 3-class checkpoint `20250903.keras`. The production prediction script (`prediction_ftype_2_2_0.py`) loads a later 5-class checkpoint `20250910.keras` produced from a follow-up training run. Documentation below covers both — the **Inference** sections describe the 5-class pipeline, the **Training** section describes the 3-class notebook.

## Model Architecture

**Hybrid CNN** — dual-input architecture

```
Input 1: Time Series (21, 10) ──► [CNN / LSTM branch] ─┐
                                                         ├──► Dense ──► Softmax (5 classes)
Input 2: Static (2,)          ──► [Dense branch]       ─┘
```

- Invoked as: `model.predict([X_timeseries_scaled, X_static_scaled])`

## Features

### Input 1 — Time Series: `(samples, 21, 10)`

**21 timesteps** = last 4 ISO weeks of previous year + first 17 ISO weeks of target year

| Channel | Feature | Source | Description |
|---|---|---|---|
| 0 | VV | Sentinel-1 | SAR backscatter, vertical-vertical |
| 1 | VH | Sentinel-1 | SAR backscatter, vertical-horizontal |
| 2 | GVMI | Sentinel-2 | Green Vegetation Moisture Index |
| 3 | NDVI | Sentinel-2 | Normalized Difference Vegetation Index |
| 4 | SLAVI | Sentinel-2 | Specific Leaf Area Vegetation Index |
| 5 | LAI | Biophysical | Leaf Area Index |
| 6 | FAPAR | Biophysical | Fraction of Absorbed PAR |
| 7 | FCOVER | Biophysical | Fraction of Vegetation Cover |
| 8 | VH/VV | Computed | `VH / VV`, div-by-zero guard `1e-6` |
| 9 | VV/VH | Computed | `VV / VH`, div-by-zero guard `1e-6` |

> **Channel order is critical.** Mismatch causes severe class bias (e.g., 99.9% single class).

### Input 2 — Static Features: `(samples, 2)`

| Channel | Feature | Description |
|---|---|---|
| 0 | Slope | degrees |
| 1 | Elevation | meters |

## Preprocessing

### 1. Raster loading

File naming convention per feature:

| Feature | File pattern |
|---|---|
| VV, VH | `{zone}/{feature}/cube_s1_{feature}_{year}_W4before_to_W16after_21bands.tif` |
| GVMI, NDVI, SLAVI | `{zone}/{feature}/cube_s2_{feature}_{year}_W4before_to_W16after_21bands.tif` |
| LAI, FAPAR, FCOVER | `{zone}/{feature}/cube_biophys_{feature}_{year}_W4before_to_W16after_21bands.tif` |
| Elevation | `{zone}/elevation/elevation_data.tif` |
| Slope | `{zone}/slope/slope_data.tif` |

### 2. Union masking

Pixel excluded if **any** feature has nodata (`NaN`, `-9999`, or `0` for static) across all timesteps.

### 3. Reshape

```
Timeseries raster: (21, H, W) → flatten → (H*W, 21) → mask → (n_valid, 21)
Stack 8 features → (n_valid, 21, 8)
Add VH/VV + VV/VH → (n_valid, 21, 10)

Static raster: (1, H, W) → flatten → (H*W,) → mask → (n_valid,)
Stack slope + elevation → (n_valid, 2)
```

### 4. Z-score scaling

```python
X_ts_scaled = (X_ts - timeseries_mean) / timeseries_std
X_static_scaled = (X_static - static_mean) / static_std
```

- Parameters from `.npz`: keys `timeseries_mean`, `timeseries_std`, `static_mean`, `static_std`
- **Must use training-set statistics** — not recomputed at inference

## Inference Workflow

```
Datacube rasters (21 bands each)
        │
        ▼
Union mask → valid pixels only
        │
        ▼
Stack → (n_valid, 21, 8) + compute ratios → (n_valid, 21, 10)
        │
        ▼
Z-score scale (training stats)
        │
        ▼
model.predict([X_ts, X_static]) → probabilities (n_valid, 5)
        │
        ▼
argmax → class indices → reshape to (H, W) → GeoTIFF
```

## Output

| File | Format | Description |
|---|---|---|
| `pred_{year}_2.2.0-class.tif` | int16, nodata=-999 | Per-year class map |
| `pred_{year}_2.2.0-proba.tif` | float32, 5 bands, nodata=-999 | Per-class probabilities |
| `pred_majority_2.2.0.tif` | int16, nodata=-999 | Pixel-wise majority vote across years |

## Data Paths

| Data | Path |
|---|---|
| Datacube | `/fs2/ftype_mflf/shared_data/datacube/2025/1.2.0/{zone}/` |
| Output | `/fs2/ftype_mflf/result/2025/1.2.0/{zone}/model_version_2.2.0/` |

## Usage

```bash
python src/prediction_ftype_2_2_0.py \
  --zones zone_1,zone_2,zone_3 \
  --years 2019,2020,2021,2022,2023,2024 \
  --majority_vote
```

## Training Configuration (notebook `03_training_ftype-model_v2_2_0.ipynb`)

Major refactor vs v2.1.1 — first version with **combined 9-channel timeseries** and **spatial validation**.

| Item | Value |
|---|---|
| Dataset version | `2024/1.1.1/dataset` (post-filter) |
| Samples | **6,836** (same pool as v2.1.1) |
| Class mapping (notebook) | **3 classes**: `DDF` (0), `DEF` (1), `MDF` (2) — `MIX` dropped |
| Outlier filter | **IQR method**, threshold = 1.5, `per_timestep=True` (new in v2.2.0) |
| Feature layout | **Combined timeseries** `(N, T, 9)` + Static `(N, 2)` — no separate branches |
| Channel order (training) | `[VV, VH, GVMI, NDVI, SLAVI, LAI, FAPAR, FCOVER, ratio]` + (VH/VV, VV/VH added at inference to reach 10) |
| Train/val split | Stratified + **spatial validation set** (pixels from held-out spatial tiles) |
| Loss | `CategoricalFocalCrossentropy` (γ = 2.0) |
| Optimizer | Adam, LR = `1e-3` |
| Epochs | 120 |
| Batch size | 32 |
| Architecture | LSTM(32) + Conv1D(64)→Conv1D(128) (CNN branch filters **increased** for richer timeseries) |
| Class imbalance | `class_weight='balanced'` + focal loss |
| Scaling | Z-score — saved to `scaling_params_from_training.npz` (keys: `timeseries_mean/std`, `static_mean/std`) |
| Callbacks | EarlyStopping(patience=10), ReduceLROnPlateau(factor=0.5, patience=5, min_lr=1e-6), ModelCheckpoint(best_only) |
| Helper | `prepare_data_for_prediction()` utility for consistent preprocessing |
| Notebook checkpoint | `hybrid_lstm_cnn_model-20250903.keras` (3 classes) |

### Key Additions vs v2.1.1

| Area | v2.1.1 | v2.2.0 |
|---|---|---|
| Feature layout | Separate (biophys / indices / static branches) | **Combined timeseries (9 ch)** + static |
| Outlier filter | ❌ | **IQR, thr=1.5, per-timestep** |
| Validation | Stratified hold-out | **+ spatial validation** (leakage-robust) |
| Classes | 4 (DDF/DEF/MDF/MIX) | **3 (DDF/DEF/MDF)** — MIX dropped |
| CNN filter count | 32→64 | **64→128** |
| Scaling `.npz` keys | `biophys_*`, `indices_*`, `static_*` | **`timeseries_*`, `static_*`** (consolidated) |

### Production checkpoint `20250910.keras`

Retrained after the notebook run with **5-class target** for the operational prediction pipeline; hyperparameters follow the same setup. Channel order at inference adds VH/VV + VV/VH to reach 10 channels — see inference sections above.

## Limitations

- Manual path specification required (no auto-detection)
- Label encoder not loaded — output is raw class indices
- Majority vote computed pixel-by-pixel (slow for large rasters)
