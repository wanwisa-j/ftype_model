# Model v2.2.1 — Forest Type Classification

## Overview

| Item | Value |
|---|---|
| Version | 2.2.1 |
| Script | `src/prediction_ftype_2_2_1.py` |
| Model file | `model/mfl_v2_2_1/hybrid_lstm_cnn_model-20250911.keras` |
| Scaling params | `model/mfl_v2_2_1/config/scaling_params_v2_2_1-*.npz` (auto-detected) |
| Label encoder | `model/mfl_v2_2_1/config/label_encoder_v2_2_1.joblib` |
| Task | Multi-class forest type classification |
| Classes | 5 classes (DDF, DEF, MDF, + others via label encoder) |
| Supports | v2.2.0, v2.2.1, v2.2.2 (version-agnostic runner) |

## Model Architecture

**Hybrid CNN** — identical structure to v2.2.0, dual-input

```
Input 1: Time Series (21, 10) ──► [CNN / LSTM branch] ─┐
                                                         ├──► Dense ──► Softmax (5 classes)
Input 2: Static (2,)          ──► [Dense branch]       ─┘
```

- Invoked as: `model.predict([X_timeseries_scaled, X_static_scaled])`

## Features

### Input 1 — Time Series: `(samples, 21, 10)`

**21 timesteps** = last 4 ISO weeks (Mon-start) of previous year + first 17 ISO weeks of target year

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

> **Critical**: Channel order must exactly match training. Mismatch → prediction degenerates (99.9% single class). This was the root cause fixed in this version.

### Input 2 — Static Features: `(samples, 2)`

| Channel | Feature | Description |
|---|---|---|
| 0 | Slope | degrees — **must be channel 0** |
| 1 | Elevation | meters — **must be channel 1** |

> v2.0.0 had elevation/slope reversed. This version enforces correct order matching training script.

## Preprocessing

Same as v2.2.0. See [model_v2_2_0.md](model_v2_2_0.md) for raster loading, union masking, reshape details.

### Z-score scaling

```python
X_ts_scaled = (X_ts - timeseries_mean) / timeseries_std
X_static_scaled = (X_static - static_mean) / static_std
```

Parameters loaded from auto-detected `.npz` file (see Artifact Detection below).

## Artifact Auto-Detection

Script resolves model artifacts by version string automatically:

```
model/mfl_v{major}_{minor}_{patch}/
    ├── hybrid_lstm_cnn_model-*.keras      ← model (latest sorted)
    └── config/
        ├── scaling_params_v{ver}_*.npz   ← scaling (with fallback chain)
        └── label_encoder_v{ver}.joblib   ← label encoder
```

**Scaling param fallback chain** (tried in order):
1. `scaling_params_v{ver}_no_leakage-*.npz`
2. `scaling_params_v{ver}-*.npz`
3. `scaling_params_v{ver}.npz`
4. `scaling_params_from_training_hybrid_lstm_cnn_model-*.npz`
5. `scaling_params_from_training.npz`

## Label Encoder

Loaded from `label_encoder_v2_2_1.joblib`. Maps predicted class index → forest type name (e.g., `DDF`, `DEF`, `MDF`). Without it, output contains raw indices only.

## Inference Workflow

```
--model_version 2.2.1
        │
        ▼
detect_model_artifacts() → model path, scaling path, label encoder path
        │
        ▼
Load rasters → union mask → reshape → (n_valid, 21, 8)
        │
        ▼
Append VH/VV + VV/VH → (n_valid, 21, 10)
        │
        ▼
Z-score scale → model.predict([X_ts, X_static]) → (n_valid, 5)
        │
        ▼
argmax → label_encoder.inverse_transform() → class names → GeoTIFF
```

## Output

| File | Format | Description |
|---|---|---|
| `pred_{year}_{ver}-class.tif` | int16, nodata=-999 | Per-year class map |
| `pred_{year}_{ver}-proba.tif` | float32, 5 bands, nodata=-999 | Per-class probabilities |
| `pred_majority_{ver}.tif` | int16, nodata=-999 | Majority vote across years |

## Data Paths

| Data | Path |
|---|---|
| Datacube | `/fs2/ftype_mflf/shared_data/datacube/2025/1.2.0/{zone}/` |
| Output | `/fs2/ftype_mflf/result/2025/1.2.0/{zone}/model_version_{ver}/` |

## Usage

```bash
# Auto-detect artifacts (recommended)
python src/prediction_ftype_2_2_1.py \
  --model_version 2.2.1 \
  --zones zone_1,zone_2,zone_3 \
  --years 2019,2020,2021,2022,2023,2024 \
  --majority_vote

# Manual path override
python src/prediction_ftype_2_2_1.py \
  --model_version 2.2.1 \
  --model_path /path/to/model.keras \
  --config_dir /path/to/config \
  --zones zone_1 --years 2023
```

## Key Fixes vs v2.2.0

| Issue | v2.2.0 | v2.2.1 |
|---|---|---|
| Feature channel order | Not enforced | Strict enforcement + logging |
| Static feature order | `elevation, slope` (wrong) | `slope, elevation` (correct) |
| Scaling param loading | Single hardcoded path | Fallback chain + auto-detect |
| Label encoder | Not loaded | Required, auto-detected |
| Class output | Raw indices | Named classes via encoder |
| Path config | Manual only | Auto-detection by version |
| Multi-version support | v2.2.0 only | Any v2.2.x |
