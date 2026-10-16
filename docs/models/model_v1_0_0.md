# Model v1.0.0 — Forest Type Classification

## Overview

| Item | Value |
|---|---|
| Version | 1.0.0 |
| Script | `src/prediction_ftype.py` |
| Model file | `model/lstm-fcn_dataset2_phase_1-2_20240508_3.h5` |
| Config | `config/config.yaml` |
| Task | Multi-class forest type classification |

## Model Architecture

**LSTM-FCN** (Long Short-Term Memory Fully Convolutional Network)

- Single-input architecture
- Input: `(samples, timesteps=4, features=4)`
- Output: Softmax probabilities over N classes → argmax → class index

## Features

### Time Series Input — Shape: `(samples, 4, 4)`

| Channel | Feature | Source | Notes |
|---|---|---|---|
| 0 | NDVI | Sentinel-2 | Normalized Difference Vegetation Index |
| 1 | GVMI | Sentinel-2 | Green Vegetation Moisture Index |
| 2 | SLAVI | Sentinel-2 | Specific Leaf Area Vegetation Index |
| 3 | S1 VV/VH ratio | Sentinel-1 | Computed: `VV / VH` per pixel |

- **Timesteps**: 4 monthly maxima (months 1–4 of target year)
- **File pattern**: `max_s2_{index}_{year}_{mm:02}*.tif`, `max_s1_VV_{year}_{mm:02}*.tif`

### No Static Features

Elevation and slope not used in this version.

## Preprocessing

1. Load 4 monthly rasters per feature
2. For S1: compute `VV / VH` ratio
3. Reshape: `(4_months, H, W)` → `(H*W, 4, 1)` per feature → `dstack` → `(H*W, 4, 4)`
4. No scaling applied

## Inference Workflow

```
feature_path/{year}/max_s2_ndvi_*.tif  ─┐
feature_path/{year}/max_s2_gvmi_*.tif  ─┤
feature_path/{year}/max_s2_slavi_*.tif ─┼─► reshape ──► model.predict() ──► argmax ──► GeoTIFF
feature_path/{year}/max_s1_VV_*.tif    ─┤
feature_path/{year}/max_s1_VH_*.tif   ─┘  (VV/VH ratio)
```

## Output

| File | Description |
|---|---|
| `pred_{year}_{version}.tif` | Per-year class prediction raster |
| `pred_mode_{version}.tif` | Majority vote across `retroactive_years` (default: 5) |

- Nodata: same as input raster dtype/nodata
- Classes: integer indices (0-based)

## Limitations

- Only 4 monthly timesteps (coarse temporal resolution)
- No biophysical features (LAI, FAPAR, FCOVER)
- No topographic features (elevation, slope)
- No probabilistic output
- No data scaling — raw DN values fed to model
- Single input head — cannot leverage multi-source heterogeneous data
