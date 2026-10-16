# Model Documentation Index

Forest type classification models — MFL project.

## Versions

| Version | Architecture | Inputs | Timesteps | Classes | Script / Notebook |
|---|---|---|---|---|---|
| [v1.0.0](model_v1_0_0.md) | LSTM-FCN | Single: indices only (4 ch) | 4 monthly | N (raw idx) | `src/prediction_ftype.py` |
| [v2.0.0](model_v2_0_0.md) | Hybrid LSTM-CNN (126,789 params) | 3 streams: biophys + indices + static | 60 (biophys), 4 (indices) | 5 | `src/prediciton_ftype_2_0_0.py` |
| [v2.1.0](model_v2_1_0.md) | Hybrid LSTM-CNN (70,309 params) | 3 streams (same as v2.0.0) | 60 / 4 | 5 (adds `ddf`, `delh`, `mix`) | *notebook only* |
| [v2.1.1](model_v2_1_1.md) | Hybrid LSTM-CNN (70,309 params) | 3 streams (same as v2.1.0) | 60 / 4 | 4 (drops `delh`) | *notebook only* |
| [v2.2.0](model_v2_2_0.md) | Hybrid CNN | 2 inputs: timeseries (10 ch) + static | 21 ISO weeks | 5 (inference) / 3 (notebook) | `src/prediction_ftype_2_2_0.py` |
| [v2.2.1](model_v2_2_1.md) | Hybrid CNN (same as v2.2.0) | 2 inputs: timeseries (10 ch) + static | 21 ISO weeks | 5 (named via encoder) | `src/prediction_ftype_2_2_1.py` |

## Feature Channel Reference (v2.2.x)

### Time Series — 21 timesteps (W4 prev year → W17 target year)

| Ch | Feature | Type |
|---|---|---|
| 0 | VV | Sentinel-1 SAR |
| 1 | VH | Sentinel-1 SAR |
| 2 | GVMI | Sentinel-2 optical |
| 3 | NDVI | Sentinel-2 optical |
| 4 | SLAVI | Sentinel-2 optical |
| 5 | LAI | Biophysical |
| 6 | FAPAR | Biophysical |
| 7 | FCOVER | Biophysical |
| 8 | VH/VV | Computed ratio |
| 9 | VV/VH | Computed ratio |

### Static — 2 features

| Ch | Feature |
|---|---|
| 0 | Slope (degrees) |
| 1 | Elevation (meters) |

## Training Evolution (v2.x)

| Epoch of change | What changed | Where |
|---|---|---|
| v2.0.0 → v2.1.0 | Focal loss (γ=2) + stratified split + hold-out + model down-sized 45% | [v2.1.0](model_v2_1_0.md) |
| v2.1.0 → v2.1.1 | Dataset upgraded `1.1.0` → `1.1.1` (341K → 6.8K curated samples), `delh` dropped, epochs 60 → 120, scaling `.npz` persisted | [v2.1.1](model_v2_1_1.md) |
| v2.1.1 → v2.2.0 | Combined 9-ch timeseries (vs separate branches), IQR outlier filter, spatial validation, 3-class focus (notebook), larger CNN filters, `timeseries_*` scaling keys | [v2.2.0](model_v2_2_0.md) |
| v2.2.0 → v2.2.1 | Enforced channel/static order, scaling fallback chain, label encoder, version-agnostic runner | [v2.2.1](model_v2_2_1.md) |

## Critical Notes

- **Channel order** must match training exactly (v2.2.x). Wrong order → class prediction collapse.
- **Scaling params** must come from training — never recompute at inference.
- **21-week structure**: ISO Mon-start, 4 weeks from prev year + 17 weeks from target year.
- **Label encoder** required for named class output (v2.2.1+).
- **Scaling `.npz` key schema changed** at v2.2.0: pre-v2.2.0 uses `biophys_*`/`indices_*`/`static_*`; v2.2.0+ uses `timeseries_*`/`static_*`. Not interchangeable.
