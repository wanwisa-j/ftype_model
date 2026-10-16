# Forest Type Model Data Pipeline Overview

## Introduction

This document provides a comprehensive overview of the data pipeline used for the forest type classification model. The pipeline consists of two main data sources that are processed, aligned, and integrated to create training and prediction datasets.

## Pipeline Architecture

```
Data Sources → Preprocessing → Alignment → Interpolation → Dataset Extraction → Model Training/Prediction
     ↓              ↓             ↓            ↓              ↓                    ↓
[GEE Data]    [Raster Clip]  [Pixel Align] [Time Series] [Feature Extract] [Model 2.2.0]
[Biophysical] [& Align]      [Processing]  [Interpolate]  [Ground Truth]   [Prediction]
```

## Data Sources Overview

### 1. Google Earth Engine (GEE) Data
- **Sentinel-1 Features**: VV, VH (radar backscatter)
- **Sentinel-2 Features**: GVMI, NDVI, SLAVI (optical indices)
- **Static Features**: SRTM elevation and slope
- **Format**: 21-band raster (weekly timesteps)
- **Temporal Coverage**: 4 weeks before to 17 weeks after (W4before to W16after)

### 2. Internal Biophysical Features
- **Features**: LAI, FAPAR, FCOVER (derived from Sentinel-2)
- **Source**: ESA biophysical processing pipeline
- **Processing**: Tile-based calculation and weekly aggregation
- **Format**: 21-band raster (same temporal structure as GEE data)

## Data Storage Structure

### Final Datacube Storage
```
/fs2/ftype_mflf/shared_data/datacube/2025/1.2.0/
├── mfl_cmf_area_project_2025_chr_zone_1/
│   ├── fapar/
│   ├── fcover/
│   ├── lai/
│   ├── gvmi/
│   ├── ndvi/
│   ├── slavi/
│   ├── vv/
│   ├── vh/
│   ├── elevation/
│   └── slope/
└── mfl_cmf_area_project_2025_chr_zone_2/
    └── [same structure]
```

### Naming Convention
```
cube_{source}_{feature}_{year}_W{start}before_to_W{end}after_21bands.tif

Examples:
- cube_s2_gvmi_2023_W4before_to_W16after_21bands.tif
- cube_biophys_fapar_2023_W4before_to_W16after_21bands.tif
```

## Processing Workflow

### Phase 1: Data Acquisition
1. **GEE Data**: Exported directly from Google Earth Engine API
2. **Biophysical Data**: Calculated from Sentinel-2 tiles using local processing

### Phase 2: Preprocessing
1. **Raster Alignment**: `src/raster_clip_align.py`
   - Ensures consistent pixel alignment across all data sources
   - Clips data to Area of Interest (AOI)
   - Maintains spatial consistency

### Phase 3: Temporal Processing
1. **Weekly Aggregation**: Biophysical data grouped into weekly composites
2. **Time Series Creation**: 21-band temporal datacubes
3. **Gap Filling**: `src/batch_interpolate.py` for missing data interpolation

### Phase 4: Dataset Creation
1. **Feature Extraction**: `scripts/extract_forest_type_dataset.py`
   - Extracts pixel values at ground survey locations
   - Creates training/validation datasets
2. **Data Querying**: `src/dataset_builder.py`
   - Manages dataset access and querying
   - Handles ground truth integration

### Phase 5: Model Application
1. **Training**: Uses extracted datasets with ground truth labels
2. **Prediction**: `src/prediction_ftype_2_2_0.py` (current version 2.2.0)

## Key Components

### Core Scripts
- `src/raster_clip_align.py`: Spatial alignment and clipping
- `src/batch_interpolate.py`: Time series gap filling
- `scripts/extract_forest_type_dataset.py`: Dataset extraction
- `src/dataset_builder.py`: Data access and management
- `src/prediction_ftype_2_2_0.py`: Model prediction

### Configuration Files
- `config/`: Contains YAML configuration files for different processing stages

### External Dependencies
- `/fs1/tor_project/esa_biophysicalop_local/src/cal_biophysi_local.py`: Biophysical calculation
- `notebook/04_prep_data-prediction_v2_1_1.ipynb`: Biophysical weekly aggregation

## Quality Control

1. **Spatial Consistency**: All rasters aligned to same grid
2. **Temporal Consistency**: Uniform 21-band weekly structure
3. **Data Validation**: Missing data interpolation ensures complete time series
4. **Ground Truth Integration**: Pixel-level extraction at survey locations

## Next Steps

This pipeline documentation serves as the foundation for:
- Model version control and reproducibility
- Data quality assurance
- Scaling to new areas of interest
- Integration of additional data sources

For detailed information on specific components, refer to the individual documentation files in this directory.
