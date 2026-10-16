# Forest Type Model Data Pipeline Overview

## Introduction

This document provides a comprehensive overview of the data pipeline used for the forest type classification model. The pipeline consists of two main data sources that are processed, aligned, and integrated to create training and prediction datasets.

## Pipeline Architecture

```
Data Sources → Preprocessing → Alignment → Interpolation → Verification → Dataset Extraction → Model Training/Prediction
     ↓              ↓             ↓            ↓              ↓             ↓                    ↓
[GEE Data]    [Raster Clip]  [Pixel Align] [Time Series] [Datacube]   [Feature Extract] [Model 2.2.0]
[Biophysical] [& Align]      [Processing]  [Interpolate]  [Verification] [Ground Truth]   [Prediction]
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
1. **Weekly Aggregation**: `src/weekly_biophys_aggregation.py` - robust biophysical data aggregation pipeline
   - Fixes NoData issues from previous notebook approach
   - ISO week-based aggregation with proper interpolation
   - Quality assurance and validation throughout processing
2. **Time Series Creation**: 21-band temporal datacubes (W4before to W16after)  
3. **Gap Filling**: `src/batch_interpolate.py` for missing data interpolation

### Phase 4: Datacube Verification
1. **Comprehensive Verification**: `datacube_zone_verification.py`
   - Validates datacube completeness and quality
   - Checks geometric consistency and temporal structure
   - Ensures data integrity across all zones
2. **Batch Processing**: Shell scripts for parallel verification
   - `run_zone_verification_parallel.sh`: All zones simultaneously
   - `run_single_zone_verification.sh`: Individual zone testing
   - `verify_all_zones_sequential.sh`: Sequential processing

### Phase 5: Dataset Creation
1. **Feature Extraction**: `scripts/extract_forest_type_dataset.py`
   - Extracts pixel values at ground survey locations
   - Creates training/validation datasets
2. **Data Querying**: `src/dataset_builder.py`
   - Manages dataset access and querying
   - Handles ground truth integration

### Phase 6: Model Application
1. **Training**: `scripts/train_model_v2_2_1.py` - Uses extracted datasets with ground truth labels
2. **Prediction**: `src/prediction_ftype_2_2_1.py` (current version 2.2.2 - **FIXED**)

⚠️ **CRITICAL PIPELINE FIXES (September 11, 2025):**
- Fixed scaling parameters file path mismatch
- Corrected feature channel order to match training exactly
- Added mandatory label encoder loading
- Enhanced pipeline validation and error handling

## Key Components

### Core Scripts
- `src/raster_clip_align.py`: Spatial alignment and clipping
- `src/batch_interpolate.py`: Time series gap filling
- `datacube_zone_verification.py`: Comprehensive datacube verification
- `scripts/extract_forest_type_dataset.py`: Dataset extraction
- `src/dataset_builder.py`: Data access and management (point-based)
- `scripts/train_model_v2_2_1.py`: Model training with spatial separation
- `src/prediction_ftype_2_2_1.py`: Model prediction (**FIXED** - v2.2.2 compatible)

### Critical Pipeline Requirements (v2.2.2)
- **Feature Order Consistency**: Training and prediction must use identical channel orders
- **Scaling Parameters**: Version-specific scaling files are mandatory
- **Label Encoder**: Required for proper class name mapping (DDF, DEF, MDF)
- **Pipeline Validation**: Test predictions at training points before full deployment

### Configuration Files
- `config/`: Contains YAML configuration files for different processing stages

### External Dependencies
- `/fs1/tor_project/esa_biophysicalop_local/src/cal_biophysi_local.py`: Biophysical calculation
- `/fs1/tor_project/model_foresttype/src/weekly_biophys_aggregation.py`: Biophysical weekly aggregation (production pipeline)

## Quality Control

1. **Spatial Consistency**: All rasters aligned to same grid
2. **Temporal Consistency**: Uniform 21-band weekly structure
3. **Data Validation**: Missing data interpolation ensures complete time series
4. **Comprehensive Verification**: Automated datacube quality assessment
   - Structure verification (feature completeness)
   - Geometric consistency validation
   - Temporal dimension verification
   - Data quality assessment (NoData handling, value ranges)
5. **Ground Truth Integration**: Pixel-level extraction at survey locations

## ⚠️ Critical Pipeline Lessons Learned

### Training-Prediction Consistency is Mandatory
**September 11, 2025 Investigation:** A critical pipeline mismatch caused 99.9% prediction bias toward class 2.0, revealing that:

1. **Feature Channel Order** must be identical between training and prediction
   - Neural networks are extremely sensitive to input channel positions
   - Small ordering differences cause catastrophic prediction failures

2. **Scaling Parameters** must be version-consistent and properly loaded
   - Wrong scaling parameters shift features by orders of magnitude
   - Model interprets scaled features as out-of-distribution inputs

3. **Label Encoding** is mandatory for proper class interpretation
   - Raw model outputs (0, 1, 2) must map to class names (DDF, DEF, MDF)
   - Missing label encoder prevents proper result interpretation

### Pipeline Validation Strategy
- **Always test predictions at training locations first**
- **Monitor prediction class distributions for unrealistic bias**
- **Validate artifact loading (scaling, labels) before full runs**
- **Implement automated consistency checking between training and prediction**

### Key Reference Documents
- [MEMORY/Prediction_Pipeline_Mismatch_Issue.md](../MEMORY/Prediction_Pipeline_Mismatch_Issue.md): Complete investigation details
- [PREDICTION_PIPELINE.md](./PREDICTION_PIPELINE.md): Updated with critical fixes

## Next Steps

This pipeline documentation serves as the foundation for:
- Model version control and reproducibility
- Data quality assurance with mandatory consistency checking
- Scaling to new areas of interest with validated pipelines
- Integration of additional data sources with proper validation

For detailed information on specific components, refer to the individual documentation files in this directory.
