# Training Dataset Builder (`src/dataset_builder.py`)

> ย้ายมาจาก README เดิมของ repo ต้นทาง ใช้สำหรับสร้าง dataset ตอน train ไม่เกี่ยวกับการ predict ตาม AOI
> (ดู [AOI_PREDICTION.md](AOI_PREDICTION.md)) บางสคริปต์ที่อ้างถึงด้านล่าง เช่น `run_single_zone_verification.sh`
> และ `run_zone_verification_parallel.sh` ไม่มีอยู่ใน repo แล้ว ตัว builder อยู่ที่ `src/dataset_builder.py`
> ส่วน path ของ dataset ตั้งค่า default ไว้ที่ `paths.training_dataset` ใน `config/config.yaml`

This repository contains a comprehensive pipeline for forest type classification using multi-source satellite data. The pipeline includes data processing, quality verification, dataset creation, and model training components.

## Pipeline Overview

The complete pipeline processes satellite data through multiple stages:

1. **Data Sources**: Google Earth Engine exports and biophysical parameters
2. **Preprocessing**: Raster alignment and interpolation  
3. **Quality Verification**: Comprehensive datacube validation
4. **Dataset Creation**: Training dataset extraction from verified datacubes
5. **Model Training**: Forest type classification using verified data

## Key Features

- **Multi-source Data Integration**: Combines optical, SAR, and biophysical features
- **Automated Quality Assurance**: Comprehensive verification before dataset creation
- **Flexible Dataset Builder**: Create custom training datasets
- **Scalable Processing**: Support for multiple zones and years
- **Production Ready**: Robust error handling and documentation

## Quick Start

### 1. Datacube Verification
Before using any data, verify datacube quality:

```bash
# Verify single zone
./run_single_zone_verification.sh zone_1

# Verify all zones in parallel
./run_zone_verification_parallel.sh

# Sequential verification (lower resource usage)
./verify_all_zones_sequential.sh
```

### 2. Dataset Creation
After verification passes, create training datasets:

```python
from dataset_builder import ForestTypeDatasetBuilder

# Initialize builder
builder = ForestTypeDatasetBuilder()

# Create dataset from verified datacube
dataset = builder.create_dataset(
    zones=['mfl_cmf_2024', 'mfl_cmf_area_project_2025_chr_zone_1'],
    years=[2020.0, 2021.0],
    features=['vv', 'vh', 'ndvi', 'slope'],
    target='ftype_mix'
)

# Access the data
vv_features = dataset['vv_feature']      # Shape: (N, 21)
vh_features = dataset['vh_feature']      # Shape: (N, 21)
ndvi_features = dataset['ndvi_feature']  # Shape: (N, 21)
slope_features = dataset['slope_feature'] # Shape: (N, 1)
targets = dataset['target']             # Shape: (N, 1)
```

## Data Quality Assurance

The pipeline includes comprehensive quality verification to ensure datacube integrity:

- ✅ **Structure Verification**: All expected features present
- ✅ **Geometry Verification**: Consistent CRS, transforms, and shapes  
- ✅ **Temporal Verification**: Correct 21-band structure for time series
- ✅ **Data Quality**: NoData handling and interpolation validation

**Only use verified datacubes for dataset creation and model training.**

## Quick Start

```python
from dataset_builder import ForestTypeDatasetBuilder

# Initialize builder
builder = ForestTypeDatasetBuilder()

# Create dataset
dataset = builder.create_dataset(
    zones=['mfl_cmf_2024', 'mfl_cmf_area_project_2025_chr_zone_1'],
    years=[2020.0, 2021.0],
    features=['vv', 'vh', 'ndvi', 'slope'],
    target='ftype_mix'
)

# Access the data
vv_features = dataset['vv_feature']      # Shape: (N, 21)
vh_features = dataset['vh_feature']      # Shape: (N, 21)
ndvi_features = dataset['ndvi_feature']  # Shape: (N, 21)
slope_features = dataset['slope_feature'] # Shape: (N, 1)
targets = dataset['target']             # Shape: (N, 1)
```

## Available Data

### Zones
- `mfl_cmf_2024`: 1,556 points, years 2019-2024
- `mfl_cmf_area_project_2025_chr_zone_1`: 312 points, years 2019-2025
- `mfl_cmf_area_project_2025_chr_zone_2`: 145 points, years 2019-2025

### Features

**Temporal Features (21 timesteps):**
- `fapar`: Fraction of Absorbed Photosynthetically Active Radiation
- `fcover`: Fractional Cover
- `gvmi`: Global Vegetation Moisture Index
- `lai`: Leaf Area Index
- `ndvi`: Normalized Difference Vegetation Index
- `slavi`: Specific Leaf Area Vegetation Index
- `vh`: VH polarization SAR
- `vv`: VV polarization SAR

**Static Features (1 timestep):**
- `elevation`: Elevation above sea level
- `slope`: Terrain slope

### Target Variables
- `ftype_mix`: Forest type including mixed classes
- `ftype_nonmix`: Forest type excluding mixed classes

Target classes include: BAMBOO, BANANA, DDF, DEF, MDF, MIX, NONE, S

## Dataset Structure

The resulting dataset follows this structure:

```
For temporal features (e.g., vv, ndvi):
- Shape: (total_samples, 21)
- Where total_samples = sum of (zone_points × selected_years)

For static features (e.g., slope, elevation):
- Shape: (total_samples, 1)
- Static values repeated for each year-point combination

For targets:
- Shape: (total_samples, 1)
- Same target value used across all years for each point
```

## Example Calculations

For the example: zones=['mfl_cmf_2024', 'mfl_cmf_area_project_2025_chr_zone_1'], years=[2020, 2021, 2025]

**Expected samples:**
- mfl_cmf_2024: 1,556 points × 2 years (2020, 2021) = 3,112 samples
- mfl_cmf_area_project_2025_chr_zone_1: 312 points × 3 years (2020, 2021, 2025) = 936 samples
- **Total: 4,048 samples**

**Actual shapes:**
```
vv_feature: (4048, 21)     # Temporal feature
fapar_feature: (4048, 21)  # Temporal feature
slope_feature: (4048, 1)   # Static feature
target: (4048, 1)          # Target values
```

## Advanced Usage

### Get Dataset Summary
```python
summary = builder.get_dataset_summary()
print(f"Available zones: {summary['zones']}")
print(f"Available years: {summary['years']}")
print(f"Zone-year combinations: {summary['zone_year_combinations']}")
```

### Include Point Information
```python
dataset = builder.create_dataset(
    zones=['mfl_cmf_2024'],
    years=[2020.0],
    features=['vv', 'slope'],
    target='ftype_mix',
    return_point_info=True  # Include point metadata
)

# Access point information
point_info = dataset['point_info']
# Each entry contains: {'point_id', 'zone', 'year', 'utm_x', 'utm_y'}
```

### Save and Load Datasets
```python
import numpy as np

# Save dataset
np.savez_compressed('my_dataset.npz', **dataset)

# Load dataset
loaded = np.load('my_dataset.npz', allow_pickle=True)
vv_features = loaded['vv_feature']
targets = loaded['target']
```

## Data Validation

The builder automatically validates:
- Zone names exist in the dataset
- Years are available for specified zones
- Feature names are valid
- Target variable names are correct
- Data shapes are consistent across features and targets

Warnings are issued for:
- Zone-year combinations without data
- Missing temporal data (< 21 timesteps)
- Missing static data

## Best Practices

1. **Check Available Data**: Use `get_dataset_summary()` to understand what data is available before creating datasets.

2. **Handle Missing Years**: Some zones don't have data for all years. The builder will warn you and skip missing combinations.

3. **Feature Selection**: Mix temporal and static features based on your model requirements.

4. **Target Selection**: Choose between `ftype_mix` (includes mixed classes) and `ftype_nonmix` (pure classes only).

5. **Reproducibility**: Save your datasets with meaningful names and document the exact parameters used.

## Error Handling

Common errors and solutions:

**ValueError: Invalid zones/years/features**
- Check available options using `get_dataset_summary()`

**Shape mismatch errors**
- Usually indicates data inconsistency; check warnings for missing data

**No data found errors**
- Verify zone-year combinations exist in the dataset

## Performance Notes

- Loading the full dataset takes ~10-30 seconds depending on system
- Feature extraction is optimized but can take time for large combinations
- Consider saving frequently used datasets to avoid reprocessing

## Pipeline Components

### Core Scripts
- `datacube_zone_verification.py`: Comprehensive datacube verification
- `dataset_builder.py`: Training dataset creation from verified datacubes
- `src/prediction_ftype_2_2_0.py`: Forest type prediction model (v2.2.0)
- `src/raster_clip_align.py`: Spatial alignment and clipping
- `src/batch_interpolate.py`: Time series gap filling and interpolation

### Verification Tools
- `run_single_zone_verification.sh`: Individual zone testing
- `run_zone_verification_parallel.sh`: Parallel processing for all zones
- `verify_all_zones_sequential.sh`: Sequential processing (resource-friendly)

### Configuration
- `config/`: YAML configuration files for different processing stages

### Documentation
- `docs/final_pipline/`: Complete pipeline documentation
- `FINAL_VERIFICATION_SUMMARY.md`: Datacube v1.2.0 verification results
- `README.md`: This documentation

## File Structure

```
├── datacube_zone_verification.py    # Main verification script
├── dataset_builder.py               # Dataset creation from verified data
├── dataset_examples.py              # Usage examples
├── run_single_zone_verification.sh  # Single zone verification
├── run_zone_verification_parallel.sh # Parallel verification
├── verify_all_zones_sequential.sh   # Sequential verification
├── src/                             # Core processing scripts
├── config/                          # Configuration files
├── docs/final_pipline/              # Pipeline documentation
└── README.md                        # This documentation
```

## Dependencies

- pandas
- numpy
- pathlib
- warnings
- typing

All dependencies are available in the `gisworkspace-cupy` conda environment.
