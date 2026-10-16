# Biophysical Data Source Documentation

## Overview

The biophysical data source provides vegetation biophysical parameters derived from Sentinel-2 imagery using the ESA SNAP Biophysical Processor. These features complement the GEE-derived optical indices with physically-based vegetation parameters.

## Biophysical Parameters

### LAI (Leaf Area Index)
- **Definition**: Total one-sided leaf area per unit ground area
- **Units**: m²/m² (dimensionless)
- **Range**: 0-8 (typical forest values: 2-6)
- **Physical Meaning**: Indicator of vegetation density and canopy structure

### FAPAR (Fraction of Absorbed Photosynthetically Active Radiation)
- **Definition**: Fraction of incoming PAR absorbed by green vegetation
- **Units**: Fraction (0-1)
- **Range**: 0-0.95
- **Physical Meaning**: Vegetation photosynthetic activity and productivity

### FCOVER (Fraction of Green Vegetation Cover)
- **Definition**: Fraction of ground covered by green vegetation
- **Units**: Fraction (0-1)
- **Range**: 0-1
- **Physical Meaning**: Vegetation coverage and canopy density

## Processing Pipeline

### Step 1: Sentinel-2 Tile Processing
**Source Code**: `/fs1/tor_project/esa_biophysicalop_local/src/cal_biophysi_local.py`

**Input Data**:
- Sentinel-2 Level-2A tiles
- Atmospheric correction applied
- Cloud masking performed

**Processing Parameters**:
- **Spatial Resolution**: 10m and 20m bands resampled to 10m
- **Quality Flags**: Invalid pixels masked
- **Algorithm**: Neural network-based ESA processor

### Step 2: Tile Merging and Clipping
**Storage Location**: `/fs2/ftype_mflf/shared_data/temp/biophys/2025/1.0.0/`

**Directory Structure**:
```
/fs2/ftype_mflf/shared_data/temp/biophys/2025/1.0.0/
├── mfl_cmf_area_project_2025_chr_zone_1/
│   ├── fapar/
│   │   ├── 20230101_fapar_clipped.tif
│   │   ├── 20230108_fapar_clipped.tif
│   │   └── ...
│   ├── fcover/
│   └── lai/
└── mfl_cmf_area_project_2025_chr_zone_2/
    └── [same structure]
```

**Processing Steps**:
1. **Tile Merging**: Combine overlapping Sentinel-2 tiles
2. **AOI Clipping**: Clip to specific zone boundaries
3. **Quality Control**: Remove invalid/cloudy pixels
4. **Date Organization**: Store by acquisition date

### Step 3: Weekly Temporal Aggregation
**Source Code**: `/fs1/tor_project/model_foresttype/notebook/04_prep_data-prediction_v2_1_1.ipynb`
**Section**: "Biophy" markdown cell

**Aggregation Method**:
- **Temporal Window**: 7-day windows (Monday to Sunday)
- **Composite Method**: Median composite within each week
- **Gap Handling**: Linear interpolation for missing weeks
- **Quality Weighting**: Higher weight for cloud-free pixels

**Output Structure**:
```
Weekly aggregated files organized by feature and week:
- W4before: 4 weeks before reference period
- W3before, W2before, W1before: Weeks leading to reference
- W0 to W16after: Reference period through end
```

### Step 4: Datacube Creation
**Output Location**: `/fs2/ftype_mflf/shared_data/datacube/2025/1.2.0/`

**File Format**:
```
cube_biophys_{feature}_{year}_W4before_to_W16after_21bands.tif

Examples:
- cube_biophys_fapar_2023_W4before_to_W16after_21bands.tif
- cube_biophys_fcover_2023_W4before_to_W16after_21bands.tif
- cube_biophys_lai_2023_W4before_to_W16after_21bands.tif
```

**Band Organization**:
- Band 1: W4before
- Band 2: W3before
- Band 3: W2before
- Band 4: W1before
- Band 5: W0 (reference week)
- Bands 6-21: W1after through W16after

## Data Characteristics

### Temporal Resolution
- **Original**: Variable (Sentinel-2 revisit cycle: 2-5 days at equator)
- **Processed**: Weekly composites (21 weeks total)
- **Coverage Period**: Aligned with GEE data temporal windows

### Spatial Resolution
- **Original**: 10m (resampled from mixed 10m/20m bands)
- **Final**: 10m (consistent with GEE data)
- **Projection**: UTM zone-specific

### Quality Metrics
- **Cloud Coverage**: <20% per composite period
- **Data Availability**: >80% pixel coverage per week
- **Temporal Continuity**: <2 consecutive weeks missing data

## Algorithm Details

### ESA SNAP Biophysical Processor
- **Method**: Neural network trained on PROSPECT+SAIL radiative transfer model
- **Input Bands**: Sentinel-2 bands 3,4,5,6,7,8A,11,12
- **Training Dataset**: Synthetic dataset from radiative transfer simulations
- **Validation**: Extensive field campaign data

### Quality Flags
- **Algorithm Success**: Neural network convergence flag
- **Input Quality**: Valid Sentinel-2 pixel flag
- **Geometric Quality**: Geometric accuracy assessment
- **Radiometric Quality**: Radiometric quality flag

## Data Quality Control

### Pre-processing Quality Control
1. **Sentinel-2 Scene Selection**:
   - Cloud coverage <30% per scene
   - No major geometric distortions
   - Complete spectral coverage

2. **Biophysical Calculation QC**:
   - Algorithm convergence check
   - Value range validation (LAI: 0-8, FAPAR/FCOVER: 0-1)
   - Spatial continuity assessment

### Post-processing Quality Control
1. **Temporal Consistency**:
   - Outlier detection using temporal filters
   - Smooth temporal profiles expected
   - Seasonal pattern validation

2. **Spatial Consistency**:
   - Edge effect detection and removal
   - Spatial autocorrelation analysis
   - Cross-validation with reference data

## Integration with Pipeline

### Alignment Process
**Script**: `src/raster_clip_align.py`
- **Spatial Alignment**: Exact pixel registration with GEE data
- **Projection Harmonization**: Consistent coordinate reference system
- **NoData Harmonization**: Unified missing data handling

### Interpolation Process
**Script**: `src/batch_interpolate.py`
- **Missing Data Detection**: Identify gaps in time series
- **Interpolation Method**: Linear interpolation between valid observations
- **Quality Preservation**: Maintain uncertainty estimates

## Known Limitations

### Algorithm Limitations
- **Saturation**: LAI may saturate at high values (>6)
- **Soil Background**: FCOVER sensitive to soil brightness
- **Atmospheric Effects**: Residual atmospheric contamination possible

### Temporal Limitations
- **Cloud Gaps**: Extended cloudy periods create data gaps
- **Seasonal Bias**: Better performance during growing season
- **Acquisition Gaps**: Sentinel-2 orbit gaps affect some periods

### Spatial Limitations
- **Edge Effects**: Potential artifacts at tile boundaries
- **Mixed Pixels**: Challenges in heterogeneous landscapes
- **Water Bodies**: Invalid retrievals over water

## Validation and Accuracy

### Reference Data
- **Field Measurements**: Ground-based LAI measurements
- **High-Resolution Imagery**: Validation using airborne/drone data
- **Cross-Sensor Validation**: Comparison with MODIS/VIIRS products

### Accuracy Metrics
- **LAI**: RMSE ~0.5-1.0 m²/m² depending on vegetation type
- **FAPAR**: RMSE ~0.05-0.1 depending on season
- **FCOVER**: RMSE ~0.05-0.15 depending on landscape heterogeneity

## Updates and Maintenance

### Processing Schedule
- **Daily Processing**: New Sentinel-2 acquisitions processed within 48 hours
- **Weekly Aggregation**: Updated every Monday
- **Quality Assessment**: Monthly quality reports

### Version Control
- **Algorithm Versions**: ESA processor version tracking
- **Processing Versions**: Pipeline version in directory structure
- **Data Versions**: Version 1.2.0 is current production version

## Related Documentation
- [GEE Data Source](./GEE_DATA_SOURCE.md)
- [Data Alignment Process](./DATA_ALIGNMENT_PROCESS.md)
- [Temporal Interpolation](./TEMPORAL_INTERPOLATION.md)
