# Data Processing and Alignment Documentation

## Overview

This document describes the critical data processing steps that ensure spatial and temporal consistency across all data sources in the forest type classification pipeline. The alignment process is essential for integrating GEE-derived features with biophysical parameters.

## Spatial Alignment Process

### Primary Tool
**Script**: `src/raster_clip_align.py`

### Objectives
1. **Pixel Registration**: Ensure exact pixel-to-pixel correspondence
2. **Projection Consistency**: Harmonize coordinate reference systems
3. **Spatial Extent**: Clip all data to identical AOI boundaries
4. **Resolution Matching**: Maintain consistent 10m spatial resolution

### Input Data Sources
```
GEE Exports:
├── cube_s1_vv_2023_W4before_to_W16after_21bands.tif
├── cube_s1_vh_2023_W4before_to_W16after_21bands.tif
├── cube_s2_gvmi_2023_W4before_to_W16after_21bands.tif
├── cube_s2_ndvi_2023_W4before_to_W16after_21bands.tif
├── cube_s2_slavi_2023_W4before_to_W16after_21bands.tif
├── elevation_srtm.tif
└── slope_srtm.tif

Biophysical Data:
├── cube_biophys_fapar_2023_W4before_to_W16after_21bands.tif
├── cube_biophys_fcover_2023_W4before_to_W16after_21bands.tif
└── cube_biophys_lai_2023_W4before_to_W16after_21bands.tif
```

### Processing Steps

#### Step 1: Reference Grid Definition
```python
# Pseudo-code workflow
reference_raster = load_gee_export(vv_data)  # Use VV as spatial reference
target_crs = reference_raster.crs
target_transform = reference_raster.transform
target_bounds = reference_raster.bounds
target_shape = reference_raster.shape
```

#### Step 2: Spatial Resampling
- **Method**: Nearest neighbor for categorical data, bilinear for continuous
- **Target Resolution**: 10m x 10m pixels
- **Registration**: Sub-pixel accuracy (<0.1 pixel offset)

#### Step 3: Extent Clipping
- **AOI Boundaries**: Exact match to GEE export extent
- **NoData Handling**: Consistent masking across all layers
- **Edge Treatment**: Buffer removal to avoid edge effects

#### Step 4: Quality Validation
- **Pixel Count Verification**: Ensure identical dimensions
- **Coordinate Validation**: Check corner coordinates match
- **Projection Validation**: Verify CRS consistency

## Temporal Interpolation Process

### Primary Tool
**Script**: `src/batch_interpolate.py`

### Interpolation Strategy
The pipeline handles missing data through sophisticated temporal interpolation to ensure complete time series for all features.

### Gap Detection
```python
# Gap types handled:
- Single week gaps: Linear interpolation
- Multi-week gaps (2-3 weeks): Spline interpolation  
- Extended gaps (>3 weeks): Seasonal interpolation using climatology
- Start/end gaps: Backward/forward filling with trend analysis
```

### Interpolation Methods

#### Linear Interpolation
- **Application**: Single week gaps
- **Method**: Simple linear interpolation between adjacent valid observations
- **Quality**: Maintains temporal trends, suitable for short gaps

#### Spline Interpolation  
- **Application**: 2-3 week gaps
- **Method**: Cubic spline fitting through valid observations
- **Quality**: Smooth transitions, preserves local patterns

#### Seasonal Interpolation
- **Application**: Extended gaps (>3 weeks)
- **Method**: Use climatological patterns from multi-year data
- **Quality**: Maintains seasonal patterns, less precise for specific year

#### Quality Flags
Each interpolated pixel receives quality flags:
- `0`: Original observation
- `1`: Linear interpolation
- `2`: Spline interpolation  
- `3`: Seasonal interpolation
- `9`: No interpolation possible

### Processing Workflow

#### Batch Processing Structure
```bash
# Example batch processing command
python src/batch_interpolate.py \
  --input-dir /fs2/ftype_mflf/shared_data/datacube/2025/1.2.0/ \
  --zone mfl_cmf_area_project_2025_chr_zone_1 \
  --features fapar,fcover,lai,gvmi,ndvi,slavi,vv,vh \
  --output-dir /fs2/ftype_mflf/shared_data/datacube/2025/1.2.0/ \
  --quality-flags
```

#### Processing Order
1. **Feature-by-Feature**: Process each feature independently
2. **Pixel-by-Pixel**: Individual time series interpolation
3. **Quality Assessment**: Generate interpolation quality maps
4. **Validation**: Check temporal continuity and value ranges

## Output Data Structure

### Final Datacube Organization
```
/fs2/ftype_mflf/shared_data/datacube/2025/1.2.0/
├── mfl_cmf_area_project_2025_chr_zone_1/
│   ├── fapar/
│   │   ├── cube_biophys_fapar_2023_W4before_to_W16after_21bands.tif
│   │   └── cube_biophys_fapar_2023_W4before_to_W16after_21bands_quality.tif
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

### Data Format Specifications
- **File Format**: GeoTIFF with LZW compression
- **Data Type**: Float32 for continuous variables
- **NoData Value**: -9999 (consistent across all layers)
- **Band Count**: 21 bands per feature (temporal series)
- **Spatial Reference**: UTM zone-specific (EPSG:32600+ zone number)

## Quality Control and Validation

### Spatial Validation
```python
# Validation checks performed:
1. Pixel registration accuracy (<0.1 pixel offset)
2. Extent consistency (identical bounds)
3. Resolution verification (10m pixels)
4. Projection validation (matching CRS)
5. NoData consistency (same mask across features)
```

### Temporal Validation
```python
# Temporal quality checks:
1. Complete time series (21 bands present)
2. Value range validation (feature-specific limits)
3. Temporal continuity (no abrupt changes)
4. Interpolation quality assessment
5. Gap distribution analysis
```

### Cross-Feature Validation
- **Correlation Analysis**: Expected relationships between features
- **Physical Consistency**: Biophysical parameter relationships
- **Spatial Patterns**: Consistent landscape patterns across features

## Performance Optimization

### Processing Efficiency
- **Parallel Processing**: Multi-core utilization for pixel-wise operations
- **Memory Management**: Chunk-based processing for large rasters
- **I/O Optimization**: Efficient reading/writing with GDAL

### Storage Optimization
- **Compression**: LZW compression for ~50% size reduction
- **Tiling**: Internal tiling for faster access
- **Overviews**: Pyramid layers for visualization

## Error Handling and Recovery

### Common Issues and Solutions

#### Projection Mismatches
- **Detection**: Automatic CRS validation
- **Solution**: Reproject to reference grid
- **Prevention**: Standardized export procedures

#### Missing Data Patterns
- **Cloud Contamination**: Temporal interpolation
- **Sensor Gaps**: Cross-sensor data fusion
- **Processing Failures**: Automatic retry mechanisms

#### Memory Limitations
- **Large Files**: Chunk-based processing
- **Multiple Zones**: Sequential processing
- **Resource Management**: Automatic cleanup

## Configuration Management

### Configuration Files
```yaml
# Example processing configuration
spatial_alignment:
  reference_feature: "vv"
  target_resolution: 10
  resampling_method: "bilinear"
  
temporal_interpolation:
  max_gap_linear: 1
  max_gap_spline: 3
  use_climatology: true
  quality_flags: true

output:
  compression: "lzw"
  tiled: true
  overviews: true
```

### Zone-Specific Parameters
Different AOI zones may require specific processing parameters stored in zone-specific configuration files.

## Monitoring and Logging

### Processing Logs
- **Spatial Alignment**: Pixel registration accuracy metrics
- **Interpolation**: Gap statistics and quality metrics
- **Validation**: Quality control results
- **Performance**: Processing time and resource usage

### Quality Reports
Generated after each processing run:
- Alignment accuracy summary
- Interpolation statistics
- Data completeness metrics
- Quality flag distributions

## Related Documentation
- [GEE Data Source](./GEE_DATA_SOURCE.md)
- [Biophysical Data Source](./BIOPHYSICAL_DATA_SOURCE.md)
- [Dataset Extraction](./DATASET_EXTRACTION.md)
