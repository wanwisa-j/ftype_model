# Google Earth Engine (GEE) Data Source Documentation

## Overview

The Google Earth Engine data source provides satellite-derived features for the forest type classification model. This includes both radar and optical sensor data, as well as static topographic features.

## Data Products

### Sentinel-1 Radar Features
- **VV Polarization**: Vertical transmit, vertical receive backscatter
- **VH Polarization**: Vertical transmit, horizontal receive backscatter
- **Processing Level**: Ground Range Detected (GRD)
- **Spatial Resolution**: ~10 meters
- **Temporal Resolution**: Weekly composites

### Sentinel-2 Optical Indices
- **GVMI (Global Vegetation Moisture Index)**
  - Formula: `(NIR + 0.1) - (SWIR + 0.02) / (NIR + 0.1) + (SWIR + 0.02)`
  - Sensitive to vegetation water content
  
- **NDVI (Normalized Difference Vegetation Index)**
  - Formula: `(NIR - Red) / (NIR + Red)`
  - Classic vegetation index
  
- **SLAVI (Specific Leaf Area Vegetation Index)**
  - Formula: `NIR / (Red + SWIR)`
  - Related to leaf structure and biochemistry

### Static Topographic Features
- **Elevation**: SRTM Digital Elevation Model (30m resolution)
- **Slope**: Calculated from SRTM DEM using GEE terrain algorithms

## Data Processing in GEE

### Temporal Aggregation
- **Weekly Composites**: Data aggregated into weekly periods
- **Composite Method**: Median compositing to reduce cloud/noise effects
- **Temporal Window**: 21 weeks total
  - 4 weeks before the reference period (W4before)
  - 17 weeks after the start (W16after)

### Quality Filtering
- **Cloud Masking**: Sentinel-2 cloud probability < 20%
- **Data Availability**: Minimum pixel count thresholds
- **Temporal Consistency**: Gap-filled using linear interpolation

## Export Configuration

### Area of Interest (AOI)
- **Zone Definition**: Based on MFL-CMF project areas
- **Example Zones**:
  - `mfl_cmf_area_project_2025_chr_zone_1`
  - `mfl_cmf_area_project_2025_chr_zone_2`

### Export Parameters
- **Spatial Resolution**: 10 meters
- **Coordinate System**: UTM (zone-specific)
- **Export Format**: GeoTIFF (21 bands)
- **File Naming**: `cube_s2_{feature}_{year}_W4before_to_W16after_21bands.tif`

## Storage Structure

### Primary Storage Location
```
/fs2/ftype_mflf/shared_data/gdrive/GEE_Exports_{zone}/
├── cube_s1_vv_2023_W4before_to_W16after_21bands.tif
├── cube_s1_vh_2023_W4before_to_W16after_21bands.tif
├── cube_s2_gvmi_2023_W4before_to_W16after_21bands.tif
├── cube_s2_ndvi_2023_W4before_to_W16after_21bands.tif
├── cube_s2_slavi_2023_W4before_to_W16after_21bands.tif
├── elevation_srtm.tif
└── slope_srtm.tif
```

### Band Structure
Each 21-band file contains:
- Band 1-4: Weeks before reference period (W4before to W1before)
- Band 5-21: Reference period and after (W0 to W16after)

## Data Quality Characteristics

### Temporal Coverage
- **Complete Time Series**: 21 consecutive weeks
- **Seasonal Representation**: Captures phenological changes
- **Reference Period**: Aligned with ground survey timing

### Spatial Coverage
- **Full AOI Coverage**: Complete coverage of each zone
- **Consistent Grid**: All features on same spatial grid
- **No Data Handling**: Masked pixels for water bodies, urban areas

### Data Gaps
- **Cloud Coverage**: Managed through temporal compositing
- **Sensor Gaps**: Landsat-8 backup for critical periods
- **Processing Gaps**: Interpolated using neighboring time periods

## Integration with Processing Pipeline

### Input to Alignment Process
GEE exports serve as input to `src/raster_clip_align.py` for:
- Pixel-level alignment with biophysical data
- Consistent spatial reference system
- Uniform nodata handling

### Quality Control Steps
1. **Visual Inspection**: Automated quality checks
2. **Temporal Continuity**: Check for missing time steps
3. **Spatial Consistency**: Verify geographic alignment
4. **Value Range Validation**: Ensure realistic feature values

## Access and Authentication

### GEE Account Requirements
- **Service Account**: For automated processing
- **Asset Access**: Permissions for Sentinel collections
- **Export Quota**: Sufficient for large area processing

### Processing Scripts
- **GEE JavaScript**: Main export scripts
- **Python API**: Batch processing and monitoring
- **Configuration**: YAML files for different AOIs

## Known Limitations

### Temporal Constraints
- **Sentinel-1 Revisit**: 6-12 day cycle affects weekly composites
- **Seasonal Gaps**: Some periods with limited data availability
- **Processing Delays**: 2-3 day lag for recent data

### Spatial Constraints
- **Resolution Limits**: 10m minimum for Sentinel data
- **Edge Effects**: Potential issues at AOI boundaries
- **Projection Distortion**: Minimal at study area latitudes

## Updates and Maintenance

### Data Refresh Schedule
- **Annual Updates**: New year data processing
- **Seasonal Updates**: Critical growing season periods
- **Ad-hoc Updates**: As needed for model improvements

### Version Control
- **Export Versions**: Tracked in filename conventions
- **Processing Versions**: Documented in configuration files
- **Quality Metrics**: Stored with each export batch

## Related Documentation
- [Biophysical Data Source](./BIOPHYSICAL_DATA_SOURCE.md)
- [Data Alignment Process](./DATA_ALIGNMENT_PROCESS.md)
- [Feature Engineering](./feature_engineering_and_models.md)
