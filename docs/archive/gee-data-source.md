# Google Earth Engine (GEE) Data Source Documentation

## Overview

The Google Earth Engine data source provides the majority of features for the forest type classification model, including Sentinel-1 SAR data, Sentinel-2 optical indices, and SRTM terrain data. This document details the GEE data acquisition, processing, and storage workflows.

## 🛰️ Data Products

### Sentinel-1 SAR Features
**Features**: `[vv, vh]`

- **Product**: Sentinel-1 Ground Range Detected (GRD)
- **Polarizations**: VV (Vertical-Vertical), VH (Vertical-Horizontal)
- **Resolution**: ~10m spatial resolution
- **Units**: dB (decibel backscatter coefficient)
- **Temporal**: Weekly composites
- **Processing**: Multi-temporal speckle filtering

### Sentinel-2 Optical Indices  
**Features**: `[gvmi, ndvi, slavi]`

- **Product**: Sentinel-2 MSI Level-2A (Surface Reflectance)
- **Resolution**: 10m spatial resolution (resampled where needed)
- **Units**: Dimensionless indices (0-1 range typical)
- **Temporal**: Weekly composites with cloud masking
- **Quality**: Cloud-free pixel selection

#### Index Definitions
```javascript
// GVMI - Green Vegetation Moisture Index
var gvmi = image.select('B8').subtract(image.select('B11'))
                .divide(image.select('B8').add(image.select('B11')))
                .rename('GVMI');

// NDVI - Normalized Difference Vegetation Index  
var ndvi = image.normalizedDifference(['B8', 'B4']).rename('NDVI');

// SLAVI - Specific Leaf Area Vegetation Index
var slavi = image.select('B8')
                 .divide(image.select('B4').add(image.select('B11')))
                 .rename('SLAVI');
```

### SRTM Terrain Data
**Features**: `[elevation, slope]`

- **Product**: SRTM Digital Elevation Model v4.1
- **Resolution**: ~30m spatial resolution (resampled to 10m)
- **Units**: 
  - Elevation: meters above sea level
  - Slope: degrees (0-90°)
- **Temporal**: Static (single value per pixel)
- **Processing**: Slope derived from elevation using GEE terrain algorithms

## 📁 Data Storage Structure

### Directory Organization
```
/fs2/ftype_mflf/shared_data/gdrive/GEE_Exports_[zone]/
├── cube_s1_vv_[year]_W4before_to_W16after_21bands.tif
├── cube_s1_vh_[year]_W4before_to_W16after_21bands.tif
├── cube_s2_gvmi_[year]_W4before_to_W16after_21bands.tif
├── cube_s2_ndvi_[year]_W4before_to_W16after_21bands.tif
├── cube_s2_slavi_[year]_W4before_to_W16after_21bands.tif
├── cube_srtm_elevation_[year]_W4before_to_W16after_21bands.tif
└── cube_srtm_slope_[year]_W4before_to_W16after_21bands.tif
```

### File Naming Convention
```
cube_[sensor]_[feature]_[year]_W4before_to_W16after_21bands.tif
```

**Components:**
- `sensor`: s1 (Sentinel-1), s2 (Sentinel-2), srtm (SRTM)
- `feature`: vv, vh, gvmi, ndvi, slavi, elevation, slope
- `year`: 2019, 2020, 2021, 2022, 2023, 2024, 2025
- `21bands`: 21 temporal bands (weeks)

### Example Files by Zone
```bash
# Zone: mfl_cmf_area_project_2025_chr_zone_1
/fs2/ftype_mflf/shared_data/gdrive/GEE_Exports_mfl_cmf_area_project_2025_chr_zone_1/
├── cube_s2_gvmi_2023_W4before_to_W16after_21bands.tif
├── cube_s1_vv_2023_W4before_to_W16after_21bands.tif
└── cube_srtm_elevation_2023_W4before_to_W16after_21bands.tif

# Zone: mfl_cmf_area_project_2025_chr_zone_2  
/fs2/ftype_mflf/shared_data/gdrive/GEE_Exports_mfl_cmf_area_project_2025_chr_zone_2/
└── [similar structure]
```

## ⏰ Temporal Processing

### Weekly Time Series Structure
```
21 Bands = [W-4, W-3, W-2, W-1] + [W1, W2, ..., W17]
```

**Temporal Logic:**
- **Previous Year Weeks**: Last 4 weeks of year N-1 (e.g., Dec 2022 for 2023 data)
- **Current Year Weeks**: First 17 weeks of year N (e.g., Jan-Apr 2023)
- **Total Coverage**: ~21 weeks spanning growing season

### Example for 2023 Data
```
Band 1:  Week 49, 2022 (Dec 5-11, 2022)
Band 2:  Week 50, 2022 (Dec 12-18, 2022)  
Band 3:  Week 51, 2022 (Dec 19-25, 2022)
Band 4:  Week 52, 2022 (Dec 26, 2022 - Jan 1, 2023)
Band 5:  Week 1, 2023  (Jan 2-8, 2023)
Band 6:  Week 2, 2023  (Jan 9-15, 2023)
...
Band 21: Week 17, 2023 (Apr 24-30, 2023)
```

## 🔄 GEE Processing Workflow

### 1. Data Collection and Filtering

#### Sentinel-1 Processing
```javascript
// Load Sentinel-1 collection
var s1Collection = ee.ImageCollection('COPERNICUS/S1_GRD')
  .filter(ee.Filter.eq('instrumentMode', 'IW'))
  .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV'))
  .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VH'))
  .filter(ee.Filter.eq('orbitProperties_pass', 'DESCENDING'))
  .filterBounds(aoi)
  .filterDate(startDate, endDate);

// Apply speckle filtering
var s1Filtered = s1Collection.map(function(image) {
  var vv = image.select('VV').focal_median(50, 'circle', 'meters');
  var vh = image.select('VH').focal_median(50, 'circle', 'meters');
  return image.addBands(vv.rename('VV_filtered'))
              .addBands(vh.rename('VH_filtered'));
});
```

#### Sentinel-2 Processing  
```javascript
// Load Sentinel-2 collection with cloud masking
var s2Collection = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
  .filterBounds(aoi)
  .filterDate(startDate, endDate)
  .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20))
  .map(maskS2clouds);

// Calculate vegetation indices
var addIndices = function(image) {
  var ndvi = image.normalizedDifference(['B8', 'B4']).rename('NDVI');
  var gvmi = image.select('B8').subtract(image.select('B11'))
                  .divide(image.select('B8').add(image.select('B11')))
                  .rename('GVMI');
  var slavi = image.select('B8')
                   .divide(image.select('B4').add(image.select('B11')))
                   .rename('SLAVI');
  return image.addBands([ndvi, gvmi, slavi]);
};
```

### 2. Weekly Compositing
```javascript  
// Create weekly composites
var weeklyComposites = ee.List.sequence(startWeek, endWeek).map(function(week) {
  var weekStart = startDate.advance(ee.Number(week).multiply(7), 'day');
  var weekEnd = weekStart.advance(7, 'day');
  
  var weeklyCollection = s2Collection
    .filterDate(weekStart, weekEnd)
    .map(addIndices);
  
  // Use median composite to reduce noise
  var composite = weeklyCollection.median();
  
  return composite.set('week', week)
                  .set('system:time_start', weekStart.millis());
});
```

### 3. Export Configuration
```javascript
// Export as multi-band GeoTIFF
Export.image.toDrive({
  image: weeklyStack.select(featureList),
  description: 'cube_s2_gvmi_2023_W4before_to_W16after_21bands',
  folder: 'GEE_Exports_' + zoneName,
  fileNamePrefix: 'cube_s2_gvmi_2023_W4before_to_W16after_21bands',
  region: aoi,
  scale: 10,
  crs: 'EPSG:32647',  // UTM Zone 47N for Thailand
  maxPixels: 1e9,
  fileFormat: 'GeoTIFF'
});
```

## 📊 Data Quality and Processing Standards

### Spatial Standards
- **Coordinate System**: UTM Zone 47N (EPSG:32647)
- **Pixel Size**: 10m × 10m
- **Resampling**: Bilinear for continuous, nearest for categorical
- **Extent**: Clipped to AOI boundaries

### Temporal Standards
- **Compositing Period**: 7 days (Monday to Sunday)
- **Missing Data**: Filled using temporal interpolation
- **Quality Filtering**: 
  - S1: Orbit consistency, speckle filtering
  - S2: Cloud masking, shadow masking

### Radiometric Standards
- **S1 Backscatter**: Calibrated to σ⁰ in dB
- **S2 Reflectance**: Surface reflectance (Level-2A)
- **Indices**: Calculated from calibrated reflectance
- **Range Validation**: Typical value ranges enforced

## 🔧 Processing Parameters

### Key Configuration Settings
```python
# Temporal parameters
WEEKS_BEFORE = 4      # Previous year weeks
WEEKS_AFTER = 17      # Current year weeks  
TOTAL_BANDS = 21      # Total temporal bands

# Spatial parameters  
PIXEL_SIZE = 10       # meters
CRS = 'EPSG:32647'   # UTM Zone 47N
RESAMPLING = 'bilinear'

# Quality parameters
S2_CLOUD_THRESHOLD = 20    # % cloud coverage
S1_SPECKLE_FILTER = 50     # meters radius
COMPOSITE_METHOD = 'median'
```

### Processing Constraints
- **Memory Limits**: 1e9 pixels max per export
- **Time Limits**: 6-hour GEE processing timeout
- **Storage Limits**: Google Drive capacity dependent
- **API Limits**: GEE rate limiting applies

## 📈 Data Characteristics

### Typical Value Ranges
| Feature | Units | Typical Range | No-Data Value |
|---------|--------|---------------|---------------|
| VV | dB | -25 to -5 | -9999 |
| VH | dB | -30 to -10 | -9999 |
| GVMI | dimensionless | -0.5 to 0.5 | -9999 |
| NDVI | dimensionless | -0.2 to 0.9 | -9999 |
| SLAVI | dimensionless | 0 to 10 | -9999 |
| Elevation | meters | 0 to 2000 | -9999 |
| Slope | degrees | 0 to 90 | -9999 |

### Data Quality Metrics
```python
# Expected completeness by feature
EXPECTED_COMPLETENESS = {
    'vv': 0.95,      # 95% valid pixels
    'vh': 0.95,      # 95% valid pixels  
    'gvmi': 0.85,    # 85% valid pixels (cloud dependent)
    'ndvi': 0.85,    # 85% valid pixels (cloud dependent)
    'slavi': 0.80,   # 80% valid pixels (cloud dependent)
    'elevation': 1.0, # 100% valid pixels (static)
    'slope': 1.0     # 100% valid pixels (static)
}
```

## 🚨 Known Issues and Limitations

### Current Issues
1. **Cloud Coverage**: Optical indices affected by persistent clouds
2. **SAR Speckle**: Residual noise in S1 data despite filtering  
3. **Temporal Gaps**: Missing weeks due to processing failures
4. **Geometric Accuracy**: Minor co-registration errors between sensors

### Limitations
1. **Revisit Time**: Limited by satellite orbits (6-12 day repeat)
2. **Weather Dependency**: Optical data affected by clouds/haze
3. **Processing Delays**: GEE processing can take hours for large areas
4. **Storage Constraints**: Google Drive limits affect batch processing

### Quality Monitoring
```python
# Automated quality checks
def check_gee_data_quality(raster_path):
    """Check GEE data quality metrics"""
    with rasterio.open(raster_path) as src:
        data = src.read()
        
        # Check for no-data prevalence  
        nodata_pct = np.sum(data == -9999) / data.size * 100
        
        # Check value ranges
        valid_data = data[data != -9999]
        value_range = (valid_data.min(), valid_data.max())
        
        # Check temporal completeness
        temporal_completeness = np.sum(~np.all(data == -9999, axis=(1,2))) / data.shape[0]
        
        return {
            'nodata_percentage': nodata_pct,
            'value_range': value_range,
            'temporal_completeness': temporal_completeness
        }
```

## 🔄 Integration with Pipeline

### Interface with Processing Chain
1. **Input to Alignment**: GEE exports → `raster_clip_align.py`
2. **Input to Interpolation**: Aligned rasters → `batch_interpolate.py`  
3. **Output Format**: Compatible with `dataset_builder.py`
4. **Quality Checks**: Automated validation in extraction pipeline

### Data Flow Summary
```
GEE API → Weekly Composites → Multi-band Export → 
Google Drive → Local Storage → Alignment → Interpolation → 
Final Datacube → Training Extraction
```

## 📚 Best Practices

### Data Acquisition
1. **Batch Processing**: Process multiple zones/years together
2. **Error Handling**: Implement retry logic for GEE timeouts  
3. **Quality Monitoring**: Regular validation of exported data
4. **Version Control**: Track GEE script versions and parameters

### Storage Management  
1. **Naming Consistency**: Follow strict naming conventions
2. **Backup Strategy**: Multiple copies of critical data
3. **Space Monitoring**: Regular cleanup of temporary files
4. **Access Control**: Proper permissions for shared storage

### Performance Optimization
1. **Parallel Exports**: Multiple concurrent GEE tasks
2. **Efficient Filtering**: Minimize collection sizes before processing
3. **Memory Management**: Respect GEE memory limits
4. **Caching Strategy**: Reuse intermediate results where possible

---

*Last Updated: January 28, 2025*
*GEE Scripts Version: 2.1*
*Integration Status: Stable with current pipeline*