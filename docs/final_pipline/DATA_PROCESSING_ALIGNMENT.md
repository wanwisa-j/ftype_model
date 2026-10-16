# Data Processing and Alignment Documentation

## Overview

This document describes the critical data processing steps that ensure spatial and temporal consistency across all data sources in the forest type classification pipeline. The alignment process is essential for integrating GEE-derived features with biophysical parameters.

## Spatial Alignment Process

### Primary Tool
**Script**: `src/raster_clip_align.py`
**Main Function**: `raster_pixels_alignment_with_clip()`

### Key Features
- **Automatic UTM Zone Detection**: Determines optimal UTM zone from AOI centroid
- **Automatic Pixel Size Detection**: Extracts pixel size from input raster
- **Flexible CRS Support**: Works with any geographic location worldwide
- **Multiple Processing Modes**: Single, batch, zone, and config-based processing

### Processing Capabilities
1. **UTM Zone Auto-Detection**: `get_utm_zone_from_aoi()` calculates appropriate UTM zone
2. **Pixel Size Auto-Detection**: `auto_determine_pixel_size()` extracts from raster metadata
3. **AOI Clipping**: Uses `rio.mask.mask()` with all_touched=True for precise clipping
4. **Grid Alignment**: Creates aligned transform using `rio.transform.from_bounds()`
5. **Multi-band Support**: Processes all bands simultaneously with reproject()

### Objectives
1. **Pixel Registration**: Ensure exact pixel-to-pixel correspondence through aligned grid
2. **Projection Consistency**: Auto-determine and harmonize coordinate reference systems
3. **Spatial Extent**: Clip all data to identical AOI boundaries using precise masking
4. **Resolution Matching**: Maintain consistent pixel spacing across all data sources

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

#### Step 1: Automatic Parameter Detection
```python
# Auto-determine target CRS from AOI centroid
target_crs = get_utm_zone_from_aoi(aoi_areas)
# Example: 'EPSG:32647' for UTM Zone 47N

# Auto-determine pixel size from input raster
pixel_size = auto_determine_pixel_size(raster_path)
# Extracts from raster transform (min of x/y pixel sizes)
```

#### Step 2: AOI Preparation and Grid Definition
```python
# Ensure AOI is in target CRS
if aoi_areas.crs != target_crs:
    aoi_areas = aoi_areas.to_crs(target_crs)

# Calculate aligned grid from AOI bounds
bbox = aoi_areas.total_bounds  # x_min, y_min, x_max, y_max
num_pixels_x = int(np.ceil((x_max - x_min) / pixel_size))
num_pixels_y = int(np.ceil((y_max - y_min) / pixel_size))
align_transform = rio.transform.from_bounds(*bbox, width=num_pixels_x, height=num_pixels_y)
```

#### Step 3: Raster Clipping and Reprojection
```python
# Clip raster with AOI before reprojection
clipped_data, clipped_transform = rio.mask.mask(
    src, aoi_clip.geometry, crop=True, nodata=src.nodata, all_touched=True
)

# Reproject each band to aligned grid
for band in range(num_bands):
    reproject(
        source=clipped_data[band],
        destination=dst_data[band],
        src_transform=clipped_transform,
        src_crs=src_crs,
        dst_transform=align_transform,
        dst_crs=target_crs_obj,
        resampling=resampling_method,
        num_threads=2
    )
```

#### Step 4: Output Generation and Validation
```python
# Update metadata for aligned raster
aligned_meta = {
    'crs': target_crs_obj,
    'transform': align_transform,
    'width': num_pixels_x,
    'height': num_pixels_y,
    'count': num_bands
}

# Save aligned raster if output directory provided
if out_dir_cube:
    with rio.open(output_path, 'w', **aligned_meta) as dst:
        dst.write(dst_data)
```

## Temporal Interpolation Process

### Primary Tools
**Wrapper Script**: `src/batch_interpolate.py`
**Core Engine**: `src/raster_interpolation_ultra_fast.py`

### Processing Strategy
The pipeline uses ultra-fast vectorized linear interpolation to handle invalid values (NaN, inf, -inf, nodata) across the temporal dimension of raster datacubes.

### Key Features
- **Vectorized Processing**: Full numpy vectorization for maximum performance
- **Memory Efficient**: Chunk-based processing with configurable chunk sizes
- **Edge Extrapolation**: Linear interpolation with edge value extension
- **Conda Environment**: Automatic activation of gisworkspace-cupy environment

### Interpolation Algorithm

#### Core Interpolation Logic
```python
def interpolate_chunk_vectorized(data_chunk, time_indices, output_nodata):
    """
    Ultra-fast vectorized interpolation for a data chunk.
    """
    # Convert to float64 for precision
    arr = data_chunk.astype(np.float64)
    
    # Mark invalid values: nan, inf, -inf, nodata
    invalid_elem = ~np.isfinite(arr)
    if output_nodata is not None:
        invalid_elem |= (arr == output_nodata)
    
    # Identify pixels with all timesteps invalid
    mask_invalid = np.all(invalid_elem, axis=0)
    
    # Process columns needing interpolation
    cols_to_fix = pixels_with_some_valid_data
    for col in cols_to_fix:
        # Linear interpolation with edge extension
        series_filled = np.interp(time_indices, valid_times, valid_values)
```

#### Batch Processing Workflow
```python
# Command structure from batch_interpolate.py
cmd = [
    'python', 
    '/fs1/tor_project/model_foresttype/src/raster_interpolation_ultra_fast.py',
    str(input_file),
    str(output_file),
    '--chunk_size', str(chunk_size)
]

# Environment activation and execution
full_cmd = f"/bin/bash -c 'source /opt/miniconda3/etc/profile.d/conda.sh && conda activate gisworkspace-cupy && {' '.join(cmd)}'"
```

### Processing Workflow

#### Batch Processing Command Structure
```bash
# Example batch interpolation command
python /fs1/tor_project/model_foresttype/src/batch_interpolate.py \
  /fs2/ftype_mflf/shared_data/gdrive/GEE_Exports_mfl_cmf_area_project_2025_chr_zone_2/ \
  /fs2/ftype_mflf/shared_data/interpolated/ \
  --pattern "cube_*.tif" \
  --chunk_size 512 \
  --suffix "_interpolated"
```

#### Processing Parameters
- **Input Directory**: Source directory containing raster datacubes
- **Output Directory**: Target directory for interpolated results
- **File Pattern**: Glob pattern to match files (default: "*.tif")
- **Chunk Size**: Memory management chunk size (default: 512)
- **NoData Value**: Target nodata value (default: -9999)

#### Processing Logic
1. **File Discovery**: Scan input directory for matching raster files
2. **Environment Setup**: Activate gisworkspace-cupy conda environment
3. **Sequential Processing**: Process each file using ultra-fast engine
4. **Memory Management**: Chunk-based processing for large rasters
5. **Output Generation**: Write interpolated results with suffix naming

#### Quality Control
- **Invalid Value Handling**: Comprehensive detection of NaN, inf, -inf, nodata
- **Edge Extrapolation**: Use edge values for start/end gaps
- **Pixel-wise Processing**: Individual time series interpolation per pixel
- **Data Type Management**: Float64 precision for interpolation, Float32 output

## Output Data Structure

### Final Datacube Organization
```
/fs2/ftype_mflf/shared_data/datacube/2025/1.2.0/
├── mfl_cmf_area_project_2025_chr_zone_1/
│   ├── fapar/
│   │   └── cube_biophys_fapar_2023_W4before_to_W16after_21bands.tif
│   ├── fcover/
│   ├── lai/
│   ├── gvmi/
│   ├── ndvi/
│   ├── slavi/
│   ├── vv/
│   ├── vh/
│   ├── elevation/
│   │   └── elevation_data.tif
│   └── slope/
│       └── slope_data.tif
└── mfl_cmf_area_project_2025_chr_zone_2/
    └── [same structure]
```

### Interpolated Data Organization
```
/fs2/ftype_mflf/shared_data/interpolated/
├── cube_s1_vv_2023_W4before_to_W16after_21bands_interpolated.tif
├── cube_s1_vh_2023_W4before_to_W16after_21bands_interpolated.tif
├── cube_s2_gvmi_2023_W4before_to_W16after_21bands_interpolated.tif
├── cube_s2_ndvi_2023_W4before_to_W16after_21bands_interpolated.tif
└── cube_s2_slavi_2023_W4before_to_W16after_21bands_interpolated.tif
```

### Data Format Specifications
- **File Format**: GeoTIFF (default rasterio output)
- **Data Type**: Float32 for interpolated data (converted from Float64 processing)
- **NoData Value**: -9999 (configurable, set consistently across pipeline)
- **Band Count**: 21 bands per feature for temporal series, 1 band for static features
- **Spatial Reference**: Auto-determined UTM zone (EPSG:32600+ zone number)
- **Compression**: None by default (can be added via rasterio profile updates)
- **Chunk Size**: Configurable for memory management (default: 512x512)

## Quality Control and Validation

### Spatial Alignment Validation
```python
# Validation capabilities in raster_clip_align.py:
1. Automatic UTM zone detection from AOI centroid
2. Pixel size extraction from source rasters
3. Aligned grid generation using rio.transform.from_bounds()
4. Precise AOI clipping with all_touched=True
5. Multi-band reprojection with consistent transform
6. Metadata preservation and updating
```

### Temporal Interpolation Validation
```python
# Quality checks in interpolation pipeline:
1. Invalid value detection (NaN, inf, -inf, nodata)
2. Pixel-wise time series analysis
3. Edge value extrapolation for boundary gaps
4. Memory-efficient chunk processing
5. Data type consistency (Float64->Float32 conversion)
6. Progress monitoring and logging
```

### Processing Modes Available

#### Single Raster Processing
```bash
python raster_clip_align.py --mode single \
  --raster input.tif --aoi boundary.shp --output output_dir/
```

#### Batch Directory Processing
```bash
python raster_clip_align.py --mode batch \
  --input-dir raster_dir/ --aoi study_area.parquet --output processed_dir/
```

#### Zone-specific Processing
```bash
python raster_clip_align.py --mode zone \
  --zone zone_1 --input-dir gdrive_exports/ --output datacube_dir/
```

## Performance Optimization

### Spatial Processing Efficiency
- **Multi-threading**: 2 threads for rasterio reproject operations
- **Memory Management**: Automatic clipping before reprojection to reduce data volume
- **UTM Auto-detection**: Eliminates manual CRS specification
- **Pixel Size Auto-detection**: Extracts optimal resolution from source data

### Interpolation Processing Efficiency
- **Ultra-fast Vectorization**: Full numpy vectorization using `interpolate_chunk_vectorized()`
- **Chunk-based Processing**: Configurable chunk sizes (default: 2048x2048)
- **Memory Optimization**: Float64 processing with Float32 output
- **Batch Processing**: Automated conda environment activation and file processing

### Conda Environment Management
```bash
# Automatic environment activation in batch_interpolate.py
source /opt/miniconda3/etc/profile.d/conda.sh
conda activate gisworkspace-cupy
```

## Error Handling and Recovery

### Spatial Alignment Error Handling
```python
# Clipping fallback mechanism
try:
    clipped_data, clipped_transform = rio.mask.mask(
        src, aoi_clip.geometry, crop=True, nodata=src.nodata, all_touched=True
    )
except Exception as e:
    logger.warning(f"Clipping failed, using original raster: {e}")
    clipped_data = src.read()
    clipped_transform = src.transform
```

### Interpolation Error Handling
```python
# File processing with error capture
try:
    result = subprocess.run(full_cmd, shell=True, capture_output=True, text=True)
    if result.returncode == 0:
        logger.info(f"Successfully processed: {input_file.name}")
        return True
    else:
        logger.error(f"Failed to process {input_file.name}: {result.stderr}")
        return False
except Exception as e:
    logger.error(f"Error processing {input_file.name}: {e}")
    return False
```

### Common Issues and Solutions

#### File Access Issues
- **Detection**: File existence checking before processing
- **Solution**: Automatic directory creation with `mkdir(parents=True, exist_ok=True)`
- **Recovery**: Skip processing if input files not found

#### Memory Limitations
- **Large Rasters**: Automatic chunk-based processing
- **System Resources**: Configurable chunk sizes for different hardware
- **Environment Issues**: Conda environment validation and activation

## Configuration Management

### Command Line Interface

#### Spatial Alignment Configuration
```bash
# Single raster with auto-detection
python raster_clip_align.py --mode single \
  --raster input.tif --aoi boundary.shp --output output_dir/

# Single raster with specific parameters
python raster_clip_align.py --mode single \
  --raster input.tif --aoi boundary.parquet --output output_dir/ \
  --target-crs EPSG:32633 --pixel-size 30.0 --resampling bilinear

# Batch processing with pattern matching
python raster_clip_align.py --mode batch \
  --input-dir raster_dir/ --aoi study_area.geojson --output processed_dir/ \
  --pattern "*.tif" --resampling nearest

# Zone processing for multiple raster types
python raster_clip_align.py --mode zone \
  --zone zone_1 --input-dir gdrive_dir/ --output datacube_dir/ --aoi aoi_base_path/
```

#### Interpolation Configuration
```bash
# Basic batch interpolation
python batch_interpolate.py input_dir output_dir

# Advanced configuration
python batch_interpolate.py input_dir output_dir \
  --pattern "cube_*.tif" \
  --nodata_value -9999 \
  --chunk_size 1024 \
  --suffix "_interpolated" \
  --verbose
```

### Processing Parameters

#### Spatial Alignment Parameters
- `--target-crs`: Target CRS (auto-determined if not specified)
- `--pixel-size`: Target pixel size (auto-determined if not specified)
- `--resampling`: Resampling method (nearest, bilinear, cubic, average)

#### Interpolation Parameters
- `--pattern`: File pattern for batch processing (default: "*.tif")
- `--nodata_value`: NoData value for output (default: -9999)
- `--chunk_size`: Processing chunk size (default: 512)
- `--suffix`: Output filename suffix (default: "_interpolated")

### Configuration Template Generation
```bash
# Generate YAML configuration template
python raster_clip_align.py --mode config --config config.yaml
```

## Related Documentation
- [GEE Data Source](./GEE_DATA_SOURCE.md)
- [Biophysical Data Source](./BIOPHYSICAL_DATA_SOURCE.md)
- [Dataset Extraction](./DATASET_EXTRACTION.md)
