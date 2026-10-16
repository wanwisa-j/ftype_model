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

### Step 1: Data Retrieval and Preparation
**Source Code**: `/fs1/tor_project/esa_biophysicalop_local/src/cal_biophysi_local.py`

**Data Query Process**:
- **API Endpoint**: `http://172.16.99.36:3002/api/v1`
- **Query Period**: November 1st (previous year) to April 30th (target year)
- **Spatial Filter**: AOI bounding box geometry
- **Bands Retrieved**: 10m bands (B02, B03, B04, B08) and 20m bands (B05, B06, B07, B8A, B11, B12, SCL)

**Input Data Requirements**:
- Sentinel-2 Level-2A tiles from local API
- AOI geometry file (.shp or .parquet format)
- MGRS tile coverage within AOI bounds

**Processing Parameters**:
- **Spatial Resolution**: 20m bands downscaled to 10m using nearest neighbor resampling
- **Target Grid**: 10980 x 10980 pixels per tile
- **Projection**: UTM zone-specific, then reprojected to EPSG:4326

### Step 2: Biophysical Processing and Calculation
**Processing Function**: `processing_single_date()` in `cal_biophysi_local.py`

**Band Processing**:
1. **Band Loading**: Load 10m and 20m Sentinel-2 bands
2. **Spatial Resampling**: Downscale 20m bands to 10m (10980x10980 pixels) using nearest neighbor
3. **Baseline Correction**: Apply temporal baseline correction based on reference date (2022-01-25)
4. **Projection**: Reproject from UTM to EPSG:4326
5. **AOI Clipping**: Clip to specific zone boundaries using `rio.clip()`

**Angle Extraction**:
- **Metadata Source**: Extract viewing and solar angles from MTD_TL.xml files
- **Sun Angles**: Zenith and azimuth angles from Mean_Sun_Angle
- **Viewing Angles**: Mean viewing incidence angles per band from XML metadata
- **Namespace Handling**: Support for both psd-14 and psd-15 schema versions

**Data Stack Creation**:
- **Band Order**: B02, B03, B04, B05, B06, B07, B08, B8A, B11, B12 (spectral bands only)
- **Temporal Dimension**: Multi-temporal array creation for same-date acquisitions
- **Metadata Integration**: Include acquisition time, angles, and quality flags

**Biophysical Calculation**:
- **Algorithm**: ESA SNAP Biophysical Processor neural network
- **Variables Calculated**: LAI, FAPAR, FCOVER simultaneously
- **Input Requirements**: All 10 spectral bands + viewing geometry

## Detailed Processing Workflow

### Main Processing Function
**Function**: `main(aoi_dir, year, output_dir)` in `cal_biophysi_local.py`

**Workflow Steps**:

1. **AOI Setup**:
   - Read AOI geometry from .shp or .parquet file
   - Ensure CRS is EPSG:4326
   - Extract AOI name from directory path

2. **Data Query**:
   - Query Sentinel-2 data from local API (`http://172.16.99.36:3002/api/v1`)
   - Time range: November 1st (year-1) to April 30th (year)
   - Spatial filter: AOI bounding box
   - Retrieve both 10m and 20m bands plus SCL

3. **DataFrame Creation**:
   - Convert API response to pandas DataFrame
   - Generate MGRS tile IDs from zone_number + tile_first + tile_second
   - Add file paths for each band
   - Create orbit path and rounded acquisition dates

4. **Daily Processing Loop**:
   - Group data by acquisition date
   - Check if output already exists (skip if present)
   - Process each acquisition date through `processing_single_date()`

5. **Multi-Scene Handling**:
   - For multiple scenes per date: merge using `merge_arrays()`
   - For single scenes: direct processing
   - Save outputs to organized directory structure

### Single Date Processing Function
**Function**: `processing_single_date(args)` where `args = (row, aoi_areas)`

**Processing Steps**:

1. **Band Data Loading**:
   ```python
   # Load 10m bands directly
   if band in s2.S2_BANDS_10_LOCAL:
       band_data[band] = rxr.open_rasterio(band_path)
   
   # Downscale 20m bands to 10m
   if band in s2.S2_BANDS_20_LOCAL:
       band_data[band] = s2_downscale(rxr.open_rasterio(band_path), row['UTM Zone'])
   ```

2. **Quality Control Check**:
   - Check if clipped data is all null/zero for each band
   - Skip processing if no valid data in AOI

3. **Data Preprocessing**:
   - Apply baseline correction: `apply_baseline_data(band_data[band], row['Acquisition Date'])`
   - Reproject to EPSG:4326: `reproject_raster_to_4326(band_data[band])`
   - Clip to AOI: `band_data[band].rio.clip(aoi_areas.geometry.values)`

4. **Metadata Extraction**:
   ```python
   xml_path = f'{row["Parent File"]}/GRANULE/{row["sub_granule_dir"]}/MTD_TL.xml'
   tile_angles = get_tile_angles(xml_path)
   ```
   - Extract sun zenith/azimuth angles
   - Calculate mean viewing angles across all bands

5. **Data Stacking**:
   - Order bands: ['B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B8A', 'B11', 'B12']
   - Create xarray Dataset with time, band, y, x dimensions
   - Include metadata (assetid, productid, angles, etc.)

6. **Biophysical Calculation**:
   ```python
   for variable in [LAI, FAPAR, FCOVER]:
       ds = satellitetools.biophys.run_snap_biophys(ds, variable)
   ```

7. **Cloud Masking** (optional):
   - If SCL band available: downscale, reproject, clip
   - Mask clouds: exclude values [3, 8, 9, 10, 255]
   - Apply mask to all biophysical variables

### Output Organization
**Base Directory**: User-specified output directory
**Structure**:
```
{output_dir}/{aoi_name}/
├── fapar/
│   ├── s2_fapar_2023_01_01.tif
│   ├── s2_fapar_2023_01_02.tif
│   └── ...
├── fcover/
│   ├── s2_fcover_2023_01_01.tif
│   └── ...
└── lai/
    ├── s2_lai_2023_01_01.tif
    └── ...
### Step 3: Weekly Temporal Aggregation
**Source Code**: `/fs1/tor_project/model_foresttype/src/weekly_biophys_aggregation.py`
**Replaces**: `/fs1/tor_project/model_foresttype/notebook/04_prep_data-prediction_v2_1_1.ipynb` (notebook approach)

**Key Implementation**: `WeeklyBiophysicalAggregator` class - robust pipeline for processing daily biophysical files into weekly aggregated datacubes

**Processing Workflow**:

1. **Input Data Collection**:
   - Read daily biophysical files from `/fs2/ftype_mflf/shared_data/temp/biophys/2025/1.0.0/{zone}/`
   - Create DataFrame mapping dates to file paths using `create_biophys_df()`
   - Filter data by date range: November 15th (previous year) to April 30th (target year)

2. **Weekly Aggregation Method**:
   - **Temporal Window**: ISO week-based aggregation using Monday as week start (W-MON)
   - **Composite Method**: Mean composite within each week using `np.mean()`
   - **Week Definition**: ISO calendar weeks to ensure consistent temporal alignment

3. **21-Week Selection Process**:
   - **Target Structure**: Last 4 weeks of previous year + first 17 weeks of project year
   - **Week Calculation**: 
     ```python
     # Determine desired week starts
     year_start = pd.Timestamp(f'{year}-01-01')
     jan4 = pd.Timestamp(f'{year}-01-04')
     week1_monday = jan4 - pd.Timedelta(days=jan4.weekday())
     desired_weeks_list = [week1_monday + pd.Timedelta(weeks=w) for w in range(-4, 17)]
     ```

4. **Gap Handling and Interpolation**:
   - **Missing Week Detection**: Identify gaps in weekly data availability
   - **Interpolation Method**: Nearest neighbor interpolation for missing weeks
   - **Pixel-level Processing**: Linear interpolation for NaN values in time series
   - **Quality Assurance**: Data validation and quality reporting throughout pipeline

5. **Spatial Processing**:
   - **Geospatial Metadata**: Preserve coordinate reference system and geotransform
   - **Data Type Optimization**: Convert to float32 for memory efficiency
   - **Compression**: Apply LZW compression for output files

**Output Structure**:
```
{output_dir}/{zone}/
├── fapar/
│   └── cube_biophys_fapar_{year}_W4before_to_W16after_21bands.tif
├── fcover/
│   └── cube_biophys_fcover_{year}_W4before_to_W16after_21bands.tif
└── lai/
    └── cube_biophys_lai_{year}_W4before_to_W16after_21bands.tif
```

**Output Location**: `/fs2/ftype_mflf/shared_data/datacube/2025/1.2.0/`

**Band Organization** (21 bands total):
- Bands 1-4: W4before to W1before (last 4 weeks of previous year)
- Band 5: W0 (first week of project year) 
- Bands 6-21: W1after to W16after (weeks 2-17 of project year)

**Processing Parameters**:
- **Data Type**: float32 for memory efficiency
- **NoData Handling**: NaN values preserved and interpolated
- **Compression**: LZW compression for output files
- **Quality Control**: ~98-99% valid data retention after fixing NoData issues

**Command Line Usage**:
```bash
# Single zone/year processing
python weekly_biophys_aggregation.py \
  --input-dir /fs2/ftype_mflf/shared_data/temp/biophys/2025/1.0.0/{zone} \
  --output-dir /fs2/ftype_mflf/shared_data/datacube/2025/1.2.0/{zone} \
  --year 2022

# Batch processing multiple years
./run_weekly_biophys_batch.sh
```

**Key Improvements Over Notebook Approach**:
- **Fixed NoData Issues**: Resolved ~96-98% NoData problem in zones 1 & 2
- **Robust Error Handling**: Comprehensive validation and logging
- **Parallel Processing**: Efficient batch processing with configurable concurrency
- **Quality Monitoring**: Real-time data quality assessment and reporting
- **Consistent Output**: Standardized 21-band weekly datacubes across all zones

## Data Characteristics

### Temporal Resolution
- **Original**: Variable (Sentinel-2 revisit cycle: 2-5 days at equator)
- **Processed**: Daily single-date processing, then weekly composites (21 weeks total)
- **Coverage Period**: November 1st (previous year) to April 30th (target year)
- **Weekly Aggregation**: Aligned with GEE data temporal windows

### Spatial Resolution
- **Original**: Mixed 10m and 20m bands
- **Processing**: 20m bands downscaled to 10m using nearest neighbor resampling
- **Final**: 10m (consistent with GEE data)
- **Target Grid**: 10980 x 10980 pixels per tile

### Projection Handling
- **Source**: UTM zone-specific (32600 + zone_number)
- **Final**: EPSG:4326 (WGS84 Geographic)
- **Resampling Method**: Nearest neighbor for band resampling

### Quality Metrics
- **Cloud Coverage**: Masked using SCL band (values 3, 8, 9, 10, 255)
- **Data Availability**: Checked per band for null/zero values
- **Temporal Continuity**: Daily processing with gap handling in weekly aggregation

## Algorithm Details

### ESA SNAP Biophysical Processor
- **Method**: Neural network trained on PROSPECT+SAIL radiative transfer model
- **Input Bands**: Sentinel-2 bands B02, B03, B04, B05, B06, B07, B08, B8A, B11, B12
- **Training Dataset**: Synthetic dataset from radiative transfer simulations
- **Validation**: Extensive field campaign data
- **Implementation**: `satellitetools.biophys.run_snap_biophys()` function

### Baseline Correction Algorithm
**Function**: `apply_baseline_data(band_data, date_acq)`
- **Reference Date**: 2022-01-25
- **Correction Logic**:
  - If acquisition date > reference date: subtract 1000 from raw values
  - If acquisition date ≤ reference date: no offset applied
  - Scale by factor of 10000 to convert to reflectance
  - Mask zero and -0.1 values as NaN

### Spatial Downscaling Function
**Function**: `s2_downscale(img_raster, utm_zone)`
- **Method**: Nearest neighbor resampling
- **Target Grid**: 10980 x 10980 pixels
- **CRS Handling**: Auto-assign UTM CRS if missing (32600 + zone_number)

### Angle Extraction Function
**Function**: `get_tile_angles(xml_path)`
- **Metadata Source**: MTD_TL.xml files from Sentinel-2 GRANULE directories
- **Namespaces Supported**: Both psd-14 and psd-15 schema versions
- **Outputs**: Sun zenith/azimuth angles and per-band viewing angles

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

## Command Line Usage

### Running the Biophysical Processor
```bash
python /fs1/tor_project/esa_biophysicalop_local/src/cal_biophysi_local.py \
    --aoi_dir /path/to/aoi_file.shp \
    --year 2023 \
    --output_dir /fs2/ftype_mflf/shared_data/temp/biophys/2025/1.0.0/
```

**Parameters**:
- `--aoi_dir`: Path to AOI geometry file (.shp or .parquet)
- `--year`: Target year for processing (int)
- `--output_dir`: Base output directory for results

**Output Structure**:
```
{output_dir}/{aoi_name}/
├── fapar/
│   ├── s2_fapar_YYYY_MM_DD.tif
│   └── ...
├── fcover/
│   ├── s2_fcover_YYYY_MM_DD.tif
│   └── ...
└── lai/
    ├── s2_lai_YYYY_MM_DD.tif
    └── ...
```

## Related Documentation
- [GEE Data Source](./GEE_DATA_SOURCE.md)
- [Data Alignment Process](./DATA_PROCESSING_ALIGNMENT.md)
- [Temporal Interpolation](./DATA_PROCESSING_ALIGNMENT.md)
- [Data Pipeline Overview](./DATA_PIPELINE_OVERVIEW.md)
