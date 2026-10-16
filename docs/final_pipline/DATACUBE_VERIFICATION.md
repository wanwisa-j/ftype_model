# Datacube Verification Documentation

## Overview

The Comprehensive Datacube Verification System ensures the quality, completeness, and consistency of processed datacubes before they are used for dataset extraction and model training. This critical quality assurance step validates that all data processing phases have completed successfully and that the datacube meets the forest type model's requirements.

## Verification Architecture

### Verification Workflow
```
Processed Datacube → Structure Check → Geometry Validation → Temporal Verification → Quality Assessment → Final Report
         ↓                ↓                ↓                   ↓                    ↓               ↓
   [Zone Directory] [Feature Count] [CRS/Transform] [Band Structure] [Data Integrity] [Pass/Fail]
```

## Core Verification Components

### 1. Structure Verification
**Purpose**: Ensures all expected features are present with correct file counts

**Checks**:
- **Biophysical Features**: fapar, fcover, lai (7 files each per zone)
- **Optical Features**: gvmi, ndvi, slavi (7 files each per zone)
- **SAR Features**: vh, vv (7 files each per zone)
- **Static Features**: elevation, slope (1 file each per zone)
- **Directory Structure**: Proper organization and naming conventions

**Implementation**:
```python
def verify_structure(self) -> Dict[str, Any]:
    # Check each expected feature directory
    for feature in self.all_features:
        feature_path = self.zone_dir / feature
        if feature_path.exists():
            tif_files = list(feature_path.glob("*.tif"))
            structure_results['file_counts'][feature] = len(tif_files)
```

### 2. Geometry Verification
**Purpose**: Validates spatial consistency across all features

**Checks**:
- **CRS Consistency**: Same coordinate reference system (EPSG:32647)
- **Transform Consistency**: Identical pixel alignment and grid structure
- **Shape Consistency**: Uniform dimensions across all rasters
- **Spatial Bounds**: Consistent coverage within each zone

**Implementation**:
```python
def verify_geometry(self) -> Dict[str, Any]:
    # Compare metadata across all features
    if metadata['crs'] != reference_metadata['crs']:
        geometry['crs_consistent'] = False
    if metadata['transform'] != reference_metadata['transform']:
        geometry['transform_consistent'] = False
```

### 3. Temporal Dimension Verification
**Purpose**: Ensures correct temporal structure for time series analysis

**Checks**:
- **Temporal Features**: Must have exactly 21 bands (W4before to W16after)
- **Static Features**: Must have exactly 1 band
- **Band Structure**: Consistent across all files within feature types
- **Temporal Coverage**: Complete time series without gaps

**Implementation**:
```python
def verify_temporal_dimensions(self) -> Dict[str, Any]:
    # Check temporal features (21 bands)
    for feature in self.temporal_features:
        all_correct = all(count == 21 for count in band_counts)
    # Check static features (1 band)  
    for feature in self.static_features:
        all_correct = all(count == 1 for count in band_counts)
```

### 4. Data Quality Verification
**Purpose**: Assesses data integrity and interpolation quality

**Checks**:
- **NoData Detection**: Identifies and quantifies missing data patterns
- **Valid Data Percentage**: Calculates proportion of valid vs. invalid pixels
- **Value Range Analysis**: Ensures values are within expected bounds
- **Interpolation Quality**: Verifies successful gap filling

**Implementation**:
```python
def analyze_data_quality(self) -> Dict[str, Any]:
    # Check for various NoData patterns
    for ndv in self.nodata_values:
        nodata_count += np.sum(data == ndv)
    valid_percentage = (valid_pixels / total_values) * 100
```

## Verification Tools

### Main Verification Script
**File**: `datacube_zone_verification.py`

**Features**:
- Comprehensive analysis of individual zones
- JSON results export for automated processing
- Human-readable text reports
- Visualization generation (plots and charts)
- Command-line interface with flexible options

**Usage**:
```bash
python datacube_zone_verification.py <zone_directory> [options]
```

**Options**:
- `--output`: Custom output directory
- `--max-files`: Number of files to analyze per feature
- `--verbose`: Enable detailed logging
- `--no-plots`: Skip visualization generation

### Execution Scripts

#### Single Zone Verification
**File**: `run_single_zone_verification.sh`

**Purpose**: Test individual zones or re-run after fixes

**Usage**:
```bash
./run_single_zone_verification.sh zone_1
./run_single_zone_verification.sh zone_2 /custom/output/path
```

**Features**:
- Environment activation (gisworkspace-cupy)
- Error handling and status reporting
- Quick results summary
- Integration with conda environment

#### Parallel Zone Verification
**File**: `run_zone_verification_parallel.sh`

**Purpose**: Process all zones simultaneously for maximum efficiency

**Usage**:
```bash
./run_zone_verification_parallel.sh
```

**Features**:
- Configurable parallel job limit (default: 3)
- Background processing with PID tracking
- Consolidated summary report generation
- Resource-efficient parallel execution

#### Sequential Zone Verification
**File**: `verify_all_zones_sequential.sh`

**Purpose**: Process zones one-by-one for resource-constrained environments

**Usage**:
```bash
./verify_all_zones_sequential.sh
```

**Features**:
- Lower memory usage
- Optional stop-on-failure mode
- Detailed progress tracking
- Comprehensive logging

## Verification Outputs

### JSON Results
**File**: `{zone}_verification_results.json`

**Content**:
- Complete verification metadata
- Detailed analysis results
- Error information and diagnostics
- Processing timestamps
- Machine-readable format for automation

**Structure**:
```json
{
  "zone": "zone_1",
  "status": "PASS",
  "verification_time": "2025-09-10T05:01:00.961528",
  "structure": {...},
  "geometry": {...},
  "temporal": {...},
  "quality": {...},
  "summary": {...}
}
```

### Text Reports
**File**: `{zone}_verification_report.txt`

**Content**:
- Executive summary
- Component status breakdown
- Detailed findings
- Issues and recommendations
- Human-readable format

### Visualizations
**File**: `{zone}_verification_summary.png`

**Content**:
- Feature completeness charts
- Data quality metrics
- Verification status dashboard
- File count distributions

## Quality Thresholds

### Pass/Fail Criteria

#### Structure Requirements
- ✅ **PASS**: All 10 features present with correct file counts
- ❌ **FAIL**: Any missing features or incorrect file counts

#### Geometry Requirements
- ✅ **PASS**: CRS, transform, and shape consistent across all features
- ❌ **FAIL**: Any geometric inconsistencies detected

#### Temporal Requirements
- ✅ **PASS**: Temporal features have 21 bands, static features have 1 band
- ❌ **FAIL**: Incorrect band structure in any feature

#### Quality Requirements
- ✅ **PASS**: >90% valid data, successful interpolation
- ❌ **FAIL**: <90% valid data or significant quality issues

### Warning Thresholds
- **Structure**: Missing non-critical files
- **Quality**: 80-90% valid data (acceptable but flagged)
- **Geometry**: Minor coordinate precision differences

## Integration with Pipeline

### Position in Workflow
The verification step occurs after interpolation but before dataset extraction:

1. **Data Processing**: Raster alignment and interpolation complete
2. **📍 Verification**: Quality assurance and validation
3. **Dataset Extraction**: Ground truth dataset creation
4. **Model Training**: Use verified datacube for training

### Automated Quality Gates
```bash
# Example pipeline integration
./run_interpolation.sh
if ./run_zone_verification_parallel.sh; then
    echo "✅ Verification passed - proceeding to dataset extraction"
    ./extract_dataset.sh
else
    echo "❌ Verification failed - fix issues before proceeding"
    exit 1
fi
```

### Continuous Integration
- **Pre-commit**: Verify sample zones before data commits
- **Scheduled**: Regular verification of production datacubes
- **Version Control**: Track verification results with datacube versions

## Troubleshooting

### Common Issues and Solutions

#### Missing Features
**Symptom**: Features missing from zone directories  
**Cause**: Incomplete data processing or file transfer issues  
**Solution**: Re-run data processing pipeline for missing features

#### Geometry Inconsistencies
**Symptom**: CRS or transform mismatches  
**Cause**: Inconsistent spatial processing  
**Solution**: Re-run `raster_clip_align.py` to ensure consistent alignment

#### Temporal Structure Issues
**Symptom**: Incorrect band counts  
**Cause**: Processing errors during temporal aggregation  
**Solution**: Regenerate temporal datacubes with correct band structure

#### Low Data Quality
**Symptom**: High NoData percentages  
**Cause**: Insufficient interpolation or source data issues  
**Solution**: Re-run interpolation or investigate source data quality

### Error Diagnostics

#### Log Analysis
```bash
# Check verification logs
tail -f /tmp/zone_verification_logs/zone_1_verification.log

# Search for specific errors
grep -i error /tmp/zone_verification_logs/*.log
```

#### Manual Inspection
```python
# Quick manual check
import rasterio
with rasterio.open('datacube_file.tif') as src:
    print(f"Bands: {src.count}, Shape: {src.shape}, CRS: {src.crs}")
```

## Performance Optimization

### Resource Management
- **Memory**: Configurable sampling for large rasters
- **CPU**: Parallel processing with job limits
- **I/O**: Efficient file access patterns

### Scalability
- **Zone-based**: Independent verification per zone
- **Incremental**: Verify only changed data
- **Distributed**: Scale across multiple compute nodes

## Version History

### v1.0 (2025-09-10)
- Initial comprehensive verification system
- Support for all 10 feature types
- Parallel and sequential execution modes
- Complete documentation and examples

### Future Enhancements
- **Multi-version Support**: Compare different datacube versions
- **Automated Fixes**: Self-healing for minor issues
- **Advanced Analytics**: Trend analysis and quality metrics over time
- **Cloud Integration**: Support for cloud-based datacubes

## Related Documentation

- [Data Pipeline Overview](./DATA_PIPELINE_OVERVIEW.md)
- [Data Processing Alignment](./DATA_PROCESSING_ALIGNMENT.md)
- [Final Verification Summary](../../FINAL_VERIFICATION_SUMMARY.md)

---

*This documentation is part of the Forest Type Model Pipeline v1.2.0*  
*Last updated: 2025-09-10*