# Conversation Memory: NoData Investigation - September 8, 2025

## Context & Problem Statement
- **Issue**: Biophysical data in `mfl_cmf_area_project_2025_chr_zone_1` and `mfl_cmf_area_project_2025_chr_zone_2` contains excessive NoData (-9999) values
- **Impact**: ~25% of training samples contaminated with invalid biophysical data
- **Dataset**: `/fs2/ftype_mflf/shared_data/dataset/1.2.0/raw/forest_type_dataset_raw_20250903_145707.cleaned.parquet`

## Investigation Process Completed

### 1. Documentation Review ✅
- **Files reviewed**: 
  - `/fs1/tor_project/model_foresttype/docs/final_pipline/BIOPHYSICAL_DATA_SOURCE.md`
  - `/fs1/tor_project/model_foresttype/docs/final_pipline/DATA_PROCESSING_ALIGNMENT.md`
  - `/fs1/tor_project/model_foresttype/docs/final_pipline/DATASET_EXTRACTION.md`

### 2. Existing Analysis Results ✅
- **Used script**: `/fs1/tor_project/model_foresttype/analysis_lai_fapar_fcover_distribution.py`
- **Results location**: `/fs1/tor_project/model_foresttype/analysis_output/biophysical_analysis_report.txt`
- **Key finding**: Zone-wise contamination patterns identified

### 3. Aggregated Datacube Analysis ✅
- **Location**: `/fs2/ftype_mflf/shared_data/datacube/2025/1.2.0/`
- **Script created**: `/fs1/tor_project/model_foresttype/quick_nodata_check.py`
- **Results**: 96-98% NoData in final datacubes

### 4. Raw Daily Data Analysis ✅ - BREAKTHROUGH
- **Location**: `/fs2/ftype_mflf/shared_data/temp/biophys/2025/1.0.0/`
- **Script created**: `/fs1/tor_project/model_foresttype/quick_raw_data_check.py`
- **KEY FINDING**: Raw daily data has 0.0% NoData - problem is in aggregation!

## Root Cause Identified

### **DEFINITIVE CONCLUSION**: 
NoData issue introduced during **Weekly Temporal Aggregation**, NOT in source data processing.

**Evidence:**
- Raw daily FCOVER files: 0.0% NoData, healthy 0.0-0.9 ranges
- Final weekly datacubes: 96-98% NoData
- Problem location: `prepare_biophys_dataset()` function in notebook

## Key Files Created During Investigation

### Analysis Scripts:
- `/fs1/tor_project/model_foresttype/investigate_datacube_nodata.py` - Full investigation (too slow)
- `/fs1/tor_project/model_foresttype/quick_nodata_check.py` - Quick aggregated data check
- `/fs1/tor_project/model_foresttype/quick_raw_data_check.py` - **CRITICAL** raw data analysis
- `/fs1/tor_project/model_foresttype/check_raw_biophys_nodata.py` - Detailed raw analysis (unused)

### Reports & Documentation:
- `/fs1/tor_project/model_foresttype/nodata_investigation_summary.md` - Initial analysis
- `/fs1/tor_project/model_foresttype/DEFINITIVE_NODATA_ROOT_CAUSE.md` - **FINAL ROOT CAUSE**

### Environment Configuration:
- Updated `/fs1/tor_project/model_foresttype/.claude/settings.local.json` with:
  - Geospatial: `conda activate gisworkspace-cupy`
  - TensorFlow: `source /fs1/tor_project/model_foresttype/tf/bin/activate`

## Next Steps (URGENT)

### 1. Fix Weekly Aggregation Bug
**Target**: `/fs1/tor_project/model_foresttype/notebook/04_prep_data-prediction_v2_1_1.ipynb`
- Section: "Biophy" markdown cell
- Function: `prepare_biophys_dataset()`
- Check: 21-week selection algorithm, gap-filling interpolation

### 2. Test Areas to Investigate:
- Daily to weekly mean composite calculation
- Missing week interpolation (nearest neighbor)
- Data type conversion (Float64→Float32)
- Mask application issues
- CRS/spatial alignment problems

### 3. Validation After Fix:
- Run aggregation on subset of daily data
- Verify output datacube has ~55-72% valid data (matching raw daily)
- Check FCOVER ranges remain 0.0-1.0
- Confirm all 21 bands are properly populated

## Zone-Specific Data Quality

### Raw Daily Data (2022 FCOVER sample):
- **Zone 1**: 0.0% NoData, 72.1% valid, ranges [0.0000, 0.9107]
- **Zone 2**: 0.0% NoData, 55.6% valid, ranges [0.0136, 0.8847]

### Aggregated Weekly Data:
- **Zone 1**: 96.6% NoData (LAI: 96.1%, FAPAR: 96.9%, FCOVER: 96.7%)
- **Zone 2**: 98.2% NoData (LAI: 97.5%, FAPAR: 98.4%, FCOVER: 98.6%)

### Healthy Reference:
- **mfl_cmf_2024**: Normal ranges, no NoData contamination

## Directory Structure Verified

```
/fs2/ftype_mflf/shared_data/temp/biophys/2025/1.0.0/
├── mfl_cmf_area_project_2025_chr_zone_1/
│   ├── fapar/     # Daily files: s2_fapar_YYYY_MM_DD.tif
│   ├── fcover/    # Daily files: s2_fcover_YYYY_MM_DD.tif  
│   └── lai/       # Daily files: s2_lai_YYYY_MM_DD.tif
└── mfl_cmf_area_project_2025_chr_zone_2/
    └── [same structure]

/fs2/ftype_mflf/shared_data/datacube/2025/1.2.0/
├── mfl_cmf_area_project_2025_chr_zone_1/
│   ├── fapar/     # Weekly cubes: cube_biophys_fapar_YYYY_W4before_to_W16after_21bands.tif
│   ├── fcover/    # Weekly cubes: cube_biophys_fcover_YYYY_W4before_to_W16after_21bands.tif
│   └── lai/       # Weekly cubes: cube_biophys_lai_YYYY_W4before_to_W16after_21bands.tif
```

## Commands to Reactivate Investigation

```bash
# Activate geospatial environment
source /home/warrat/yes/etc/profile.d/conda.sh && conda activate gisworkspace-cupy

# Quick check current status
python /fs1/tor_project/model_foresttype/quick_nodata_check.py

# Check raw daily data
python /fs1/tor_project/model_foresttype/quick_raw_data_check.py

# View previous analysis results
cat /fs1/tor_project/model_foresttype/analysis_output/biophysical_analysis_report.txt
```

## Key Insights for Future Reference

1. **Investigation Strategy**: Always check raw data BEFORE processed data to identify where issues are introduced
2. **Environment**: Use `gisworkspace-cupy` conda environment for geospatial scripts
3. **Pattern**: NoData contamination can occur at ANY processing step - don't assume it's from source data
4. **Tools**: Quick sampling scripts (500-2000 pixels) are much faster than full raster analysis

## Status
- **Investigation**: COMPLETE ✅
- **Root Cause**: IDENTIFIED ✅  
- **Next Action**: Fix weekly aggregation code (URGENT)
- **Files**: All analysis scripts and reports saved for future reference

---
*Investigation completed: September 8, 2025*  
*Environment: `/fs1/tor_project/model_foresttype/` with conda gisworkspace-cupy*