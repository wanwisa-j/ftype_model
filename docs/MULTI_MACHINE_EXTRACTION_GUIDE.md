# Forest Type Dataset Extraction - Multi-Machine Setup
# Generated on: September 3, 2025

## Overview
This document provides instructions for running forest type dataset extraction across 3 machines for optimal performance. The extraction is split by zones and years to distribute the computational load.

## Data Distribution Strategy

### Available Zones and Years:
- **mfl_cmf_2024**: 2019-2024 (6 years)
- **mfl_cmf_area_project_2025_chr_zone_1**: 2019-2025 (7 years) 
- **mfl_cmf_area_project_2025_chr_zone_2**: 2019-2025 (7 years)
- **zone_1** to **zone_5**: 2019-2025 (7 years each, 35 years total)

### Machine Distribution:
**Machine 1** (Primary):
- mfl_cmf_2024 (all years: 2019-2024)
- zone_1 (all years: 2019-2025)
- zone_2 (all years: 2019-2025)

**Machine 2**:
- mfl_cmf_area_project_2025_chr_zone_1 (all years: 2019-2025)
- zone_3 (all years: 2019-2025)
- zone_4 (all years: 2019-2025)

**Machine 3**:
- mfl_cmf_area_project_2025_chr_zone_2 (all years: 2019-2025)
- zone_5 (all years: 2019-2025)

## Prerequisites for Each Machine

### 1. Environment Setup
```bash
# Activate conda environment
conda activate gisworkspace-cupy

# Verify Python packages
python -c "import rasterio, geopandas, pandas, numpy; print('All packages available')"
```

### 2. Directory Structure
Ensure the following directories exist on each machine:
```
/fs1/tor_project/model_foresttype/          # Project root
/fs2/ftype_mflf/shared_data/                # Shared data directory
/fs2/ftype_mflf/shared_data/dataset/1.2.0/  # Output directory
```

### 3. Required Files
Copy these files to each machine:
- `scripts/extract_forest_type_dataset.py`
- `config/machine_X_config.yaml` (specific to each machine)
- `scripts/run_extraction_machine_X.sh` (execution script)

## Execution Instructions

### For Machine 1:
```bash
cd /fs1/tor_project/model_foresttype
conda activate gisworkspace-cupy
bash scripts/run_extraction_machine_1.sh
```

### For Machine 2:
```bash
cd /fs1/tor_project/model_foresttype
conda activate gisworkspace-cupy
bash scripts/run_extraction_machine_2.sh
```

### For Machine 3:
```bash
cd /fs1/tor_project/model_foresttype
conda activate gisworkspace-cupy
bash scripts/run_extraction_machine_3.sh
```

## Expected Output

Each machine will generate:
- Raw dataset: `/fs2/ftype_mflf/shared_data/dataset/1.2.0/raw/forest_type_dataset_raw_machine_X_[timestamp].parquet`
- Wide dataset: `/fs2/ftype_mflf/shared_data/dataset/1.2.0/wide/forest_type_dataset_wide_machine_X_[timestamp].parquet`
- Logs: `/fs1/tor_project/model_foresttype/logs/extraction_machine_X_[timestamp].log`

## Data Consolidation

After all machines complete, run on Machine 1:
```bash
python scripts/consolidate_datasets.py
```

This will merge all datasets into final combined files.

## Monitoring Progress

Check progress on each machine:
```bash
tail -f /fs1/tor_project/model_foresttype/logs/extraction_machine_X_*.log
```

## Estimated Runtime

- **Machine 1**: ~3-4 hours (3 zones, 20 years total)
- **Machine 2**: ~3-4 hours (3 zones, 21 years total)  
- **Machine 3**: ~2-3 hours (2 zones, 14 years total)

Total estimated time: 3-4 hours (parallel execution)

## Error Handling

If a machine fails:
1. Check the log file for the specific error
2. Resume from the failed zone using the zone-specific config
3. Contact the team if datacube files are missing

## Contact Information

For issues or questions:
- Primary contact: [Your contact info]
- Data location: /fs2/ftype_mflf/shared_data/
- Project repository: model_foresttype (dev branch)
