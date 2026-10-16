# 🚀 DEPLOYMENT PACKAGE READY - Multi-Machine Forest Type Dataset Extraction

**Generated:** September 3, 2025  
**Status:** ✅ ALL PRE-FLIGHT TESTS PASSED  
**Ready for:** 3-Machine Distributed Extraction

## 📊 EXTRACTION OVERVIEW

### Data Distribution:
- **Total Zones:** 8 zones across 2024-2025 datacubes
- **Total Estimated Records:** 500,000+ raw records  
- **Total Estimated Points:** 50,000+ unique spatial points
- **Features:** LAI, NDVI (21 weeks time-series)
- **Static Features:** Elevation, Slope

### Machine Workload:
| Machine | Zones | Zone-Years | Est. Time |
|---------|-------|------------|-----------|
| Machine 1 | `mfl_cmf_2024`, `zone_1`, `zone_2` | 20 | 3-4 hours |
| Machine 2 | `mfl_cmf_area_project_2025_chr_zone_1`, `zone_3`, `zone_4` | 21 | 3-4 hours |
| Machine 3 | `mfl_cmf_area_project_2025_chr_zone_2`, `zone_5` | 14 | 2-3 hours |

## 📁 FILES TO DEPLOY

### Required Files on Each Machine:
```bash
# Core extraction script
scripts/extract_forest_type_dataset.py

# Machine-specific configs
config/machine_1_config.yaml  # Machine 1
config/machine_2_config.yaml  # Machine 2  
config/machine_3_config.yaml  # Machine 3

# Machine-specific runners
scripts/run_extraction_machine_1.sh  # Machine 1
scripts/run_extraction_machine_2.sh  # Machine 2
scripts/run_extraction_machine_3.sh  # Machine 3

# Consolidation script (Machine 1 only)
scripts/consolidate_datasets.py
```

### Documentation:
```bash
docs/MULTI_MACHINE_EXTRACTION_GUIDE.md
docs/AGENTIC_AI_INSTRUCTIONS.md
```

## 🎯 QUICK START COMMANDS

### Machine 1:
```bash
cd /fs1/tor_project/model_foresttype
conda activate gisworkspace-cupy
bash scripts/run_extraction_machine_1.sh
```

### Machine 2:
```bash
cd /fs1/tor_project/model_foresttype
conda activate gisworkspace-cupy
bash scripts/run_extraction_machine_2.sh
```

### Machine 3:
```bash
cd /fs1/tor_project/model_foresttype
conda activate gisworkspace-cupy
bash scripts/run_extraction_machine_3.sh
```

### Consolidation (after all complete):
```bash
cd /fs1/tor_project/model_foresttype
conda activate gisworkspace-cupy
python scripts/consolidate_datasets.py
```

## ✅ VALIDATION CHECKPOINTS

### Pre-Execution Validation:
- [x] Configuration files valid
- [x] Data directories accessible
- [x] All 8 zones available in datacubes
- [x] Execution scripts have proper permissions
- [x] Label file accessible: `ftype_total_2024_2025_train_model_20250826.csv`
- [x] Boundary files accessible in `/fs3/datacenter/ftype_mflf/shared_data/boundary/`

### Execution Monitoring:
- [ ] All 3 machines start successfully
- [ ] Log files show zone processing progress
- [ ] Output files appear in `/fs2/ftype_mflf/shared_data/dataset/1.2.0/`
- [ ] No critical errors in logs
- [ ] Sufficient disk space maintained

### Post-Execution Validation:
- [ ] 6 machine-specific parquet files generated
- [ ] Consolidation produces 2 combined files
- [ ] Combined dataset has 500,000+ records
- [ ] All zones represented in final data
- [ ] Data quality checks pass

## 📈 EXPECTED OUTPUTS

### Per Machine:
```
/fs2/ftype_mflf/shared_data/dataset/1.2.0/raw/
├── forest_type_dataset_raw_machine_1_[timestamp].parquet
├── forest_type_dataset_raw_machine_2_[timestamp].parquet
└── forest_type_dataset_raw_machine_3_[timestamp].parquet

/fs2/ftype_mflf/shared_data/dataset/1.2.0/wide/
├── forest_type_dataset_wide_machine_1_[timestamp].parquet
├── forest_type_dataset_wide_machine_2_[timestamp].parquet
└── forest_type_dataset_wide_machine_3_[timestamp].parquet
```

### Final Consolidated:
```
/fs2/ftype_mflf/shared_data/dataset/1.2.0/raw/
└── forest_type_dataset_raw_combined_[timestamp].parquet

/fs2/ftype_mflf/shared_data/dataset/1.2.0/wide/
└── forest_type_dataset_wide_combined_[timestamp].parquet
```

## 🔍 MONITORING COMMANDS

### Progress Monitoring:
```bash
# Real-time log monitoring
tail -f /fs1/tor_project/model_foresttype/logs/extraction_machine_X_*.log

# Check output file sizes
ls -lh /fs2/ftype_mflf/shared_data/dataset/1.2.0/raw/
ls -lh /fs2/ftype_mflf/shared_data/dataset/1.2.0/wide/

# System resource monitoring
htop
df -h /fs2/ftype_mflf/shared_data/
```

## 🚨 TROUBLESHOOTING

### Common Issues:
1. **Environment not found:** `conda activate gisworkspace-cupy`
2. **Permission denied:** `chmod +x scripts/run_extraction_machine_*.sh`
3. **Out of memory:** Reduce `chunk_size` in config from 5000 to 2000
4. **Missing datacube:** Check zone availability with `ls /fs2/ftype_mflf/shared_data/datacube/*/1.2.0/`

### Recovery Strategy:
If any machine fails, edit its config to process only the remaining zones and restart.

## 🎉 SUCCESS CRITERIA

The extraction is successful when:
1. ✅ All 3 machines complete without errors
2. ✅ Combined raw dataset has 500,000+ records  
3. ✅ Combined wide dataset has 50,000+ unique points
4. ✅ All 8 zones represented in final data
5. ✅ Time series data has 21 weeks for LAI/NDVI
6. ✅ No missing values in critical columns

---

## 🚀 DEPLOYMENT IS READY!

**All systems are GO for the 3-machine distributed extraction.**

The setup has been validated and is ready for immediate deployment. Each machine has been allocated an optimal workload to complete the full dataset extraction in 3-4 hours of parallel processing.

**Next Step:** Deploy files to machines and execute the extraction simultaneously!
