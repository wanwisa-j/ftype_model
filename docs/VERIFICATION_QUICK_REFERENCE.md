# Datacube Verification Quick Reference

## 🚀 Quick Commands

### Single Zone Verification
```bash
# Basic usage
./run_single_zone_verification.sh zone_1

# With custom output
./run_single_zone_verification.sh zone_2 ./my_verification_output
```

### All Zones - Parallel (Fastest)
```bash
# Run all zones simultaneously
./run_zone_verification_parallel.sh
```

### All Zones - Sequential (Resource-friendly)
```bash
# Run zones one by one
./verify_all_zones_sequential.sh
```

### Manual Python Execution
```bash
# Direct Python execution
python datacube_zone_verification.py /fs2/ftype_mflf/shared_data/datacube/2025/1.2.0/zone_1

# With options
python datacube_zone_verification.py /fs2/ftype_mflf/shared_data/datacube/2025/1.2.0/zone_1 \
  --output ./verification_output \
  --max-files 3 \
  --verbose
```

## 📊 Understanding Results

### Success Indicators
- ✅ **PASS**: All checks successful
- **Exit code 0**: Script completed successfully
- **Features found: 10/10**: All expected features present
- **Issues count: 0**: No problems detected

### Failure Indicators  
- ❌ **FAIL**: One or more checks failed
- **Exit code 1**: Script detected issues
- **Features found: X/10** (X < 10): Missing features
- **Issues count: > 0**: Problems detected

## 🔍 What Gets Verified

### ✅ Structure Check
- **Biophysical**: fapar, fcover, lai (7 files each)
- **Optical**: gvmi, ndvi, slavi (7 files each)
- **SAR**: vh, vv (7 files each)
- **Static**: elevation, slope (1 file each)

### ✅ Geometry Check
- **CRS**: EPSG:32647 consistent across all features
- **Transform**: Perfect pixel alignment
- **Shape**: Same dimensions across features

### ✅ Temporal Check
- **Temporal features**: 21 bands each (W4before to W16after)
- **Static features**: 1 band each

### ✅ Quality Check
- **Valid data percentage**: Should be >90%
- **NoData patterns**: Proper interpolation
- **Value ranges**: Within expected bounds

## 📁 Output Files

Each verification generates:
```
verification_results/zone_X/
├── zone_X_verification_results.json    # Machine-readable results
├── zone_X_verification_report.txt      # Human-readable report
└── zone_X_verification_summary.png     # Visual charts
```

## 🛠️ Troubleshooting

### Common Issues

#### Script Permission Denied
```bash
chmod +x run_single_zone_verification.sh
chmod +x run_zone_verification_parallel.sh
chmod +x verify_all_zones_sequential.sh
```

#### Conda Environment Not Found
```bash
# Activate manually
source /opt/miniconda3/etc/profile.d/conda.sh
conda activate gisworkspace-cupy
```

#### Zone Directory Not Found
```bash
# Check available zones
ls -la /fs2/ftype_mflf/shared_data/datacube/2025/1.2.0/
```

### Quick Diagnostics
```bash
# Check if Python script works
python datacube_zone_verification.py --help

# Test single file
ls -la /fs2/ftype_mflf/shared_data/datacube/2025/1.2.0/zone_1/fapar/

# Check environment
which python
conda info --envs
```

## 📋 Expected Results (v1.2.0)

For a healthy datacube, expect:
- **Status**: PASS for all zones
- **Features**: 10/10 for each zone  
- **Files**: 58 total per zone (7×8 temporal + 2 static)
- **Quality**: 100% valid data
- **Processing time**: 5-25 seconds per zone

## ⚡ Performance Tips

### Parallel Execution
- **Max jobs**: Default 3, adjust in script if needed
- **Memory**: ~2GB per concurrent job
- **Time**: ~1-2 minutes for all zones

### Sequential Execution  
- **Memory**: Lower usage, single zone at a time
- **Time**: ~2-5 minutes for all zones
- **Reliability**: Better error isolation

### Single Zone Testing
- **Use case**: Quick checks, debugging specific issues
- **Time**: 5-25 seconds per zone
- **Best for**: Development and troubleshooting

## 🎯 Integration Examples

### Before Dataset Extraction
```bash
# Verify datacube before extracting datasets
if ./run_zone_verification_parallel.sh; then
    echo "✅ Proceeding with dataset extraction"
    python extract_datasets.py
else
    echo "❌ Fix datacube issues first"
    exit 1
fi
```

### In Automation Scripts
```bash
#!/bin/bash
# Pipeline automation example

echo "Step 1: Processing data..."
./run_data_processing.sh

echo "Step 2: Verifying datacube..."
if ! ./run_zone_verification_parallel.sh; then
    echo "ERROR: Datacube verification failed"
    exit 1
fi

echo "Step 3: Creating datasets..."
python create_training_datasets.py

echo "✅ Pipeline completed successfully"
```

### Continuous Integration
```yaml
# Example CI/CD step
- name: Verify Datacube Quality
  run: |
    chmod +x ./run_zone_verification_parallel.sh
    ./run_zone_verification_parallel.sh
```

---

**💡 Pro Tip**: Always run verification before dataset extraction or model training to ensure data quality!

*Quick Reference for Forest Type Model Pipeline v1.2.0*