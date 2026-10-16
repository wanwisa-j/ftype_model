# Model v2.3.0 Training Changes - Data Leakage Fix
**Date:** 2025-09-10  
**Purpose:** Fix data leakage in forest type classification model  
**Status:** **READY FOR TRAINING**

## Changes Made to Training Notebook

### File Modified
- **Notebook**: `/fs1/tor_project/model_foresttype/notebook/03_training_ftype-model_v2_2_0.ipynb`
- **Version**: Upgraded from v2.2.0 → v2.3.0

### Key Changes

#### 1. **Training Zone Configuration** (Cell: Dataset Creation)
**Before (v2.2.0):**
```python
zones = ['mfl_cmf_2024', 'mfl_cmf_area_project_2025_chr_zone_1', 'mfl_cmf_area_project_2025_chr_zone_2']
```

**After (v2.3.0):**
```python
training_zones = [
    'mfl_cmf_2024',                                    # Different geographic area
    'mfl_cmf_area_project_2025_chr_zone_2'            # Different zone in same project
    # EXCLUDED: 'mfl_cmf_area_project_2025_chr_zone_1'  # Reserved for prediction ONLY
]
```

#### 2. **Model Save Path** (Cell: Model Saving)
**Before:**
```python
model.save('/fs1/tor_project/model_foresttype/model/mfl_v2_2_0/hybrid_lstm_cnn_model-20250910.keras')
```

**After:**
```python
model_version = "2.3.0"  # New version - spatially separated training
save_path = f'/fs1/tor_project/model_foresttype/model/mfl_v{model_version.replace(".", "_")}/hybrid_lstm_cnn_model-20250910-no-leakage.keras'
```

#### 3. **Scaling Parameters Path** (Cell: Scaling Parameters)
**Before:**
```python
save_path = '/fs1/tor_project/model_foresttype/model/mfl_v2_2_0/config/scaling_params_from_training_hybrid_lstm_cnn_model-20250910.npz'
```

**After:**
```python
save_path = f'{config_dir}/scaling_params_v{model_version.replace(".", "_")}_no_leakage-20250910.npz'
```

## Spatial Separation Strategy

### Training Data (Model v2.3.0)
- **Zone 1**: `mfl_cmf_2024` (Different geographic area)
- **Zone 2**: `mfl_cmf_area_project_2025_chr_zone_2` (Different zone in project)
- **Years**: 2020-2025 (where data available)
- **Samples**: ~8,200 samples (reduced from 10,522 but spatially independent)

### Prediction Target (Held Out)
- **Zone**: `mfl_cmf_area_project_2025_chr_zone_1` 
- **Status**: **COMPLETELY EXCLUDED** from training
- **Purpose**: Zero data leakage for legitimate predictions

## Expected Training Impact

### Sample Size Reduction
- **v2.2.0**: 10,522 samples (with data leakage)
- **v2.3.0**: ~8,200 samples (without zone_1 data)
- **Reduction**: ~22% fewer samples but **legitimate training**

### Class Distribution Impact
The exclusion of `mfl_cmf_area_project_2025_chr_zone_1` may slightly change class distribution:
- Original zone_1 had: MDF (79%), DDF (16%), DEF (3%)
- Remaining zones should still provide balanced representation
- Class weights in training remain the same

## Output Files (v2.3.0)

### Model Files
```
/fs1/tor_project/model_foresttype/model/mfl_v2_3_0/
├── hybrid_lstm_cnn_model-20250910-no-leakage.keras
└── config/
    └── scaling_params_v2_3_0_no_leakage-20250910.npz
```

### Metadata Included
- Model version: 2.3.0
- Training zones: ['mfl_cmf_2024', 'mfl_cmf_area_project_2025_chr_zone_2']
- Excluded prediction zone: 'mfl_cmf_area_project_2025_chr_zone_1'
- Data leakage status: FIXED

## Validation Strategy

### Expected Results
With proper spatial separation, predictions on `mfl_cmf_area_project_2025_chr_zone_1` should show:
- **Realistic diversity**: Not 100% Class 2
- **Approximate ground truth distribution**: ~79% MDF, ~16% DDF, ~3% DEF
- **Model confidence variation**: Different confidence levels across pixels

### Validation Test
After training v2.3.0, run the same validation as before:
```python
# This should show diverse predictions, not 100% Class 2
label_data_predict['predicted_class'].value_counts()
```

## Next Steps

### 1. **Train Model v2.3.0**
- Execute the modified notebook
- Monitor training metrics
- Save model and scaling parameters

### 2. **Update Prediction Script**
- Modify `src/prediction_ftype_2_2_0.py` to use v2.3.0 model
- Update scaling parameters path
- Update model path

### 3. **Validate Results**
- Run predictions on `mfl_cmf_area_project_2025_chr_zone_1`
- Verify diverse class predictions (not 100% MDF)
- Compare with ground truth distribution

### 4. **Production Deployment**
- Replace v2.2.0 model with v2.3.0 in production
- Update documentation
- Monitor prediction quality

## Risk Mitigation

### Potential Issues
1. **Reduced training data**: 22% fewer samples
2. **Geographic bias**: Training zones may not fully represent prediction zone
3. **Class imbalance**: May be affected by zone exclusion

### Monitoring
- Watch training/validation metrics for degradation
- Check class prediction diversity in validation
- Monitor model confidence distributions

---
**Status**: Ready for training  
**Expected outcome**: Legitimate, diverse forest type predictions  
**Risk level**: Low (proper ML practices implemented)