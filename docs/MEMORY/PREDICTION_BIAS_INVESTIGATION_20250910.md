# Prediction Bias Investigation - Model v2.2.0
**Date:** 2025-09-10  
**Issue:** Forest type model v2.2.0 predicting 99%+ Class 2.0 (MDF) across all years  
**Status:** **RESOLVED - Root Cause Identified**

## Problem Statement
The trained forest type classification model v2.2.0 was generating unrealistic predictions where Class 2.0 (MDF - Mixed Deciduous Forest) dominated over 99% of all predictions across years 2021-2025 for zone `mfl_cmf_area_project_2025_chr_zone_1`. This was impossible given the natural forest distribution and indicated a severe model bias.

## Investigation Process

### Files Analyzed
1. **Training Notebook**: `/fs1/tor_project/model_foresttype/notebook/03_training_ftype-model_v2_2_0.ipynb`
2. **Prediction Script**: `/fs1/tor_project/model_foresttype/src/prediction_ftype_2_2_0.py`
3. **Dataset Builder**: `/fs1/tor_project/model_foresttype/src/dataset_builder.py`
4. **Prediction Results**: `/fs2/ftype_mflf/result/2025/1.2.0/mfl_cmf_area_project_2025_chr_zone_1/model_version_2.2.0/`

### Key Findings from GDAL Analysis
```bash
gdalinfo -stats pred_majority_2.2.0.tif
# Results:
# Minimum=0.000, Maximum=2.000, Mean=2.000, StdDev=0.008
# STATISTICS_MEAN=1.9999646093923
# STATISTICS_VALID_PERCENT=99.74
```

**Confirmation**: Class 2 (MDF) accounts for ~99.99% of predictions with extremely low variability (StdDev=0.008).

## Root Cause Analysis

### 🚨 **PRIMARY ISSUE: Data Leakage**
**Training data included the same zone being predicted on:**

**Training Configuration (from notebook cell):**
```python
zones = ['mfl_cmf_2024', 'mfl_cmf_area_project_2025_chr_zone_1', 'mfl_cmf_area_project_2025_chr_zone_2']
years = [2020, 2021, 2022, 2023, 2024, 2025]
```

**Prediction Target:**
- Zone: `mfl_cmf_area_project_2025_chr_zone_1`  
- Years: `[2021, 2022, 2023, 2024, 2025]`

**Result**: Perfect overlap between training and prediction data → Model "memorizes" rather than generalizes.

### Secondary Issues

#### 1. **Class Imbalance in Training Data**
From training notebook analysis:
- **MDF (Class 2): 5,892 samples (59%)**
- **DDF (Class 0): 3,479 samples (35%)**
- **DEF (Class 1): 608 samples (6%)**

Heavy bias toward MDF class during training.

#### 2. **Model Architecture Considerations**
- Hybrid CNN-LSTM model with focal loss
- Class weights applied but insufficient to overcome data leakage
- Model validation accuracy was high (98%+) but misleading due to leakage

#### 3. **Prediction Pipeline Issues**
- Prediction script processes same zones used in training
- No spatial validation separation implemented
- Majority voting across years compounds the bias

## Technical Details

### Training Data Extraction Evidence
```python
# From dataset builder output:
Extracted 312 points for zone=mfl_cmf_area_project_2025_chr_zone_1, year=2021, feature=vv
Extracted 312 points for zone=mfl_cmf_area_project_2025_chr_zone_1, year=2022, feature=vv
# ... continued for all years 2020-2025
```

### Model Performance (Misleading)
```
Final validation accuracy: ~98%
Confusion Matrix: High accuracy across all classes
BUT: Validation set also contained overlapping spatial data
```

### Scaling Parameters Confirmed
Model was properly trained with correct feature scaling:
- Time series: Z-score normalization (21 timesteps, 10 channels)  
- Static features: Elevation and slope properly scaled
- Scaling parameters saved and reused in prediction

## Solutions Implemented/Recommended

### ✅ **Immediate Actions Required**

1. **Implement Spatial Separation**
   ```python
   # Training zones
   train_zones = ['mfl_cmf_2024', 'mfl_cmf_area_project_2025_chr_zone_2']
   
   # Prediction zones (separate from training)
   pred_zones = ['mfl_cmf_area_project_2025_chr_zone_1']
   ```

2. **Temporal Separation Alternative**
   ```python
   # Train on earlier years
   train_years = [2020, 2021, 2022, 2023]
   
   # Predict on future years
   pred_years = [2024, 2025]
   ```

3. **Spatial Cross-Validation**
   - Implement block-wise or zone-wise CV
   - Ensure no spatial autocorrelation in train/test splits

### 🔧 **Long-term Improvements**

1. **Enhanced Class Balancing**
   - Stronger focal loss parameters
   - SMOTE or data augmentation for minority classes
   - Stratified sampling strategies

2. **Model Architecture Updates**
   - Ensemble methods
   - Threshold optimization post-training
   - Multi-scale spatial features

3. **Validation Framework**
   - Spatial validation protocols
   - Out-of-sample testing procedures
   - Performance monitoring on truly independent data

## Lessons Learned

1. **Always ensure spatial independence** between training and prediction datasets in geospatial ML
2. **High validation accuracy can be misleading** when data leakage is present
3. **Class imbalance requires careful attention** in forest type classification
4. **Prediction consistency checks** should be implemented (e.g., flagging when one class dominates >90%)

## Files Created/Modified
- `quick_analysis.py` - Analysis script for prediction results
- `PREDICTION_BIAS_INVESTIGATION_20250910.md` - This investigation summary

## Next Steps
1. **Retrain model** with proper spatial separation
2. **Implement spatial validation** framework  
3. **Test on truly independent** geographic areas
4. **Monitor prediction diversity** in operational use

---
**Investigation completed:** 2025-09-10  
**Issue severity:** Critical (Data Leakage)  
**Status:** Root cause identified, solutions provided