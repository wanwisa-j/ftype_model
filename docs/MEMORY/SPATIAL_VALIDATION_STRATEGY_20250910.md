# Spatial Validation Strategy - Model v2.3.0
**Date:** 2025-09-10  
**Purpose:** Implement proper spatial validation for forest type classification  
**Status:** **IMPLEMENTED IN NOTEBOOK**

## Spatial Validation Strategy

### Problem Addressed
- Previous model (v2.2.0) had data leakage - same areas used for training and testing
- Need independent geographic validation before deploying model
- Standard random train/test split doesn't account for spatial autocorrelation

### Solution: Three-Tier Validation System

#### 1. **Training Set**
- **Zones**: `mfl_cmf_2024`, `mfl_cmf_area_project_2025_chr_zone_2`
- **Purpose**: Model training only
- **Samples**: ~8,200 (reduced from original 10,522 due to zone exclusion)

#### 2. **In-Training Validation** (20% Random Holdout)
- **Source**: 20% random sample from training zones
- **Purpose**: Early stopping, hyperparameter tuning, training monitoring
- **Type**: Standard random split within training zones
- **Usage**: `validation_data` in model.fit()

#### 3. **Spatial Validation** (Independent Zone)
- **Zone**: `mfl_cmf_area_project_2025_chr_zone_1` (completely held out)
- **Purpose**: True performance evaluation before deployment
- **Type**: Geographic/spatial holdout
- **Usage**: Final model evaluation and data leakage detection

## Implementation Details

### Data Flow
```python
# 1. Create training dataset (excludes zone_1)
training_zones = ['mfl_cmf_2024', 'mfl_cmf_area_project_2025_chr_zone_2']
dataset = builder.create_dataset(zones=training_zones, ...)

# 2. Create spatial validation dataset  
spatial_val_zones = ['mfl_cmf_area_project_2025_chr_zone_1']
spatial_val_dataset = builder.create_dataset(zones=spatial_val_zones, ...)

# 3. Standard train/test split within training zones
X_train, X_test, y_train, y_test = train_test_split(training_data, test_size=0.2)

# 4. Apply scaling (computed from training set only)
# All three sets scaled using training statistics
```

### Key Features

#### Consistent Preprocessing
- Same feature engineering for all sets (VH/VV ratio, filtering, etc.)
- Scaling parameters computed from training set only
- Applied consistently to training, in-training validation, and spatial validation

#### Comprehensive Evaluation
```python
# In-training validation (during training)
model.fit(x=train_features, validation_data=(test_features, test_targets))

# Spatial validation (after training)  
spatial_predictions = model.predict([X_spatial_val_timeseries, X_spatial_val_static])
```

#### Data Leakage Detection
Automated checks for:
- Single-class predictions (100% one class)
- Extreme class dominance (>95% one class)
- Comparison with ground truth distribution

## Expected Results

### Healthy Model (No Leakage)
```
Spatial Validation Results:
Predicted class distribution:
   MDF (Class 2): ~240 predictions (77%)
   DDF (Class 0): ~50 predictions (16%) 
   DEF (Class 1): ~22 predictions (7%)

Ground truth distribution:
   MDF (Class 2): 247 samples (79%)
   DDF (Class 0): 51 samples (16%)
   DEF (Class 1): 8 samples (3%)
```

### Data Leakage Indicators
```
⚠️ WARNING Signs:
- Model predicts only one class (e.g., 100% MDF)
- One class dominates >95% of predictions
- Perfect accuracy on spatial validation (unrealistic)
```

## Benefits

### 1. **Realistic Performance Assessment**
- Spatial validation provides true performance on independent geographic area
- Accounts for spatial autocorrelation in forest data
- Prevents overoptimistic accuracy estimates

### 2. **Data Leakage Detection**
- Automated checks flag potential leakage issues
- Comparison of prediction vs ground truth distributions
- Early warning system for model deployment

### 3. **Deployment Confidence**
- Model validated on actual prediction target area
- Known performance characteristics before full deployment
- Reduced risk of production failures

### 4. **Scientific Rigor**
- Follows best practices for spatial machine learning
- Addresses spatial autocorrelation properly
- Enables meaningful performance reporting

## Validation Metrics

### Performance Metrics
- **Classification accuracy** on spatial validation set
- **Precision, Recall, F1-score** per class
- **Confusion matrix** analysis
- **Class distribution** comparison

### Leakage Detection Metrics
- **Prediction entropy**: Should be > 0 for healthy model
- **Dominant class percentage**: Should be < 90% for balanced predictions
- **Distribution similarity**: Predictions should roughly match ground truth ratios

## Files Modified

### Notebook Cells Updated
1. **Dataset Creation**: Spatial separation of zones
2. **Feature Scaling**: Include spatial validation in scaling pipeline  
3. **Label Encoding**: Handle all three datasets consistently
4. **Model Evaluation**: Comprehensive evaluation with leakage detection

### New Variables Added
```python
# Spatial validation data
X_spatial_val_timeseries    # Time series features for zone_1
X_spatial_val_static        # Static features for zone_1  
y_spatial_val_encoded       # Target labels for zone_1
spatial_val_results         # Evaluation results storage
```

## Success Criteria

### Model Acceptance
- ✅ **Diverse predictions**: Multiple classes predicted (not 100% one class)
- ✅ **Reasonable accuracy**: >70% on spatial validation
- ✅ **Balanced predictions**: No single class >90% of predictions
- ✅ **Consistent performance**: Similar accuracy on in-training vs spatial validation

### Model Rejection
- ❌ **Single class dominance**: >95% predictions in one class
- ❌ **Perfect accuracy**: 100% accuracy (indicates leakage)
- ❌ **Zero entropy**: All predictions identical
- ❌ **Massive performance gap**: Training accuracy >> spatial accuracy

---
**Implementation Status**: Complete  
**Next Step**: Run training with spatial validation  
**Expected Outcome**: Legitimate, diverse forest predictions on zone_1