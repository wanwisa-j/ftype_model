# Forest Type Model - Investigation Memory

This directory contains detailed investigations and resolutions of critical issues encountered during the forest type classification model development and deployment.

## Investigation Timeline

### September 10, 2025
- **[Dataset_vs_Datacube_Differences.md](./Dataset_vs_Datacube_Differences.md)**: Resolved differences between dataset extraction and datacube access methods
- **[DATA_LEAKAGE_VALIDATION_PROOF_20250910.md](./DATA_LEAKAGE_VALIDATION_PROOF_20250910.md)**: Validated spatial separation strategy
- **[SPATIAL_VALIDATION_STRATEGY_20250910.md](./SPATIAL_VALIDATION_STRATEGY_20250910.md)**: Documented anti-leakage approach
- **[PREDICTION_BIAS_INVESTIGATION_20250910.md](./PREDICTION_BIAS_INVESTIGATION_20250910.md)**: Initial prediction bias analysis

### September 11, 2025  
- **[Prediction_Pipeline_Mismatch_Issue.md](./Prediction_Pipeline_Mismatch_Issue.md)**: **CRITICAL** - Resolved 99.9% class 2.0 prediction bias caused by training-prediction pipeline inconsistencies

## Critical Issues Resolved

### 1. Dataset vs Datacube Differences ✅ RESOLVED
- **Issue**: Different values extracted by dataset builder vs direct datacube access
- **Root Cause**: Comparison script bug, not actual data differences  
- **Resolution**: Updated extraction methods to use consistent pixel indexing

### 2. Prediction Pipeline Mismatch ✅ RESOLVED  
- **Issue**: 99.9% of predictions classified as class 2.0 (MDF)
- **Root Cause**: Multiple pipeline inconsistencies between training and prediction
- **Resolution**: Fixed scaling parameters, feature order, and label encoder loading

## Key Learnings

### Pipeline Consistency is Critical
- **Feature channel order** must match exactly between training and prediction
- **Scaling parameters** must be version-consistent and properly loaded
- **Label encoders** are mandatory for proper class mapping
- **Small inconsistencies** can cause catastrophic prediction failures

### Validation Strategy  
- **Point-based testing** at known training locations catches pipeline issues early
- **Distribution analysis** reveals prediction bias patterns
- **Cross-validation** between training and prediction preprocessing is essential

### Documentation Requirements
- **Explicit feature ordering** prevents channel misalignment
- **Version compatibility matrices** track model-data relationships  
- **Investigation memory** helps prevent recurring issues

## Quick Reference

### Common Issues & Solutions

**Symptom**: Prediction bias toward single class  
**Check**: Feature channel order, scaling parameters, label encoder loading

**Symptom**: Predictions don't match training points  
**Check**: Coordinate transformation consistency, data source alignment

**Symptom**: Model performance degrades in production  
**Check**: Training vs prediction pipeline consistency, data distribution shifts

### Critical Files for Pipeline Consistency
- `scripts/train_model_v2_2_1.py`: Training pipeline reference
- `src/prediction_ftype_2_2_1.py`: Prediction pipeline (fixed)
- `model/mfl_v2_2_2/config/`: Model artifacts and scaling parameters

### Investigation Methodology
1. **Isolate the problem**: Specific symptoms and scope
2. **Compare pipelines**: Training vs prediction step-by-step
3. **Validate hypotheses**: Test specific components
4. **Document solutions**: Comprehensive resolution records
5. **Prevent recurrence**: Add validation and monitoring

---

**These investigations ensure robust, consistent forest type classification workflows.** 🎯