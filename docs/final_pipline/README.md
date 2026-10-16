# Forest Type Model - Complete Data Pipeline Documentation

This directory contains comprehensive documentation for the complete forest type classification data pipeline, covering all stages from data acquisition through model prediction.

⚠️ **CRITICAL UPDATES (September 11, 2025):** Major pipeline fixes applied to resolve 99.9% class 2.0 prediction bias. All prediction documentation updated with critical fixes and validation requirements.

## 📚 Documentation Files

### 🔍 Start Here
- **[DATA_PIPELINE_OVERVIEW.md](./DATA_PIPELINE_OVERVIEW.md)** - Complete pipeline architecture and workflow
- **[DOCUMENTATION_INDEX.md](./DOCUMENTATION_INDEX.md)** - Index of all documentation files and their coverage

### 📊 Data Sources
- **[GEE_DATA_SOURCE.md](./GEE_DATA_SOURCE.md)** - Google Earth Engine data collection (Sentinel-1, Sentinel-2, SRTM)
- **[BIOPHYSICAL_DATA_SOURCE.md](./BIOPHYSICAL_DATA_SOURCE.md)** - Internal biophysical parameter processing (LAI, FAPAR, FCOVER)

### ⚙️ Data Processing
- **[DATA_PROCESSING_ALIGNMENT.md](./DATA_PROCESSING_ALIGNMENT.md)** - Spatial alignment and temporal interpolation

### 🤖 Model Application
- **[DATASET_EXTRACTION.md](./DATASET_EXTRACTION.md)** - Training dataset creation and feature engineering
- **[PREDICTION_PIPELINE.md](./PREDICTION_PIPELINE.md)** - Model prediction and deployment (v2.2.2 - **FIXED**)

### 🚨 Critical Issues & Solutions
- **[../MEMORY/Prediction_Pipeline_Mismatch_Issue.md](../MEMORY/Prediction_Pipeline_Mismatch_Issue.md)** - Complete investigation of 99.9% prediction bias
- **Pipeline consistency requirements** - Feature order, scaling parameters, label encoding

## 🚀 Quick Navigation

### For New Users
1. **Start**: [DATA_PIPELINE_OVERVIEW.md](./DATA_PIPELINE_OVERVIEW.md) - Understand the complete system
2. **Data Sources**: Read [GEE_DATA_SOURCE.md](./GEE_DATA_SOURCE.md) and [BIOPHYSICAL_DATA_SOURCE.md](./BIOPHYSICAL_DATA_SOURCE.md)
3. **Processing**: Follow [DATA_PROCESSING_ALIGNMENT.md](./DATA_PROCESSING_ALIGNMENT.md)
4. **Implementation**: Use [DATASET_EXTRACTION.md](./DATASET_EXTRACTION.md) and [PREDICTION_PIPELINE.md](./PREDICTION_PIPELINE.md)

### For Developers
- **Architecture**: [DATA_PIPELINE_OVERVIEW.md](./DATA_PIPELINE_OVERVIEW.md)
- **Scripts**: Each document covers specific processing scripts
- **Integration**: Cross-references between all components

### For System Administrators
- **Deployment**: [PREDICTION_PIPELINE.md](./PREDICTION_PIPELINE.md)
- **Storage**: [DATA_PROCESSING_ALIGNMENT.md](./DATA_PROCESSING_ALIGNMENT.md)
- **Monitoring**: Quality control sections in each document

## 📋 Pipeline Summary

### Data Flow
```
GEE Data → ┐
           ├→ Spatial Alignment → Temporal Interpolation → Dataset Extraction → Model Training/Prediction
Biophysical → ┘
```

### Key Features Documented
- **10 Input Features**: VV, VH, GVMI, NDVI, SLAVI, LAI, FAPAR, FCOVER, elevation, slope
- **21-Band Time Series**: Weekly temporal datacubes (W4before to W16after)
- **Multi-Zone Processing**: Scalable across different AOI zones
- **Model v2.2.2**: Current production prediction pipeline (**FIXED** for critical issues)

### Key Scripts Covered
- `src/raster_clip_align.py` - Spatial alignment
- `src/batch_interpolate.py` - Temporal interpolation  
- `scripts/extract_forest_type_dataset.py` - Dataset extraction
- `src/dataset_builder.py` - Dataset management
- `scripts/train_model_v2_2_1.py` - Model training with spatial separation
- `src/prediction_ftype_2_2_1.py` - Model prediction (**FIXED** for v2.2.2 compatibility)

## 📁 Data Locations

### Input Data
```
GEE Exports: /fs2/ftype_mflf/shared_data/gdrive/GEE_Exports_*/
Biophysical: /fs2/ftype_mflf/shared_data/temp/biophys/2025/1.0.0/
```

### Final Datacube
```
/fs2/ftype_mflf/shared_data/datacube/2025/1.2.0/
├── mfl_cmf_area_project_2025_chr_zone_1/
├── mfl_cmf_area_project_2025_chr_zone_2/
└── ...
```

### Model Assets (v2.2.2)
```
model/mfl_v2_2_2/
├── hybrid_lstm_cnn_model-20250910-no-leakage.keras
├── config/
│   ├── scaling_params_v2_2_2_no_leakage-20250910.npz   # CRITICAL
│   └── label_encoder_v2_2_2.joblib                     # CRITICAL
└── best_model.h5
```

## 🔧 Implementation Checklist

### Data Preparation
- [ ] GEE data exports validated
- [ ] Biophysical processing completed
- [ ] Spatial alignment executed
- [ ] Temporal interpolation applied

### Model Training
- [ ] Ground truth data prepared
- [ ] Features extracted at survey points
- [ ] Training dataset created
- [ ] Model training completed

### Prediction Deployment (v2.2.2)
- [ ] Model v2.2.2 assets ready (model + scaling + label encoder)
- [ ] Target area datacube prepared
- [ ] Prediction pipeline configured with EXACT feature order
- [ ] Pipeline consistency validated (training vs prediction)
- [ ] Test predictions at training locations
- [ ] Output validation performed

## 📊 Quality Metrics

### Data Quality
- **Spatial Consistency**: <0.1 pixel registration error
- **Temporal Completeness**: >98% time series coverage
- **Interpolation Quality**: <20% interpolated values per pixel

### Model Performance
- **Overall Accuracy**: 80-95% (depending on area and data quality)
- **Confidence Scores**: Available for uncertainty assessment
- **Cross-Validation**: Documented validation procedures

## 🔄 Version Information

- **Documentation Version**: 1.1 (Updated with critical fixes)
- **Creation Date**: September 8, 2025
- **Last Updated**: September 11, 2025 (Critical pipeline fixes)
- **Pipeline Version**: 2.2.2 (current production - FIXED)
- **Datacube Version**: 1.2.0 (current production)

## 📞 Support

### Getting Started
1. Read [DATA_PIPELINE_OVERVIEW.md](./DATA_PIPELINE_OVERVIEW.md) first
2. Identify your use case (training vs. prediction)
3. Follow relevant documentation sections
4. Check cross-references for dependencies

### Troubleshooting
- **Prediction Bias Issues**: See [MEMORY/Prediction_Pipeline_Mismatch_Issue.md](../MEMORY/Prediction_Pipeline_Mismatch_Issue.md)
- **Feature Order Mismatches**: Verify training vs prediction consistency
- **Missing Model Artifacts**: Check scaling parameters and label encoder files
- Each document includes error handling sections
- Check processing logs for specific issues
- Validate input data quality and formats

### Updates and Maintenance
- Documentation synchronized with code changes
- Version tracking for pipeline components
- Regular validation of documentation accuracy

---

**Created**: September 8, 2025  
**Status**: Complete and ready for use  
**Coverage**: End-to-end forest type classification pipeline  
**Total Files**: 7 comprehensive documentation files
