# Forest Type Classification Data Pipeline Documentation

This documentation package provides comprehensive guides for the forest type classification data pipeline, covering all stages from raw data acquisition to model prediction.

## 🔍 Recent Critical Findings

**Data Quality Issues Identified (January 2025):**
- Zones 1 and 2 have severe no-data problems (96%+ invalid pixels)
- Root cause: Interpolation pipeline failure for biophysical features  
- Current solution: Use only clean zones (mfl_cmf_2024) and implement robust outlier filtering

## 📚 Complete Documentation Structure

### 🆕 **Complete Pipeline Documentation (Comprehensive)**
**📁 Location**: [**./final_pipline/**](./final_pipline/) - **All comprehensive pipeline documentation**

- [**DATA_PIPELINE_OVERVIEW.md**](./final_pipline/DATA_PIPELINE_OVERVIEW.md) - **Complete pipeline architecture and workflow**
- [**GEE_DATA_SOURCE.md**](./final_pipline/GEE_DATA_SOURCE.md) - **Google Earth Engine data collection and processing**
- [**BIOPHYSICAL_DATA_SOURCE.md**](./final_pipline/BIOPHYSICAL_DATA_SOURCE.md) - **Internal biophysical parameter processing**
- [**DATA_PROCESSING_ALIGNMENT.md**](./final_pipline/DATA_PROCESSING_ALIGNMENT.md) - **Spatial alignment and temporal interpolation**
- [**DATASET_EXTRACTION.md**](./final_pipline/DATASET_EXTRACTION.md) - **Training dataset creation and feature engineering**
- [**PREDICTION_PIPELINE.md**](./final_pipline/PREDICTION_PIPELINE.md) - **Model prediction and deployment (v2.2.0)**
- [**DOCUMENTATION_INDEX.md**](./final_pipline/DOCUMENTATION_INDEX.md) - **Complete documentation index and cross-references**

### 📋 **Legacy Documentation (Reference)**

### 📋 [Data Pipeline Overview](./data_pipeline_overview.md)
- **Purpose**: High-level architecture and workflow description
- **Key Content**:
  - Pipeline stages and data flow
  - Data sources and formats
  - Quality assurance measures
  - Performance optimization strategies

### 🗄️ [Data Storage and Query Methods](./data_storage_and_query.md)
- **Purpose**: Storage architecture and data access patterns
- **Key Content**:
  - Multi-tiered storage strategy
  - Data formats and encoding
  - Query APIs and access patterns
  - Performance benchmarks

### 🚀 [Implementation Guide](./implementation_guide.md)
- **Purpose**: Step-by-step implementation instructions
- **Key Content**:
  - Quick start guide
  - Configuration options
  - Troubleshooting guide
  - Integration examples

### 🔬 [Feature Engineering and Models](./feature_engineering_and_models.md)
- **Purpose**: Advanced ML techniques and model architectures
- **Key Content**:
  - Advanced feature engineering methods
  - Model architecture recommendations
  - Evaluation strategies
  - Performance optimization

## Pipeline Summary

### Data Sources
- **Ground Truth Labels**: Forest type classifications at point locations
- **Satellite Datacubes**: Multi-temporal time-series features (2019-2024, selectable by year)
- **Static Features**: Elevation and slope data (selectable)
- **Spatial Boundaries**: Processing zone definitions (selectable zones)

### Key Features with Selection Flexibility
- **Time-Series Features** (8 available): LAI, FAPAR, FCOVER, GVMI, NDVI, SLAVI, VH, VV
  - **Individual year selection**: Choose specific years for each feature
  - **Feature selection**: Include/exclude specific indices
- **Static Features** (2 available): Elevation, Slope
  - **Individual selection**: Choose which static features to include
- **Temporal Coverage**: Flexible selection from 6 years × 21 weekly timesteps
- **Spatial Coverage**: Selectable from 8 processing zones

### Processing Capabilities with Raw Data Preservation
- **Selective Extraction**: Choose zones, features, and years
- **Raw Data Preservation**: No scaling/normalization applied
- **Pre-interpolated Data**: Time-series data is already interpolated
- **Parallel Processing**: Multi-threaded feature extraction
- **Batch Processing**: Memory-efficient large dataset handling
- **Quality Assurance**: Comprehensive data validation

### Output Formats
- **Raw Features**: Point-based extracted values in original units
- **Selectable Datasets**: Custom combinations of zones/features/years
- **Multiple Targets**: Support for different classification schemes
- **Selection Metadata**: Complete record of data selection choices

## Implementation Checklist

### Prerequisites ✅
- [ ] Python environment with required packages
- [ ] Access to datacube directories
- [ ] Ground truth label file
- [ ] Processing zone boundary files

### Core Implementation 🔧
- [x] Feature extraction system
- [x] Data preprocessing pipeline
- [x] Main pipeline orchestrator
- [x] Error handling and logging

### Testing and Validation 🧪
- [ ] Unit tests for core components
- [ ] Integration testing with sample data
- [ ] Performance benchmarking
- [ ] Data quality validation

### Production Deployment 🚀
- [ ] Configuration management
- [ ] Monitoring and alerting
- [ ] Documentation review
- [ ] Team training

## Quick Start Commands

### Basic Pipeline Execution with Selection
```bash
# Navigate to project directory
cd /fs1/tor_project/model_foresttype

# Run pipeline with custom selection
python src/dataset_pipeline.py \
    --label-file /fs1/tor_project/model_foresttype/data/2025_data_label_20250726/groud_label_data_ftype_for_model_20250826/ftype_total_2024_2025_train_model_20250826.csv \
    --output-dir ./output/dataset_$(date +%Y%m%d) \
    --selected-zones "zone_1,zone_2,mfl_cmf_2024" \
    --selected-features "lai,ndvi,vh,elevation" \
    --selected-years "lai:2020,2021,2022;ndvi:2019,2021,2023;vh:2021,2022" \
    --max-workers 8 \
    --batch-size 200 \
    --no-scaling \
    --preserve-raw-values
```

### Expected Outputs
```
output_directory/
├── complete_features_dataset.pkl      # Raw extracted features (selected subset)
├── ml_dataset_ftype_mix.pkl          # ML-ready dataset (mixed) - original values
├── ml_dataset_ftype_nonmix.pkl       # ML-ready dataset (non-mixed) - original values
├── selection_metadata.json           # Record of selected years/zones/features
├── feature_statistics.csv           # Data quality metrics for selected data
└── pipeline_YYYYMMDD_HHMMSS.log      # Execution log
```

## Key Design Decisions

### 1. Flexible Data Selection
- **Individual Year Selection**: Choose specific years for each feature type
- **Zone Selection**: Select which processing zones to include
- **Feature Selection**: Choose which indices/features to extract
- **Raw Data Preservation**: Keep original values without scaling

### 2. Data Handling Strategy
- **Point-Based Sampling**: Efficient extraction at labeled locations
- **Pre-interpolated Data**: Time-series data is already processed and ready
- **Selective Processing**: Only extract and process selected combinations
- **Original Value Preservation**: No scaling or normalization applied

### 3. Quality Assurance
- **Selection Metadata**: Complete record of choices made
- **Data Validation**: Automatic quality checks for selected data
- **Flexible Configuration**: Easy to modify selection criteria

### 4. Performance Optimization
- **Selective Loading**: Only process selected data combinations
- **Parallel Processing**: Multi-threaded feature extraction
- **Memory Management**: Batch processing for large selections

## Integration Examples

### Machine Learning Workflow with Raw Data
```python
# Load dataset with original values preserved
import pickle
with open('ml_dataset_ftype_mix.pkl', 'rb') as f:
    dataset = pickle.load(f)

# Access raw time-series data (no scaling applied)
X_train_raw = dataset['train']['X_raw']
y_train = dataset['train']['y']

# Check what was selected
selection_config = dataset['selection_config']
print(f"Selected features: {selection_config['selected_features']}")
print(f"Selected years: {selection_config['selected_years']}")
print(f"Selected zones: {selection_config['selected_zones']}")

# Train model with original values
from sklearn.ensemble import RandomForestClassifier
rf_model = RandomForestClassifier(n_estimators=200)

# Use specific time-series data
lai_2021 = dataset['train']['X_timeseries']['lai_2021']  # Shape: (n_samples, 21)
ndvi_2022 = dataset['train']['X_timeseries']['ndvi_2022']  # Shape: (n_samples, 21)

# Combine features as needed
import numpy as np
X_combined = np.concatenate([
    lai_2021.reshape(len(lai_2021), -1),  # Flatten time series
    ndvi_2022.reshape(len(ndvi_2022), -1)
], axis=1)

rf_model.fit(X_combined, y_train)
print(f"Model trained with original data values - no scaling applied")
```

### Time-Series Analysis with Selected Data
```python
# Analyze temporal patterns for selected feature-year combinations
import matplotlib.pyplot as plt

def plot_selected_timeseries(dataset, sample_idx=0):
    """Plot time series for selected features"""
    X_timeseries = dataset['train']['X_timeseries']
    
    fig, axes = plt.subplots(len(X_timeseries), 1, figsize=(12, 3*len(X_timeseries)))
    if len(X_timeseries) == 1:
        axes = [axes]
    
    weeks = range(1, 22)
    
    for i, (feature_year, data) in enumerate(X_timeseries.items()):
        axes[i].plot(weeks, data[sample_idx], 'b-', linewidth=2)
        axes[i].set_title(f'{feature_year.upper()} - Original Values')
        axes[i].set_xlabel('Week')
        axes[i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

# Plot example
plot_selected_timeseries(dataset)
```

## Performance Expectations

### Processing Speed
- **Feature Extraction**: 100-1000 points/minute (depending on selections and hardware)
- **Selection Processing**: Faster processing with fewer selected features
- **Memory Usage**: 2-8 GB RAM depending on selection scope

### Dataset Characteristics
- **Flexible Size**: Depends on zone/feature/year selections
- **Feature Dimensions**: Variable based on selections (typically 100-2000 features per point)
- **Original Values**: Data preserved in original physical units
- **File Sizes**: 10MB-10GB depending on selection scope

### Accuracy Expectations
- **Baseline Models**: 70-85% accuracy with minimal feature selection
- **Comprehensive Selection**: 80-95% accuracy with multi-year, multi-feature data
- **Original Value Benefits**: Better interpretability and physical meaning

## Next Steps and Recommendations

### Immediate Actions
1. **Review Documentation**: Ensure understanding of selection capabilities
2. **Define Selection Criteria**: Choose zones, features, and years for your use case
3. **Test with Minimal Selection**: Start with small subset for validation
4. **Configure Parameters**: Adjust selection settings for your requirements

### Future Enhancements
1. **Expand Selections**: Add new zones or features as they become available
2. **Optimize Performance**: Profile and optimize for your specific selections
3. **Develop Selection APIs**: Create tools for easier selection management
4. **Add Selection Validation**: Implement checks for selection compatibility

### Maintenance
1. **Monitor Selection Quality**: Regular validation of selected data combinations
2. **Update Selections**: Modify as new data becomes available
3. **Version Control Selections**: Track different selection configurations
4. **Document Custom Selections**: Keep records of successful selection strategies

## Support and Contact

### Getting Help
- **Technical Issues**: Check troubleshooting guide in implementation documentation
- **Performance Problems**: Review optimization recommendations
- **New Features**: Follow extension patterns in the codebase

### Documentation Updates
- Keep documentation current with implementation changes
- Add examples for new use cases
- Update performance benchmarks as system evolves

---

**Document Version**: 1.0  
**Last Updated**: September 3, 2025  
**Authors**: Forest Type Classification Team  
**Status**: Ready for Implementation
