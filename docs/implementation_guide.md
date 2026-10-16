# Implementation Plan and Usage Guide

## Implementation Phases

### Phase 1: Core Infrastructure (Estimated: 2-3 days)
**Status**: ✅ Completed
- [x] Feature extraction system (`feature_extractor.py`)
- [x] Data preprocessing pipeline (`data_preprocessor.py`)
- [x] Main pipeline orchestrator (`dataset_pipeline.py`)
- [x] Basic error handling and logging

### Phase 2: Testing and Validation (Estimated: 1-2 days)
**Status**: 🔄 Next Phase
- [ ] Unit tests for core components
- [ ] Integration testing with sample data
- [ ] Performance benchmarking
- [ ] Data quality validation

### Phase 3: Optimization and Enhancement (Estimated: 1-2 days)
**Status**: 📋 Planned
- [ ] Memory optimization for large datasets
- [ ] Advanced caching mechanisms
- [ ] Query API development
- [ ] Documentation completion

### Phase 4: Production Deployment (Estimated: 1 day)
**Status**: 📋 Planned
- [ ] Configuration management
- [ ] Monitoring and alerting
- [ ] User training and documentation
- [ ] Production data processing

## Quick Start Guide

### Prerequisites
```bash
# Required Python packages
pip install rasterio geopandas pandas numpy scikit-learn tqdm xarray rioxarray

# Optional for advanced features
pip install dask distributed joblib
```

### Basic Usage

#### 1. Complete Pipeline Execution with Selection
```bash
cd /fs1/tor_project/model_foresttype

# Run complete pipeline with feature/year/zone selection
python src/dataset_pipeline.py \
    --label-file /fs1/tor_project/model_foresttype/data/2025_data_label_20250726/groud_label_data_ftype_for_model_20250826/ftype_total_2024_2025_train_model_20250826.csv \
    --output-dir ./output/dataset_$(date +%Y%m%d) \
    --selected-zones "zone_1,zone_2,mfl_cmf_2024" \
    --selected-features "lai,ndvi,vh,elevation" \
    --selected-years "lai:2020,2021,2022;ndvi:2019,2021,2023;vh:2021,2022" \
    --max-workers 8 \
    --batch-size 200 \
    --test-size 0.2 \
    --no-scaling \
    --preserve-raw-values
```

#### 2. Feature Extraction with Custom Selection
```bash
# Extract only specific features and years
python src/dataset_pipeline.py \
    --label-file path/to/labels.csv \
    --output-dir ./output/custom_features \
    --selected-zones "zone_1,zone_3" \
    --selected-features "lai,ndvi,elevation" \
    --selected-years "lai:2020,2022;ndvi:2021,2023" \
    --max-workers 4 \
    --raw-data-only
```

#### 3. Using Existing Features with Different Selection
```bash
# Use existing features but apply different selection criteria
python src/dataset_pipeline.py \
    --label-file path/to/labels.csv \
    --output-dir ./output/reselected \
    --skip-extraction \
    --features-file ./output/all_features/complete_features_dataset.pkl \
    --selected-features "ndvi,vh,slope" \
    --selected-years "ndvi:2020,2021;vh:2022,2023"
```

### Advanced Usage Examples

#### Custom Feature and Year Selection
```python
from src.feature_extractor import ForestTypeFeatureExtractor

# Define custom selection
selection_config = {
    'zones': ['zone_1', 'zone_2', 'mfl_cmf_2024'],
    'features': {
        'lai': [2020, 2021, 2022],        # LAI for 3 specific years
        'ndvi': [2019, 2021, 2023],       # NDVI for different years
        'vh': [2021, 2022],               # VH for 2 years only
        'elevation': True,                 # Include elevation
        'slope': False                     # Exclude slope
    }
}

# Initialize extractor with selection
extractor = ForestTypeFeatureExtractor(
    datacube_config=datacube_config,
    bbox_areas_config=bbox_areas_config,
    selection_config=selection_config,
    preserve_raw_values=True,              # No scaling
    years_range=(2019, 2024)               # Available range
)
```

#### Raw Data Processing (No Preprocessing)
```python
from src.data_processor import RawDataProcessor

# Initialize processor for raw data handling
processor = RawDataProcessor(
    apply_scaling=False,                   # Keep original values
    apply_imputation=False,                # Data is pre-interpolated
    preserve_temporal_structure=True       # Keep time-series format
)

# Process with selection
processed_data = processor.process_raw_features(
    features_df,
    selection_config=selection_config,
    target_col='ftype_mix'
)
```

## Configuration Options

### Pipeline Configuration
```python
# Dataset pipeline configuration with selection capabilities
CONFIG = {
    'data_selection': {
        'zones': ['zone_1', 'zone_2', 'mfl_cmf_2024'],  # Selected zones
        'features': {
            'lai': [2020, 2021, 2022],      # Feature with specific years
            'ndvi': [2019, 2021, 2023],     # Different years per feature
            'vh': [2021, 2022],             # Subset of years
            'elevation': True,               # Include static feature
            'slope': False                   # Exclude static feature
        },
        'targets': ['ftype_mix', 'ftype_nonmix'],  # Both classification targets
        'min_data_completeness': 0.7        # Minimum data quality threshold
    },
    'extraction': {
        'max_workers': 8,                   # Parallel processing threads
        'batch_size': 200,                  # Points per batch
        'preserve_raw_values': True,        # No scaling/normalization
        'handle_missing': 'skip'            # Skip missing combinations
    },
    'processing': {
        'apply_scaling': False,             # Keep original data values
        'apply_imputation': False,          # Data is pre-interpolated
        'preserve_temporal_structure': True, # Keep time-series format
        'test_size': 0.2,                  # Train/test split ratio
        'random_state': 42                 # Reproducibility
    },
    'output': {
        'save_intermediate': True,          # Save batch results
        'save_selection_metadata': True,    # Record selection choices
        'save_raw_format': True,           # Keep unprocessed format
        'compress_output': False           # No compression for interpretability
    }
}
```

### Selection Configuration Examples
```python
# Example 1: Minimal selection for quick testing
MINIMAL_CONFIG = {
    'zones': ['zone_1'],
    'features': {
        'ndvi': [2021, 2022],
        'elevation': True
    }
}

# Example 2: Comprehensive multi-year analysis
COMPREHENSIVE_CONFIG = {
    'zones': ['zone_1', 'zone_2', 'zone_3', 'mfl_cmf_2024'],
    'features': {
        'lai': [2019, 2020, 2021, 2022, 2023, 2024],
        'fapar': [2020, 2021, 2022, 2023],
        'ndvi': [2019, 2020, 2021, 2022, 2023, 2024],
        'gvmi': [2021, 2022, 2023],
        'vh': [2020, 2021, 2022, 2023],
        'vv': [2020, 2021, 2022, 2023],
        'elevation': True,
        'slope': True
    }
}

# Example 3: Focused forest type analysis
FOREST_FOCUS_CONFIG = {
    'zones': ['mfl_cmf_2024', 'zone_1', 'zone_2'],
    'features': {
        'lai': [2021, 2022, 2023],          # Vegetation structure
        'ndvi': [2021, 2022, 2023],         # Vegetation vigor
        'fapar': [2021, 2022, 2023],        # Photosynthetic activity
        'vh': [2021, 2022],                 # Forest structure (SAR)
        'elevation': True,                   # Topographic context
        'slope': True
    }
}
```

### Zone and Datacube Configuration
```python
# Default configuration (can be customized)
DATACUBE_CONFIG = {
    'mfl_cmf_2024': '/fs2/ftype_mflf/shared_data/datacube/2024/1.2.0/mfl_cmf_2024',
    'zone_1': '/fs2/ftype_mflf/shared_data/datacube/2025/1.2.0/zone_1',
    # ... additional zones
}

BBOX_AREAS_CONFIG = {
    'mfl_cmf_2024': '/fs3/datacenter/ftype_mflf/shared_data/boundary/mfl_cmf_2024.parquet',
    'zone_1': '/fs3/datacenter/ftype_mflf/shared_data/boundary/mfl_cmf_project_2025_phase2/split/zone_1.parquet',
    # ... additional boundaries
}
```

## Output Structure and Usage

### Generated Files
```
output_directory/
├── complete_features_dataset.pkl      # Raw extracted features (selected subset)
├── complete_features_dataset.csv      # Human-readable version
├── ml_dataset_ftype_mix.pkl          # ML-ready dataset (mixed forest)
├── ml_dataset_ftype_nonmix.pkl       # ML-ready dataset (non-mixed forest)
├── selection_metadata.json           # Record of selected years/zones/features
├── feature_statistics.csv           # Data quality metrics for selected data
├── dataset_summary_ftype_mix.txt     # Dataset summary (mixed)
├── dataset_summary_ftype_nonmix.txt  # Dataset summary (non-mixed)
├── features_batch_*.pkl              # Intermediate batch results
└── pipeline_YYYYMMDD_HHMMSS.log      # Execution log
```

### Loading Processed Datasets
```python
import pandas as pd
import pickle

# Load ML-ready dataset with selection metadata
with open('ml_dataset_ftype_mix.pkl', 'rb') as f:
    dataset = pickle.load(f)

# Access raw time-series data (no scaling applied)
X_train_raw = dataset['train']['X_raw']           # Original values preserved
X_train_timeseries = dataset['train']['X_timeseries']  # Individual time series
y_train = dataset['train']['y']

# Access test data
X_test_raw = dataset['test']['X_raw']
X_test_timeseries = dataset['test']['X_timeseries']
y_test = dataset['test']['y']

# Get selection information
selection_config = dataset['selection_config']
selected_years = selection_config['selected_years']      # Which years for each feature
selected_zones = selection_config['selected_zones']      # Which zones included
selected_features = selection_config['selected_features'] # Which features included

# Get feature names and target classes
feature_names = dataset['train']['feature_names']
target_classes = dataset['train']['target_classes']

print(f"Selected features: {selected_features}")
print(f"Selected zones: {selected_zones}")
print(f"Selected years per feature: {selected_years}")
print(f"Data preserved in original units: {dataset['config']['no_scaling_applied']}")
```

### Working with Raw Time-Series Data
```python
# Access individual time series for specific feature-year combinations
lai_2021_data = X_train_timeseries['lai_2021']  # Shape: (n_samples, 21)
ndvi_2022_data = X_train_timeseries['ndvi_2022']  # Shape: (n_samples, 21)

# Plot example time series
import matplotlib.pyplot as plt

sample_idx = 0
weeks = range(1, 22)  # 21 weeks

plt.figure(figsize=(12, 6))
plt.plot(weeks, lai_2021_data[sample_idx], 'g-', label='LAI 2021', linewidth=2)
plt.plot(weeks, ndvi_2022_data[sample_idx], 'r-', label='NDVI 2022', linewidth=2)
plt.xlabel('Week')
plt.ylabel('Original Values (No Scaling)')
plt.title(f'Sample Time Series - Point {sample_idx}')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
```

### Using Selection for New Data Processing
```python
# Load selection metadata for consistent processing
with open('selection_metadata.json', 'r') as f:
    import json
    selection_metadata = json.load(f)

# Apply same selection to new dataset
def apply_consistent_selection(new_dataset_path, selection_metadata):
    """Apply same selection criteria to new data"""
    new_data = pd.read_pickle(new_dataset_path)
    
    # Filter zones
    filtered_data = new_data[new_data['zone'].isin(selection_metadata['selected_zones'])]
    
    # Select same feature columns
    base_columns = ['point_id', 'utm_x', 'utm_y', 'zone', 'ftype_mix', 'ftype_nonmix']
    feature_columns = []
    
    # Add selected time-series features
    for feature, years in selection_metadata['selected_years'].items():
        for year in years:
            col_name = f"{feature}_{year}"
            if col_name in filtered_data.columns:
                feature_columns.append(col_name)
    
    # Add selected static features
    for feature in selection_metadata['selected_features']:
        if feature in ['elevation', 'slope'] and feature in filtered_data.columns:
            feature_columns.append(feature)
    
    return filtered_data[base_columns + feature_columns]
```

## Performance Monitoring

### Built-in Monitoring
- Progress bars for long-running operations
- Batch processing status updates
- Memory usage monitoring
- Error rate tracking

### Custom Monitoring
```python
import logging
import time
from memory_profiler import profile

# Setup custom logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('detailed_pipeline.log'),
        logging.StreamHandler()
    ]
)

# Performance profiling
@profile
def extract_features_with_profiling():
    # Your feature extraction code
    pass
```

## Troubleshooting Guide

### Common Issues and Solutions

#### 1. Memory Issues
**Problem**: Out of memory during processing
```bash
# Solutions:
# 1. Reduce batch size
--batch-size 50

# 2. Reduce parallel workers
--max-workers 2

# 3. Process zones separately
# Edit datacube_config to include only one zone at a time
```

#### 2. Missing Data Files
**Problem**: Datacube files not found
```python
# Check datacube structure
import os
for zone, path in DATACUBE_CONFIG.items():
    print(f"Zone {zone}: {os.path.exists(path)}")
    if os.path.exists(path):
        print(f"  Features: {os.listdir(path)}")
```

#### 3. Coordinate System Issues
**Problem**: Spatial join fails
```python
# Verify CRS consistency
import geopandas as gpd

# Check label data CRS
label_gdf = gpd.read_file('labels.csv')
print(f"Label CRS: {label_gdf.crs}")

# Check boundary CRS
boundary_gdf = gpd.read_parquet('boundary.parquet')
print(f"Boundary CRS: {boundary_gdf.crs}")
```

#### 4. Feature Extraction Errors
**Problem**: High error rate in feature extraction
```python
# Enable detailed logging
logging.getLogger('feature_extractor').setLevel(logging.DEBUG)

# Check sample extraction manually
from src.feature_extractor import ForestTypeFeatureExtractor
extractor = ForestTypeFeatureExtractor(...)
single_point_result = extractor.extract_features_for_point(sample_point, years)
```

### Performance Optimization Tips

1. **Optimize I/O**:
   - Use SSD storage for datacube directories
   - Enable compression for large datasets
   - Use batch processing to reduce file operations

2. **Memory Management**:
   - Monitor memory usage during processing
   - Use appropriate batch sizes for available RAM
   - Clear intermediate variables when possible

3. **Parallel Processing**:
   - Match worker count to CPU cores
   - Consider I/O vs CPU bound operations
   - Use distributed processing for very large datasets

## Integration with ML Workflows

### Scikit-learn Integration with Raw Data
```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import numpy as np

# Load preprocessed data (original values preserved)
X_train_raw = dataset['train']['X_raw']
y_train = dataset['train']['y']
X_test_raw = dataset['test']['X_raw']
y_test = dataset['test']['y']

# Option 1: Use raw time-series data directly
# Flatten time-series for traditional ML algorithms
X_train_flattened = []
X_test_flattened = []

for sample in X_train_raw:
    # Each sample contains selected feature-year combinations
    flattened_features = []
    for feature_name in dataset['train']['feature_names']:
        if feature_name in ['elevation', 'slope']:
            # Static feature - single value
            flattened_features.append(sample[feature_name])
        else:
            # Time-series feature - 21 values
            flattened_features.extend(sample[feature_name])
    X_train_flattened.append(flattened_features)

# Same for test data
for sample in X_test_raw:
    flattened_features = []
    for feature_name in dataset['test']['feature_names']:
        if feature_name in ['elevation', 'slope']:
            flattened_features.append(sample[feature_name])
        else:
            flattened_features.extend(sample[feature_name])
    X_test_flattened.append(flattened_features)

X_train_array = np.array(X_train_flattened)
X_test_array = np.array(X_test_flattened)

# Train model with original values
rf_model = RandomForestClassifier(n_estimators=200, random_state=42)
rf_model.fit(X_train_array, y_train)

# Evaluate
y_pred = rf_model.predict(X_test_array)
print(classification_report(y_test, y_pred, 
                          target_names=dataset['train']['target_classes']))

print(f"Using original data values from selected features:")
print(f"  Selected zones: {dataset['selection_config']['selected_zones']}")
print(f"  Selected features: {dataset['selection_config']['selected_features']}")
print(f"  Selected years: {dataset['selection_config']['selected_years']}")
```

### Time-Series Specific Analysis
```python
import matplotlib.pyplot as plt
import seaborn as sns

# Analyze time-series patterns by forest type
def analyze_temporal_patterns(dataset, feature_name, year):
    """Analyze temporal patterns for specific feature-year combination"""
    
    X_timeseries = dataset['train']['X_timeseries']
    y_train = dataset['train']['y']
    target_classes = dataset['train']['target_classes']
    
    feature_key = f"{feature_name}_{year}"
    if feature_key not in X_timeseries:
        print(f"Feature {feature_key} not available in selected data")
        return
    
    timeseries_data = X_timeseries[feature_key]  # Shape: (n_samples, 21)
    
    # Plot by forest type
    plt.figure(figsize=(15, 8))
    
    for class_idx, class_name in enumerate(target_classes):
        class_mask = (y_train == class_idx)
        class_data = timeseries_data[class_mask]
        
        # Calculate mean and std for this class
        mean_ts = np.nanmean(class_data, axis=0)
        std_ts = np.nanstd(class_data, axis=0)
        
        weeks = range(1, 22)
        plt.plot(weeks, mean_ts, label=f'{class_name} (n={np.sum(class_mask)})', linewidth=2)
        plt.fill_between(weeks, mean_ts - std_ts, mean_ts + std_ts, alpha=0.2)
    
    plt.xlabel('Week')
    plt.ylabel(f'{feature_name.upper()} {year} (Original Values)')
    plt.title(f'Temporal Patterns: {feature_name.upper()} {year} by Forest Type')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
    
    return timeseries_data

# Example usage
if 'lai_2021' in dataset['train']['X_timeseries']:
    lai_data = analyze_temporal_patterns(dataset, 'lai', 2021)
```

### Working with Selected Zones
```python
def analyze_spatial_distribution(dataset):
    """Analyze distribution of data across selected zones"""
    
    # Load the complete dataset to see zone distribution
    complete_features = pd.read_pickle('complete_features_dataset.pkl')
    
    selected_zones = dataset['selection_config']['selected_zones']
    
    print("Zone Distribution in Selected Data:")
    zone_counts = complete_features[complete_features['zone'].isin(selected_zones)]['zone'].value_counts()
    print(zone_counts)
    
    # Plot zone distribution by forest type
    zone_target_crosstab = pd.crosstab(
        complete_features[complete_features['zone'].isin(selected_zones)]['zone'],
        complete_features[complete_features['zone'].isin(selected_zones)]['ftype_mix']
    )
    
    plt.figure(figsize=(12, 6))
    zone_target_crosstab.plot(kind='bar', stacked=True)
    plt.title('Forest Type Distribution Across Selected Zones')
    plt.xlabel('Zone')
    plt.ylabel('Number of Points')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
    
    return zone_counts, zone_target_crosstab

# Analyze selected zones
zone_stats = analyze_spatial_distribution(dataset)
```

## Next Steps and Recommendations

### Immediate Actions
1. **Test the pipeline** with a small subset of data
2. **Validate outputs** against expected data ranges
3. **Benchmark performance** on your hardware
4. **Review feature statistics** for data quality

### Future Enhancements
1. **Add new features** as they become available
2. **Implement advanced preprocessing** (e.g., PCA, feature selection)
3. **Develop specialized query APIs** for common use cases
4. **Add support for streaming data** processing

### Maintenance
1. **Monitor data quality** over time
2. **Update preprocessing** as data distributions change
3. **Version control datasets** for reproducibility
4. **Document any customizations** for team knowledge
