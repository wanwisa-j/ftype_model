# Dataset Extraction and Model Training Documentation

## Overview

This document describes the process of extracting training datasets from the aligned datacube and preparing them for model training. The extraction process links spatially-aligned satellite features with ground survey data to create labeled training datasets.

## Dataset Extraction Pipeline

### Primary Tool
**Script**: `scripts/extract_forest_type_dataset.py`

### Input Requirements

#### Ground Survey Data
- **Location**: Point coordinates (latitude, longitude)
- **Forest Type Labels**: Categorical classification (target variable)
- **Survey Date**: Temporal reference for feature extraction
- **Quality Flags**: Data quality indicators

#### Datacube Features
- **Spatial Coverage**: Complete AOI coverage for all features
- **Temporal Alignment**: 21-band time series for each feature
- **Feature List**: VV, VH, GVMI, NDVI, SLAVI, LAI, FAPAR, FCOVER, elevation, slope

### Extraction Process

#### Step 1: Spatial Intersection
```python
# Pseudo-code workflow
ground_points = load_survey_data()
for point in ground_points:
    # Extract pixel values at survey location
    pixel_coords = transform_to_raster_coords(point.lat, point.lon)
    
    for feature in features:
        datacube = load_feature_datacube(feature, point.zone)
        time_series = extract_pixel_values(datacube, pixel_coords)
        point.features[feature] = time_series
```

#### Step 2: Temporal Alignment
- **Reference Week**: Align extraction to survey week
- **Temporal Window**: Extract full 21-week time series
- **Quality Check**: Validate temporal completeness

#### Step 3: Feature Vector Creation
Each training sample consists of:
```python
feature_vector = {
    'point_id': unique_identifier,
    'coordinates': (lat, lon),
    'survey_date': date,
    'forest_type': target_label,
    'features': {
        'vv': [21_band_time_series],
        'vh': [21_band_time_series], 
        'gvmi': [21_band_time_series],
        'ndvi': [21_band_time_series],
        'slavi': [21_band_time_series],
        'lai': [21_band_time_series],
        'fapar': [21_band_time_series],
        'fcover': [21_band_time_series],
        'elevation': [single_value],
        'slope': [single_value]
    }
}
```

### Data Management System

#### Primary Tool
**Script**: `src/dataset_builder.py`

#### Dataset Organization
```python
class ForestTypeDataset:
    def __init__(self, config):
        self.feature_datacubes = load_datacubes(config.datacube_path)
        self.ground_truth = load_ground_truth(config.survey_data)
        self.extracted_features = {}
        
    def extract_features(self, point_list):
        """Extract features for list of survey points"""
        
    def create_training_set(self, split_ratio=0.8):
        """Create train/validation split"""
        
    def get_feature_matrix(self):
        """Return scikit-learn compatible feature matrix"""
```

#### Query Interface
```python
# Example usage
dataset = ForestTypeDataset(config)
train_X, train_y = dataset.get_training_data()
val_X, val_y = dataset.get_validation_data()
test_X, test_y = dataset.get_test_data()
```

## Feature Engineering

### Temporal Feature Extraction
From the 21-band time series, multiple features are derived:

#### Statistical Features
```python
temporal_features = {
    'mean': np.mean(time_series),
    'std': np.std(time_series),
    'min': np.min(time_series),
    'max': np.max(time_series),
    'median': np.median(time_series),
    'skewness': scipy.stats.skew(time_series),
    'kurtosis': scipy.stats.kurtosis(time_series)
}
```

#### Phenological Features
```python
phenology_features = {
    'growing_season_start': detect_green_up(time_series),
    'growing_season_end': detect_senescence(time_series),
    'peak_value': np.max(time_series),
    'peak_timing': np.argmax(time_series),
    'seasonal_amplitude': np.max(time_series) - np.min(time_series)
}
```

#### Trend Features
```python
trend_features = {
    'linear_trend': linear_regression_slope(time_series),
    'trend_significance': trend_p_value,
    'seasonal_trend': seasonal_decomposition_trend(time_series)
}
```

### Multi-Feature Integration
```python
integrated_features = {
    'vegetation_indices': [gvmi, ndvi, slavi],
    'biophysical': [lai, fapar, fcover],
    'radar': [vv, vh],
    'topographic': [elevation, slope],
    'derived_indices': calculate_derived_indices()
}
```

## Data Quality Control

### Extraction Quality Checks

#### Temporal Quality  
- **Completeness**: Verify all 21 bands present
- **Interpolation Flags**: Track interpolated vs. observed values
- **Survey Timing**: Validate temporal alignment

#### Feature Quality
- **Value Ranges**: Check for realistic feature values
- **Missing Data**: Handle NoData values appropriately
- **Outlier Detection**: Identify and flag extreme values

### Dataset Validation
```python
def validate_dataset(features, labels):
    checks = {
        'completeness': check_missing_values(features),
        'consistency': check_feature_consistency(features),
        'balance': check_class_balance(labels),
        'quality': assess_data_quality(features)
    }
    return checks
```

## Training Data Characteristics

### Sample Distribution
Typical dataset characteristics:
- **Total Samples**: 5,000-15,000 points per zone
- **Class Distribution**: Variable by forest type
- **Spatial Distribution**: Representative of AOI
- **Temporal Coverage**: Multi-year for robustness

### Feature Dimensions
```python
feature_dimensions = {
    'temporal_features': 21 * 8,  # 8 features × 21 time steps
    'statistical_features': 8 * 7,  # 8 features × 7 statistics
    'phenological_features': 8 * 5,  # 8 features × 5 phenology metrics
    'static_features': 2,  # elevation, slope
    'total': ~300_features
}
```

## Model Training Integration

### Data Preparation
```python
# Prepare data for machine learning
X = dataset.get_feature_matrix()
y = dataset.get_labels()

# Feature scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train/validation split
X_train, X_val, y_train, y_val = train_test_split(
    X_scaled, y, test_size=0.2, stratify=y, random_state=42
)
```

### Feature Selection
```python
# Feature importance analysis
selector = SelectKBest(f_classif, k=100)
X_selected = selector.fit_transform(X_train, y_train)

# Or use model-based selection
rf_selector = RandomForestClassifier(n_estimators=100)
rf_selector.fit(X_train, y_train)
feature_importance = rf_selector.feature_importances_
```

## Storage and Versioning

### Dataset Storage Format
```
/fs1/tor_project/model_foresttype/
├── forest_type_training_data.npz  # Current dataset
├── datasets/
│   ├── v1.0/
│   │   ├── training_features.npy
│   │   ├── training_labels.npy
│   │   ├── validation_features.npy
│   │   ├── validation_labels.npy
│   │   └── metadata.json
│   └── v1.1/
└── extraction_logs/
    ├── extraction_20250903.log
    └── quality_report_20250903.json
```

### Metadata Management
```json
{
    "dataset_version": "1.1",
    "creation_date": "2025-09-03",
    "datacube_version": "1.2.0",
    "zones_included": ["zone_1", "zone_2"],
    "feature_count": 298,
    "sample_count": 12450,
    "class_distribution": {
        "primary_forest": 3200,
        "secondary_forest": 2800,
        "plantation": 2100,
        "mixed_forest": 1950,
        "other": 2400
    },
    "quality_metrics": {
        "completeness": 0.98,
        "interpolation_ratio": 0.15,
        "spatial_accuracy": 0.95
    }
}
```

## Performance Monitoring

### Extraction Performance
- **Processing Time**: Track extraction speed per sample
- **Memory Usage**: Monitor memory requirements
- **I/O Performance**: Datacube access efficiency

### Dataset Quality Metrics
- **Completeness**: Percentage of complete feature vectors
- **Spatial Coverage**: Geographic distribution of samples
- **Temporal Coverage**: Time series completeness
- **Class Balance**: Distribution across forest types

## Error Handling

### Common Issues and Solutions

#### Missing Datacube Files
- **Detection**: File existence validation
- **Solution**: Skip missing features or halt extraction
- **Prevention**: Pre-processing validation

#### Coordinate Projection Issues
- **Detection**: Bounds checking
- **Solution**: Automatic reprojection
- **Prevention**: Standardized coordinate handling

#### Memory Limitations
- **Detection**: Memory usage monitoring
- **Solution**: Batch processing
- **Prevention**: Chunk-based extraction

## Related Documentation
- [Data Processing and Alignment](./DATA_PROCESSING_ALIGNMENT.md)
- [Model Training Guide](./MODEL_TRAINING.md)
- [Prediction Pipeline](./PREDICTION_PIPELINE.md)
