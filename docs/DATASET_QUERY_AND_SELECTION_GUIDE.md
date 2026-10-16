# Forest Type Dataset Query and Selection Guide

## Overview

This guide provides comprehensive instructions for querying and selecting data from the comprehensive forest type dataset v1.2.0 for machine learning training. The dataset contains over 2.1 million records with 10 satellite indices across multiple zones and years.

## Dataset Structure

### File Locations
```
/fs2/ftype_mflf/shared_data/dataset/1.2.0/
├── raw/                     # Raw format datasets
│   └── forest_type_dataset_raw_20250903_104423.parquet
├── processed/               # Wide format datasets  
│   └── forest_type_dataset_wide_20250903_104423.parquet
└── metadata/               # Extraction metadata
    └── extraction_metadata_20250903_104423.json
```

### Dataset Statistics
- **Total records**: 2,109,906
- **Unique points**: 1,807 ground truth locations
- **Active zones**: 3 zones (mfl_cmf_2024, mfl_cmf_area_project_2025_chr_zone_1, mfl_cmf_area_project_2025_chr_zone_2)
- **Features**: 10 comprehensive satellite indices
- **Time range**: 2019-2025 (7 years)

## Available Features

### Time Series Features (8 indices)
| Feature | Description | Records | Years Available |
|---------|-------------|---------|-----------------|
| `lai` | Leaf Area Index | 263,235 | 2019-2024, 2025* |
| `fapar` | Fraction of Absorbed PAR | 263,235 | 2019-2024, 2025* |
| `fcover` | Fraction Cover | 263,235 | 2019-2024, 2025* |
| `ndvi` | Normalized Difference Vegetation Index | 263,235 | 2019-2024, 2025* |
| `gvmi` | Global Vegetation Moisture Index | 263,235 | 2019-2024, 2025* |
| `slavi` | Specific Leaf Area Vegetation Index | 263,235 | 2019-2024, 2025* |
| `vv` | Sentinel-1 VV Polarization | 263,235 | 2019-2024, 2025* |
| `vh` | Sentinel-1 VH Polarization | 263,235 | 2019-2024, 2025* |

*Note: 2025 data availability varies by zone

### Static Features (2 indices)
| Feature | Description | Records |
|---------|-------------|---------|
| `elevation` | SRTM Elevation | 2,013 |
| `slope` | SRTM Slope | 2,013 |

## Data Loading and Basic Queries

### Python Setup
```python
import pandas as pd
import numpy as np

# Load the comprehensive dataset
df = pd.read_parquet('/fs2/ftype_mflf/shared_data/dataset/1.2.0/raw/forest_type_dataset_raw_20250903_104423.parquet')
```

### Basic Dataset Exploration
```python
# Dataset overview
print(f"Dataset shape: {df.shape}")
print(f"Unique points: {df.point_id.nunique()}")
print(f"Available zones: {df.zone.unique()}")
print(f"Available features: {sorted(df.feature.unique())}")
print(f"Year range: {df.year.min():.0f}-{df.year.max():.0f}")

# Check data completeness by zone
zone_summary = df.groupby(['zone', 'feature']).size().unstack(fill_value=0)
print(zone_summary)
```

## Query Strategies

### 1. Zone-Based Selection

#### Select Single Zone
```python
# Focus on largest zone (mfl_cmf_2024)
zone_data = df[df.zone == 'mfl_cmf_2024']
print(f"Zone mfl_cmf_2024: {zone_data.point_id.nunique()} points")
```

#### Select Multiple Zones
```python
# Select specific zones for training
selected_zones = ['mfl_cmf_2024', 'mfl_cmf_area_project_2025_chr_zone_1']
multi_zone_data = df[df.zone.isin(selected_zones)]
print(f"Selected zones: {multi_zone_data.point_id.nunique()} points")
```

### 2. Feature-Based Selection

#### Select Specific Features
```python
# Biophysical indices only
bio_features = ['lai', 'fapar', 'fcover']
bio_data = df[df.feature.isin(bio_features)]

# Vegetation indices only  
veg_features = ['ndvi', 'gvmi', 'slavi']
veg_data = df[df.feature.isin(veg_features)]

# SAR data only
sar_features = ['vv', 'vh']
sar_data = df[df.feature.isin(sar_features)]

# All time series (exclude static)
time_series_features = ['lai', 'fapar', 'fcover', 'ndvi', 'gvmi', 'slavi', 'vv', 'vh']
ts_data = df[df.feature.isin(time_series_features)]
```

#### Include Static Features
```python
# Add static features to any selection
static_features = ['elevation', 'slope']
static_data = df[df.feature.isin(static_features)]

# Combined time series + static
all_features = time_series_features + static_features
complete_data = df[df.feature.isin(all_features)]
```

### 3. Time-Based Selection

#### Select Specific Years
```python
# Focus on recent years (2020-2023)
recent_years = [2020, 2021, 2022, 2023]
recent_data = df[df.year.isin(recent_years) | df.year.isna()]  # Include static (NaN years)

# Pre-COVID data only
pre_covid = df[df.year.isin([2019, 2020]) | df.year.isna()]

# Post-COVID recovery
post_covid = df[df.year.isin([2021, 2022, 2023, 2024]) | df.year.isna()]
```

#### Handle Missing Years
```python
# Check data availability by year and zone
year_availability = df.groupby(['zone', 'year', 'feature']).size().unstack(fill_value=0)
print("Data availability by year:")
print(year_availability.sum(axis=1))
```

### 4. Quality-Based Selection

#### Remove Incomplete Points
```python
# Points with all required features
required_features = ['lai', 'ndvi', 'elevation', 'slope']
complete_points = df.groupby('point_id')['feature'].apply(
    lambda x: all(feat in x.values for feat in required_features)
)
high_quality_points = complete_points[complete_points].index
quality_data = df[df.point_id.isin(high_quality_points)]
```

#### Filter by Data Quality
```python
# Remove extreme outliers (example for NDVI)
ndvi_data = df[df.feature == 'ndvi']
q1, q3 = ndvi_data['value'].quantile([0.01, 0.99])
clean_points = ndvi_data[(ndvi_data['value'] >= q1) & (ndvi_data['value'] <= q3)]['point_id']
clean_data = df[df.point_id.isin(clean_points)]
```

## Training Dataset Preparation

### 1. Wide Format Conversion
```python
def create_wide_format(df_subset):
    """Convert long format to wide format for ML training"""
    
    # Separate time series and static features
    ts_data = df_subset[df_subset.feature.isin(time_series_features)]
    static_data = df_subset[df_subset.feature.isin(['elevation', 'slope'])]
    
    # Pivot time series data
    ts_wide = ts_data.pivot_table(
        index=['point_id', 'utm_x', 'utm_y', 'ftype_mix', 'ftype_nonmix'],
        columns=['feature', 'week', 'year'],
        values='value',
        aggfunc='first'
    ).reset_index()
    
    # Pivot static data
    static_wide = static_data.pivot_table(
        index='point_id',
        columns='feature',
        values='value',
        aggfunc='first'
    ).reset_index()
    
    # Merge time series and static
    final_wide = ts_wide.merge(static_wide, on='point_id', how='left')
    
    return final_wide

# Example usage
training_data = create_wide_format(complete_data)
```

### 2. Feature Engineering Examples
```python
# Calculate seasonal statistics
def add_seasonal_features(df_wide):
    """Add seasonal aggregations to wide format data"""
    
    # Example for LAI
    lai_cols = [col for col in df_wide.columns if 'lai' in str(col)]
    df_wide['lai_mean'] = df_wide[lai_cols].mean(axis=1)
    df_wide['lai_std'] = df_wide[lai_cols].std(axis=1)
    df_wide['lai_max'] = df_wide[lai_cols].max(axis=1)
    df_wide['lai_min'] = df_wide[lai_cols].min(axis=1)
    
    return df_wide

# Apply feature engineering
enhanced_training_data = add_seasonal_features(training_data)
```

### 3. Train-Test Splitting Strategies

#### Random Split
```python
from sklearn.model_selection import train_test_split

X = enhanced_training_data.drop(['point_id', 'utm_x', 'utm_y', 'ftype_mix', 'ftype_nonmix'], axis=1)
y = enhanced_training_data['ftype_mix']  # or 'ftype_nonmix'

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
```

#### Spatial Split
```python
# Split by zone to avoid spatial autocorrelation
train_zones = ['mfl_cmf_2024']
test_zones = ['mfl_cmf_area_project_2025_chr_zone_1']

# Get zone info for each point
point_zones = df.groupby('point_id')['zone'].first()
train_points = point_zones[point_zones.isin(train_zones)].index
test_points = point_zones[point_zones.isin(test_zones)].index

train_data = enhanced_training_data[enhanced_training_data.point_id.isin(train_points)]
test_data = enhanced_training_data[enhanced_training_data.point_id.isin(test_points)]
```

#### Temporal Split
```python
# Train on early years, test on recent years
early_years = [2019, 2020, 2021]
recent_years = [2022, 2023, 2024]

# Filter original data by years
train_temporal = df[df.year.isin(early_years) | df.year.isna()]
test_temporal = df[df.year.isin(recent_years) | df.year.isna()]

# Convert to wide format
train_wide = create_wide_format(train_temporal)
test_wide = create_wide_format(test_temporal)
```

## Common Training Configurations

### Configuration 1: Comprehensive Multi-Index
```python
# All features, all zones, recent years
config_1 = {
    'zones': ['mfl_cmf_2024', 'mfl_cmf_area_project_2025_chr_zone_1', 'mfl_cmf_area_project_2025_chr_zone_2'],
    'features': time_series_features + static_features,
    'years': [2020, 2021, 2022, 2023],
    'target': 'ftype_mix'
}

data_1 = df[
    (df.zone.isin(config_1['zones'])) &
    (df.feature.isin(config_1['features'])) &
    (df.year.isin(config_1['years']) | df.year.isna())
]
```

### Configuration 2: Biophysical Focus
```python
# Biophysical indices only, largest zone
config_2 = {
    'zones': ['mfl_cmf_2024'],
    'features': ['lai', 'fapar', 'fcover', 'elevation', 'slope'],
    'years': [2019, 2020, 2021, 2022, 2023, 2024],
    'target': 'ftype_nonmix'
}

data_2 = df[
    (df.zone.isin(config_2['zones'])) &
    (df.feature.isin(config_2['features'])) &
    (df.year.isin(config_2['years']) | df.year.isna())
]
```

### Configuration 3: SAR-Optical Fusion
```python
# Combined SAR and optical indices
config_3 = {
    'zones': ['mfl_cmf_2024', 'mfl_cmf_area_project_2025_chr_zone_1'],
    'features': ['ndvi', 'gvmi', 'vv', 'vh', 'elevation', 'slope'],
    'years': [2020, 2021, 2022, 2023],
    'target': 'ftype_mix'
}

data_3 = df[
    (df.zone.isin(config_3['zones'])) &
    (df.feature.isin(config_3['features'])) &
    (df.year.isin(config_3['years']) | df.year.isna())
]
```

## Performance Optimization

### Memory Management
```python
# Load data in chunks for large datasets
def load_in_chunks(file_path, chunk_size=50000):
    """Load large parquet files in chunks"""
    return pd.read_parquet(file_path, chunksize=chunk_size)

# Use specific columns only
def load_subset(file_path, zones=None, features=None, years=None):
    """Load only required subset"""
    df = pd.read_parquet(file_path)
    
    if zones:
        df = df[df.zone.isin(zones)]
    if features:
        df = df[df.feature.isin(features)]
    if years:
        df = df[df.year.isin(years) | df.year.isna()]
    
    return df
```

### Caching Strategy
```python
import pickle

def cache_processed_data(data, cache_path):
    """Cache processed data for reuse"""
    with open(cache_path, 'wb') as f:
        pickle.dump(data, f)

def load_cached_data(cache_path):
    """Load cached processed data"""
    with open(cache_path, 'rb') as f:
        return pickle.load(f)

# Example usage
cache_path = '/fs1/tor_project/model_foresttype/data/processed_cache.pkl'
cache_processed_data(enhanced_training_data, cache_path)
```

## Validation and Quality Checks

### Data Completeness Check
```python
def validate_dataset(df_subset):
    """Validate dataset quality"""
    
    print("=== Dataset Validation Report ===")
    print(f"Total records: {len(df_subset):,}")
    print(f"Unique points: {df_subset.point_id.nunique():,}")
    print(f"Features: {sorted(df_subset.feature.unique())}")
    print(f"Zones: {sorted(df_subset.zone.unique())}")
    
    # Check for missing values
    missing_by_feature = df_subset.groupby('feature')['value'].apply(lambda x: x.isna().sum())
    print(f"\nMissing values by feature:")
    print(missing_by_feature)
    
    # Check temporal coverage
    temporal_coverage = df_subset.groupby(['zone', 'feature'])['year'].apply(
        lambda x: x.dropna().nunique()
    ).unstack(fill_value=0)
    print(f"\nTemporal coverage (years per feature by zone):")
    print(temporal_coverage)
    
    return True

# Validate your selection
validate_dataset(data_1)
```

## Integration with Training Notebooks

### Export for Model Training
```python
def export_for_training(df_subset, output_path, format='parquet'):
    """Export processed dataset for model training"""
    
    if format == 'parquet':
        df_subset.to_parquet(output_path, index=False)
    elif format == 'csv':
        df_subset.to_csv(output_path, index=False)
    elif format == 'hdf5':
        df_subset.to_hdf(output_path, key='training_data', mode='w')
    
    print(f"Dataset exported to: {output_path}")
    print(f"Shape: {df_subset.shape}")

# Export comprehensive dataset
export_for_training(
    enhanced_training_data, 
    '/fs1/tor_project/model_foresttype/data/training_dataset_v1_2_0.parquet'
)
```

### Integration Code Template
```python
# Template for training notebook integration
def load_training_data(config):
    """Load and prepare training data based on configuration"""
    
    # Load raw data
    df = pd.read_parquet('/fs2/ftype_mflf/shared_data/dataset/1.2.0/raw/forest_type_dataset_raw_20250903_104423.parquet')
    
    # Apply filters
    filtered_data = df[
        (df.zone.isin(config['zones'])) &
        (df.feature.isin(config['features'])) &
        (df.year.isin(config['years']) | df.year.isna())
    ]
    
    # Convert to wide format
    wide_data = create_wide_format(filtered_data)
    
    # Feature engineering
    enhanced_data = add_seasonal_features(wide_data)
    
    # Prepare X and y
    feature_cols = [col for col in enhanced_data.columns 
                   if col not in ['point_id', 'utm_x', 'utm_y', 'ftype_mix', 'ftype_nonmix']]
    
    X = enhanced_data[feature_cols]
    y = enhanced_data[config['target']]
    
    return X, y, enhanced_data

# Example usage in training notebook
config = {
    'zones': ['mfl_cmf_2024'],
    'features': ['lai', 'ndvi', 'elevation', 'slope'],
    'years': [2020, 2021, 2022, 2023],
    'target': 'ftype_mix'
}

X, y, full_data = load_training_data(config)
```

## Best Practices

1. **Always validate your selection** - Use the validation functions to ensure data quality
2. **Consider spatial autocorrelation** - Use spatial splits when possible
3. **Handle missing data appropriately** - Different strategies for time series vs static features
4. **Cache processed datasets** - Avoid reprocessing large datasets repeatedly
5. **Document your selections** - Keep track of configurations used for reproducibility
6. **Monitor memory usage** - Use chunking for very large selections
7. **Validate temporal consistency** - Ensure consistent temporal coverage across features

## Troubleshooting

### Common Issues and Solutions

1. **Memory errors**: Use chunking or reduce dataset size
2. **Missing features**: Check zone-specific feature availability
3. **Inconsistent years**: Some zones have limited temporal coverage
4. **Large file sizes**: Use parquet format and compression
5. **Slow queries**: Index by commonly used columns

---

**Last Updated**: September 3, 2025  
**Dataset Version**: 1.2.0  
**Contact**: Forest Type Modeling Team
