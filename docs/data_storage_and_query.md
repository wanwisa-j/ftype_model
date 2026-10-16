# Data Storage Architecture and Query Methods

## Storage Strategy Overview

The forest type classification dataset employs a multi-tiered storage architecture designed for efficient data access, scalability, and ML model integration. The strategy balances raw data preservation, processing efficiency, and query performance.

## Storage Tiers

### Tier 1: Raw Datacube Storage
**Location**: `/fs2/ftype_mflf/shared_data/datacube/`

```
datacube/
├── 2024/1.2.0/mfl_cmf_2024/
│   ├── lai/
│   │   ├── cube_biophy_lai_2019_W4before_to_W16after_21bands.tif
│   │   ├── cube_biophy_lai_2020_W4before_to_W16after_21bands.tif
│   │   └── ...
│   ├── elevation/
│   │   └── elevation_data.tif
│   └── ...
├── 2025/1.2.0/zone_1/
└── ...
```

**Characteristics**:
- Immutable source data
- Zone-based organization
- Feature-specific subdirectories
- Year-based file naming
- High storage capacity requirements

### Tier 2: Processed Feature Cache
**Location**: `output_directory/processed_features/`

```
processed_features/
├── zone_features/
│   ├── mfl_cmf_2024_features.pkl
│   ├── zone_1_features.pkl
│   └── ...
├── point_features/
│   ├── features_batch_1.pkl
│   ├── features_batch_2.pkl
│   └── ...
└── complete_features_dataset.pkl
```

**Characteristics**:
- Intermediate processing results
- Point-based feature storage
- Batch organization for memory management
- Pickle format for Python optimization
- Enables pipeline restart/recovery

### Tier 3: ML-Ready Datasets
**Location**: `output_directory/ml_datasets/`

```
ml_datasets/
├── ftype_mix/
│   ├── tabular_features/
│   │   ├── train_X.npy
│   │   ├── train_y.npy
│   │   ├── test_X.npy
│   │   └── test_y.npy
│   ├── sequential_features/
│   │   ├── train_X_seq.npy
│   │   ├── test_X_seq.npy
│   │   └── metadata.json
│   └── ml_dataset_ftype_mix.pkl
├── ftype_nonmix/
└── preprocessors.pkl
```

**Characteristics**:
- Model-ready format
- Separate tabular and sequential data
- Train/test splits maintained
- Preprocessing artifacts preserved
- Multiple target support

## Data Formats and Encoding

### Raw Features Storage
```python
# Structure of complete_features_dataset.pkl
{
    'point_id': int,           # Unique point identifier
    'utm_x': float,            # UTM X coordinate
    'utm_y': float,            # UTM Y coordinate
    'zone': str,               # Processing zone
    'year': int,               # Label year
    'ftype_mix': str,          # Target label (mix)
    'ftype_nonmix': str,       # Target label (nonmix)
    
    # Time-series features (21 timesteps each) - only selected years
    'lai_2019': np.ndarray,    # Shape: (21,) - only if 2019 selected for LAI
    'lai_2020': np.ndarray,    # Shape: (21,) - only if 2020 selected for LAI
    'ndvi_2022': np.ndarray,   # Shape: (21,) - only if 2022 selected for NDVI
    ...                        # Only selected feature-year combinations
    
    # Static features - only if selected
    'elevation': float,        # Only if elevation selected
    'slope': float,           # Only if slope selected
    
    # Selection metadata
    'selected_years': dict,    # {'lai': [2019, 2020], 'ndvi': [2022, 2023], ...}
    'selected_zones': list,    # ['zone_1', 'zone_3', ...]
    'selected_features': list  # ['lai', 'ndvi', 'elevation', ...]
}
```

### ML Dataset Storage
```python
# Structure of ml_dataset_{target}.pkl
{
    'train': {
        'X_raw': np.ndarray,              # Shape: (n_samples, selected_features)
        'X_timeseries': dict,             # {'feature_year': np.ndarray(n_samples, 21)}
        'y': np.ndarray,                  # Shape: (n_samples,)
        'feature_names': List[str],       # Names of selected features
        'target_classes': List[str],
        'target_col': str
    },
    'test': {
        # Same structure as train
    },
    'selection_config': {
        'selected_years': dict,           # Which years for each feature
        'selected_zones': list,           # Which zones included
        'selected_features': list,        # Which features included
        'selection_criteria': dict       # Original selection parameters
    },
    'config': {
        'target_col': str,
        'n_train_samples': int,
        'n_test_samples': int,
        'no_scaling_applied': True,       # Indicates original values preserved
        'pre_interpolated': True         # Indicates data was pre-processed
    }
}
```

## Query Methods and Data Access Patterns

### 1. Feature Selection Queries

#### Year-Feature Selection
```python
def query_by_feature_years(feature_selection_config):
    """Extract specific feature-year combinations"""
    # feature_selection_config = {
    #     'lai': [2020, 2021, 2022],
    #     'ndvi': [2019, 2023, 2024],
    #     'vh': [2021, 2022],
    #     'elevation': True,  # Static feature
    #     'slope': False      # Exclude this static feature
    # }
    
    selected_columns = []
    
    # Process time-series features
    for feature, years in feature_selection_config.items():
        if feature in ['elevation', 'slope']:
            if years:  # If True, include static feature
                selected_columns.append(feature)
        else:
            # Time-series feature
            for year in years:
                col_name = f"{feature}_{year}"
                if col_name in dataset.columns:
                    selected_columns.append(col_name)
    
    return dataset[selected_columns]
```

#### Zone-Based Filtering with Selection
```python
def query_by_selected_zones(zone_list, feature_config=None):
    """Get data from specific zones with feature selection"""
    # Filter by zones first
    zone_filtered = dataset[dataset['zone'].isin(zone_list)]
    
    # Apply feature selection if provided
    if feature_config:
        return query_by_feature_years_on_data(zone_filtered, feature_config)
    
    return zone_filtered
```

### 2. Raw Data Access (No Scaling)

#### Direct Time-Series Access
```python
def get_raw_timeseries(point_ids, feature_year_combinations):
    """Get raw time-series data without any preprocessing"""
    # feature_year_combinations = [('lai', 2020), ('ndvi', 2021), ...]
    
    result = {}
    for feature, year in feature_year_combinations:
        col_name = f"{feature}_{year}"
        if col_name in dataset.columns:
            # Return original values as stored
            result[col_name] = dataset.loc[dataset['point_id'].isin(point_ids), col_name].values
    
    return result
```

#### Static Feature Access
```python
def get_static_features(point_ids, features=['elevation', 'slope']):
    """Get static features in original units"""
    available_features = [f for f in features if f in dataset.columns]
    return dataset.loc[dataset['point_id'].isin(point_ids), available_features]
```

### 2. Spatial Queries

#### Zone-Based Filtering
```python
def query_by_zone(zone_name, feature_type=None):
    """Get all points within a specific zone"""
    filtered_data = dataset[dataset['zone'] == zone_name]
    return filtered_data
```

#### Spatial Intersection
```python
def query_by_geometry(polygon_geometry):
    """Find points within custom polygon"""
    # Using geopandas spatial operations
    return intersected_points
```

### 3. Temporal Queries with Selection

#### Flexible Year Selection
```python
def query_by_custom_years(feature_year_mapping):
    """Filter features by custom year combinations"""
    # feature_year_mapping = {
    #     'lai': [2019, 2021, 2023],
    #     'ndvi': [2020, 2022, 2024],
    #     'vh': [2021, 2022]
    # }
    
    selected_columns = ['point_id', 'utm_x', 'utm_y', 'zone', 'ftype_mix', 'ftype_nonmix']
    
    for feature, years in feature_year_mapping.items():
        for year in years:
            col_name = f"{feature}_{year}"
            if col_name in dataset.columns:
                selected_columns.append(col_name)
    
    return dataset[selected_columns]
```

#### Seasonal Window Extraction
```python
def query_seasonal_windows(feature_year_combinations, week_ranges):
    """Extract specific seasonal windows from time series"""
    # week_ranges = [(1, 8), (10, 16)]  # Early and late season
    result = {}
    
    for feature, year in feature_year_combinations:
        col_name = f"{feature}_{year}"
        if col_name in dataset.columns:
            for start_week, end_week in week_ranges:
                seasonal_data = dataset[col_name].apply(
                    lambda ts: ts[start_week-1:end_week] if len(ts) >= end_week else np.full(end_week-start_week+1, np.nan)
                )
                result[f"{feature}_{year}_weeks_{start_week}_{end_week}"] = seasonal_data
    
    return result
```

### 4. Selection-Based Queries

#### Configuration-Driven Access
```python
def query_by_selection_config(selection_config):
    """Query data based on comprehensive selection configuration"""
    # selection_config = {
    #     'zones': ['zone_1', 'zone_3', 'mfl_cmf_2024'],
    #     'years': {
    #         'lai': [2020, 2021, 2022],
    #         'ndvi': [2019, 2021, 2023],
    #         'vh': [2021, 2022]
    #     },
    #     'static_features': ['elevation'],  # Exclude slope
    #     'exclude_missing': True
    # }
    
    # Filter by zones
    data = dataset[dataset['zone'].isin(selection_config['zones'])]
    
    # Select feature columns
    feature_columns = ['point_id', 'utm_x', 'utm_y', 'zone', 'ftype_mix', 'ftype_nonmix']
    
    # Add time-series features
    for feature, years in selection_config['years'].items():
        for year in years:
            col_name = f"{feature}_{year}"
            if col_name in data.columns:
                feature_columns.append(col_name)
    
    # Add static features
    for static_feature in selection_config.get('static_features', []):
        if static_feature in data.columns:
            feature_columns.append(static_feature)
    
    result = data[feature_columns]
    
    # Optionally exclude rows with missing data
    if selection_config.get('exclude_missing', False):
        result = result.dropna()
    
    return result
```

## Data Access APIs

### High-Level Dataset API
```python
class ForestTypeDataset:
    def __init__(self, dataset_path):
        self.data = pd.read_pickle(dataset_path)
        self.selection_metadata = self._load_selection_metadata()
        
    def query_points(self, point_ids):
        """Query by point IDs"""
        return self.data[self.data['point_id'].isin(point_ids)]
        
    def query_spatial(self, zones=None, coords_bounds=None):
        """Spatial queries with zone selection"""
        if zones:
            return self.data[self.data['zone'].isin(zones)]
        # Additional spatial logic
        
    def query_by_selection(self, selection_config):
        """Query with flexible feature/year/zone selection"""
        return query_by_selection_config(selection_config)
        
    def get_raw_timeseries(self, feature_year_combinations):
        """Get raw time-series data without preprocessing"""
        return get_raw_timeseries(self.data, feature_year_combinations)
        
    def get_available_combinations(self):
        """Get all available feature-year combinations in dataset"""
        timeseries_cols = [col for col in self.data.columns if '_20' in col]
        combinations = []
        for col in timeseries_cols:
            parts = col.split('_')
            if len(parts) >= 2:
                feature = '_'.join(parts[:-1])
                year = parts[-1]
                combinations.append((feature, int(year)))
        return combinations
```

### Selection-Aware Data Access
```python
class SelectiveFeatureLoader:
    def __init__(self, base_dataset_path):
        self.base_data = pd.read_pickle(base_dataset_path)
        
    def create_custom_dataset(self, selection_config, output_path=None):
        """Create custom dataset based on selection criteria"""
        # selection_config = {
        #     'zones': ['zone_1', 'zone_2'],
        #     'features': {
        #         'lai': [2020, 2021, 2022],
        #         'ndvi': [2019, 2020, 2021],
        #         'elevation': True
        #     },
        #     'targets': ['ftype_mix'],  # or ['ftype_nonmix'] or both
        #     'min_data_quality': 0.8   # Minimum data completeness
        # }
        
        filtered_data = self._apply_selection(selection_config)
        
        if output_path:
            # Save with selection metadata
            dataset_info = {
                'data': filtered_data,
                'selection_config': selection_config,
                'creation_date': datetime.now(),
                'source_dataset': self.base_data_path,
                'data_statistics': self._calculate_statistics(filtered_data)
            }
            pd.to_pickle(dataset_info, output_path)
            
        return filtered_data
        
    def _apply_selection(self, config):
        """Apply selection criteria to base dataset"""
        # Zone filtering
        data = self.base_data[self.base_data['zone'].isin(config['zones'])]
        
        # Feature selection
        selected_cols = ['point_id', 'utm_x', 'utm_y', 'zone']
        selected_cols.extend(config.get('targets', ['ftype_mix', 'ftype_nonmix']))
        
        for feature, years_or_flag in config['features'].items():
            if feature in ['elevation', 'slope']:
                if years_or_flag:  # True means include
                    selected_cols.append(feature)
            else:
                # Time-series feature
                for year in years_or_flag:
                    col_name = f"{feature}_{year}"
                    if col_name in data.columns:
                        selected_cols.append(col_name)
        
        return data[selected_cols]
```

### Streaming Data Access
```python
class FeatureStreamer:
    def __init__(self, batch_size=1000):
        self.batch_size = batch_size
        
    def stream_batches(self, query_params):
        """Stream data in batches for large datasets"""
        for batch in self.get_batches(query_params):
            yield batch
            
    def lazy_load_features(self, point_list):
        """Lazy loading for memory efficiency"""
        # Generator-based loading
```

## Performance Optimization

### Indexing Strategy
```python
# Spatial indexing for fast coordinate queries
spatial_index = {
    'zone': BTreeIndex,           # Zone-based partitioning
    'coordinates': RTreeIndex,    # Spatial R-tree for coordinates
    'temporal': HashIndex         # Year-based hash index
}
```

### Caching Strategy
```python
# Multi-level caching
cache_levels = {
    'memory': LRUCache(maxsize=1000),     # In-memory feature cache
    'disk': DiskCache('/tmp/features'),   # Disk-based cache
    'batch': BatchCache()                 # Batch result cache
}
```

### Query Optimization
1. **Spatial Partitioning**: Zone-based data organization
2. **Temporal Partitioning**: Year-based file structure
3. **Feature Grouping**: Co-locate related features
4. **Batch Processing**: Minimize I/O operations
5. **Parallel Queries**: Multi-threaded access patterns

## Data Consistency and Integrity

### Version Control
- Dataset versioning with semantic versioning
- Provenance tracking for feature extraction
- Configuration preservation for reproducibility

### Data Validation
```python
def validate_dataset_integrity():
    """Comprehensive data validation"""
    checks = [
        validate_coordinate_bounds(),
        validate_feature_completeness(),
        validate_temporal_consistency(),
        validate_target_distribution(),
        validate_spatial_coverage()
    ]
    return all(checks)
```

### Backup and Recovery
- Incremental backups of processed features
- Checkpoint-based pipeline recovery
- Redundant storage for critical datasets

## Query Performance Benchmarks

### Expected Performance Metrics
- **Point Query**: <1ms per point (cached)
- **Batch Query** (1000 points): <10 seconds
- **Zone Query**: <100ms for zone filtering
- **Feature Extraction**: 100-1000 points/minute
- **ML Dataset Loading**: <30 seconds for full dataset

### Scalability Targets
- **Dataset Size**: Up to 1M labeled points
- **Feature Dimensions**: 1000+ features per point
- **Temporal Coverage**: 10+ years of data
- **Concurrent Users**: 10+ simultaneous queries
- **Storage Growth**: Linear scaling with data volume
