
# Corrected Data Preparation for 21-Timestep Structure

import pandas as pd
import numpy as np

# Load dataset
dataset_path = "/fs2/ftype_mflf/shared_data/dataset/1.2.0/raw/forest_type_dataset_raw_20250903_104423.parquet"
df = pd.read_parquet(dataset_path)

# Define feature groups
biophysical_features = ['lai', 'fapar', 'fcover']
indices_features = ['ndvi', 'gvmi', 'slavi']
temporal_features = biophysical_features + indices_features

# Filter and clean temporal data
temporal_data = df[df['feature'].isin(temporal_features)].copy()
temporal_data = temporal_data.dropna(subset=['value', 'week', 'year'])

# Get unique dimensions
weeks = sorted(temporal_data['week'].unique())  # Should be 1-21
years = sorted(temporal_data['year'].unique())
points = sorted(temporal_data['point_id'].unique())

print(f"Temporal structure: {len(points)} points, {len(years)} years, {len(weeks)} weeks")
print(f"Expected timesteps per point: {len(years) * len(weeks)}")

# Prepare corrected feature arrays
def prepare_feature_group(features, data):
    """Prepare feature group with correct temporal structure."""
    
    # Filter data for this feature group
    group_data = data[data['feature'].isin(features)]
    
    # Pivot to get features as columns
    pivoted = group_data.pivot_table(
        index=['point_id', 'year', 'week'],
        columns='feature',
        values='value',
        aggfunc='mean'
    ).reset_index()
    
    # Create temporal index (combine year + week)
    pivoted['temporal_index'] = (pivoted['year'] - min(years)) * 21 + (pivoted['week'] - 1)
    
    # Initialize 3D array
    n_points = len(points)
    n_timesteps = len(years) * 21
    n_features = len(features)
    
    feature_array = np.full((n_points, n_timesteps, n_features), np.nan, dtype=np.float32)
    
    # Create mappings
    point_map = {pid: idx for idx, pid in enumerate(points)}
    feature_map = {feat: idx for idx, feat in enumerate(features)}
    
    # Fill array
    for _, row in pivoted.iterrows():
        point_idx = point_map[row['point_id']]
        temporal_idx = int(row['temporal_index'])
        
        for feat in features:
            if feat in row and not pd.isna(row[feat]):
                feat_idx = feature_map[feat]
                feature_array[point_idx, temporal_idx, feat_idx] = row[feat]
    
    return feature_array

# Create corrected biophysical array (should be: points, 126_timesteps, 3_features)
X_biophysical = prepare_feature_group(biophysical_features, temporal_data)

# Create corrected indices array (should be: points, 126_timesteps, 6_features)  
X_indices = prepare_feature_group(indices_features, temporal_data)

print(f"Corrected biophysical shape: {X_biophysical.shape}")
print(f"Corrected indices shape: {X_indices.shape}")

# Verify 21-week structure
timesteps_per_year = X_biophysical.shape[1] // len(years)
print(f"Timesteps per year: {timesteps_per_year} (should be 21)")

if timesteps_per_year == 21:
    print("✅ SUCCESS: Achieved correct 21-week temporal structure!")
else:
    print(f"❌ Still have issues: {timesteps_per_year} timesteps per year instead of 21")
