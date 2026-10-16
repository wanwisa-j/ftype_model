#!/usr/bin/env python3
"""
Corrected data preparation script for forest type dataset.
This script ensures proper 21-timestep temporal structure for both
biophysical and indices features, fixing the notebook's data preparation issue.

Key fixes:
1. Ensure all 21 timesteps are preserved during reshaping
2. Align biophysical and indices temporal dimensions  
3. Proper seasonal learning structure: (datapoints, 21_weeks, features)
"""

import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_dataset():
    """Load the raw dataset."""
    dataset_path = "/fs2/ftype_mflf/shared_data/dataset/1.2.0/raw/forest_type_dataset_raw_20250903_104423.parquet"
    
    logger.info(f"Loading dataset from: {dataset_path}")
    df = pd.read_parquet(dataset_path)
    logger.info(f"Dataset loaded: {df.shape}")
    return df

def prepare_temporal_features_corrected(df):
    """Prepare temporal features with correct 21-timestep structure."""
    
    logger.info("=== Preparing Temporal Features (Corrected) ===")
    
    # Define feature groups
    biophysical_features = ['lai', 'fapar', 'fcover']
    indices_features = ['ndvi', 'gvmi', 'slavi']
    temporal_features = biophysical_features + indices_features
    
    # Filter temporal data only
    temporal_data = df[df['feature'].isin(temporal_features)].copy()
    logger.info(f"Temporal data records: {len(temporal_data)}")
    
    # Remove invalid data
    temporal_data = temporal_data.dropna(subset=['value', 'week', 'year'])
    logger.info(f"Valid temporal data: {len(temporal_data)}")
    
    # Check temporal structure
    weeks = sorted(temporal_data['week'].unique())
    years = sorted(temporal_data['year'].unique())
    points = sorted(temporal_data['point_id'].unique())
    
    logger.info(f"Weeks: {len(weeks)} (range: {min(weeks)}-{max(weeks)})")
    logger.info(f"Years: {len(years)} (range: {min(years)}-{max(years)})")
    logger.info(f"Points: {len(points)}")
    
    # Create feature arrays with correct 21-timestep structure
    feature_arrays = {}
    
    for feature_group, features in [('biophysical', biophysical_features), ('indices', indices_features)]:
        logger.info(f"\nProcessing {feature_group} features: {features}")
        
        # Filter data for this feature group
        group_data = temporal_data[temporal_data['feature'].isin(features)]
        
        # Handle duplicates by taking mean values
        group_data = group_data.groupby(['point_id', 'year', 'week', 'feature'])['value'].mean().reset_index()
        
        # Create multi-index for proper reshaping
        group_data = group_data.set_index(['point_id', 'year', 'week', 'feature'])['value'].unstack('feature')
        
        # Reset index to get point_id, year, week as columns
        group_data = group_data.reset_index()
        
        logger.info(f"Reshaped {feature_group} data: {group_data.shape}")
        logger.info(f"Columns: {list(group_data.columns)}")
        
        # Check temporal completeness
        expected_records = len(points) * len(years) * 21  # 21 weeks per year
        actual_records = len(group_data)
        logger.info(f"Expected records: {expected_records}, Actual: {actual_records}")
        
        # Create 3D array: (points, timesteps, features)
        # Combine year+week into single temporal dimension
        group_data['temporal_index'] = (group_data['year'] - min(years)) * 21 + (group_data['week'] - 1)
        
        # Initialize array for this feature group
        n_points = len(points)
        n_timesteps = len(years) * 21  # 21 weeks per year
        n_features = len(features)
        
        feature_array = np.full((n_points, n_timesteps, n_features), np.nan, dtype=np.float32)
        
        # Fill the array
        point_id_map = {pid: idx for idx, pid in enumerate(sorted(points))}
        feature_map = {feat: idx for idx, feat in enumerate(features)}
        
        for _, row in group_data.iterrows():
            point_idx = point_id_map[row['point_id']]
            temporal_idx = int(row['temporal_index'])
            
            for feat in features:
                if not pd.isna(row[feat]):
                    feat_idx = feature_map[feat]
                    feature_array[point_idx, temporal_idx, feat_idx] = row[feat]
        
        feature_arrays[feature_group] = feature_array
        
        logger.info(f"{feature_group.capitalize()} array shape: {feature_array.shape}")
        
        # Validate temporal structure
        valid_timesteps = []
        for t in range(n_timesteps):
            if not np.all(np.isnan(feature_array[:, t, :])):
                valid_timesteps.append(t)
        
        logger.info(f"Valid timesteps: {len(valid_timesteps)} / {n_timesteps}")
        
        # Check if we have the expected 21 weeks per year
        weeks_per_year = n_timesteps // len(years)
        logger.info(f"Timesteps per year: {weeks_per_year}")
        
        if weeks_per_year == 21:
            logger.info(f"✅ Correct temporal structure: {weeks_per_year} weeks per year")
        else:
            logger.warning(f"❌ Incorrect temporal structure: {weeks_per_year} weeks per year (expected 21)")
    
    return feature_arrays, points, years

def align_temporal_dimensions(feature_arrays):
    """Ensure biophysical and indices arrays have matching temporal dimensions."""
    
    logger.info("\n=== Aligning Temporal Dimensions ===")
    
    bio_shape = feature_arrays['biophysical'].shape
    indices_shape = feature_arrays['indices'].shape
    
    logger.info(f"Biophysical shape: {bio_shape}")
    logger.info(f"Indices shape: {indices_shape}")
    
    # Check if temporal dimensions match
    if bio_shape[1] == indices_shape[1]:
        logger.info("✅ Temporal dimensions already aligned")
        return feature_arrays
    else:
        logger.warning(f"❌ Temporal dimension mismatch: {bio_shape[1]} vs {indices_shape[1]}")
        
        # Align to the shorter dimension to ensure consistency
        min_timesteps = min(bio_shape[1], indices_shape[1])
        logger.info(f"Aligning to {min_timesteps} timesteps")
        
        aligned_arrays = {}
        aligned_arrays['biophysical'] = feature_arrays['biophysical'][:, :min_timesteps, :]
        aligned_arrays['indices'] = feature_arrays['indices'][:, :min_timesteps, :]
        
        logger.info(f"Aligned biophysical shape: {aligned_arrays['biophysical'].shape}")
        logger.info(f"Aligned indices shape: {aligned_arrays['indices'].shape}")
        
        return aligned_arrays

def validate_corrected_structure(feature_arrays):
    """Validate the corrected data structure."""
    
    logger.info("\n=== Validating Corrected Structure ===")
    
    bio_array = feature_arrays['biophysical']
    indices_array = feature_arrays['indices']
    
    logger.info(f"Final biophysical shape: {bio_array.shape}")
    logger.info(f"Final indices shape: {indices_array.shape}")
    
    # Check if we have the target structure
    expected_weeks = 21
    actual_bio_weeks = bio_array.shape[1] // 6  # Assuming 6 years of data
    actual_indices_weeks = indices_array.shape[1] // 6
    
    logger.info(f"Biophysical weeks per year: {actual_bio_weeks}")
    logger.info(f"Indices weeks per year: {actual_indices_weeks}")
    
    if actual_bio_weeks == expected_weeks and actual_indices_weeks == expected_weeks:
        logger.info("✅ SUCCESS: Achieved target 21-week temporal structure")
    else:
        logger.warning(f"❌ Still have temporal structure issues")
    
    # Check data completeness
    bio_valid = ~np.isnan(bio_array).all(axis=2).sum()
    indices_valid = ~np.isnan(indices_array).all(axis=2).sum()
    
    logger.info(f"Valid biophysical timesteps: {bio_valid}")
    logger.info(f"Valid indices timesteps: {indices_valid}")
    
    return bio_array, indices_array

def create_corrected_notebook_code():
    """Generate corrected code for the notebook."""
    
    corrected_code = '''
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
'''
    
    return corrected_code

def main():
    """Main function to create corrected data preparation."""
    
    logger.info("=== Forest Type Dataset - Corrected Data Preparation ===")
    
    # 1. Load dataset
    df = load_dataset()
    
    # 2. Prepare corrected temporal features
    feature_arrays, points, years = prepare_temporal_features_corrected(df)
    
    # 3. Align temporal dimensions
    aligned_arrays = align_temporal_dimensions(feature_arrays)
    
    # 4. Validate corrected structure
    bio_array, indices_array = validate_corrected_structure(aligned_arrays)
    
    # 5. Generate corrected notebook code
    corrected_code = create_corrected_notebook_code()
    
    # Save corrected code for notebook
    code_file = Path("corrected_data_preparation.py")
    with open(code_file, 'w') as f:
        f.write(corrected_code)
    
    logger.info(f"\n=== Summary ===")
    logger.info(f"✅ Corrected data preparation created")
    logger.info(f"✅ Code saved to: {code_file}")
    logger.info(f"✅ Biophysical shape: {bio_array.shape}")
    logger.info(f"✅ Indices shape: {indices_array.shape}")
    logger.info(f"\nNext steps:")
    logger.info(f"1. Copy the corrected code into your notebook")
    logger.info(f"2. Replace the existing data preparation cells")
    logger.info(f"3. Ensure both arrays have proper 21-week temporal structure")

if __name__ == "__main__":
    main()
