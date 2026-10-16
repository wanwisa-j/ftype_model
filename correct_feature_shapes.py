#!/usr/bin/env python3
"""
Correct Feature Reshaping for Forest Type Classification
========================================================

This script demonstrates the CORRECT approach where each year for each ground point
is treated as a separate datapoint, giving us:

CORRECT SHAPES:
- Biophysical: (10,290, 21, 3) = 1,715 points × 6 years × 21 weeks × 3 features
- Indices: (10,290, 21, 5) = 1,715 points × 6 years × 21 weeks × 5 features  
- Labels: (10,290,) = same target repeated for each year of same point

This approach maximizes training data while maintaining the logical weekly structure.
"""

import numpy as np
import pandas as pd
import os

def load_and_process_correct_shapes():
    """Demonstrate the correct reshaping approach"""
    print("🎯 CORRECT Feature Reshaping Demonstration")
    print("=" * 60)
    
    # Load dataset (using same path as verification script)
    dataset_path = '/fs2/ftype_mflf/shared_data/dataset/1.2.0/raw/forest_type_dataset_raw_20250903_104423.parquet'
    
    if not os.path.exists(dataset_path):
        print(f"❌ Dataset not found: {dataset_path}")
        return
    
    print("📁 Loading dataset...")
    df_raw = pd.read_parquet(dataset_path)
    
    # Apply same filtering as before
    print("🔧 Applying forest type filtering...")
    TARGET_FOREST_TYPES = ['MDF', 'DDF', 'DEF', 'MIX']
    
    def normalize_forest_type(ftype_series):
        normalized = ftype_series.str.upper().str.strip()
        type_mapping = {'MDF': 'MDF', 'DDF': 'DDF', 'DEF': 'DEF', 'MIX': 'MIX', 'MIXED': 'MIX'}
        return normalized.map(type_mapping)
    
    point_metadata = df_raw.groupby('point_id')[['ftype_mix', 'ftype_nonmix', 'zone']].first().reset_index()
    point_metadata['ftype_mix_normalized'] = normalize_forest_type(point_metadata['ftype_mix'])
    valid_ftype_mix = point_metadata['ftype_mix_normalized'].isin(TARGET_FOREST_TYPES)
    valid_points = point_metadata[valid_ftype_mix]['point_id'].values
    df_filtered = df_raw[df_raw['point_id'].isin(valid_points)].copy()
    
    print(f"✅ Filtered dataset: {len(df_filtered):,} records, {df_filtered['point_id'].nunique():,} points")
    
    return df_filtered, point_metadata

def simulate_correct_reshape():
    """Simulate the correct reshaping process"""
    print("\n🔄 Simulating CORRECT Reshape Process")
    print("=" * 50)
    
    # Simulate current shapes (from notebook)
    n_ground_points = 1715
    n_years = 6
    weeks_per_year = 21
    total_timesteps = n_years * weeks_per_year  # 126
    
    print(f"📊 Current (WRONG) shapes from notebook:")
    print(f"  Biophysical: ({n_ground_points}, {total_timesteps}, 3)")
    print(f"  Indices:     ({n_ground_points}, {total_timesteps}, 5)")
    print(f"  Logic: All years concatenated into long sequences")
    
    # Create dummy data to demonstrate reshape
    print(f"\n🔄 Creating dummy data to demonstrate reshape...")
    
    # Current wrong shape
    biophys_wrong = np.random.random((n_ground_points, total_timesteps, 3))
    indices_wrong = np.random.random((n_ground_points, total_timesteps, 5))
    labels_wrong = np.random.choice(['MDF', 'DDF', 'DEF', 'MIX'], n_ground_points)
    point_ids_wrong = np.arange(n_ground_points)
    
    print(f"Current shapes (WRONG approach):")
    print(f"  biophys_wrong: {biophys_wrong.shape}")
    print(f"  indices_wrong: {indices_wrong.shape}")
    print(f"  labels_wrong: {labels_wrong.shape}")
    print(f"  point_ids_wrong: {point_ids_wrong.shape}")
    
    # CORRECT RESHAPE
    print(f"\n✅ Applying CORRECT reshape...")
    
    # Step 1: Reshape to (points, years, weeks, features)
    biophys_reshaped = biophys_wrong.reshape(n_ground_points, n_years, weeks_per_year, 3)
    indices_reshaped = indices_wrong.reshape(n_ground_points, n_years, weeks_per_year, 5)
    
    print(f"Step 1 - Intermediate reshape:")
    print(f"  biophys_reshaped: {biophys_reshaped.shape}")
    print(f"  indices_reshaped: {indices_reshaped.shape}")
    
    # Step 2: Flatten to (points × years, weeks, features) 
    n_total_samples = n_ground_points * n_years
    biophys_correct = biophys_reshaped.reshape(n_total_samples, weeks_per_year, 3)
    indices_correct = indices_reshaped.reshape(n_total_samples, weeks_per_year, 5)
    
    # Step 3: Replicate labels and point IDs for each year
    labels_correct = np.repeat(labels_wrong, n_years)
    point_ids_correct = np.repeat(point_ids_wrong, n_years)
    
    # Add year information
    years_array = np.tile(np.arange(2019, 2019 + n_years), n_ground_points)
    
    print(f"\nStep 2 - CORRECT final shapes:")
    print(f"  biophys_correct: {biophys_correct.shape}")
    print(f"  indices_correct: {indices_correct.shape}")
    print(f"  labels_correct: {labels_correct.shape}")
    print(f"  point_ids_correct: {point_ids_correct.shape}")
    print(f"  years_array: {years_array.shape}")
    
    # Verification
    print(f"\n🔍 Verification:")
    print(f"  Total samples: {n_total_samples:,} = {n_ground_points:,} points × {n_years} years")
    print(f"  Each sample has {weeks_per_year} timesteps (weeks)")
    print(f"  Biophysical features per timestep: 3 (LAI, FAPAR, FCOVER)")
    print(f"  Indices features per timestep: 5 (GVMI, NDVI, VV, VH, SLAVI)")
    
    # Show data structure
    print(f"\n📋 Data structure example:")
    print(f"  Sample 0: Point {point_ids_correct[0]}, Year {years_array[0]}, Label: {labels_correct[0]}")
    print(f"  Sample 1: Point {point_ids_correct[1]}, Year {years_array[1]}, Label: {labels_correct[1]}")
    print(f"  Sample 6: Point {point_ids_correct[6]}, Year {years_array[6]}, Label: {labels_correct[6]}")
    print(f"  ...")
    print(f"  Sample {n_total_samples-1}: Point {point_ids_correct[-1]}, Year {years_array[-1]}, Label: {labels_correct[-1]}")
    
    return biophys_correct, indices_correct, labels_correct, point_ids_correct, years_array

def create_implementation_function():
    """Create the actual implementation function"""
    print(f"\n📝 Implementation Function")
    print("=" * 40)
    
    implementation = """
def reshape_to_yearly_samples(biophys_data, indices_data, labels, point_ids):
    '''
    Reshape time series data to treat each year as a separate sample
    
    Input shapes:
    - biophys_data: (n_points, total_timesteps, 3) where total_timesteps = n_years * 21
    - indices_data: (n_points, total_timesteps, 5) 
    - labels: (n_points,)
    - point_ids: (n_points,)
    
    Output shapes:
    - biophys_samples: (n_points * n_years, 21, 3)
    - indices_samples: (n_points * n_years, 21, 5)
    - labels_samples: (n_points * n_years,)
    - point_ids_samples: (n_points * n_years,)
    - years_samples: (n_points * n_years,)
    '''
    n_points, total_timesteps, _ = biophys_data.shape
    
    # Determine number of years from total timesteps
    weeks_per_year = 21
    n_years = total_timesteps // weeks_per_year
    
    print(f"Reshaping {n_points} points × {n_years} years = {n_points * n_years} samples")
    
    # Reshape to (points, years, weeks, features)
    biophys_reshaped = biophys_data.reshape(n_points, n_years, weeks_per_year, -1)
    indices_reshaped = indices_data.reshape(n_points, n_years, weeks_per_year, -1)
    
    # Flatten to (points × years, weeks, features)
    biophys_samples = biophys_reshaped.reshape(n_points * n_years, weeks_per_year, -1)
    indices_samples = indices_reshaped.reshape(n_points * n_years, weeks_per_year, -1)
    
    # Replicate labels and point IDs
    labels_samples = np.repeat(labels, n_years)
    point_ids_samples = np.repeat(point_ids, n_years)
    
    # Create year array
    years_samples = np.tile(np.arange(n_years), n_points)
    
    return biophys_samples, indices_samples, labels_samples, point_ids_samples, years_samples

# Usage in notebook:
# biophys_yearly, indices_yearly, labels_yearly, point_ids_yearly, years_yearly = reshape_to_yearly_samples(
#     biophys_data_all, indices_data_all, label_data_all, FID_data_all
# )
"""
    
    print(implementation)
    
    return implementation

def demonstrate_train_test_split():
    """Show how to properly split data to avoid leakage"""
    print(f"\n⚠️  Important: Proper Train/Test Split to Avoid Data Leakage")
    print("=" * 60)
    
    explanation = """
🚨 DATA LEAKAGE RISK:
Since the same spatial point appears in multiple years, you CANNOT use random 
train/test split or you'll have the same location in both training and testing!

✅ CORRECT SPLIT STRATEGY:
Split by POINT_ID, not by samples, to ensure spatial independence.

Example code:
```python
# Get unique point IDs
unique_points = np.unique(point_ids_yearly)

# Split points (not samples) into train/test
from sklearn.model_selection import train_test_split
train_points, test_points = train_test_split(
    unique_points, test_size=0.2, random_state=42, 
    stratify=labels_yearly[::n_years]  # Use one label per point for stratification
)

# Create boolean masks for samples
train_mask = np.isin(point_ids_yearly, train_points)
test_mask = np.isin(point_ids_yearly, test_points)

# Split data
X_biophys_train = biophys_yearly[train_mask]
X_biophys_test = biophys_yearly[test_mask]
y_train = labels_yearly[train_mask]
y_test = labels_yearly[test_mask]
```

This ensures:
- No spatial point appears in both train and test
- Proper evaluation of model generalization
- Maintains temporal information within training"""
    
    print(explanation)

def main():
    """Main demonstration"""
    print("🎯 CORRECT Feature Shapes for Forest Type Classification")
    print("=" * 65)
    print("User Understanding: Each year per ground point = separate datapoint")
    print("Expected Shape: (datapoints, 21_weeks, features)")
    print()
    
    # Load and process data
    df_filtered, point_metadata = load_and_process_correct_shapes()
    
    # Demonstrate correct reshape
    biophys_correct, indices_correct, labels_correct, point_ids_correct, years_array = simulate_correct_reshape()
    
    # Show implementation function
    implementation = create_implementation_function()
    
    # Show proper train/test split
    demonstrate_train_test_split()
    
    print(f"\n✅ SUMMARY")
    print("=" * 30)
    print(f"CORRECT shapes achieved:")
    print(f"  Biophysical: {biophys_correct.shape} ✅")
    print(f"  Indices:     {indices_correct.shape} ✅") 
    print(f"  Labels:      {labels_correct.shape} ✅")
    print()
    print(f"Benefits:")
    print(f"  ✅ Matches your expectation: (datapoints, 21, features)")
    print(f"  ✅ 6x more training data: 10,290 vs 1,715")
    print(f"  ✅ Captures year-to-year variations")
    print(f"  ✅ Logical weekly structure preserved")
    print(f"  ✅ Same target label for same point across years")

if __name__ == "__main__":
    main()
