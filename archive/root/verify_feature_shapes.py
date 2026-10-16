#!/usr/bin/env python3
"""
Feature Shape Verification Script
=================================

This script verifies the expected shapes of features for the forest type classification model.
Expected shapes:
- Biophysical data: (data_points, 21_timesteps, 3_features) for LAI, FAPAR, FCOVER  
- Indices data: (data_points, 21_timesteps, 5_features) for GVMI, NDVI, VV, VH, SLAVI
- Static data: (data_points, 2) for elevation, slope
"""

import numpy as np
import pandas as pd
import sys
import os
from pathlib import Path

def load_dataset():
    """Load the comprehensive dataset v1.2.0"""
    dataset_path = '/fs2/ftype_mflf/shared_data/dataset/1.2.0/raw/forest_type_dataset_raw_20250903_104423.parquet'
    
    if not os.path.exists(dataset_path):
        print(f"❌ Dataset not found: {dataset_path}")
        return None
        
    print(f"📁 Loading dataset: {dataset_path}")
    df_raw = pd.read_parquet(dataset_path)
    print(f"✅ Loaded {len(df_raw):,} records")
    
    return df_raw

def apply_forest_type_filtering(df_raw):
    """Apply the same filtering and normalization as in the notebook"""
    print("\n🔧 Applying forest type filtering...")
    
    # Target forest types
    TARGET_FOREST_TYPES = ['MDF', 'DDF', 'DEF', 'MIX']
    
    def normalize_forest_type(ftype_series):
        """Normalize forest type labels"""
        normalized = ftype_series.str.upper().str.strip()
        type_mapping = {
            'MDF': 'MDF',
            'DDF': 'DDF', 
            'DEF': 'DEF',
            'MIX': 'MIX',
            'MIXED': 'MIX',
        }
        return normalized.map(type_mapping)
    
    # Get unique point metadata
    point_metadata = df_raw.groupby('point_id')[['ftype_mix', 'ftype_nonmix', 'zone']].first().reset_index()
    
    # Normalize forest types
    point_metadata['ftype_mix_normalized'] = normalize_forest_type(point_metadata['ftype_mix'])
    
    # Filter to target forest types
    valid_ftype_mix = point_metadata['ftype_mix_normalized'].isin(TARGET_FOREST_TYPES)
    valid_points = point_metadata[valid_ftype_mix]['point_id'].values
    
    # Filter the dataset
    df_filtered = df_raw[df_raw['point_id'].isin(valid_points)].copy()
    
    # Update ftype columns with normalized values
    point_id_to_ftype_mix = dict(zip(point_metadata['point_id'], point_metadata['ftype_mix_normalized']))
    df_filtered['ftype_mix'] = df_filtered['point_id'].map(point_id_to_ftype_mix)
    
    print(f"📍 Points after filtering: {df_filtered['point_id'].nunique():,}")
    print(f"📊 Records after filtering: {len(df_filtered):,}")
    
    return df_filtered

def analyze_temporal_structure(df_filtered):
    """Analyze the temporal structure of the data"""
    print("\n📅 Analyzing temporal structure...")
    
    # Check years and weeks available
    years_available = sorted(df_filtered[df_filtered['year'].notna()]['year'].unique())
    weeks_available = sorted(df_filtered[df_filtered['week'].notna()]['week'].unique())
    
    print(f"Available years: {years_available}")
    print(f"Available weeks: {weeks_available}")
    print(f"Expected weeks per year: 21 (based on data extraction)")
    print(f"Actual weeks range: {min(weeks_available)} to {max(weeks_available)}")
    
    # Check temporal coverage by feature
    print(f"\n📊 Temporal coverage by feature:")
    temporal_analysis = df_filtered[df_filtered['year'].notna()].groupby(['feature', 'year']).size().unstack(fill_value=0)
    print(temporal_analysis)
    
    return years_available, weeks_available

def verify_feature_groups(df_filtered):
    """Verify feature groups and their expected shapes"""
    print("\n🔍 Verifying feature groups...")
    
    # Define expected feature groups
    biophys_features = ['lai', 'fapar', 'fcover']
    indices_features = ['gvmi', 'ndvi', 'vv', 'vh', 'slavi']
    static_features = ['elevation', 'slope']
    
    all_features = sorted(df_filtered['feature'].unique())
    print(f"All features in dataset: {all_features}")
    
    # Check feature availability
    print(f"\n📊 Feature group analysis:")
    print(f"Expected biophysical features: {biophys_features}")
    biophys_available = [f for f in biophys_features if f in all_features]
    print(f"Available biophysical features: {biophys_available}")
    
    print(f"\nExpected indices features: {indices_features}")
    indices_available = [f for f in indices_features if f in all_features]
    print(f"Available indices features: {indices_available}")
    
    print(f"\nExpected static features: {static_features}")
    static_available = [f for f in static_features if f in all_features]
    print(f"Available static features: {static_available}")
    
    return biophys_available, indices_available, static_available

def simulate_time_series_conversion(df_filtered, biophys_features, indices_features, years_to_use):
    """Simulate the time series conversion to check expected shapes"""
    print(f"\n🔄 Simulating time series conversion...")
    print(f"Using years: {years_to_use}")
    
    # Get unique points
    unique_points = df_filtered['point_id'].unique()
    n_points = len(unique_points)
    print(f"Number of unique points: {n_points}")
    
    # Calculate expected timesteps
    n_years = len(years_to_use)
    weeks_per_year = 21  # Based on extraction configuration
    total_timesteps = n_years * weeks_per_year
    
    print(f"Expected timesteps calculation:")
    print(f"  Years: {n_years}")
    print(f"  Weeks per year: {weeks_per_year}")
    print(f"  Total timesteps: {total_timesteps}")
    
    # Expected shapes
    expected_biophys_shape = (n_points, total_timesteps, len(biophys_features))
    expected_indices_shape = (n_points, total_timesteps, len(indices_features))
    
    print(f"\n🎯 Expected shapes:")
    print(f"  Biophysical data: {expected_biophys_shape}")
    print(f"  Indices data: {expected_indices_shape}")
    print(f"  Static data: ({n_points}, {len(['elevation', 'slope'])})")
    
    # Check actual data availability for a sample point
    sample_point = unique_points[0]
    sample_data = df_filtered[df_filtered['point_id'] == sample_point]
    
    print(f"\n🔍 Sample point analysis (point_id: {sample_point}):")
    
    for feature in biophys_features + indices_features:
        feature_data = sample_data[sample_data['feature'] == feature]
        if len(feature_data) > 0:
            years_in_data = sorted(feature_data[feature_data['year'].notna()]['year'].unique())
            weeks_in_data = sorted(feature_data[feature_data['week'].notna()]['week'].unique())
            print(f"  {feature}: {len(feature_data)} records, years: {years_in_data}, weeks: {len(weeks_in_data)} unique")
        else:
            print(f"  {feature}: No data found")
    
    return expected_biophys_shape, expected_indices_shape

def check_actual_pivot_result(df_filtered, biophys_features, indices_features, years_to_use):
    """Actually perform a pivot to see the real shapes"""
    print(f"\n🔄 Performing actual pivot to verify shapes...")
    
    # Separate time series and static data
    ts_features = biophys_features + indices_features
    ts_data = df_filtered[df_filtered.feature.isin(ts_features)]
    
    print(f"Time series records: {len(ts_data):,}")
    
    # Filter by years
    ts_data_filtered = ts_data[
        (ts_data.year.isin(years_to_use)) | (ts_data.year.isna())
    ]
    
    print(f"Time series records after year filtering: {len(ts_data_filtered):,}")
    
    # Check data structure for biophysical features
    print(f"\n📊 Checking biophysical data structure:")
    bio_data = ts_data_filtered[ts_data_filtered.feature.isin(biophys_features)]
    if len(bio_data) > 0:
        print(f"  Records: {len(bio_data):,}")
        bio_pivot_sample = bio_data.head(1000)  # Sample for pivot test
        try:
            bio_wide = bio_pivot_sample.pivot_table(
                index='point_id',
                columns=['feature', 'week', 'year'],
                values='value',
                aggfunc='first'
            )
            print(f"  Sample pivot shape: {bio_wide.shape}")
            print(f"  Columns sample: {list(bio_wide.columns[:5])}")
        except Exception as e:
            print(f"  Pivot error: {e}")
    
    # Check data structure for indices features  
    print(f"\n📊 Checking indices data structure:")
    indices_data = ts_data_filtered[ts_data_filtered.feature.isin(indices_features)]
    if len(indices_data) > 0:
        print(f"  Records: {len(indices_data):,}")
        indices_pivot_sample = indices_data.head(1000)  # Sample for pivot test
        try:
            indices_wide = indices_pivot_sample.pivot_table(
                index='point_id',
                columns=['feature', 'week', 'year'],
                values='value',
                aggfunc='first'
            )
            print(f"  Sample pivot shape: {indices_wide.shape}")
            print(f"  Columns sample: {list(indices_wide.columns[:5])}")
        except Exception as e:
            print(f"  Pivot error: {e}")

def main():
    """Main verification function"""
    print("🔍 Feature Shape Verification Script")
    print("=" * 50)
    
    # Load dataset
    df_raw = load_dataset()
    if df_raw is None:
        return
    
    # Apply filtering
    df_filtered = apply_forest_type_filtering(df_raw)
    
    # Analyze temporal structure
    years_available, weeks_available = analyze_temporal_structure(df_filtered)
    
    # Verify feature groups
    biophys_features, indices_features, static_features = verify_feature_groups(df_filtered)
    
    # Use years with good coverage (same logic as notebook)
    years_to_use = [2019, 2020, 2021, 2022, 2023, 2024]  # Based on notebook analysis
    
    # Simulate conversion
    expected_biophys_shape, expected_indices_shape = simulate_time_series_conversion(
        df_filtered, biophys_features, indices_features, years_to_use
    )
    
    # Check actual pivot
    check_actual_pivot_result(df_filtered, biophys_features, indices_features, years_to_use)
    
    print(f"\n✅ Verification complete!")
    print(f"\n🎯 Summary:")
    print(f"Expected biophysical shape: {expected_biophys_shape}")
    print(f"Expected indices shape: {expected_indices_shape}")
    print(f"Your understanding: biophys (points, 21, 3), indices (points, 21, 5)")
    
    # Compare with your expectation
    n_points = expected_biophys_shape[0]
    your_expected_biophys = (n_points, 21, 3)
    your_expected_indices = (n_points, 21, 5)
    
    print(f"\n🔍 Comparison with your expectation:")
    print(f"Biophysical - Expected: {expected_biophys_shape}, Your expectation: {your_expected_biophys}")
    print(f"Indices - Expected: {expected_indices_shape}, Your expectation: {your_expected_indices}")
    
    if expected_biophys_shape == your_expected_biophys:
        print("✅ Biophysical shape matches your expectation!")
    else:
        print("❌ Biophysical shape differs from your expectation")
        print(f"   Difference likely due to: multiple years ({len(years_to_use)} years × 21 weeks = {len(years_to_use) * 21} timesteps)")
    
    if expected_indices_shape == your_expected_indices:
        print("✅ Indices shape matches your expectation!")
    else:
        print("❌ Indices shape differs from your expectation")
        print(f"   Difference likely due to: multiple years ({len(years_to_use)} years × 21 weeks = {len(years_to_use) * 21} timesteps)")

if __name__ == "__main__":
    main()
