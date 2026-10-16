#!/usr/bin/env python3
"""
Debug the dataset structure to understand why pivot failed
"""

import pandas as pd
import numpy as np

# Load the backup (original long format)
backup_path = "/fs2/ftype_mflf/shared_data/dataset/1.2.0/raw/forest_type_dataset_raw_20250903_145707.cleaned_with_new_biophy_20250909_071747_LONG_FORMAT_BACKUP_20250909_084631.parquet"

print(f"Loading backup dataset: {backup_path}")
df = pd.read_parquet(backup_path)

print(f"Dataset shape: {df.shape}")
print(f"Columns: {list(df.columns)}")

# Check sample data
print(f"\nFirst 10 rows:")
print(df.head(10))

# Check unique combinations
print(f"\nChecking for duplicates...")
id_cols = ['point_id', 'band', 'week', 'file_path', 'utm_x', 'utm_y', 
           'ftype_mix', 'ftype_nonmix', 'timestep', 'zone', 'year']

# Check if there are duplicates in the ID columns + feature combination
print(f"Checking duplicates in ID + feature combination...")
df_check = df.dropna(subset=['zone', 'feature', 'value'])
duplicate_check = df_check.groupby(id_cols + ['feature']).size()
duplicates = duplicate_check[duplicate_check > 1]

print(f"Number of duplicate combinations: {len(duplicates)}")
if len(duplicates) > 0:
    print(f"Sample duplicates:")
    print(duplicates.head())

# Check specific zone data
problem_zones = ['mfl_cmf_area_project_2025_chr_zone_1', 'mfl_cmf_area_project_2025_chr_zone_2']
biophy_features = ['lai', 'fapar', 'fcover']

print(f"\n=== ZONE DATA ANALYSIS ===")
for zone in problem_zones:
    zone_data = df[df['zone'] == zone]
    print(f"\n{zone}: {len(zone_data):,} records")
    
    # Check 2021 data
    zone_2021 = zone_data[zone_data['year'] == 2021]
    print(f"  2021 records: {len(zone_2021):,}")
    
    if len(zone_2021) > 0:
        print(f"  Sample 2021 record:")
        print(zone_2021.head(1))
        
        for feature in biophy_features:
            feature_data = zone_2021[zone_2021['feature'] == feature]
            print(f"    {feature}: {len(feature_data):,} records")
            if len(feature_data) > 0:
                values = feature_data['value'].dropna()
                if len(values) > 0:
                    print(f"      Value range: [{values.min():.3f}, {values.max():.3f}]")

# Check if there are NaN values in critical columns
print(f"\n=== NaN CHECK ===")
for col in ['point_id', 'zone', 'feature', 'year']:
    nan_count = df[col].isna().sum()
    print(f"{col}: {nan_count:,} NaN values")

# Check unique point_ids per zone
print(f"\n=== UNIQUE POINTS PER ZONE ===")
for zone in problem_zones:
    zone_data = df[df['zone'] == zone]
    unique_points = zone_data['point_id'].nunique()
    print(f"{zone}: {unique_points} unique points")
