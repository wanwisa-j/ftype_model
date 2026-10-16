#!/usr/bin/env python3
"""
Clean Dataset by Keeping Latest Data and Removing NaN/NoData

This script:
1. Drops records with NaN or 'nodata' values
2. For duplicate band combinations, keeps the last/latest data (most recently merged)
3. Ensures each point has exactly 21 bands per feature-year combination
"""

import pandas as pd
import numpy as np

def clean_dataset_keep_latest():
    """Clean dataset by removing NaN/nodata and keeping latest duplicate data"""
    
    print("=== DATASET CLEANING: KEEP LATEST DATA ===")
    
    # Load dataset
    dataset_path = '/fs2/ftype_mflf/shared_data/dataset/1.2.0/raw/forest_type_dataset_raw_20250903_145707.parquet'
    backup_path = '/fs2/ftype_mflf/shared_data/dataset/1.2.0/raw/forest_type_dataset_raw_20250903_145707_BEFORE_LATEST_CLEAN.parquet'
    
    print(f"Loading dataset: {dataset_path}")
    df = pd.read_parquet(dataset_path)
    print(f"Original dataset shape: {df.shape}")
    
    # Create backup
    print(f"Creating backup...")
    df.to_parquet(backup_path)
    print(f"Backup created: {backup_path}")
    
    # Step 1: Remove NaN and nodata values
    print(f"\n=== STEP 1: REMOVING NaN/NODATA VALUES ===")
    
    # Check for NaN values
    nan_values = df['value'].isna().sum()
    print(f"NaN values in 'value' column: {nan_values:,}")
    
    # Check for 'nodata' string values
    if df['value'].dtype == 'object':
        nodata_values = (df['value'] == 'nodata').sum()
        print(f"'nodata' string values: {nodata_values:,}")
    else:
        nodata_values = 0
        print(f"'nodata' string values: 0 (numeric column)")
    
    # Check target columns for nodata
    target_nodata = 0
    if 'ftype_mix' in df.columns:
        mix_nodata = (df['ftype_mix'] == 'nodata').sum()
        print(f"'nodata' in ftype_mix: {mix_nodata:,}")
        target_nodata += mix_nodata
    
    if 'ftype_nonmix' in df.columns:
        nonmix_nodata = (df['ftype_nonmix'] == 'nodata').sum()
        print(f"'nodata' in ftype_nonmix: {nonmix_nodata:,}")
        target_nodata += nonmix_nodata
    
    # Remove NaN values (if any)
    if nan_values > 0:
        df_clean = df.dropna(subset=['value'])
        print(f"After removing NaN values: {len(df_clean):,} records")
    else:
        df_clean = df.copy()
        print(f"No NaN values to remove")
    
    # Remove 'nodata' string values (if any)
    if nodata_values > 0:
        df_clean = df_clean[df_clean['value'] != 'nodata']
        print(f"After removing 'nodata' values: {len(df_clean):,} records")
    
    # Step 2: Handle duplicate bands by keeping the last occurrence
    print(f"\n=== STEP 2: HANDLING DUPLICATE BANDS (KEEP LAST) ===")
    
    # Define key columns for identifying unique records
    key_columns = ['point_id', 'zone', 'feature', 'year', 'band']
    
    # Check for duplicates
    duplicates_mask = df_clean.duplicated(subset=key_columns, keep=False)
    total_duplicates = duplicates_mask.sum()
    print(f"Total records with duplicates: {total_duplicates:,}")
    
    if total_duplicates > 0:
        print(f"Duplicate combinations by key columns:")
        duplicate_counts = df_clean[duplicates_mask].groupby(key_columns).size()
        print(f"Duplicate count distribution:")
        print(duplicate_counts.value_counts().sort_index())
        
        # Keep the last occurrence of each duplicate
        # This keeps the most recently merged/added data
        df_clean = df_clean.drop_duplicates(subset=key_columns, keep='last')
        print(f"After removing duplicates (keeping last): {len(df_clean):,} records")
    else:
        print("No duplicates found")
    
    # Step 3: Verify the structure, especially VV/VH 2020/2023
    print(f"\n=== STEP 3: VERIFYING CLEANED STRUCTURE ===")
    
    # Focus on previously problematic areas
    problematic = df_clean[
        (df_clean['feature'].isin(['vv', 'vh'])) &
        (df_clean['year'].isin([2020, 2023])) &
        (df_clean['zone'].isin(['mfl_cmf_area_project_2025_chr_zone_1', 'mfl_cmf_area_project_2025_chr_zone_2']))
    ]
    
    if len(problematic) > 0:
        print(f"VV/VH 2020/2023 records after cleaning: {len(problematic):,}")
        
        # Check bands per point
        bands_per_point = problematic.groupby(['zone', 'feature', 'year', 'point_id']).size()
        print(f"\nBands per point distribution:")
        band_distribution = bands_per_point.value_counts().sort_index()
        print(band_distribution)
        
        # Check for perfect 21-band points
        perfect_points = (bands_per_point == 21).sum()
        total_points = len(bands_per_point)
        
        print(f"\nPoints with exactly 21 bands: {perfect_points}/{total_points} ({perfect_points/total_points*100:.1f}%)")
        
        if perfect_points == total_points:
            print("✅ All VV/VH points now have exactly 21 bands!")
        else:
            print("⚠️  Some points still don't have 21 bands")
            
            # Show problematic points (up to 10)
            problematic_points = bands_per_point[bands_per_point != 21]
            print(f"\nProblematic points ({len(problematic_points)} total):")
            for idx, count in problematic_points.head(10).items():
                zone, feature, year, point_id = idx
                print(f"  Point {point_id} ({zone}, {feature} {int(year)}): {count} bands")
    
    # Step 4: Overall dataset verification
    print(f"\n=== STEP 4: OVERALL DATASET VERIFICATION ===")
    
    # Check missing values
    missing_values = df_clean['value'].isna().sum()
    print(f"Missing values after cleaning: {missing_values:,}")
    
    # Check all temporal features for 21-band structure
    temporal_features = ['fapar', 'fcover', 'gvmi', 'lai', 'ndvi', 'slavi', 'vh', 'vv']
    static_features = ['elevation', 'slope']
    
    print(f"\nTemporal structure verification:")
    all_temporal_perfect = True
    
    for feature in temporal_features:
        feature_data = df_clean[(df_clean['feature'] == feature) & (~df_clean['year'].isna())]
        if len(feature_data) > 0:
            bands_per_point = feature_data.groupby(['zone', 'year', 'point_id']).size()
            perfect_points = (bands_per_point == 21).sum()
            total_points = len(bands_per_point)
            
            if perfect_points == total_points:
                print(f"  ✅ {feature}: {perfect_points}/{total_points} (100%) perfect")
            else:
                print(f"  ⚠️  {feature}: {perfect_points}/{total_points} ({perfect_points/total_points*100:.1f}%) perfect")
                all_temporal_perfect = False
    
    if all_temporal_perfect:
        print("✅ All temporal features have perfect 21-band structure!")
    else:
        print("⚠️  Some temporal features still have structural issues")
    
    # Save cleaned dataset
    print(f"\n=== SAVING CLEANED DATASET ===")
    df_clean.to_parquet(dataset_path)
    print(f"Cleaned dataset saved: {dataset_path}")
    
    # Final summary
    original_size = len(df)
    cleaned_size = len(df_clean)
    removed_records = original_size - cleaned_size
    
    print(f"\n=== FINAL SUMMARY ===")
    print(f"Original records: {original_size:,}")
    print(f"Cleaned records: {cleaned_size:,}")
    print(f"Removed records: {removed_records:,}")
    print(f"Reduction: {removed_records/original_size*100:.2f}%")
    print(f"Missing values: {missing_values:,}")
    
    if missing_values == 0 and all_temporal_perfect:
        print(f"\n🎉 DATASET CLEANING COMPLETE - PERFECT STRUCTURE!")
    elif missing_values == 0:
        print(f"\n✅ DATASET CLEANING COMPLETE - NO MISSING VALUES!")
    else:
        print(f"\n⚠️  DATASET CLEANING COMPLETE - SOME ISSUES REMAIN")
    
    print(f"Backup: {backup_path}")
    print(f"Cleaned: {dataset_path}")
    
    return df_clean

if __name__ == "__main__":
    result = clean_dataset_keep_latest()