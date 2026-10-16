#!/usr/bin/env python3
"""
Clean Duplicate VV/VH Entries Script

This script fixes the duplicate VV/VH entries for 2020 and 2023 in zones 1 and 2.
The issue is that we have:
- 2020: 42 bands per point (should be 21) - duplicated data
- 2023: 63 bands per point (should be 21) - tripled data

We need to remove the duplicates and keep only 21 bands per point.
"""

import pandas as pd
import numpy as np
from pathlib import Path

def analyze_duplicate_structure(df):
    """Analyze the structure of duplicates to understand the pattern"""
    
    print("=== ANALYZING DUPLICATE STRUCTURE ===")
    
    # Focus on problematic combinations
    problematic = df[
        (df['feature'].isin(['vv', 'vh'])) &
        (df['year'].isin([2020, 2023])) &
        (df['zone'].isin(['mfl_cmf_area_project_2025_chr_zone_1', 'mfl_cmf_area_project_2025_chr_zone_2']))
    ]
    
    print(f"Problematic records: {len(problematic):,}")
    
    # Analyze band structure
    band_analysis = problematic.groupby(['zone', 'feature', 'year', 'point_id']).agg({
        'band': ['count', 'min', 'max', list],
        'value': 'first'
    }).reset_index()
    
    band_analysis.columns = ['zone', 'feature', 'year', 'point_id', 'band_count', 'min_band', 'max_band', 'band_list', 'sample_value']
    
    print(f"\nBand count distribution:")
    print(band_analysis['band_count'].value_counts().sort_index())
    
    # Show examples of band patterns
    print(f"\nExample band patterns:")
    examples = band_analysis.groupby(['zone', 'feature', 'year', 'band_count']).first().reset_index()
    
    for _, row in examples.head(10).iterrows():
        bands = row['band_list']
        if len(bands) > 25:  # Only show first and last few for long lists
            band_str = f"[{bands[0]}, {bands[1]}, ..., {bands[-2]}, {bands[-1]}] (total: {len(bands)})"
        else:
            band_str = str(bands)
        
        print(f"  {row['zone']} {row['feature']} {int(row['year'])}: {row['band_count']} bands")
        print(f"    Bands: {band_str}")
        print(f"    Range: {row['min_band']}-{row['max_band']}")
    
    return band_analysis

def clean_duplicates(df, dry_run=False):
    """Clean duplicate VV/VH entries, keeping only bands 1-21"""
    
    print(f"\n=== CLEANING DUPLICATES (Dry run: {dry_run}) ===")
    
    # Create a copy for modification
    df_clean = df.copy()
    
    # Identify records to clean
    duplicate_conditions = (
        (df_clean['feature'].isin(['vv', 'vh'])) &
        (df_clean['year'].isin([2020, 2023])) &
        (df_clean['zone'].isin(['mfl_cmf_area_project_2025_chr_zone_1', 'mfl_cmf_area_project_2025_chr_zone_2']))
    )
    
    duplicate_records = df_clean[duplicate_conditions]
    print(f"Records to process: {len(duplicate_records):,}")
    
    # Strategy: Keep only bands 1-21 for each point-feature-year combination
    print(f"\nCleaning strategy: Keep only bands 1-21 for each point")
    
    # Find records to keep (bands 1-21)
    keep_conditions = duplicate_conditions & (df_clean['band'] <= 21)
    records_to_keep = df_clean[keep_conditions]
    
    # Find records to remove (bands > 21)
    remove_conditions = duplicate_conditions & (df_clean['band'] > 21)
    records_to_remove = df_clean[remove_conditions]
    
    print(f"Records to keep (bands 1-21): {len(records_to_keep):,}")
    print(f"Records to remove (bands >21): {len(records_to_remove):,}")
    
    if not dry_run:
        # Remove duplicate records
        df_clean = df_clean[~remove_conditions]
        print(f"After cleaning: {len(df_clean):,} total records")
    
    # Verify the result
    print(f"\nVerifying cleaned structure...")
    
    if not dry_run:
        verify_df = df_clean
    else:
        verify_df = df_clean[~remove_conditions]  # Simulate the result
    
    # Check the bands per point after cleaning
    cleaned_problematic = verify_df[
        (verify_df['feature'].isin(['vv', 'vh'])) &
        (verify_df['year'].isin([2020, 2023])) &
        (verify_df['zone'].isin(['mfl_cmf_area_project_2025_chr_zone_1', 'mfl_cmf_area_project_2025_chr_zone_2']))
    ]
    
    if len(cleaned_problematic) > 0:
        cleaned_bands = cleaned_problematic.groupby(['zone', 'feature', 'year', 'point_id']).size()
        print(f"Bands per point after cleaning:")
        print(cleaned_bands.value_counts().sort_index())
        
        # Check if all have 21 bands
        perfect_points = (cleaned_bands == 21).sum()
        total_points = len(cleaned_bands)
        
        print(f"Points with 21 bands: {perfect_points}/{total_points} ({perfect_points/total_points*100:.1f}%)")
        
        if perfect_points == total_points:
            print("✅ All points now have exactly 21 bands!")
        else:
            print("⚠️  Some points still don't have 21 bands")
    
    if not dry_run:
        return df_clean
    else:
        return df

def main():
    """Main function to clean duplicates"""
    
    print("=== VV/VH DUPLICATE CLEANING SCRIPT ===")
    
    # Load dataset
    dataset_path = '/fs2/ftype_mflf/shared_data/dataset/1.2.0/raw/forest_type_dataset_raw_20250903_145707.parquet'
    backup_path = '/fs2/ftype_mflf/shared_data/dataset/1.2.0/raw/forest_type_dataset_raw_20250903_145707_BEFORE_DUPLICATE_CLEAN.parquet'
    
    print(f"Loading dataset: {dataset_path}")
    df = pd.read_parquet(dataset_path)
    print(f"Dataset shape: {df.shape}")
    
    # Analyze duplicates first
    band_analysis = analyze_duplicate_structure(df)
    
    # Create backup
    print(f"\nCreating backup...")
    df.to_parquet(backup_path)
    print(f"Backup created: {backup_path}")
    
    # Dry run first
    print(f"\n" + "="*60)
    print("PERFORMING DRY RUN...")
    clean_duplicates(df, dry_run=True)
    
    # Ask for confirmation (simulate user input)
    print(f"\n" + "="*60)
    print("DRY RUN COMPLETE - PROCEEDING WITH ACTUAL CLEANING...")
    
    # Perform actual cleaning
    df_cleaned = clean_duplicates(df, dry_run=False)
    
    # Save cleaned dataset
    print(f"\nSaving cleaned dataset...")
    df_cleaned.to_parquet(dataset_path)
    print(f"Cleaned dataset saved: {dataset_path}")
    
    # Final summary
    original_size = len(df)
    cleaned_size = len(df_cleaned)
    removed_records = original_size - cleaned_size
    
    print(f"\n=== CLEANING SUMMARY ===")
    print(f"Original records: {original_size:,}")
    print(f"Cleaned records: {cleaned_size:,}")
    print(f"Removed duplicates: {removed_records:,}")
    print(f"Reduction: {removed_records/original_size*100:.2f}%")
    
    print(f"\n✅ DUPLICATE CLEANING COMPLETE!")
    print(f"Backup: {backup_path}")
    print(f"Cleaned: {dataset_path}")
    
    return df_cleaned

if __name__ == "__main__":
    result = main()