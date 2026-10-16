#!/usr/bin/env python3
"""
Clean Exact Duplicate VV/VH Records

The issue is that we have exact duplicate records (same point_id, zone, feature, year, band)
Rather than different band numbers. We need to remove the duplicate rows.
"""

import pandas as pd
import numpy as np

def analyze_exact_duplicates(df):
    """Analyze exact duplicate records"""
    
    print("=== ANALYZING EXACT DUPLICATES ===")
    
    # Focus on problematic combinations
    problematic = df[
        (df['feature'].isin(['vv', 'vh'])) &
        (df['year'].isin([2020, 2023])) &
        (df['zone'].isin(['mfl_cmf_area_project_2025_chr_zone_1', 'mfl_cmf_area_project_2025_chr_zone_2']))
    ]
    
    print(f"Problematic records: {len(problematic):,}")
    
    # Check for exact duplicates based on key columns
    key_columns = ['point_id', 'zone', 'feature', 'year', 'band']
    
    # Find duplicates
    duplicated_records = problematic[problematic.duplicated(subset=key_columns, keep=False)]
    unique_combinations = problematic.drop_duplicates(subset=key_columns)
    
    print(f"Total problematic records: {len(problematic):,}")
    print(f"Unique combinations: {len(unique_combinations):,}")
    print(f"Duplicate records: {len(duplicated_records):,}")
    
    if len(duplicated_records) > 0:
        # Show duplicate pattern
        duplicate_counts = problematic.groupby(key_columns).size()
        print(f"\nDuplicate counts distribution:")
        print(duplicate_counts.value_counts().sort_index())
        
        # Show examples
        example_duplicates = duplicate_counts[duplicate_counts > 1].head()
        print(f"\nExample duplicates:")
        for idx, count in example_duplicates.items():
            print(f"  {idx}: {count} copies")
    
    return unique_combinations

def clean_exact_duplicates(df, dry_run=False):
    """Remove exact duplicate records, keeping only one copy of each unique combination"""
    
    print(f"\n=== CLEANING EXACT DUPLICATES (Dry run: {dry_run}) ===")
    
    original_size = len(df)
    
    # Define key columns that should be unique
    key_columns = ['point_id', 'zone', 'feature', 'year', 'band']
    
    # Check for duplicates across the entire dataset
    total_duplicates = df.duplicated(subset=key_columns).sum()
    print(f"Total duplicate records in dataset: {total_duplicates:,}")
    
    if total_duplicates == 0:
        print("✅ No duplicates found!")
        return df
    
    # Remove duplicates, keeping the first occurrence
    if not dry_run:
        df_cleaned = df.drop_duplicates(subset=key_columns, keep='first')
        print(f"After deduplication: {len(df_cleaned):,} records")
    else:
        df_cleaned = df  # Don't modify for dry run
        expected_size = len(df) - total_duplicates
        print(f"Expected size after cleaning: {expected_size:,} records")
    
    # Verify problematic areas specifically
    print(f"\nVerifying VV/VH 2020/2023 structure...")
    
    if not dry_run:
        verify_df = df_cleaned
    else:
        verify_df = df.drop_duplicates(subset=key_columns, keep='first')  # Simulate result
    
    # Check the problematic combinations
    cleaned_problematic = verify_df[
        (verify_df['feature'].isin(['vv', 'vh'])) &
        (verify_df['year'].isin([2020, 2023])) &
        (verify_df['zone'].isin(['mfl_cmf_area_project_2025_chr_zone_1', 'mfl_cmf_area_project_2025_chr_zone_2']))
    ]
    
    if len(cleaned_problematic) > 0:
        # Check bands per point
        bands_per_point = cleaned_problematic.groupby(['zone', 'feature', 'year', 'point_id']).size()
        print(f"Bands per point after cleaning:")
        band_distribution = bands_per_point.value_counts().sort_index()
        print(band_distribution)
        
        # Check for 21-band points
        perfect_points = (bands_per_point == 21).sum()
        total_points = len(bands_per_point)
        
        print(f"Points with 21 bands: {perfect_points}/{total_points} ({perfect_points/total_points*100:.1f}%)")
        
        if perfect_points == total_points:
            print("✅ All points now have exactly 21 bands!")
        else:
            print("⚠️  Some points still don't have 21 bands")
            
            # Show problematic points
            problematic_points = bands_per_point[bands_per_point != 21]
            if len(problematic_points) <= 10:
                print("Problematic points:")
                for idx, count in problematic_points.items():
                    print(f"  {idx}: {count} bands")
    
    if not dry_run:
        return df_cleaned
    else:
        return df

def main():
    """Main function to clean exact duplicates"""
    
    print("=== EXACT DUPLICATE CLEANING SCRIPT ===")
    
    # Load dataset
    dataset_path = '/fs2/ftype_mflf/shared_data/dataset/1.2.0/raw/forest_type_dataset_raw_20250903_145707.parquet'
    backup_path = '/fs2/ftype_mflf/shared_data/dataset/1.2.0/raw/forest_type_dataset_raw_20250903_145707_BEFORE_EXACT_DUPLICATE_CLEAN.parquet'
    
    print(f"Loading dataset: {dataset_path}")
    df = pd.read_parquet(dataset_path)
    print(f"Dataset shape: {df.shape}")
    
    # Analyze exact duplicates
    unique_combinations = analyze_exact_duplicates(df)
    
    # Create backup
    print(f"\nCreating backup...")
    df.to_parquet(backup_path)
    print(f"Backup created: {backup_path}")
    
    # Dry run first
    print(f"\n" + "="*60)
    print("PERFORMING DRY RUN...")
    clean_exact_duplicates(df, dry_run=True)
    
    # Perform actual cleaning
    print(f"\n" + "="*60)
    print("PROCEEDING WITH ACTUAL CLEANING...")
    
    df_cleaned = clean_exact_duplicates(df, dry_run=False)
    
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
    
    print(f"\n✅ EXACT DUPLICATE CLEANING COMPLETE!")
    print(f"Backup: {backup_path}")
    print(f"Cleaned: {dataset_path}")
    
    return df_cleaned

if __name__ == "__main__":
    result = main()