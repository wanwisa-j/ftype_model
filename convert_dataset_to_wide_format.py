#!/usr/bin/env python3
"""
Convert the long format biophysical dataset to wide format for proper verification
"""

import pandas as pd
import numpy as np
from datetime import datetime
import sys

def convert_long_to_wide_format():
    """Convert the dataset from long format to wide format."""
    
    # Load the current dataset
    dataset_path = "/fs2/ftype_mflf/shared_data/dataset/1.2.0/raw/forest_type_dataset_raw_20250903_145707.cleaned_with_new_biophy_20250909_071747.parquet"
    
    print(f"Loading dataset: {dataset_path}")
    df_long = pd.read_parquet(dataset_path)
    print(f"Original dataset shape: {df_long.shape}")
    
    # Create backup
    backup_path = dataset_path.replace('.parquet', f'_LONG_FORMAT_BACKUP_{datetime.now().strftime("%Y%m%d_%H%M%S")}.parquet')
    df_long.to_parquet(backup_path)
    print(f"Backup created: {backup_path}")
    
    # Check current structure
    print(f"\nCurrent dataset structure:")
    print(f"Columns: {list(df_long.columns)}")
    print(f"Features available: {sorted(df_long['feature'].unique())}")
    print(f"Years available: {sorted(df_long['year'].dropna().unique())}")
    print(f"Zones available: {sorted(df_long['zone'].unique())}")
    
    # Convert to wide format
    print(f"\nConverting to wide format...")
    
    # First, handle the NaN years by filling them with a placeholder or removing them
    print(f"NaN years found: {df_long['year'].isna().sum()}")
    
    # For static features (elevation, slope), year might be NaN, so let's handle this
    # We'll fill NaN years with -1 for static features or drop them for temporal features
    df_clean = df_long.copy()
    
    # For static features, fill NaN year with -1
    static_features = ['elevation', 'slope']
    temporal_features = ['lai', 'fapar', 'fcover', 'ndvi', 'vh', 'vv', 'gvmi', 'slavi']
    
    # Fill NaN years for static features
    mask_static = df_clean['feature'].isin(static_features)
    df_clean.loc[mask_static, 'year'] = df_clean.loc[mask_static, 'year'].fillna(-1)
    
    # Remove rows with NaN year for temporal features
    mask_temporal = df_clean['feature'].isin(temporal_features)
    df_clean = df_clean[~(mask_temporal & df_clean['year'].isna())]
    
    # Remove rows with missing critical information
    df_clean = df_clean.dropna(subset=['zone', 'feature', 'value'])
    print(f"After cleaning missing values: {df_clean.shape}")
    
    # Group by the ID columns and pivot on feature
    id_cols = ['point_id', 'band', 'week', 'file_path', 'utm_x', 'utm_y', 
               'ftype_mix', 'ftype_nonmix', 'timestep', 'zone', 'year']
    
    # Pivot the data
    df_wide = df_clean.pivot_table(
        index=id_cols,
        columns='feature',
        values='value',
        aggfunc='first'  # Use first value in case of duplicates
    ).reset_index()
    
    # Flatten column names
    df_wide.columns.name = None
    
    print(f"Wide format shape: {df_wide.shape}")
    print(f"Columns in wide format: {list(df_wide.columns)}")
    
    # Check for missing data in biophysical features
    biophy_features = ['lai', 'fapar', 'fcover']
    problem_zones = ['mfl_cmf_area_project_2025_chr_zone_1', 'mfl_cmf_area_project_2025_chr_zone_2']
    
    print(f"\n=== DATA COMPLETENESS CHECK ===")
    for zone in problem_zones:
        zone_data = df_wide[df_wide['zone'] == zone]
        print(f"\n{zone}: {len(zone_data):,} records")
        
        for feature in biophy_features:
            if feature in df_wide.columns:
                missing = zone_data[feature].isna().sum()
                total = len(zone_data)
                pct_complete = ((total - missing) / total * 100) if total > 0 else 0
                status = "✅" if missing == 0 else "❌"
                print(f"  {status} {feature}: {total - missing:,}/{total:,} ({pct_complete:.1f}% complete)")
                
                if missing == 0 and total > 0:
                    values = zone_data[feature].dropna()
                    if len(values) > 0:
                        print(f"    Range: [{values.min():.3f}, {values.max():.3f}]")
            else:
                print(f"  ⚠️ {feature}: Column not found")
    
    # Check 2021 data specifically
    print(f"\n=== 2021 DATA CHECK ===")
    df_2021 = df_wide[df_wide['year'] == 2021]
    print(f"Total 2021 records: {len(df_2021):,}")
    
    for zone in problem_zones:
        zone_2021 = df_2021[df_2021['zone'] == zone]
        if len(zone_2021) > 0:
            print(f"\n{zone} (2021): {len(zone_2021):,} records")
            for feature in biophy_features:
                if feature in df_wide.columns:
                    missing = zone_2021[feature].isna().sum()
                    total = len(zone_2021)
                    status = "✅" if missing == 0 else "❌"
                    print(f"  {status} {feature}: {total - missing:,}/{total:,} complete")
    
    # Save the wide format dataset
    wide_dataset_path = dataset_path.replace('.parquet', '_WIDE_FORMAT.parquet')
    df_wide.to_parquet(wide_dataset_path)
    print(f"\n✅ Wide format dataset saved: {wide_dataset_path}")
    
    # Also overwrite the original with wide format
    df_wide.to_parquet(dataset_path)
    print(f"✅ Original dataset updated with wide format: {dataset_path}")
    
    return df_wide

def main():
    """Main function."""
    try:
        df_wide = convert_long_to_wide_format()
        print(f"\n🎉 Dataset conversion completed successfully!")
        print(f"Final dataset shape: {df_wide.shape}")
    except Exception as e:
        print(f"\n❌ Error during conversion: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
