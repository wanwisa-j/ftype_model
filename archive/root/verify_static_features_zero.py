#!/usr/bin/env python3
"""
Verification and analysis of static features (slope, elevation) with value 0 or empty (NaN).
Reports which zones and years have points with 0 or missing static features.
"""
import pandas as pd
import numpy as np
from pathlib import Path

def verify_static_features_zero():
    print("=== VERIFY STATIC FEATURES: ZERO OR EMPTY ===")
    # Load the dataset
    df = pd.read_parquet('/fs2/ftype_mflf/shared_data/dataset/1.2.0/raw/forest_type_dataset_raw_20250903_145707.cleaned.parquet')
    static_features = ['slope', 'elevation']
    

    # Additional verification for mfl_cmf_area_project_2025_chr_zone_1
    print("\n=== VERIFY: mfl_cmf_area_project_2025_chr_zone_1 static features for all non-NaN years ===")
    for zone in ['mfl_cmf_2024', 'mfl_cmf_area_project_2025_chr_zone_1', 'mfl_cmf_area_project_2025_chr_zone_2']:
        zone_df = df[df['zone'] == zone]
        non_nan_years = zone_df[~zone_df['year'].isna()]['year'].unique()
        for feature in static_features:
            print(f"\nChecking {feature} for all points and years in {zone} (excluding NaN years)...")
            for year in non_nan_years:
                year_df = zone_df[(zone_df['year'] == year) & (zone_df['feature'] == feature)]
                # For static features, there should be one value per point
                bad = year_df[(year_df['value'] == 0) | (year_df['value'].isna())]
                if not bad.empty:
                    print(f"  Year {int(year)}: {len(bad)} points with {feature}=0 or NaN")
                    print(bad[['point_id', 'utm_x', 'utm_y']].head(10))
                else:
                    print(f"  Year {int(year)}: OK (no 0 or NaN values)")

if __name__ == "__main__":
    verify_static_features_zero()
