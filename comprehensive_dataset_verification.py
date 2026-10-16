#!/usr/bin/env python3
"""
Comprehensive Forest Type Dataset Verification Script

This script performs complete verification of the forest type classification dataset:
- Verifies all features are complete across all zones
- Checks target variables (ftype_mix, ftype_nonmix) completeness
- Validates temporal structure (21 bands/timesteps for time-series features)
- Analyzes dataset shape and consistency
"""

import pandas as pd
import numpy as np
import os
from pathlib import Path
from collections import defaultdict

def comprehensive_dataset_verification():
    """Complete verification of the forest type dataset."""
    
    print("=== COMPREHENSIVE FOREST TYPE DATASET VERIFICATION ===")
    
    # Load the dataset
    print("Loading dataset...")
    df = pd.read_parquet('/fs2/ftype_mflf/shared_data/dataset/1.2.0/raw/forest_type_dataset_raw_20250909_102320.parquet')
    
    print(f"Dataset shape: {df.shape}")
    print(f"Dataset columns: {list(df.columns)}")
    
    # Basic dataset overview
    print(f"\n=== DATASET OVERVIEW ===")
    unique_zones = sorted(df['zone'].unique())
    unique_features = sorted(df['feature'].unique())
    unique_years = sorted([y for y in df['year'].unique() if not pd.isna(y)])
    unique_points = df['point_id'].nunique()
    
    print(f"Unique zones ({len(unique_zones)}): {unique_zones}")
    print(f"Unique features ({len(unique_features)}): {unique_features}")
    print(f"Unique years ({len(unique_years)}): {unique_years}")
    print(f"Unique points: {unique_points:,}")
    
    # Define feature categories
    static_features = ['elevation', 'slope']
    temporal_features = [f for f in unique_features if f not in static_features]

    print(f"Static features: {static_features}")
    print(f"Temporal features: {temporal_features}")
    
    # Check for missing values
    print(f"\n=== MISSING VALUES ANALYSIS ===")
    missing_data = df[df['value'].isna()]
    total_missing = len(missing_data)
    total_records = len(df)
    
    print(f"Total records: {total_records:,}")
    print(f"Records with missing values: {total_missing:,} ({total_missing/total_records*100:.2f}%)")
    
    if total_missing == 0:
        print("✅ PERFECT! No missing values found in the dataset!")
    else:
        print("⚠️  Missing values detected - analyzing...")
        
        # Missing by feature
        missing_by_feature = missing_data.groupby('feature').size().sort_values(ascending=False)
        print(f"\nMissing records by feature:")
        for feature, count in missing_by_feature.items():
            print(f"  {feature}: {count:,} records")
    
    # Verify target variables
    print(f"\n=== TARGET VARIABLES VERIFICATION ===")
    target_columns = ['ftype_mix', 'ftype_nonmix']
    
    for target in target_columns:
        if target in df.columns:
            target_missing = df[target].isna().sum()
            target_total = len(df)
            unique_values = df[target].dropna().nunique()
            value_counts = df[target].value_counts().sort_index()
            
            print(f"\n{target}:")
            print(f"  Missing values: {target_missing:,} ({target_missing/target_total*100:.2f}%)")
            print(f"  Unique values: {unique_values}")
            print(f"  Value distribution:")
            for value, count in value_counts.items():
                print(f"    {value}: {count:,} records")
        else:
            print(f"❌ Target column '{target}' not found in dataset!")
    
    # Verify feature completeness by zone
    print(f"\n=== FEATURE COMPLETENESS BY ZONE ===")
    
    for zone in unique_zones:
        zone_data = df[df['zone'] == zone]
        zone_points = zone_data['point_id'].nunique()
        
        print(f"\n--- Zone: {zone} ({zone_points:,} points) ---")
        
        # Check each feature
        for feature in unique_features:
            feature_data = zone_data[zone_data['feature'] == feature]
            
            if feature in static_features:
                # Static features should have 1 record per point
                expected_records = zone_points
                actual_records = len(feature_data)
                missing_records = feature_data['value'].isna().sum()
                
                status = "✅" if missing_records == 0 and actual_records == expected_records else "❌"
                print(f"  {feature} (static): {status} {actual_records:,}/{expected_records:,} records, {missing_records:,} missing")
                
            else:
                # Temporal features - need to check by year
                feature_years = sorted([y for y in feature_data['year'].unique() if not pd.isna(y)])
                total_missing = feature_data['value'].isna().sum()
                
                if len(feature_years) == 0:
                    print(f"  {feature} (temporal): ❌ No data for any year")
                else:
                    status = "✅" if total_missing == 0 else "⚠️ "
                    print(f"  {feature} (temporal): {status} Years: {feature_years}, {total_missing:,} missing values")
    
    # Verify temporal structure (21 bands check)
    print(f"\n=== TEMPORAL STRUCTURE VERIFICATION (21 BANDS CHECK) ===")
    
    temporal_structure = {}
    
    for zone in unique_zones:
        zone_data = df[df['zone'] == zone]
        print(f"\n--- Zone: {zone} ---")
        
        for feature in temporal_features:
            for year in unique_years:
                feature_year_data = zone_data[
                    (zone_data['feature'] == feature) & 
                    (zone_data['year'] == year)
                ]
                
                if len(feature_year_data) > 0:
                    # Group by point to check bands per point
                    bands_per_point = feature_year_data.groupby('point_id').size()
                    
                    if len(bands_per_point) > 0:
                        min_bands = bands_per_point.min()
                        max_bands = bands_per_point.max()
                        mean_bands = bands_per_point.mean()
                        points_with_21_bands = (bands_per_point == 21).sum()
                        total_points = len(bands_per_point)
                        
                        status = "✅" if min_bands == max_bands == 21 else "⚠️ "
                        print(f"  {feature} {year}: {status} Points: {total_points:,}, Bands: {min_bands}-{max_bands} (avg: {mean_bands:.1f}), 21-band points: {points_with_21_bands}")
                        
                        # Store for summary
                        key = f"{zone}_{feature}_{year}"
                        temporal_structure[key] = {
                            'zone': zone,
                            'feature': feature,
                            'year': year,
                            'total_points': total_points,
                            'min_bands': min_bands,
                            'max_bands': max_bands,
                            'points_with_21_bands': points_with_21_bands,
                            'complete': min_bands == max_bands == 21
                        }
    
    # Summary of temporal structure issues
    print(f"\n=== TEMPORAL STRUCTURE SUMMARY ===")
    
    complete_temporal = 0
    incomplete_temporal = 0
    
    for key, info in temporal_structure.items():
        if info['complete']:
            complete_temporal += 1
        else:
            incomplete_temporal += 1
            print(f"⚠️  {info['zone']} {info['feature']} {info['year']}: {info['min_bands']}-{info['max_bands']} bands per point")
    
    total_temporal = complete_temporal + incomplete_temporal
    
    print(f"\nTemporal structure completeness:")
    print(f"  Complete (21 bands): {complete_temporal:,}/{total_temporal:,} ({complete_temporal/total_temporal*100:.1f}%)")
    print(f"  Incomplete: {incomplete_temporal:,}/{total_temporal:,} ({incomplete_temporal/total_temporal*100:.1f}%)")
    
    # Dataset shape analysis
    print(f"\n=== DATASET SHAPE ANALYSIS ===")
    print(f"Expected structure: [zone, year, feature, point] combinations")
    
    # Group by key dimensions
    shape_analysis = df.groupby(['zone', 'feature']).agg({
        'point_id': 'nunique',
        'year': lambda x: len([y for y in x.unique() if not pd.isna(y)]),
        'value': ['count', lambda x: x.isna().sum()]
    }).round(2)
    
    shape_analysis.columns = ['unique_points', 'unique_years', 'total_records', 'missing_values']
    
    print(f"\nShape analysis by zone and feature:")
    print(shape_analysis)
    
    # Final summary
    print(f"\n=== FINAL DATASET QUALITY SUMMARY ===")
    
    quality_checks = {
        'No missing values': total_missing == 0,
        'All zones present': len(unique_zones) >= 3,  # Expecting at least 3 zones
        'All features present': len(unique_features) >= 10,  # Expecting multiple features
        'Target variables complete': all(df[col].isna().sum() == 0 for col in target_columns if col in df.columns),
        'Temporal structure complete': incomplete_temporal == 0
    }
    
    passed_checks = sum(quality_checks.values())
    total_checks = len(quality_checks)
    
    print(f"\nQuality Checks Passed: {passed_checks}/{total_checks}")
    
    for check, passed in quality_checks.items():
        status = "✅" if passed else "❌"
        print(f"  {status} {check}")
    
    if passed_checks == total_checks:
        print(f"\n🎉 DATASET VERIFICATION COMPLETE - ALL CHECKS PASSED!")
        print(f"Dataset is ready for forest type classification training!")
    else:
        print(f"\n⚠️  Dataset has {total_checks - passed_checks} quality issues that need attention.")
    
    return df, quality_checks

def check_datacube_availability():
    """Check datacube file availability."""
    
    print(f"\n=== DATACUBE AVAILABILITY CHECK ===")
    
    base_path = Path('/fs2/ftype_mflf/shared_data/datacube')
    
    if not base_path.exists():
        print(f"❌ Datacube base path not found: {base_path}")
        return
    
    years = [d.name for d in base_path.iterdir() if d.is_dir()]
    print(f"Available datacube years: {sorted(years)}")
    
    datacube_status = {}
    
    for year in sorted(years):
        year_path = base_path / year
        versions = [d.name for d in year_path.iterdir() if d.is_dir()]
        
        for version in sorted(versions):
            version_path = year_path / version
            zones = [d.name for d in version_path.iterdir() if d.is_dir()]
            
            for zone in zones:
                zone_path = version_path / zone
                
                # Check key features
                features_to_check = {
                    'slope': 'slope/slope_data.tif',
                    'elevation': 'elevation/elevation_data.tif',
                    'vv_2020': 'vv/cube_s1_vv_2020_W4before_to_W16after_21bands.tif',
                    'vh_2020': 'vh/cube_s1_vh_2020_W4before_to_W16after_21bands.tif',
                    'vv_2023': 'vv/cube_s1_vv_2023_W4before_to_W16after_21bands.tif',
                    'vh_2023': 'vh/cube_s1_vh_2023_W4before_to_W16after_21bands.tif'
                }
                
                zone_status = {}
                for feature_name, feature_path in features_to_check.items():
                    full_path = zone_path / feature_path
                    zone_status[feature_name] = full_path.exists()
                
                datacube_status[f"{year}/{version}/{zone}"] = zone_status
    
    # Summary of critical datacube files
    print(f"\nKey datacube file availability:")
    critical_zones = ['mfl_cmf_area_project_2025_chr_zone_1', 'mfl_cmf_area_project_2025_chr_zone_2']
    
    for zone_name in critical_zones:
        print(f"\n{zone_name}:")
        found_zone = False
        
        for key, status in datacube_status.items():
            if zone_name in key:
                found_zone = True
                print(f"  {key}:")
                for feature, available in status.items():
                    icon = "✅" if available else "❌"
                    print(f"    {icon} {feature}")
        
        if not found_zone:
            print(f"  ❌ Zone not found in datacube")

if __name__ == "__main__":
    # Execute the verification
    print("Starting comprehensive dataset verification...")
    try:
        df, quality_checks = comprehensive_dataset_verification()
        check_datacube_availability()
        print("\n" + "="*60)
        print("VERIFICATION COMPLETED SUCCESSFULLY!")
        print("="*60)
    except Exception as e:
        print(f"❌ Error during verification: {e}")
        import traceback
        traceback.print_exc()