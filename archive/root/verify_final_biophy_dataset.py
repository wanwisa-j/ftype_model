#!/usr/bin/env python3
"""
Verify the final biophysical dataset after merging 2021 data
"""

import pandas as pd
import numpy as np

# Load the final dataset
dataset_path = "/fs2/ftype_mflf/shared_data/dataset/1.2.0/raw/forest_type_dataset_raw_20250903_145707.cleaned_with_new_biophy_20250909_071747.parquet"
print(f"Loading dataset: {dataset_path}")
df = pd.read_parquet(dataset_path)

print(f"\n=== FINAL DATASET VERIFICATION ===")
print(f"Total records: {len(df):,}")
print(f"Dataset shape: {df.shape}")

# Check zones
zones = df['zone'].unique()
print(f"\nZones in dataset: {sorted(zones)}")

# Check years
years = sorted(df['year'].unique())
print(f"Years in dataset: {years}")

# Check features available in the dataset
features_available = df['feature'].unique()
print(f"Features available: {sorted(features_available)}")

print(f"\n=== BIOPHYSICAL DATA COMPLETENESS BY ZONE AND YEAR ===")
biophy_features = ['lai', 'fapar', 'fcover']
problem_zones = ['mfl_cmf_area_project_2025_chr_zone_1', 'mfl_cmf_area_project_2025_chr_zone_2']

for zone in problem_zones:
    zone_data = df[df['zone'] == zone]
    print(f"\n📍 Zone: {zone}")
    print(f"   Total records: {len(zone_data):,}")
    
    for year in sorted(zone_data['year'].dropna().unique()):
        year_data = zone_data[zone_data['year'] == year]
        print(f"   Year {int(year)}: {len(year_data):,} records")
        
        for feature in biophy_features:
            feature_data = year_data[year_data['feature'] == feature]
            missing = feature_data['value'].isna().sum()
            total = len(feature_data)
            pct_complete = ((total - missing) / total * 100) if total > 0 else 0
            status = "✅" if missing == 0 and total > 0 else "❌"
            print(f"     {status} {feature}: {total - missing:,}/{total:,} ({pct_complete:.1f}% complete)")

print(f"\n=== DATA QUALITY SUMMARY ===")
for zone in problem_zones:
    zone_data = df[df['zone'] == zone]
    print(f"\n🎯 {zone}:")
    
    for feature in biophy_features:
        feature_data = zone_data[zone_data['feature'] == feature]
        total_records = len(feature_data)
        missing_records = feature_data['value'].isna().sum() if total_records > 0 else 0
        pct_complete = ((total_records - missing_records) / total_records * 100) if total_records > 0 else 0
        
        print(f"   {feature.upper()}: {total_records - missing_records:,}/{total_records:,} ({pct_complete:.2f}% complete)")
        
        if missing_records == 0 and total_records > 0:
            values = feature_data['value'].dropna()
            if len(values) > 0:
                print(f"     Range: [{values.min():.3f}, {values.max():.3f}]")
                print(f"     Mean: {values.mean():.3f}, Std: {values.std():.3f}")

# Check for any remaining missing data in problem zones
print(f"\n=== MISSING DATA CHECK ===")
for zone in problem_zones:
    zone_data = df[df['zone'] == zone]
    print(f"\n{zone}:")
    for feature in biophy_features:
        feature_data = zone_data[zone_data['feature'] == feature]
        missing = feature_data['value'].isna().sum() if len(feature_data) > 0 else 0
        total = len(feature_data)
        if missing > 0:
            print(f"  ❌ {feature}: {missing:,}/{total:,} missing values")
        elif total > 0:
            print(f"  ✅ {feature}: No missing values ({total:,} records)")
        else:
            print(f"  ⚠️ {feature}: No data found")

# Verify 2021 data specifically
print(f"\n=== 2021 DATA VERIFICATION ===")
df_2021 = df[df['year'] == 2021]
print(f"Total 2021 records: {len(df_2021):,}")

for zone in problem_zones:
    zone_2021 = df_2021[df_2021['zone'] == zone]
    if len(zone_2021) > 0:
        print(f"\n{zone} (2021): {len(zone_2021):,} records")
        for feature in biophy_features:
            feature_2021 = zone_2021[zone_2021['feature'] == feature]
            missing = feature_2021['value'].isna().sum() if len(feature_2021) > 0 else 0
            total = len(feature_2021)
            status = "✅" if missing == 0 and total > 0 else "❌"
            print(f"  {status} {feature}: {total - missing:,}/{total:,} complete")
    else:
        print(f"\n❌ {zone}: No 2021 data found!")

print(f"\n🎉 Final dataset verification complete!")
print(f"📊 Total dataset size: {len(df):,} records across {len(zones)} zones and {len(years)} years")
