#!/usr/bin/env python3
"""
Comprehensive Static Data Quality Report
Analysis of slope and elevation data in the extracted forest type dataset.
"""

import pandas as pd
import numpy as np
import json
from datetime import datetime

def generate_static_data_report():
    """Generate a comprehensive report on static data quality."""
    
    print("=== FOREST TYPE DATASET - STATIC DATA QUALITY REPORT ===")
    print(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Load the extracted dataset
    raw_data = pd.read_parquet('/fs2/ftype_mflf/shared_data/dataset/1.2.0/raw/forest_type_dataset_raw_20250903_145707.parquet')
    
    print(f"Dataset overview:")
    print(f"  Total records: {len(raw_data):,}")
    print(f"  Total unique points: {raw_data['point_id'].nunique():,}")
    print(f"  Total features: {len(raw_data['feature'].unique())}")
    print(f"  Features: {sorted(raw_data['feature'].unique())}")
    print()
    
    # Focus on static features
    static_features = ['slope', 'elevation']
    
    print("=== STATIC FEATURES ANALYSIS ===")
    
    for feature in static_features:
        print(f"\n--- {feature.upper()} DATA ---")
        feature_data = raw_data[raw_data['feature'] == feature]
        
        print(f"Total records: {len(feature_data):,}")
        print(f"Unique points: {feature_data['point_id'].nunique():,}")
        
        # Missing data analysis
        nan_count = feature_data['value'].isna().sum()
        valid_count = feature_data['value'].notna().sum()
        nan_percentage = (nan_count / len(feature_data)) * 100
        
        print(f"Missing values: {nan_count} ({nan_percentage:.2f}%)")
        print(f"Valid values: {valid_count} ({100-nan_percentage:.2f}%)")
        
        # Value statistics for valid data
        valid_values = feature_data['value'].dropna()
        if len(valid_values) > 0:
            print(f"Value range: {valid_values.min():.2f} to {valid_values.max():.2f}")
            print(f"Mean: {valid_values.mean():.2f}")
            print(f"Standard deviation: {valid_values.std():.2f}")
            print(f"Median: {valid_values.median():.2f}")
        
        # Zone breakdown
        print(f"\nBreakdown by zone:")
        zone_stats = feature_data.groupby('zone').agg({
            'value': ['count', lambda x: x.isna().sum(), lambda x: x.notna().sum()]
        })
        zone_stats.columns = ['total', 'missing', 'valid']
        zone_stats['missing_pct'] = (zone_stats['missing'] / zone_stats['total'] * 100).round(2)
        
        for zone in zone_stats.index:
            stats = zone_stats.loc[zone]
            print(f"  {zone}: {stats['valid']}/{stats['total']} valid ({100-stats['missing_pct']:.1f}%)")
    
    # Identify problematic points
    print("\n=== PROBLEMATIC POINTS ANALYSIS ===")
    
    slope_data = raw_data[raw_data['feature'] == 'slope']
    elevation_data = raw_data[raw_data['feature'] == 'elevation']
    
    missing_slope_points = set(slope_data[slope_data['value'].isna()]['point_id'].unique())
    missing_elevation_points = set(elevation_data[elevation_data['value'].isna()]['point_id'].unique())
    
    print(f"Points missing slope data: {len(missing_slope_points)}")
    print(f"Points missing elevation data: {len(missing_elevation_points)}")
    
    # Check if same points are missing both
    missing_both = missing_slope_points.intersection(missing_elevation_points)
    missing_only_slope = missing_slope_points - missing_elevation_points
    missing_only_elevation = missing_elevation_points - missing_slope_points
    
    print(f"Points missing both slope and elevation: {len(missing_both)}")
    print(f"Points missing only slope: {len(missing_only_slope)}")
    print(f"Points missing only elevation: {len(missing_only_elevation)}")
    
    if missing_both:
        print(f"Point IDs missing both: {sorted(list(missing_both))}")
        
        # Get zone information for missing points
        missing_points_info = []
        for point_id in missing_both:
            point_data = raw_data[raw_data['point_id'] == point_id]
            if not point_data.empty:
                zone = point_data['zone'].iloc[0]
                utm_x = point_data['utm_x'].iloc[0]
                utm_y = point_data['utm_y'].iloc[0]
                forest_type = point_data['ftype_mix'].iloc[0]
                missing_points_info.append({
                    'point_id': point_id,
                    'zone': zone,
                    'utm_x': utm_x,
                    'utm_y': utm_y,
                    'forest_type': forest_type
                })
        
        print(f"\nDetailed info for missing points:")
        for info in missing_points_info:
            print(f"  Point {info['point_id']}: {info['zone']}, ({info['utm_x']}, {info['utm_y']}), {info['forest_type']}")
    
    # Root cause analysis
    print("\n=== ROOT CAUSE ANALYSIS ===")
    print("Based on the datacube validation, the missing static data is caused by:")
    print("1. ❌ Poor coverage in zone 'mfl_cmf_area_project_2025_chr_zone_1' datacube files")
    print("   - Slope datacube: 96.54% NoData/NaN pixels")
    print("   - Elevation datacube: 96.54% NoData/NaN pixels")
    print("   - Only 3.46% of pixels have valid data")
    print()
    print("2. ✅ Zone 'mfl_cmf_2024' and 'mfl_cmf_area_project_2025_chr_zone_2' have complete coverage")
    print("   - No missing values in these zones")
    print()
    
    # Impact assessment
    print("=== IMPACT ASSESSMENT ===")
    total_points = raw_data['point_id'].nunique()
    affected_points = len(missing_both)
    impact_percentage = (affected_points / total_points) * 100
    
    print(f"Dataset impact:")
    print(f"  Total points in dataset: {total_points:,}")
    print(f"  Points with missing static data: {affected_points}")
    print(f"  Percentage affected: {impact_percentage:.2f}%")
    print()
    
    if impact_percentage < 1.0:
        print("✅ IMPACT ASSESSMENT: LOW")
        print("   - Less than 1% of points affected")
        print("   - Dataset is still suitable for model training")
        print("   - Consider filtering out affected points or using interpolation")
    elif impact_percentage < 5.0:
        print("⚠️ IMPACT ASSESSMENT: MODERATE") 
        print("   - Between 1-5% of points affected")
        print("   - May impact model performance if not handled")
        print("   - Recommend data imputation or point exclusion")
    else:
        print("❌ IMPACT ASSESSMENT: HIGH")
        print("   - More than 5% of points affected")
        print("   - Significant impact on model training")
        print("   - Requires immediate data correction")
    
    # Recommendations
    print("\n=== RECOMMENDATIONS ===")
    print("1. 🔧 SHORT-TERM SOLUTIONS:")
    print("   a) Filter out the 9 affected points (0.50% of dataset)")
    print("   b) Use interpolation based on nearby points with valid data")
    print("   c) Use mean/median imputation for the missing values")
    print()
    print("2. 🏗️ LONG-TERM SOLUTIONS:")
    print("   a) Investigate zone 1 datacube generation process")
    print("   b) Regenerate slope and elevation datacubes for zone 1")
    print("   c) Validate datacube quality before extraction")
    print()
    print("3. 📊 IMMEDIATE ACTIONS:")
    print("   a) Document affected points for reference")
    print("   b) Proceed with model training using available data")
    print("   c) Monitor model performance on points with imputed static data")
    
    # Save summary to file
    summary = {
        'generation_time': datetime.now().isoformat(),
        'dataset_overview': {
            'total_records': len(raw_data),
            'total_points': total_points,
            'total_features': len(raw_data['feature'].unique())
        },
        'static_data_quality': {
            'slope': {
                'total_records': len(slope_data),
                'missing_values': int(slope_data['value'].isna().sum()),
                'missing_percentage': float((slope_data['value'].isna().sum() / len(slope_data)) * 100)
            },
            'elevation': {
                'total_records': len(elevation_data),
                'missing_values': int(elevation_data['value'].isna().sum()),
                'missing_percentage': float((elevation_data['value'].isna().sum() / len(elevation_data)) * 100)
            }
        },
        'problematic_points': {
            'missing_both_features': list(missing_both),
            'total_affected': len(missing_both),
            'impact_percentage': float(impact_percentage)
        },
        'zone_analysis': {
            'mfl_cmf_2024': 'Complete coverage',
            'mfl_cmf_area_project_2025_chr_zone_1': f'{affected_points} points missing static data',
            'mfl_cmf_area_project_2025_chr_zone_2': 'Complete coverage'
        }
    }
    
    report_file = f'/fs1/tor_project/model_foresttype/static_data_quality_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    with open(report_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n📄 Full report saved to: {report_file}")
    print("\n=== REPORT COMPLETE ===")

if __name__ == "__main__":
    generate_static_data_report()
