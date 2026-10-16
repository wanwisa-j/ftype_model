#!/usr/bin/env python3
"""
LAI Time Series Pattern Analysis
Shows temporal patterns in LAI data for different forest types
"""

import pandas as pd
import numpy as np
from pathlib import Path

def analyze_lai_patterns():
    """Analyze LAI temporal patterns"""
    
    # Load the test data
    data_dir = Path('/fs2/ftype_mflf/shared_data/dataset/1.2.0/test/raw')
    parquet_files = list(data_dir.glob('*.parquet'))
    latest_file = max(parquet_files, key=lambda x: x.stat().st_mtime)
    
    df = pd.read_parquet(latest_file)
    lai_data = df[df['feature'] == 'lai'].copy()
    
    if lai_data.empty:
        print("No LAI data found!")
        return
    
    print("🍃 LAI Time Series Pattern Analysis")
    print("=" * 50)
    
    # Check temporal distribution
    print(f"📅 Temporal Coverage:")
    print(f"   Total LAI records: {len(lai_data):,}")
    
    # Week distribution
    if 'week' in lai_data.columns:
        week_counts = lai_data['week'].value_counts().sort_index()
        print(f"   Weeks covered: {len(week_counts)} weeks")
        print(f"   Week range: {week_counts.index.min()} - {week_counts.index.max()}")
        
        print(f"\n📊 Weekly Distribution:")
        # Show weeks with data
        for week in sorted(week_counts.index):
            count = week_counts[week]
            bar_length = int(count / week_counts.max() * 30)
            bar = '█' * bar_length
            print(f"   Week {int(week):2d}: {bar} ({count:,} records)")
    
    # Sample some time series for specific forest types
    print(f"\n🌲 Sample Time Series by Forest Type:")
    
    top_forest_types = lai_data['ftype_mix'].value_counts().head(4).index
    
    for ftype in top_forest_types:
        print(f"\n   🌳 {ftype} Forest:")
        ftype_data = lai_data[lai_data['ftype_mix'] == ftype]
        
        # Get a few sample points
        sample_points = ftype_data['point_id'].unique()[:3]
        
        for point_id in sample_points:
            point_data = ftype_data[ftype_data['point_id'] == point_id].sort_values('week')
            if len(point_data) > 1:
                values = point_data['value'].values
                weeks = point_data['week'].values
                
                print(f"     Point {point_id}:")
                print(f"       Weeks: {list(weeks)}")
                print(f"       LAI:   {[f'{v:.2f}' for v in values]}")
                
                # Simple trend analysis
                if len(values) > 2:
                    trend = np.polyfit(range(len(values)), values, 1)[0]
                    if trend > 0.01:
                        trend_desc = "📈 increasing"
                    elif trend < -0.01:
                        trend_desc = "📉 decreasing"
                    else:
                        trend_desc = "➡️  stable"
                    print(f"       Trend: {trend_desc} ({trend:.3f} per step)")
    
    # Statistical summary by forest type and week
    print(f"\n📈 LAI Statistics by Forest Type:")
    lai_stats = lai_data.groupby('ftype_mix')['value'].agg([
        'count', 'mean', 'std', 'min', 'max'
    ]).round(3)
    
    lai_stats = lai_stats.sort_values('count', ascending=False)
    
    print(f"{'Type':<10} {'Count':<8} {'Mean':<6} {'Std':<6} {'Min':<6} {'Max':<6}")
    print("-" * 50)
    for ftype, row in lai_stats.head(8).iterrows():
        print(f"{ftype:<10} {row['count']:<8.0f} {row['mean']:<6.3f} {row['std']:<6.3f} "
              f"{row['min']:<6.3f} {row['max']:<6.3f}")
    
    # Check for seasonal patterns (if multiple weeks available)
    if 'week' in lai_data.columns and lai_data['week'].nunique() > 1:
        print(f"\n🗓️  Weekly LAI Patterns:")
        weekly_stats = lai_data.groupby('week')['value'].agg(['count', 'mean', 'std']).round(3)
        
        print(f"{'Week':<6} {'Count':<8} {'Mean LAI':<10} {'Std':<6}")
        print("-" * 35)
        for week, row in weekly_stats.iterrows():
            print(f"{week:<6.0f} {row['count']:<8.0f} {row['mean']:<10.3f} {row['std']:<6.3f}")
    
    # Data completeness analysis
    print(f"\n🔍 Data Completeness Analysis:")
    total_points = lai_data['point_id'].nunique()
    
    if 'week' in lai_data.columns:
        total_weeks = lai_data['week'].nunique()
        expected_records = total_points * total_weeks
        actual_records = len(lai_data)
        completeness = (actual_records / expected_records) * 100
        
        print(f"   Points: {total_points:,}")
        print(f"   Weeks: {total_weeks}")
        print(f"   Expected records: {expected_records:,}")
        print(f"   Actual records: {actual_records:,}")
        print(f"   Completeness: {completeness:.1f}%")
        
        # Points with complete time series
        point_weeks = lai_data.groupby('point_id')['week'].nunique()
        complete_points = (point_weeks == total_weeks).sum()
        print(f"   Points with complete time series: {complete_points:,} ({complete_points/total_points*100:.1f}%)")
    
    print(f"\n✅ LAI pattern analysis complete!")

if __name__ == "__main__":
    analyze_lai_patterns()
