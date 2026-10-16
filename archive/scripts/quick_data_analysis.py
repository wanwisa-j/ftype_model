#!/usr/bin/env python3
"""
Quick Data Analysis - Forest Type Dataset
Shows basic statistics and simple visualizations for the test dataset
"""

import pandas as pd
import numpy as np
from pathlib import Path

def analyze_test_data():
    """Analyze the test dataset and show key statistics"""
    
    # Load the most recent test data
    data_dir = Path('/fs2/ftype_mflf/shared_data/dataset/1.2.0/test/raw')
    parquet_files = list(data_dir.glob('*.parquet'))
    if not parquet_files:
        print("No test data found!")
        return
    
    latest_file = max(parquet_files, key=lambda x: x.stat().st_mtime)
    print(f"📊 Analyzing: {latest_file.name}")
    print("=" * 60)
    
    # Load data
    df = pd.read_parquet(latest_file)
    
    # Basic info
    print(f"📈 Dataset Overview:")
    print(f"   Total records: {len(df):,}")
    print(f"   Unique points: {df['point_id'].nunique():,}")
    print(f"   Features: {', '.join(sorted(df['feature'].unique()))}")
    print(f"   Zone: {df['zone'].unique()[0]}")
    
    # Forest type distribution
    print(f"\n🌲 Forest Type Distribution:")
    print("   Mix Types:")
    mix_counts = df['ftype_mix'].value_counts().head(8)
    for ftype, count in mix_counts.items():
        percentage = (count / len(df)) * 100
        print(f"     {ftype:10s}: {count:5,d} ({percentage:4.1f}%)")
    
    print("   Non-Mix Types:")
    nonmix_counts = df['ftype_nonmix'].value_counts().head(8)
    for ftype, count in nonmix_counts.items():
        percentage = (count / len(df)) * 100
        print(f"     {ftype:10s}: {count:5,d} ({percentage:4.1f}%)")
    
    # LAI analysis
    lai_data = df[df['feature'] == 'lai']
    if not lai_data.empty:
        print(f"\n🍃 LAI Analysis:")
        print(f"   Records: {len(lai_data):,}")
        print(f"   Mean: {lai_data['value'].mean():.3f}")
        print(f"   Std:  {lai_data['value'].std():.3f}")
        print(f"   Min:  {lai_data['value'].min():.3f}")
        print(f"   Max:  {lai_data['value'].max():.3f}")
        
        # LAI by forest type
        print(f"\n   LAI by Forest Mix Type:")
        lai_by_type = lai_data.groupby('ftype_mix')['value'].agg(['count', 'mean', 'std'])
        lai_by_type = lai_by_type.sort_values('count', ascending=False).head(6)
        for ftype, row in lai_by_type.iterrows():
            print(f"     {ftype:10s}: {row['mean']:5.3f} ± {row['std']:5.3f} (n={row['count']:,})")
    
    # NDVI analysis
    ndvi_data = df[df['feature'] == 'ndvi']
    if not ndvi_data.empty:
        print(f"\n🌿 NDVI Analysis:")
        print(f"   Records: {len(ndvi_data):,}")
        print(f"   Mean: {ndvi_data['value'].mean():.3f}")
        print(f"   Std:  {ndvi_data['value'].std():.3f}")
        print(f"   Min:  {ndvi_data['value'].min():.3f}")
        print(f"   Max:  {ndvi_data['value'].max():.3f}")
        
        # NDVI by forest type
        print(f"\n   NDVI by Forest Mix Type:")
        ndvi_by_type = ndvi_data.groupby('ftype_mix')['value'].agg(['count', 'mean', 'std'])
        ndvi_by_type = ndvi_by_type.sort_values('count', ascending=False).head(6)
        for ftype, row in ndvi_by_type.iterrows():
            print(f"     {ftype:10s}: {row['mean']:5.3f} ± {row['std']:5.3f} (n={row['count']:,})")
    
    # Static features
    static_features = df[df['feature'].isin(['elevation', 'slope'])]
    if not static_features.empty:
        print(f"\n🏔️  Static Features:")
        for feature in ['elevation', 'slope']:
            feat_data = df[df['feature'] == feature]['value']
            if not feat_data.empty:
                print(f"   {feature.capitalize():10s}: {feat_data.mean():7.1f} ± {feat_data.std():5.1f} "
                      f"[{feat_data.min():.1f} - {feat_data.max():.1f}]")
    
    # Data quality check
    print(f"\n🔍 Data Quality:")
    print(f"   Missing values: {df['value'].isna().sum():,}")
    print(f"   Zero values: {(df['value'] == 0).sum():,}")
    print(f"   Negative values: {(df['value'] < 0).sum():,}")
    
    # Simple histogram for LAI values
    if not lai_data.empty:
        print(f"\n📊 LAI Distribution (histogram):")
        values = lai_data['value'].values
        hist, bins = np.histogram(values, bins=20)
        max_count = max(hist)
        scale = 50 / max_count  # Scale to fit in 50 characters
        
        for i in range(len(hist)):
            bar_length = int(hist[i] * scale)
            bar = '█' * bar_length
            print(f"   {bins[i]:4.2f}-{bins[i+1]:4.2f}: {bar} ({hist[i]:,})")
    
    print(f"\n✅ Analysis complete!")
    print(f"💾 Plots saved in: /fs1/tor_project/model_foresttype/plots/")

if __name__ == "__main__":
    analyze_test_data()
