#!/usr/bin/env python3
"""
Simple Data Consistency Check
=============================

Check if training data (dataset_builder) and prediction data come from same sources.
"""

import sys
import os
import numpy as np
import pandas as pd

sys.path.insert(0, '/fs1/tor_project/model_foresttype/src')
from dataset_builder import ForestTypeDatasetBuilder

def main():
    print("🔍 SIMPLE DATA CONSISTENCY CHECK")
    print("=" * 50)
    
    # 1. Check what's in the training dataset
    print("\n📊 TRAINING DATASET (via dataset_builder):")
    
    builder = ForestTypeDatasetBuilder()
    zone = 'mfl_cmf_area_project_2025_chr_zone_1'
    
    # Extract training data for zone_1 (2025 only)
    dataset = builder.create_dataset(
        zones=[zone],
        years=[2025],
        features=['vv', 'vh', 'ndvi'],
        target='ftype_nonmix',
        return_point_info=True
    )
    
    print(f"   Zone: {zone}")
    print(f"   Samples: {dataset['target'].shape[0]}")
    
    # Check target distribution
    targets = dataset['target'].ravel()
    unique, counts = np.unique(targets, return_counts=True)
    print(f"   Target distribution:")
    for cls, count in zip(unique, counts):
        print(f"      {cls}: {count} samples")
    
    # Check coordinates
    if 'point_info' in dataset and len(dataset['point_info']) > 0:
        coords = [(p['utm_x'], p['utm_y']) for p in dataset['point_info'][:10]]
        print(f"   Sample coordinates (first 10): {coords}")
        
        # Check coordinate bounds
        all_x = [p['utm_x'] for p in dataset['point_info']]
        all_y = [p['utm_y'] for p in dataset['point_info']]
        print(f"   X range: {min(all_x):.0f} to {max(all_x):.0f}")
        print(f"   Y range: {min(all_y):.0f} to {max(all_y):.0f}")
    
    # Check feature statistics
    vv_data = dataset['vv_feature']
    vh_data = dataset['vh_feature']
    ndvi_data = dataset['ndvi_feature']
    
    print(f"\n   Feature statistics:")
    print(f"      VV: shape={vv_data.shape}, mean={np.nanmean(vv_data):.6f}, std={np.nanstd(vv_data):.6f}")
    print(f"      VH: shape={vh_data.shape}, mean={np.nanmean(vh_data):.6f}, std={np.nanstd(vh_data):.6f}")
    print(f"      NDVI: shape={ndvi_data.shape}, mean={np.nanmean(ndvi_data):.6f}, std={np.nanstd(ndvi_data):.6f}")
    
    # 2. Check what prediction pipeline should access
    print(f"\n📁 PREDICTION DATA SOURCES:")
    datacube_base = '/fs2/ftype_mflf/shared_data/datacube/2025/1.2.0'
    zone_path = f"{datacube_base}/{zone}"
    
    print(f"   Zone directory: {zone_path}")
    print(f"   Directory exists: {os.path.exists(zone_path)}")
    
    if os.path.exists(zone_path):
        features = ['vv', 'vh', 'ndvi']
        for feature in features:
            feature_dir = f"{zone_path}/{feature}"
            print(f"   Feature '{feature}' dir: {os.path.exists(feature_dir)}")
            
            if os.path.exists(feature_dir):
                files = os.listdir(feature_dir)
                tif_files = [f for f in files if f.endswith('.tif')]
                print(f"      TIF files: {len(tif_files)}")
                for tif_file in tif_files:
                    file_path = f"{feature_dir}/{tif_file}"
                    file_size = os.path.getsize(file_path) / (1024*1024)  # MB
                    print(f"         {tif_file}: {file_size:.1f} MB")
    
    # 3. Key questions
    print(f"\n❓ KEY INVESTIGATION QUESTIONS:")
    print(f"   1. Does training dataset use same raster files as prediction?")
    print(f"   2. Are coordinate systems consistent?")
    print(f"   3. Are the 312 training points a subset of full prediction area?")
    print(f"   4. Is scaling applied consistently in both pipelines?")
    
    # 4. Check if the 312 training points are just the labeled points
    print(f"\n🎯 HYPOTHESIS CHECK:")
    print(f"   Training samples: {dataset['target'].shape[0]}")
    print(f"   Expected: Should be 312 (matching your validation)")
    
    if dataset['target'].shape[0] == 312:
        print(f"   ✅ CONFIRMED: Training uses exactly 312 points")
        print(f"   💡 INSIGHT: These are likely the same 312 labeled points")
        print(f"   🚨 ISSUE: Model trained only on labeled points, not full spatial area")
        print(f"   ")
        print(f"   🔍 ROOT CAUSE HYPOTHESIS:")
        print(f"   - Training: Uses 312 labeled points from zone_1")
        print(f"   - Spatial validation: Uses same 312 labeled points") 
        print(f"   - Full prediction: Samples entire raster (millions of pixels)")
        print(f"   - RESULT: Model learned patterns specific to labeled locations")
        print(f"             but these don't represent the full area")
    
    print(f"\n💡 NEXT STEPS:")
    print(f"   1. Verify if training dataset is limited to labeled point locations")
    print(f"   2. Check if full prediction samples different spatial distribution") 
    print(f"   3. Consider training on spatial samples beyond just labeled points")
    print(f"   4. Implement stratified spatial sampling for training")

if __name__ == "__main__":
    main()