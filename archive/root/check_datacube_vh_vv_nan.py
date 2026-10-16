#!/usr/bin/env python3
"""
Check VV and VH datacube bands for NaN data in specific zones and years
"""

import os
import rasterio
import numpy as np
from pathlib import Path

def check_datacube_bands():
    """Check VV and VH bands for NaN data"""
    
    base_path = '/fs2/ftype_mflf/shared_data/datacube/2025/1.2.0'
    zones = ['mfl_cmf_area_project_2025_chr_zone_1', 'mfl_cmf_area_project_2025_chr_zone_2']
    years = [2020, 2023]
    features = ['vv', 'vh']
    
    print("=== DATACUBE VV/VH NaN ANALYSIS ===")
    print(f"Base path: {base_path}")
    print(f"Zones: {zones}")
    print(f"Years: {years}")
    print(f"Features: {features}")
    print()
    
    for zone in zones:
        print(f"--- ZONE: {zone} ---")
        
        for feature in features:
            print(f"  Feature: {feature}")
            
            for year in years:
                file_path = f'{base_path}/{zone}/{feature}/cube_s1_{feature}_{year}_W4before_to_W16after_21bands.tif'
                print(f"    Year {year}:")
                print(f"      File: {file_path}")
                
                if not os.path.exists(file_path):
                    print(f"      Status: ❌ FILE NOT FOUND")
                    continue
                
                try:
                    with rasterio.open(file_path) as src:
                        print(f"      Status: ✅ FILE EXISTS")
                        print(f"      Shape: {src.shape}")
                        print(f"      Bands: {src.count}")
                        print(f"      CRS: {src.crs}")
                        print(f"      Bounds: {src.bounds}")
                        
                        # Check each band for NaN values
                        total_pixels = src.width * src.height
                        bands_with_nan = []
                        
                        for band_idx in range(1, min(src.count + 1, 22)):  # Check up to 21 bands
                            band_data = src.read(band_idx)
                            
                            # Count NaN values
                            nan_count = np.isnan(band_data).sum()
                            nan_percentage = (nan_count / total_pixels) * 100
                            
                            if nan_count > 0:
                                bands_with_nan.append(band_idx)
                                print(f"        Band {band_idx:2d}: {nan_count:8d} NaN pixels ({nan_percentage:6.2f}%)")
                            else:
                                print(f"        Band {band_idx:2d}: No NaN pixels")
                        
                        if bands_with_nan:
                            print(f"      Summary: {len(bands_with_nan)} bands with NaN data")
                        else:
                            print(f"      Summary: All bands clean (no NaN)")
                            
                except Exception as e:
                    print(f"      Status: ❌ ERROR READING FILE")
                    print(f"      Error: {str(e)}")
                
                print()
        
        print()

if __name__ == "__main__":
    check_datacube_bands()