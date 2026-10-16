#!/usr/bin/env python3
"""
Quick script to check raster statistics and identify invalid values
"""

import numpy as np
import rasterio
import sys
from pathlib import Path

def check_raster_stats(file_path, sample_pixels=10000):
    """Check basic statistics of a raster file"""
    print(f"Analyzing: {Path(file_path).name}")
    
    with rasterio.open(file_path) as src:
        print(f"  Dimensions: {src.count} bands, {src.height}H x {src.width}W")
        print(f"  Data type: {src.dtypes[0]}")
        print(f"  Nodata value: {src.nodata}")
        
        # Read a sample of the data
        total_pixels = src.height * src.width
        if total_pixels > sample_pixels:
            # Read a smaller sample for speed
            sample_window = rasterio.windows.Window(0, 0, 
                                                  min(100, src.width), 
                                                  min(100, src.height))
            data = src.read(window=sample_window)
        else:
            data = src.read()
            
        print(f"  Sample shape: {data.shape}")
        print(f"  Value range: {np.nanmin(data):.6f} to {np.nanmax(data):.6f}")
        
        # Check for invalid values
        nan_count = np.sum(np.isnan(data))
        inf_count = np.sum(np.isinf(data))
        nodata_count = 0
        if src.nodata is not None:
            nodata_count = np.sum(data == src.nodata)
            
        total_invalid = nan_count + inf_count + nodata_count
        total_values = data.size
        
        print(f"  NaN values: {nan_count} ({100*nan_count/total_values:.2f}%)")
        print(f"  Inf values: {inf_count} ({100*inf_count/total_values:.2f}%)")
        print(f"  Nodata values: {nodata_count} ({100*nodata_count/total_values:.2f}%)")
        print(f"  Total invalid: {total_invalid} ({100*total_invalid/total_values:.2f}%)")
        
        if total_invalid == 0:
            print("  → No invalid values found - interpolation may not be needed")
        else:
            print("  → Invalid values found - interpolation recommended")
            
        print()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python check_raster_stats.py <raster_file>")
        sys.exit(1)
        
    check_raster_stats(sys.argv[1])