#!/usr/bin/env python3
"""
Validate static datacube files (slope and elevation) for missing data issues.
Check specific points that have NaN values in the extracted dataset.
"""

import pandas as pd
import numpy as np
import rasterio
from rasterio.windows import from_bounds
import os
from pathlib import Path

def validate_static_datacube():
    """Check static datacube files for the points with missing data."""
    
    # Load the extracted dataset to get the problematic points
    print("Loading extracted dataset...")
    raw_data = pd.read_parquet('/fs2/ftype_mflf/shared_data/dataset/1.2.0/raw/forest_type_dataset_raw_20250903_145707.parquet')
    
    # Get points with missing slope/elevation data
    slope_data = raw_data[raw_data['feature'] == 'slope']
    elevation_data = raw_data[raw_data['feature'] == 'elevation']
    
    missing_slope_points = slope_data[slope_data['value'].isna()]['point_id'].unique()
    missing_elevation_points = elevation_data[elevation_data['value'].isna()]['point_id'].unique()
    
    print(f"Points with missing slope: {len(missing_slope_points)}")
    print(f"Points with missing elevation: {len(missing_elevation_points)}")
    print(f"Missing points: {list(missing_slope_points)}")
    
    # Get the UTM coordinates for these points
    missing_points_data = []
    for point_id in missing_slope_points:
        point_records = raw_data[raw_data['point_id'] == point_id]
        if not point_records.empty:
            utm_x = point_records['utm_x'].iloc[0]
            utm_y = point_records['utm_y'].iloc[0]
            zone = point_records['zone'].iloc[0]
            missing_points_data.append({
                'point_id': point_id,
                'utm_x': utm_x,
                'utm_y': utm_y,
                'zone': zone
            })
    
    print(f"\n=== CHECKING DATACUBE FILES ===")
    
    # Check slope and elevation files for the problematic zone
    zone = "mfl_cmf_area_project_2025_chr_zone_1"
    slope_file = f'/fs2/ftype_mflf/shared_data/datacube/2025/1.2.0/{zone}/slope/slope_data.tif'
    elevation_file = f'/fs2/ftype_mflf/shared_data/datacube/2025/1.2.0/{zone}/elevation/elevation_data.tif'
    
    print(f"Checking files for zone: {zone}")
    print(f"Slope file: {slope_file}")
    print(f"Elevation file: {elevation_file}")
    
    # Check if files exist
    if not os.path.exists(slope_file):
        print(f"❌ Slope file not found: {slope_file}")
        return
    if not os.path.exists(elevation_file):
        print(f"❌ Elevation file not found: {elevation_file}")
        return
    
    print("✅ Both files exist")
    
    # Check the files
    for file_path, feature_name in [(slope_file, 'slope'), (elevation_file, 'elevation')]:
        print(f"\n--- Checking {feature_name} file ---")
        
        try:
            with rasterio.open(file_path) as src:
                print(f"File info:")
                print(f"  Shape: {src.shape}")
                print(f"  CRS: {src.crs}")
                print(f"  Bounds: {src.bounds}")
                print(f"  NoData value: {src.nodata}")
                print(f"  Data type: {src.dtypes[0]}")
                
                # Check values at specific locations
                print(f"\nChecking values at missing points:")
                for point_data in missing_points_data[:5]:  # Check first 5 points
                    utm_x = point_data['utm_x']
                    utm_y = point_data['utm_y']
                    point_id = point_data['point_id']
                    
                    try:
                        # Get pixel coordinates
                        row, col = src.index(utm_x, utm_y)
                        
                        # Check if coordinates are within bounds
                        if 0 <= row < src.height and 0 <= col < src.width:
                            # Read the value at this location
                            value = src.read(1, window=rasterio.windows.Window(col, row, 1, 1))[0, 0]
                            
                            print(f"  Point {point_id} ({utm_x}, {utm_y}): row={row}, col={col}, value={value}")
                            
                            # Check if it's a NoData value
                            if src.nodata is not None and value == src.nodata:
                                print(f"    ❌ NoData value at this location")
                            elif np.isnan(value):
                                print(f"    ❌ NaN value at this location")
                            else:
                                print(f"    ✅ Valid value found")
                        else:
                            print(f"  Point {point_id} ({utm_x}, {utm_y}): ❌ Outside raster bounds")
                            print(f"    Row {row} (max: {src.height-1}), Col {col} (max: {src.width-1})")
                    
                    except Exception as e:
                        print(f"  Point {point_id} ({utm_x}, {utm_y}): ❌ Error reading value: {e}")
                
                # Get overall statistics
                print(f"\nOverall statistics for {feature_name}:")
                data = src.read(1)
                
                # Mask NoData values
                if src.nodata is not None:
                    masked_data = np.ma.masked_equal(data, src.nodata)
                else:
                    masked_data = np.ma.masked_invalid(data)
                
                valid_data = masked_data.compressed()
                
                print(f"  Total pixels: {data.size}")
                print(f"  Valid pixels: {len(valid_data)}")
                print(f"  NoData/NaN pixels: {data.size - len(valid_data)}")
                print(f"  NoData percentage: {(data.size - len(valid_data))/data.size*100:.2f}%")
                
                if len(valid_data) > 0:
                    print(f"  Value range: {valid_data.min():.2f} to {valid_data.max():.2f}")
                    print(f"  Mean: {valid_data.mean():.2f}")
                    print(f"  Std: {valid_data.std():.2f}")
        
        except Exception as e:
            print(f"❌ Error reading {feature_name} file: {e}")

if __name__ == "__main__":
    validate_static_datacube()
