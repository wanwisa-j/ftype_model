import numpy as np
import rasterio
from collections import Counter
import os

# File path
file_path = "/fs2/ftype_mflf/result/2025/1.2.0/mfl_cmf_area_project_2025_chr_zone_1/model_version_2.2.0/pred_majority_2.2.0.tif"

# Check if file exists
print(f"Checking if file exists: {file_path}")
print(f"File exists: {os.path.exists(file_path)}")

if os.path.exists(file_path):
    print(f"File size: {os.path.getsize(file_path) / (1024*1024):.2f} MB")
    
    # Read the GeoTIFF file
    print("\nReading GeoTIFF file...")
    with rasterio.open(file_path) as src:
        print(f"Raster dimensions: {src.width} x {src.height}")
        print(f"Number of bands: {src.count}")
        print(f"Data type: {src.dtypes[0]}")
        print(f"CRS: {src.crs}")
        print(f"NoData value: {src.nodata}")
        
        # Read the data
        data = src.read(1)  # Read first band
        
    print(f"Data shape: {data.shape}")
    print(f"Data type: {data.dtype}")
    
    # Handle NoData values
    if src.nodata is not None:
        # Mask out NoData values
        valid_data = data[data != src.nodata]
        print(f"Total pixels: {data.size:,}")
        print(f"Valid pixels (excluding NoData): {valid_data.size:,}")
        print(f"NoData pixels: {data.size - valid_data.size:,}")
    else:
        valid_data = data.flatten()
        print(f"Total pixels: {data.size:,}")
    
    # Calculate class distribution
    print("\nCalculating class distribution...")
    unique_classes, counts = np.unique(valid_data, return_counts=True)
    
    total_valid_pixels = valid_data.size
    
    print(f"\nClass Distribution:")
    print(f"{'Class':<10} {'Count':<15} {'Percentage':<12}")
    print("-" * 37)
    
    for cls, count in zip(unique_classes, counts):
        percentage = (count / total_valid_pixels) * 100
        print(f"{cls:<10} {count:<15,} {percentage:<12.4f}%")
    
    # Check if class 2.0 dominates over 99%
    if 2.0 in unique_classes:
        class_2_idx = np.where(unique_classes == 2.0)[0][0]
        class_2_count = counts[class_2_idx]
        class_2_percentage = (class_2_count / total_valid_pixels) * 100
        
        print(f"\nSpecific analysis for class 2.0:")
        print(f"Class 2.0 count: {class_2_count:,}")
        print(f"Class 2.0 percentage: {class_2_percentage:.6f}%")
        print(f"Does class 2.0 dominate >99%: {class_2_percentage > 99}")
    else:
        print("\nClass 2.0 not found in the data")
        
    # Additional statistics
    print(f"\nAdditional Statistics:")
    print(f"Number of unique classes: {len(unique_classes)}")
    print(f"Min value: {np.min(valid_data)}")
    print(f"Max value: {np.max(valid_data)}")
    print(f"Mean value: {np.mean(valid_data):.4f}")
    print(f"Median value: {np.median(valid_data):.4f}")
        
else:
    print("File does not exist!")