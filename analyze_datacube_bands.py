#!/usr/bin/env python3
"""
Analyze datacube band structure to understand temporal dimension mismatch.
Expected: 21 bands per raster (21 weeks)
Actual: 18 timesteps in biophysical data

This script investigates:
1. Actual number of bands in datacube rasters
2. How extraction process handles these bands
3. Why we get 18 instead of 21 timesteps
"""

import os
import sys
import rasterio
import numpy as np
import pandas as pd
from pathlib import Path
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def analyze_datacube_raster(raster_path):
    """Analyze a single datacube raster file."""
    try:
        with rasterio.open(raster_path) as src:
            logger.info(f"Analyzing: {raster_path}")
            logger.info(f"  Bands: {src.count}")
            logger.info(f"  Shape: {src.width} x {src.height}")
            logger.info(f"  CRS: {src.crs}")
            logger.info(f"  Transform: {src.transform}")
            
            # Read band names if available
            band_descriptions = [src.tags(i) for i in range(1, src.count + 1)]
            logger.info(f"  Band descriptions: {band_descriptions[:3]}...")  # Show first 3
            
            return {
                'path': raster_path,
                'bands': src.count,
                'width': src.width,
                'height': src.height,
                'crs': str(src.crs),
                'band_descriptions': band_descriptions
            }
    except Exception as e:
        logger.error(f"Error analyzing {raster_path}: {e}")
        return None

def analyze_datacube_structure():
    """Analyze the structure of datacube files."""
    datacube_base = "/fs2/ftype_mflf/shared_data/datacube/2024/1.2.0/mfl_cmf_2024"
    
    # Features to analyze
    features = ['lai', 'fapar', 'fcover', 'ndvi', 'gvmi', 'slavi']
    years = [2019, 2020, 2021, 2022, 2023, 2024]
    
    results = []
    
    for feature in features:
        feature_dir = Path(datacube_base) / feature
        if not feature_dir.exists():
            logger.warning(f"Feature directory not found: {feature_dir}")
            continue
            
        logger.info(f"\n=== Analyzing {feature.upper()} ===")
        
        # Look for raster files
        raster_files = list(feature_dir.glob("*.tif")) + list(feature_dir.glob("*.tiff"))
        logger.info(f"Found {len(raster_files)} raster files")
        
        for raster_file in raster_files[:2]:  # Analyze first 2 files per feature
            result = analyze_datacube_raster(str(raster_file))
            if result:
                result['feature'] = feature
                results.append(result)
    
    return results

def analyze_sample_extraction():
    """Simulate how extraction works to understand band->timestep conversion."""
    sample_raster = "/fs2/ftype_mflf/shared_data/datacube/2024/1.2.0/mfl_cmf_2024/lai/cube_biophy_lai_2019_W4before_to_W16after_21bands.tif"
    
    if not os.path.exists(sample_raster):
        logger.error(f"Sample raster not found: {sample_raster}")
        return None
        
    logger.info(f"\n=== Detailed Analysis of Sample Raster ===")
    
    try:
        with rasterio.open(sample_raster) as src:
            logger.info(f"File: {sample_raster}")
            logger.info(f"Bands: {src.count}")
            logger.info(f"Shape: ({src.height}, {src.width})")
            
            # Extract sample point (center of raster)
            center_row = src.height // 2
            center_col = src.width // 2
            
            # Read all bands for the center pixel
            pixel_values = []
            for band_idx in range(1, src.count + 1):
                band_data = src.read(band_idx)
                pixel_value = band_data[center_row, center_col]
                pixel_values.append(pixel_value)
                
            logger.info(f"Sample pixel values (all {len(pixel_values)} bands):")
            logger.info(f"  Values: {pixel_values}")
            logger.info(f"  Non-zero values: {sum(1 for v in pixel_values if v != 0)}")
            logger.info(f"  Valid values (not NaN/fill): {sum(1 for v in pixel_values if not np.isnan(v) and v != -9999)}")
            
            # Check for pattern in band values
            valid_indices = [i for i, v in enumerate(pixel_values) if not np.isnan(v) and v != -9999 and v != 0]
            logger.info(f"  Valid band indices: {valid_indices}")
            logger.info(f"  Number of valid bands: {len(valid_indices)}")
            
            return {
                'total_bands': src.count,
                'valid_bands': len(valid_indices),
                'valid_indices': valid_indices,
                'pixel_values': pixel_values
            }
            
    except Exception as e:
        logger.error(f"Error in detailed analysis: {e}")
        return None

def main():
    """Main analysis function."""
    logger.info("=== Datacube Band Structure Analysis ===")
    
    # 1. Analyze datacube structure
    logger.info("\n1. Analyzing datacube raster structure...")
    datacube_results = analyze_datacube_structure()
    
    if datacube_results:
        df = pd.DataFrame(datacube_results)
        logger.info(f"\n=== Summary of Datacube Analysis ===")
        logger.info(f"Total files analyzed: {len(df)}")
        logger.info(f"Band counts by feature:")
        for feature in df['feature'].unique():
            feature_data = df[df['feature'] == feature]
            band_counts = feature_data['bands'].unique()
            logger.info(f"  {feature}: {band_counts}")
    
    # 2. Detailed analysis of sample raster
    logger.info("\n2. Detailed analysis of sample raster...")
    sample_result = analyze_sample_extraction()
    
    if sample_result:
        logger.info(f"\n=== Band->Timestep Conversion Analysis ===")
        logger.info(f"Total bands in raster: {sample_result['total_bands']}")
        logger.info(f"Valid/usable bands: {sample_result['valid_bands']}")
        logger.info(f"Potential reason for 18 vs 21: {21 - sample_result['valid_bands']} bands may be invalid/padding")
    
    # 3. Check extraction script logic
    logger.info("\n3. Recommendations:")
    logger.info("   - Check if extraction script filters out invalid bands")
    logger.info("   - Verify if some bands represent padding or metadata")
    logger.info("   - Consider if temporal alignment drops certain weeks")
    logger.info("   - Look for missing data handling that reduces 21->18")

if __name__ == "__main__":
    main()
