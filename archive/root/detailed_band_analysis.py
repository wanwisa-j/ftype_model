#!/usr/bin/env python3
"""
Detailed Band Analysis Script

Analyzes which specific bands contain valid vs invalid data in the problematic raster files.
This will help identify which weeks/bands are missing for VV/VH radar data in 2023.
"""

import rasterio
import numpy as np
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def analyze_band_by_band(raster_path, sample_points=100):
    """Analyze each band individually to identify which ones have valid data."""
    
    logger.info(f"🔍 Analyzing: {Path(raster_path).name}")
    
    with rasterio.open(raster_path) as src:
        height, width = src.height, src.width
        
        # Sample multiple points across the raster
        sample_rows = np.linspace(0, height-1, int(np.sqrt(sample_points)), dtype=int)
        sample_cols = np.linspace(0, width-1, int(np.sqrt(sample_points)), dtype=int)
        
        band_analysis = []
        
        for band_idx in range(1, src.count + 1):
            valid_pixels = 0
            total_pixels = len(sample_rows) * len(sample_cols)
            
            try:
                for row in sample_rows:
                    for col in sample_cols:
                        pixel_value = src.read(band_idx, window=((row, row+1), (col, col+1)))[0, 0]
                        
                        # Check if pixel has valid data
                        if not np.isnan(pixel_value) and pixel_value != src.nodata and pixel_value != 0:
                            valid_pixels += 1
                
                valid_percentage = (valid_pixels / total_pixels) * 100
                has_data = valid_pixels > 0
                
                band_info = {
                    'band': band_idx,
                    'week': band_idx,  # Assuming band == week for radar data
                    'valid_pixels': valid_pixels,
                    'total_sampled': total_pixels,
                    'valid_percentage': valid_percentage,
                    'has_data': has_data,
                    'status': 'VALID' if has_data else 'EMPTY'
                }
                
                band_analysis.append(band_info)
                
                status_icon = "✅" if has_data else "❌"
                logger.info(f"  Band {band_idx:2d} (Week {band_idx:2d}): {status_icon} {valid_percentage:5.1f}% valid pixels")
                
            except Exception as e:
                logger.error(f"  Band {band_idx:2d}: Error reading - {str(e)}")
                band_analysis.append({
                    'band': band_idx,
                    'week': band_idx,
                    'error': str(e),
                    'status': 'ERROR'
                })
        
        # Summary
        valid_bands = [b for b in band_analysis if b.get('has_data', False)]
        empty_bands = [b for b in band_analysis if b.get('status') == 'EMPTY']
        error_bands = [b for b in band_analysis if b.get('status') == 'ERROR']
        
        logger.info(f"📊 Summary:")
        logger.info(f"  Valid bands: {len(valid_bands)}/21")
        logger.info(f"  Empty bands: {len(empty_bands)}/21")
        logger.info(f"  Error bands: {len(error_bands)}/21")
        
        if empty_bands:
            empty_weeks = [b['week'] for b in empty_bands]
            logger.warning(f"  Empty weeks: {empty_weeks}")
        
        return band_analysis

def main():
    """Analyze all problematic files."""
    
    logger.info("🔍 DETAILED BAND-BY-BAND ANALYSIS")
    logger.info("=" * 50)
    
    # Problematic files identified by validation
    problematic_files = [
        "/fs2/ftype_mflf/shared_data/datacube/2025/1.2.0/mfl_cmf_area_project_2025_chr_zone_1/vv/cube_s1_vv_2023_W4before_to_W16after_21bands.tif",
        "/fs2/ftype_mflf/shared_data/datacube/2025/1.2.0/mfl_cmf_area_project_2025_chr_zone_2/vv/cube_s1_vv_2023_W4before_to_W16after_21bands.tif", 
        "/fs2/ftype_mflf/shared_data/datacube/2025/1.2.0/mfl_cmf_area_project_2025_chr_zone_1/vh/cube_s1_vh_2023_W4before_to_W16after_21bands.tif",
        "/fs2/ftype_mflf/shared_data/datacube/2025/1.2.0/mfl_cmf_area_project_2025_chr_zone_2/vh/cube_s1_vh_2023_W4before_to_W16after_21bands.tif"
    ]
    
    all_results = {}
    
    for file_path in problematic_files:
        logger.info(f"\n" + "="*80)
        result = analyze_band_by_band(file_path)
        all_results[file_path] = result
    
    # Cross-file analysis
    logger.info(f"\n" + "="*80)
    logger.info("🔍 CROSS-FILE PATTERN ANALYSIS")
    
    # Find common empty bands across files
    empty_bands_per_file = {}
    for file_path, results in all_results.items():
        empty_bands = [r['band'] for r in results if r.get('status') == 'EMPTY']
        empty_bands_per_file[Path(file_path).name] = empty_bands
        logger.info(f"{Path(file_path).name}: Empty bands {empty_bands}")
    
    # Find bands that are consistently empty across all files
    if empty_bands_per_file:
        all_empty_bands = set(list(empty_bands_per_file.values())[0])
        for empty_bands in empty_bands_per_file.values():
            all_empty_bands = all_empty_bands.intersection(set(empty_bands))
        
        if all_empty_bands:
            logger.error(f"🚨 Bands consistently empty across ALL files: {sorted(all_empty_bands)}")
        else:
            logger.info("ℹ️  No bands are consistently empty across all files")
    
    return all_results

if __name__ == "__main__":
    results = main()
