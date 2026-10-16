#!/usr/bin/env python3
"""
Compare original and interpolated raster data and create visualization plots.
"""

import numpy as np
import rasterio
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_raster_info(file_path):
    """Load basic information about a raster file."""
    with rasterio.open(file_path) as src:
        return {
            'shape': (src.count, src.height, src.width),
            'bounds': src.bounds,
            'crs': src.crs,
            'dtype': src.dtypes[0],
            'nodata': src.nodata,
            'transform': src.transform
        }

def count_pixels_by_type(data, nodata_val):
    """Count different pixel types in the data."""
    if nodata_val is not None:
        valid_mask = ~np.isclose(data, nodata_val, equal_nan=True)
    else:
        valid_mask = ~np.isnan(data)
    
    nan_mask = np.isnan(data)
    
    total_pixels = data.size
    valid_pixels = np.sum(valid_mask)
    nan_pixels = np.sum(nan_mask)
    nodata_pixels = total_pixels - valid_pixels - nan_pixels
    
    return {
        'total': total_pixels,
        'valid': valid_pixels,
        'nan': nan_pixels,
        'nodata': nodata_pixels
    }

def compare_rasters(original_path, interpolated_path, sample_bands=None):
    """Compare original and interpolated raster files."""
    
    logger.info(f"Loading original file: {original_path}")
    orig_info = load_raster_info(original_path)
    
    logger.info(f"Loading interpolated file: {interpolated_path}")
    interp_info = load_raster_info(interpolated_path)
    
    # Print basic info comparison
    print("\n=== RASTER COMPARISON ===")
    print(f"Original shape: {orig_info['shape']}")
    print(f"Interpolated shape: {interp_info['shape']}")
    print(f"Original dtype: {orig_info['dtype']}")
    print(f"Interpolated dtype: {interp_info['dtype']}")
    print(f"Original nodata: {orig_info['nodata']}")
    print(f"Interpolated nodata: {interp_info['nodata']}")
    
    # Load sample bands for detailed comparison
    if sample_bands is None:
        # Select a few bands across the time series
        total_bands = orig_info['shape'][0]
        sample_bands = [0, total_bands//4, total_bands//2, 3*total_bands//4, total_bands-1]
    
    with rasterio.open(original_path) as orig_src, rasterio.open(interpolated_path) as interp_src:
        
        print(f"\n=== BAND-WISE COMPARISON ===")
        print("Band | Original Stats | Interpolated Stats | Change")
        print("-" * 70)
        
        fig, axes = plt.subplots(len(sample_bands), 3, figsize=(15, 4*len(sample_bands)))
        if len(sample_bands) == 1:
            axes = axes.reshape(1, -1)
        
        for i, band_idx in enumerate(sample_bands):
            # Read band data (band indices are 1-based in rasterio)
            orig_band = orig_src.read(band_idx + 1).astype(np.float32)
            interp_band = interp_src.read(band_idx + 1).astype(np.float32)
            
            # Convert nodata to NaN for consistent handling
            if orig_info['nodata'] is not None:
                orig_band[np.isclose(orig_band, orig_info['nodata'])] = np.nan
            if interp_info['nodata'] is not None:
                interp_band[np.isclose(interp_band, interp_info['nodata'])] = np.nan
            
            # Count pixels
            orig_counts = count_pixels_by_type(orig_band, orig_info['nodata'])
            interp_counts = count_pixels_by_type(interp_band, interp_info['nodata'])
            
            # Calculate statistics for valid pixels only
            orig_valid = orig_band[~np.isnan(orig_band)]
            interp_valid = interp_band[~np.isnan(interp_band)]
            
            print(f"{band_idx:4d} | Valid: {orig_counts['valid']:8d}, NaN: {orig_counts['nan']:6d} | Valid: {interp_counts['valid']:8d}, NaN: {interp_counts['nan']:6d} | +{interp_counts['valid'] - orig_counts['valid']:6d}")
            
            if len(orig_valid) > 0:
                print(f"     | Mean: {np.mean(orig_valid):8.2f}, Std: {np.std(orig_valid):7.2f} | Mean: {np.mean(interp_valid):8.2f}, Std: {np.std(interp_valid):7.2f} |")
            
            # Create subplot for this band
            row = i
            
            # Original data
            im1 = axes[row, 0].imshow(orig_band, cmap='viridis', interpolation='nearest')
            axes[row, 0].set_title(f'Original Band {band_idx}')
            axes[row, 0].set_axis_off()
            plt.colorbar(im1, ax=axes[row, 0], shrink=0.6)
            
            # Interpolated data  
            im2 = axes[row, 1].imshow(interp_band, cmap='viridis', interpolation='nearest')
            axes[row, 1].set_title(f'Interpolated Band {band_idx}')
            axes[row, 1].set_axis_off()
            plt.colorbar(im2, ax=axes[row, 1], shrink=0.6)
            
            # Difference
            diff = interp_band - orig_band
            # Only show differences where both have valid data
            mask = np.isnan(orig_band) | np.isnan(interp_band)
            diff[mask] = np.nan
            
            im3 = axes[row, 2].imshow(diff, cmap='RdBu_r', interpolation='nearest')
            axes[row, 2].set_title(f'Difference Band {band_idx}')
            axes[row, 2].set_axis_off()
            plt.colorbar(im3, ax=axes[row, 2], shrink=0.6)
        
        plt.tight_layout()
        output_plot = Path(interpolated_path).parent / 'interpolation_comparison.png'
        plt.savefig(output_plot, dpi=150, bbox_inches='tight')
        logger.info(f"Comparison plot saved to: {output_plot}")
        plt.show()
        
        # Create a pixel change analysis plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Count changes across all bands
        total_orig_valid = 0
        total_interp_valid = 0
        band_changes = []
        
        for band_idx in range(orig_info['shape'][0]):
            orig_band = orig_src.read(band_idx + 1).astype(np.float32)
            interp_band = interp_src.read(band_idx + 1).astype(np.float32)
            
            if orig_info['nodata'] is not None:
                orig_band[np.isclose(orig_band, orig_info['nodata'])] = np.nan
            if interp_info['nodata'] is not None:
                interp_band[np.isclose(interp_band, interp_info['nodata'])] = np.nan
            
            orig_counts = count_pixels_by_type(orig_band, orig_info['nodata'])
            interp_counts = count_pixels_by_type(interp_band, interp_info['nodata'])
            
            total_orig_valid += orig_counts['valid']
            total_interp_valid += interp_counts['valid']
            band_changes.append(interp_counts['valid'] - orig_counts['valid'])
        
        # Plot valid pixel counts by band
        band_numbers = list(range(len(band_changes)))
        ax1.bar(band_numbers, band_changes, alpha=0.7)
        ax1.set_xlabel('Band Index')
        ax1.set_ylabel('Change in Valid Pixels')
        ax1.set_title('Valid Pixel Count Changes by Band')
        ax1.grid(True, alpha=0.3)
        
        # Overall summary
        total_change = total_interp_valid - total_orig_valid
        ax2.bar(['Original', 'Interpolated'], [total_orig_valid, total_interp_valid], 
                color=['blue', 'orange'], alpha=0.7)
        ax2.set_ylabel('Total Valid Pixels')
        ax2.set_title(f'Overall Valid Pixels\n(Change: +{total_change:,})')
        
        for i, v in enumerate([total_orig_valid, total_interp_valid]):
            ax2.text(i, v + max(total_orig_valid, total_interp_valid) * 0.01, 
                    f'{v:,}', ha='center', va='bottom')
        
        plt.tight_layout()
        summary_plot = Path(interpolated_path).parent / 'interpolation_summary.png'
        plt.savefig(summary_plot, dpi=150, bbox_inches='tight')
        logger.info(f"Summary plot saved to: {summary_plot}")
        plt.show()

if __name__ == "__main__":
    original_file = "/fs2/ftype_mflf/shared_data/datacube/2025/1.2.0/mfl_cmf_area_project_2025_chr_zone_2/vv/cube_s1_vv_2019_W4before_to_W16after_21bands.tif"
    interpolated_file = "/fs2/ftype_mflf/shared_data/interpolated/cube_s1_vv_2019_W4before_to_W16after_21bands_interpolated.tif"
    
    # Sample bands to visualize (will show bands 0, 5, 10, 15, 20 from 21-band cube)
    sample_bands = [0, 5, 10, 15, 20]
    
    compare_rasters(original_file, interpolated_file, sample_bands)