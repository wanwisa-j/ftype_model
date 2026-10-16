#!/usr/bin/env python3
"""
Compare original, interpolated, and extrapolated raster results.
"""

import numpy as np
import rasterio
import matplotlib.pyplot as plt
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def safe_load_raster_info(file_path):
    """Safely load raster info, handle errors."""
    try:
        with rasterio.open(file_path) as src:
            return {
                'exists': True,
                'shape': (src.count, src.height, src.width),
                'bounds': src.bounds,
                'crs': src.crs,
                'dtype': src.dtypes[0],
                'nodata': src.nodata,
                'file_size': Path(file_path).stat().st_size
            }
    except Exception as e:
        return {
            'exists': False,
            'error': str(e),
            'file_size': Path(file_path).stat().st_size if Path(file_path).exists() else 0
        }

def analyze_all_files():
    """Compare all three files: original, interpolated, and extrapolated."""
    
    files = {
        'original': "/fs2/ftype_mflf/shared_data/datacube/2025/1.2.0/mfl_cmf_area_project_2025_chr_zone_2/vv/cube_s1_vv_2019_W4before_to_W16after_21bands.tif",
        'interpolated': "/fs2/ftype_mflf/shared_data/interpolated/cube_s1_vv_2019_W4before_to_W16after_21bands_interpolated.tif",
        'extrapolated': "/fs2/ftype_mflf/shared_data/interpolated/cube_s1_vv_2019_W4before_to_W16after_21bands_extrapolated.tif"
    }
    
    print("=== FILE COMPARISON ===")
    print("File Type     | Exists | Shape | Size (MB) | Status")
    print("-" * 60)
    
    file_info = {}
    for name, path in files.items():
        info = safe_load_raster_info(path)
        file_info[name] = info
        
        if info['exists']:
            size_mb = info['file_size'] / (1024*1024)
            print(f"{name:12s} | {'Yes':6s} | {str(info['shape']):15s} | {size_mb:7.1f} | OK")
        else:
            size_mb = info['file_size'] / (1024*1024)
            print(f"{name:12s} | {'No':6s} | {'N/A':15s} | {size_mb:7.1f} | ERROR: {info['error']}")
    
    # If extrapolated file is too small, something went wrong
    if file_info['extrapolated']['exists'] and file_info['extrapolated']['file_size'] < 1000000:  # < 1MB
        print(f"\n⚠️  WARNING: Extrapolated file is suspiciously small ({file_info['extrapolated']['file_size']} bytes)")
        print("This suggests the extrapolation process may have failed or been interrupted.")
        return
    
    # Continue with detailed analysis only if all files exist and are reasonable size
    if not all(info['exists'] for info in file_info.values()):
        print("\n❌ Cannot proceed with detailed analysis - some files are missing or corrupted.")
        return
    
    # Detailed band-by-band analysis
    print(f"\n=== BAND-BY-BAND ANALYSIS ===")
    
    try:
        with rasterio.open(files['original']) as orig_src, \
             rasterio.open(files['interpolated']) as interp_src, \
             rasterio.open(files['extrapolated']) as extrap_src:
            
            total_bands = orig_src.count
            print("Band | Original NaN | Interpolated NaN | Extrapolated NaN | Improvement")
            print("-" * 70)
            
            band_results = []
            
            for band_idx in range(min(total_bands, 21)):  # Limit to prevent memory issues
                orig_band = orig_src.read(band_idx + 1).astype(np.float32)
                interp_band = interp_src.read(band_idx + 1).astype(np.float32)
                extrap_band = extrap_src.read(band_idx + 1).astype(np.float32)
                
                orig_nan = np.sum(np.isnan(orig_band))
                interp_nan = np.sum(np.isnan(interp_band)) + np.sum(np.isclose(interp_band, interp_src.nodata or -9999))
                extrap_nan = np.sum(np.isnan(extrap_band)) + np.sum(np.isclose(extrap_band, extrap_src.nodata or -9999))
                
                improvement = orig_nan - extrap_nan
                
                print(f"{band_idx:4d} | {orig_nan:10,d} | {interp_nan:12,d} | {extrap_nan:12,d} | {improvement:+9,d}")
                
                band_results.append({
                    'band': band_idx,
                    'orig_nan': orig_nan,
                    'interp_nan': interp_nan,
                    'extrap_nan': extrap_nan,
                    'improvement': improvement
                })
            
            # Create summary visualization
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            
            bands = [r['band'] for r in band_results]
            orig_nan = [r['orig_nan'] for r in band_results]
            interp_nan = [r['interp_nan'] for r in band_results]
            extrap_nan = [r['extrap_nan'] for r in band_results]
            
            # Plot 1: Missing pixels comparison
            axes[0,0].plot(bands, orig_nan, 'o-', label='Original', color='red', alpha=0.7)
            axes[0,0].plot(bands, interp_nan, 's-', label='Interpolated', color='blue', alpha=0.7)
            axes[0,0].plot(bands, extrap_nan, '^-', label='Extrapolated', color='green', alpha=0.7)
            axes[0,0].set_xlabel('Band Index')
            axes[0,0].set_ylabel('Missing Pixels')
            axes[0,0].set_title('Missing Pixels by Band')
            axes[0,0].legend()
            axes[0,0].grid(True, alpha=0.3)
            
            # Plot 2: Improvement over original
            interp_improvement = [orig_nan[i] - interp_nan[i] for i in range(len(bands))]
            extrap_improvement = [orig_nan[i] - extrap_nan[i] for i in range(len(bands))]
            
            x = np.arange(len(bands))
            width = 0.35
            
            axes[0,1].bar(x - width/2, interp_improvement, width, label='Interpolation Only', alpha=0.7)
            axes[0,1].bar(x + width/2, extrap_improvement, width, label='With Extrapolation', alpha=0.7)
            axes[0,1].set_xlabel('Band Index')
            axes[0,1].set_ylabel('Pixels Recovered')
            axes[0,1].set_title('Pixel Recovery Comparison')
            axes[0,1].legend()
            axes[0,1].set_xticks(x)
            axes[0,1].set_xticklabels(bands)
            axes[0,1].grid(True, alpha=0.3)
            
            # Plot 3: Focus on edge bands (18-20)
            edge_bands = [18, 19, 20]
            if all(b < len(band_results) for b in edge_bands):
                edge_orig = [band_results[b]['orig_nan'] for b in edge_bands]
                edge_interp = [band_results[b]['interp_nan'] for b in edge_bands]
                edge_extrap = [band_results[b]['extrap_nan'] for b in edge_bands]
                
                x_edge = np.arange(len(edge_bands))
                axes[1,0].bar(x_edge - 0.25, edge_orig, 0.25, label='Original', color='red', alpha=0.7)
                axes[1,0].bar(x_edge, edge_interp, 0.25, label='Interpolated', color='blue', alpha=0.7)
                axes[1,0].bar(x_edge + 0.25, edge_extrap, 0.25, label='Extrapolated', color='green', alpha=0.7)
                axes[1,0].set_xlabel('Band Index')
                axes[1,0].set_ylabel('Missing Pixels')
                axes[1,0].set_title('Edge Bands (18-20) Comparison')
                axes[1,0].legend()
                axes[1,0].set_xticks(x_edge)
                axes[1,0].set_xticklabels(edge_bands)
                axes[1,0].grid(True, alpha=0.3)
            
            # Plot 4: Total summary
            total_orig = sum(orig_nan)
            total_interp = sum(interp_nan)
            total_extrap = sum(extrap_nan)
            
            methods = ['Original', 'Interpolated', 'Extrapolated']
            totals = [total_orig, total_interp, total_extrap]
            colors = ['red', 'blue', 'green']
            
            axes[1,1].bar(methods, totals, color=colors, alpha=0.7)
            axes[1,1].set_ylabel('Total Missing Pixels')
            axes[1,1].set_title('Overall Comparison')
            
            for i, v in enumerate(totals):
                axes[1,1].text(i, v + max(totals) * 0.01, f'{v:,}', ha='center', va='bottom')
            
            plt.tight_layout()
            
            # Save results
            output_plot = Path(files['extrapolated']).parent / 'complete_interpolation_comparison.png'
            plt.savefig(output_plot, dpi=150, bbox_inches='tight')
            logger.info(f"Complete comparison plot saved to: {output_plot}")
            plt.show()
            
            print(f"\n=== SUMMARY ===")
            print(f"Total missing pixels - Original: {total_orig:,}")
            print(f"Total missing pixels - Interpolated: {total_interp:,}")
            print(f"Total missing pixels - Extrapolated: {total_extrap:,}")
            print(f"Improvement with interpolation: {total_orig - total_interp:,} pixels")
            print(f"Additional improvement with extrapolation: {total_interp - total_extrap:,} pixels")
            print(f"Overall improvement: {total_orig - total_extrap:,} pixels ({100*(total_orig - total_extrap)/total_orig:.1f}%)")
            
    except Exception as e:
        print(f"❌ Error during detailed analysis: {e}")
        logger.error(f"Detailed analysis failed: {e}")

if __name__ == "__main__":
    analyze_all_files()