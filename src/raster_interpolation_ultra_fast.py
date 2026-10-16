#!/usr/bin/env python3
"""
Ultra-fast linear interpolation for raster datacubes using full vectorization.

This approach processes entire chunks at once using numpy operations.
"""

import numpy as np
import rasterio
import argparse
import logging
from pathlib import Path
import sys

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def interpolate_chunk_vectorized(data_chunk, time_indices, output_nodata):
    """
    Vectorized interpolation with edge value extrapolation for a data chunk.
    
    Args:
        data_chunk: numpy array of shape (bands, height, width)
        time_indices: array of time indices [0, 1, 2, ...]
        output_nodata: value to use for nodata pixels
        
    Returns:
        interpolated data chunk with same shape as input
    """
    bands, height, width = data_chunk.shape
    
    # Prepare float64 for interpolation precision
    arr = data_chunk.astype(np.float64)
    
    # Mark invalid values: nan, inf, -inf, nodata (if defined)
    invalid_elem = ~np.isfinite(arr)
    if output_nodata is not None:
        invalid_elem |= (arr == output_nodata)
    
    # Pixels with all timesteps invalid -> mask_invalid True
    mask_invalid = np.all(invalid_elem, axis=0)  # (height, width)
    
    # Flatten for efficient column-wise processing
    arr_flat = arr.reshape(bands, -1)
    invalid_flat = invalid_elem.reshape(bands, -1)
    mask_all_invalid_flat = mask_invalid.ravel()
    
    # Columns needing interpolation (at least one valid & one invalid)
    cols_to_fix = np.where(~mask_all_invalid_flat & np.any(invalid_flat, axis=0))[0]
    
    for col in cols_to_fix:
        series = arr_flat[:, col]
        valid_mask = np.isfinite(series)
        if not valid_mask.any():
            continue  # should not happen due to ~mask_all_invalid_flat
        if valid_mask.all():
            continue  # no interpolation needed
        
        valid_times = time_indices[valid_mask]
        valid_values = series[valid_mask]
        
        # Linear interpolation with edge extension (extrapolation)
        series_filled = np.interp(time_indices, valid_times, valid_values)
        arr_flat[:, col] = series_filled
    
    # Set fully invalid pixels to output_nodata
    if mask_all_invalid_flat.any():
        arr_flat[:, mask_all_invalid_flat] = output_nodata
    
    # Cast back to float32 and reshape
    result = arr_flat.reshape(bands, height, width).astype(np.float32)
    
    return result


def interpolate_raster_ultra_fast(input_path, output_path, nodata_value=None, chunk_size=2048):
    """
    Ultra-fast interpolation of raster datacube.
    """
    logger.info(f"Starting ultra-fast interpolation of {input_path}")
    
    with rasterio.open(input_path) as src:
        profile = src.profile.copy()
        height, width = src.height, src.width
        count = src.count
        
        logger.info(f"Raster dimensions: {count} bands, {height}H x {width}W")
        
        # Set output parameters
        output_nodata = nodata_value if nodata_value is not None else -9999
        profile['nodata'] = output_nodata
        profile['dtype'] = 'float32'
        
        time_indices = np.arange(count, dtype=np.float32)
        
        with rasterio.open(output_path, 'w', **profile) as dst:
            
            # Process in chunks to manage memory
            for row_start in range(0, height, chunk_size):
                row_end = min(row_start + chunk_size, height)
                
                for col_start in range(0, width, chunk_size):
                    col_end = min(col_start + chunk_size, width)
                    
                    # Read chunk
                    window = rasterio.windows.Window(col_start, row_start, 
                                                   col_end - col_start, row_end - row_start)
                    data_chunk = src.read(window=window).astype(np.float32)
                    
                    # Interpolate chunk
                    interpolated_chunk = interpolate_chunk_vectorized(
                        data_chunk, time_indices, output_nodata
                    )
                    
                    # Write result
                    dst.write(interpolated_chunk, window=window)
                
                # Progress
                progress = (row_end / height) * 100
                logger.info(f"Progress: {progress:.1f}%")
            
            logger.info(f"Interpolation complete! Output: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Ultra-fast raster interpolation')
    parser.add_argument('input_file', help='Input raster file')
    parser.add_argument('output_file', help='Output raster file')
    parser.add_argument('--nodata_value', type=float, default=None,
                       help='Nodata value (default: -9999)')
    parser.add_argument('--chunk_size', type=int, default=2048,
                       help='Chunk size (default: 2048)')
    
    args = parser.parse_args()
    
    input_path = Path(args.input_file)
    if not input_path.exists():
        logger.error(f"Input file not found: {input_path}")
        sys.exit(1)
    
    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        interpolate_raster_ultra_fast(
            str(input_path), 
            str(output_path),
            nodata_value=args.nodata_value,
            chunk_size=args.chunk_size
        )
    except Exception as e:
        logger.error(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()