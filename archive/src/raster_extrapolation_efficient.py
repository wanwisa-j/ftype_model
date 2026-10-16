#!/usr/bin/env python3
"""
Efficient linear interpolation and extrapolation using numpy's interp with bounds_error=False.
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


def interpolate_chunk_efficient_extrapolation(data_chunk, time_indices, output_nodata):
    """
    Efficient vectorized interpolation and extrapolation for a data chunk.
    """
    bands, height, width = data_chunk.shape
    result = data_chunk.copy()
    
    # Create invalid mask
    invalid_mask = np.isnan(data_chunk)
    
    # Process each spatial pixel
    for h in range(height):
        for w in range(width):
            pixel_timeseries = data_chunk[:, h, w]
            pixel_invalid = invalid_mask[:, h, w]
            
            # Skip if no invalid values
            if not np.any(pixel_invalid):
                continue
                
            # Get valid values
            valid_mask = ~pixel_invalid
            valid_count = np.sum(valid_mask)
            
            if valid_count <= 1:
                # Set all to nodata if insufficient valid values
                result[:, h, w] = output_nodata
            else:
                # Get valid indices and values
                valid_indices = time_indices[valid_mask]
                valid_values = pixel_timeseries[valid_mask]
                
                # Get all invalid time indices
                invalid_indices = time_indices[pixel_invalid]
                
                # Use numpy.interp which automatically extrapolates beyond range
                # by using the boundary values
                interpolated_values = np.interp(invalid_indices, valid_indices, valid_values)
                
                # Assign interpolated/extrapolated values
                result[pixel_invalid, h, w] = interpolated_values
    
    return result


def interpolate_raster_efficient_extrapolation(input_path, output_path, nodata_value=None, chunk_size=512):
    """
    Efficient interpolation and extrapolation of raster datacube.
    """
    logger.info(f"Starting efficient interpolation with extrapolation of {input_path}")
    
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
                    
                    # Interpolate and extrapolate chunk
                    interpolated_chunk = interpolate_chunk_efficient_extrapolation(
                        data_chunk, time_indices, output_nodata
                    )
                    
                    # Write result
                    dst.write(interpolated_chunk, window=window)
                
                # Progress
                progress = (row_end / height) * 100
                logger.info(f"Progress: {progress:.1f}%")
            
            logger.info(f"Efficient interpolation with extrapolation complete! Output: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Efficient raster interpolation with extrapolation')
    parser.add_argument('input_file', help='Input raster file')
    parser.add_argument('output_file', help='Output raster file')
    parser.add_argument('--nodata_value', type=float, default=None,
                       help='Nodata value (default: -9999)')
    parser.add_argument('--chunk_size', type=int, default=512,
                       help='Chunk size (default: 512)')
    
    args = parser.parse_args()
    
    input_path = Path(args.input_file)
    if not input_path.exists():
        logger.error(f"Input file not found: {input_path}")
        sys.exit(1)
    
    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        interpolate_raster_efficient_extrapolation(
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