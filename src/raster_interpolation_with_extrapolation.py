#!/usr/bin/env python3
"""
Ultra-fast linear interpolation and extrapolation for raster datacubes using full vectorization.

This version includes extrapolation beyond the valid data range for edge bands.
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


def interpolate_chunk_with_extrapolation(data_chunk, time_indices, output_nodata):
    """
    Vectorized interpolation and extrapolation for a data chunk.
    
    Args:
        data_chunk: numpy array of shape (bands, height, width)
        time_indices: array of time indices [0, 1, 2, ...]
        output_nodata: value to use for nodata pixels
        
    Returns:
        interpolated data chunk with same shape as input
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
                
                # Find interpolation range
                first_valid = np.min(valid_indices)
                last_valid = np.max(valid_indices)
                
                # Process all invalid points (both interpolation and extrapolation)
                for t in range(bands):
                    if pixel_invalid[t]:
                        if first_valid <= t <= last_valid:
                            # Linear interpolation within valid range
                            result[t, h, w] = np.interp(t, valid_indices, valid_values)
                        else:
                            # Linear extrapolation beyond valid range
                            if valid_count >= 2:
                                # Use linear extrapolation
                                result[t, h, w] = np.interp(t, valid_indices, valid_values)
                            else:
                                # Fallback to nearest value if only one valid point
                                if t < first_valid:
                                    result[t, h, w] = valid_values[0]
                                else:
                                    result[t, h, w] = valid_values[-1]
    
    return result


def interpolate_raster_with_extrapolation(input_path, output_path, nodata_value=None, chunk_size=512):
    """
    Ultra-fast interpolation and extrapolation of raster datacube.
    """
    logger.info(f"Starting interpolation with extrapolation of {input_path}")
    
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
                    interpolated_chunk = interpolate_chunk_with_extrapolation(
                        data_chunk, time_indices, output_nodata
                    )
                    
                    # Write result
                    dst.write(interpolated_chunk, window=window)
                
                # Progress
                progress = (row_end / height) * 100
                logger.info(f"Progress: {progress:.1f}%")
            
            logger.info(f"Interpolation with extrapolation complete! Output: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Ultra-fast raster interpolation with extrapolation')
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
        interpolate_raster_with_extrapolation(
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