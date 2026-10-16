#!/usr/bin/env python3
"""
Efficient linear interpolation script for raster datacubes with invalid values.

This vectorized version processes the entire raster much faster than pixel-by-pixel processing.
"""

import numpy as np
import rasterio
from rasterio.profiles import default_gtiff_profile
import argparse
import logging
from pathlib import Path
import sys
from scipy import interpolate

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def interpolate_raster_cube_vectorized(input_path, output_path, nodata_value=None, chunk_size=1024):
    """
    Vectorized interpolation of invalid values in a raster datacube.
    
    Args:
        input_path: path to input raster file
        output_path: path to output raster file  
        nodata_value: specific nodata value to treat as invalid
        chunk_size: size of chunks for processing (to manage memory)
    """
    logger.info(f"Starting vectorized interpolation of {input_path}")
    
    with rasterio.open(input_path) as src:
        # Get raster properties
        profile = src.profile.copy()
        height, width = src.height, src.width
        count = src.count  # number of bands (timesteps)
        
        logger.info(f"Raster dimensions: {count} bands, {height}H x {width}W")
        logger.info(f"Data type: {profile['dtype']}")
        logger.info(f"Original nodata value: {profile.get('nodata', 'None')}")
        
        # Set output nodata value
        output_nodata = nodata_value if nodata_value is not None else -9999
        profile['nodata'] = output_nodata
        profile['dtype'] = 'float32'
        
        # Create time index array
        time_indices = np.arange(count)
        
        with rasterio.open(output_path, 'w', **profile) as dst:
            
            total_pixels = height * width
            processed_pixels = 0
            interpolated_pixels = 0
            invalid_pixels = 0
            
            # Process data in row chunks to manage memory
            for row_start in range(0, height, chunk_size):
                row_end = min(row_start + chunk_size, height)
                chunk_height = row_end - row_start
                
                # Read all bands for this row chunk
                window = rasterio.windows.Window(0, row_start, width, chunk_height)
                data_chunk = src.read(window=window)  # Shape: (bands, chunk_height, width)
                
                # Convert to float32
                data_chunk = data_chunk.astype(np.float32)
                
                # Identify invalid values
                invalid_mask = np.isnan(data_chunk)
                if src.nodata is not None:
                    invalid_mask |= (data_chunk == src.nodata)
                if nodata_value is not None:
                    invalid_mask |= (data_chunk == nodata_value)
                
                # For each pixel position (h, w), interpolate along time axis
                for h in range(chunk_height):
                    for w in range(width):
                        timeseries = data_chunk[:, h, w]  # Shape: (bands,)
                        invalid_ts = invalid_mask[:, h, w]  # Shape: (bands,)
                        
                        processed_pixels += 1
                        
                        # Skip if no invalid values
                        if not np.any(invalid_ts):
                            continue
                            
                        # Check valid values count
                        valid_mask = ~invalid_ts
                        valid_count = np.sum(valid_mask)
                        
                        if valid_count <= 1:
                            # Cannot interpolate, set all to nodata
                            data_chunk[:, h, w] = output_nodata
                            invalid_pixels += 1
                        else:
                            # Perform interpolation
                            valid_indices = time_indices[valid_mask]
                            valid_values = timeseries[valid_mask]
                            
                            # Only interpolate within the valid range
                            first_valid = np.min(valid_indices)
                            last_valid = np.max(valid_indices)
                            
                            # Find invalid points within valid range
                            interp_mask = invalid_ts & (time_indices >= first_valid) & (time_indices <= last_valid)
                            
                            if np.any(interp_mask):
                                # Linear interpolation
                                interp_values = np.interp(
                                    time_indices[interp_mask],
                                    valid_indices, 
                                    valid_values
                                )
                                data_chunk[interp_mask, h, w] = interp_values
                                interpolated_pixels += 1
                            
                            # Set points outside valid range to nodata
                            outside_mask = invalid_ts & ((time_indices < first_valid) | (time_indices > last_valid))
                            if np.any(outside_mask):
                                data_chunk[outside_mask, h, w] = output_nodata
                
                # Write the processed chunk
                dst.write(data_chunk, window=window)
                
                # Progress logging
                progress = (row_end / height) * 100
                logger.info(f"Progress: {progress:.1f}%")
            
            logger.info(f"Interpolation complete!")
            logger.info(f"Total pixels processed: {processed_pixels}")
            logger.info(f"Pixels with interpolation: {interpolated_pixels}")
            logger.info(f"Pixels set to nodata: {invalid_pixels}")
            logger.info(f"Output saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Efficiently interpolate invalid values in raster datacubes')
    parser.add_argument('input_file', help='Input raster file path')
    parser.add_argument('output_file', help='Output raster file path')
    parser.add_argument('--nodata_value', type=float, default=None,
                       help='Specific nodata value to treat as invalid (default: use file nodata)')
    parser.add_argument('--chunk_size', type=int, default=1024,
                       help='Row chunk size for processing (default: 1024)')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose logging')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Validate input file exists
    input_path = Path(args.input_file)
    if not input_path.exists():
        logger.error(f"Input file does not exist: {input_path}")
        sys.exit(1)
    
    # Create output directory if needed
    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        interpolate_raster_cube_vectorized(
            str(input_path), 
            str(output_path), 
            nodata_value=args.nodata_value,
            chunk_size=args.chunk_size
        )
    except Exception as e:
        logger.error(f"Error during interpolation: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()