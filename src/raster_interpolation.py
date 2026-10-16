#!/usr/bin/env python3
"""
Linear interpolation script for raster datacubes with invalid values.

This script performs linear interpolation along the time dimension for raster datacubes
with 3D structure (time, height, width). It handles invalid values (NaN, -inf, inf, nodata)
by interpolating between valid timesteps. Pixels with entirely invalid timesteps remain
as NaN/nodata.

Usage:
    source /opt/miniconda3/etc/profile.d/conda.sh
    conda activate gisworkspace-cupy
    python raster_interpolation.py input.tif output.tif [--nodata_value VALUE]
    
Example:
    python raster_interpolation.py /fs2/ftype_mflf/shared_data/datacube/2025/1.2.0/mfl_cmf_area_project_2025_chr_zone_1/gvmi/cube_s2_gvmi_2023_W4before_to_W16after_21bands.tif output_interpolated.tif
"""

import numpy as np
import rasterio
from rasterio.profiles import default_gtiff_profile
import argparse
import logging
from pathlib import Path
import sys

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def is_invalid_value(data, nodata_value=None):
    """
    Identify invalid values in the data array.
    
    Args:
        data: numpy array
        nodata_value: specific nodata value to consider invalid
        
    Returns:
        boolean mask where True indicates invalid values
    """
    invalid_mask = np.isnan(data) | np.isinf(data)
    
    if nodata_value is not None:
        invalid_mask |= (data == nodata_value)
    
    return invalid_mask


def interpolate_timeseries(timeseries, time_indices=None):
    """
    Perform linear interpolation on a 1D timeseries.
    
    Args:
        timeseries: 1D numpy array with time dimension
        time_indices: array of time indices (default: 0, 1, 2, ...)
        
    Returns:
        interpolated timeseries with same shape as input
    """
    if time_indices is None:
        time_indices = np.arange(len(timeseries))
    
    # Find valid (non-invalid) values
    valid_mask = ~is_invalid_value(timeseries)
    
    # If no valid values or only one valid value, return original (cannot interpolate)
    if np.sum(valid_mask) <= 1:
        return timeseries.copy()
    
    # If all values are valid, return original
    if np.all(valid_mask):
        return timeseries.copy()
    
    # Perform interpolation
    result = timeseries.copy()
    valid_indices = time_indices[valid_mask]
    valid_values = timeseries[valid_mask]
    
    # Interpolate only between the first and last valid points
    first_valid_idx = np.min(valid_indices)
    last_valid_idx = np.max(valid_indices)
    
    # Find points that need interpolation (between first and last valid)
    interpolate_mask = (time_indices >= first_valid_idx) & (time_indices <= last_valid_idx) & (~valid_mask)
    
    if np.any(interpolate_mask):
        # Use numpy's interp function for linear interpolation
        interpolated_values = np.interp(
            time_indices[interpolate_mask],
            valid_indices,
            valid_values
        )
        result[interpolate_mask] = interpolated_values
    
    return result


def interpolate_raster_cube(input_path, output_path, nodata_value=None, chunk_size=512):
    """
    Interpolate invalid values in a raster datacube along the time dimension.
    
    Args:
        input_path: path to input raster file
        output_path: path to output raster file
        nodata_value: specific nodata value to treat as invalid
        chunk_size: size of chunks for processing (to manage memory)
    """
    logger.info(f"Starting interpolation of {input_path}")
    
    with rasterio.open(input_path) as src:
        # Get raster properties
        profile = src.profile.copy()
        height, width = src.height, src.width
        count = src.count  # number of bands (timesteps)
        
        logger.info(f"Raster dimensions: {count} bands, {height}H x {width}W")
        logger.info(f"Data type: {profile['dtype']}")
        logger.info(f"Original nodata value: {profile.get('nodata', 'None')}")
        
        # Update profile for output
        output_nodata = nodata_value if nodata_value is not None else profile.get('nodata', -9999)
        profile['nodata'] = output_nodata
            
        # Use float32 for processing to handle interpolated values
        if profile['dtype'] in ['uint8', 'uint16', 'int16', 'int32']:
            profile['dtype'] = 'float32'
        
        # Create output file
        with rasterio.open(output_path, 'w', **profile) as dst:
            
            # Process data in chunks to manage memory
            total_pixels = 0
            interpolated_pixels = 0
            invalid_pixels = 0
            
            for row_start in range(0, height, chunk_size):
                row_end = min(row_start + chunk_size, height)
                
                for col_start in range(0, width, chunk_size):
                    col_end = min(col_start + chunk_size, width)
                    
                    # Read chunk data for all bands
                    window = rasterio.windows.Window(col_start, row_start, 
                                                   col_end - col_start, row_end - row_start)
                    
                    # Read all bands for this window
                    data_chunk = src.read(window=window, masked=False)  # Shape: (bands, height, width)
                    
                    # Convert to float32 for processing
                    data_chunk = data_chunk.astype(np.float32)
                    
                    # Transpose to (height, width, bands) for easier processing
                    data_chunk = data_chunk.transpose(1, 2, 0)
                    
                    chunk_height, chunk_width, bands = data_chunk.shape
                    
                    # Process each pixel's timeseries
                    for h in range(chunk_height):
                        for w in range(chunk_width):
                            pixel_timeseries = data_chunk[h, w, :].copy()
                            original_timeseries = pixel_timeseries.copy()
                            
                            # Check if pixel has any valid values
                            valid_mask = ~is_invalid_value(pixel_timeseries, src.nodata)
                            
                            total_pixels += 1
                            
                            if np.sum(valid_mask) <= 1:
                                # Pixel has 0 or 1 valid values - cannot interpolate
                                invalid_pixels += 1
                                # Set entire timeseries to nodata
                                pixel_timeseries[:] = output_nodata
                            else:
                                # Interpolate the timeseries
                                interpolated_series = interpolate_timeseries(pixel_timeseries)
                                
                                # Check if any interpolation actually happened
                                if not np.array_equal(original_timeseries, interpolated_series):
                                    interpolated_pixels += 1
                                
                                pixel_timeseries = interpolated_series
                            
                            data_chunk[h, w, :] = pixel_timeseries
                    
                    # Transpose back to (bands, height, width) and write
                    data_chunk = data_chunk.transpose(2, 0, 1)
                    dst.write(data_chunk, window=window)
                
                # Progress logging
                progress = ((row_end) / height) * 100
                logger.info(f"Progress: {progress:.1f}%")
            
            logger.info(f"Interpolation complete!")
            logger.info(f"Total pixels processed: {total_pixels}")
            logger.info(f"Pixels with interpolation: {interpolated_pixels}")
            logger.info(f"Pixels remaining invalid: {invalid_pixels}")
            logger.info(f"Output saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Interpolate invalid values in raster datacubes')
    parser.add_argument('input_file', help='Input raster file path')
    parser.add_argument('output_file', help='Output raster file path') 
    parser.add_argument('--nodata_value', type=float, default=None,
                       help='Specific nodata value to treat as invalid (default: use file nodata)')
    parser.add_argument('--chunk_size', type=int, default=512,
                       help='Chunk size for processing (default: 512)')
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
        interpolate_raster_cube(
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