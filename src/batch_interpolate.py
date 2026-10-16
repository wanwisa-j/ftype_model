#!/usr/bin/env python3
"""
Batch processing script for interpolating multiple raster datacubes.

This script processes multiple raster files in a directory, applying linear interpolation
to handle invalid values (NaN, -inf, inf, nodata) along the time dimension.

Usage:
    source /opt/miniconda3/etc/profile.d/conda.sh
    conda activate gisworkspace-cupy
    python batch_interpolate.py input_dir output_dir [--pattern "*.tif"] [--nodata_value VALUE]
    
Example:
    python batch_interpolate.py /fs2/ftype_mflf/shared_data/gdrive/GEE_Exports_mfl_cmf_area_project_2025_chr_zone_2/ /fs2/ftype_mflf/shared_data/interpolated/ --pattern "cube_*.tif"
"""

import argparse
import logging
from pathlib import Path
import sys
import subprocess
import glob

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def process_file(input_file, output_file, nodata_value=None, chunk_size=512):
    """
    Process a single raster file using the interpolation script.
    
    Args:
        input_file: path to input raster file
        output_file: path to output raster file
        nodata_value: specific nodata value to treat as invalid
        chunk_size: chunk size for processing
        
    Returns:
        True if successful, False if failed
    """
    try:
        # Build command using the ultra-fast version
        cmd = [
            'python', 
            '/fs1/tor_project/model_foresttype/src/raster_interpolation_ultra_fast.py',
            str(input_file),
            str(output_file),
            '--chunk_size', str(chunk_size)
        ]
        
        if nodata_value is not None:
            cmd.extend(['--nodata_value', str(nodata_value)])
            
        # Use bash explicitly and add conda activation
        full_cmd = f"/bin/bash -c 'source /opt/miniconda3/etc/profile.d/conda.sh && conda activate gisworkspace-cupy && {' '.join(cmd)}'"
        
        logger.info(f"Processing: {input_file.name}")
        result = subprocess.run(full_cmd, shell=True, capture_output=True, text=True, executable='/bin/bash')
        
        if result.returncode == 0:
            logger.info(f"Successfully processed: {input_file.name}")
            return True
        else:
            logger.error(f"Failed to process {input_file.name}: {result.stderr}")
            return False
            
    except Exception as e:
        logger.error(f"Error processing {input_file.name}: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description='Batch interpolate invalid values in raster datacubes')
    parser.add_argument('input_dir', help='Input directory containing raster files')
    parser.add_argument('output_dir', help='Output directory for processed files')
    parser.add_argument('--pattern', default='*.tif', help='File pattern to match (default: *.tif)')
    parser.add_argument('--nodata_value', type=float, default=None,
                       help='Specific nodata value to treat as invalid (default: use file nodata)')
    parser.add_argument('--chunk_size', type=int, default=512,
                       help='Chunk size for processing (default: 512)')
    parser.add_argument('--suffix', default='_interpolated',
                       help='Suffix to add to output filenames (default: _interpolated)')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose logging')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Validate input directory
    input_dir = Path(args.input_dir)
    if not input_dir.exists() or not input_dir.is_dir():
        logger.error(f"Input directory does not exist: {input_dir}")
        sys.exit(1)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find matching files
    pattern_path = input_dir / args.pattern
    input_files = list(input_dir.glob(args.pattern))
    
    if not input_files:
        logger.error(f"No files found matching pattern: {pattern_path}")
        sys.exit(1)
    
    logger.info(f"Found {len(input_files)} files to process")
    
    # Process each file
    successful = 0
    failed = 0
    
    for input_file in input_files:
        # Create output filename
        stem = input_file.stem
        suffix = input_file.suffix
        output_filename = f"{stem}{args.suffix}{suffix}"
        output_file = output_dir / output_filename
        
        # Skip if output already exists
        if output_file.exists():
            logger.info(f"Output file already exists, skipping: {output_filename}")
            continue
            
        # Process the file
        if process_file(input_file, output_file, args.nodata_value, args.chunk_size):
            successful += 1
        else:
            failed += 1
    
    logger.info(f"Batch processing complete!")
    logger.info(f"Successfully processed: {successful} files")
    logger.info(f"Failed to process: {failed} files")
    
    if failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()