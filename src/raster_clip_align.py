#!/usr/bin/env python3
"""
Raster Clipping and Pixel Alignment Script

This script handles clipping raster data with AOI boundaries and aligning pixels.
Automatically detects appropriate UTM zone based on AOI location and determines
optimal pixel size from input raster.

Features:
- Automatic UTM zone detection from AOI centroid
- Automatic pixel size determination from input raster
- Support for any geographic location worldwide
- Flexible processing modes (single, batch, zone, config)

Author: Generated for Raster Processing
Date: 2025-09-03
"""

import rasterio as rio
import numpy as np
import os
import geopandas as gpd
import rasterio.mask
from rasterio.warp import reproject, Resampling
from rasterio.crs import CRS
from typing import Tuple, Optional, Union
import argparse
import yaml
from pathlib import Path
import logging
import math

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def get_utm_zone_from_aoi(aoi_gdf: gpd.GeoDataFrame) -> str:
    """
    Determine the appropriate UTM zone CRS based on AOI centroid.
    
    Args:
        aoi_gdf: GeoDataFrame containing AOI polygons
        
    Returns:
        EPSG code string for the appropriate UTM zone
    """
    # Ensure AOI is in WGS84 for centroid calculation
    if aoi_gdf.crs != 'EPSG:4326':
        aoi_wgs84 = aoi_gdf.to_crs('EPSG:4326')
    else:
        aoi_wgs84 = aoi_gdf
    
    # Get centroid of the union of all geometries
    union_geom = aoi_wgs84.geometry.union_all()
    centroid = union_geom.centroid
    
    # Extract longitude and latitude
    lon = centroid.x
    lat = centroid.y
    
    # Calculate UTM zone number
    utm_zone = int((lon + 180) / 6) + 1
    
    # Determine hemisphere (North or South)
    if lat >= 0:
        # Northern hemisphere
        epsg_code = f'EPSG:{32600 + utm_zone}'
    else:
        # Southern hemisphere  
        epsg_code = f'EPSG:{32700 + utm_zone}'
    
    logger.info(f"AOI centroid: ({lon:.6f}, {lat:.6f}), determined UTM zone: {epsg_code}")
    return epsg_code


def auto_determine_pixel_size(raster_path: str) -> float:
    """
    Automatically determine appropriate pixel size from input raster.
    
    Args:
        raster_path: Path to input raster file
        
    Returns:
        Pixel size in CRS units
    """
    with rio.open(raster_path) as src:
        # Get pixel size from transform (assumes square pixels)
        transform = src.transform
        pixel_size_x = abs(transform.a)
        pixel_size_y = abs(transform.e)
        pixel_size = min(pixel_size_x, pixel_size_y)
        
        logger.info(f"Auto-determined pixel size: {pixel_size:.2f} (from raster {os.path.basename(raster_path)})")
        return pixel_size


def raster_pixels_alignment_with_clip(raster_path: str, 
                                    aoi_areas: gpd.GeoDataFrame, 
                                    out_dir_cube: Optional[str] = None,
                                    target_crs: Optional[str] = None,
                                    pixel_size: Optional[float] = None,
                                    resampling_method: Resampling = Resampling.bilinear) -> Tuple[np.ndarray, dict, Optional[str]]:
    """
    Clip raster with AOI and align pixels to a standardized grid.
    Automatically determines UTM zone and pixel size if not provided.
    
    Args:
        raster_path: Path to input raster file
        aoi_areas: GeoDataFrame containing AOI polygons
        out_dir_cube: Output directory for aligned raster (optional)
        target_crs: Target coordinate reference system (auto-determined if None)
        pixel_size: Target pixel size in CRS units (auto-determined if None)
        resampling_method: Resampling method for reprojection
        
    Returns:
        Tuple of (aligned_data, metadata, output_path)
    """
    logger.info(f"Processing raster: {raster_path}")
    
    # Auto-determine target CRS if not provided
    if target_crs is None:
        target_crs = get_utm_zone_from_aoi(aoi_areas)
        logger.info(f"Auto-determined target CRS: {target_crs}")
    
    # Auto-determine pixel size if not provided
    if pixel_size is None:
        pixel_size = auto_determine_pixel_size(raster_path)
        logger.info(f"Auto-determined pixel size: {pixel_size}")
    
    # Ensure AOI is in target CRS
    if aoi_areas.crs != target_crs:
        logger.info(f"Reprojecting AOI from {aoi_areas.crs} to {target_crs}")
        aoi_areas = aoi_areas.to_crs(target_crs)
    
    # Get bounding box for pixel alignment
    bbox = aoi_areas.total_bounds
    x_min, y_min, x_max, y_max = bbox
    
    # Calculate aligned grid dimensions
    num_pixels_x = int(np.ceil((x_max - x_min) / pixel_size))
    num_pixels_y = int(np.ceil((y_max - y_min) / pixel_size))
    
    # Create aligned transform
    align_transform = rio.transform.from_bounds(*bbox, width=num_pixels_x, height=num_pixels_y)
    
    target_crs_obj = CRS.from_string(target_crs)
    
    # Read and clip the raster
    with rio.open(raster_path) as src:
        logger.info(f"Original raster CRS: {src.crs}, Shape: {src.shape}")
        
        # Reproject AOI to raster CRS for clipping
        if aoi_areas.crs != src.crs:
            aoi_clip = aoi_areas.to_crs(src.crs)
        else:
            aoi_clip = aoi_areas
        
        # Clip raster with AOI
        try:
            clipped_data, clipped_transform = rio.mask.mask(
                src, aoi_clip.geometry, crop=True, nodata=src.nodata, all_touched=True
            )
            logger.info(f"Clipped raster shape: {clipped_data.shape}")
        except Exception as e:
            logger.warning(f"Clipping failed, using original raster: {e}")
            clipped_data = src.read()
            clipped_transform = src.transform
        
        # Get original metadata
        src_crs = src.crs
        original_meta = src.meta.copy()
    
    # Prepare for reprojection to aligned grid
    num_bands = clipped_data.shape[0]
    dst_data = np.empty((num_bands, num_pixels_y, num_pixels_x), dtype=clipped_data.dtype)
    
    # Reproject each band
    for b in range(num_bands):
        reproject(
            source=clipped_data[b],
            destination=dst_data[b],
            src_transform=clipped_transform,
            src_crs=src_crs,
            dst_transform=align_transform,
            dst_crs=target_crs_obj,
            resampling=resampling_method,
            num_threads=2
        )
    
    # Update metadata
    aligned_meta = original_meta.copy()
    aligned_meta.update({
        'crs': target_crs_obj,
        'transform': align_transform,
        'width': num_pixels_x,
        'height': num_pixels_y,
        'count': num_bands
    })
    
    logger.info(f"Aligned raster shape: {dst_data.shape}")
    
    # Save if output directory provided
    out_path = None
    if out_dir_cube:
        os.makedirs(out_dir_cube, exist_ok=True)
        out_path = os.path.join(out_dir_cube, f'{os.path.basename(raster_path)}')
        
        with rio.open(out_path, 'w', **aligned_meta) as dst:
            dst.write(dst_data)
        
        logger.info(f"Saved aligned raster to: {out_path}")
    
    return dst_data, aligned_meta, out_path


def batch_process_rasters(input_dir: str,
                         aoi_path: str,
                         output_dir: str,
                         pattern: str = "*.tif",
                         **alignment_kwargs):
    """
    Batch process multiple rasters in a directory.
    
    Args:
        input_dir: Directory containing input rasters
        aoi_path: Path to AOI file (parquet, shapefile, etc.)
        output_dir: Output directory for processed rasters
        pattern: File pattern to match
        **alignment_kwargs: Additional arguments for alignment function
    """
    # Load AOI
    if aoi_path.endswith('.parquet'):
        aoi_areas = gpd.read_parquet(aoi_path)
    else:
        aoi_areas = gpd.read_file(aoi_path)
    
    logger.info(f"Loaded AOI with {len(aoi_areas)} features, CRS: {aoi_areas.crs}")
    
    # Find raster files
    input_path = Path(input_dir)
    raster_files = list(input_path.glob(pattern))
    
    if not raster_files:
        logger.warning(f"No raster files found matching pattern '{pattern}' in {input_dir}")
        return
    
    logger.info(f"Found {len(raster_files)} raster files to process")
    
    # Process each raster
    successful = 0
    failed = 0
    
    for raster_file in raster_files:
        try:
            logger.info(f"Processing: {raster_file.name}")
            
            # Create output subdirectory if needed
            relative_path = raster_file.relative_to(input_path)
            output_subdir = Path(output_dir) / relative_path.parent
            
            aligned_data, aligned_meta, out_path = raster_pixels_alignment_with_clip(
                str(raster_file),
                aoi_areas,
                str(output_subdir),
                **alignment_kwargs
            )
            
            successful += 1
            logger.info(f"Successfully processed: {raster_file.name}")
            
        except Exception as e:
            failed += 1
            logger.error(f"Failed to process {raster_file.name}: {e}")
            continue
    
    logger.info(f"Batch processing completed: {successful} successful, {failed} failed")


def process_zone_rasters(zone: str,
                        base_gdrive_dir: str,
                        base_output_dir: str,
                        aoi_base_path: str):
    """
    Process all rasters for a specific zone.
    
    Args:
        zone: Zone identifier (e.g., 'zone_1')
        base_gdrive_dir: Base directory containing rasters
        base_output_dir: Base output directory for processed data
        aoi_base_path: Base path for AOI files
    """
    logger.info(f"Processing zone: {zone}")
    
    # Setup paths
    aoi_path = f"{aoi_base_path}/{zone}.parquet"
    output_dir = f"{base_output_dir}/{zone}/"
    
    # Load AOI
    try:
        aoi_areas = gpd.read_parquet(aoi_path)
        logger.info(f"Loaded AOI for {zone}: {len(aoi_areas)} features, CRS: {aoi_areas.crs}")
    except Exception as e:
        logger.error(f"Failed to load AOI for {zone}: {e}")
        return
    
    # Find and process raster directories
    gdrive_path = Path(base_gdrive_dir)
    processed_count = 0
    
    for subdir in gdrive_path.iterdir():
        if not subdir.is_dir():
            continue
            
        logger.info(f"Checking directory: {subdir.name}")
        
        # Look for raster files
        raster_files = list(subdir.glob("*.tif"))
        if not raster_files:
            continue
        
        logger.info(f"Found {len(raster_files)} raster files in {subdir.name}")
        
        # Process each raster type
        raster_types = {}
        for raster_file in raster_files:
            # Extract raster type from filename (e.g., 'vh', 'vv', 'ndvi', etc.)
            filename = raster_file.stem
            if 'vh' in filename.lower():
                raster_type = 'vh'
            elif 'vv' in filename.lower():
                raster_type = 'vv'
            elif 'ndvi' in filename.lower():
                raster_type = 'ndvi'
            elif 'gvmi' in filename.lower():
                raster_type = 'gvmi'
            elif 'slavi' in filename.lower():
                raster_type = 'slavi'
            else:
                raster_type = 'other'
            
            if raster_type not in raster_types:
                raster_types[raster_type] = []
            raster_types[raster_type].append(raster_file)
        
        # Process each raster type
        for raster_type, files in raster_types.items():
            output_subdir = f"{output_dir}/{raster_type}/"
            os.makedirs(output_subdir, exist_ok=True)
            
            for raster_file in files:
                try:
                    logger.info(f"Processing {raster_type}: {raster_file.name}")
                    
                    aligned_data, aligned_meta, out_path = raster_pixels_alignment_with_clip(
                        str(raster_file),
                        aoi_areas,
                        output_subdir
                    )
                    
                    processed_count += 1
                    logger.info(f"Successfully processed: {raster_file.name}")
                    
                except Exception as e:
                    logger.error(f"Failed to process {raster_file.name}: {e}")
                    continue
    
    logger.info(f"Completed processing zone {zone}: {processed_count} rasters processed")


def create_config_template(output_path: str):
    """Create a YAML configuration template."""
    config = {
        'processing': {
            'target_crs': None,  # Auto-determine from AOI
            'pixel_size': None,  # Auto-determine from raster
            'resampling_method': 'bilinear'
        },
        'paths': {
            'input_dir': '/path/to/raster/directory/',
            'output_dir': '/path/to/output/directory/',
            'aoi_base_path': '/path/to/aoi/files/'
        },
        'zones': ['zone_1', 'zone_2'],
        'raster_types': ['band1', 'band2', 'ndvi']
    }
    
    with open(output_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    logger.info(f"Configuration template created: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Raster Clipping and Pixel Alignment',
        epilog='''
Examples:
  # Process single raster (auto-detect UTM zone and pixel size)
  python raster_clip_align.py --mode single --raster input.tif --aoi boundary.shp --output output_dir/
  
  # Process single raster with specific CRS and pixel size
  python raster_clip_align.py --mode single --raster input.tif --aoi boundary.parquet --output output_dir/ --target-crs EPSG:32633 --pixel-size 30.0
  
  # Batch process directory
  python raster_clip_align.py --mode batch --input-dir raster_dir/ --aoi study_area.geojson --output processed_dir/ --pattern "*.tif"
  
  # Create configuration template
  python raster_clip_align.py --mode config --config config.yaml
        ''',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('--mode', choices=['single', 'batch', 'zone', 'config'], 
                       required=True, help='Processing mode')
    
    # Single raster processing
    parser.add_argument('--raster', help='Path to input raster file')
    parser.add_argument('--aoi', help='Path to AOI file')
    parser.add_argument('--output', help='Output directory')
    
    # Batch processing
    parser.add_argument('--input-dir', help='Input directory for batch processing')
    parser.add_argument('--pattern', default='*.tif', help='File pattern for batch processing')
    
    # Zone processing
    parser.add_argument('--zone', help='Zone identifier for zone processing')
    
    # Configuration
    parser.add_argument('--config', help='Path to configuration file')
    
    # Processing parameters
    parser.add_argument('--target-crs', default=None, help='Target CRS (auto-determined if not specified)')
    parser.add_argument('--pixel-size', type=float, default=None, help='Target pixel size (auto-determined if not specified)')
    parser.add_argument('--resampling', default='bilinear', 
                       choices=['nearest', 'bilinear', 'cubic', 'average'],
                       help='Resampling method')
    
    args = parser.parse_args()
    
    # Map resampling method
    resampling_map = {
        'nearest': Resampling.nearest,
        'bilinear': Resampling.bilinear,
        'cubic': Resampling.cubic,
        'average': Resampling.average
    }
    resampling_method = resampling_map[args.resampling]
    
    try:
        if args.mode == 'single':
            if not all([args.raster, args.aoi, args.output]):
                raise ValueError("Single mode requires --raster, --aoi, and --output")
            
            aoi_areas = gpd.read_parquet(args.aoi) if args.aoi.endswith('.parquet') else gpd.read_file(args.aoi)
            
            aligned_data, aligned_meta, out_path = raster_pixels_alignment_with_clip(
                args.raster, aoi_areas, args.output,
                target_crs=args.target_crs,
                pixel_size=args.pixel_size,
                resampling_method=resampling_method
            )
            
            logger.info(f"Single raster processing completed: {out_path}")
        
        elif args.mode == 'batch':
            if not all([args.input_dir, args.aoi, args.output]):
                raise ValueError("Batch mode requires --input-dir, --aoi, and --output")
            
            batch_process_rasters(
                args.input_dir, args.aoi, args.output, args.pattern,
                target_crs=args.target_crs,
                pixel_size=args.pixel_size,
                resampling_method=resampling_method
            )
        
        elif args.mode == 'zone':
            if not all([args.zone, args.input_dir, args.output, args.aoi]):
                raise ValueError("Zone mode requires --zone, --input-dir, --output, and --aoi")
            
            # Extract base path from AOI argument
            aoi_base_path = str(Path(args.aoi).parent)
            
            process_zone_rasters(
                args.zone, 
                args.input_dir, 
                args.output, 
                aoi_base_path
            )
        
        elif args.mode == 'config':
            if not args.config:
                raise ValueError("Config mode requires --config")
            
            create_config_template(args.config)
        
        logger.info("Processing completed successfully")
        
    except Exception as e:
        logger.error(f"Processing failed: {e}")
        raise


if __name__ == "__main__":
    main()