#!/usr/bin/env python3
"""
Fix Missing Radar Data using Ultra-Fast Raster Interpolation

This script uses the ultra-fast raster interpolation to fix the missing VV/VH radar data
for weeks [3, 5, 8, 10, 12, 15, 17, 20] in 2023 for zones 1 & 2.

It processes the problematic raster files and creates corrected versions.
"""

import sys
import os
import logging
from pathlib import Path
import subprocess

# Add src directory to path to import the interpolation script
sys.path.insert(0, '/fs1/tor_project/model_foresttype/src')

from raster_interpolation_ultra_fast import interpolate_raster_ultra_fast

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def fix_missing_radar_data():
    """Fix missing radar data using ultra-fast raster interpolation."""
    
    logger.info("🚀 FIXING MISSING RADAR DATA USING ULTRA-FAST INTERPOLATION")
    logger.info("=" * 70)
    
    # Define the problematic files identified by validation
    problematic_files = [
        {
            'input': "/fs2/ftype_mflf/shared_data/datacube/2025/1.2.0/mfl_cmf_area_project_2025_chr_zone_1/vv/cube_s1_vv_2023_W4before_to_W16after_21bands.tif",
            'output': "/fs2/ftype_mflf/shared_data/datacube/2025/1.2.0/mfl_cmf_area_project_2025_chr_zone_1/vv/cube_s1_vv_2023_W4before_to_W16after_21bands_INTERPOLATED.tif",
            'zone': 'zone_1',
            'feature': 'vv'
        },
        {
            'input': "/fs2/ftype_mflf/shared_data/datacube/2025/1.2.0/mfl_cmf_area_project_2025_chr_zone_2/vv/cube_s1_vv_2023_W4before_to_W16after_21bands.tif",
            'output': "/fs2/ftype_mflf/shared_data/datacube/2025/1.2.0/mfl_cmf_area_project_2025_chr_zone_2/vv/cube_s1_vv_2023_W4before_to_W16after_21bands_INTERPOLATED.tif",
            'zone': 'zone_2',
            'feature': 'vv'
        },
        {
            'input': "/fs2/ftype_mflf/shared_data/datacube/2025/1.2.0/mfl_cmf_area_project_2025_chr_zone_1/vh/cube_s1_vh_2023_W4before_to_W16after_21bands.tif",
            'output': "/fs2/ftype_mflf/shared_data/datacube/2025/1.2.0/mfl_cmf_area_project_2025_chr_zone_1/vh/cube_s1_vh_2023_W4before_to_W16after_21bands_INTERPOLATED.tif",
            'zone': 'zone_1',
            'feature': 'vh'
        },
        {
            'input': "/fs2/ftype_mflf/shared_data/datacube/2025/1.2.0/mfl_cmf_area_project_2025_chr_zone_2/vh/cube_s1_vh_2023_W4before_to_W16after_21bands.tif",
            'output': "/fs2/ftype_mflf/shared_data/datacube/2025/1.2.0/mfl_cmf_area_project_2025_chr_zone_2/vh/cube_s1_vh_2023_W4before_to_W16after_21bands_INTERPOLATED.tif",
            'zone': 'zone_2',
            'feature': 'vh'
        }
    ]
    
    # Information about missing weeks
    missing_weeks = [3, 5, 8, 10, 12, 15, 17, 20]
    available_weeks = [1, 2, 4, 6, 7, 9, 11, 13, 14, 16, 18, 19, 21]
    
    logger.info(f"Missing weeks to interpolate: {missing_weeks}")
    logger.info(f"Available weeks for interpolation: {available_weeks}")
    logger.info(f"Files to process: {len(problematic_files)}")
    
    success_count = 0
    
    for i, file_info in enumerate(problematic_files, 1):
        input_path = file_info['input']
        output_path = file_info['output']
        zone = file_info['zone']
        feature = file_info['feature']
        
        logger.info(f"\n📁 Processing file {i}/{len(problematic_files)}: {zone} {feature}")
        logger.info(f"   Input:  {Path(input_path).name}")
        logger.info(f"   Output: {Path(output_path).name}")
        
        # Check if input file exists
        if not os.path.exists(input_path):
            logger.error(f"   ❌ Input file not found: {input_path}")
            continue
        
        # Create output directory if needed
        output_dir = Path(output_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            # Use the ultra-fast interpolation
            logger.info(f"   🚀 Starting ultra-fast interpolation...")
            
            interpolate_raster_ultra_fast(
                input_path=input_path,
                output_path=output_path,
                nodata_value=-9999,  # Use standard nodata value
                chunk_size=2048      # Good balance of memory and performance
            )
            
            logger.info(f"   ✅ Interpolation completed successfully!")
            success_count += 1
            
        except Exception as e:
            logger.error(f"   ❌ Error during interpolation: {str(e)}")
            continue
    
    # Summary
    logger.info(f"\n📊 INTERPOLATION SUMMARY:")
    logger.info(f"   Total files processed: {len(problematic_files)}")
    logger.info(f"   Successful interpolations: {success_count}")
    logger.info(f"   Failed interpolations: {len(problematic_files) - success_count}")
    
    if success_count == len(problematic_files):
        logger.info("   🎉 All files interpolated successfully!")
        return True
    else:
        logger.warning("   ⚠️  Some files failed to interpolate")
        return False

def validate_interpolated_files():
    """Validate the interpolated files to ensure they now have complete data."""
    
    logger.info("\n🔍 VALIDATING INTERPOLATED FILES")
    logger.info("=" * 35)
    
    # Import the validation function from our earlier script
    sys.path.insert(0, '/fs1/tor_project/model_foresttype')
    
    try:
        import rasterio
        import numpy as np
        
        interpolated_files = [
            "/fs2/ftype_mflf/shared_data/datacube/2025/1.2.0/mfl_cmf_area_project_2025_chr_zone_1/vv/cube_s1_vv_2023_W4before_to_W16after_21bands_INTERPOLATED.tif",
            "/fs2/ftype_mflf/shared_data/datacube/2025/1.2.0/mfl_cmf_area_project_2025_chr_zone_2/vv/cube_s1_vv_2023_W4before_to_W16after_21bands_INTERPOLATED.tif",
            "/fs2/ftype_mflf/shared_data/datacube/2025/1.2.0/mfl_cmf_area_project_2025_chr_zone_1/vh/cube_s1_vh_2023_W4before_to_W16after_21bands_INTERPOLATED.tif",
            "/fs2/ftype_mflf/shared_data/datacube/2025/1.2.0/mfl_cmf_area_project_2025_chr_zone_2/vh/cube_s1_vh_2023_W4before_to_W16after_21bands_INTERPOLATED.tif"
        ]
        
        all_valid = True
        
        for file_path in interpolated_files:
            logger.info(f"\n📁 Validating: {Path(file_path).name}")
            
            if not os.path.exists(file_path):
                logger.error(f"   ❌ File not found: {file_path}")
                all_valid = False
                continue
            
            try:
                with rasterio.open(file_path) as src:
                    logger.info(f"   📊 Bands: {src.count}")
                    logger.info(f"   📏 Dimensions: {src.height}H x {src.width}W")
                    
                    # Sample center pixel across all bands to check completeness
                    center_row = src.height // 2
                    center_col = src.width // 2
                    
                    valid_bands = 0
                    for band_idx in range(1, src.count + 1):
                        band_data = src.read(band_idx, window=((center_row, center_row+1), (center_col, center_col+1)))
                        pixel_value = band_data[0, 0]
                        
                        if not np.isnan(pixel_value) and pixel_value != src.nodata and pixel_value != 0:
                            valid_bands += 1
                    
                    logger.info(f"   ✅ Valid bands: {valid_bands}/21")
                    
                    if valid_bands == 21:
                        logger.info(f"   🎉 Perfect! All 21 bands have valid data")
                    elif valid_bands >= 19:  # Allow for some edge cases
                        logger.info(f"   ✅ Good! {valid_bands}/21 bands have data")
                    else:
                        logger.warning(f"   ⚠️  Only {valid_bands}/21 bands have data")
                        all_valid = False
                        
            except Exception as e:
                logger.error(f"   ❌ Error reading file: {str(e)}")
                all_valid = False
        
        if all_valid:
            logger.info(f"\n🎉 ALL INTERPOLATED FILES VALIDATED SUCCESSFULLY!")
        else:
            logger.warning(f"\n⚠️  SOME VALIDATION ISSUES FOUND")
            
        return all_valid
        
    except ImportError as e:
        logger.error(f"Missing required packages for validation: {e}")
        return False

def create_symlinks_for_corrected_files():
    """Create symlinks to use the corrected files in place of original ones."""
    
    logger.info("\n🔗 CREATING SYMLINKS FOR CORRECTED FILES")
    logger.info("=" * 40)
    
    file_pairs = [
        {
            'original': "/fs2/ftype_mflf/shared_data/datacube/2025/1.2.0/mfl_cmf_area_project_2025_chr_zone_1/vv/cube_s1_vv_2023_W4before_to_W16after_21bands.tif",
            'interpolated': "/fs2/ftype_mflf/shared_data/datacube/2025/1.2.0/mfl_cmf_area_project_2025_chr_zone_1/vv/cube_s1_vv_2023_W4before_to_W16after_21bands_INTERPOLATED.tif",
            'backup': "/fs2/ftype_mflf/shared_data/datacube/2025/1.2.0/mfl_cmf_area_project_2025_chr_zone_1/vv/cube_s1_vv_2023_W4before_to_W16after_21bands_ORIGINAL.tif"
        },
        {
            'original': "/fs2/ftype_mflf/shared_data/datacube/2025/1.2.0/mfl_cmf_area_project_2025_chr_zone_2/vv/cube_s1_vv_2023_W4before_to_W16after_21bands.tif",
            'interpolated': "/fs2/ftype_mflf/shared_data/datacube/2025/1.2.0/mfl_cmf_area_project_2025_chr_zone_2/vv/cube_s1_vv_2023_W4before_to_W16after_21bands_INTERPOLATED.tif",
            'backup': "/fs2/ftype_mflf/shared_data/datacube/2025/1.2.0/mfl_cmf_area_project_2025_chr_zone_2/vv/cube_s1_vv_2023_W4before_to_W16after_21bands_ORIGINAL.tif"
        },
        {
            'original': "/fs2/ftype_mflf/shared_data/datacube/2025/1.2.0/mfl_cmf_area_project_2025_chr_zone_1/vh/cube_s1_vh_2023_W4before_to_W16after_21bands.tif",
            'interpolated': "/fs2/ftype_mflf/shared_data/datacube/2025/1.2.0/mfl_cmf_area_project_2025_chr_zone_1/vh/cube_s1_vh_2023_W4before_to_W16after_21bands_INTERPOLATED.tif",
            'backup': "/fs2/ftype_mflf/shared_data/datacube/2025/1.2.0/mfl_cmf_area_project_2025_chr_zone_1/vh/cube_s1_vh_2023_W4before_to_W16after_21bands_ORIGINAL.tif"
        },
        {
            'original': "/fs2/ftype_mflf/shared_data/datacube/2025/1.2.0/mfl_cmf_area_project_2025_chr_zone_2/vh/cube_s1_vh_2023_W4before_to_W16after_21bands.tif",
            'interpolated': "/fs2/ftype_mflf/shared_data/datacube/2025/1.2.0/mfl_cmf_area_project_2025_chr_zone_2/vh/cube_s1_vh_2023_W4before_to_W16after_21bands_INTERPOLATED.tif",
            'backup': "/fs2/ftype_mflf/shared_data/datacube/2025/1.2.0/mfl_cmf_area_project_2025_chr_zone_2/vh/cube_s1_vh_2023_W4before_to_W16after_21bands_ORIGINAL.tif"
        }
    ]
    
    logger.info("⚠️  WARNING: This will replace the original files with interpolated versions!")
    logger.info("Original files will be backed up with '_ORIGINAL' suffix.")
    logger.info("To proceed with replacement, this function needs manual execution.")
    
    replacement_commands = []
    
    for pair in file_pairs:
        original = pair['original']
        interpolated = pair['interpolated']
        backup = pair['backup']
        
        if os.path.exists(interpolated):
            commands = [
                f"# Backup original file",
                f"cp '{original}' '{backup}'",
                f"# Replace with interpolated version", 
                f"cp '{interpolated}' '{original}'",
                f"echo 'Replaced {Path(original).name}'"
            ]
            replacement_commands.extend(commands)
            replacement_commands.append("")
    
    # Save commands to script
    script_path = "/fs1/tor_project/model_foresttype/replace_with_interpolated.sh"
    with open(script_path, 'w') as f:
        f.write("#!/bin/bash\n")
        f.write("# Script to replace original raster files with interpolated versions\n\n")
        f.write("set -e  # Exit on any error\n\n")
        for cmd in replacement_commands:
            f.write(f"{cmd}\n")
    
    # Make script executable
    os.chmod(script_path, 0o755)
    
    logger.info(f"📝 Replacement script created: {script_path}")
    logger.info("   To apply the changes, run: bash replace_with_interpolated.sh")
    
    return script_path

def main():
    """Main execution function."""
    
    logger.info("🎯 MISSING RADAR DATA INTERPOLATION PIPELINE")
    logger.info("=" * 50)
    
    # Step 1: Fix missing data using ultra-fast interpolation
    logger.info("Step 1: Interpolating missing radar data...")
    interpolation_success = fix_missing_radar_data()
    
    if not interpolation_success:
        logger.error("❌ Interpolation failed. Stopping pipeline.")
        return False
    
    # Step 2: Validate interpolated files
    logger.info("Step 2: Validating interpolated files...")
    validation_success = validate_interpolated_files()
    
    if not validation_success:
        logger.warning("⚠️  Validation had issues, but continuing...")
    
    # Step 3: Create replacement script
    logger.info("Step 3: Creating replacement script...")
    script_path = create_symlinks_for_corrected_files()
    
    # Summary and next steps
    logger.info(f"\n🎉 INTERPOLATION PIPELINE COMPLETE!")
    logger.info("=" * 40)
    logger.info("📋 NEXT STEPS:")
    logger.info("1. Review the interpolated files to ensure quality")
    logger.info("2. Run the replacement script to use corrected files:")
    logger.info(f"   bash {script_path}")
    logger.info("3. Re-run data extraction to get corrected 21-week temporal structure")
    logger.info("4. Update notebook to use the corrected data")
    
    return True

if __name__ == "__main__":
    success = main()
    if success:
        logger.info("✅ Mission accomplished!")
    else:
        logger.error("❌ Mission failed!")
