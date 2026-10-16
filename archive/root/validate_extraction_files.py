#!/usr/bin/env python3
"""
Datacube and Extraction Files Validation Script

This script validates:
1. Datacube raster files - checks if they have the expected 21 bands
2. Extraction consistency - verifies temporal structure across zones and years
3. Identifies missing or corrupted data that causes shape mismatches

Addresses the specific issue:
- Zone 1 & 2 have 13 timesteps instead of 21 for VV/VH in 2023
- Expected: 21 bands per raster file
"""

import os
import sys
import rasterio
import numpy as np
import pandas as pd
from pathlib import Path
import logging
from collections import defaultdict

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def validate_raster_file(raster_path):
    """Validate a single raster file for band count and data integrity."""
    try:
        with rasterio.open(raster_path) as src:
            info = {
                'path': str(raster_path),
                'bands': src.count,
                'width': src.width,
                'height': src.height,
                'crs': str(src.crs),
                'nodata': src.nodata,
                'has_data': False,
                'valid_bands': 0,
                'issues': []
            }
            
            # Check if we have the expected 21 bands
            if src.count != 21:
                info['issues'].append(f"BAND_COUNT_MISMATCH: Expected 21 bands, found {src.count}")
            
            # Sample center pixel to check data availability
            center_row = src.height // 2
            center_col = src.width // 2
            
            valid_band_count = 0
            for band_idx in range(1, min(src.count + 1, 22)):  # Check up to 21 bands
                try:
                    band_data = src.read(band_idx, window=((center_row, center_row+1), (center_col, center_col+1)))
                    pixel_value = band_data[0, 0]
                    
                    # Check if pixel has valid data (not NaN, not nodata value)
                    if not np.isnan(pixel_value) and pixel_value != src.nodata:
                        valid_band_count += 1
                        info['has_data'] = True
                except Exception as e:
                    info['issues'].append(f"BAND_{band_idx}_READ_ERROR: {str(e)}")
            
            info['valid_bands'] = valid_band_count
            
            if valid_band_count == 0:
                info['issues'].append("NO_VALID_DATA: All bands appear to be empty or invalid")
            elif valid_band_count < 21:
                info['issues'].append(f"PARTIAL_DATA: Only {valid_band_count}/21 bands have valid data")
            
            return info
            
    except Exception as e:
        return {
            'path': str(raster_path),
            'bands': 0,
            'issues': [f"FILE_READ_ERROR: {str(e)}"],
            'has_data': False,
            'valid_bands': 0
        }

def validate_datacube_structure():
    """Validate the entire datacube structure."""
    
    logger.info("=== DATACUBE STRUCTURE VALIDATION ===")
    
    # Define zones and features to check
    base_path = Path("/fs2/ftype_mflf/shared_data/datacube")
    datacube_versions = ["2024/1.2.0", "2025/1.2.0"]
    zones = ["mfl_cmf_2024", "mfl_cmf_area_project_2025_chr_zone_1", "mfl_cmf_area_project_2025_chr_zone_2"]
    features = ['lai', 'fapar', 'fcover', 'ndvi', 'gvmi', 'slavi', 'vv', 'vh']
    years = [2019, 2020, 2021, 2022, 2023, 2024]
    
    validation_results = []
    issues_summary = defaultdict(list)
    
    for version in datacube_versions:
        version_path = base_path / version
        if not version_path.exists():
            logger.warning(f"Datacube version not found: {version_path}")
            continue
            
        logger.info(f"Checking datacube version: {version}")
        
        for zone in zones:
            zone_path = version_path / zone
            if not zone_path.exists():
                logger.warning(f"Zone not found: {zone_path}")
                continue
                
            logger.info(f"  Checking zone: {zone}")
            
            for feature in features:
                feature_path = zone_path / feature
                if not feature_path.exists():
                    logger.warning(f"  Feature directory not found: {feature_path}")
                    issues_summary[f"{zone}_{feature}"].append("DIRECTORY_MISSING")
                    continue
                
                # Find raster files for this feature
                raster_files = list(feature_path.glob("*.tif")) + list(feature_path.glob("*.tiff"))
                
                if not raster_files:
                    logger.warning(f"  No raster files found in: {feature_path}")
                    issues_summary[f"{zone}_{feature}"].append("NO_RASTER_FILES")
                    continue
                
                logger.info(f"    Checking feature: {feature} ({len(raster_files)} files)")
                
                for raster_file in raster_files:
                    # Extract year from filename if possible
                    filename = raster_file.name
                    file_year = None
                    for year in years:
                        if str(year) in filename:
                            file_year = year
                            break
                    
                    # Validate the raster file
                    validation_result = validate_raster_file(raster_file)
                    validation_result.update({
                        'datacube_version': version,
                        'zone': zone,
                        'feature': feature,
                        'year': file_year,
                        'filename': filename
                    })
                    
                    validation_results.append(validation_result)
                    
                    # Log issues
                    if validation_result['issues']:
                        key = f"{zone}_{feature}_{file_year}"
                        issues_summary[key].extend(validation_result['issues'])
                        
                        for issue in validation_result['issues']:
                            logger.warning(f"    {filename}: {issue}")
    
    return validation_results, issues_summary

def validate_specific_problematic_files():
    """Validate the specific files mentioned in the error message."""
    
    logger.info("=== SPECIFIC PROBLEMATIC FILES VALIDATION ===")
    
    # Files mentioned in the error
    problematic_files = [
        "/fs2/ftype_mflf/shared_data/datacube/2025/1.2.0/mfl_cmf_area_project_2025_chr_zone_1/vv/cube_s1_vv_2023_W4before_to_W16after_21bands.tif",
        "/fs2/ftype_mflf/shared_data/datacube/2025/1.2.0/mfl_cmf_area_project_2025_chr_zone_2/vv/cube_s1_vv_2023_W4before_to_W16after_21bands.tif",
        "/fs2/ftype_mflf/shared_data/datacube/2025/1.2.0/mfl_cmf_area_project_2025_chr_zone_1/vh/cube_s1_vh_2023_W4before_to_W16after_21bands.tif",
        "/fs2/ftype_mflf/shared_data/datacube/2025/1.2.0/mfl_cmf_area_project_2025_chr_zone_2/vh/cube_s1_vh_2023_W4before_to_W16after_21bands.tif"
    ]
    
    results = []
    
    for file_path in problematic_files:
        logger.info(f"Validating: {file_path}")
        
        if not os.path.exists(file_path):
            logger.error(f"FILE NOT FOUND: {file_path}")
            results.append({
                'path': file_path,
                'exists': False,
                'bands': 0,
                'issues': ['FILE_NOT_FOUND']
            })
            continue
        
        result = validate_raster_file(file_path)
        result['exists'] = True
        results.append(result)
        
        # Detailed analysis for these specific files
        logger.info(f"  File exists: ✅")
        logger.info(f"  Bands: {result['bands']}")
        logger.info(f"  Valid bands: {result['valid_bands']}")
        logger.info(f"  Has data: {result['has_data']}")
        
        if result['issues']:
            logger.error(f"  Issues found:")
            for issue in result['issues']:
                logger.error(f"    - {issue}")
        else:
            logger.info(f"  ✅ No issues found")
    
    return results

def analyze_extraction_consistency():
    """Analyze the extraction dataset for consistency issues."""
    
    logger.info("=== EXTRACTION CONSISTENCY ANALYSIS ===")
    
    # Load the raw dataset
    dataset_path = "/fs2/ftype_mflf/shared_data/dataset/1.2.0/raw/forest_type_dataset_raw_20250903_104423.parquet"
    
    if not os.path.exists(dataset_path):
        logger.error(f"Dataset not found: {dataset_path}")
        return None
    
    logger.info(f"Loading dataset: {dataset_path}")
    df = pd.read_parquet(dataset_path)
    
    # Focus on the problematic features and zones
    radar_features = ['vv', 'vh']
    problematic_zones = ['mfl_cmf_area_project_2025_chr_zone_1', 'mfl_cmf_area_project_2025_chr_zone_2']
    problematic_year = 2023
    
    radar_data = df[df['feature'].isin(radar_features)]
    
    logger.info("Analyzing temporal structure by zone, year, and feature:")
    
    structure_analysis = []
    
    for zone in radar_data['zone'].unique():
        for year in radar_data['year'].unique():
            if pd.isna(year):
                continue
            year = int(year)
            
            for feature in radar_features:
                zone_year_feature_data = radar_data[
                    (radar_data['zone'] == zone) & 
                    (radar_data['year'] == year) & 
                    (radar_data['feature'] == feature)
                ]
                
                if len(zone_year_feature_data) == 0:
                    continue
                
                # Count unique weeks/bands
                weeks = zone_year_feature_data['week'].dropna().unique()
                bands = zone_year_feature_data['band'].dropna().unique()
                points = zone_year_feature_data['point_id'].unique()
                
                analysis_result = {
                    'zone': zone,
                    'year': year,
                    'feature': feature,
                    'total_records': len(zone_year_feature_data),
                    'unique_weeks': len(weeks),
                    'unique_bands': len(bands),
                    'unique_points': len(points),
                    'week_range': f"{min(weeks)}-{max(weeks)}" if len(weeks) > 0 else "N/A",
                    'band_range': f"{min(bands)}-{max(bands)}" if len(bands) > 0 else "N/A",
                    'is_problematic': zone in problematic_zones and year == problematic_year and feature in radar_features
                }
                
                structure_analysis.append(analysis_result)
                
                # Log problematic cases
                if analysis_result['is_problematic']:
                    logger.warning(f"PROBLEMATIC: {zone}, {year}, {feature}")
                    logger.warning(f"  Records: {analysis_result['total_records']}")
                    logger.warning(f"  Weeks: {analysis_result['unique_weeks']} ({analysis_result['week_range']})")
                    logger.warning(f"  Bands: {analysis_result['unique_bands']} ({analysis_result['band_range']})")
                    logger.warning(f"  Points: {analysis_result['unique_points']}")
                    
                    if analysis_result['unique_weeks'] != 21 or analysis_result['unique_bands'] != 21:
                        logger.error(f"  ❌ SHAPE_MISMATCH: Expected 21, found {analysis_result['unique_weeks']} weeks, {analysis_result['unique_bands']} bands")
    
    return structure_analysis

def generate_validation_report(datacube_results, specific_files_results, extraction_analysis):
    """Generate a comprehensive validation report."""
    
    logger.info("=== GENERATING VALIDATION REPORT ===")
    
    report = {
        'summary': {
            'total_files_checked': len(datacube_results) if datacube_results else 0,
            'files_with_issues': len([r for r in datacube_results if r['issues']]) if datacube_results else 0,
            'specific_problematic_files': len(specific_files_results),
            'problematic_files_found': len([r for r in specific_files_results if r.get('exists', False)]),
        },
        'datacube_validation': datacube_results,
        'specific_files_validation': specific_files_results,
        'extraction_analysis': extraction_analysis
    }
    
    # Print summary
    logger.info(f"📊 VALIDATION SUMMARY:")
    logger.info(f"  Total files checked: {report['summary']['total_files_checked']}")
    logger.info(f"  Files with issues: {report['summary']['files_with_issues']}")
    logger.info(f"  Specific problematic files checked: {report['summary']['specific_problematic_files']}")
    logger.info(f"  Problematic files found: {report['summary']['problematic_files_found']}")
    
    # Identify key issues
    if specific_files_results:
        logger.info(f"\n🔍 SPECIFIC FILE ANALYSIS:")
        for result in specific_files_results:
            filename = Path(result['path']).name
            if result.get('exists', False):
                if result['bands'] != 21:
                    logger.error(f"  ❌ {filename}: {result['bands']} bands (expected 21)")
                else:
                    logger.info(f"  ✅ {filename}: {result['bands']} bands (correct)")
            else:
                logger.error(f"  ❌ {filename}: File not found")
    
    return report

def main():
    """Main validation function."""
    
    logger.info("🔍 DATACUBE AND EXTRACTION FILES VALIDATION")
    logger.info("=" * 60)
    
    # 1. Validate datacube structure
    logger.info("Step 1: Validating datacube structure...")
    datacube_results, issues_summary = validate_datacube_structure()
    
    # 2. Validate specific problematic files
    logger.info("Step 2: Validating specific problematic files...")
    specific_files_results = validate_specific_problematic_files()
    
    # 3. Analyze extraction consistency
    logger.info("Step 3: Analyzing extraction consistency...")
    extraction_analysis = analyze_extraction_consistency()
    
    # 4. Generate comprehensive report
    logger.info("Step 4: Generating validation report...")
    report = generate_validation_report(datacube_results, specific_files_results, extraction_analysis)
    
    # 5. Recommendations
    logger.info("\n🛠️  RECOMMENDATIONS:")
    logger.info("1. Check if problematic raster files actually have 21 bands")
    logger.info("2. Verify if extraction script correctly handles all bands")
    logger.info("3. Consider re-extracting data for problematic zones/years")
    logger.info("4. Update extraction script to handle missing/corrupted bands")
    
    return report

if __name__ == "__main__":
    report = main()
