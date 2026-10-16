#!/usr/bin/env python3
"""
Check Raw Biophysical Data Before Weekly Aggregation
Analyze daily FCOVER data for 2022 to identify where NoData is introduced
"""

import rasterio
import numpy as np
from pathlib import Path
from datetime import datetime
import pandas as pd

def analyze_daily_raster(raster_path, sample_size=2000):
    """Analyze a single daily biophysical raster for NoData patterns."""
    try:
        with rasterio.open(raster_path) as src:
            # Read the data (should be single band for daily files)
            data = src.read(1)
            
            # Sample random pixels for efficiency
            flat_data = data.flatten()
            if len(flat_data) > sample_size:
                sample_indices = np.random.choice(len(flat_data), sample_size, replace=False)
                sample_data = flat_data[sample_indices]
            else:
                sample_data = flat_data
            
            # Analyze data quality
            nodata_count = np.sum(sample_data == -9999)
            nan_count = np.sum(np.isnan(sample_data))
            zero_count = np.sum(sample_data == 0)
            
            # Valid data (excluding NoData, NaN, but including 0 as valid for FCOVER)
            valid_mask = (sample_data != -9999) & ~np.isnan(sample_data)
            valid_data = sample_data[valid_mask]
            
            # Additional quality checks
            negative_count = np.sum(sample_data < 0) - nodata_count  # Negative but not -9999
            above_one_count = np.sum(sample_data > 1.0)  # FCOVER should be 0-1
            
            return {
                'file': raster_path.name,
                'date': extract_date_from_filename(raster_path.name),
                'total_sampled': len(sample_data),
                'nodata_count': nodata_count,
                'nodata_percentage': nodata_count / len(sample_data) * 100,
                'nan_count': nan_count,
                'zero_count': zero_count,
                'valid_count': len(valid_data),
                'valid_percentage': len(valid_data) / len(sample_data) * 100,
                'negative_count': negative_count,
                'above_one_count': above_one_count,
                'valid_range': (valid_data.min(), valid_data.max()) if len(valid_data) > 0 else (None, None),
                'valid_mean': valid_data.mean() if len(valid_data) > 0 else None,
                'shape': data.shape,
                'dtype': str(data.dtype)
            }
    except Exception as e:
        return {
            'file': raster_path.name,
            'error': str(e)
        }

def extract_date_from_filename(filename):
    """Extract date from filename like s2_fcover_2022_01_15.tif"""
    try:
        parts = filename.split('_')
        if len(parts) >= 5:
            year = int(parts[2])
            month = int(parts[3])
            day = int(parts[4].split('.')[0])
            return datetime(year, month, day)
    except:
        pass
    return None

def check_raw_fcover_data(base_dir, year=2022):
    """Check all daily FCOVER files for a specific year."""
    zone_name = Path(base_dir).name
    fcover_dir = Path(base_dir) / "fcover"
    
    print(f"🔍 Checking Raw FCOVER Data - {zone_name}")
    print(f"📁 Directory: {fcover_dir}")
    print(f"📅 Year: {year}")
    print("="*80)
    
    if not fcover_dir.exists():
        print(f"❌ FCOVER directory not found: {fcover_dir}")
        return None
    
    # Find all FCOVER files for the specified year
    pattern = f"s2_fcover_{year}_*.tif"
    fcover_files = list(fcover_dir.glob(pattern))
    fcover_files.sort()
    
    if not fcover_files:
        print(f"❌ No FCOVER files found for year {year}")
        print(f"   Pattern searched: {pattern}")
        # Show what files are actually there
        all_files = list(fcover_dir.glob("*.tif"))
        print(f"   Files found in directory: {len(all_files)}")
        if all_files:
            print(f"   Sample files: {[f.name for f in all_files[:5]]}")
        return None
    
    print(f"📊 Found {len(fcover_files)} FCOVER files for {year}")
    print(f"   Date range: {fcover_files[0].name} to {fcover_files[-1].name}")
    
    # Analyze each file
    all_results = []
    good_files = 0
    problematic_files = 0
    
    for i, file_path in enumerate(fcover_files):
        if i % 10 == 0:  # Progress indicator
            print(f"   Processing file {i+1}/{len(fcover_files)}: {file_path.name}")
        
        result = analyze_daily_raster(file_path)
        all_results.append(result)
        
        if 'error' not in result:
            if result['nodata_percentage'] < 10:  # Less than 10% NoData is considered good
                good_files += 1
            else:
                problematic_files += 1
    
    # Summary statistics
    valid_results = [r for r in all_results if 'error' not in r]
    
    if not valid_results:
        print("❌ No files could be processed successfully")
        return None
    
    # Calculate statistics
    nodata_percentages = [r['nodata_percentage'] for r in valid_results]
    valid_percentages = [r['valid_percentage'] for r in valid_results]
    
    print(f"\n📈 SUMMARY STATISTICS:")
    print(f"   Files processed: {len(valid_results)}")
    print(f"   Files with <10% NoData: {good_files}")
    print(f"   Files with ≥10% NoData: {problematic_files}")
    print(f"   Average NoData percentage: {np.mean(nodata_percentages):.1f}%")
    print(f"   Median NoData percentage: {np.median(nodata_percentages):.1f}%")
    print(f"   Min NoData percentage: {np.min(nodata_percentages):.1f}%")
    print(f"   Max NoData percentage: {np.max(nodata_percentages):.1f}%")
    
    # Find best and worst files
    best_file = min(valid_results, key=lambda x: x['nodata_percentage'])
    worst_file = max(valid_results, key=lambda x: x['nodata_percentage'])
    
    print(f"\n🏆 BEST FILE (least NoData):")
    print(f"   {best_file['file']}: {best_file['nodata_percentage']:.1f}% NoData, {best_file['valid_percentage']:.1f}% valid")
    if best_file['valid_range'][0] is not None:
        print(f"   Valid range: [{best_file['valid_range'][0]:.4f}, {best_file['valid_range'][1]:.4f}]")
        print(f"   Valid mean: {best_file['valid_mean']:.4f}")
    
    print(f"\n🔴 WORST FILE (most NoData):")
    print(f"   {worst_file['file']}: {worst_file['nodata_percentage']:.1f}% NoData, {worst_file['valid_percentage']:.1f}% valid")
    if worst_file['valid_range'][0] is not None:
        print(f"   Valid range: [{worst_file['valid_range'][0]:.4f}, {worst_file['valid_range'][1]:.4f}]")
        print(f"   Valid mean: {worst_file['valid_mean']:.4f}")
    
    # Check for temporal patterns
    print(f"\n📅 TEMPORAL PATTERN ANALYSIS:")
    monthly_stats = {}
    for result in valid_results:
        if result['date']:
            month = result['date'].month
            if month not in monthly_stats:
                monthly_stats[month] = []
            monthly_stats[month].append(result['nodata_percentage'])
    
    if monthly_stats:
        print("   Average NoData by month:")
        for month in sorted(monthly_stats.keys()):
            avg_nodata = np.mean(monthly_stats[month])
            file_count = len(monthly_stats[month])
            month_name = datetime(2022, month, 1).strftime('%B')
            print(f"   {month_name:>9}: {avg_nodata:5.1f}% NoData (n={file_count})")
    
    # Quality issues summary
    files_with_negatives = sum(1 for r in valid_results if r['negative_count'] > 0)
    files_with_above_one = sum(1 for r in valid_results if r['above_one_count'] > 0)
    
    print(f"\n⚠️  QUALITY ISSUES:")
    print(f"   Files with negative FCOVER values: {files_with_negatives}")
    print(f"   Files with FCOVER > 1.0: {files_with_above_one}")
    
    return {
        'zone': zone_name,
        'year': year,
        'total_files': len(fcover_files),
        'processed_files': len(valid_results),
        'good_files': good_files,
        'problematic_files': problematic_files,
        'avg_nodata_pct': np.mean(nodata_percentages),
        'results': all_results,
        'best_file': best_file,
        'worst_file': worst_file,
        'monthly_stats': monthly_stats
    }

def main():
    """Check raw biophysical data before temporal aggregation."""
    base_dir = "/fs2/ftype_mflf/shared_data/temp/biophys/2025/1.0.0/mfl_cmf_area_project_2025_chr_zone_1"
    year = 2022
    
    print("🔍 RAW BIOPHYSICAL DATA INVESTIGATION")
    print("Before Weekly Temporal Aggregation")
    print("="*80)
    print(f"📁 Base directory: {base_dir}")
    print(f"🎯 Feature: FCOVER")
    print(f"📅 Year: {year}")
    print(f"🔬 Analysis: Daily raster files")
    
    if not Path(base_dir).exists():
        print(f"❌ Base directory not found: {base_dir}")
        return
    
    # Run the analysis
    results = check_raw_fcover_data(base_dir, year)
    
    if results:
        print(f"\n🎯 CONCLUSION:")
        if results['avg_nodata_pct'] < 10:
            print(f"   ✅ Raw daily data looks GOOD: {results['avg_nodata_pct']:.1f}% average NoData")
            print(f"   💡 NoData issue likely introduced during WEEKLY AGGREGATION")
        elif results['avg_nodata_pct'] < 50:
            print(f"   ⚠️  Raw daily data has MODERATE issues: {results['avg_nodata_pct']:.1f}% average NoData")
            print(f"   💡 Problem exists in DAILY DATA but gets worse during aggregation")
        else:
            print(f"   🔴 Raw daily data has SEVERE issues: {results['avg_nodata_pct']:.1f}% average NoData")
            print(f"   💡 Problem starts at the SOURCE DATA level")
        
        print(f"\n📊 FILES SUMMARY:")
        print(f"   Good files (<10% NoData): {results['good_files']}/{results['processed_files']}")
        print(f"   Problematic files (≥10% NoData): {results['problematic_files']}/{results['processed_files']}")
        
        print(f"\n💡 RECOMMENDED NEXT STEPS:")
        if results['avg_nodata_pct'] < 10:
            print("   1. Investigate weekly temporal aggregation code")
            print("   2. Check gap-filling and interpolation algorithms")
            print("   3. Verify temporal alignment in aggregation process")
        else:
            print("   1. Check source Sentinel-2 data availability")
            print("   2. Verify cloud masking parameters")
            print("   3. Investigate biophysical processor inputs")
    else:
        print("❌ Analysis failed - check directory structure and file patterns")

if __name__ == "__main__":
    main()