#!/usr/bin/env python3
"""
Final Dataset Validation Script
Comprehensive validation of the complete forest type dataset
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging
from datetime import datetime

def setup_logging():
    """Setup logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(f'/fs1/tor_project/model_foresttype/logs/dataset_validation_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def validate_dataset():
    """Comprehensive dataset validation"""
    logger = setup_logging()
    
    logger.info("🔍 Starting comprehensive dataset validation...")
    
    # Load datasets
    wide_path = '/fs2/ftype_mflf/shared_data/dataset/1.2.0/processed/forest_type_dataset_wide_combined_20250903_095530.parquet'
    raw_path = '/fs2/ftype_mflf/shared_data/dataset/1.2.0/raw/forest_type_dataset_raw_combined_20250903_095530.parquet'
    
    logger.info(f"📖 Loading wide dataset: {Path(wide_path).name}")
    wide_df = pd.read_parquet(wide_path)
    
    logger.info(f"📖 Loading raw dataset: {Path(raw_path).name}")
    raw_df = pd.read_parquet(raw_path)
    
    # Basic dataset information
    logger.info("\n" + "="*60)
    logger.info("📊 DATASET OVERVIEW")
    logger.info("="*60)
    
    logger.info(f"Wide Dataset: {wide_df.shape[0]:,} points × {wide_df.shape[1]:,} features")
    logger.info(f"Raw Dataset: {raw_df.shape[0]:,} records × {raw_df.shape[1]:,} columns")
    
    # Spatial coverage validation
    logger.info("\n" + "="*60)
    logger.info("🗺️  SPATIAL COVERAGE VALIDATION")
    logger.info("="*60)
    
    zones = sorted(wide_df['zone'].unique())
    logger.info(f"Zones covered: {zones}")
    
    for zone in zones:
        zone_points = len(wide_df[wide_df['zone'] == zone])
        logger.info(f"  - {zone}: {zone_points:,} points")
        
        # Check coordinate ranges
        zone_data = wide_df[wide_df['zone'] == zone]
        utm_x_range = (zone_data['utm_x'].min(), zone_data['utm_x'].max())
        utm_y_range = (zone_data['utm_y'].min(), zone_data['utm_y'].max())
        logger.info(f"    UTM X range: {utm_x_range[0]:.0f} - {utm_x_range[1]:.0f}")
        logger.info(f"    UTM Y range: {utm_y_range[0]:.0f} - {utm_y_range[1]:.0f}")
    
    # Temporal coverage validation
    logger.info("\n" + "="*60)
    logger.info("📅 TEMPORAL COVERAGE VALIDATION")
    logger.info("="*60)
    
    # Extract years from feature columns
    ndvi_cols = [col for col in wide_df.columns if 'ndvi_' in col and '_w' in col]
    lai_cols = [col for col in wide_df.columns if 'lai_' in col and '_w' in col]
    
    ndvi_years = sorted(list(set([int(col.split('_')[1]) for col in ndvi_cols if col.split('_')[1].isdigit()])))
    lai_years = sorted(list(set([int(col.split('_')[1]) for col in lai_cols if col.split('_')[1].isdigit()])))
    
    logger.info(f"NDVI years: {ndvi_years}")
    logger.info(f"LAI years: {lai_years}")
    logger.info(f"NDVI features per year: {len(ndvi_cols) // len(ndvi_years)}")
    logger.info(f"LAI features per year: {len(lai_cols) // len(lai_years)}")
    
    # Feature completeness validation
    logger.info("\n" + "="*60)
    logger.info("🔍 FEATURE COMPLETENESS VALIDATION")
    logger.info("="*60)
    
    # Check missing values by year and zone
    for year in ndvi_years:
        ndvi_year_cols = [col for col in ndvi_cols if f'_{year}_' in col]
        lai_year_cols = [col for col in lai_cols if f'_{year}_' in col]
        
        ndvi_missing = wide_df[ndvi_year_cols].isnull().sum().sum()
        lai_missing = wide_df[lai_year_cols].isnull().sum().sum()
        
        total_ndvi_cells = len(wide_df) * len(ndvi_year_cols)
        total_lai_cells = len(wide_df) * len(lai_year_cols)
        
        ndvi_completeness = ((total_ndvi_cells - ndvi_missing) / total_ndvi_cells) * 100
        lai_completeness = ((total_lai_cells - lai_missing) / total_lai_cells) * 100
        
        logger.info(f"Year {year}:")
        logger.info(f"  - NDVI completeness: {ndvi_completeness:.1f}% ({ndvi_missing:,} missing)")
        logger.info(f"  - LAI completeness: {lai_completeness:.1f}% ({lai_missing:,} missing)")
    
    # Forest type validation
    logger.info("\n" + "="*60)
    logger.info("🌳 FOREST TYPE VALIDATION")
    logger.info("="*60)
    
    # Forest type distribution
    ftype_mix_dist = wide_df['ftype_mix'].value_counts()
    ftype_nonmix_dist = wide_df['ftype_nonmix'].value_counts()
    
    logger.info("Forest Type Mix Distribution:")
    for ftype, count in ftype_mix_dist.head(10).items():
        pct = (count / len(wide_df)) * 100
        logger.info(f"  - {ftype}: {count:,} ({pct:.1f}%)")
    
    logger.info("\nForest Type Non-Mix Distribution:")
    for ftype, count in ftype_nonmix_dist.head(10).items():
        pct = (count / len(wide_df)) * 100
        logger.info(f"  - {ftype}: {count:,} ({pct:.1f}%)")
    
    # Data quality validation
    logger.info("\n" + "="*60)
    logger.info("📈 DATA QUALITY VALIDATION")
    logger.info("="*60)
    
    # Check NDVI value ranges
    ndvi_sample_cols = [col for col in ndvi_cols if '2023' in col][:5]  # Sample from 2023
    for col in ndvi_sample_cols:
        valid_data = wide_df[col].dropna()
        if len(valid_data) > 0:
            logger.info(f"{col}: min={valid_data.min():.3f}, max={valid_data.max():.3f}, mean={valid_data.mean():.3f}")
    
    # Check for outliers
    logger.info("\nOutlier Detection (NDVI > 1.0 or < -1.0):")
    outlier_count = 0
    for col in ndvi_sample_cols:
        outliers = wide_df[(wide_df[col] > 1.0) | (wide_df[col] < -1.0)][col].count()
        outlier_count += outliers
        if outliers > 0:
            logger.warning(f"  - {col}: {outliers} outliers")
    
    if outlier_count == 0:
        logger.info("  ✅ No NDVI outliers detected")
    
    # Consistency validation
    logger.info("\n" + "="*60)
    logger.info("🔄 CONSISTENCY VALIDATION")
    logger.info("="*60)
    
    # Check if raw and wide datasets have same number of unique points
    raw_points = raw_df['point_id'].nunique()
    wide_points = wide_df['point_id'].nunique()
    
    logger.info(f"Raw dataset unique points: {raw_points:,}")
    logger.info(f"Wide dataset unique points: {wide_points:,}")
    
    if raw_points == wide_points:
        logger.info("✅ Point consistency check passed")
    else:
        logger.warning(f"⚠️  Point count mismatch: {abs(raw_points - wide_points)} difference")
    
    # Performance metrics
    logger.info("\n" + "="*60)
    logger.info("⚡ PERFORMANCE METRICS")
    logger.info("="*60)
    
    # File sizes
    wide_size_mb = Path(wide_path).stat().st_size / (1024*1024)
    raw_size_mb = Path(raw_path).stat().st_size / (1024*1024)
    
    logger.info(f"Wide dataset size: {wide_size_mb:.1f} MB")
    logger.info(f"Raw dataset size: {raw_size_mb:.1f} MB")
    logger.info(f"Total dataset size: {wide_size_mb + raw_size_mb:.1f} MB")
    
    # Memory usage estimation
    memory_usage_mb = wide_df.memory_usage(deep=True).sum() / (1024*1024)
    logger.info(f"Wide dataset memory usage: {memory_usage_mb:.1f} MB")
    
    # Final validation summary
    logger.info("\n" + "="*60)
    logger.info("✅ VALIDATION SUMMARY")
    logger.info("="*60)
    
    validation_results = {
        'total_points': len(wide_df),
        'unique_points': wide_df['point_id'].nunique(),
        'zones_covered': len(zones),
        'years_covered': len(ndvi_years),
        'ndvi_features': len(ndvi_cols),
        'lai_features': len(lai_cols),
        'total_features': len(wide_df.columns),
        'dataset_size_mb': wide_size_mb + raw_size_mb,
        'forest_type_classes': len(ftype_mix_dist) + len(ftype_nonmix_dist)
    }
    
    logger.info(f"📊 Total Points: {validation_results['total_points']:,}")
    logger.info(f"📍 Unique Points: {validation_results['unique_points']:,}")
    logger.info(f"🗺️  Zones: {validation_results['zones_covered']}")
    logger.info(f"📅 Years: {validation_results['years_covered']}")
    logger.info(f"📈 NDVI Features: {validation_results['ndvi_features']}")
    logger.info(f"🍃 LAI Features: {validation_results['lai_features']}")
    logger.info(f"🔢 Total Features: {validation_results['total_features']}")
    logger.info(f"💾 Dataset Size: {validation_results['dataset_size_mb']:.1f} MB")
    logger.info(f"🌳 Forest Classes: {validation_results['forest_type_classes']}")
    
    logger.info("\n🎉 Dataset validation completed successfully!")
    logger.info("📁 Dataset is ready for machine learning model training!")
    
    return validation_results

if __name__ == "__main__":
    results = validate_dataset()
    print("\n✅ Validation completed successfully!")
