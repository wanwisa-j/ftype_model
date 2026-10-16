#!/usr/bin/env python3
"""
Dataset Consolidation Script
Merges datasets from multiple machines into final combined files
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
from datetime import datetime
import glob

def setup_logging():
    """Setup logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(Path(__file__).resolve().parent.parent / 'logs' / f'consolidation_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def find_machine_files(data_dir, pattern):
    """Find all machine-specific files matching the pattern"""
    files = glob.glob(str(data_dir / pattern))
    return sorted(files, key=lambda x: Path(x).stat().st_mtime, reverse=True)

def consolidate_datasets():
    """Consolidate datasets from all machines"""
    logger = setup_logging()
    
    logger.info("🔄 Starting dataset consolidation...")
    
    # Define paths
    base_dir = Path('/fs2/ftype_mflf/shared_data/dataset/1.2.0')
    raw_dir = base_dir / 'raw'
    wide_dir = base_dir / 'processed'
    
    # Check if directories exist
    if not raw_dir.exists():
        logger.error(f"❌ Raw directory not found: {raw_dir}")
        return False
        
    if not wide_dir.exists():
        logger.error(f"❌ Wide directory not found: {wide_dir}")
        return False
    
    # Find machine files
    logger.info("🔍 Searching for machine-specific files...")
    
    raw_files = find_machine_files(raw_dir, "forest_type_dataset_raw_*.parquet")
    wide_files = find_machine_files(base_dir / 'processed', "forest_type_dataset_wide_*.parquet")
    
    logger.info(f"Found {len(raw_files)} raw files and {len(wide_files)} wide files")
    
    if not raw_files:
        logger.error("❌ No raw machine files found")
        return False
        
    if not wide_files:
        logger.error("❌ No wide machine files found")
        return False
    
    # Print found files
    logger.info("📁 Raw files found:")
    for f in raw_files:
        size_mb = Path(f).stat().st_size / (1024*1024)
        logger.info(f"   - {Path(f).name} ({size_mb:.1f} MB)")
    
    logger.info("📁 Wide files found:")
    for f in wide_files:
        size_mb = Path(f).stat().st_size / (1024*1024)
        logger.info(f"   - {Path(f).name} ({size_mb:.1f} MB)")
    
    # Consolidate raw files
    logger.info("🔄 Consolidating raw datasets...")
    raw_dfs = []
    total_raw_records = 0
    
    for file_path in raw_files:
        logger.info(f"📖 Loading {Path(file_path).name}...")
        try:
            df = pd.read_parquet(file_path)
            logger.info(f"   - Records: {len(df):,}")
            logger.info(f"   - Columns: {list(df.columns)}")
            logger.info(f"   - Zones: {df['zone'].unique().tolist()}")
            
            raw_dfs.append(df)
            total_raw_records += len(df)
            
        except Exception as e:
            logger.error(f"❌ Error loading {file_path}: {e}")
            return False
    
    # Combine raw datasets
    logger.info(f"🔗 Combining {len(raw_dfs)} raw datasets...")
    combined_raw = pd.concat(raw_dfs, ignore_index=True)
    logger.info(f"✅ Combined raw dataset: {len(combined_raw):,} records")
    
    # Consolidate wide files
    logger.info("🔄 Consolidating wide datasets...")
    wide_dfs = []
    total_wide_records = 0
    
    for file_path in wide_files:
        logger.info(f"📖 Loading {Path(file_path).name}...")
        try:
            df = pd.read_parquet(file_path)
            logger.info(f"   - Records: {len(df):,}")
            logger.info(f"   - Features: {[col for col in df.columns if 'week' in col or 'lai' in col or 'ndvi' in col][:5]}...")
            logger.info(f"   - Zones: {df['zone'].unique().tolist()}")
            
            wide_dfs.append(df)
            total_wide_records += len(df)
            
        except Exception as e:
            logger.error(f"❌ Error loading {file_path}: {e}")
            return False
    
    # Combine wide datasets
    logger.info(f"🔗 Combining {len(wide_dfs)} wide datasets...")
    combined_wide = pd.concat(wide_dfs, ignore_index=True)
    logger.info(f"✅ Combined wide dataset: {len(combined_wide):,} records")
    
    # Generate final filenames
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    final_raw_file = raw_dir / f"forest_type_dataset_raw_combined_{timestamp}.parquet"
    final_wide_file = wide_dir / f"forest_type_dataset_wide_combined_{timestamp}.parquet"
    
    # Save combined datasets
    logger.info(f"💾 Saving combined raw dataset...")
    combined_raw.to_parquet(final_raw_file, index=False)
    raw_size_mb = final_raw_file.stat().st_size / (1024*1024)
    logger.info(f"✅ Saved: {final_raw_file.name} ({raw_size_mb:.1f} MB)")
    
    logger.info(f"💾 Saving combined wide dataset...")
    combined_wide.to_parquet(final_wide_file, index=False)
    wide_size_mb = final_wide_file.stat().st_size / (1024*1024)
    logger.info(f"✅ Saved: {final_wide_file.name} ({wide_size_mb:.1f} MB)")
    
    # Generate summary statistics
    logger.info("\n📊 CONSOLIDATION SUMMARY")
    logger.info("=" * 50)
    logger.info(f"📈 Raw Dataset:")
    logger.info(f"   - Total records: {len(combined_raw):,}")
    logger.info(f"   - Unique points: {combined_raw['point_id'].nunique():,}")
    logger.info(f"   - Zones: {sorted(combined_raw['zone'].unique())}")
    logger.info(f"   - Features: {sorted(combined_raw['feature'].unique())}")
    logger.info(f"   - Years: {sorted(combined_raw['year'].unique())}")
    logger.info(f"   - File size: {raw_size_mb:.1f} MB")
    
    logger.info(f"\n📊 Wide Dataset:")
    logger.info(f"   - Total records: {len(combined_wide):,}")
    logger.info(f"   - Unique points: {combined_wide['point_id'].nunique():,}")
    logger.info(f"   - Zones: {sorted(combined_wide['zone'].unique())}")
    if 'year' in combined_wide.columns:
        logger.info(f"   - Years: {sorted(combined_wide['year'].unique())}")
    else:
        # Extract years from column names
        year_cols = [col for col in combined_wide.columns if any(str(y) in col for y in range(2019, 2026))]
        years = sorted(list(set([int(col.split('_')[1]) for col in year_cols if '_' in col and col.split('_')[1].isdigit()])))
        logger.info(f"   - Years (from features): {years}")
    logger.info(f"   - Time series columns: {len([col for col in combined_wide.columns if 'week' in col or 'w' in col])}")
    logger.info(f"   - File size: {wide_size_mb:.1f} MB")
    
    # Forest type distribution
    logger.info(f"\n🌳 Forest Type Distribution:")
    for col in ['ftype_mix', 'ftype_nonmix']:
        if col in combined_wide.columns:
            dist = combined_wide[col].value_counts()
            logger.info(f"   {col}:")
            for ftype, count in dist.items():
                pct = (count / len(combined_wide)) * 100
                logger.info(f"      - {ftype}: {count:,} ({pct:.1f}%)")
    
    # Data quality checks
    logger.info(f"\n🔍 Data Quality Checks:")
    
    # Check for missing values in raw data
    raw_missing = combined_raw.isnull().sum()
    missing_cols = raw_missing[raw_missing > 0]
    if len(missing_cols) > 0:
        logger.warning(f"⚠️  Missing values in raw data:")
        for col, count in missing_cols.items():
            logger.warning(f"      - {col}: {count:,} missing")
    else:
        logger.info(f"✅ No missing values in raw data")
    
    # Check for missing values in wide data
    wide_missing = combined_wide.isnull().sum()
    missing_cols = wide_missing[wide_missing > 0]
    if len(missing_cols) > 0:
        logger.warning(f"⚠️  Missing values in wide data:")
        for col, count in missing_cols.items():
            if count > 0:
                logger.warning(f"      - {col}: {count:,} missing")
    else:
        logger.info(f"✅ No missing values in wide data")
    
    logger.info(f"\n🎉 Dataset consolidation completed successfully!")
    logger.info(f"📁 Output files:")
    logger.info(f"   - Raw: {final_raw_file}")
    logger.info(f"   - Wide: {final_wide_file}")
    
    return True

if __name__ == "__main__":
    success = consolidate_datasets()
    if success:
        print("\n✅ Consolidation completed successfully!")
    else:
        print("\n❌ Consolidation failed!")
        exit(1)
