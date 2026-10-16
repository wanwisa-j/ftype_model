#!/usr/bin/env python3
"""
Missing Data Handler Script

Creates a solution for handling missing radar data in 2023 VV/VH.
Options:
1. Interpolate missing weeks using available data
2. Filter out 2023 radar data entirely
3. Use alternative imputation strategies
4. Update extraction to handle missing bands gracefully
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def analyze_missing_data_impact():
    """Analyze the impact of missing radar data on the dataset."""
    
    logger.info("🔍 ANALYZING MISSING DATA IMPACT")
    logger.info("=" * 50)
    
    # Load the dataset
    dataset_path = "/fs2/ftype_mflf/shared_data/dataset/1.2.0/raw/forest_type_dataset_raw_20250903_104423.parquet"
    df = pd.read_parquet(dataset_path)
    
    # Define the missing weeks
    missing_weeks = [3, 5, 8, 10, 12, 15, 17, 20]
    available_weeks = [1, 2, 4, 6, 7, 9, 11, 13, 14, 16, 18, 19, 21]
    
    logger.info(f"Missing weeks: {missing_weeks}")
    logger.info(f"Available weeks: {available_weeks}")
    
    # Focus on radar data in 2023
    radar_2023 = df[
        (df['feature'].isin(['vv', 'vh'])) & 
        (df['year'] == 2023) &
        (df['zone'].isin(['mfl_cmf_area_project_2025_chr_zone_1', 'mfl_cmf_area_project_2025_chr_zone_2']))
    ]
    
    logger.info(f"\n📊 2023 Radar Data Overview:")
    logger.info(f"Total records: {len(radar_2023):,}")
    logger.info(f"Unique points: {radar_2023['point_id'].nunique():,}")
    logger.info(f"Features: {radar_2023['feature'].unique()}")
    logger.info(f"Zones: {radar_2023['zone'].unique()}")
    
    # Check data completeness by week
    week_completeness = {}
    for week in range(1, 22):
        week_data = radar_2023[radar_2023['week'] == week]
        completeness = len(week_data) / (len(radar_2023) / 21) * 100 if len(radar_2023) > 0 else 0
        week_completeness[week] = {
            'records': len(week_data),
            'completeness_pct': completeness,
            'is_missing': week in missing_weeks
        }
        
        status = "❌ MISSING" if week in missing_weeks else "✅ AVAILABLE"
        logger.info(f"  Week {week:2d}: {status} - {len(week_data):,} records ({completeness:.1f}%)")
    
    return week_completeness, radar_2023

def generate_interpolation_strategy():
    """Generate interpolation strategy for missing radar data."""
    
    logger.info("\n🛠️  INTERPOLATION STRATEGY")
    logger.info("=" * 30)
    
    missing_weeks = [3, 5, 8, 10, 12, 15, 17, 20]
    available_weeks = [1, 2, 4, 6, 7, 9, 11, 13, 14, 16, 18, 19, 21]
    
    interpolation_plan = []
    
    for missing_week in missing_weeks:
        # Find nearest available weeks
        before_weeks = [w for w in available_weeks if w < missing_week]
        after_weeks = [w for w in available_weeks if w > missing_week]
        
        before_week = max(before_weeks) if before_weeks else None
        after_week = min(after_weeks) if after_weeks else None
        
        strategy = {
            'missing_week': missing_week,
            'before_week': before_week,
            'after_week': after_week,
            'method': 'linear'
        }
        
        if before_week and after_week:
            strategy['interpolation'] = f"Linear between week {before_week} and {after_week}"
        elif before_week:
            strategy['interpolation'] = f"Forward fill from week {before_week}"
            strategy['method'] = 'forward_fill'
        elif after_week:
            strategy['interpolation'] = f"Backward fill from week {after_week}"
            strategy['method'] = 'backward_fill'
        else:
            strategy['interpolation'] = "No nearby weeks available"
            strategy['method'] = 'zero_fill'
        
        interpolation_plan.append(strategy)
        logger.info(f"Week {missing_week:2d}: {strategy['interpolation']}")
    
    return interpolation_plan

def create_data_filtering_options():
    """Create options for handling missing data."""
    
    logger.info("\n🎯 DATA HANDLING OPTIONS")
    logger.info("=" * 25)
    
    options = {
        'option1_interpolate': {
            'name': 'Interpolate Missing Weeks',
            'description': 'Use linear interpolation to fill missing radar weeks',
            'pros': ['Maintains 21-week structure', 'Uses available temporal patterns'],
            'cons': ['Introduces artificial data', 'May affect model accuracy'],
            'implementation': 'Update data preparation to interpolate missing weeks'
        },
        
        'option2_filter_2023': {
            'name': 'Exclude 2023 Radar Data',
            'description': 'Remove all 2023 VV/VH data from training',
            'pros': ['Only uses real data', 'Simple implementation'],
            'cons': ['Reduces dataset size', 'Loses 2023 temporal patterns'],
            'implementation': 'Filter out 2023 radar data during preparation'
        },
        
        'option3_partial_temporal': {
            'name': 'Use 13-Week Radar Structure',
            'description': 'Adapt model to handle variable temporal lengths',
            'pros': ['Uses all available real data', 'More complex modeling'],
            'cons': ['Complex implementation', 'Model architecture changes needed'],
            'implementation': 'Modify model to handle variable-length sequences'
        },
        
        'option4_hybrid': {
            'name': 'Hybrid Approach',
            'description': 'Use interpolation for training, real data for validation',
            'pros': ['Balanced approach', 'Real data for testing'],
            'cons': ['Complex implementation', 'Training/validation mismatch'],
            'implementation': 'Different processing for train/validation sets'
        }
    }
    
    for option_key, option in options.items():
        logger.info(f"\n{option['name']}:")
        logger.info(f"  Description: {option['description']}")
        logger.info(f"  Pros: {', '.join(option['pros'])}")
        logger.info(f"  Cons: {', '.join(option['cons'])}")
        logger.info(f"  Implementation: {option['implementation']}")
    
    return options

def generate_corrected_extraction_code():
    """Generate code to handle missing data during extraction."""
    
    logger.info("\n📝 GENERATING CORRECTED EXTRACTION CODE")
    logger.info("=" * 40)
    
    corrected_code = '''
def handle_missing_radar_data(radar_data, missing_weeks=[3, 5, 8, 10, 12, 15, 17, 20]):
    """
    Handle missing radar data by interpolation.
    
    Args:
        radar_data: DataFrame with radar data (vv/vh)
        missing_weeks: List of weeks with missing data
    
    Returns:
        DataFrame with interpolated values for missing weeks
    """
    import pandas as pd
    import numpy as np
    
    # For each point and feature, interpolate missing weeks
    interpolated_data = []
    
    for (point_id, zone, year, feature), group in radar_data.groupby(['point_id', 'zone', 'year', 'feature']):
        
        # Skip if not 2023 radar data
        if year != 2023 or feature not in ['vv', 'vh']:
            interpolated_data.append(group)
            continue
        
        # Create complete week range
        complete_weeks = list(range(1, 22))
        existing_weeks = group['week'].tolist()
        
        # Create complete dataframe with missing weeks
        complete_group = pd.DataFrame({
            'week': complete_weeks,
            'point_id': point_id,
            'zone': zone,
            'year': year,
            'feature': feature,
            'band': complete_weeks  # band == week for radar
        })
        
        # Merge with existing data
        merged = complete_group.merge(group, on=['week', 'point_id', 'zone', 'year', 'feature', 'band'], how='left')
        
        # Interpolate missing values
        value_cols = [col for col in merged.columns if col.startswith('value') or col == 'pixel_value']
        
        for col in value_cols:
            if col in merged.columns:
                # Linear interpolation for missing values
                merged[col] = merged[col].interpolate(method='linear')
                
                # Forward/backward fill for edge cases
                merged[col] = merged[col].fillna(method='ffill').fillna(method='bfill')
        
        interpolated_data.append(merged)
    
    # Combine all interpolated data
    result = pd.concat(interpolated_data, ignore_index=True)
    return result

def prepare_radar_data_with_interpolation(df):
    """
    Main function to prepare radar data with missing week interpolation.
    """
    logger.info("🔧 Applying radar data interpolation...")
    
    # Separate radar and non-radar data
    radar_data = df[df['feature'].isin(['vv', 'vh'])]
    other_data = df[~df['feature'].isin(['vv', 'vh'])]
    
    # Handle missing radar data
    corrected_radar = handle_missing_radar_data(radar_data)
    
    # Combine back
    corrected_df = pd.concat([corrected_radar, other_data], ignore_index=True)
    
    logger.info(f"✅ Interpolation complete. Total records: {len(corrected_df):,}")
    return corrected_df
'''
    
    logger.info("Generated code for handling missing radar data with interpolation")
    return corrected_code

def main():
    """Main analysis and solution generation."""
    
    logger.info("🚨 MISSING RADAR DATA ANALYSIS & SOLUTION")
    logger.info("=" * 60)
    
    # 1. Analyze impact
    week_completeness, radar_2023 = analyze_missing_data_impact()
    
    # 2. Generate interpolation strategy
    interpolation_plan = generate_interpolation_strategy()
    
    # 3. Create filtering options
    options = create_data_filtering_options()
    
    # 4. Generate corrected code
    corrected_code = generate_corrected_extraction_code()
    
    # 5. Recommendations
    logger.info("\n🎯 RECOMMENDATIONS")
    logger.info("=" * 20)
    logger.info("1. IMMEDIATE FIX: Use Option 1 (Interpolation) to maintain 21-week structure")
    logger.info("2. ALTERNATIVE: Use Option 2 (Filter 2023) for pure real-data approach")
    logger.info("3. IMPLEMENTATION: Update notebook data preparation with interpolation logic")
    logger.info("4. VALIDATION: Test model performance with interpolated vs filtered data")
    
    # Summary of exactly what weeks are problematic
    logger.info("\n📋 EXACT ISSUE SUMMARY:")
    logger.info(f"- Problem: VV/VH radar data in 2023 missing 8 specific weeks")
    logger.info(f"- Missing weeks: {[3, 5, 8, 10, 12, 15, 17, 20]}")
    logger.info(f"- Available weeks: {[1, 2, 4, 6, 7, 9, 11, 13, 14, 16, 18, 19, 21]}")
    logger.info(f"- Affected zones: zone_1 and zone_2")
    logger.info(f"- Root cause: Source radar data incomplete for those weeks in 2023")
    
    return {
        'week_completeness': week_completeness,
        'interpolation_plan': interpolation_plan,
        'options': options,
        'corrected_code': corrected_code
    }

if __name__ == "__main__":
    results = main()
