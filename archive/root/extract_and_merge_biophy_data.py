#!/usr/bin/env python3
"""
Extract and Merge Biophysical Data Script

This script extracts biophysical data (LAI, FAPAR, FCOVER) from the new datacubes
for mfl_cmf_area_project_2025_chr_zone_1 and mfl_cmf_area_project_2025_chr_zone_2
and merges it with the existing cleaned dataset, replacing the problematic biophy data.

Usage:
    python extract_and_merge_biophy_data.py

Features:
- Extracts LAI, FAPAR, FCOVER for zones 1 and 2
- Supports multiple years (2020, 2021, 2022, 2023)
- Creates backup before merging
- Validates data quality after extraction
- Preserves all other data in the cleaned dataset
"""

import os
import sys
import numpy as np
import pandas as pd
import rasterio
from rasterio.sample import sample_gen
from rasterio.warp import transform
from datetime import datetime
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

class BiophysicalDataExtractorMerger:
    """Extract biophysical data from new datacubes and merge with existing dataset."""
    
    def __init__(self):
        """Initialize extractor with paths and configurations."""
        # File paths
        self.dataset_path = "/fs2/ftype_mflf/shared_data/dataset/1.2.0/raw/forest_type_dataset_raw_20250903_145707.cleaned_with_new_biophy_20250909_071747.parquet"
        self.datacube_base_path = "/fs2/ftype_mflf/shared_data/datacube/2025/1.2.0"
        
        # Zones to update
        self.zones_to_update = [
            "mfl_cmf_area_project_2025_chr_zone_1",
            "mfl_cmf_area_project_2025_chr_zone_2"
        ]
        
        # Biophysical features to extract
        self.biophy_features = ["lai", "fapar", "fcover"]
        
        # Years to extract (all available years in the new datacubes, including 2021)
        self.years = [2019, 2020, 2021, 2022, 2023, 2024, 2025]
        
        print(f"=== BIOPHYSICAL DATA EXTRACTOR AND MERGER ===")
        print(f"Dataset: {self.dataset_path}")
        print(f"Datacube base: {self.datacube_base_path}")
        print(f"Zones to update: {self.zones_to_update}")
        print(f"Features: {self.biophy_features}")
        print(f"Years: {self.years}")
    
    def load_existing_dataset(self):
        """Load the existing cleaned dataset."""
        print(f"\n1. Loading existing dataset...")
        if not os.path.exists(self.dataset_path):
            raise FileNotFoundError(f"Dataset not found: {self.dataset_path}")
        
        self.df = pd.read_parquet(self.dataset_path)
        print(f"   Dataset loaded: {self.df.shape}")
        print(f"   Zones in dataset: {sorted(self.df['zone'].unique())}")
        print(f"   Features in dataset: {sorted(self.df['feature'].unique())}")
        print(f"   Years in dataset: {sorted([y for y in self.df['year'].unique() if not pd.isna(y)])}")
        
        return self.df
    
    def create_backup(self):
        """Create backup of the existing dataset."""
        print(f"\n2. Creating backup...")
        backup_path = self.dataset_path.replace('.parquet', f'_BACKUP_{datetime.now().strftime("%Y%m%d_%H%M%S")}.parquet')
        self.df.to_parquet(backup_path)
        print(f"   Backup created: {backup_path}")
        return backup_path
    
    def get_points_for_extraction(self, zone):
        """Get unique point coordinates for a specific zone."""
        print(f"\n   Getting points for zone: {zone}")
        
        # Get points from the dataset for this zone
        zone_data = self.df[self.df['zone'] == zone]
        if len(zone_data) == 0:
            print(f"   Warning: No data found for zone {zone}")
            return pd.DataFrame()
        
        # Get unique points with UTM coordinates
        points_df = zone_data.groupby('point_id').agg({
            'utm_x': 'first',
            'utm_y': 'first',
            'ftype_mix': 'first',
            'ftype_nonmix': 'first'
        }).reset_index()
        
        # Keep UTM coordinates for rasterio sampling (datacubes are in UTM EPSG:32647)
        # No conversion needed - rasterio will sample using UTM coordinates
        points_df['latitude'] = points_df['utm_y']  # For compatibility with later code
        points_df['longitude'] = points_df['utm_x']  # For compatibility with later code
        
        print(f"   Found {len(points_df)} unique points for {zone}")
        print(f"   UTM X range: {points_df['utm_x'].min():.0f} to {points_df['utm_x'].max():.0f}")
        print(f"   UTM Y range: {points_df['utm_y'].min():.0f} to {points_df['utm_y'].max():.0f}")
        
        return points_df
    
    def extract_feature_for_zone_year(self, zone, year, feature):
        """Extract biophysical feature for a specific zone and year."""
        print(f"     Extracting {feature} for {zone}, year {year}")
        
        # Construct datacube file path
        datacube_path = f"{self.datacube_base_path}/{zone}/{feature}/cube_biophys_{feature}_{year}_W4before_to_W16after_21bands.tif"
        
        if not os.path.exists(datacube_path):
            print(f"     Warning: Datacube not found: {datacube_path}")
            return pd.DataFrame()
        
        # Get points for this zone
        points_df = self.get_points_for_extraction(zone)
        if len(points_df) == 0:
            return pd.DataFrame()
        
        # Open datacube and extract values
        with rasterio.open(datacube_path) as src:
            print(f"     Datacube shape: {src.shape}, bands: {src.count}")
            
            # Prepare coordinates for sampling
            coords = [(lon, lat) for lon, lat in zip(points_df['longitude'], points_df['latitude'])]
            
            # Extract values for all bands (21 bands)
            extracted_data = []
            
            try:
                # Sample all bands at once
                sampled_values = list(sample_gen(src, coords))
                
                for point_idx, point_values in enumerate(sampled_values):
                    point_id = points_df.iloc[point_idx]['point_id']
                    
                    # point_values contains 21 values (one per band/timestep)
                    if len(point_values) == 21:
                        for timestep, value in enumerate(point_values):
                            extracted_data.append({
                                'point_id': point_id,
                                'zone': zone,
                                'year': float(year),
                                'feature': feature,
                                'timestep': timestep,
                                'value': float(value) if not np.isnan(value) and value != -9999 else np.nan,
                                'utm_x': points_df.iloc[point_idx]['utm_x'],
                                'utm_y': points_df.iloc[point_idx]['utm_y'],
                                'ftype_mix': points_df.iloc[point_idx]['ftype_mix'],
                                'ftype_nonmix': points_df.iloc[point_idx]['ftype_nonmix']
                            })
                    else:
                        print(f"     Warning: Point {point_id} has {len(point_values)} values instead of 21")
                
                print(f"     Extracted {len(extracted_data)} records for {feature}")
                
            except Exception as e:
                print(f"     Error extracting {feature}: {e}")
                return pd.DataFrame()
        
        return pd.DataFrame(extracted_data)
    
    def check_existing_data(self, zone, year, feature):
        """Check if data already exists for a specific zone, year, and feature."""
        existing_data = self.df[
            (self.df['zone'] == zone) &
            (self.df['feature'] == feature) &
            (self.df['year'] == float(year))
        ]
        
        if len(existing_data) > 0:
            # Check data quality - if all values are valid, skip extraction
            valid_values = existing_data['value'].notna().sum()
            total_values = len(existing_data)
            completion_rate = (valid_values / total_values * 100) if total_values > 0 else 0
            
            print(f"     Existing data: {total_values} records, {completion_rate:.1f}% complete")
            
            # Only skip if data is nearly complete (>95%) to avoid overwriting good data
            if completion_rate > 95:
                print(f"     Skipping extraction - existing data is {completion_rate:.1f}% complete")
                return True
            else:
                print(f"     Will re-extract - existing data only {completion_rate:.1f}% complete")
                return False
        else:
            print(f"     No existing data found - will extract")
            return False
    
    def extract_all_biophy_data(self):
        """Extract all biophysical data for the specified zones and years."""
        print(f"\n3. Extracting biophysical data...")
        
        all_new_data = []
        
        for zone in self.zones_to_update:
            print(f"\n  Processing zone: {zone}")
            
            for year in self.years:
                print(f"   Processing year: {year}")
                
                for feature in self.biophy_features:
                    # Check if we already have good data for this combination
                    if not self.check_existing_data(zone, year, feature):
                        feature_data = self.extract_feature_for_zone_year(zone, year, feature)
                        if len(feature_data) > 0:
                            all_new_data.append(feature_data)
                    else:
                        print(f"     Skipping {feature} {year} for {zone} - good data already exists")
        
        if not all_new_data:
            print(f"   No new data extracted!")
            return pd.DataFrame()
        
        # Combine all extracted data
        new_biophy_df = pd.concat(all_new_data, ignore_index=True)
        print(f"\n   Total new biophy records: {len(new_biophy_df)}")
        
        # Validate extracted data
        self.validate_extracted_data(new_biophy_df)
        
        return new_biophy_df
    
    def validate_extracted_data(self, new_data):
        """Validate the quality of extracted data."""
        print(f"\n   Validating extracted data...")
        
        # Check data completeness
        total_records = len(new_data)
        missing_values = new_data['value'].isna().sum()
        completion_rate = (total_records - missing_values) / total_records * 100
        
        print(f"   Total records: {total_records}")
        print(f"   Missing values: {missing_values}")
        print(f"   Data completion rate: {completion_rate:.2f}%")
        
        # Check by zone and feature
        for zone in self.zones_to_update:
            for feature in self.biophy_features:
                zone_feature_data = new_data[
                    (new_data['zone'] == zone) & 
                    (new_data['feature'] == feature)
                ]
                if len(zone_feature_data) > 0:
                    zone_missing = zone_feature_data['value'].isna().sum()
                    zone_completion = (len(zone_feature_data) - zone_missing) / len(zone_feature_data) * 100
                    print(f"   {zone} {feature}: {len(zone_feature_data)} records, {zone_completion:.1f}% complete")
        
        # Check value ranges
        for feature in self.biophy_features:
            feature_data = new_data[new_data['feature'] == feature]['value'].dropna()
            if len(feature_data) > 0:
                print(f"   {feature} value range: [{feature_data.min():.3f}, {feature_data.max():.3f}]")
    
    def remove_old_biophy_data(self):
        """Remove old biophysical data for the specified zones."""
        print(f"\n4. Removing old biophysical data...")
        
        # Identify records to remove
        conditions_to_remove = (
            (self.df['zone'].isin(self.zones_to_update)) &
            (self.df['feature'].isin(self.biophy_features)) &
            (self.df['year'].isin([float(y) for y in self.years]))
        )
        
        records_to_remove = self.df[conditions_to_remove]
        print(f"   Records to remove: {len(records_to_remove)}")
        
        # Show breakdown
        for zone in self.zones_to_update:
            for feature in self.biophy_features:
                count = len(records_to_remove[
                    (records_to_remove['zone'] == zone) &
                    (records_to_remove['feature'] == feature)
                ])
                if count > 0:
                    print(f"   {zone} {feature}: {count} records")
        
        # Remove old records
        self.df_cleaned = self.df[~conditions_to_remove].copy()
        print(f"   Dataset after removal: {self.df_cleaned.shape}")
        
        return self.df_cleaned
    
    def merge_new_data(self, new_biophy_df):
        """Merge new biophysical data with the cleaned dataset."""
        print(f"\n5. Merging new biophysical data...")
        
        print(f"   Dataset before merge: {self.df_cleaned.shape}")
        print(f"   New data to add: {len(new_biophy_df)}")
        
        # Combine datasets
        self.df_final = pd.concat([self.df_cleaned, new_biophy_df], ignore_index=True)
        print(f"   Final dataset shape: {self.df_final.shape}")
        
        return self.df_final
    
    def validate_final_dataset(self):
        """Validate the final merged dataset."""
        print(f"\n6. Validating final dataset...")
        
        # Check biophysical data for updated zones
        for zone in self.zones_to_update:
            for feature in self.biophy_features:
                zone_feature_data = self.df_final[
                    (self.df_final['zone'] == zone) &
                    (self.df_final['feature'] == feature) &
                    (self.df_final['year'].isin([float(y) for y in self.years]))
                ]
                
                missing_count = zone_feature_data['value'].isna().sum()
                total_count = len(zone_feature_data)
                completion_rate = (total_count - missing_count) / total_count * 100 if total_count > 0 else 0
                
                print(f"   {zone} {feature}: {total_count} records, {completion_rate:.1f}% complete")
                
                if missing_count == 0:
                    print(f"   ✅ {zone} {feature}: No missing values!")
                else:
                    print(f"   ⚠️ {zone} {feature}: {missing_count} missing values")
    
    def save_final_dataset(self):
        """Save the final merged dataset."""
        print(f"\n7. Saving final dataset...")
        
        # Save to original path
        self.df_final.to_parquet(self.dataset_path)
        print(f"   Final dataset saved: {self.dataset_path}")
        print(f"   Final shape: {self.df_final.shape}")
        
        # Create additional backup with new data info
        new_data_path = self.dataset_path.replace('.parquet', f'_with_new_biophy_{datetime.now().strftime("%Y%m%d_%H%M%S")}.parquet')
        self.df_final.to_parquet(new_data_path)
        print(f"   Additional copy saved: {new_data_path}")
    
    def run_extraction_and_merge(self):
        """Run the complete extraction and merge process."""
        print(f"\n{'='*60}")
        print(f"STARTING BIOPHYSICAL DATA EXTRACTION AND MERGE")
        print(f"{'='*60}")
        
        try:
            # Step 1: Load existing dataset
            self.load_existing_dataset()
            
            # Step 2: Create backup
            backup_path = self.create_backup()
            
            # Step 3: Extract new biophysical data
            new_biophy_df = self.extract_all_biophy_data()
            
            if len(new_biophy_df) == 0:
                print(f"\n❌ No new data extracted. Aborting merge.")
                return False
            
            # Step 4: Remove old biophysical data
            self.remove_old_biophy_data()
            
            # Step 5: Merge new data
            self.merge_new_data(new_biophy_df)
            
            # Step 6: Validate final dataset
            self.validate_final_dataset()
            
            # Step 7: Save final dataset
            self.save_final_dataset()
            
            print(f"\n{'='*60}")
            print(f"✅ EXTRACTION AND MERGE COMPLETED SUCCESSFULLY!")
            print(f"{'='*60}")
            print(f"Original dataset backed up: {backup_path}")
            print(f"Updated dataset: {self.dataset_path}")
            print(f"Final dataset shape: {self.df_final.shape}")
            
            return True
            
        except Exception as e:
            print(f"\n❌ ERROR during extraction and merge: {e}")
            import traceback
            traceback.print_exc()
            return False

def main():
    """Main function to run the extraction and merge process."""
    extractor = BiophysicalDataExtractorMerger()
    success = extractor.run_extraction_and_merge()
    
    if success:
        print(f"\n🎉 Process completed successfully!")
    else:
        print(f"\n💥 Process failed. Check the logs above.")
        sys.exit(1)

if __name__ == "__main__":
    main()
