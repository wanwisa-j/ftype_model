#!/usr/bin/env python3
"""
Datacube Comparison Script for Forest Type Classification

This script builds a full datacube for the entire mfl_cmf_area_project_2025_chr_zone_1 zone
and compares it with the existing dataset at the same sample points.

Key Features:
- Loads timeseries and static features for entire zone
- Creates full datacube with all spatial pixels
- Samples datacube at same points as existing dataset
- Compares values between datacube samples and dataset

Author: Generated for forest type classification project
Date: 2025-09-10
"""

import os
import sys
import numpy as np
import pandas as pd
import rasterio
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Add project paths
sys.path.append('/fs1/tor_project/model_foresttype/src')
from dataset_builder import ForestTypeDatasetBuilder
from prediction_ftype_2_2_1 import load_raster_data, create_union_mask, reshape_and_stack_features

class DatacubeComparison:
    """
    Compare full datacube with sampled dataset for mfl_cmf_area_project_2025_chr_zone_1
    """
    
    def __init__(self, use_extract_file_logic=False):
        self.zone = 'mfl_cmf_area_project_2025_chr_zone_1'
        self.year = 2025
        self.datacube_dir = '/fs2/ftype_mflf/shared_data/datacube/2025/1.2.0/mfl_cmf_area_project_2025_chr_zone_1'
        self.use_extract_file_logic = use_extract_file_logic
        
        # Feature lists from prediction script
        self.timeseries_features = ['vv', 'vh', 'gvmi', 'ndvi', 'slavi', 'lai', 'fapar', 'fcover']
        self.static_features = ['elevation', 'slope']
        self.all_features = self.timeseries_features + self.static_features
        
        print(f"Initialized DatacubeComparison for {self.zone}, year {self.year}")
        print(f"Using extract file selection logic: {use_extract_file_logic}")
        
    def load_full_datacube(self):
        """Load full datacube for the entire zone"""
        print("=" * 60)
        print("LOADING FULL DATACUBE")
        print("=" * 60)
        
        datacube_data = {}
        
        # Load timeseries features
        timeseries_dicts = []
        for feature_name in self.timeseries_features:
            print(f"Loading timeseries feature: {feature_name}")
            try:
                if self.use_extract_file_logic:
                    data_dict = self._load_raster_data_extract_logic(self.datacube_dir, self.year, feature_name)
                else:
                    data_dict = load_raster_data(self.datacube_dir, self.year, feature_name)
                timeseries_dicts.append(data_dict)
                datacube_data[feature_name] = data_dict
                print(f"  ✅ Loaded {feature_name}: {data_dict['data'].shape}")
            except Exception as e:
                print(f"  ❌ Failed to load {feature_name}: {e}")
                
        # Load static features
        static_dicts = []
        for feature_name in self.static_features:
            print(f"Loading static feature: {feature_name}")
            try:
                if self.use_extract_file_logic:
                    data_dict = self._load_raster_data_extract_logic(self.datacube_dir, self.year, feature_name)
                else:
                    data_dict = load_raster_data(self.datacube_dir, self.year, feature_name)
                static_dicts.append(data_dict)
                datacube_data[feature_name] = data_dict
                print(f"  ✅ Loaded {feature_name}: {data_dict['data'].shape}")
            except Exception as e:
                print(f"  ❌ Failed to load {feature_name}: {e}")
        
        # Create union mask
        all_dicts = timeseries_dicts + static_dicts
        if all_dicts:
            union_mask = create_union_mask(all_dicts)
            print(f"Created union mask with {np.sum(~union_mask)} valid pixels out of {union_mask.size} total")
            
            # Get spatial dimensions
            first_data = timeseries_dicts[0]['data']
            n_timesteps = first_data.shape[0]
            height, width = union_mask.shape
            
            # Store datacube info
            self.datacube_info = {
                'data': datacube_data,
                'timeseries_dicts': timeseries_dicts,
                'static_dicts': static_dicts,
                'union_mask': union_mask,
                'n_timesteps': n_timesteps,
                'height': height,
                'width': width,
                'n_valid_pixels': np.sum(~union_mask)
            }
            
            print(f"Datacube dimensions: {n_timesteps} timesteps, {height}x{width} pixels")
            print(f"Valid pixels: {self.datacube_info['n_valid_pixels']}")
            
            return True
        else:
            print("❌ Failed to load any datacube features")
            return False
    
    def _load_raster_data_extract_logic(self, datacube_dir, year, feature_name):
        """Load raster data using extract_forest_type_dataset.py file selection logic"""
        import rasterio
        
        feature_dir = Path(datacube_dir) / feature_name
        
        if not feature_dir.exists():
            raise FileNotFoundError(f"Feature directory not found: {feature_dir}")
        
        # For static features, don't look for year-specific files
        if feature_name in self.static_features:
            all_files = list(feature_dir.glob("*.tif"))
            if not all_files:
                raise FileNotFoundError(f"No files found in {feature_dir}")
            
            # Use the first available file (static features don't vary by year)
            main_file = all_files[0]
        else:
            # Look for year-specific files for temporal features
            year_files = list(feature_dir.glob(f"*{year}*.tif"))
            
            if not year_files:
                raise FileNotFoundError(f"No files found for year {year} in {feature_dir}")
            
            # Filter to prioritize main files and avoid duplicates
            # Priority order: main file > interpolated > original > backup
            main_file = None
            for file_path in year_files:
                filename = file_path.name.lower()
                # Skip backup, original, and other duplicate files - prioritize main file
                if not any(suffix in filename for suffix in ['_backup', '_original', '_interpolated']):
                    main_file = file_path
                    break
            
            # If no main file found, try interpolated, then original (but never backup)
            if main_file is None:
                for file_path in year_files:
                    filename = file_path.name.lower()
                    if '_interpolated' in filename:
                        main_file = file_path
                        break
                if main_file is None:
                    for file_path in year_files:
                        filename = file_path.name.lower()
                        if '_original' in filename and '_backup' not in filename:
                            main_file = file_path
                            break
            
            if main_file is None:
                raise FileNotFoundError(f"No suitable file found for {feature_name} year {year}")
        
        print(f"  📁 Using extract logic: {main_file.name}")
        
        # Load the selected file
        with rasterio.open(main_file) as src:
            data = src.read()  # Read all bands
            transform = src.transform
            crs = src.crs
            nodata = src.nodata
            bounds = src.bounds
            
            # Handle nodata values
            if nodata is not None:
                data = np.where(data == nodata, np.nan, data)
            
            metadata = {
                'transform': transform,
                'crs': crs,
                'bounds': bounds,
                'nodata': nodata,
                'shape': data.shape,
                'width': src.width,
                'height': src.height
            }
        
        return {
            'data': data,
            'metadata': metadata
        }
    
    def load_sample_dataset(self):
        """Load the existing sample dataset - trying both ForestTypeDatasetBuilder and direct parquet load"""
        print("\n" + "=" * 60)
        print("LOADING SAMPLE DATASET")
        print("=" * 60)
        
        try:
            # First, try using ForestTypeDatasetBuilder
            builder = ForestTypeDatasetBuilder()
            
            # Create dataset for the specific zone and year
            dataset = builder.create_dataset(
                zones=[self.zone],
                years=[self.year],
                features=self.all_features,
                target='ftype_nonmix',
                return_point_info=True
            )
            
            self.sample_dataset = dataset
            print(f"✅ Loaded sample dataset with {dataset['target'].shape[0]} points")
            
            # Display point info sample
            if 'point_info' in dataset:
                print("Sample point info:")
                for i, point in enumerate(dataset['point_info'][:3]):
                    print(f"  Point {i+1}: ID={point['point_id']}, X={point['utm_x']}, Y={point['utm_y']}")
            
            return True
            
        except Exception as e:
            print(f"⚠️  ForestTypeDatasetBuilder failed: {e}")
            print("Trying to load from extract_forest_type_dataset.py output...")
            
            # Try to load from test dataset if available
            test_dataset_dir = '/fs1/tor_project/model_foresttype/test_dataset'
            if os.path.exists(test_dataset_dir):
                return self._load_from_extract_dataset()
            else:
                print("❌ No test dataset available")
                raise e
    
    def _load_from_extract_dataset(self):
        """Load dataset from extract_forest_type_dataset.py output"""
        print("📁 Loading from extract_forest_type_dataset.py output...")
        
        # Look for the most recent parquet file
        test_dir = Path('/fs1/tor_project/model_foresttype/test_dataset')
        raw_dir = test_dir / 'raw'
        
        if not raw_dir.exists():
            raise FileNotFoundError("Raw dataset directory not found")
        
        # Find most recent parquet file
        parquet_files = list(raw_dir.glob('forest_type_dataset_raw_*.parquet'))
        if not parquet_files:
            raise FileNotFoundError("No raw dataset parquet files found")
        
        latest_file = max(parquet_files, key=lambda p: p.stat().st_mtime)
        print(f"Loading from: {latest_file}")
        
        # Load parquet file
        import pandas as pd
        df = pd.read_parquet(latest_file)
        print(f"Loaded {len(df)} records from extract dataset")
        
        # Filter for our zone and year
        df_filtered = df[
            (df['zone'] == self.zone) & 
            (df['year'] == self.year)
        ].copy()
        print(f"Filtered to {len(df_filtered)} records for {self.zone} year {self.year}")
        
        # Convert to format compatible with comparison
        self.sample_dataset = self._convert_extract_to_dataset_format(df_filtered)
        
        print(f"✅ Loaded extract dataset with {len(self.sample_dataset['point_info'])} points")
        return True
    
    def _convert_extract_to_dataset_format(self, df):
        """Convert extract dataset format to comparison format"""
        print("🔄 Converting extract dataset format...")
        
        # Get unique points
        point_cols = ['point_id', 'utm_x', 'utm_y', 'zone', 'ftype_mix', 'ftype_nonmix']
        points_df = df[point_cols].drop_duplicates().reset_index(drop=True)
        
        # Create point info
        point_info = []
        for _, row in points_df.iterrows():
            point_info.append({
                'point_id': row['point_id'],
                'utm_x': row['utm_x'], 
                'utm_y': row['utm_y'],
                'zone': row['zone']
            })
        
        # Create feature arrays
        dataset = {'point_info': point_info}
        
        for feature in self.all_features:
            feature_df = df[df['feature'] == feature].copy()
            if len(feature_df) == 0:
                print(f"⚠️  No data for feature {feature}")
                continue
                
            if feature in self.static_features:
                # Static features - single value per point
                feature_values = []
                for point_id in points_df['point_id']:
                    point_data = feature_df[feature_df['point_id'] == point_id]
                    if len(point_data) > 0:
                        feature_values.append([point_data['value'].iloc[0]])
                    else:
                        feature_values.append([np.nan])
                dataset[f'{feature}_feature'] = np.array(feature_values)
            else:
                # Time series features - multiple values per point
                feature_values = []
                for point_id in points_df['point_id']:
                    point_data = feature_df[feature_df['point_id'] == point_id].sort_values('week')
                    if len(point_data) > 0:
                        values = point_data['value'].values
                        # Pad or truncate to 21 timesteps
                        if len(values) < 21:
                            values = np.pad(values, (0, 21 - len(values)), constant_values=np.nan)
                        elif len(values) > 21:
                            values = values[:21]
                        feature_values.append(values)
                    else:
                        feature_values.append(np.full(21, np.nan))
                dataset[f'{feature}_feature'] = np.array(feature_values)
            
            print(f"  {feature}_feature shape: {dataset[f'{feature}_feature'].shape}")
        
        # Add target
        dataset['target'] = points_df[['ftype_nonmix']].values
        
        return dataset
    
    def get_pixel_coordinates_from_points(self):
        """Convert UTM coordinates to pixel coordinates in the datacube"""
        print("\n" + "=" * 60)
        print("CONVERTING COORDINATES TO PIXEL INDICES")
        print("=" * 60)
        
        # Get reference metadata from first timeseries feature
        reference_metadata = self.datacube_info['timeseries_dicts'][0]['metadata']
        transform = reference_metadata['transform']
        
        point_info = self.sample_dataset['point_info']
        pixel_coordinates = []
        valid_points = []
        
        for i, point in enumerate(point_info):
            # Convert UTM to pixel coordinates
            utm_x, utm_y = point['utm_x'], point['utm_y']
            
            # Use rasterio transform to get pixel coordinates
            col, row = ~transform * (utm_x, utm_y)
            col, row = int(round(col)), int(round(row))
            
            # Check if pixel is within raster bounds
            height, width = self.datacube_info['height'], self.datacube_info['width']
            if 0 <= row < height and 0 <= col < width:
                # Check if pixel is not masked (valid)
                if not self.datacube_info['union_mask'][row, col]:
                    pixel_coordinates.append((row, col))
                    valid_points.append(i)
                    if len(pixel_coordinates) <= 3:  # Show first 3 examples
                        print(f"  Point {point['point_id']}: UTM({utm_x:.0f}, {utm_y:.0f}) -> Pixel({row}, {col})")
                else:
                    print(f"  Point {point['point_id']}: UTM({utm_x:.0f}, {utm_y:.0f}) -> Pixel({row}, {col}) [MASKED]")
            else:
                print(f"  Point {point['point_id']}: UTM({utm_x:.0f}, {utm_y:.0f}) -> Pixel({row}, {col}) [OUT OF BOUNDS]")
        
        self.pixel_coordinates = pixel_coordinates
        self.valid_point_indices = valid_points
        
        print(f"✅ Found {len(pixel_coordinates)} valid points within datacube bounds")
        return len(pixel_coordinates) > 0
    
    def sample_datacube_at_points(self):
        """Sample datacube values at the specific points"""
        print("\n" + "=" * 60)
        print("SAMPLING DATACUBE AT DATASET POINTS")
        print("=" * 60)
        
        datacube_samples = {}
        
        # Sample timeseries features
        for i, feature_name in enumerate(self.timeseries_features):
            print(f"Sampling {feature_name}...")
            feature_data = self.datacube_info['timeseries_dicts'][i]['data']
            
            point_samples = []
            for row, col in self.pixel_coordinates:
                # Extract time series for this pixel (all timesteps)
                pixel_timeseries = feature_data[:, row, col]
                point_samples.append(pixel_timeseries)
            
            datacube_samples[feature_name] = np.array(point_samples)
            print(f"  ✅ {feature_name}: {datacube_samples[feature_name].shape}")
        
        # Sample static features  
        for i, feature_name in enumerate(self.static_features):
            print(f"Sampling {feature_name}...")
            
            # Find the correct static feature data by matching feature names loaded during datacube loading
            # The static_dicts are loaded in the same order as static_features in load_full_datacube()
            if i < len(self.datacube_info['static_dicts']):
                feature_data = self.datacube_info['static_dicts'][i]['data']
                print(f"  📍 Using static_dict index {i} for {feature_name}")
            else:
                print(f"  ❌ No static data available for {feature_name} (index {i} >= {len(self.datacube_info['static_dicts'])})")
                continue
            
            # Handle single band static data
            if feature_data.ndim == 3 and feature_data.shape[0] == 1:
                feature_data = feature_data.squeeze(axis=0)
            
            point_samples = []
            for row, col in self.pixel_coordinates:
                # Extract static value for this pixel
                pixel_value = feature_data[row, col]
                point_samples.append([pixel_value])  # Keep as array for consistency
            
            datacube_samples[feature_name] = np.array(point_samples)
            print(f"  ✅ {feature_name}: {datacube_samples[feature_name].shape}")
        
        self.datacube_samples = datacube_samples
        print(f"✅ Sampled datacube at {len(self.pixel_coordinates)} points")
        return True
    
    def compare_samples_with_dataset(self):
        """Compare datacube samples with dataset values"""
        print("\n" + "=" * 60)
        print("COMPARING DATACUBE SAMPLES WITH DATASET")  
        print("=" * 60)
        
        comparison_results = {}
        
        for feature_name in self.all_features:
            print(f"\nComparing {feature_name}:")
            print("-" * 30)
            
            # Get datacube samples
            datacube_values = self.datacube_samples[feature_name]
            
            # Get dataset values for valid points
            dataset_feature = self.sample_dataset[f'{feature_name}_feature']
            dataset_values = dataset_feature[self.valid_point_indices]
            
            # Compare shapes
            print(f"Datacube shape: {datacube_values.shape}")
            print(f"Dataset shape:  {dataset_values.shape}")
            
            if datacube_values.shape != dataset_values.shape:
                print(f"  ⚠️  Shape mismatch - cannot compare directly")
                continue
            
            # Calculate differences
            diff = datacube_values - dataset_values
            rmse = np.sqrt(np.mean(diff**2))
            mae = np.mean(np.abs(diff))
            max_diff = np.max(np.abs(diff))
            
            # Handle NaN values
            valid_mask = ~(np.isnan(datacube_values) | np.isnan(dataset_values))
            if np.sum(valid_mask) > 0:
                valid_diff = diff[valid_mask]
                rmse_valid = np.sqrt(np.mean(valid_diff**2))
                mae_valid = np.mean(np.abs(valid_diff))
                max_diff_valid = np.max(np.abs(valid_diff))
                
                print(f"  RMSE (all):   {rmse:.6f}")
                print(f"  MAE (all):    {mae:.6f}")
                print(f"  Max diff:     {max_diff:.6f}")
                print(f"  RMSE (valid): {rmse_valid:.6f}")
                print(f"  MAE (valid):  {mae_valid:.6f}")
                print(f"  Max diff (valid): {max_diff_valid:.6f}")
                print(f"  Valid points: {np.sum(valid_mask)} / {valid_mask.size}")
                
                # Show sample values
                print(f"  Sample comparisons (first 3 valid points):")
                valid_indices = np.where(valid_mask.ravel())[0][:3]
                for idx in valid_indices:
                    if datacube_values.ndim == 2:
                        dc_val = datacube_values.ravel()[idx]
                        ds_val = dataset_values.ravel()[idx]
                    else:
                        dc_val = datacube_values[idx // datacube_values.shape[1], idx % datacube_values.shape[1]]
                        ds_val = dataset_values[idx // dataset_values.shape[1], idx % dataset_values.shape[1]]
                    print(f"    Point {idx}: Datacube={dc_val:.6f}, Dataset={ds_val:.6f}, Diff={dc_val-ds_val:.6f}")
            else:
                print(f"  ⚠️  No valid points for comparison")
                rmse_valid = mae_valid = max_diff_valid = np.nan
            
            comparison_results[feature_name] = {
                'rmse': rmse,
                'mae': mae,
                'max_diff': max_diff,
                'rmse_valid': rmse_valid,
                'mae_valid': mae_valid,
                'max_diff_valid': max_diff_valid,
                'n_valid': np.sum(valid_mask) if 'valid_mask' in locals() else 0,
                'n_total': valid_mask.size if 'valid_mask' in locals() else 0
            }
        
        self.comparison_results = comparison_results
        return comparison_results
    
    def generate_summary_report(self):
        """Generate a summary report of the comparison"""
        print("\n" + "=" * 60)
        print("SUMMARY REPORT")
        print("=" * 60)
        
        print(f"Zone: {self.zone}")
        print(f"Year: {self.year}")
        print(f"Datacube directory: {self.datacube_dir}")
        print(f"Total dataset points: {len(self.sample_dataset['point_info'])}")
        print(f"Valid points for comparison: {len(self.pixel_coordinates)}")
        print(f"Datacube dimensions: {self.datacube_info['height']}x{self.datacube_info['width']} pixels")
        print(f"Valid pixels in datacube: {self.datacube_info['n_valid_pixels']}")
        
        print("\nFEATURE COMPARISON SUMMARY:")
        print("-" * 40)
        
        for feature_name, results in self.comparison_results.items():
            if np.isnan(results['rmse_valid']):
                status = "❌ No valid data"
            elif results['rmse_valid'] < 1e-6:
                status = "✅ Perfect match"
            elif results['rmse_valid'] < 1e-3:
                status = "✅ Excellent match"
            elif results['rmse_valid'] < 1e-2:
                status = "⚠️  Good match"
            else:
                status = "⚠️  Poor match"
                
            print(f"{feature_name:10s}: RMSE={results['rmse_valid']:.6f}, MAE={results['mae_valid']:.6f} {status}")
        
        # Save detailed report
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = f'/fs1/tor_project/model_foresttype/reports/datacube_comparison_{self.zone}_{self.year}_{timestamp}.txt'
        
        os.makedirs(os.path.dirname(report_path), exist_ok=True)
        
        with open(report_path, 'w') as f:
            f.write("DATACUBE COMPARISON REPORT\n")
            f.write("=" * 60 + "\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Zone: {self.zone}\n")
            f.write(f"Year: {self.year}\n")
            f.write(f"Datacube directory: {self.datacube_dir}\n\n")
            
            f.write("SUMMARY STATISTICS:\n")
            f.write("-" * 20 + "\n")
            f.write(f"Total dataset points: {len(self.sample_dataset['point_info'])}\n")
            f.write(f"Valid points for comparison: {len(self.pixel_coordinates)}\n")
            f.write(f"Datacube dimensions: {self.datacube_info['height']}x{self.datacube_info['width']} pixels\n")
            f.write(f"Valid pixels in datacube: {self.datacube_info['n_valid_pixels']}\n\n")
            
            f.write("FEATURE COMPARISON RESULTS:\n")
            f.write("-" * 30 + "\n")
            for feature_name, results in self.comparison_results.items():
                f.write(f"{feature_name}:\n")
                f.write(f"  RMSE (all): {results['rmse']:.6f}\n")
                f.write(f"  MAE (all): {results['mae']:.6f}\n")
                f.write(f"  Max diff: {results['max_diff']:.6f}\n")
                f.write(f"  RMSE (valid): {results['rmse_valid']:.6f}\n")
                f.write(f"  MAE (valid): {results['mae_valid']:.6f}\n")
                f.write(f"  Max diff (valid): {results['max_diff_valid']:.6f}\n")
                f.write(f"  Valid points: {results['n_valid']} / {results['n_total']}\n\n")
        
        print(f"\n📄 Detailed report saved to: {report_path}")
        return report_path
    
    def run_comparison(self):
        """Run the complete comparison workflow"""
        print("🚀 STARTING DATACUBE COMPARISON")
        print("=" * 80)
        
        steps = [
            ("Loading full datacube", self.load_full_datacube),
            ("Loading sample dataset", self.load_sample_dataset), 
            ("Converting coordinates", self.get_pixel_coordinates_from_points),
            ("Sampling datacube at points", self.sample_datacube_at_points),
            ("Comparing samples", self.compare_samples_with_dataset),
            ("Generating report", self.generate_summary_report)
        ]
        
        for step_name, step_func in steps:
            print(f"\n🔄 {step_name}...")
            try:
                success = step_func()
                if success is False:
                    print(f"❌ Failed at step: {step_name}")
                    return False
                print(f"✅ Completed: {step_name}")
            except Exception as e:
                print(f"❌ Error in {step_name}: {e}")
                return False
        
        print("\n" + "=" * 80)
        print("🎉 DATACUBE COMPARISON COMPLETED SUCCESSFULLY!")
        print("=" * 80)
        return True

def main():
    """Main function to run the datacube comparison"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Compare datacube with dataset')
    parser.add_argument('--use-extract-logic', action='store_true', 
                       help='Use extract_forest_type_dataset.py file selection logic')
    parser.add_argument('--compare-both', action='store_true',
                       help='Compare both prediction and extract file selection methods')
    
    args = parser.parse_args()
    
    print("Forest Type Datacube Comparison Tool")
    print("=" * 40)
    
    if args.compare_both:
        print("🔄 Running comparison with both methods...")
        
        print("\n" + "=" * 80)
        print("COMPARISON 1: Using prediction_ftype_2_2_1.py file logic (hardcoded filenames)")
        print("=" * 80)
        comparator1 = DatacubeComparison(use_extract_file_logic=False)
        success1 = comparator1.run_comparison()
        
        print("\n" + "=" * 80)
        print("COMPARISON 2: Using extract_forest_type_dataset.py file logic (prioritized selection)")
        print("=" * 80)
        comparator2 = DatacubeComparison(use_extract_file_logic=True)
        success2 = comparator2.run_comparison()
        
        if success1 and success2:
            print("\n🎯 BOTH COMPARISONS COMPLETED - CHECK DIFFERENCES IN REPORTS")
            return 0
        else:
            print("\n❌ One or both comparisons failed!")
            return 1
            
    else:
        # Initialize and run comparison
        comparator = DatacubeComparison(use_extract_file_logic=args.use_extract_logic)
        
        success = comparator.run_comparison()
        
        if success:
            print("\n✅ Comparison completed successfully!")
            print("The datacube and dataset have been compared at the same spatial points.")
        else:
            print("\n❌ Comparison failed!")
            return 1
            
        return 0

if __name__ == "__main__":
    exit(main())