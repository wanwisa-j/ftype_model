#!/usr/bin/env python3
"""
Data Consistency Investigation Script
=====================================

Purpose: Investigate if data extracted via dataset_builder.py matches
         data extracted via prediction pipeline for the same points.

Issue: Model v2.2.1 passed spatial validation but still shows bias in 
       full predictions, suggesting data inconsistency between training
       and prediction pipelines.

Investigation Plan:
1. Load ground truth points from label CSV
2. Extract data using dataset_builder.py (training method)  
3. Extract data using prediction pipeline method (raster sampling)
4. Compare values at same geographic locations
5. Apply scaling from model config and compare
6. Identify source of inconsistency

Author: Generated for data consistency investigation
Date: 2025-09-10
"""

import os
import sys
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.sample import sample_gen
from rasterio.transform import rowcol
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Add project paths
sys.path.append('/fs1/tor_project/model_foresttype/src')
from dataset_builder import ForestTypeDatasetBuilder

class DataConsistencyInvestigator:
    """Investigate data consistency between training and prediction pipelines."""
    
    def __init__(self):
        self.label_data_path = '/fs1/tor_project/model_foresttype/data/2025_data_label_20250726/groud_label_data_ftype_for_model_20250826/ftype_total_2024_2025_train_model_20250826.csv'
        self.zone = 'mfl_cmf_area_project_2025_chr_zone_1'
        self.datacube_base = '/fs2/ftype_mflf/shared_data/datacube/2025/1.2.0'
        self.model_config_dir = '/fs1/tor_project/model_foresttype/model/mfl_v2_3_0/config'
        
        # Features to investigate
        self.timeseries_features = ['vv', 'vh', 'gvmi', 'ndvi', 'slavi', 'lai', 'fapar', 'fcover']
        self.static_features = ['slope', 'elevation']
        self.years = [2021, 2022, 2023, 2024, 2025]
        
        self.results = {}
        
    def load_ground_truth_points(self):
        """Load ground truth points for the investigation zone."""
        print(f"📍 Loading ground truth points for zone: {self.zone}")
        
        # Load label data
        label_df = pd.read_csv(self.label_data_path)
        print(f"   Loaded {len(label_df)} total label points")
        
        # Filter for ground data only
        label_df = label_df[label_df['data'] == 'ground'].copy()
        print(f"   Filtered to {len(label_df)} ground points")
        
        # Create GeoDataFrame  
        label_df['utm_x'] = pd.to_numeric(label_df['utm_x'], errors='coerce')
        label_df['utm_y'] = pd.to_numeric(label_df['utm_y'], errors='coerce')
        
        label_gdf = gpd.GeoDataFrame(
            label_df,
            geometry=gpd.points_from_xy(label_df['utm_x'], label_df['utm_y']),
            crs="EPSG:32647"
        )
        
        # For investigation, take a representative sample
        sample_size = min(50, len(label_gdf))  # Use 50 points for detailed comparison
        self.sample_points = label_gdf.sample(n=sample_size, random_state=42).reset_index(drop=True)
        
        print(f"   Selected {len(self.sample_points)} points for investigation")
        print(f"   Sample points class distribution:")
        if 'ftype_nonmix' in self.sample_points.columns:
            print(self.sample_points['ftype_nonmix'].value_counts().to_dict())
        
        return self.sample_points
    
    def extract_via_dataset_builder(self):
        """Extract data using dataset_builder.py method (training pipeline)."""
        print(f"\n🔄 METHOD 1: Extracting data via dataset_builder.py")
        
        try:
            builder = ForestTypeDatasetBuilder()
            
            # Extract features for our zone using dataset_builder
            dataset = builder.create_dataset(
                zones=[self.zone],
                years=self.years,
                features=self.timeseries_features + self.static_features,
                target='ftype_nonmix',
                return_point_info=True
            )
            
            print(f"   ✅ Dataset extracted: {dataset['target'].shape[0]} samples")
            
            # Convert to DataFrame for easier comparison
            dataset_df = pd.DataFrame(dataset['point_info'])
            
            # Add feature data (taking first year/timestep for simplicity)
            for feature in self.timeseries_features:
                feature_data = dataset[f'{feature}_feature']
                if feature_data.ndim > 1:
                    # Take mean across timesteps for comparison
                    dataset_df[f'{feature}_mean'] = feature_data.mean(axis=1)
                    dataset_df[f'{feature}_first'] = feature_data[:, 0] if feature_data.shape[1] > 0 else np.nan
                else:
                    dataset_df[f'{feature}_value'] = feature_data
            
            for feature in self.static_features:
                feature_data = dataset[f'{feature}_feature']
                if feature_data.ndim > 1:
                    dataset_df[f'{feature}_value'] = feature_data[:, 0]
                else:
                    dataset_df[f'{feature}_value'] = feature_data
            
            # Add target values
            dataset_df['ftype_nonmix'] = dataset['target'].ravel()
            
            print(f"   Dataset builder columns: {list(dataset_df.columns)}")
            
            self.results['dataset_builder'] = {
                'data': dataset_df,
                'raw_dataset': dataset
            }
            
            return dataset_df
            
        except Exception as e:
            print(f"   ❌ Error in dataset_builder extraction: {e}")
            return None
    
    def extract_via_prediction_pipeline(self):
        """Extract data using prediction pipeline method (direct raster sampling)."""
        print(f"\n🔄 METHOD 2: Extracting data via prediction pipeline method")
        
        try:
            # Initialize results DataFrame
            prediction_df = self.sample_points[['utm_x', 'utm_y', 'ftype_nonmix']].copy()
            
            # Extract each feature using direct raster sampling
            for feature in self.timeseries_features:
                print(f"   Processing feature: {feature}")
                feature_values = self._sample_feature_raster(feature, year=2025)
                prediction_df[f'{feature}_mean'] = feature_values['mean']
                prediction_df[f'{feature}_first'] = feature_values['first']
            
            for feature in self.static_features:
                print(f"   Processing static feature: {feature}")  
                feature_values = self._sample_feature_raster(feature, year=None)
                prediction_df[f'{feature}_value'] = feature_values['value']
            
            print(f"   ✅ Prediction pipeline extraction complete")
            print(f"   Prediction pipeline columns: {list(prediction_df.columns)}")
            
            self.results['prediction_pipeline'] = {
                'data': prediction_df
            }
            
            return prediction_df
            
        except Exception as e:
            print(f"   ❌ Error in prediction pipeline extraction: {e}")
            return None
    
    def _sample_feature_raster(self, feature, year=None):
        """Sample feature raster at sample point locations."""
        try:
            # Determine file path based on feature type
            feature_dir = Path(self.datacube_base) / self.zone / feature
            
            if year is not None:
                # Time series feature
                if feature.lower() in ['vv', 'vh']:
                    filename = f"cube_s1_{feature.lower()}_{year}_W4before_to_W16after_21bands.tif"
                elif feature.lower() in ['gvmi', 'ndvi', 'slavi']:
                    filename = f"cube_s2_{feature.lower()}_{year}_W4before_to_W16after_21bands.tif"
                elif feature.lower() in ['lai', 'fapar', 'fcover']:
                    filename = f"cube_biophys_{feature.lower()}_{year}_W4before_to_W16after_21bands.tif"
                else:
                    print(f"     ⚠️  Unknown timeseries feature: {feature}")
                    return {'mean': [np.nan] * len(self.sample_points), 'first': [np.nan] * len(self.sample_points)}
            else:
                # Static feature
                if feature.lower() == 'elevation':
                    filename = "elevation_data.tif"
                elif feature.lower() == 'slope':
                    filename = "slope_data.tif"
                else:
                    print(f"     ⚠️  Unknown static feature: {feature}")
                    return {'value': [np.nan] * len(self.sample_points)}
            
            file_path = feature_dir / filename
            
            if not file_path.exists():
                print(f"     ⚠️  File not found: {file_path}")
                if year is not None:
                    return {'mean': [np.nan] * len(self.sample_points), 'first': [np.nan] * len(self.sample_points)}
                else:
                    return {'value': [np.nan] * len(self.sample_points)}
            
            # Sample raster at point locations
            coords = [(row.utm_x, row.utm_y) for _, row in self.sample_points.iterrows()]
            
            with rasterio.open(file_path) as src:
                # Sample at coordinates
                samples = list(sample_gen(src, coords))
                
                if year is not None:
                    # Time series - compute statistics
                    means = []
                    firsts = []
                    for sample in samples:
                        if sample is not None and len(sample) > 0:
                            valid_values = [v for v in sample if v is not None and not np.isnan(v)]
                            if valid_values:
                                means.append(np.mean(valid_values))
                                firsts.append(valid_values[0])
                            else:
                                means.append(np.nan)
                                firsts.append(np.nan)
                        else:
                            means.append(np.nan)
                            firsts.append(np.nan)
                    
                    return {'mean': means, 'first': firsts}
                else:
                    # Static feature
                    values = []
                    for sample in samples:
                        if sample is not None and len(sample) > 0:
                            values.append(sample[0])
                        else:
                            values.append(np.nan)
                    
                    return {'value': values}
                    
        except Exception as e:
            print(f"     ❌ Error sampling {feature}: {e}")
            if year is not None:
                return {'mean': [np.nan] * len(self.sample_points), 'first': [np.nan] * len(self.sample_points)}
            else:
                return {'value': [np.nan] * len(self.sample_points)}
    
    def compare_data_sources(self):
        """Compare data from both extraction methods."""
        print(f"\n📊 COMPARING DATA SOURCES")
        print("=" * 60)
        
        if 'dataset_builder' not in self.results or 'prediction_pipeline' not in self.results:
            print("❌ Cannot compare - one or both extraction methods failed")
            return
        
        db_data = self.results['dataset_builder']['data']
        pp_data = self.results['prediction_pipeline']['data']
        
        print(f"Dataset builder samples: {len(db_data)}")
        print(f"Prediction pipeline samples: {len(pp_data)}")
        
        # Find matching points (by coordinates)
        matches = []
        for i, pp_row in pp_data.iterrows():
            # Find closest point in dataset_builder data
            if len(db_data) > 0:
                distances = np.sqrt((db_data['utm_x'] - pp_row['utm_x'])**2 + 
                                  (db_data['utm_y'] - pp_row['utm_y'])**2)
                closest_idx = distances.idxmin()
                if distances[closest_idx] < 10:  # Within 10 meters
                    matches.append((i, closest_idx))
        
        print(f"Found {len(matches)} matching points")
        
        if len(matches) == 0:
            print("❌ No matching points found between methods!")
            return
        
        # Compare feature values
        comparison_results = {}
        
        # Compare timeseries features
        for feature in self.timeseries_features:
            mean_col = f'{feature}_mean'
            first_col = f'{feature}_first'
            
            if mean_col in db_data.columns and mean_col in pp_data.columns:
                pp_vals = []
                db_vals = []
                
                for pp_idx, db_idx in matches:
                    pp_val = pp_data.loc[pp_idx, mean_col]
                    db_val = db_data.loc[db_idx, mean_col]
                    
                    if not (np.isnan(pp_val) or np.isnan(db_val)):
                        pp_vals.append(pp_val)
                        db_vals.append(db_val)
                
                if len(pp_vals) > 0:
                    correlation = np.corrcoef(pp_vals, db_vals)[0, 1] if len(pp_vals) > 1 else np.nan
                    rmse = np.sqrt(np.mean((np.array(pp_vals) - np.array(db_vals))**2))
                    mean_diff = np.mean(np.array(pp_vals) - np.array(db_vals))
                    
                    comparison_results[feature] = {
                        'correlation': correlation,
                        'rmse': rmse, 
                        'mean_difference': mean_diff,
                        'n_samples': len(pp_vals),
                        'pp_mean': np.mean(pp_vals),
                        'db_mean': np.mean(db_vals)
                    }
                    
                    print(f"\n{feature.upper()}:")
                    print(f"   Correlation: {correlation:.4f}")
                    print(f"   RMSE: {rmse:.6f}")
                    print(f"   Mean difference: {mean_diff:.6f}")
                    print(f"   Prediction pipeline mean: {np.mean(pp_vals):.6f}")
                    print(f"   Dataset builder mean: {np.mean(db_vals):.6f}")
                    
                    # Flag significant differences
                    if abs(correlation) < 0.95 or abs(mean_diff) > 0.001:
                        print(f"   🚨 SIGNIFICANT DIFFERENCE DETECTED!")
        
        # Compare static features
        for feature in self.static_features:
            val_col = f'{feature}_value'
            
            if val_col in db_data.columns and val_col in pp_data.columns:
                pp_vals = []
                db_vals = []
                
                for pp_idx, db_idx in matches:
                    pp_val = pp_data.loc[pp_idx, val_col]
                    db_val = db_data.loc[db_idx, val_col]
                    
                    if not (np.isnan(pp_val) or np.isnan(db_val)):
                        pp_vals.append(pp_val)
                        db_vals.append(db_val)
                
                if len(pp_vals) > 0:
                    correlation = np.corrcoef(pp_vals, db_vals)[0, 1] if len(pp_vals) > 1 else np.nan
                    rmse = np.sqrt(np.mean((np.array(pp_vals) - np.array(db_vals))**2))
                    mean_diff = np.mean(np.array(pp_vals) - np.array(db_vals))
                    
                    print(f"\n{feature.upper()} (static):")
                    print(f"   Correlation: {correlation:.4f}")
                    print(f"   RMSE: {rmse:.6f}")
                    print(f"   Mean difference: {mean_diff:.6f}")
                    print(f"   Prediction pipeline mean: {np.mean(pp_vals):.6f}")
                    print(f"   Dataset builder mean: {np.mean(db_vals):.6f}")
                    
                    if abs(correlation) < 0.95 or abs(mean_diff) > 1.0:  # Larger threshold for static features
                        print(f"   🚨 SIGNIFICANT DIFFERENCE DETECTED!")
        
        self.results['comparison'] = comparison_results
    
    def apply_scaling_comparison(self):
        """Apply model scaling to both datasets and compare."""
        print(f"\n🔧 APPLYING MODEL SCALING COMPARISON")
        print("=" * 60)
        
        # Load scaling parameters
        scaling_path = Path(self.model_config_dir) / 'scaling_params_v2_3_0_no_leakage-20250910.npz'
        if not scaling_path.exists():
            # Try alternative path
            scaling_path = Path('/fs1/tor_project/model_foresttype/model/mfl_v2_2_0/config/scaling_params_from_training_hybrid_lstm_cnn_model-20250910.npz')
            
        if not scaling_path.exists():
            print(f"❌ Scaling parameters not found at {scaling_path}")
            return
        
        print(f"📂 Loading scaling parameters from: {scaling_path}")
        
        try:
            scaling_data = np.load(scaling_path)
            
            timeseries_mean = scaling_data['timeseries_mean']  # Shape: (1, 1, 10)
            timeseries_std = scaling_data['timeseries_std']    # Shape: (1, 1, 10)
            static_mean = scaling_data['static_mean']          # Shape: (1, 2)  
            static_std = scaling_data['static_std']            # Shape: (1, 2)
            
            print(f"   Timeseries scaling: mean={timeseries_mean.shape}, std={timeseries_std.shape}")
            print(f"   Static scaling: mean={static_mean.shape}, std={static_std.shape}")
            print(f"   ✅ Scaling parameters loaded successfully")
            
            # Apply scaling and compare
            # This is a simplified comparison - in practice, you'd need to reconstruct
            # the full feature arrays and apply the exact scaling used in training
            
            print("\n📊 SCALING ANALYSIS:")
            print("   Note: This shows the scaling parameters that would be applied")
            print("   Full scaling comparison would require reconstructing complete feature arrays")
            
            # Show scaling parameters for reference
            print(f"\n   Timeseries means (first 8 channels): {timeseries_mean.flatten()[:8]}")
            print(f"   Timeseries stds (first 8 channels): {timeseries_std.flatten()[:8]}")
            print(f"   Static means: {static_mean.flatten()}")
            print(f"   Static stds: {static_std.flatten()}")
            
        except Exception as e:
            print(f"❌ Error loading scaling parameters: {e}")
    
    def generate_summary_report(self):
        """Generate summary report of investigation findings."""
        print(f"\n" + "="*80)
        print("📋 INVESTIGATION SUMMARY REPORT")
        print("="*80)
        
        print(f"🎯 Investigation Focus: {self.zone}")
        print(f"📍 Sample Points: {len(self.sample_points)}")
        print(f"🗓️  Years Investigated: {self.years}")
        print(f"📊 Features: {len(self.timeseries_features)} timeseries + {len(self.static_features)} static")
        
        if 'comparison' in self.results:
            print(f"\n🔍 KEY FINDINGS:")
            
            significant_diffs = []
            for feature, metrics in self.results['comparison'].items():
                if abs(metrics.get('correlation', 1.0)) < 0.95 or abs(metrics.get('mean_difference', 0)) > 0.001:
                    significant_diffs.append(feature)
            
            if significant_diffs:
                print(f"   🚨 SIGNIFICANT DIFFERENCES FOUND in {len(significant_diffs)} features:")
                for feature in significant_diffs:
                    print(f"      - {feature}")
                print(f"\n   💡 CONCLUSION: Data inconsistency detected between training and prediction pipelines!")
                print(f"      This explains why model shows bias in predictions despite passing validation.")
            else:
                print(f"   ✅ NO SIGNIFICANT DIFFERENCES found in feature values")
                print(f"   💡 CONCLUSION: Data extraction methods are consistent")
                print(f"      Issue may be in scaling application or model implementation.")
        else:
            print(f"   ❌ Comparison not completed - check extraction errors above")
        
        print(f"\n📋 RECOMMENDATIONS:")
        if 'comparison' in self.results and len([f for f, m in self.results['comparison'].items() 
                                                if abs(m.get('correlation', 1.0)) < 0.95]) > 0:
            print(f"   1. 🔧 FIX data extraction pipeline inconsistencies")
            print(f"   2. 🔄 Re-train model with consistent data extraction")
            print(f"   3. 🧪 Validate that training and prediction use identical data processing")
            print(f"   4. 📊 Re-run predictions with fixed pipeline")
        else:
            print(f"   1. 🔍 Investigate scaling application in prediction pipeline")
            print(f"   2. 🧪 Check model loading and inference pipeline")
            print(f"   3. 📊 Verify feature preprocessing matches training exactly")
            print(f"   4. 🔧 Check for other sources of bias (e.g., spatial sampling differences)")
        
        print("="*80)
    
    def run_investigation(self):
        """Run the complete investigation."""
        print("🕵️ STARTING DATA CONSISTENCY INVESTIGATION")
        print("="*80)
        
        # Step 1: Load sample points
        self.load_ground_truth_points()
        
        # Step 2: Extract via dataset_builder
        self.extract_via_dataset_builder()
        
        # Step 3: Extract via prediction pipeline
        self.extract_via_prediction_pipeline()
        
        # Step 4: Compare methods
        self.compare_data_sources()
        
        # Step 5: Analyze scaling
        self.apply_scaling_comparison()
        
        # Step 6: Generate report
        self.generate_summary_report()
        
        print(f"\n✅ Investigation complete!")
        return self.results

def main():
    """Run the data consistency investigation."""
    investigator = DataConsistencyInvestigator()
    results = investigator.run_investigation()
    return results

if __name__ == "__main__":
    results = main()