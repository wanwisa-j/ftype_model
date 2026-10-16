"""
Feature Extraction System for Forest Type Classification Dataset
Extracts time-series and static features from datacubes at labeled point locations
"""

import os
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.sample import sample_gen
import xarray as xr
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import pickle
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import logging

warnings.filterwarnings('ignore', category=rasterio.errors.NotGeoreferencedWarning)

class ForestTypeFeatureExtractor:
    """
    Extract features from datacubes for forest type classification training
    """
    
    def __init__(self, 
                 datacube_config: Dict[str, str],
                 bbox_areas_config: Dict[str, str],
                 time_series_features: List[str] = None,
                 static_features: List[str] = None,
                 years_range: Tuple[int, int] = (2019, 2024)):
        """
        Initialize the feature extractor
        
        Args:
            datacube_config: Dict mapping zone names to datacube directory paths
            bbox_areas_config: Dict mapping zone names to boundary file paths
            time_series_features: List of time-series feature names
            static_features: List of static feature names
            years_range: Tuple of (start_year, end_year) for data collection
        """
        self.datacube_config = datacube_config
        self.bbox_areas_config = bbox_areas_config
        self.years_range = years_range
        
        self.time_series_features = time_series_features or [
            'lai', 'fapar', 'fcover', 'gvmi', 'ndvi', 'slavi', 'vh', 'vv'
        ]
        
        self.static_features = static_features or [
            'elevation', 'slope'
        ]
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
    def load_label_data(self, label_file_path: str) -> gpd.GeoDataFrame:
        """Load and preprocess label data"""
        self.logger.info(f"Loading label data from {label_file_path}")
        
        # Load the label data
        label_df = pd.read_csv(label_file_path)
        
        # Convert to numeric coordinates
        label_df['utm_x'] = pd.to_numeric(label_df['utm_x'], errors='coerce')
        label_df['utm_y'] = pd.to_numeric(label_df['utm_y'], errors='coerce')
        
        # Create GeoDataFrame
        label_gdf = gpd.GeoDataFrame(
            label_df,
            geometry=gpd.points_from_xy(label_df['utm_x'], label_df['utm_y']),
            crs="EPSG:32647"
        )
        
        # Filter for ground data only
        label_gdf = label_gdf[label_gdf['data'] == 'ground'].copy()
        
        self.logger.info(f"Loaded {len(label_gdf)} ground truth points")
        return label_gdf
        
    def assign_zones_to_points(self, label_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """Assign zone information to label points through spatial join"""
        self.logger.info("Assigning zones to label points")
        
        # Load all boundary areas
        bbox_areas_gdf = gpd.GeoDataFrame()
        for zone, path in self.bbox_areas_config.items():
            zone_gdf = gpd.read_parquet(path)
            zone_gdf['zone'] = zone
            
            if bbox_areas_gdf.empty:
                bbox_areas_gdf = zone_gdf.copy()
            else:
                if zone_gdf.crs != bbox_areas_gdf.crs:
                    zone_gdf = zone_gdf.to_crs(bbox_areas_gdf.crs)
                bbox_areas_gdf = pd.concat([bbox_areas_gdf, zone_gdf], ignore_index=True)
        
        # Ensure CRS match
        if label_gdf.crs != bbox_areas_gdf.crs:
            label_gdf = label_gdf.to_crs(bbox_areas_gdf.crs)
        
        # Spatial join
        joined_gdf = gpd.sjoin(label_gdf, bbox_areas_gdf, how='left', predicate='within')
        
        # Remove points that don't fall within any zone
        joined_gdf = joined_gdf.dropna(subset=['zone']).copy()
        
        # Add datacube path information
        joined_gdf['datacube_path'] = joined_gdf['zone'].map(self.datacube_config)
        
        self.logger.info(f"Assigned zones to {len(joined_gdf)} points")
        return joined_gdf
        
    def extract_time_series_features_at_point(self, 
                                            point_coords: Tuple[float, float],
                                            datacube_path: str,
                                            feature_name: str,
                                            years: List[int]) -> Dict[str, np.ndarray]:
        """Extract time-series features at a single point location"""
        point_features = {}
        
        for year in years:
            # Construct file path based on feature type
            if feature_name in ['vh', 'vv']:
                file_pattern = f"cube_s1_{feature_name}_{year}_W4before_to_W16after_21bands.tif"
            elif feature_name in ['gvmi', 'ndvi', 'slavi']:
                file_pattern = f"cube_s2_{feature_name}_{year}_W4before_to_W16after_21bands.tif"
            elif feature_name in ['lai', 'fapar', 'fcover']:
                file_pattern = f"cube_biophy_{feature_name}_{year}_W4before_to_W16after_21bands.tif"
            else:
                continue
                
            file_path = os.path.join(datacube_path, feature_name, file_pattern)
            
            if os.path.exists(file_path):
                try:
                    with rasterio.open(file_path) as src:
                        # Sample at point location
                        samples = list(sample_gen(src, [point_coords]))
                        if samples and len(samples[0]) > 0:
                            time_series = np.array(samples[0])
                            # Handle nodata values
                            if src.nodata is not None:
                                time_series = np.where(time_series == src.nodata, np.nan, time_series)
                            point_features[f"{feature_name}_{year}"] = time_series
                        else:
                            point_features[f"{feature_name}_{year}"] = np.full(21, np.nan)
                except Exception as e:
                    self.logger.warning(f"Error reading {file_path}: {e}")
                    point_features[f"{feature_name}_{year}"] = np.full(21, np.nan)
            else:
                self.logger.warning(f"File not found: {file_path}")
                point_features[f"{feature_name}_{year}"] = np.full(21, np.nan)
                
        return point_features
        
    def extract_static_features_at_point(self, 
                                       point_coords: Tuple[float, float],
                                       datacube_path: str,
                                       feature_name: str) -> float:
        """Extract static features at a single point location"""
        file_path = os.path.join(datacube_path, feature_name, f"{feature_name}_data.tif")
        
        if os.path.exists(file_path):
            try:
                with rasterio.open(file_path) as src:
                    samples = list(sample_gen(src, [point_coords]))
                    if samples and len(samples[0]) > 0:
                        value = samples[0][0]
                        if src.nodata is not None and value == src.nodata:
                            return np.nan
                        return float(value)
                    else:
                        return np.nan
            except Exception as e:
                self.logger.warning(f"Error reading {file_path}: {e}")
                return np.nan
        else:
            self.logger.warning(f"File not found: {file_path}")
            return np.nan
            
    def extract_features_for_point(self, 
                                 point_row: pd.Series,
                                 years: List[int]) -> Dict:
        """Extract all features for a single point"""
        point_coords = (point_row.geometry.x, point_row.geometry.y)
        datacube_path = point_row['datacube_path']
        
        features = {
            'point_id': point_row.name,
            'utm_x': point_row.geometry.x,
            'utm_y': point_row.geometry.y,
            'zone': point_row['zone'],
            'year': point_row.get('year', None),
            'ftype_mix': point_row.get('ftype_mix', None),
            'ftype_nonmix': point_row.get('ftype_nonmix', None),
        }
        
        # Extract time-series features
        for feature_name in self.time_series_features:
            ts_features = self.extract_time_series_features_at_point(
                point_coords, datacube_path, feature_name, years
            )
            features.update(ts_features)
            
        # Extract static features
        for feature_name in self.static_features:
            static_value = self.extract_static_features_at_point(
                point_coords, datacube_path, feature_name
            )
            features[feature_name] = static_value
            
        return features
        
    def extract_dataset(self, 
                       label_gdf: gpd.GeoDataFrame,
                       output_dir: str,
                       max_workers: int = 4,
                       batch_size: int = 100) -> pd.DataFrame:
        """Extract features for all points and create training dataset"""
        self.logger.info("Starting feature extraction")
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        years = list(range(self.years_range[0], self.years_range[1] + 1))
        all_features = []
        
        # Process in batches to manage memory
        total_points = len(label_gdf)
        processed = 0
        
        for batch_start in range(0, total_points, batch_size):
            batch_end = min(batch_start + batch_size, total_points)
            batch_gdf = label_gdf.iloc[batch_start:batch_end]
            
            self.logger.info(f"Processing batch {batch_start//batch_size + 1}: "
                           f"points {batch_start}-{batch_end-1}")
            
            # Use thread pool for parallel processing
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {
                    executor.submit(self.extract_features_for_point, row, years): idx
                    for idx, row in batch_gdf.iterrows()
                }
                
                batch_features = []
                for future in tqdm(as_completed(futures), 
                                 total=len(futures),
                                 desc=f"Extracting features batch {batch_start//batch_size + 1}"):
                    try:
                        result = future.result()
                        batch_features.append(result)
                        processed += 1
                    except Exception as e:
                        self.logger.error(f"Error processing point: {e}")
                        
            all_features.extend(batch_features)
            
            # Save intermediate results
            if batch_features:
                batch_df = pd.DataFrame(batch_features)
                batch_file = os.path.join(output_dir, f"features_batch_{batch_start//batch_size + 1}.pkl")
                batch_df.to_pickle(batch_file)
                self.logger.info(f"Saved batch to {batch_file}")
        
        # Combine all features into final dataset
        if all_features:
            features_df = pd.DataFrame(all_features)
            
            # Save the complete dataset
            output_file = os.path.join(output_dir, "complete_features_dataset.pkl")
            features_df.to_pickle(output_file)
            
            # Also save as CSV for easy inspection
            csv_file = os.path.join(output_dir, "complete_features_dataset.csv")
            features_df.to_csv(csv_file, index=False)
            
            self.logger.info(f"Feature extraction complete. Dataset saved to {output_file}")
            self.logger.info(f"Dataset shape: {features_df.shape}")
            
            return features_df
        else:
            self.logger.error("No features extracted")
            return pd.DataFrame()
            
    def create_training_datasets(self, 
                               features_df: pd.DataFrame,
                               output_dir: str,
                               test_size: float = 0.2,
                               random_state: int = 42) -> Dict[str, Dict]:
        """Create training datasets for both classification targets"""
        self.logger.info("Creating training datasets")
        
        from sklearn.model_selection import train_test_split
        
        results = {}
        
        for target in ['ftype_mix', 'ftype_nonmix']:
            # Filter out rows with missing target values
            target_df = features_df.dropna(subset=[target]).copy()
            
            if len(target_df) == 0:
                self.logger.warning(f"No valid data for target {target}")
                continue
                
            # Prepare feature columns
            feature_cols = []
            
            # Add time-series features
            for feature_name in self.time_series_features:
                for year in range(self.years_range[0], self.years_range[1] + 1):
                    col_name = f"{feature_name}_{year}"
                    if col_name in target_df.columns:
                        feature_cols.append(col_name)
                        
            # Add static features
            for feature_name in self.static_features:
                if feature_name in target_df.columns:
                    feature_cols.append(feature_name)
            
            # Create feature matrix
            X = target_df[feature_cols].values
            y = target_df[target].values
            
            # Split into train/test
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state, stratify=y
            )
            
            # Store results
            results[target] = {
                'X_train': X_train,
                'X_test': X_test,
                'y_train': y_train,
                'y_test': y_test,
                'feature_names': feature_cols,
                'target_name': target,
                'class_names': list(target_df[target].unique())
            }
            
            # Save dataset
            dataset_file = os.path.join(output_dir, f"training_dataset_{target}.pkl")
            with open(dataset_file, 'wb') as f:
                pickle.dump(results[target], f)
                
            self.logger.info(f"Created dataset for {target}: "
                           f"Train: {X_train.shape}, Test: {X_test.shape}")
            
        return results
        
    def get_feature_statistics(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """Get statistics about the extracted features"""
        stats = []
        
        # Time-series feature statistics
        for feature_name in self.time_series_features:
            for year in range(self.years_range[0], self.years_range[1] + 1):
                col_name = f"{feature_name}_{year}"
                if col_name in features_df.columns:
                    feature_data = features_df[col_name].apply(
                        lambda x: x if isinstance(x, np.ndarray) else np.array([])
                    )
                    
                    # Calculate statistics for each timestep
                    valid_series = [x for x in feature_data if len(x) == 21 and not np.all(np.isnan(x))]
                    
                    if valid_series:
                        ts_array = np.stack(valid_series)
                        stats.append({
                            'feature': col_name,
                            'type': 'time_series',
                            'n_samples': len(valid_series),
                            'mean_across_time': np.nanmean(ts_array),
                            'std_across_time': np.nanstd(ts_array),
                            'missing_ratio': 1 - len(valid_series) / len(features_df)
                        })
        
        # Static feature statistics
        for feature_name in self.static_features:
            if feature_name in features_df.columns:
                feature_data = features_df[feature_name]
                stats.append({
                    'feature': feature_name,
                    'type': 'static',
                    'n_samples': feature_data.notna().sum(),
                    'mean_across_time': feature_data.mean(),
                    'std_across_time': feature_data.std(),
                    'missing_ratio': feature_data.isna().sum() / len(feature_data)
                })
                
        return pd.DataFrame(stats)
