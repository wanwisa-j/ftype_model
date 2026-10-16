#!/usr/bin/env python3
"""
Forest Type Classification Dataset Extraction Script

This script extracts satellite time-series features and static features at ground truth
point locations for forest type classification. It supports flexible selection of zones,
years, and features while preserving raw values without scaling.

Usage:
    python extract_forest_type_dataset.py [--config config.yaml] [--output /path/to/output]

Author: Generated for forest type classification project
Date: September 2025
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from datetime import datetime
import yaml
import json
from multiprocessing import Pool, cpu_count
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import functools
from typing import List, Dict, Tuple, Any, Optional

import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.sample import sample_gen
import xarray as xr
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('dataset_extraction.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def extract_feature_for_zone_year(args: Tuple[str, str, int, Dict, gpd.GeoDataFrame]) -> pd.DataFrame:
    """
    Extract features for a specific zone-year combination.
    This function is designed to be used with multiprocessing.
    
    Args:
        args: Tuple containing (feature_name, zone, year, config, zone_points)
    
    Returns:
        DataFrame with extracted feature data
    """
    feature_name, zone, year, config, zone_points = args
    
    feature_data = []
    
    try:
        datacube_dir = config['datacube_dirs'][zone]
        feature_dir = Path(datacube_dir) / feature_name
        
        if not feature_dir.exists():
            return pd.DataFrame()
        
        # Look for year-specific files
        year_files = list(feature_dir.glob(f"*{year}*.tif"))
        
        if not year_files:
            return pd.DataFrame()
        
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
            return pd.DataFrame()
        
        # Process the selected main file
        file_path = main_file
        
        with rasterio.open(file_path) as src:
            # Convert coordinates to match raster CRS
            if zone_points.crs != src.crs:
                zone_points_raster_crs = zone_points.to_crs(src.crs)
                coords = [(point.x, point.y) for point in zone_points_raster_crs.geometry]
            else:
                coords = [(point.x, point.y) for point in zone_points.geometry]
            
            # Sample raster at point locations
            samples = list(sample_gen(src, coords))
            
            # Handle multi-band data (time series)
            if src.count > 1:
                # Multi-band file (weekly/temporal data)
                for band_idx in range(src.count):
                    band_values = [sample[band_idx] if sample is not None else np.nan 
                                 for sample in samples]
                    
                    # Create feature records
                    for i, (_, point_row) in enumerate(zone_points.iterrows()):
                        feature_data.append({
                            'point_id': point_row.name,
                            'zone': zone,
                            'feature': feature_name,
                            'year': year,
                            'band': band_idx + 1,
                            'week': band_idx + 1,  # Assuming weekly data
                            'value': band_values[i],
                            'file_path': str(file_path),
                            'utm_x': point_row['utm_x'],
                            'utm_y': point_row['utm_y']
                        })
            else:
                # Single band file
                band_values = [sample[0] if sample is not None else np.nan 
                             for sample in samples]
                
                for i, (_, point_row) in enumerate(zone_points.iterrows()):
                    feature_data.append({
                        'point_id': point_row.name,
                        'zone': zone,
                        'feature': feature_name,
                        'year': year,
                        'band': 1,
                        'week': None,
                        'value': band_values[i],
                        'file_path': str(file_path),
                        'utm_x': point_row['utm_x'],
                        'utm_y': point_row['utm_y']
                    })
                    
    except Exception as e:
        # Log error but don't fail completely
        logger.error(f"Error processing {feature_name} {zone} {year}: {e}")
        
    return pd.DataFrame(feature_data)

def extract_static_feature_for_zone(args: Tuple[str, str, Dict, gpd.GeoDataFrame]) -> pd.DataFrame:
    """
    Extract static features for a specific zone.
    This function is designed to be used with multiprocessing.
    
    Args:
        args: Tuple containing (feature_name, zone, config, zone_points)
    
    Returns:
        DataFrame with extracted static feature data
    """
    feature_name, zone, config, zone_points = args
    
    feature_data = []
    
    try:
        datacube_dir = config['datacube_dirs'][zone]
        feature_dir = Path(datacube_dir) / feature_name
        
        if not feature_dir.exists():
            return pd.DataFrame()
        
        # Look for static feature files
        feature_files = list(feature_dir.glob("*.tif"))
        
        if not feature_files:
            return pd.DataFrame()
        
        # Use the first available file
        file_path = feature_files[0]
        
        with rasterio.open(file_path) as src:
            # Convert coordinates to match raster CRS
            if zone_points.crs != src.crs:
                zone_points_raster_crs = zone_points.to_crs(src.crs)
                coords = [(point.x, point.y) for point in zone_points_raster_crs.geometry]
            else:
                coords = [(point.x, point.y) for point in zone_points.geometry]
            
            # Sample raster at point locations
            samples = list(sample_gen(src, coords))
            band_values = [sample[0] if sample is not None else np.nan 
                         for sample in samples]
            
            # Create feature records
            for i, (_, point_row) in enumerate(zone_points.iterrows()):
                feature_data.append({
                    'point_id': point_row.name,
                    'zone': zone,
                    'feature': feature_name,
                    'year': None,  # Static feature
                    'band': 1,
                    'week': None,
                    'value': band_values[i],
                    'file_path': str(file_path),
                    'utm_x': point_row['utm_x'],
                    'utm_y': point_row['utm_y']
                })
                
    except Exception as e:
        logger.error(f"Error processing static feature {feature_name} {zone}: {e}")
        
    return pd.DataFrame(feature_data)

class ForestTypeDatasetExtractor:
    """
    Extract forest type classification dataset from satellite datacubes and ground truth labels.
    """
    
    def __init__(self, config_path=None, n_workers=None):
        """Initialize extractor with configuration."""
        self.config = self._load_config(config_path)
        self._setup_paths()
        self._validate_config()
        
        # Set up parallel processing
        self.n_workers = n_workers if n_workers is not None else min(cpu_count(), 8)  # Limit to 8 cores max by default
        logger.info(f"Initialized with {self.n_workers} worker processes")
        
    def _load_config(self, config_path):
        """Load configuration from file or use defaults."""
        default_config = {
            'dataset_output_dir': '/fs2/ftype_mflf/shared_data/dataset/1.2.0',
            'label_data_path': '/fs1/tor_project/model_foresttype/data/2025_data_label_20250726/groud_label_data_ftype_for_model_20250826/ftype_total_2024_2025_train_model_20250826.csv',
            'bbox_areas': {
                'mfl_cmf_2024': '/fs3/datacenter/ftype_mflf/shared_data/boundary/mfl_cmf_2024.parquet',
                'mfl_cmf_area_project_2025_chr_zone_1': '/fs3/datacenter/ftype_mflf/shared_data/boundary/mfl_cmf_area_project_2025_chr_zone_1-fit_areas.parquet',
                'mfl_cmf_area_project_2025_chr_zone_2': '/fs3/datacenter/ftype_mflf/shared_data/boundary/mfl_cmf_area_project_2025_chr_zone_2-fit_areas.parquet',
                'zone_1': '/fs3/datacenter/ftype_mflf/shared_data/boundary/mfl_cmf_project_2025_phase2/split/zone_1.parquet',
                'zone_2': '/fs3/datacenter/ftype_mflf/shared_data/boundary/mfl_cmf_project_2025_phase2/split/zone_2.parquet',
                'zone_3': '/fs3/datacenter/ftype_mflf/shared_data/boundary/mfl_cmf_project_2025_phase2/split/zone_3.parquet',
                'zone_4': '/fs3/datacenter/ftype_mflf/shared_data/boundary/mfl_cmf_project_2025_phase2/split/zone_4.parquet',
                'zone_5': '/fs3/datacenter/ftype_mflf/shared_data/boundary/mfl_cmf_project_2025_phase2/split/zone_5.parquet'
            },
            'datacube_dirs': {
                'mfl_cmf_2024': '/fs2/ftype_mflf/shared_data/datacube/2024/1.2.0/mfl_cmf_2024',
                'mfl_cmf_area_project_2025_chr_zone_1': '/fs2/ftype_mflf/shared_data/datacube/2025/1.2.0/mfl_cmf_area_project_2025_chr_zone_1',
                'mfl_cmf_area_project_2025_chr_zone_2': '/fs2/ftype_mflf/shared_data/datacube/2025/1.2.0/mfl_cmf_area_project_2025_chr_zone_2',
                'zone_1': '/fs2/ftype_mflf/shared_data/datacube/2025/1.2.0/zone_1',
                'zone_2': '/fs2/ftype_mflf/shared_data/datacube/2025/1.2.0/zone_2',
                'zone_3': '/fs2/ftype_mflf/shared_data/datacube/2025/1.2.0/zone_3',
                'zone_4': '/fs2/ftype_mflf/shared_data/datacube/2025/1.2.0/zone_4',
                'zone_5': '/fs2/ftype_mflf/shared_data/datacube/2025/1.2.0/zone_5'
            },
            'selection': {
                'zones': ['mfl_cmf_2024', 'mfl_cmf_area_project_2025_chr_zone_1', 'mfl_cmf_area_project_2025_chr_zone_2'],
                'features': {
                    # Time-series features with zone-specific year ranges
                    'lai': {'default': [2019, 2020, 2021, 2022, 2023, 2024, 2025], 'mfl_cmf_2024': [2019, 2020, 2021, 2022, 2023, 2024]},
                    'fapar': {'default': [2019, 2020, 2021, 2022, 2023, 2024, 2025], 'mfl_cmf_2024': [2019, 2020, 2021, 2022, 2023, 2024]},
                    'fcover': {'default': [2019, 2020, 2021, 2022, 2023, 2024, 2025], 'mfl_cmf_2024': [2019, 2020, 2021, 2022, 2023, 2024]},
                    'ndvi': {'default': [2019, 2020, 2021, 2022, 2023, 2024, 2025], 'mfl_cmf_2024': [2019, 2020, 2021, 2022, 2023, 2024]},
                    'gvmi': {'default': [2019, 2020, 2021, 2022, 2023, 2024, 2025], 'mfl_cmf_2024': [2019, 2020, 2021, 2022, 2023, 2024]},
                    'slavi': {'default': [2019, 2020, 2021, 2022, 2023, 2024, 2025], 'mfl_cmf_2024': [2019, 2020, 2021, 2022, 2023, 2024]},
                    'vh': {'default': [2019, 2020, 2021, 2022, 2023, 2024, 2025], 'mfl_cmf_2024': [2019, 2020, 2021, 2022, 2023, 2024]},
                    'vv': {'default': [2019, 2020, 2021, 2022, 2023, 2024, 2025], 'mfl_cmf_2024': [2019, 2020, 2021, 2022, 2023, 2024]},
                    'elevation': True,
                    'slope': True
                },
                'targets': ['ftype_mix', 'ftype_nonmix'],
                'preserve_raw_values': True
            },
            'parallel_processing': {
                'enabled': True,
                'n_workers': None,  # Auto-detect CPU cores
                'chunk_size': 1,
                'use_processes': True  # True for ProcessPool, False for ThreadPool
            }
        }
        
        if config_path and Path(config_path).exists():
            with open(config_path, 'r') as f:
                user_config = yaml.safe_load(f)
                # Deep merge configurations
                default_config.update(user_config)
        
        return default_config
    
    def _setup_paths(self):
        """Setup and create necessary directories."""
        self.output_dir = Path(self.config['dataset_output_dir'])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        (self.output_dir / 'raw').mkdir(exist_ok=True)
        (self.output_dir / 'processed').mkdir(exist_ok=True)
        (self.output_dir / 'metadata').mkdir(exist_ok=True)
    
    def _validate_config(self):
        """Validate configuration and paths."""
        logger.info("Validating configuration...")
        
        # Check label data path
        if not Path(self.config['label_data_path']).exists():
            raise FileNotFoundError(f"Label data not found: {self.config['label_data_path']}")
        
        # Check selected zones exist in configuration
        for zone in self.config['selection']['zones']:
            if zone not in self.config['bbox_areas']:
                raise ValueError(f"Zone '{zone}' not found in bbox_areas configuration")
            if zone not in self.config['datacube_dirs']:
                raise ValueError(f"Zone '{zone}' not found in datacube_dirs configuration")
        
        logger.info("Configuration validation passed")
    
    def load_and_prepare_labels(self):
        """Load ground truth labels and perform spatial joins with zones."""
        logger.info("Loading ground truth labels...")
        
        # Load label data
        label_df = gpd.read_file(self.config['label_data_path'])
        
        # Filter to ground data only
        label_df = label_df[label_df['data'] == 'ground'].copy()
        
        # Ensure numeric coordinates
        label_df['utm_x'] = pd.to_numeric(label_df['utm_x'], errors='coerce')
        label_df['utm_y'] = pd.to_numeric(label_df['utm_y'], errors='coerce')
        
        # Create geometry and set CRS
        label_df = gpd.GeoDataFrame(
            label_df,
            geometry=gpd.points_from_xy(label_df['utm_x'], label_df['utm_y']),
            crs="EPSG:32647"
        )
        
        logger.info(f"Loaded {len(label_df)} ground truth points")
        
        # Load and combine boundary zones
        logger.info("Loading boundary zones...")
        bbox_gdf = gpd.GeoDataFrame()
        
        for zone, path in self.config['bbox_areas'].items():
            if not Path(path).exists():
                logger.warning(f"Boundary file not found for zone {zone}: {path}")
                continue
                
            zone_gdf = gpd.read_parquet(path)
            zone_gdf['zone'] = zone
            
            if bbox_gdf.empty:
                bbox_gdf = zone_gdf.copy()
            else:
                if zone_gdf.crs != bbox_gdf.crs:
                    zone_gdf = zone_gdf.to_crs(bbox_gdf.crs)
                bbox_gdf = pd.concat([bbox_gdf, zone_gdf], ignore_index=True)
        
        # Perform spatial join
        logger.info("Performing spatial join...")
        if label_df.crs != bbox_gdf.crs:
            label_df = label_df.to_crs(bbox_gdf.crs)
        
        joined_df = gpd.sjoin(label_df, bbox_gdf, how='left', predicate='within')
        
        # Filter to selected zones
        selected_zones = self.config['selection']['zones']
        joined_df = joined_df[joined_df['zone'].isin(selected_zones)].copy()
        
        # Add datacube paths
        joined_df['datacube_path'] = joined_df['zone'].map(self.config['datacube_dirs'])
        
        logger.info(f"Points by zone after filtering:")
        for zone in selected_zones:
            count = len(joined_df[joined_df['zone'] == zone])
            logger.info(f"  {zone}: {count} points")
        
        self.points_df = joined_df
        return joined_df
    
    def get_years_for_zone_feature(self, feature_name, zone):
        """Get the appropriate years for a specific zone and feature."""
        feature_config = self.config['selection']['features'][feature_name]
        
        if isinstance(feature_config, dict):
            # Zone-specific configuration
            if zone in feature_config:
                return feature_config[zone]
            else:
                return feature_config.get('default', [])
        elif isinstance(feature_config, list):
            # Simple list of years
            return feature_config
        else:
            # Static feature (True/False)
            return None
    
    def extract_time_series_features(self, feature_name):
        """Extract time-series features with zone-specific years using parallel processing."""
        logger.info(f"Extracting {feature_name} features")
        
        # Check if parallel processing is enabled
        parallel_config = self.config.get('parallel_processing', {})
        use_parallel = parallel_config.get('enabled', True)
        use_processes = parallel_config.get('use_processes', True)
        
        if use_parallel and self.n_workers > 1:
            return self._extract_time_series_features_parallel(feature_name, use_processes)
        else:
            return self._extract_time_series_features_serial(feature_name)
    
    def _extract_time_series_features_parallel(self, feature_name, use_processes=True):
        """Extract time-series features using parallel processing."""
        logger.info(f"Using {'process' if use_processes else 'thread'} parallelization with {self.n_workers} workers")
        
        # Prepare tasks for parallel execution
        tasks = []
        for zone in self.config['selection']['zones']:
            zone_points = self.points_df[self.points_df['zone'] == zone].copy()
            if len(zone_points) == 0:
                continue
            
            # Get years for this specific zone and feature
            years = self.get_years_for_zone_feature(feature_name, zone)
            if not years:
                continue
            
            logger.info(f"Preparing {feature_name} tasks for zone {zone} with years {years} ({len(zone_points)} points)")
            
            for year in years:
                tasks.append((feature_name, zone, year, self.config, zone_points))
        
        if not tasks:
            logger.warning(f"No tasks prepared for feature {feature_name}")
            return pd.DataFrame()
        
        logger.info(f"Executing {len(tasks)} parallel tasks for {feature_name}")
        
        # Execute tasks in parallel
        all_results = []
        
        if use_processes:
            # Use ProcessPoolExecutor for CPU-bound tasks
            with ProcessPoolExecutor(max_workers=self.n_workers) as executor:
                # Submit all tasks
                future_to_task = {executor.submit(extract_feature_for_zone_year, task): task 
                                for task in tasks}
                
                # Collect results with progress bar
                for future in tqdm(as_completed(future_to_task), 
                                 total=len(tasks), 
                                 desc=f"Processing {feature_name}"):
                    try:
                        result = future.result()
                        if not result.empty:
                            all_results.append(result)
                    except Exception as e:
                        task = future_to_task[future]
                        logger.error(f"Task failed for {task[0]} {task[1]} {task[2]}: {e}")
        else:
            # Use ThreadPoolExecutor for I/O-bound tasks
            with ThreadPoolExecutor(max_workers=self.n_workers) as executor:
                # Submit all tasks
                future_to_task = {executor.submit(extract_feature_for_zone_year, task): task 
                                for task in tasks}
                
                # Collect results with progress bar
                for future in tqdm(as_completed(future_to_task), 
                                 total=len(tasks), 
                                 desc=f"Processing {feature_name}"):
                    try:
                        result = future.result()
                        if not result.empty:
                            all_results.append(result)
                    except Exception as e:
                        task = future_to_task[future]
                        logger.error(f"Task failed for {task[0]} {task[1]} {task[2]}: {e}")
        
        # Combine all results
        if all_results:
            combined_df = pd.concat(all_results, ignore_index=True)
            logger.info(f"Extracted {len(combined_df)} records for {feature_name}")
            return combined_df
        else:
            logger.warning(f"No data extracted for {feature_name}")
            return pd.DataFrame()
    
    def _extract_time_series_features_serial(self, feature_name):
        """Extract time-series features using serial processing (original method)."""
        logger.info(f"Using serial processing for {feature_name}")
        
        feature_data = []
        
        for zone in self.config['selection']['zones']:
            zone_points = self.points_df[self.points_df['zone'] == zone].copy()
            if len(zone_points) == 0:
                continue
            
            # Get years for this specific zone and feature
            years = self.get_years_for_zone_feature(feature_name, zone)
            if not years:
                continue
                
            logger.info(f"Processing {feature_name} for zone {zone} with years {years} ({len(zone_points)} points)")
            
            for year in years:
                logger.info(f"Processing {feature_name} {zone} {year}")
                
                # Use the parallel processing function for individual zone-year
                task = (feature_name, zone, year, self.config, zone_points)
                result_df = extract_feature_for_zone_year(task)
                
                if not result_df.empty:
                    feature_data.append(result_df)
        
        if feature_data:
            return pd.concat(feature_data, ignore_index=True)
        else:
            return pd.DataFrame()
    
    def extract_static_features(self, feature_name):
        """Extract static features (elevation, slope) using parallel processing."""
        logger.info(f"Extracting static feature: {feature_name}")
        
        # Check if parallel processing is enabled
        parallel_config = self.config.get('parallel_processing', {})
        use_parallel = parallel_config.get('enabled', True)
        use_processes = parallel_config.get('use_processes', True)
        
        if use_parallel and self.n_workers > 1:
            return self._extract_static_features_parallel(feature_name, use_processes)
        else:
            return self._extract_static_features_serial(feature_name)
    
    def _extract_static_features_parallel(self, feature_name, use_processes=True):
        """Extract static features using parallel processing."""
        logger.info(f"Using {'process' if use_processes else 'thread'} parallelization for static {feature_name}")
        
        # Prepare tasks for parallel execution
        tasks = []
        for zone in self.config['selection']['zones']:
            zone_points = self.points_df[self.points_df['zone'] == zone].copy()
            if len(zone_points) == 0:
                continue
            
            tasks.append((feature_name, zone, self.config, zone_points))
        
        if not tasks:
            logger.warning(f"No tasks prepared for static feature {feature_name}")
            return pd.DataFrame()
        
        logger.info(f"Executing {len(tasks)} parallel tasks for static {feature_name}")
        
        # Execute tasks in parallel
        all_results = []
        
        if use_processes:
            # Use ProcessPoolExecutor for CPU-bound tasks
            with ProcessPoolExecutor(max_workers=self.n_workers) as executor:
                # Submit all tasks
                future_to_task = {executor.submit(extract_static_feature_for_zone, task): task 
                                for task in tasks}
                
                # Collect results
                for future in tqdm(as_completed(future_to_task), 
                                 total=len(tasks), 
                                 desc=f"Processing static {feature_name}"):
                    try:
                        result = future.result()
                        if not result.empty:
                            all_results.append(result)
                    except Exception as e:
                        task = future_to_task[future]
                        logger.error(f"Static task failed for {task[0]} {task[1]}: {e}")
        else:
            # Use ThreadPoolExecutor for I/O-bound tasks
            with ThreadPoolExecutor(max_workers=self.n_workers) as executor:
                # Submit all tasks
                future_to_task = {executor.submit(extract_static_feature_for_zone, task): task 
                                for task in tasks}
                
                # Collect results
                for future in tqdm(as_completed(future_to_task), 
                                 total=len(tasks), 
                                 desc=f"Processing static {feature_name}"):
                    try:
                        result = future.result()
                        if not result.empty:
                            all_results.append(result)
                    except Exception as e:
                        task = future_to_task[future]
                        logger.error(f"Static task failed for {task[0]} {task[1]}: {e}")
        
        # Combine all results
        if all_results:
            combined_df = pd.concat(all_results, ignore_index=True)
            logger.info(f"Extracted {len(combined_df)} records for static {feature_name}")
            return combined_df
        else:
            logger.warning(f"No data extracted for static {feature_name}")
            return pd.DataFrame()
    
    def _extract_static_features_serial(self, feature_name):
        """Extract static features using serial processing."""
        logger.info(f"Using serial processing for static {feature_name}")
        
        all_results = []
        
        for zone in self.config['selection']['zones']:
            zone_points = self.points_df[self.points_df['zone'] == zone].copy()
            if len(zone_points) == 0:
                continue
                
            logger.info(f"Processing static {feature_name} for zone {zone} ({len(zone_points)} points)")
            
            # Use the parallel processing function for individual zone
            task = (feature_name, zone, self.config, zone_points)
            result_df = extract_static_feature_for_zone(task)
            
            if not result_df.empty:
                all_results.append(result_df)
        
        if all_results:
            return pd.concat(all_results, ignore_index=True)
        else:
            return pd.DataFrame()
    
    def extract_all_features(self):
        """Extract all configured features."""
        logger.info("Starting feature extraction...")
        
        all_features = []
        features_config = self.config['selection']['features']
        
        # Extract time-series features
        for feature_name, config in features_config.items():
            if isinstance(config, (list, dict)):  # Time-series feature
                feature_df = self.extract_time_series_features(feature_name)
                if not feature_df.empty:
                    all_features.append(feature_df)
            elif config is True:  # Static feature
                feature_df = self.extract_static_features(feature_name)
                if not feature_df.empty:
                    all_features.append(feature_df)
        
        if all_features:
            combined_features = pd.concat(all_features, ignore_index=True)
            logger.info(f"Extracted {len(combined_features)} feature records")
            return combined_features
        else:
            logger.warning("No features were extracted")
            return pd.DataFrame()
    
    def create_dataset(self):
        """Create the final dataset by combining features with labels."""
        logger.info("Creating final dataset...")
        
        # Extract features
        features_df = self.extract_all_features()
        
        if features_df.empty:
            logger.error("No features extracted. Cannot create dataset.")
            return None
        
        # Merge with label data
        labels_subset = self.points_df[['ftype_mix', 'ftype_nonmix', 'utm_x', 'utm_y', 'zone']].copy()
        labels_subset = labels_subset.reset_index().rename(columns={'index': 'point_id'})
        
        # Create final dataset
        dataset = features_df.merge(
            labels_subset,
            on=['point_id', 'zone', 'utm_x', 'utm_y'],
            how='left'
        )
        
        logger.info(f"Final dataset shape: {dataset.shape}")
        logger.info(f"Features summary:")
        logger.info(dataset.groupby(['feature', 'year'])['value'].agg(['count', 'mean', 'std']))
        
        return dataset
    
    def save_dataset(self, dataset):
        """Save dataset and metadata."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save raw dataset
        raw_path = self.output_dir / 'raw' / f'forest_type_dataset_raw_{timestamp}.parquet'
        dataset.to_parquet(raw_path, index=False)
        logger.info(f"Raw dataset saved to: {raw_path}")
        
        # Save pivot table format for ML
        pivot_features = ['feature', 'year', 'week']
        pivot_features = [col for col in pivot_features if col in dataset.columns]
        
        if len(pivot_features) > 1:
            # Create feature names for pivoting
            dataset['feature_name'] = dataset.apply(
                lambda row: f"{row['feature']}_" + 
                           (f"{int(row['year'])}_" if pd.notna(row['year']) else "static_") +
                           (f"w{int(row['week'])}" if pd.notna(row['week']) else ""),
                axis=1
            )
            
            # Pivot to wide format
            wide_dataset = dataset.pivot_table(
                index=['point_id', 'zone', 'utm_x', 'utm_y', 'ftype_mix', 'ftype_nonmix'],
                columns='feature_name',
                values='value',
                aggfunc='mean'  # In case of duplicates
            ).reset_index()
            
            # Save wide format
            wide_path = self.output_dir / 'processed' / f'forest_type_dataset_wide_{timestamp}.parquet'
            wide_dataset.to_parquet(wide_path, index=False)
            logger.info(f"Wide format dataset saved to: {wide_path}")
        
        # Save metadata
        metadata = {
            'extraction_timestamp': timestamp,
            'config': self.config,
            'dataset_shape': list(dataset.shape),
            'feature_summary': dataset.groupby('feature')['value'].agg(['count', 'mean', 'std']).to_dict(),
            'zone_summary': dataset.groupby('zone')['point_id'].nunique().to_dict(),
            'total_points': dataset['point_id'].nunique(),
            'total_features': len(dataset['feature'].unique())
        }
        
        metadata_path = self.output_dir / 'metadata' / f'extraction_metadata_{timestamp}.json'
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        
        logger.info(f"Metadata saved to: {metadata_path}")
        
        return {
            'raw_dataset_path': raw_path,
            'wide_dataset_path': wide_path if 'wide_dataset' in locals() else None,
            'metadata_path': metadata_path
        }
    
    def run(self):
        """Run the complete extraction process."""
        logger.info("Starting forest type dataset extraction...")
        logger.info(f"Output directory: {self.output_dir}")
        
        try:
            # Load and prepare labels
            self.load_and_prepare_labels()
            
            # Create dataset
            dataset = self.create_dataset()
            
            if dataset is not None and not dataset.empty:
                # Save results
                paths = self.save_dataset(dataset)
                
                logger.info("Dataset extraction completed successfully!")
                logger.info("Generated files:")
                for key, path in paths.items():
                    if path:
                        logger.info(f"  {key}: {path}")
                
                return paths
            else:
                logger.error("Dataset creation failed")
                return None
                
        except Exception as e:
            logger.error(f"Dataset extraction failed: {e}")
            raise

def main():
    """Main function with command line interface."""
    parser = argparse.ArgumentParser(description='Extract forest type classification dataset')
    parser.add_argument('--config', type=str, help='Path to configuration YAML file')
    parser.add_argument('--output', type=str, help='Output directory path')
    parser.add_argument('--workers', type=int, help='Number of parallel workers (default: auto-detect)')
    parser.add_argument('--serial', action='store_true', help='Force serial processing (disable parallelization)')
    parser.add_argument('--threads', action='store_true', help='Use thread-based parallelization instead of process-based')
    parser.add_argument('--verbose', '-v', action='store_true', help='Enable verbose logging')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Initialize extractor
    n_workers = 1 if args.serial else args.workers
    extractor = ForestTypeDatasetExtractor(config_path=args.config, n_workers=n_workers)
    
    # Override parallel processing settings if specified
    if args.serial:
        extractor.config['parallel_processing']['enabled'] = False
    if args.threads:
        extractor.config['parallel_processing']['use_processes'] = False
        
    # Override output directory if provided
    if args.output:
        extractor.config['dataset_output_dir'] = args.output
        extractor._setup_paths()
    
    # Log configuration
    parallel_config = extractor.config.get('parallel_processing', {})
    if parallel_config.get('enabled', True):
        mode = "processes" if parallel_config.get('use_processes', True) else "threads"
        logger.info(f"Parallel processing: {extractor.n_workers} {mode}")
    else:
        logger.info("Parallel processing: disabled (serial mode)")
    
    # Run extraction
    result = extractor.run()
    
    if result:
        print("\n✅ Dataset extraction completed successfully!")
        print(f"📁 Output directory: {extractor.output_dir}")
    else:
        print("\n❌ Dataset extraction failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()
