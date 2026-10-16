#!/usr/bin/env python3
"""
Forest Type Classification Dataset Builder

This module provides flexible dataset creation for forest type classification training.
It allows selection of specific zones, years, features, and target variables from the
cleaned parquet dataset.

Features:
- Flexible zone, year, and feature selection
- Handles both temporal (21 timesteps) and static (1 timestep) features
- Creates properly shaped arrays for training
- Supports both ftype_mix and ftype_nonmix targets
- Data validation and consistency checks
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional
import warnings
from pathlib import Path

class ForestTypeDatasetBuilder:
    """
    A flexible dataset builder for forest type classification.
    
    This class loads the cleaned dataset and provides methods to extract
    training data based on specified zones, years, features, and targets.
    """
    
    def __init__(self, dataset_path: str = None):
        """
        Initialize the dataset builder.
        
        Args:
            dataset_path: Path to the cleaned parquet dataset
        """
        if dataset_path is None:
            dataset_path = '/fs2/ftype_mflf/shared_data/dataset/1.2.0/raw/forest_type_dataset_raw_20250909_102320.parquet'
        
        self.dataset_path = dataset_path
        print(f"Loading dataset from: {dataset_path}")
        self.df = pd.read_parquet(dataset_path)
        
        
        # Get available options
        self.available_zones = sorted(self.df['zone'].unique())
        self.available_features = sorted(self.df['feature'].unique())
        self.available_years = sorted([y for y in self.df['year'].unique() if not pd.isna(y)])
        self.available_targets = ['ftype_mix', 'ftype_nonmix']
        
        # Define feature categories
        self.static_features = ['elevation', 'slope']
        self.temporal_features = [f for f in self.available_features if f not in self.static_features]
        
        print(f"Dataset loaded successfully!")
        print(f"Available zones: {self.available_zones}")
        print(f"Available years: {self.available_years}")
        print(f"Available features: {self.available_features}")
        print(f"Static features: {self.static_features}")
        print(f"Temporal features: {self.temporal_features}")
        print(f"Available targets: {self.available_targets}")
    
    def validate_inputs(self, zones: List[str], years: List[float], features: List[str], target: str) -> bool:
        """
        Validate input parameters.
        
        Args:
            zones: List of zone names
            years: List of years
            features: List of feature names
            target: Target variable name
            
        Returns:
            True if all inputs are valid
        """
        # Check zones
        invalid_zones = [z for z in zones if z not in self.available_zones]
        if invalid_zones:
            raise ValueError(f"Invalid zones: {invalid_zones}. Available: {self.available_zones}")
        
        # Check years
        invalid_years = [y for y in years if y not in self.available_years]
        if invalid_years:
            raise ValueError(f"Invalid years: {invalid_years}. Available: {self.available_years}")
        
        # Check features
        invalid_features = [f for f in features if f not in self.available_features]
        if invalid_features:
            raise ValueError(f"Invalid features: {invalid_features}. Available: {self.available_features}")
        
        # Check target
        if target not in self.available_targets:
            raise ValueError(f"Invalid target: {target}. Available: {self.available_targets}")
        
        # Check zone-year combinations
        for zone in zones:
            zone_data = self.df[self.df['zone'] == zone]
            zone_years = [y for y in zone_data['year'].unique() if not pd.isna(y)]
            invalid_zone_years = [y for y in years if y not in zone_years]
            if invalid_zone_years:
                warnings.warn(f"Zone '{zone}' does not have data for years: {invalid_zone_years}")
        
        return True
    
    def get_feature_data(self, zones: List[str], years: List[float], feature: str) -> np.ndarray:
        """
        Extract feature data for specified zones and years.
        
        Args:
            zones: List of zone names
            years: List of years
            feature: Feature name
            
        Returns:
            NumPy array with shape (total_points, timesteps)
            - For temporal features: timesteps = 21
            - For static features: timesteps = 1
        """
        all_data = []
        
        for zone in zones:
            if feature in self.static_features:
                # Static features: get one value per point (no year dependency)
                # But repeat for each year to match the temporal data structure
                zone_feature_data = self.df[
                    (self.df['zone'] == zone) & 
                    (self.df['feature'] == feature)
                ]
                # Only use static rows where year is None/NaN
                zone_feature_data = zone_feature_data[pd.isna(zone_feature_data['year'])]
                if len(zone_feature_data) == 0:
                    print(f"Warning: No static data found for zone={zone}, feature={feature}")
                    continue
                # Get static values for each point
                zone_static_values = {}
                for point_id in sorted(zone_feature_data['point_id'].unique()):
                    point_values = zone_feature_data[zone_feature_data['point_id'] == point_id]['value'].values
                    if len(point_values) >= 1:
                        zone_static_values[point_id] = point_values[0]
                    else:
                        zone_static_values[point_id] = np.nan
                # Repeat static values for each year in years (only years explicitly requested)
                for year in years:
                    # Check if this zone has data for this year (temporal rows)
                    year_check = self.df[
                        (self.df['zone'] == zone) & 
                        (self.df['year'] == year)
                    ]
                    if len(year_check) == 0:
                        print(f"Warning: Zone {zone} has no data for year {year}")
                        continue
                    # Get points that have data in this year
                    year_points = sorted(year_check['point_id'].unique())
                    for point_id in year_points:
                        if point_id in zone_static_values:
                            all_data.append([zone_static_values[point_id]])
                        else:
                            all_data.append([np.nan])
                    print(f"Extracted {len(year_points)} points for zone={zone}, year={year}, feature={feature} (static)")
            else:
                # Temporal features: get 21 timesteps per point for each year (only years explicitly requested)
                for year in years:
                    zone_feature_data = self.df[
                        (self.df['zone'] == zone) & 
                        (self.df['feature'] == feature) & 
                        (self.df['year'] == year)
                    ]
                    if len(zone_feature_data) == 0:
                        print(f"Warning: No data found for zone={zone}, year={year}, feature={feature}")
                        continue
                    # Group by point_id and extract values
                    point_data = []
                    for point_id in sorted(zone_feature_data['point_id'].unique()):
                        point_values = zone_feature_data[zone_feature_data['point_id'] == point_id]['value'].values
                        # Temporal feature should have 21 values
                        if len(point_values) == 21:
                            point_data.append(point_values)
                        else:
                            print(f"Warning: Temporal feature {feature} has {len(point_values)} values for point {point_id} (expected 21)")
                            # Pad or truncate to 21 values
                            if len(point_values) < 21:
                                padded_values = np.pad(point_values, (0, 21 - len(point_values)), constant_values=np.nan)
                                point_data.append(padded_values)
                            else:
                                point_data.append(point_values[:21])
                    if point_data:
                        all_data.extend(point_data)
                        print(f"Extracted {len(point_data)} points for zone={zone}, year={year}, feature={feature}")
        
        if not all_data:
            raise ValueError(f"No data found for feature {feature} with specified zones and years")
        
        feature_array = np.vstack(all_data)
        print(f"Feature '{feature}' final shape: {feature_array.shape}")
        return feature_array
    
    def get_target_data(self, zones: List[str], years: List[float], target: str) -> np.ndarray:
        """
        Extract target data for specified zones and years.
        
        Args:
            zones: List of zone names
            years: List of years
            target: Target variable name (ftype_mix or ftype_nonmix)
            
        Returns:
            NumPy array with shape (total_points, 1)
        """
        all_targets = []
        
        for zone in zones:
            # Get zone target data (target is consistent across all years for each point)
            zone_data = self.df[self.df['zone'] == zone]
            if len(zone_data) == 0:
                print(f"Warning: No data found for zone={zone}")
                continue
            # Get unique points and their target values
            point_targets = zone_data.groupby('point_id')[target].first()
            for year in years:
                # Only use explicitly requested years (not None/NaN)
                if pd.isna(year):
                    continue
                # Check if this zone has data for this year
                year_check = self.df[
                    (self.df['zone'] == zone) & 
                    (self.df['year'] == year)
                ]
                if len(year_check) == 0:
                    print(f"Warning: Zone {zone} has no data for year {year}")
                    continue
                # Get points that have data in this year (to match feature extraction)
                year_points = sorted(year_check['point_id'].unique())
                # Extract targets for these points
                year_targets = []
                for point_id in year_points:
                    if point_id in point_targets.index:
                        year_targets.append(point_targets[point_id])
                    else:
                        year_targets.append('NONE')  # Default value for missing targets
                all_targets.extend(year_targets)
                print(f"Extracted {len(year_targets)} target values for zone={zone}, year={year}")
        
        if not all_targets:
            raise ValueError(f"No target data found for {target} with specified zones and years")
        
        target_array = np.array(all_targets).reshape(-1, 1)
        print(f"Target '{target}' final shape: {target_array.shape}")
        return target_array
    
    def create_dataset(self, 
                      zones: List[str], 
                      years: List[float], 
                      features: List[str], 
                      target: str,
                      return_point_info: bool = False) -> Dict[str, np.ndarray]:
        """
        Create a complete dataset for training.
        
        Args:
            zones: List of zone names to include
            years: List of years to include
            features: List of features to extract
            target: Target variable (ftype_mix or ftype_nonmix)
            return_point_info: If True, return point metadata
            
        Returns:
            Dictionary containing:
            - Feature arrays (one per feature)
            - Target array
            - Optionally point information
        """
        print(f"\n=== CREATING DATASET ===")
        print(f"Zones: {zones}")
        print(f"Years: {years}")
        print(f"Features: {features}")
        print(f"Target: {target}")
        
        # Validate inputs
        self.validate_inputs(zones, years, features, target)
        
        # Extract feature data
        dataset = {}
        for feature in features:
            print(f"\nExtracting feature: {feature}")
            dataset[f"{feature}_feature"] = self.get_feature_data(zones, years, feature)
        
        # Extract target data
        print(f"\nExtracting target: {target}")
        dataset['target'] = self.get_target_data(zones, years, target)
        
        # Verify shapes are consistent
        target_samples = dataset['target'].shape[0]
        for feature in features:
            feature_samples = dataset[f"{feature}_feature"].shape[0]
            if feature_samples != target_samples:
                raise ValueError(f"Shape mismatch: feature '{feature}' has {feature_samples} samples, "
                               f"but target has {target_samples} samples")
        
        print(f"\n=== DATASET CREATED SUCCESSFULLY ===")
        print(f"Total samples: {target_samples}")
        for feature in features:
            shape = dataset[f"{feature}_feature"].shape
            print(f"  {feature}_feature shape: {shape}")
        print(f"  target shape: {dataset['target'].shape}")
        
        # Add point information if requested
        if return_point_info:
            dataset['point_info'] = self._get_point_info(zones, years)
        
        return dataset
    
    def _get_point_info(self, zones: List[str], years: List[float]) -> List[Dict]:
        """
        Get point metadata for the dataset.
        
        Args:
            zones: List of zone names
            years: List of years
            
        Returns:
            List of dictionaries with point information
        """
        point_info = []
        
        for zone in zones:
            for year in years:
                zone_year_data = self.df[
                    (self.df['zone'] == zone) & 
                    (self.df['year'] == year)
                ]
                
                if len(zone_year_data) == 0:
                    continue
                
                for point_id in sorted(zone_year_data['point_id'].unique()):
                    point_data = zone_year_data[zone_year_data['point_id'] == point_id].iloc[0]
                    point_info.append({
                        'point_id': point_id,
                        'zone': zone,
                        'year': year,
                        'utm_x': point_data['utm_x'],
                        'utm_y': point_data['utm_y']
                    })
        
        return point_info
    
    def get_dataset_summary(self) -> Dict:
        """
        Get a summary of the dataset.
        
        Returns:
            Dictionary with dataset statistics
        """
        summary = {
            'total_records': len(self.df),
            'unique_points': self.df['point_id'].nunique(),
            'zones': self.available_zones,
            'years': self.available_years,
            'features': self.available_features,
            'static_features': self.static_features,
            'temporal_features': self.temporal_features,
            'targets': self.available_targets
        }
        
        # Zone-year combinations
        zone_year_combinations = {}
        for zone in self.available_zones:
            zone_data = self.df[self.df['zone'] == zone]
            zone_years = sorted([y for y in zone_data['year'].unique() if not pd.isna(y)])
            zone_points = zone_data['point_id'].nunique()
            zone_year_combinations[zone] = {
                'years': zone_years,
                'unique_points': zone_points
            }
        
        summary['zone_year_combinations'] = zone_year_combinations
        
        return summary

def example_usage():
    """
    Example of how to use the ForestTypeDatasetBuilder.
    """
    # Initialize builder
    builder = ForestTypeDatasetBuilder()
    
    # Example 1: Create dataset as specified in the requirements
    zones = ['mfl_cmf_2024', 'mfl_cmf_area_project_2025_chr_zone_1']
    years = [2020.0, 2021.0, 2025.0]
    features = ['vv', 'fapar', 'slope']
    target = 'ftype_mix'
    
    dataset = builder.create_dataset(
        zones=zones,
        years=years,
        features=features,
        target=target,
        return_point_info=True
    )
    
    print("\n=== EXAMPLE DATASET ===")
    for key, value in dataset.items():
        if isinstance(value, np.ndarray):
            print(f"{key}: {value.shape}")
        else:
            print(f"{key}: {len(value)} records")
    
    return dataset

if __name__ == "__main__":
    # Run example
    dataset = example_usage()
