"""
Data Preprocessing Module for Forest Type Classification
Handles feature engineering, normalization, and time-series preparation
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder, RobustScaler
from sklearn.impute import SimpleImputer
from typing import Dict, List, Tuple, Optional, Union
import pickle
import logging

class ForestTypeDataPreprocessor:
    """
    Preprocess extracted features for machine learning models
    """
    
    def __init__(self, 
                 time_series_features: List[str] = None,
                 static_features: List[str] = None,
                 years_range: Tuple[int, int] = (2019, 2024),
                 imputation_strategy: str = 'median',
                 scaling_method: str = 'standard'):
        """
        Initialize the data preprocessor
        
        Args:
            time_series_features: List of time-series feature names
            static_features: List of static feature names
            years_range: Tuple of (start_year, end_year)
            imputation_strategy: Strategy for handling missing values ('median', 'mean', 'forward_fill')
            scaling_method: Method for feature scaling ('standard', 'robust', 'minmax')
        """
        self.time_series_features = time_series_features or [
            'lai', 'fapar', 'fcover', 'gvmi', 'ndvi', 'slavi', 'vh', 'vv'
        ]
        
        self.static_features = static_features or [
            'elevation', 'slope'
        ]
        
        self.years_range = years_range
        self.imputation_strategy = imputation_strategy
        self.scaling_method = scaling_method
        
        # Initialize preprocessing components
        self.static_scaler = None
        self.ts_scalers = {}
        self.label_encoders = {}
        self.imputers = {}
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
    def _create_scaler(self):
        """Create scaler based on scaling method"""
        if self.scaling_method == 'standard':
            return StandardScaler()
        elif self.scaling_method == 'robust':
            return RobustScaler()
        elif self.scaling_method == 'minmax':
            from sklearn.preprocessing import MinMaxScaler
            return MinMaxScaler()
        else:
            raise ValueError(f"Unknown scaling method: {self.scaling_method}")
            
    def _interpolate_time_series(self, ts_data: np.ndarray) -> np.ndarray:
        """Interpolate missing values in time series data"""
        if len(ts_data) != 21:
            return np.full(21, np.nan)
            
        # Create a copy to avoid modifying original
        interpolated = ts_data.copy()
        
        # Find valid (non-NaN) indices
        valid_mask = ~np.isnan(interpolated)
        
        if not valid_mask.any():
            # All values are NaN - cannot interpolate
            return interpolated
            
        if valid_mask.all():
            # No missing values
            return interpolated
            
        # Perform linear interpolation
        valid_indices = np.where(valid_mask)[0]
        valid_values = interpolated[valid_indices]
        
        # Interpolate missing values
        all_indices = np.arange(len(interpolated))
        interpolated = np.interp(all_indices, valid_indices, valid_values)
        
        return interpolated.astype(np.float32)
        
    def _extract_time_series_matrix(self, features_df: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Extract time-series data into 3D matrices"""
        ts_matrices = {}
        
        for feature_name in self.time_series_features:
            feature_data = []
            
            # Collect data from all years for this feature
            for year in range(self.years_range[0], self.years_range[1] + 1):
                col_name = f"{feature_name}_{year}"
                if col_name in features_df.columns:
                    year_data = features_df[col_name].apply(
                        lambda x: self._interpolate_time_series(x) if isinstance(x, np.ndarray) else np.full(21, np.nan)
                    ).tolist()
                    feature_data.append(np.stack(year_data))
                else:
                    # Fill missing year with NaN
                    n_samples = len(features_df)
                    feature_data.append(np.full((n_samples, 21), np.nan))
            
            if feature_data:
                # Stack years: shape (n_samples, n_years, 21_timesteps)
                ts_matrices[feature_name] = np.stack(feature_data, axis=1)
                
        return ts_matrices
        
    def _create_aggregated_features(self, ts_matrices: Dict[str, np.ndarray]) -> pd.DataFrame:
        """Create aggregated features from time series data"""
        aggregated_features = []
        
        for feature_name, ts_data in ts_matrices.items():
            n_samples, n_years, n_timesteps = ts_data.shape
            
            for sample_idx in range(n_samples):
                sample_data = ts_data[sample_idx]  # (n_years, n_timesteps)
                
                # Aggregate across all years and timesteps
                sample_features = {
                    f'{feature_name}_mean': np.nanmean(sample_data),
                    f'{feature_name}_std': np.nanstd(sample_data),
                    f'{feature_name}_min': np.nanmin(sample_data),
                    f'{feature_name}_max': np.nanmax(sample_data),
                    f'{feature_name}_median': np.nanmedian(sample_data),
                    f'{feature_name}_q25': np.nanpercentile(sample_data, 25),
                    f'{feature_name}_q75': np.nanpercentile(sample_data, 75),
                }
                
                # Seasonal features (assuming weeks 1-21 represent seasonal cycle)
                if not np.all(np.isnan(sample_data)):
                    # Average across years for each week
                    weekly_avg = np.nanmean(sample_data, axis=0)
                    
                    # Find peak and valley weeks
                    if not np.all(np.isnan(weekly_avg)):
                        sample_features[f'{feature_name}_peak_week'] = np.nanargmax(weekly_avg)
                        sample_features[f'{feature_name}_valley_week'] = np.nanargmin(weekly_avg)
                        sample_features[f'{feature_name}_seasonal_range'] = np.nanmax(weekly_avg) - np.nanmin(weekly_avg)
                    else:
                        sample_features[f'{feature_name}_peak_week'] = np.nan
                        sample_features[f'{feature_name}_valley_week'] = np.nan
                        sample_features[f'{feature_name}_seasonal_range'] = np.nan
                else:
                    sample_features[f'{feature_name}_peak_week'] = np.nan
                    sample_features[f'{feature_name}_valley_week'] = np.nan
                    sample_features[f'{feature_name}_seasonal_range'] = np.nan
                
                aggregated_features.append(sample_features)
                
        return pd.DataFrame(aggregated_features)
        
    def _prepare_sequential_data(self, ts_matrices: Dict[str, np.ndarray]) -> np.ndarray:
        """Prepare sequential data for LSTM/CNN models"""
        # Concatenate all time series features
        all_ts_data = []
        
        for feature_name in self.time_series_features:
            if feature_name in ts_matrices:
                ts_data = ts_matrices[feature_name]
                n_samples, n_years, n_timesteps = ts_data.shape
                
                # Reshape to (n_samples, n_years * n_timesteps)
                reshaped = ts_data.reshape(n_samples, -1)
                all_ts_data.append(reshaped)
        
        if all_ts_data:
            # Concatenate along feature dimension
            sequential_data = np.concatenate(all_ts_data, axis=1)
            return sequential_data
        else:
            return np.array([])
            
    def fit_transform_train(self, features_df: pd.DataFrame, target_col: str) -> Dict:
        """Fit preprocessors and transform training data"""
        self.logger.info("Fitting preprocessors on training data")
        
        # Extract time series matrices
        ts_matrices = self._extract_time_series_matrix(features_df)
        
        # Create aggregated features
        aggregated_df = self._create_aggregated_features(ts_matrices)
        
        # Combine with static features
        static_cols = [col for col in self.static_features if col in features_df.columns]
        static_df = features_df[static_cols].copy()
        
        # Combine all tabular features
        tabular_features = pd.concat([aggregated_df, static_df], axis=1)
        
        # Handle missing values
        self.imputers['tabular'] = SimpleImputer(strategy=self.imputation_strategy)
        tabular_features_imputed = pd.DataFrame(
            self.imputers['tabular'].fit_transform(tabular_features),
            columns=tabular_features.columns,
            index=tabular_features.index
        )
        
        # Scale tabular features
        self.static_scaler = self._create_scaler()
        tabular_features_scaled = pd.DataFrame(
            self.static_scaler.fit_transform(tabular_features_imputed),
            columns=tabular_features_imputed.columns,
            index=tabular_features_imputed.index
        )
        
        # Prepare sequential data for deep learning models
        sequential_data = self._prepare_sequential_data(ts_matrices)
        
        # Handle missing values in sequential data
        if sequential_data.size > 0:
            # Replace NaN with 0 for neural networks (after normalization)
            self.imputers['sequential'] = SimpleImputer(strategy='constant', fill_value=0)
            sequential_data_imputed = self.imputers['sequential'].fit_transform(sequential_data)
            
            # Scale sequential data
            self.ts_scalers['sequential'] = self._create_scaler()
            sequential_data_scaled = self.ts_scalers['sequential'].fit_transform(sequential_data_imputed)
        else:
            sequential_data_scaled = np.array([])
        
        # Encode target labels
        self.label_encoders[target_col] = LabelEncoder()
        y_encoded = self.label_encoders[target_col].fit_transform(features_df[target_col])
        
        # Store feature names
        self.tabular_feature_names = list(tabular_features_scaled.columns)
        self.n_sequential_features = sequential_data_scaled.shape[1] if sequential_data_scaled.size > 0 else 0
        
        result = {
            'X_tabular': tabular_features_scaled.values,
            'X_sequential': sequential_data_scaled,
            'y': y_encoded,
            'feature_names_tabular': self.tabular_feature_names,
            'n_sequential_features': self.n_sequential_features,
            'target_classes': list(self.label_encoders[target_col].classes_),
            'target_col': target_col
        }
        
        self.logger.info(f"Training data prepared:")
        self.logger.info(f"  Tabular features: {result['X_tabular'].shape}")
        self.logger.info(f"  Sequential features: {result['X_sequential'].shape}")
        self.logger.info(f"  Target classes: {result['target_classes']}")
        
        return result
        
    def transform_test(self, features_df: pd.DataFrame, target_col: str) -> Dict:
        """Transform test data using fitted preprocessors"""
        self.logger.info("Transforming test data")
        
        if not hasattr(self, 'static_scaler') or self.static_scaler is None:
            raise ValueError("Preprocessors not fitted. Call fit_transform_train first.")
        
        # Extract time series matrices
        ts_matrices = self._extract_time_series_matrix(features_df)
        
        # Create aggregated features
        aggregated_df = self._create_aggregated_features(ts_matrices)
        
        # Combine with static features
        static_cols = [col for col in self.static_features if col in features_df.columns]
        static_df = features_df[static_cols].copy()
        
        # Combine all tabular features
        tabular_features = pd.concat([aggregated_df, static_df], axis=1)
        
        # Ensure same columns as training
        for col in self.tabular_feature_names:
            if col not in tabular_features.columns:
                tabular_features[col] = np.nan
        tabular_features = tabular_features[self.tabular_feature_names]
        
        # Handle missing values
        tabular_features_imputed = pd.DataFrame(
            self.imputers['tabular'].transform(tabular_features),
            columns=tabular_features.columns,
            index=tabular_features.index
        )
        
        # Scale tabular features
        tabular_features_scaled = pd.DataFrame(
            self.static_scaler.transform(tabular_features_imputed),
            columns=tabular_features_imputed.columns,
            index=tabular_features_imputed.index
        )
        
        # Prepare sequential data
        sequential_data = self._prepare_sequential_data(ts_matrices)
        
        if sequential_data.size > 0 and 'sequential' in self.imputers:
            # Handle missing values in sequential data
            sequential_data_imputed = self.imputers['sequential'].transform(sequential_data)
            
            # Scale sequential data
            sequential_data_scaled = self.ts_scalers['sequential'].transform(sequential_data_imputed)
        else:
            sequential_data_scaled = np.array([])
        
        # Encode target labels
        y_encoded = self.label_encoders[target_col].transform(features_df[target_col])
        
        result = {
            'X_tabular': tabular_features_scaled.values,
            'X_sequential': sequential_data_scaled,
            'y': y_encoded,
            'feature_names_tabular': list(tabular_features_scaled.columns),
            'n_sequential_features': sequential_data_scaled.shape[1] if sequential_data_scaled.size > 0 else 0,
            'target_classes': list(self.label_encoders[target_col].classes_),
            'target_col': target_col
        }
        
        return result
        
    def save_preprocessors(self, output_path: str):
        """Save fitted preprocessors"""
        preprocessor_data = {
            'static_scaler': self.static_scaler,
            'ts_scalers': self.ts_scalers,
            'label_encoders': self.label_encoders,
            'imputers': self.imputers,
            'tabular_feature_names': getattr(self, 'tabular_feature_names', []),
            'n_sequential_features': getattr(self, 'n_sequential_features', 0),
            'config': {
                'time_series_features': self.time_series_features,
                'static_features': self.static_features,
                'years_range': self.years_range,
                'imputation_strategy': self.imputation_strategy,
                'scaling_method': self.scaling_method
            }
        }
        
        with open(output_path, 'wb') as f:
            pickle.dump(preprocessor_data, f)
            
        self.logger.info(f"Preprocessors saved to {output_path}")
        
    def load_preprocessors(self, input_path: str):
        """Load fitted preprocessors"""
        with open(input_path, 'rb') as f:
            preprocessor_data = pickle.load(f)
            
        self.static_scaler = preprocessor_data['static_scaler']
        self.ts_scalers = preprocessor_data['ts_scalers']
        self.label_encoders = preprocessor_data['label_encoders']
        self.imputers = preprocessor_data['imputers']
        self.tabular_feature_names = preprocessor_data['tabular_feature_names']
        self.n_sequential_features = preprocessor_data['n_sequential_features']
        
        # Update config
        config = preprocessor_data['config']
        self.time_series_features = config['time_series_features']
        self.static_features = config['static_features']
        self.years_range = config['years_range']
        self.imputation_strategy = config['imputation_strategy']
        self.scaling_method = config['scaling_method']
        
        self.logger.info(f"Preprocessors loaded from {input_path}")
