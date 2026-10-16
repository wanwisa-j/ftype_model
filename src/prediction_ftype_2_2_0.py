#!/usr/bin/env python3
"""
Forest Type Prediction Script for Model v2.2.0
==============================================

This script processes raster data in v1.2.0 format and makes predictions using
the trained forest type classification model v2.2.0.

Key differences from v2.0.0:
- Uses raster data format with timesteps (timesteps, height, width)
- No pre-existing mask files - creates masks from input data
- Processes all zones and years with majority voting
- Handles separate directories for each feature type
- Integrates with comprehensive data preparation functions

Usage:
    python prediction_ftype_2_2_0.py --zone zone_1 --years 2019,2020,2021 --output_dir /path/to/output

Author: Generated for model v2.2.0
Date: 2025-09-10
"""

# ==============================================================================
# DATA PREPARATION FOR PREDICTION - DOCUMENTATION & UTILITY FUNCTIONS
# ==============================================================================

"""
This section provides comprehensive functions and documentation for preparing
data for forest type classification prediction using the trained hybrid CNN model.

The model expects:
1. Time series features: 8 original channels + 1 VH/VV ratio = 9 total channels
2. Static features: slope and elevation (2 channels)
3. Proper scaling using training set statistics

FEATURES OVERVIEW:
================

Time Series Features (21 timesteps, 9 channels):
- Channel 0: VV (SAR backscatter, vertical-vertical polarization)
- Channel 1: VH (SAR backscatter, vertical-horizontal polarization)  
- Channel 2: GVMI (Green Vegetation Moisture Index)
- Channel 3: NDVI (Normalized Difference Vegetation Index)
- Channel 4: SLAVI (Specific Leaf Area Vegetation Index)
- Channel 5: LAI (Leaf Area Index)
- Channel 6: FAPAR (Fraction of Absorbed Photosynthetically Active Radiation)
- Channel 7: FCOVER (Fraction of vegetation cover)
- Channel 8: VH/VV Ratio (computed ratio, added during preprocessing)
- Channel 9: VV/VH Ratio (optional, not used in current model)

Static Features (2 channels):
- Channel 0: Slope (degrees)
- Channel 1: Elevation (meters)

SCALING:
========
- Time series: Z-score normalization per channel using training statistics
- Static features: Z-score normalization using training statistics
- All scaling parameters saved during training and must be reused for prediction
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras import models
import rasterio
from rasterio.transform import from_bounds
from rasterio.warp import reproject, Resampling
import os
import argparse
from scipy.stats import mode
import sys
sys.path.append('/fs1/tor_project/model_foresttype/src')
from dataset_builder import ForestTypeDatasetBuilder


# ==============================================================================
# PREDICTION DATA PREPARATION FUNCTIONS
# ==============================================================================

def load_scaling_parameters(config_path=None):
    """
    Load scaling parameters computed from training data.
    
    Args:
        config_path (str): Path to the scaling parameters file. 
                          If None, uses default path for v2.2.0 model.
    
    Returns:
        dict: Dictionary containing scaling parameters
    """
    if config_path is None:
        config_path = '/fs1/tor_project/model_foresttype/model/mfl_v2_2_0/config/scaling_params_from_training_hybrid_lstm_cnn_model-20250910.npz'
    
    try:
        with np.load(config_path) as data:
            scaling_params = {
                'static_mean': data['static_mean'],
                'static_std': data['static_std'],
                'timeseries_mean': data['timeseries_mean'],
                'timeseries_std': data['timeseries_std'],
            }
        print(f"✅ Loaded scaling parameters from: {config_path}")
        return scaling_params
    except FileNotFoundError:
        raise FileNotFoundError(f"Scaling parameters not found at: {config_path}")

def extract_prediction_features(zones, years, features=None, target='ftype_nonmix'):
    """
    Extract features for prediction using ForestTypeDatasetBuilder.
    
    Args:
        zones (list): List of zone names
        years (list): List of years to extract
        features (list): List of features to extract. If None, uses all available features.
        target (str): Target variable (used for data structure consistency)
    
    Returns:
        dict: Dataset dictionary with feature arrays
    """
    if features is None:
        features = ['vv', 'vh', 'gvmi', 'ndvi', 'slavi', 'lai', 'fapar', 'fcover', 'slope', 'elevation']
    
    print(f"🔄 Extracting features for prediction...")
    print(f"   Zones: {zones}")
    print(f"   Years: {years}")
    print(f"   Features: {features}")
    
    builder = ForestTypeDatasetBuilder()
    dataset = builder.create_dataset(
        zones=zones,
        years=years,
        features=features,
        target=target,
        return_point_info=True
    )
    
    print(f"✅ Extracted {dataset['target'].shape[0]} samples")
    return dataset

def prepare_timeseries_data(dataset):
    """
    Prepare time series features with VH/VV ratio computation.
    
    Args:
        dataset (dict): Dataset from ForestTypeDatasetBuilder
    
    Returns:
        np.ndarray: Time series array with shape (samples, timesteps, channels)
                   where channels = 8 original + 1 VH/VV ratio = 9 total
    """
    timeseries_features = ['vv', 'vh', 'gvmi', 'ndvi', 'slavi', 'lai', 'fapar', 'fcover']
    
    print("🔄 Preparing time series features...")
    
    # Stack original time series features
    X_timeseries = np.dstack([dataset[f'{feat}_feature'] for feat in timeseries_features])
    print(f"   Original timeseries shape: {X_timeseries.shape}")
    
    # Compute VH/VV ratio (Channel 8)
    vv = X_timeseries[..., 0]  # VV is channel 0
    vh = X_timeseries[..., 1]  # VH is channel 1
    
    # Avoid division by zero
    denom = np.where(vv == 0, 1e-6, vv)
    vh_vv_ratio = vh / denom
    
    # Add VH/VV ratio as 9th channel
    X_timeseries = np.concatenate([X_timeseries, vh_vv_ratio[..., np.newaxis]], axis=-1)
    
    print(f"   Final timeseries shape with VH/VV ratio: {X_timeseries.shape}")
    print(f"   ✅ Added VH/VV ratio as channel 8")

    # Compute VV/VH ratio (Channel 9)
    denom = np.where(vh == 0, 1e-6, vh)
    vv_vh_ratio = vv / denom

    X_timeseries = np.concatenate([X_timeseries, vv_vh_ratio[..., np.newaxis]], axis=-1)

    print(f"   ✅ Added VV/VH ratio as channel 9")

    return X_timeseries

def prepare_static_data(dataset):
    """
    Prepare static features (slope and elevation).
    
    Args:
        dataset (dict): Dataset from ForestTypeDatasetBuilder
    
    Returns:
        np.ndarray: Static features array with shape (samples, 2)
    """
    static_features = ['slope', 'elevation']
    
    print("🔄 Preparing static features...")
    X_static = np.hstack([dataset[f'{feat}_feature'] for feat in static_features])
    print(f"   Static features shape: {X_static.shape}")
    print(f"   ✅ Features: slope (channel 0), elevation (channel 1)")
    
    return X_static

def apply_scaling(X_timeseries, X_static, scaling_params):
    """
    Apply z-score scaling using training set statistics.
    
    Args:
        X_timeseries (np.ndarray): Time series features (samples, timesteps, channels)
        X_static (np.ndarray): Static features (samples, channels)
        scaling_params (dict): Scaling parameters from training
    
    Returns:
        tuple: (X_timeseries_scaled, X_static_scaled)
    """
    print("🔄 Applying scaling transformations...")
    
    # Scale time series features
    timeseries_mean = scaling_params['timeseries_mean']
    timeseries_std = scaling_params['timeseries_std']
    X_timeseries_scaled = (X_timeseries - timeseries_mean) / timeseries_std
    
    # Scale static features
    static_mean = scaling_params['static_mean']
    static_std = scaling_params['static_std']
    X_static_scaled = (X_static - static_mean) / static_std
    
    print(f"   ✅ Applied timeseries scaling: mean={timeseries_mean.shape}, std={timeseries_std.shape}")
    print(f"   ✅ Applied static scaling: mean={static_mean.shape}, std={static_std.shape}")
    
    return X_timeseries_scaled, X_static_scaled

def filter_forest_classes(dataset, X_timeseries, X_static, classes=['DDF', 'DEF', 'MDF']):
    """
    Filter data to include only specified forest classes.
    
    Args:
        dataset (dict): Dataset with target labels
        X_timeseries (np.ndarray): Time series features
        X_static (np.ndarray): Static features
        classes (list): List of classes to keep
    
    Returns:
        tuple: (X_timeseries_filtered, X_static_filtered, targets_filtered, mask)
    """
    print(f"🔄 Filtering for classes: {classes}")
    
    targets = dataset['target'].ravel()
    mask = np.isin(targets, classes)
    
    X_timeseries_filtered = X_timeseries[mask]
    X_static_filtered = X_static[mask]
    targets_filtered = targets[mask]
    
    print(f"   Original samples: {len(targets)}")
    print(f"   Filtered samples: {mask.sum()}")
    
    # Show class distribution
    unique, counts = np.unique(targets_filtered, return_counts=True)
    print(f"   Class distribution: {dict(zip(unique, counts))}")
    
    return X_timeseries_filtered, X_static_filtered, targets_filtered, mask

def prepare_prediction_data(zones=None, years=None, features=None, classes=['DDF', 'DEF', 'MDF'], 
                           scaling_params=None, config_path=None, 
                           timeseries_data=None, static_data=None):
    """
    Complete data preparation pipeline for forest type prediction.
    
    This function handles the entire preprocessing pipeline:
    1. Feature extraction using ForestTypeDatasetBuilder (if zones/years provided)
    2. Time series preparation with VH/VV ratio computation  
    3. Static features preparation
    4. Class filtering (if specified)
    5. Scaling using training statistics
    
    Args:
        zones (list): Zone names to extract data from (if using ForestTypeDatasetBuilder)
        years (list): Years to include in the analysis (if using ForestTypeDatasetBuilder)
        features (list): Feature names to extract. If None, uses all available.
        classes (list): Forest classes to filter for. If None, keeps all classes.
        scaling_params (dict): Pre-loaded scaling parameters. If None, loads from config_path.
        config_path (str): Path to scaling parameters file. Uses default if None.
        timeseries_data (np.ndarray): Raw timeseries data (alternative to zones/years)
        static_data (np.ndarray): Raw static data (alternative to zones/years)
    
    Returns:
        dict: Dictionary containing processed features and metadata
    
    Example:
        # Prepare data for all zones and recent years
        zones = ['mfl_cmf_2024', 'mfl_cmf_area_project_2025_chr_zone_1']
        years = [2020, 2021, 2022, 2023, 2024]
        
        prediction_data = prepare_prediction_data(zones, years)
        
        # Use with model
        X_timeseries = prediction_data['X_timeseries']
        X_static = prediction_data['X_static']
        predictions = model.predict([X_timeseries, X_static])
    """
    print("="*80)
    print("🚀 FOREST TYPE PREDICTION DATA PREPARATION")
    print("="*80)
    
    # Step 1: Load scaling parameters
    if scaling_params is None:
        scaling_params = load_scaling_parameters(config_path)
    
    # Step 2: Prepare features
    if timeseries_data is not None and static_data is not None:
        # Use provided raw data
        print("🔄 Using provided raw data...")
        
        # Validate input shapes
        if timeseries_data.ndim != 3:
            raise ValueError(f"timeseries_data must be 3D array, got shape {timeseries_data.shape}")
        if static_data.ndim != 2:
            raise ValueError(f"static_data must be 2D array, got shape {static_data.shape}")
        if timeseries_data.shape[0] != static_data.shape[0]:
            raise ValueError("Number of samples must match between timeseries and static data")
        
        expected_timeseries_features = 8  # vv, vh, gvmi, ndvi, slavi, lai, fapar, fcover
        expected_static_features = 2      # slope, elevation
        
        if timeseries_data.shape[2] != expected_timeseries_features:
            raise ValueError(f"Expected {expected_timeseries_features} timeseries features, got {timeseries_data.shape[2]}")
        if static_data.shape[1] != expected_static_features:
            raise ValueError(f"Expected {expected_static_features} static features, got {static_data.shape[1]}")
        
        print(f"Processing {timeseries_data.shape[0]} samples...")
        print(f"Timeseries shape: {timeseries_data.shape}")
        print(f"Static shape: {static_data.shape}")
        
        # Extract VV and VH channels (first two channels)
        vv = timeseries_data[:, :, 0]  # VV channel
        vh = timeseries_data[:, :, 1]  # VH channel
        
        # Calculate VH/VV ratio with division by zero protection
        denom = np.where(vv == 0, 1e-6, vv)
        vh_vv_ratio = vh / denom
        
        # Calculate VV/VH ratio with division by zero protection
        denom_vh = np.where(vh == 0, 1e-6, vh)
        vv_vh_ratio = vv / denom_vh
        
        # Append both ratios as new channels to match training format (10 channels total)
        X_timeseries = np.concatenate([
            timeseries_data, 
            vh_vv_ratio[:, :, np.newaxis],
            vv_vh_ratio[:, :, np.newaxis]
        ], axis=-1)
        
        print(f"Added VH/VV and VV/VH ratio channels. New timeseries shape: {X_timeseries.shape}")
        
        X_static = static_data
        dataset = None
        
    else:
        # Use ForestTypeDatasetBuilder
        if zones is None or years is None:
            raise ValueError("Either provide timeseries_data/static_data OR zones/years")
            
        dataset = extract_prediction_features(zones, years, features)
        
        # Step 3: Prepare time series with VH/VV ratio
        X_timeseries = prepare_timeseries_data(dataset)
        
        # Step 4: Prepare static features
        X_static = prepare_static_data(dataset)
    
    # Step 5: Filter classes (optional)
    mask = None
    targets = None
    if classes is not None and dataset is not None:
        X_timeseries, X_static, targets, mask = filter_forest_classes(
            dataset, X_timeseries, X_static, classes
        )
    
    # Step 6: Apply scaling
    X_timeseries_scaled, X_static_scaled = apply_scaling(X_timeseries, X_static, scaling_params)
    
    # Prepare return dictionary
    result = {
        'X_timeseries': X_timeseries_scaled,
        'X_static': X_static_scaled,
        'scaling_params': scaling_params
    }
    
    if dataset is not None:
        result['point_info'] = dataset['point_info']
    if targets is not None:
        result['targets'] = targets
    if mask is not None:
        result['mask'] = mask
    
    print("="*80)
    print("✅ DATA PREPARATION COMPLETE")
    print("="*80)
    print(f"📊 Final shapes:")
    print(f"   Time series: {X_timeseries_scaled.shape} (samples, timesteps, channels)")
    print(f"   Static features: {X_static_scaled.shape} (samples, features)")
    if targets is not None:
        print(f"   Targets: {targets.shape}")
    if dataset is not None:
        print(f"   Point info: {len(result['point_info'])} records")
    
    return result


def load_raster_data(feature_dir, year, feature_name):
    """
    Load raster data for a specific feature and year.
    
    Parameters:
    -----------
    feature_dir : str
        Directory containing the feature data
    year : int
        Year to load data for
    feature_name : str
        Name of the feature (e.g., 'VV', 'VH', 'lai', etc.)
    
    Returns:
    --------
    dict
        Dictionary containing raster data, transform, crs, and metadata
    """
    feature_path = os.path.join(feature_dir, feature_name)
    
    if feature_name.lower() in ['vv', 'vh']:
        filename = f"cube_s1_{feature_name.lower()}_{year}_W4before_to_W16after_21bands.tif"
    elif feature_name.lower() in ['gvmi', 'ndvi', 'slavi']:
        filename = f"cube_s2_{feature_name.lower()}_{year}_W4before_to_W16after_21bands.tif"
    elif feature_name.lower() in ['lai', 'fapar', 'fcover']:
        filename = f"cube_biophys_{feature_name.lower()}_{year}_W4before_to_W16after_21bands.tif"
    elif feature_name.lower() == 'elevation':
        filename = "elevation_data.tif"
    elif feature_name.lower() == 'slope':
        filename = "slope_data.tif"
    else:
        raise ValueError(f"Unknown feature: {feature_name}")
    
    file_path = os.path.join(feature_path, filename)
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Data file not found: {file_path}")
    
    print(f"Loading {feature_name} data from: {file_path}")
    
    with rasterio.open(file_path) as src:
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


def create_union_mask(data_dicts):
    """
    Create a union mask from multiple data dictionaries.
    
    Parameters:
    -----------
    data_dicts : list
        List of data dictionaries from load_raster_data
    
    Returns:
    --------
    numpy.ndarray
        Boolean mask where True indicates no-data pixels
    """
    print("Creating union mask from all features...")
    
    masks = []
    for i, data_dict in enumerate(data_dicts):
        data = data_dict['data']
        if data.ndim == 3:  # Timeseries data
            # Create mask where all timesteps are NaN or invalid
            mask = np.isnan(data).all(axis=0) | (data == -9999).all(axis=0)
        else:  # Static data
            # Create mask for single layer, squeeze if needed
            if data.ndim == 3 and data.shape[0] == 1:
                data = data.squeeze(axis=0)
            mask = np.isnan(data) | (data == -9999) | (data == 0)
        masks.append(mask)
        print(f"Feature {i+1}: {np.sum(mask)} no-data pixels")
    
    # Union of all masks (True where any feature has no-data)
    union_mask = np.logical_or.reduce(masks)
    print(f"Union mask: {np.sum(union_mask)} total no-data pixels")
    print(f"Valid pixels: {np.sum(~union_mask)} pixels")
    
    return union_mask


def reshape_and_stack_features(timeseries_dicts, static_dicts, mask):
    """
    Reshape and stack features for model input.
    
    Parameters:
    -----------
    timeseries_dicts : list
        List of timeseries data dictionaries
    static_dicts : list
        List of static data dictionaries
    mask : numpy.ndarray
        Boolean mask for valid pixels
    
    Returns:
    --------
    tuple
        (timeseries_array, static_array) ready for model input
    """
    print("Reshaping and stacking features...")
    
    # Get dimensions from first timeseries feature
    first_data = timeseries_dicts[0]['data']
    n_timesteps = first_data.shape[0]
    height, width = mask.shape
    n_valid_pixels = np.sum(~mask)
    
    print(f"Data dimensions: {n_timesteps} timesteps, {height}x{width} pixels")
    print(f"Valid pixels: {n_valid_pixels}")
    
    # Stack timeseries features
    timeseries_stack = []
    feature_names = ['vv', 'vh', 'gvmi', 'ndvi', 'slavi', 'lai', 'fapar', 'fcover']
    
    for i, data_dict in enumerate(timeseries_dicts):
        data = data_dict['data']
        # Reshape from (timesteps, height, width) to (n_pixels, timesteps)
        reshaped = data.reshape(n_timesteps, -1).T
        # Extract valid pixels only
        valid_data = reshaped[~mask.ravel()]
        timeseries_stack.append(valid_data)
        print(f"Stacked {feature_names[i]}: shape {valid_data.shape}")
    
    # Combine timeseries features: (n_valid_pixels, n_timesteps, n_features)
    timeseries_array = np.stack(timeseries_stack, axis=-1)
    print(f"Final timeseries array shape: {timeseries_array.shape}")
    
    # Stack static features
    static_stack = []
    static_names = ['elevation', 'slope']
    
    for i, data_dict in enumerate(static_dicts):
        data = data_dict['data']
        # Handle single band static data
        if data.ndim == 3 and data.shape[0] == 1:
            data = data.squeeze(axis=0)
        # Reshape and extract valid pixels
        valid_data = data.ravel()[~mask.ravel()]
        static_stack.append(valid_data)
        print(f"Stacked {static_names[i]}: shape {valid_data.shape}")
    
    # Combine static features: (n_valid_pixels, n_static_features)
    static_array = np.stack(static_stack, axis=-1)
    print(f"Final static array shape: {static_array.shape}")
    
    return timeseries_array, static_array


def predict_forest_type(model, timeseries_data, static_data, scaling_params, batch_size=128):
    """
    Predict forest type classifications using the trained model (legacy function).
    
    Parameters:
    -----------
    model : keras.Model
        Trained forest type classification model
    timeseries_data : numpy.ndarray
        Preprocessed timeseries data
    static_data : numpy.ndarray
        Preprocessed static data  
    scaling_params : dict
        Scaling parameters for data preprocessing
    batch_size : int
        Batch size for prediction
    
    Returns:
    --------
    tuple
        (predicted_classes, prediction_probabilities)
    """
    
    # Prepare data for prediction using the updated function
    prediction_data = prepare_prediction_data(
        timeseries_data=timeseries_data,
        static_data=static_data,
        scaling_params=scaling_params
    )
    
    X_timeseries_scaled = prediction_data['X_timeseries']
    X_static_scaled = prediction_data['X_static']
    
    # Make predictions
    print("Making predictions...")
    probabilities = model.predict([X_timeseries_scaled, X_static_scaled], 
                                 batch_size=batch_size, verbose=1)
    
    # Get predicted class indices
    predicted_indices = np.argmax(probabilities, axis=1)
    
    print(f"Prediction complete. Predicted {len(predicted_indices)} samples")
    
    return predicted_indices, probabilities


def reshape_predictions_to_raster(predictions, probabilities, mask, reference_metadata):
    """
    Reshape predictions back to raster format.
    
    Parameters:
    -----------
    predictions : numpy.ndarray
        Predicted class indices
    probabilities : numpy.ndarray
        Prediction probabilities
    mask : numpy.ndarray
        Boolean mask used for valid pixels
    reference_metadata : dict
        Reference metadata for spatial information
    
    Returns:
    --------
    tuple
        (prediction_raster, probability_raster, metadata) as numpy arrays and metadata dict
    """
    print("Reshaping predictions to raster format...")
    
    height, width = mask.shape
    n_classes = probabilities.shape[1]
    
    # Initialize output arrays with nodata
    pred_raster = np.full((height, width), -999, dtype=np.int16)
    prob_raster = np.full((n_classes, height, width), -999.0, dtype=np.float32)
    
    # Fill valid pixels
    valid_indices = ~mask.ravel()
    pred_raster.ravel()[valid_indices] = predictions
    
    for class_idx in range(n_classes):
        prob_raster[class_idx].ravel()[valid_indices] = probabilities[:, class_idx]
    
    print(f"Reshaped predictions: {pred_raster.shape}")
    print(f"Reshaped probabilities: {prob_raster.shape}")
    
    return pred_raster, prob_raster, reference_metadata


def save_geotiff(data_array, metadata, output_path, dtype='int16', nodata_value=-999):
    """
    Save numpy array as GeoTIFF using rasterio.
    
    Parameters:
    -----------
    data_array : numpy.ndarray
        Data to save
    metadata : dict
        Spatial metadata including transform, crs
    output_path : str
        Output file path
    dtype : str
        Data type for output
    nodata_value : float
        Nodata value
    """
    print(f"Saving GeoTIFF: {output_path}")
    
    # Handle different array dimensions
    if data_array.ndim == 2:
        # Single band
        height, width = data_array.shape
        count = 1
        data_to_write = data_array[np.newaxis, :, :]  # Add band dimension
    else:
        # Multi-band
        count, height, width = data_array.shape
        data_to_write = data_array
    
    with rasterio.open(
        output_path,
        'w',
        driver='GTiff',
        height=height,
        width=width,
        count=count,
        dtype=dtype,
        crs=metadata['crs'],
        transform=metadata['transform'],
        nodata=nodata_value,
        compress='LZW'
    ) as dst:
        dst.write(data_to_write)


def majority_vote_predictions(prediction_files, output_path, reference_metadata):
    """
    Perform majority voting across multiple years of predictions.
    
    Parameters:
    -----------
    prediction_files : list
        List of prediction file paths
    output_path : str
        Output path for majority vote result
    reference_metadata : dict
        Reference metadata for spatial information
    """
    print("Performing majority voting across years...")
    
    predictions = []
    for pred_file in prediction_files:
        if os.path.exists(pred_file):
            with rasterio.open(pred_file) as src:
                pred_data = src.read(1)  # Read first band
                predictions.append(pred_data)
                print(f"Loaded prediction: {pred_file}")
        else:
            print(f"Warning: Prediction file not found: {pred_file}")
    
    if len(predictions) == 0:
        print("No prediction files found for majority voting")
        return
    
    # Stack predictions and compute mode
    prediction_stack = np.stack(predictions, axis=-1)
    
    # Handle nodata values
    valid_mask = prediction_stack != -999
    
    # Initialize output
    height, width = prediction_stack.shape[:2]
    majority_prediction = np.full((height, width), -999, dtype=np.int16)
    
    # Compute majority vote pixel by pixel
    for i in range(height):
        for j in range(width):
            pixel_values = prediction_stack[i, j, valid_mask[i, j]]
            if len(pixel_values) > 0:
                mode_result = mode(pixel_values, axis=0, keepdims=False)
                majority_prediction[i, j] = mode_result.mode
    
    save_geotiff(majority_prediction, reference_metadata, output_path, dtype='int16')
    print(f"Majority vote result saved: {output_path}")


def load_scaling_params(config_dir):
    """
    Load scaling parameters from the model configuration (legacy function).
    
    Parameters:
    -----------
    config_dir : str
        Directory containing scaling parameters
    
    Returns:
    --------
    dict
        Scaling parameters dictionary
    """
    scaling_file = os.path.join(config_dir, 'scaling_params_from_training_hybrid_lstm_cnn_model-20250910.npz')
    
    # Fallback to old filename if new one doesn't exist
    if not os.path.exists(scaling_file):
        scaling_file = os.path.join(config_dir, 'scaling_params_from_training.npz')
    
    if not os.path.exists(scaling_file):
        raise FileNotFoundError(f"Scaling parameters not found: {scaling_file}")
    
    print(f"Loading scaling parameters from: {scaling_file}")
    scaling_data = np.load(scaling_file)
    
    scaling_params = {
        'timeseries_mean': scaling_data['timeseries_mean'],
        'timeseries_std': scaling_data['timeseries_std'],
        'static_mean': scaling_data['static_mean'],
        'static_std': scaling_data['static_std']
    }
    
    print("Loaded scaling parameters:")
    for key, value in scaling_params.items():
        print(f"  {key}: shape {value.shape}")
    
    return scaling_params


def process_zone_year(data_dir, zone, year, model, scaling_params, output_dir):
    """
    Process a single zone and year.
    
    Parameters:
    -----------
    data_dir : str
        Base data directory
    zone : str
        Zone identifier (e.g., 'zone_1')
    year : int
        Year to process
    model : keras.Model
        Trained model
    scaling_params : dict
        Scaling parameters
    output_dir : str
        Output directory
    
    Returns:
    --------
    str
        Path to saved prediction file
    """
    print(f"\nProcessing {zone} - {year}")
    print("=" * 50)
    
    zone_dir = os.path.join(data_dir, zone)
    
    if not os.path.exists(zone_dir):
        print(f"Zone directory not found: {zone_dir}")
        return None
    
    try:
        # Load timeseries features
        timeseries_dicts = []
        feature_names = ['vv', 'vh', 'gvmi', 'ndvi', 'slavi', 'lai', 'fapar', 'fcover']
        
        for feature_name in feature_names:
            data_dict = load_raster_data(zone_dir, year, feature_name)
            timeseries_dicts.append(data_dict)
        
        # Load static features
        static_dicts = []
        static_names = ['elevation', 'slope']
        
        for feature_name in static_names:
            data_dict = load_raster_data(zone_dir, year, feature_name)
            static_dicts.append(data_dict)
        
        # Create union mask
        all_dicts = timeseries_dicts + static_dicts
        union_mask = create_union_mask(all_dicts)
        
        # Reshape and stack features
        timeseries_array, static_array = reshape_and_stack_features(
            timeseries_dicts, static_dicts, union_mask
        )
        
        # Make predictions using the updated prepare_prediction_data function
        prediction_data = prepare_prediction_data(
            timeseries_data=timeseries_array,
            static_data=static_array,
            scaling_params=scaling_params
        )
        
        X_timeseries_scaled = prediction_data['X_timeseries']
        X_static_scaled = prediction_data['X_static']
        
        # Make predictions
        print("Making predictions...")
        probabilities = model.predict([X_timeseries_scaled, X_static_scaled], 
                                     batch_size=128, verbose=1)
        
        # Get predicted class indices
        predictions = np.argmax(probabilities, axis=1)
        
        print(f"Prediction complete. Predicted {len(predictions)} samples")
        
        # Reshape to raster format
        pred_raster, prob_raster, metadata = reshape_predictions_to_raster(
            predictions, probabilities, union_mask, timeseries_dicts[0]['metadata']
        )
        
        # Save results
        zone_output_dir = os.path.join(output_dir, zone, 'model_version_2.2.0')
        os.makedirs(zone_output_dir, exist_ok=True)
        
        pred_file = os.path.join(zone_output_dir, f'pred_{year}_2.2.0-class.tif')
        prob_file = os.path.join(zone_output_dir, f'pred_{year}_2.2.0-proba.tif')
        
        save_geotiff(pred_raster, metadata, pred_file, dtype='int16')
        save_geotiff(prob_raster, metadata, prob_file, dtype='float32')
        
        print(f"Saved predictions: {pred_file}")
        print(f"Saved probabilities: {prob_file}")
        
        return pred_file
        
    except Exception as e:
        print(f"Error processing {zone} - {year}: {str(e)}")
        return None


def main():
    """
    Main function to run forest type prediction.
    """
    parser = argparse.ArgumentParser(description='Forest Type Prediction v2.2.0')
    parser.add_argument('--data_dir', default='/fs2/ftype_mflf/shared_data/datacube/2025/1.2.0',
                       help='Base data directory')
    parser.add_argument('--zones', default='zone_1,zone_2,zone_3,zone_4,zone_5',
                       help='Comma-separated list of zones to process')
    parser.add_argument('--years', default='2019,2020,2021,2022,2023,2024,2025',
                       help='Comma-separated list of years to process')
    parser.add_argument('--model_path', 
                       default='/fs1/tor_project/model_foresttype/model/mfl_v2_2_0/hybrid_lstm_cnn_model-20250910.keras',
                       help='Path to trained model')
    parser.add_argument('--config_dir',
                       default='/fs1/tor_project/model_foresttype/model/mfl_v2_2_0/config',
                       help='Directory containing model configuration')
    parser.add_argument('--output_dir', default='/fs2/ftype_mflf/result/2025/1.2.0',
                       help='Output directory')
    parser.add_argument('--batch_size', type=int, default=128,
                       help='Batch size for prediction')
    parser.add_argument('--majority_vote', action='store_true',
                       help='Perform majority voting across years')
    
    args = parser.parse_args()
    
    # Parse zones and years
    zones = [z.strip() for z in args.zones.split(',')]
    years = [int(y.strip()) for y in args.years.split(',')]
    
    print("Forest Type Prediction v2.2.0")
    print("=" * 40)
    print(f"Data directory: {args.data_dir}")
    print(f"Model path: {args.model_path}")
    print(f"Config directory: {args.config_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Zones: {zones}")
    print(f"Years: {years}")
    print(f"Batch size: {args.batch_size}")
    print(f"Majority vote: {args.majority_vote}")
    
    # Load model and scaling parameters
    print("\nLoading model and scaling parameters...")
    
    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"Model not found: {args.model_path}")
    
    model = models.load_model(args.model_path, compile=False)
    print(f"Loaded model from: {args.model_path}")
    
    scaling_params = load_scaling_params(args.config_dir)
    
    # Process each zone and year
    os.makedirs(args.output_dir, exist_ok=True)
    
    all_prediction_files = {}
    
    for zone in zones:
        print(f"\nProcessing zone: {zone}")
        prediction_files = []
        
        for year in years:
            pred_file = process_zone_year(
                args.data_dir, zone, year, model, scaling_params, args.output_dir
            )
            if pred_file:
                prediction_files.append(pred_file)
        
        all_prediction_files[zone] = prediction_files
        
        # Majority voting if requested
        if args.majority_vote and len(prediction_files) > 1:
            zone_output_dir = os.path.join(args.output_dir, zone, 'model_version_2.2.0')
            majority_file = os.path.join(zone_output_dir, 'pred_majority_2.2.0.tif')
            
            # Use first prediction file as reference for metadata
            with rasterio.open(prediction_files[0]) as src:
                reference_metadata = {
                    'transform': src.transform,
                    'crs': src.crs,
                    'width': src.width,
                    'height': src.height
                }
            
            majority_vote_predictions(prediction_files, majority_file, reference_metadata)
    
    print("\nPrediction complete!")
    print("Summary of results:")
    for zone, files in all_prediction_files.items():
        print(f"  {zone}: {len(files)} years processed")


# ==============================================================================
# USAGE EXAMPLES AND DOCUMENTATION
# ==============================================================================

def example_usage():
    """
    Example usage of the prediction data preparation functions.
    """
    print("="*80)
    print("📖 EXAMPLE USAGE")
    print("="*80)
    
    # Example 1: Prepare data for specific zones and years
    print("Example 1: Basic prediction data preparation")
    print("-" * 50)
    
    example_zones = ['mfl_cmf_area_project_2025_chr_zone_1']
    example_years = [2023, 2024, 2025]
    
    # This would prepare data for prediction
    # prediction_data = prepare_prediction_data(example_zones, example_years)
    
    print("Code:")
    print("zones = ['mfl_cmf_area_project_2025_chr_zone_1']")
    print("years = [2023, 2024, 2025]")
    print("prediction_data = prepare_prediction_data(zones, years)")
    print("")
    print("# Use with trained model")
    print("predictions = model.predict([prediction_data['X_timeseries'],")
    print("                           prediction_data['X_static']])")
    
    print("\nExample 2: Manual step-by-step preparation")
    print("-" * 50)
    print("# Load scaling parameters")
    print("scaling_params = load_scaling_parameters()")
    print("")
    print("# Extract features")  
    print("dataset = extract_prediction_features(zones, years)")
    print("")
    print("# Prepare features")
    print("X_timeseries = prepare_timeseries_data(dataset)")
    print("X_static = prepare_static_data(dataset)")
    print("")
    print("# Apply scaling")
    print("X_ts_scaled, X_static_scaled = apply_scaling(X_timeseries, X_static, scaling_params)")
    
    print("\nExample 3: Feature channel mapping")
    print("-" * 50)
    print("Time Series Channels (10 total):")
    channels = [
        "Channel 0: VV (SAR backscatter)",
        "Channel 1: VH (SAR backscatter)", 
        "Channel 2: GVMI (Green Vegetation Moisture Index)",
        "Channel 3: NDVI (Normalized Difference Vegetation Index)",
        "Channel 4: SLAVI (Specific Leaf Area Vegetation Index)",
        "Channel 5: LAI (Leaf Area Index)",
        "Channel 6: FAPAR (Fraction of Absorbed PAR)",
        "Channel 7: FCOVER (Fraction of vegetation cover)",
        "Channel 8: VH/VV Ratio (computed)",
        "Channel 9: VV/VH Ratio (computed)"
    ]
    for channel in channels:
        print(f"  {channel}")
    
    print("\nStatic Features (2 total):")
    print("  Channel 0: Slope (degrees)")
    print("  Channel 1: Elevation (meters)")

if __name__ == "__main__":
    # Run example documentation on import
    print("\n" + "="*80)
    print("📚 PREDICTION DATA PREPARATION - READY FOR USE")
    print("="*80)
    print("All functions are now available for preparing prediction data.")
    print("Key functions:")
    print("  • prepare_prediction_data() - Complete pipeline")
    print("  • load_scaling_parameters() - Load training statistics") 
    print("  • extract_prediction_features() - Extract features from datacube")
    print("  • prepare_timeseries_data() - Process time series + VH/VV ratio")
    print("  • prepare_static_data() - Process slope and elevation")
    print("  • apply_scaling() - Apply z-score normalization")
    print("\nRun example_usage() to see detailed examples.")
    print("="*80)
    
    main()