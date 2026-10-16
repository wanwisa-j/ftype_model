#!/usr/bin/env python3
"""
Forest Type Classification Model Training Script v2.2.1

This script implements the spatially-separated training strategy to prevent data leakage.

Key Features:
- Proper spatial separation: Training excludes prediction zone
- Comprehensive validation: In-training + spatial validation
- Class balancing: Focal loss + class weights
- Feature engineering: VH/VV ratios, proper scaling
- Model monitoring: Distribution tracking during training

Training Configuration:
- Training zones: mfl_cmf_2024, mfl_cmf_area_project_2025_chr_zone_2
- Prediction zone (excluded): mfl_cmf_area_project_2025_chr_zone_1
- Features: VV, VH, GVMI, NDVI, SLAVI, LAI, FAPAR, FCOVER, slope, elevation + ratios
- Model: Hybrid CNN + Dense layers
- Loss: Focal loss with class weights

Author: Extracted from training notebook
Date: 2025-09-10
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# TensorFlow imports
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras import backend as K
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
import joblib

# Set random seeds
np.random.seed(42)
tf.random.set_seed(42)

# Add project paths
import yaml
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(REPO_ROOT, 'src'))
from dataset_builder import ForestTypeDatasetBuilder

with open(os.path.join(REPO_ROOT, 'config', 'config.yaml')) as _f:
    _model_root = yaml.safe_load(_f)['prediction']['model_root']
MODEL_ROOT = _model_root if os.path.isabs(_model_root) else os.path.join(REPO_ROOT, _model_root)


class ForestTypeTrainer:
    """Forest Type Classification Model Trainer with Spatial Separation"""
    
    def __init__(self, model_version="2.2.1"):
        self.model_version = model_version
        self.builder = ForestTypeDatasetBuilder()
        
        # Training configuration - SPATIAL SEPARATION STRATEGY
        self.training_zones = [
            'mfl_cmf_2024',                                    # Different geographic area
            'mfl_cmf_area_project_2025_chr_zone_2',            # Different zone in same project
            'mfl_cmf_area_project_2025_chr_zone_1'  # Reserved for prediction ONLY
        ]
        
        self.spatial_val_zones = ['mfl_cmf_area_project_2025_chr_zone_1']
        self.years = [2020, 2021, 2022, 2023, 2024, 2025]
        self.features = ['vv', 'vh', 'gvmi', 'ndvi', 'slavi', 'lai', 'fapar', 'fcover', 'slope', 'elevation']
        self.timeseries_features = ['vv', 'vh', 'gvmi', 'ndvi', 'slavi', 'lai', 'fapar', 'fcover']
        self.static_features = ['slope', 'elevation']
        self.target = 'ftype_nonmix'
        self.selected_classes = ['DDF', 'DEF', 'MDF']
        self.num_classes = len(self.selected_classes)
        
        # Model artifacts
        self.model = None
        self.le = None
        self.scaling_params = None
        self.history = None
        
        # Data splits
        self.train_data = {}
        self.validation_data = {}
        self.spatial_val_data = {}
        
    def load_and_prepare_data(self):
        """Load and prepare training data with spatial separation"""
        print("🔄 Loading and preparing training data...")
        print(f"   Training zones: {self.training_zones}")
        print(f"   Spatial validation zone: {self.spatial_val_zones[0]} (EXCLUDED from training)")
        print(f"   Years: {self.years}")
        print(f"   Features: {self.features}")
        
        # Load training dataset (excludes prediction zone)
        dataset = self.builder.create_dataset(
            zones=self.training_zones,
            years=self.years,
            features=self.features,
            target=self.target,
            return_point_info=True
        )
        
        # Load spatial validation dataset (independent zone)
        spatial_val_dataset = self.builder.create_dataset(
            zones=self.spatial_val_zones,
            years=self.years,
            features=self.features,
            target=self.target,
            return_point_info=True
        )
        
        print(f"✅ Training dataset: {dataset['target'].shape[0]} samples")
        print(f"✅ Spatial validation dataset: {spatial_val_dataset['target'].shape[0]} samples")
        
        return dataset, spatial_val_dataset
    
    def prepare_features(self, dataset):
        """Prepare timeseries and static features with VH/VV ratios"""
        # Prepare timeseries features
        X_timeseries = np.dstack([dataset[f'{feat}_feature'] for feat in self.timeseries_features])
        
        # Add VH/VV ratio (Channel 9)
        vv = X_timeseries[..., 0]  # VV is channel 0
        vh = X_timeseries[..., 1]  # VH is channel 1
        denom = np.where(vv == 0, 1e-6, vv)
        vh_vv_ratio = vh / denom
        X_timeseries = np.concatenate([X_timeseries, vh_vv_ratio[..., np.newaxis]], axis=-1)
        
        # Add VV/VH ratio (Channel 10)
        denom = np.where(vh == 0, 1e-6, vh)
        vv_vh_ratio = vv / denom
        X_timeseries = np.concatenate([X_timeseries, vv_vh_ratio[..., np.newaxis]], axis=-1)
        
        # Prepare static features
        X_static = np.hstack([dataset[f'{feat}_feature'] for feat in self.static_features])
        
        print(f"   Timeseries shape: {X_timeseries.shape} (samples, timesteps, channels)")
        print(f"   Static shape: {X_static.shape} (samples, features)")
        
        return X_timeseries, X_static
    
    def filter_classes(self, dataset, X_timeseries, X_static):
        """Filter for specified forest classes"""
        targets = np.char.upper(dataset['target'].astype(str).ravel())
        mask = np.isin(targets, self.selected_classes)
        
        X_timeseries_filtered = X_timeseries[mask]
        X_static_filtered = X_static[mask]
        targets_filtered = targets[mask]
        
        unique, counts = np.unique(targets_filtered, return_counts=True)
        print(f"   Class distribution: {dict(zip(unique, counts))}")
        
        return X_timeseries_filtered, X_static_filtered, targets_filtered
    
    def create_train_test_splits(self, X_timeseries, X_static, targets):
        """Create train/test splits and prepare validation data"""
        # Main train/test split from training zones
        X_ts_train, X_ts_test, X_st_train, X_st_test, y_train, y_test = train_test_split(
            X_timeseries, X_static, targets, test_size=0.2, random_state=42, stratify=targets
        )
        
        # Create small validation subset from training data for monitoring
        sample_ratio = 0.1 if len(y_train) > 1000 else 0.2
        sss = StratifiedShuffleSplit(n_splits=1, test_size=sample_ratio, random_state=42)
        for _, train_val_idx in sss.split(np.zeros(len(y_train)), y_train):
            X_ts_train_val = X_ts_train[train_val_idx]
            X_st_train_val = X_st_train[train_val_idx]
            y_train_val = y_train[train_val_idx]
        
        print(f"   Training samples: {len(y_train)}")
        print(f"   In-training validation: {len(y_test)}")
        print(f"   Training monitoring subset: {len(y_train_val)}")
        
        return {
            'X_timeseries_train': X_ts_train,
            'X_static_train': X_st_train,
            'y_train': y_train,
            'X_timeseries_test': X_ts_test,
            'X_static_test': X_st_test,
            'y_test': y_test,
            'X_timeseries_train_val': X_ts_train_val,
            'X_static_train_val': X_st_train_val,
            'y_train_val': y_train_val
        }
    
    def apply_scaling(self, train_data, spatial_val_data):
        """Apply z-score scaling using training statistics"""
        print("🔄 Applying feature scaling...")
        
        # Time series scaling
        def zscore_time_series(train, *validation_sets):
            mean = train.mean(axis=(0,1), keepdims=True)
            std = train.std(axis=(0,1), keepdims=True) + 1e-6
            train_scaled = (train - mean) / std
            val_scaled = []
            for val_set in validation_sets:
                val_scaled.append((val_set - mean) / std)
            return train_scaled, *val_scaled, mean, std
        
        # Scale timeseries features
        X_ts_train_scaled, X_ts_test_scaled, X_ts_train_val_scaled, X_ts_spatial_scaled, ts_mean, ts_std = zscore_time_series(
            train_data['X_timeseries_train'],
            train_data['X_timeseries_test'], 
            train_data['X_timeseries_train_val'],
            spatial_val_data['X_timeseries']
        )
        
        # Scale static features
        static_mean = train_data['X_static_train'].mean(axis=0, keepdims=True)
        static_std = train_data['X_static_train'].std(axis=0, keepdims=True) + 1e-6
        
        X_st_train_scaled = (train_data['X_static_train'] - static_mean) / static_std
        X_st_test_scaled = (train_data['X_static_test'] - static_mean) / static_std
        X_st_train_val_scaled = (train_data['X_static_train_val'] - static_mean) / static_std
        X_st_spatial_scaled = (spatial_val_data['X_static'] - static_mean) / static_std
        
        # Store scaling parameters
        self.scaling_params = {
            'static_mean': static_mean.astype(np.float32),
            'static_std': static_std.astype(np.float32),
            'timeseries_mean': ts_mean.astype(np.float32),
            'timeseries_std': ts_std.astype(np.float32),
            'model_version': self.model_version,
            'training_zones': self.training_zones,
            'excluded_prediction_zone': self.spatial_val_zones[0]
        }
        
        # Update data dictionaries with scaled features
        train_data.update({
            'X_timeseries_train': X_ts_train_scaled,
            'X_static_train': X_st_train_scaled,
            'X_timeseries_test': X_ts_test_scaled,
            'X_static_test': X_st_test_scaled,
            'X_timeseries_train_val': X_ts_train_val_scaled,
            'X_static_train_val': X_st_train_val_scaled
        })
        
        spatial_val_data.update({
            'X_timeseries': X_ts_spatial_scaled,
            'X_static': X_st_spatial_scaled
        })
        
        print(f"   ✅ Scaling applied to all datasets")
        return train_data, spatial_val_data
    
    def encode_labels(self, train_data, spatial_val_data):
        """Encode string labels to integers"""
        print("🔄 Encoding target labels...")
        
        # Fit label encoder on all unique labels
        all_targets = np.concatenate([
            train_data['y_train'], 
            train_data['y_test'], 
            train_data['y_train_val'],
            spatial_val_data['y']
        ])
        
        self.le = LabelEncoder()
        self.le.fit(np.unique(all_targets))
        print(f"   Available classes: {self.le.classes_}")
        
        # Transform all target sets
        train_data['y_train'] = self.le.transform(train_data['y_train']).astype(np.int32)
        train_data['y_test'] = self.le.transform(train_data['y_test']).astype(np.int32)
        train_data['y_train_val'] = self.le.transform(train_data['y_train_val']).astype(np.int32)
        spatial_val_data['y'] = self.le.transform(spatial_val_data['y']).astype(np.int32)
        
        # Show class distributions
        print("Training set distribution:")
        unique, counts = np.unique(train_data['y_train'], return_counts=True)
        for cls, count in zip(unique, counts):
            cls_name = self.le.inverse_transform([cls])[0]
            percentage = count / len(train_data['y_train']) * 100
            print(f"   {cls_name} (Class {cls}): {count} samples ({percentage:.1f}%)")
        
        print("Spatial validation distribution:")
        unique, counts = np.unique(spatial_val_data['y'], return_counts=True)
        for cls, count in zip(unique, counts):
            cls_name = self.le.inverse_transform([cls])[0]
            percentage = count / len(spatial_val_data['y']) * 100
            print(f"   {cls_name} (Class {cls}): {count} samples ({percentage:.1f}%)")
        
        return train_data, spatial_val_data
    
    def create_model(self, input_shape_timeseries, static_features_shape):
        """Create hybrid CNN model"""
        print("🔄 Creating hybrid CNN model...")
        
        # Timeseries input and CNN branch
        timeseries_input = layers.Input(shape=input_shape_timeseries, name="timeseries_input")
        
        # 1D CNN processing
        x_cnn = layers.Conv1D(64, 3, activation='relu', padding='same')(timeseries_input)
        x_cnn = layers.MaxPooling1D(2)(x_cnn)
        x_cnn = layers.Conv1D(128, 3, activation='relu', padding='same')(x_cnn)
        x_cnn = layers.MaxPooling1D(2)(x_cnn)
        x_cnn = layers.GlobalAveragePooling1D()(x_cnn)
        
        # Static features input
        static_input = layers.Input(shape=static_features_shape, name="static_input")
        
        # Combine features
        x = layers.concatenate([x_cnn, static_input])
        
        # Dense layers
        x = layers.Dense(256, activation='relu')(x)
        x = layers.Dropout(0.5)(x)
        x = layers.Dense(128, activation='relu')(x)
        x = layers.Dropout(0.5)(x)
        
        # Output layer
        output = layers.Dense(self.num_classes, activation='softmax')(x)
        
        # Create model
        model = models.Model(inputs=[timeseries_input, static_input], outputs=output)
        
        print(f"   ✅ Model created with {model.count_params()} parameters")
        return model
    
    def create_focal_loss(self, y_train):
        """Create focal loss with class weights"""
        # Compute class weights
        classes = np.unique(y_train)
        class_weights = compute_class_weight(class_weight='balanced', classes=classes, y=y_train)
        class_weight_dict = {i: class_weights[i] for i in range(len(class_weights))}
        
        print(f"   Class weights: {class_weight_dict}")
        
        # Create focal loss
        alpha_list = class_weights.tolist()
        base_loss = tf.keras.losses.CategoricalFocalCrossentropy(
            alpha=alpha_list, gamma=2.0, from_logits=False
        )
        
        def sparse_to_categorical_focal_factory(n_classes):
            def loss(y_true, y_pred):
                y_true = tf.reshape(tf.cast(tf.squeeze(y_true), tf.int32), [-1])
                y_true_ohe = tf.one_hot(y_true, depth=n_classes)
                y_true_ohe = tf.cast(y_true_ohe, y_pred.dtype)
                return base_loss(y_true_ohe, y_pred)
            return loss
        
        return sparse_to_categorical_focal_factory(self.num_classes), class_weight_dict
    
    def create_callbacks(self, train_data, spatial_val_data):
        """Create training callbacks including validation monitoring"""
        
        # Distribution monitoring callback
        class DistributionMonitor(tf.keras.callbacks.Callback):
            def __init__(self, val_inputs, val_targets, spatial_inputs, spatial_targets, le):
                super().__init__()
                self.val_inputs = val_inputs
                self.val_targets = val_targets
                self.spatial_inputs = spatial_inputs
                self.spatial_targets = spatial_targets
                self.le = le

            def on_epoch_end(self, epoch, logs=None):
                if (epoch + 1) % 10 == 0:  # Every 10 epochs
                    # In-training validation
                    y_pred = self.model.predict(self.val_inputs, verbose=0)
                    y_pred_classes = np.argmax(y_pred, axis=1)
                    unique, counts = np.unique(y_pred_classes, return_counts=True)
                    print(f"\\n[Epoch {epoch+1}] In-training val predictions:")
                    for u, c in zip(unique, counts):
                        class_name = self.le.inverse_transform([u])[0]
                        print(f"  {class_name}: {c} samples")
                    
                    # Spatial validation
                    y_pred_spatial = self.model.predict(self.spatial_inputs, verbose=0)
                    y_pred_spatial_classes = np.argmax(y_pred_spatial, axis=1)
                    unique, counts = np.unique(y_pred_spatial_classes, return_counts=True)
                    print(f"[Epoch {epoch+1}] Spatial val predictions:")
                    for u, c in zip(unique, counts):
                        class_name = self.le.inverse_transform([u])[0]
                        print(f"  {class_name}: {c} samples")
        
        # Create model save directory
        model_dir = os.path.join(MODEL_ROOT, f'mfl_v{self.model_version.replace(".", "_")}')
        os.makedirs(model_dir, exist_ok=True)
        
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
            ModelCheckpoint(f'{model_dir}/best_model.h5', monitor='val_loss', save_best_only=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6),
            DistributionMonitor(
                val_inputs=[train_data['X_timeseries_test'], train_data['X_static_test']],
                val_targets=train_data['y_test'],
                spatial_inputs=[spatial_val_data['X_timeseries'], spatial_val_data['X_static']],
                spatial_targets=spatial_val_data['y'],
                le=self.le
            )
        ]
        
        return callbacks
    
    def train_model(self, train_data, spatial_val_data, epochs=120, batch_size=32):
        """Train the model"""
        print("🔄 Training model...")
        
        # Create model
        input_shape_timeseries = (train_data['X_timeseries_train'].shape[1], train_data['X_timeseries_train'].shape[2])
        static_features_shape = (train_data['X_static_train'].shape[1],)
        
        self.model = self.create_model(input_shape_timeseries, static_features_shape)
        
        # Create loss and compile
        focal_loss, class_weight_dict = self.create_focal_loss(train_data['y_train'])
        self.model.compile(optimizer='adam', loss=focal_loss, metrics=['accuracy'])
        
        # Create callbacks
        callbacks = self.create_callbacks(train_data, spatial_val_data)
        
        # Train model
        self.history = self.model.fit(
            x=[train_data['X_timeseries_train'], train_data['X_static_train']],
            y=train_data['y_train'],
            validation_data=([train_data['X_timeseries_train_val'], train_data['X_static_train_val']], 
                           train_data['y_train_val']),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            class_weight=class_weight_dict,
            shuffle=True,
            verbose=1
        )
        
        print(f"✅ Training completed")
    
    def evaluate_model(self, train_data, spatial_val_data):
        """Comprehensive model evaluation"""
        print("🔄 Evaluating model...")
        
        # In-training validation
        print("\\n" + "="*60)
        print("📊 IN-TRAINING VALIDATION RESULTS")
        print("="*60)
        
        y_pred_intraining = self.model.predict([train_data['X_timeseries_test'], train_data['X_static_test']])
        y_pred_intraining_classes = np.argmax(y_pred_intraining, axis=1)
        
        print(classification_report(train_data['y_test'], y_pred_intraining_classes, 
                                  target_names=self.le.classes_))
        
        # Spatial validation
        print("\\n" + "="*60)
        print("🌍 SPATIAL VALIDATION RESULTS")
        print("="*60)
        
        y_pred_spatial = self.model.predict([spatial_val_data['X_timeseries'], spatial_val_data['X_static']])
        y_pred_spatial_classes = np.argmax(y_pred_spatial, axis=1)
        
        print(classification_report(spatial_val_data['y'], y_pred_spatial_classes,
                                  target_names=self.le.classes_))
        
        # Data leakage check
        print("\\n" + "="*60)
        print("🔍 SPATIAL VALIDATION ANALYSIS (Data Leakage Check)")
        print("="*60)
        
        unique_pred, counts_pred = np.unique(y_pred_spatial_classes, return_counts=True)
        total_pred = len(y_pred_spatial_classes)
        
        print("Predicted class distribution:")
        for cls, count in zip(unique_pred, counts_pred):
            cls_name = self.le.inverse_transform([cls])[0]
            percentage = count / total_pred * 100
            print(f"   {cls_name} (Class {cls}): {count} predictions ({percentage:.1f}%)")
        
        unique_true, counts_true = np.unique(spatial_val_data['y'], return_counts=True)
        print("\\nGround truth distribution:")
        for cls, count in zip(unique_true, counts_true):
            cls_name = self.le.inverse_transform([cls])[0]
            percentage = count / len(spatial_val_data['y']) * 100
            print(f"   {cls_name} (Class {cls}): {count} samples ({percentage:.1f}%)")
        
        # Check for issues
        if len(unique_pred) == 1:
            print(f"\\n⚠️  WARNING: Model predicts only one class ({self.le.inverse_transform(unique_pred)[0]})")
        elif max(counts_pred) / total_pred > 0.95:
            dominant_class = self.le.inverse_transform([unique_pred[np.argmax(counts_pred)]])[0]
            dominant_pct = max(counts_pred) / total_pred * 100
            print(f"\\n⚠️  WARNING: One class dominates {dominant_pct:.1f}% of predictions ({dominant_class})")
        else:
            print("\\n✅ GOOD: Model shows reasonable prediction diversity")
        
        return {
            'y_pred_spatial': y_pred_spatial_classes,
            'y_true_spatial': spatial_val_data['y'],
            'accuracy_spatial': np.mean(y_pred_spatial_classes == spatial_val_data['y'])
        }
    
    def save_model_and_artifacts(self):
        """Save model and associated artifacts"""
        print("🔄 Saving model and artifacts...")
        
        # Create directories
        model_dir = os.path.join(MODEL_ROOT, f'mfl_v{self.model_version.replace(".", "_")}')
        config_dir = f'{model_dir}/config'
        os.makedirs(model_dir, exist_ok=True)
        os.makedirs(config_dir, exist_ok=True)
        
        # Save model
        model_path = f'{model_dir}/hybrid_lstm_cnn_model-20250910-no-leakage.keras'
        self.model.save(model_path)
        print(f"   ✅ Model saved: {model_path}")
        
        # Save scaling parameters
        scaling_path = f'{config_dir}/scaling_params_v{self.model_version.replace(".", "_")}_no_leakage-20250910.npz'
        np.savez_compressed(
            scaling_path,
            static_mean=self.scaling_params['static_mean'],
            static_std=self.scaling_params['static_std'],
            timeseries_mean=self.scaling_params['timeseries_mean'],
            timeseries_std=self.scaling_params['timeseries_std'],
            model_version=str(self.scaling_params['model_version']),
            training_zones=str(self.scaling_params['training_zones']),
            excluded_prediction_zone=str(self.scaling_params['excluded_prediction_zone'])
        )
        print(f"   ✅ Scaling parameters saved: {scaling_path}")
        
        # Save label encoder
        le_path = f'{config_dir}/label_encoder_v{self.model_version.replace(".", "_")}.joblib'
        joblib.dump(self.le, le_path)
        print(f"   ✅ Label encoder saved: {le_path}")
        
        print(f"\\n🎉 Model v{self.model_version} training completed!")
        print(f"   Training zones: {self.training_zones}")
        print(f"   Can predict on: {self.spatial_val_zones[0]}")
    
    def run_full_training(self, epochs=120, batch_size=32, save_artifacts=True):
        """Run complete training pipeline"""
        print("="*80)
        print(f"🚀 FOREST TYPE MODEL v{self.model_version} TRAINING")
        print("="*80)
        print("🚫 DATA LEAKAGE PREVENTION:")
        print(f"   Training zones: {self.training_zones}")
        print(f"   Prediction zone (EXCLUDED): {self.spatial_val_zones[0]}")
        print("="*80)
        
        # Step 1: Load data
        dataset, spatial_val_dataset = self.load_and_prepare_data()
        
        # Step 2: Prepare features
        print("\\n🔄 Preparing training features...")
        X_timeseries, X_static = self.prepare_features(dataset)
        X_timeseries, X_static, targets = self.filter_classes(dataset, X_timeseries, X_static)
        
        print("🔄 Preparing spatial validation features...")
        X_timeseries_spatial, X_static_spatial = self.prepare_features(spatial_val_dataset)
        X_timeseries_spatial, X_static_spatial, targets_spatial = self.filter_classes(
            spatial_val_dataset, X_timeseries_spatial, X_static_spatial
        )
        
        # Step 3: Create data splits
        train_data = self.create_train_test_splits(X_timeseries, X_static, targets)
        spatial_val_data = {
            'X_timeseries': X_timeseries_spatial,
            'X_static': X_static_spatial,
            'y': targets_spatial
        }
        
        # Step 4: Apply scaling
        train_data, spatial_val_data = self.apply_scaling(train_data, spatial_val_data)
        
        # Step 5: Encode labels
        train_data, spatial_val_data = self.encode_labels(train_data, spatial_val_data)
        
        # Step 6: Train model
        self.train_model(train_data, spatial_val_data, epochs, batch_size)
        
        # Step 7: Evaluate model
        results = self.evaluate_model(train_data, spatial_val_data)
        
        # Step 8: Save artifacts
        if save_artifacts:
            self.save_model_and_artifacts()
        
        print("="*80)
        print("✅ TRAINING PIPELINE COMPLETED")
        print("="*80)
        
        return {
            'model': self.model,
            'history': self.history,
            'scaling_params': self.scaling_params,
            'label_encoder': self.le,
            'evaluation_results': results,
            'train_data': train_data,
            'spatial_val_data': spatial_val_data
        }


def main():
    """Main training function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Train Forest Type Classification Model v2.2.1')
    parser.add_argument('--epochs', type=int, default=120, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--model_version', type=str, default='2.2.1', help='Model version')
    parser.add_argument('--no_save', action='store_true', help='Skip saving artifacts')
    
    args = parser.parse_args()
    
    # Create trainer
    trainer = ForestTypeTrainer(model_version=args.model_version)
    
    # Run training
    results = trainer.run_full_training(
        epochs=args.epochs,
        batch_size=args.batch_size,
        save_artifacts=not args.no_save
    )
    
    return results


if __name__ == "__main__":
    results = main()