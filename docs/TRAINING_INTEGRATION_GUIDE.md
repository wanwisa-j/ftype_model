# Training Integration Guide for Forest Type Dataset v1.2.0

## Overview

This guide provides step-by-step instructions for integrating the comprehensive forest type dataset v1.2.0 into machine learning training workflows. The dataset is now production-ready with complete feature coverage across all zones.

## Quick Start

### 1. Load the Comprehensive Dataset
```python
import pandas as pd
import numpy as np

# Load the comprehensive dataset
df = pd.read_parquet('/fs2/ftype_mflf/shared_data/dataset/1.2.0/raw/forest_type_dataset_raw_20250903_104423.parquet')

print(f"Dataset loaded: {len(df):,} records, {df.point_id.nunique():,} points")
print(f"Features: {sorted(df.feature.unique())}")
```

### 2. Select Training Configuration
```python
# Choose your training strategy
training_config = {
    'zones': ['mfl_cmf_2024', 'mfl_cmf_area_project_2025_chr_zone_1'],
    'features': ['lai', 'fapar', 'fcover', 'ndvi', 'gvmi', 'slavi', 'vv', 'vh', 'elevation', 'slope'],
    'years': [2020, 2021, 2022, 2023],
    'target': 'ftype_mix'  # or 'ftype_nonmix'
}

# Filter dataset
training_data = df[
    (df.zone.isin(training_config['zones'])) &
    (df.feature.isin(training_config['features'])) &
    ((df.year.isin(training_config['years'])) | (df.year.isna()))
]
```

### 3. Convert to ML-Ready Format
```python
def prepare_ml_dataset(df_long, target_col='ftype_mix'):
    """Convert long format to ML-ready wide format"""
    
    # Separate time series and static features
    ts_features = ['lai', 'fapar', 'fcover', 'ndvi', 'gvmi', 'slavi', 'vv', 'vh']
    static_features = ['elevation', 'slope']
    
    ts_data = df_long[df_long.feature.isin(ts_features)]
    static_data = df_long[df_long.feature.isin(static_features)]
    
    # Get point metadata
    metadata = df_long.groupby('point_id').agg({
        'utm_x': 'first',
        'utm_y': 'first',
        'ftype_mix': 'first',
        'ftype_nonmix': 'first',
        'zone': 'first'
    }).reset_index()
    
    # Pivot time series
    if len(ts_data) > 0:
        ts_wide = ts_data.pivot_table(
            index='point_id',
            columns=['feature', 'week', 'year'],
            values='value',
            aggfunc='first'
        )
        ts_wide.columns = [f"{f}_w{w}_y{int(y)}" for f, w, y in ts_wide.columns]
        ts_wide = ts_wide.reset_index()
    else:
        ts_wide = pd.DataFrame({'point_id': metadata['point_id']})
    
    # Pivot static features
    if len(static_data) > 0:
        static_wide = static_data.pivot_table(
            index='point_id',
            columns='feature',
            values='value',
            aggfunc='first'
        ).reset_index()
    else:
        static_wide = pd.DataFrame({'point_id': metadata['point_id']})
    
    # Merge all
    ml_dataset = metadata.merge(ts_wide, on='point_id', how='left')
    ml_dataset = ml_dataset.merge(static_wide, on='point_id', how='left')
    
    # Prepare features and target
    feature_cols = [col for col in ml_dataset.columns 
                   if col not in ['point_id', 'utm_x', 'utm_y', 'ftype_mix', 'ftype_nonmix', 'zone']]
    
    X = ml_dataset[feature_cols]
    y = ml_dataset[target_col]
    
    return X, y, ml_dataset

# Prepare ML dataset
X, y, full_dataset = prepare_ml_dataset(training_data, target_col='ftype_mix')
print(f"ML dataset ready: X shape {X.shape}, y shape {y.shape}")
```

## Advanced Training Scenarios

### Scenario 1: Deep Learning Time Series
```python
def prepare_time_series_arrays(df_long, sequence_length=21):
    """Prepare 3D arrays for deep learning time series models"""
    
    ts_features = ['lai', 'fapar', 'fcover', 'ndvi', 'gvmi', 'slavi', 'vv', 'vh']
    
    # Filter to time series only
    ts_data = df_long[df_long.feature.isin(ts_features)]
    
    # Get unique points and prepare arrays
    points = sorted(ts_data.point_id.unique())
    features = sorted(ts_features)
    years = sorted(ts_data.year.dropna().unique())
    
    # Initialize arrays
    n_points = len(points)
    n_features = len(features)
    n_timesteps = len(years) * sequence_length
    
    X_3d = np.full((n_points, n_timesteps, n_features), np.nan)
    point_metadata = []
    
    for i, point_id in enumerate(points):
        point_data = ts_data[ts_data.point_id == point_id]
        
        # Get metadata
        point_meta = df_long[df_long.point_id == point_id].iloc[0]
        point_metadata.append({
            'point_id': point_id,
            'ftype_mix': point_meta['ftype_mix'],
            'ftype_nonmix': point_meta['ftype_nonmix'],
            'zone': point_meta['zone']
        })
        
        # Fill time series data
        for j, feature in enumerate(features):
            feature_data = point_data[point_data.feature == feature]
            
            for k, year in enumerate(years):
                year_data = feature_data[feature_data.year == year]
                
                for week in range(1, sequence_length + 1):
                    week_data = year_data[year_data.week == week]
                    if len(week_data) > 0:
                        timestep = k * sequence_length + (week - 1)
                        X_3d[i, timestep, j] = week_data.iloc[0]['value']
    
    metadata_df = pd.DataFrame(point_metadata)
    y = metadata_df['ftype_mix'].values
    
    return X_3d, y, metadata_df

# Prepare for deep learning
X_3d, y_dl, metadata = prepare_time_series_arrays(training_data)
print(f"Deep learning arrays: X shape {X_3d.shape}, y shape {y_dl.shape}")
```

### Scenario 2: Multi-Task Learning
```python
def prepare_multitask_dataset(df_long):
    """Prepare dataset for multi-task learning (both ftype_mix and ftype_nonmix)"""
    
    X, _, full_dataset = prepare_ml_dataset(df_long)
    
    # Multiple targets
    y_mix = full_dataset['ftype_mix']
    y_nonmix = full_dataset['ftype_nonmix']
    
    # Convert to multi-hot encoding if needed
    from sklearn.preprocessing import LabelBinarizer
    
    lb_mix = LabelBinarizer()
    lb_nonmix = LabelBinarizer()
    
    y_mix_encoded = lb_mix.fit_transform(y_mix)
    y_nonmix_encoded = lb_nonmix.fit_transform(y_nonmix)
    
    return X, {'mix': y_mix_encoded, 'nonmix': y_nonmix_encoded}, full_dataset

# Prepare multi-task dataset
X_mt, y_mt, dataset_mt = prepare_multitask_dataset(training_data)
print(f"Multi-task dataset: X {X_mt.shape}, y_mix {y_mt['mix'].shape}, y_nonmix {y_mt['nonmix'].shape}")
```

### Scenario 3: Spatial Cross-Validation
```python
def create_spatial_splits(df_long, test_zones=['mfl_cmf_area_project_2025_chr_zone_1']):
    """Create spatial train-test splits to avoid spatial autocorrelation"""
    
    # Prepare full dataset
    X, y, full_dataset = prepare_ml_dataset(df_long)
    
    # Split by zones
    train_mask = ~full_dataset['zone'].isin(test_zones)
    test_mask = full_dataset['zone'].isin(test_zones)
    
    X_train = X[train_mask]
    X_test = X[test_mask]
    y_train = y[train_mask]
    y_test = y[test_mask]
    
    print(f"Spatial split:")
    print(f"  Train: {len(X_train)} samples from zones {full_dataset[train_mask]['zone'].unique()}")
    print(f"  Test: {len(X_test)} samples from zones {full_dataset[test_mask]['zone'].unique()}")
    
    return X_train, X_test, y_train, y_test

# Create spatial splits
X_train_sp, X_test_sp, y_train_sp, y_test_sp = create_spatial_splits(training_data)
```

## Model Training Examples

### Random Forest Example
```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.impute import SimpleImputer

def train_random_forest(X_train, y_train, X_test, y_test):
    """Train Random Forest with comprehensive features"""
    
    # Handle missing values
    imputer = SimpleImputer(strategy='median')
    X_train_imputed = imputer.fit_transform(X_train)
    X_test_imputed = imputer.transform(X_test)
    
    # Train model
    rf = RandomForestClassifier(
        n_estimators=100,
        max_depth=20,
        random_state=42,
        n_jobs=-1
    )
    
    rf.fit(X_train_imputed, y_train)
    
    # Evaluate
    y_pred = rf.predict(X_test_imputed)
    
    print("Random Forest Results:")
    print(classification_report(y_test, y_pred))
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': X_train.columns,
        'importance': rf.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\nTop 10 Important Features:")
    print(feature_importance.head(10))
    
    return rf, imputer, feature_importance

# Train Random Forest
rf_model, rf_imputer, rf_importance = train_random_forest(X_train_sp, y_train_sp, X_test_sp, y_test_sp)
```

### XGBoost Example
```python
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder

def train_xgboost(X_train, y_train, X_test, y_test):
    """Train XGBoost with comprehensive features"""
    
    # Handle missing values and encode labels
    imputer = SimpleImputer(strategy='median')
    X_train_imputed = imputer.fit_transform(X_train)
    X_test_imputed = imputer.transform(X_test)
    
    le = LabelEncoder()
    y_train_encoded = le.fit_transform(y_train)
    y_test_encoded = le.transform(y_test)
    
    # Train model
    xgb_model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        random_state=42,
        eval_metric='mlogloss'
    )
    
    xgb_model.fit(
        X_train_imputed, y_train_encoded,
        eval_set=[(X_test_imputed, y_test_encoded)],
        early_stopping_rounds=10,
        verbose=False
    )
    
    # Evaluate
    y_pred_encoded = xgb_model.predict(X_test_imputed)
    y_pred = le.inverse_transform(y_pred_encoded)
    
    print("XGBoost Results:")
    print(classification_report(y_test, y_pred))
    
    return xgb_model, imputer, le

# Train XGBoost
xgb_model, xgb_imputer, xgb_encoder = train_xgboost(X_train_sp, y_train_sp, X_test_sp, y_test_sp)
```

### Deep Learning Example (TensorFlow/Keras)
```python
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import StandardScaler, LabelEncoder

def train_lstm_model(X_3d, y, test_size=0.2):
    """Train LSTM model for time series classification"""
    
    # Split data
    n_samples = X_3d.shape[0]
    split_idx = int(n_samples * (1 - test_size))
    
    X_train, X_test = X_3d[:split_idx], X_3d[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    # Handle missing values (forward fill then backward fill)
    X_train_filled = pd.DataFrame(X_train.reshape(-1, X_train.shape[-1])).fillna(method='ffill').fillna(method='bfill').values.reshape(X_train.shape)
    X_test_filled = pd.DataFrame(X_test.reshape(-1, X_test.shape[-1])).fillna(method='ffill').fillna(method='bfill').values.reshape(X_test.shape)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_filled.reshape(-1, X_train_filled.shape[-1])).reshape(X_train_filled.shape)
    X_test_scaled = scaler.transform(X_test_filled.reshape(-1, X_test_filled.shape[-1])).reshape(X_test_filled.shape)
    
    # Encode labels
    le = LabelEncoder()
    y_train_encoded = le.fit_transform(y_train)
    y_test_encoded = le.transform(y_test)
    
    n_classes = len(le.classes_)
    
    # Build model
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=(X_train_scaled.shape[1], X_train_scaled.shape[2])),
        Dropout(0.2),
        LSTM(32),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dense(n_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Train model
    history = model.fit(
        X_train_scaled, y_train_encoded,
        validation_data=(X_test_scaled, y_test_encoded),
        epochs=50,
        batch_size=32,
        verbose=1
    )
    
    # Evaluate
    test_loss, test_accuracy = model.evaluate(X_test_scaled, y_test_encoded, verbose=0)
    print(f"LSTM Test Accuracy: {test_accuracy:.4f}")
    
    return model, scaler, le, history

# Train LSTM (if deep learning arrays are available)
# lstm_model, lstm_scaler, lstm_encoder, lstm_history = train_lstm_model(X_3d, y_dl)
```

## Performance Optimization

### Memory-Efficient Loading
```python
def load_dataset_efficiently(zones=None, features=None, years=None, chunk_size=50000):
    """Load dataset efficiently for large-scale training"""
    
    chunks = []
    for chunk in pd.read_parquet(
        '/fs2/ftype_mflf/shared_data/dataset/1.2.0/raw/forest_type_dataset_raw_20250903_104423.parquet',
        chunksize=chunk_size
    ):
        # Apply filters
        if zones:
            chunk = chunk[chunk.zone.isin(zones)]
        if features:
            chunk = chunk[chunk.feature.isin(features)]
        if years:
            chunk = chunk[chunk.year.isin(years) | chunk.year.isna()]
        
        if len(chunk) > 0:
            chunks.append(chunk)
    
    return pd.concat(chunks, ignore_index=True)
```

### Feature Selection
```python
def select_important_features(X, y, method='mutual_info', top_k=50):
    """Select most important features for training"""
    
    from sklearn.feature_selection import mutual_info_classif, SelectKBest
    from sklearn.impute import SimpleImputer
    
    # Handle missing values
    imputer = SimpleImputer(strategy='median')
    X_imputed = imputer.fit_transform(X)
    
    if method == 'mutual_info':
        selector = SelectKBest(score_func=mutual_info_classif, k=top_k)
    else:
        raise ValueError("Only mutual_info method implemented")
    
    X_selected = selector.fit_transform(X_imputed, y)
    selected_features = X.columns[selector.get_support()]
    
    print(f"Selected {len(selected_features)} features from {X.shape[1]}")
    
    return X_selected, selected_features, selector, imputer
```

## Model Evaluation and Validation

### Comprehensive Evaluation
```python
def comprehensive_evaluation(model, X_test, y_test, class_names=None):
    """Comprehensive model evaluation with multiple metrics"""
    
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    # Predictions
    y_pred = model.predict(X_test)
    
    # Basic metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision, recall, f1, support = precision_recall_fscore_support(y_test, y_pred, average='weighted')
    
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'confusion_matrix': cm
    }
```

## Export and Deployment

### Model Serialization
```python
import joblib
import json

def save_training_pipeline(model, preprocessors, metadata, output_dir):
    """Save complete training pipeline for deployment"""
    
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    # Save model
    joblib.dump(model, f"{output_dir}/model.pkl")
    
    # Save preprocessors
    if 'imputer' in preprocessors:
        joblib.dump(preprocessors['imputer'], f"{output_dir}/imputer.pkl")
    if 'scaler' in preprocessors:
        joblib.dump(preprocessors['scaler'], f"{output_dir}/scaler.pkl")
    if 'encoder' in preprocessors:
        joblib.dump(preprocessors['encoder'], f"{output_dir}/encoder.pkl")
    
    # Save metadata
    with open(f"{output_dir}/metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Training pipeline saved to {output_dir}")

# Example usage
preprocessors = {
    'imputer': rf_imputer,
    # 'scaler': scaler,
    # 'encoder': encoder
}

metadata = {
    'model_type': 'RandomForest',
    'features_used': list(X.columns),
    'training_zones': training_config['zones'],
    'training_years': training_config['years'],
    'dataset_version': '1.2.0',
    'extraction_date': '2025-09-03'
}

save_training_pipeline(rf_model, preprocessors, metadata, '/fs1/tor_project/model_foresttype/model/trained_rf_v1_2_0')
```

## Integration Checklist

- [ ] Dataset loaded and validated
- [ ] Training configuration defined
- [ ] Data converted to appropriate format
- [ ] Missing values handled
- [ ] Features selected (if needed)
- [ ] Train-test split created
- [ ] Model trained and validated
- [ ] Performance evaluated
- [ ] Model and pipeline saved
- [ ] Documentation updated

---

**Dataset Version**: 1.2.0  
**Last Updated**: September 3, 2025  
**Training Ready**: ✅ Complete with all 10 satellite indices
