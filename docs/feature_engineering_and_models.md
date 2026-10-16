# Feature Engineering and Model Architecture Recommendations

## Feature Engineering Strategy

### Time-Series Feature Engineering

#### 1. Aggregated Statistics (Implemented)
```python
# Per-feature aggregations across time and years
aggregated_features = {
    f'{feature}_mean': np.nanmean(time_series_data),
    f'{feature}_std': np.nanstd(time_series_data),
    f'{feature}_min': np.nanmin(time_series_data),
    f'{feature}_max': np.nanmax(time_series_data),
    f'{feature}_median': np.nanmedian(time_series_data),
    f'{feature}_q25': np.nanpercentile(time_series_data, 25),
    f'{feature}_q75': np.nanpercentile(time_series_data, 75),
}
```

#### 2. Seasonal Patterns (Implemented)
```python
# Seasonal feature extraction
seasonal_features = {
    f'{feature}_peak_week': np.nanargmax(weekly_average),
    f'{feature}_valley_week': np.nanargmin(weekly_average),
    f'{feature}_seasonal_range': np.nanmax(weekly_average) - np.nanmin(weekly_average)
}
```

#### 3. Advanced Time-Series Features (Recommended)
```python
# Trend analysis
def extract_trend_features(time_series):
    """Extract trend-based features"""
    from scipy import stats
    
    # Linear trend
    slope, intercept, r_value, p_value, std_err = stats.linregress(range(len(time_series)), time_series)
    
    # Change points
    change_points = detect_change_points(time_series)
    
    return {
        'trend_slope': slope,
        'trend_r_squared': r_value**2,
        'trend_significance': p_value,
        'n_change_points': len(change_points),
        'max_change_magnitude': max(abs(np.diff(time_series))) if len(time_series) > 1 else 0
    }

# Frequency domain features
def extract_spectral_features(time_series):
    """Extract frequency domain features"""
    from scipy.fft import fft, fftfreq
    
    # FFT analysis
    fft_values = fft(time_series)
    frequencies = fftfreq(len(time_series))
    
    # Dominant frequency
    dominant_freq_idx = np.argmax(np.abs(fft_values[1:len(time_series)//2])) + 1
    dominant_frequency = frequencies[dominant_freq_idx]
    
    # Spectral centroid
    spectral_centroid = np.sum(frequencies * np.abs(fft_values)) / np.sum(np.abs(fft_values))
    
    return {
        'dominant_frequency': dominant_frequency,
        'spectral_centroid': spectral_centroid,
        'spectral_rolloff': calculate_spectral_rolloff(fft_values, frequencies),
        'spectral_flux': calculate_spectral_flux(fft_values)
    }
```

### Multi-Feature Interactions

#### 1. Vegetation Indices Combinations
```python
# Enhanced vegetation analysis
def calculate_vegetation_health_index(ndvi, lai, fapar):
    """Composite vegetation health indicator"""
    # Normalized combination of vegetation features
    normalized_ndvi = (ndvi - np.nanmin(ndvi)) / (np.nanmax(ndvi) - np.nanmin(ndvi))
    normalized_lai = (lai - np.nanmin(lai)) / (np.nanmax(lai) - np.nanmin(lai))
    normalized_fapar = (fapar - np.nanmin(fapar)) / (np.nanmax(fapar) - np.nanmin(fapar))
    
    # Weighted combination
    vegetation_health = 0.4 * normalized_ndvi + 0.35 * normalized_lai + 0.25 * normalized_fapar
    return vegetation_health

# Water stress indicators
def calculate_water_stress_features(gvmi, ndvi, vh, vv):
    """Multi-sensor water stress analysis"""
    # Optical-SAR water content index
    water_content_index = gvmi * (vh / (vh + vv))
    
    # Vegetation water stress
    water_stress = (1 - gvmi) * ndvi  # High vegetation with low water content
    
    return {
        'water_content_index': water_content_index,
        'water_stress_indicator': water_stress,
        'sar_vegetation_ratio': (vh - vv) / (vh + vv) * ndvi
    }
```

#### 2. Multi-Temporal Feature Engineering
```python
def extract_phenological_features(time_series_dict):
    """Extract phenological patterns across years"""
    phenology_features = {}
    
    for feature_name, multi_year_data in time_series_dict.items():
        # multi_year_data shape: (n_years, 21_weeks)
        
        # Year-to-year consistency
        yearly_peaks = [np.nanargmax(year_data) for year_data in multi_year_data]
        yearly_means = [np.nanmean(year_data) for year_data in multi_year_data]
        
        phenology_features.update({
            f'{feature_name}_peak_consistency': np.std(yearly_peaks),
            f'{feature_name}_interannual_stability': np.std(yearly_means),
            f'{feature_name}_early_season_slope': calculate_early_season_slope(multi_year_data),
            f'{feature_name}_late_season_slope': calculate_late_season_slope(multi_year_data),
            f'{feature_name}_growing_season_length': estimate_growing_season_length(multi_year_data)
        })
    
    return phenology_features
```

### Topographic Feature Enhancement

```python
def extract_enhanced_topographic_features(elevation, slope, coordinates):
    """Calculate derived topographic features"""
    # Terrain ruggedness
    terrain_ruggedness = calculate_terrain_ruggedness_index(elevation, coordinates)
    
    # Aspect (if slope is available)
    aspect = calculate_aspect(elevation, coordinates)
    
    # Topographic position index
    tpi = calculate_topographic_position_index(elevation, coordinates)
    
    return {
        'terrain_ruggedness': terrain_ruggedness,
        'aspect': aspect,
        'topographic_position': tpi,
        'slope_aspect_interaction': slope * np.cos(aspect),  # North-facing slope effect
        'elevation_normalized': (elevation - np.mean(elevation)) / np.std(elevation)
    }
```

## Model Architecture Recommendations

### 1. Hierarchical Model Architecture

```python
class HierarchicalForestClassifier:
    """Two-stage classification: broad categories then specific types"""
    
    def __init__(self):
        # Stage 1: Broad forest type (deciduous, evergreen, mixed)
        self.broad_classifier = RandomForestClassifier(n_estimators=200)
        
        # Stage 2: Specific subtypes within each broad category
        self.specific_classifiers = {
            'deciduous': RandomForestClassifier(n_estimators=150),
            'evergreen': RandomForestClassifier(n_estimators=150),
            'mixed': RandomForestClassifier(n_estimators=150)
        }
    
    def fit(self, X, y_broad, y_specific):
        # Train broad classifier
        self.broad_classifier.fit(X, y_broad)
        
        # Train specific classifiers for each broad category
        for category in ['deciduous', 'evergreen', 'mixed']:
            mask = (y_broad == category)
            if np.sum(mask) > 0:
                self.specific_classifiers[category].fit(X[mask], y_specific[mask])
    
    def predict(self, X):
        # Predict broad categories
        broad_predictions = self.broad_classifier.predict(X)
        
        # Predict specific types
        specific_predictions = np.zeros(len(X), dtype=object)
        for i, broad_pred in enumerate(broad_predictions):
            specific_pred = self.specific_classifiers[broad_pred].predict(X[i:i+1])
            specific_predictions[i] = specific_pred[0]
            
        return broad_predictions, specific_predictions
```

### 2. Multi-Input Deep Learning Architecture

```python
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, LSTM, Conv1D, Concatenate, Dropout
from tensorflow.keras.models import Model

def create_multi_input_forest_classifier(
    tabular_features_dim,
    sequential_features_dim,
    n_timesteps,
    n_classes
):
    """Multi-input model for tabular and sequential features"""
    
    # Tabular features input
    tabular_input = Input(shape=(tabular_features_dim,), name='tabular_input')
    tabular_dense = Dense(128, activation='relu')(tabular_input)
    tabular_dense = Dropout(0.3)(tabular_dense)
    tabular_dense = Dense(64, activation='relu')(tabular_dense)
    
    # Sequential features input
    sequential_input = Input(shape=(n_timesteps, sequential_features_dim), name='sequential_input')
    
    # CNN branch for local patterns
    cnn_branch = Conv1D(64, 3, activation='relu')(sequential_input)
    cnn_branch = Conv1D(32, 3, activation='relu')(cnn_branch)
    cnn_branch = tf.keras.layers.GlobalMaxPooling1D()(cnn_branch)
    
    # LSTM branch for temporal patterns
    lstm_branch = LSTM(64, return_sequences=True)(sequential_input)
    lstm_branch = LSTM(32)(lstm_branch)
    
    # Combine CNN and LSTM
    sequential_combined = Concatenate()([cnn_branch, lstm_branch])
    sequential_dense = Dense(64, activation='relu')(sequential_combined)
    
    # Combine all inputs
    combined = Concatenate()([tabular_dense, sequential_dense])
    combined = Dense(128, activation='relu')(combined)
    combined = Dropout(0.4)(combined)
    combined = Dense(64, activation='relu')(combined)
    
    # Output layer
    output = Dense(n_classes, activation='softmax', name='classification')(combined)
    
    model = Model(inputs=[tabular_input, sequential_input], outputs=output)
    
    return model

# Usage
model = create_multi_input_forest_classifier(
    tabular_features_dim=X_train_tabular.shape[1],
    sequential_features_dim=X_train_sequential.shape[1] // 21,
    n_timesteps=21,
    n_classes=len(target_classes)
)

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
```

### 3. Ensemble Methods

```python
class ForestTypeEnsemble:
    """Ensemble of different model types"""
    
    def __init__(self):
        # Different model types
        self.models = {
            'random_forest': RandomForestClassifier(n_estimators=200, random_state=42),
            'gradient_boosting': GradientBoostingClassifier(n_estimators=150, random_state=42),
            'svm': SVC(kernel='rbf', probability=True, random_state=42),
            'neural_network': MLPClassifier(hidden_layer_sizes=(200, 100, 50), random_state=42)
        }
        
        # Meta-learner for ensemble
        self.meta_learner = LogisticRegression(random_state=42)
        
    def fit(self, X_train, y_train, X_val, y_val):
        # Train base models
        base_predictions = np.zeros((len(X_val), len(self.models)))
        
        for i, (name, model) in enumerate(self.models.items()):
            print(f"Training {name}...")
            model.fit(X_train, y_train)
            base_predictions[:, i] = model.predict_proba(X_val)[:, 1]  # Assuming binary classification
        
        # Train meta-learner
        self.meta_learner.fit(base_predictions, y_val)
        
    def predict(self, X_test):
        # Get base model predictions
        base_predictions = np.zeros((len(X_test), len(self.models)))
        
        for i, (name, model) in enumerate(self.models.items()):
            base_predictions[:, i] = model.predict_proba(X_test)[:, 1]
        
        # Meta-learner final prediction
        final_predictions = self.meta_learner.predict(base_predictions)
        
        return final_predictions
```

## Model Selection Strategy

### Cross-Validation for Forest Data

```python
from sklearn.model_selection import GroupKFold, StratifiedKFold
import numpy as np

def spatial_cross_validation(X, y, zones, n_splits=5):
    """Spatial cross-validation to avoid data leakage"""
    # Group by zones to ensure spatial independence
    group_kfold = GroupKFold(n_splits=n_splits)
    
    cv_scores = []
    for train_idx, val_idx in group_kfold.split(X, y, groups=zones):
        X_train_cv, X_val_cv = X[train_idx], X[val_idx]
        y_train_cv, y_val_cv = y[train_idx], y[val_idx]
        
        # Train and evaluate model
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train_cv, y_train_cv)
        score = model.score(X_val_cv, y_val_cv)
        cv_scores.append(score)
    
    return np.mean(cv_scores), np.std(cv_scores)

def temporal_cross_validation(X, y, years, n_splits=3):
    """Temporal cross-validation for time-series data"""
    unique_years = sorted(np.unique(years))
    
    cv_scores = []
    for i in range(len(unique_years) - n_splits + 1):
        train_years = unique_years[i:i+n_splits-1]
        val_years = [unique_years[i+n_splits-1]]
        
        train_mask = np.isin(years, train_years)
        val_mask = np.isin(years, val_years)
        
        X_train_cv, X_val_cv = X[train_mask], X[val_mask]
        y_train_cv, y_val_cv = y[train_mask], y[val_mask]
        
        if len(X_train_cv) > 0 and len(X_val_cv) > 0:
            model = RandomForestClassifier(n_estimators=100, random_state=42)
            model.fit(X_train_cv, y_train_cv)
            score = model.score(X_val_cv, y_val_cv)
            cv_scores.append(score)
    
    return np.mean(cv_scores), np.std(cv_scores)
```

### Hyperparameter Optimization

```python
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint, uniform

def optimize_forest_classifier(X, y, cv_strategy='stratified'):
    """Hyperparameter optimization for forest classification"""
    
    # Define parameter distributions
    param_distributions = {
        'n_estimators': randint(100, 500),
        'max_depth': randint(5, 30),
        'min_samples_split': randint(2, 20),
        'min_samples_leaf': randint(1, 10),
        'max_features': ['sqrt', 'log2', 0.3, 0.5, 0.7],
        'bootstrap': [True, False],
        'class_weight': ['balanced', None]
    }
    
    # Base model
    rf = RandomForestClassifier(random_state=42)
    
    # Cross-validation strategy
    if cv_strategy == 'stratified':
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    elif cv_strategy == 'group':
        cv = GroupKFold(n_splits=5)
    
    # Randomized search
    random_search = RandomizedSearchCV(
        estimator=rf,
        param_distributions=param_distributions,
        n_iter=100,
        cv=cv,
        scoring='f1_macro',
        n_jobs=-1,
        random_state=42,
        verbose=1
    )
    
    random_search.fit(X, y)
    
    return random_search.best_estimator_, random_search.best_params_
```

## Evaluation Metrics for Forest Classification

```python
def comprehensive_evaluation(y_true, y_pred, y_pred_proba, class_names):
    """Comprehensive evaluation for forest type classification"""
    from sklearn.metrics import (
        classification_report, confusion_matrix, 
        roc_auc_score, average_precision_score,
        balanced_accuracy_score
    )
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    # Basic metrics
    print("Classification Report:")
    print(classification_report(y_true, y_pred, target_names=class_names))
    
    # Balanced accuracy (important for imbalanced classes)
    balanced_acc = balanced_accuracy_score(y_true, y_pred)
    print(f"\nBalanced Accuracy: {balanced_acc:.4f}")
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()
    
    # ROC-AUC for multi-class
    if len(class_names) > 2:
        roc_auc = roc_auc_score(y_true, y_pred_proba, multi_class='ovr', average='macro')
    else:
        roc_auc = roc_auc_score(y_true, y_pred_proba[:, 1])
    
    print(f"ROC-AUC Score: {roc_auc:.4f}")
    
    # Feature importance analysis
    if hasattr(model, 'feature_importances_'):
        feature_importance_analysis(model.feature_importances_, feature_names)
    
    return {
        'balanced_accuracy': balanced_acc,
        'roc_auc': roc_auc,
        'confusion_matrix': cm
    }

def feature_importance_analysis(importances, feature_names, top_n=20):
    """Analyze and visualize feature importance"""
    indices = np.argsort(importances)[::-1][:top_n]
    
    plt.figure(figsize=(12, 8))
    plt.bar(range(top_n), importances[indices])
    plt.xticks(range(top_n), [feature_names[i] for i in indices], rotation=45, ha='right')
    plt.title(f'Top {top_n} Feature Importances')
    plt.tight_layout()
    plt.show()
    
    # Group by feature type
    feature_groups = {
        'time_series': [],
        'aggregated': [],
        'static': [],
        'seasonal': []
    }
    
    for i in indices[:top_n]:
        feature_name = feature_names[i]
        importance = importances[i]
        
        if '_20' in feature_name:  # Time series
            feature_groups['time_series'].append((feature_name, importance))
        elif any(stat in feature_name for stat in ['mean', 'std', 'min', 'max']):
            feature_groups['aggregated'].append((feature_name, importance))
        elif any(static in feature_name for static in ['elevation', 'slope']):
            feature_groups['static'].append((feature_name, importance))
        elif any(seasonal in feature_name for seasonal in ['peak', 'valley', 'range']):
            feature_groups['seasonal'].append((feature_name, importance))
    
    print("\nFeature Importance by Category:")
    for category, features in feature_groups.items():
        if features:
            total_importance = sum(imp for _, imp in features)
            print(f"{category.title()}: {total_importance:.4f} ({len(features)} features)")
```

This comprehensive feature engineering and model architecture guide provides:

1. **Advanced Feature Engineering**: Trend analysis, spectral features, phenological patterns
2. **Multi-Input Architectures**: Combining tabular and sequential data effectively
3. **Ensemble Methods**: Multiple model types with meta-learning
4. **Proper Validation**: Spatial and temporal cross-validation strategies
5. **Comprehensive Evaluation**: Metrics appropriate for forest classification

The approach ensures robust model development while capturing the complex temporal and spatial patterns in forest type data.
