# Model Prediction Pipeline Documentation

## Overview

This document describes the prediction pipeline for the forest type classification model, focusing on the current model version 2.2.2. The prediction process applies trained models to new areas or time periods using the same datacube structure as training.

⚠️ **CRITICAL UPDATE (September 11, 2025):** Major pipeline fixes applied to resolve 99.9% class 2.0 prediction bias. See [MEMORY/Prediction_Pipeline_Mismatch_Issue.md](../MEMORY/Prediction_Pipeline_Mismatch_Issue.md) for full details.

## Prediction Pipeline Architecture

### Primary Tool
**Script**: `src/prediction_ftype_2_2_1.py` (UPDATED with critical fixes)

### Model Version History
- **Version 1.x**: Initial random forest implementation
- **Version 2.0**: Enhanced feature engineering
- **Version 2.1**: Improved temporal processing  
- **Version 2.2.0**: Baseline CNN model
- **Version 2.2.1**: Training script improvements
- **Version 2.2.2**: Current production version with hybrid CNN architecture

### Critical Pipeline Updates (v2.2.2)
- ✅ **Fixed scaling parameters**: Correct file path and version consistency
- ✅ **Fixed feature channel order**: Exact training-prediction alignment
- ✅ **Added label encoder**: Proper class name mapping (DDF, DEF, MDF)
- ✅ **Enhanced validation**: Pipeline consistency checking

## Input Requirements

### Datacube Structure
Prediction requires the same datacube format as training:
```
/fs2/ftype_mflf/shared_data/datacube/2025/1.2.0/
├── target_zone/
│   ├── fapar/
│   │   └── cube_biophys_fapar_2023_W4before_to_W16after_21bands.tif
│   ├── fcover/
│   ├── lai/
│   ├── gvmi/
│   ├── ndvi/
│   ├── slavi/
│   ├── vv/
│   ├── vh/
│   ├── elevation/
│   └── slope/
```

### Model Assets (v2.2.2)
```
model/mfl_v2_2_2/
├── hybrid_lstm_cnn_model-20250910-no-leakage.keras     # Trained model
├── config/
│   ├── scaling_params_v2_2_2_no_leakage-20250910.npz   # CRITICAL: Feature scaling parameters
│   └── label_encoder_v2_2_2.joblib                     # CRITICAL: Class mapping (DDF, DEF, MDF)
└── best_model.h5                                       # Alternative model format
```

⚠️ **CRITICAL REQUIREMENTS:**
- **Scaling parameters**: Mandatory for correct feature normalization
- **Label encoder**: Required for proper class name mapping
- **Version consistency**: All artifacts must be from same training run

## Prediction Workflow

### Step 1: Data Loading and Validation
```python
def load_prediction_data(zone_path):
    """Load and validate datacube for prediction"""
    
    required_features = [
        'fapar', 'fcover', 'lai', 'gvmi', 'ndvi', 
        'slavi', 'vv', 'vh', 'elevation', 'slope'
    ]
    
    datacube = {}
    for feature in required_features:
        file_path = zone_path / feature / f"cube_*_{feature}_*.tif"
        datacube[feature] = load_raster(file_path)
        
    validate_datacube_consistency(datacube)
    return datacube
```

### Step 2: Feature Extraction and Channel Construction
⚠️ **CRITICAL**: Channel order determines neural network input interpretation.

```python
def extract_prediction_features(timeseries_dicts, static_dicts, mask):
    """Extract and construct features - v2.2.2 method"""
    
    # Reshape and stack timeseries features
    timeseries_array, static_array = reshape_and_stack_features(
        timeseries_dicts, static_dicts, mask
    )
    
    # CRITICAL: Add VH/VV and VV/VH ratios exactly as in training
    vv = timeseries_array[:, :, 0]  # Channel 0: VV  
    vh = timeseries_array[:, :, 1]  # Channel 1: VH
    
    # Calculate ratios with division by zero protection
    denom = np.where(vv == 0, 1e-6, vv)
    vh_vv_ratio = vh / denom  # Channel 8
    
    denom_vh = np.where(vh == 0, 1e-6, vh) 
    vv_vh_ratio = vv / denom_vh  # Channel 9
    
    # Final timeseries: 10 channels total
    # [VV, VH, GVMI, NDVI, SLAVI, LAI, FAPAR, FCOVER, VH/VV, VV/VH]
    X_timeseries = np.concatenate([
        timeseries_array,
        vh_vv_ratio[:, :, np.newaxis], 
        vv_vh_ratio[:, :, np.newaxis]
    ], axis=-1)
    
    # Static features: [Slope, Elevation] 
    X_static = static_array
    
    return X_timeseries, X_static
```

### Step 3: Feature Scaling
⚠️ **CRITICAL**: Must use exact scaling parameters from training.

```python
def apply_scaling(X_timeseries, X_static, scaling_params):
    """Apply z-score scaling using training statistics - v2.2.2 method"""
    
    # Load scaling parameters (MANDATORY)
    timeseries_mean = scaling_params['timeseries_mean'] 
    timeseries_std = scaling_params['timeseries_std']
    static_mean = scaling_params['static_mean']
    static_std = scaling_params['static_std']
    
    # Apply timeseries scaling (per channel across samples and timesteps)
    X_timeseries_scaled = (X_timeseries - timeseries_mean) / timeseries_std
    
    # Apply static scaling (per feature across samples)
    X_static_scaled = (X_static - static_mean) / static_std
    
    return X_timeseries_scaled, X_static_scaled
    
def load_scaling_parameters():
    """Load scaling parameters - MANDATORY for v2.2.2"""
    config_path = 'model/mfl_v2_2_2/config/scaling_params_v2_2_2_no_leakage-20250910.npz'
    
    if not os.path.exists(config_path):
        raise FileNotFoundError(
            f"CRITICAL ERROR: Scaling parameters not found at: {config_path}\n"
            f"For model version 2.x.x, scaling parameters are mandatory."
        )
    
    with np.load(config_path) as data:
        return {
            'static_mean': data['static_mean'],
            'static_std': data['static_std'], 
            'timeseries_mean': data['timeseries_mean'],
            'timeseries_std': data['timeseries_std']
        }
```

### Step 4: Model Prediction and Class Mapping 
⚠️ **CRITICAL**: Label encoder required for proper class names.

```python
def predict_forest_type(model, X_timeseries_scaled, X_static_scaled, label_encoder):
    """Generate predictions - v2.2.2 method"""
    
    # Model expects two inputs: [timeseries, static]
    probabilities = model.predict([X_timeseries_scaled, X_static_scaled])
    
    # Get predicted class indices 
    predicted_indices = np.argmax(probabilities, axis=1)
    
    # Map indices to class names using label encoder
    if label_encoder is not None:
        predicted_classes = label_encoder.inverse_transform(predicted_indices)
        # Classes: ['DDF', 'DEF', 'MDF']
    else:
        predicted_classes = predicted_indices  # Raw indices [0, 1, 2]
    
    return predicted_classes, probabilities
    
def load_label_encoder():
    """Load label encoder - MANDATORY for proper class mapping"""
    le_path = 'model/mfl_v2_2_2/config/label_encoder_v2_2_2.joblib'
    
    if not os.path.exists(le_path):
        raise FileNotFoundError(
            f"CRITICAL ERROR: Label encoder not found at: {le_path}\n" 
            f"Required for mapping predictions to class names."
        )
    
    return joblib.load(le_path)
```

### Step 5: Spatial Output Generation
```python
def create_prediction_maps(predictions, probabilities, reference_raster):
    """Create spatial prediction outputs"""
    
    # Prediction map
    prediction_map = reshape_to_raster(predictions, reference_raster.shape)
    
    # Confidence map
    confidence_map = reshape_to_raster(confidence, reference_raster.shape)
    
    # Probability maps (one per class)
    probability_maps = {}
    for i, class_name in enumerate(class_labels):
        prob_map = reshape_to_raster(probabilities[:, i], reference_raster.shape)
        probability_maps[class_name] = prob_map
        
    return prediction_map, confidence_map, probability_maps
```

## Batch Processing Capabilities

### Large Area Processing
```python
def process_large_area(zone_path, output_path, chunk_size=1000):
    """Process large areas in chunks to manage memory"""
    
    # Load model assets
    model = load_model()
    scaler = load_scaler()
    selector = load_feature_selector()
    
    # Process in spatial chunks
    for chunk_bounds in generate_spatial_chunks(zone_path, chunk_size):
        
        # Load chunk data
        chunk_datacube = load_datacube_chunk(zone_path, chunk_bounds)
        
        # Extract features
        chunk_features = extract_prediction_features(chunk_datacube)
        
        # Preprocess
        processed_features = preprocess_features(chunk_features, scaler, selector)
        
        # Predict
        predictions, probabilities, confidence = predict_forest_type(
            model, processed_features
        )
        
        # Save chunk results
        save_chunk_results(predictions, probabilities, confidence, 
                          chunk_bounds, output_path)
```

### Multi-Zone Processing
```bash
# Example batch processing script
#!/bin/bash

zones=(
    "mfl_cmf_area_project_2025_chr_zone_1"
    "mfl_cmf_area_project_2025_chr_zone_2" 
    "zone_1"
)

for zone in "${zones[@]}"; do
    echo "Processing zone: $zone"
    python src/prediction_ftype_2_2_0.py \
        --zone $zone \
        --datacube-path /fs2/ftype_mflf/shared_data/datacube/2025/1.2.0/ \
        --model-path model/forest_type_classifier_v2_2_0.pkl \
        --output-path results/predictions_v2_2_0/
done
```

## Output Products

### Prediction Maps
```
results/predictions_v2_2_0/
├── mfl_cmf_area_project_2025_chr_zone_1/
│   ├── forest_type_prediction.tif          # Class predictions
│   ├── prediction_confidence.tif           # Confidence scores
│   ├── probabilities/
│   │   ├── prob_primary_forest.tif
│   │   ├── prob_secondary_forest.tif
│   │   ├── prob_plantation.tif
│   │   ├── prob_mixed_forest.tif
│   │   └── prob_other.tif
│   ├── metadata.json                       # Prediction metadata
│   └── quality_report.json                 # Quality assessment
```

### Output Specifications
- **Format**: GeoTIFF with LZW compression
- **Data Type**: 
  - Predictions: UInt8 (class codes)
  - Probabilities: Float32 (0-1 range)
  - Confidence: Float32 (0-1 range)
- **NoData Value**: 255 for predictions, -9999 for probabilities
- **Spatial Reference**: Same as input datacube

## Quality Assessment

### Prediction Quality Metrics
```python
def assess_prediction_quality(predictions, probabilities, confidence):
    """Generate quality metrics for predictions"""
    
    quality_metrics = {
        'coverage': calculate_valid_pixel_percentage(predictions),
        'confidence_distribution': analyze_confidence_distribution(confidence),
        'class_distribution': analyze_class_distribution(predictions),
        'spatial_continuity': assess_spatial_patterns(predictions),
        'uncertainty_patterns': analyze_uncertainty_patterns(confidence)
    }
    
    return quality_metrics
```

### Validation Against Ground Truth
```python
def validate_predictions(predictions, ground_truth_points):
    """Validate predictions against available ground truth"""
    
    # Extract predictions at ground truth locations
    predicted_values = extract_values_at_points(predictions, ground_truth_points)
    
    # Calculate accuracy metrics
    accuracy = accuracy_score(ground_truth_points.labels, predicted_values)
    confusion_matrix = confusion_matrix(ground_truth_points.labels, predicted_values)
    classification_report = classification_report(ground_truth_points.labels, predicted_values)
    
    return {
        'overall_accuracy': accuracy,
        'confusion_matrix': confusion_matrix,
        'classification_report': classification_report
    }
```

## Performance Optimization

### Memory Management
- **Chunk Processing**: Process large areas in spatial chunks
- **Lazy Loading**: Load data only when needed
- **Memory Monitoring**: Track memory usage during processing

### Parallel Processing
```python
from multiprocessing import Pool
import concurrent.futures

def parallel_prediction(zones, n_workers=4):
    """Process multiple zones in parallel"""
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = []
        for zone in zones:
            future = executor.submit(process_zone, zone)
            futures.append(future)
            
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            print(f"Completed zone: {result['zone']}")
```

### I/O Optimization
- **GDAL Optimization**: Use GDAL configuration for faster I/O
- **Compression**: Use appropriate compression for output files
- **Tiling**: Internal tiling for efficient access patterns

## Model Version Management

### Version Compatibility
```python
def check_model_compatibility(model_version, datacube_version):
    """Ensure model and data compatibility"""
    
    compatibility_matrix = {
        'v2.2.0': ['1.2.0', '1.1.0'],
        'v2.1.0': ['1.1.0', '1.0.0'],
        'v2.0.0': ['1.0.0']
    }
    
    if datacube_version not in compatibility_matrix[model_version]:
        raise CompatibilityError(f"Model {model_version} not compatible with datacube {datacube_version}")
```

### Model Deployment
```bash
# Model deployment script
./deploy_model.sh \
    --model-version v2.2.0 \
    --target-environment production \
    --validation-required true
```

## Monitoring and Logging

### Processing Logs
```python
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('prediction.log'),
        logging.StreamHandler()
    ]
)

# Log prediction progress
logger.info(f"Starting prediction for zone: {zone}")
logger.info(f"Model version: {model_version}")
logger.info(f"Processing {n_pixels} pixels")
logger.info(f"Prediction completed in {processing_time:.2f} seconds")
```

### Performance Monitoring
- **Processing Speed**: Pixels per second
- **Memory Usage**: Peak memory consumption
- **I/O Throughput**: Data read/write rates
- **Error Rates**: Failed pixel percentage

## Error Handling and Recovery

### Common Issues
```python
def handle_prediction_errors():
    """Handle common prediction errors"""
    
    try:
        predictions = run_prediction()
    except MemoryError:
        # Reduce chunk size and retry
        predictions = run_prediction_small_chunks()
    except FileNotFoundError:
        # Check for missing datacube files
        validate_input_files()
        raise
    except ModelCompatibilityError:
        # Load compatible model version
        model = load_compatible_model()
        predictions = run_prediction(model)
```

### Recovery Procedures
- **Checkpoint System**: Save intermediate results
- **Resume Capability**: Continue from last saved chunk
- **Validation**: Verify output completeness

## ⚠️ Critical Pipeline Issues and Solutions

### Issue: 99.9% Prediction Bias Toward Single Class
**Symptoms:**
- Almost all pixels predicted as class 2 (MDF)
- Predictions don't vary across different terrain types
- Even training locations show same class bias

**Root Causes & Solutions:**
1. **Wrong Scaling Parameters**
   - Check: `scaling_params_v2_2_2_no_leakage-20250910.npz` exists and loads correctly
   - Fix: Update file path in prediction script to match actual filename

2. **Feature Channel Order Mismatch** 
   - Check: Timeseries features loaded in exact training order: `['vv', 'vh', 'gvmi', 'ndvi', 'slavi', 'lai', 'fapar', 'fcover']`
   - Check: Static features in training order: `['slope', 'elevation']` 
   - Fix: Ensure prediction script matches training script feature order exactly

3. **Missing Label Encoder**
   - Check: `label_encoder_v2_2_2.joblib` exists and loads correctly
   - Fix: Add label encoder loading to map predictions to class names

**Validation Steps:**
```python
# Test on known training points first
test_points = load_training_locations()
predictions = predict_at_points(test_points)
validate_prediction_diversity(predictions)
```

### Issue: FileNotFoundError for Model Artifacts
**Solution:**
- Verify all required files exist in `model/mfl_v2_2_2/config/`
- Check version consistency across all artifacts
- Use absolute paths to avoid working directory issues

### Issue: Memory Errors During Large Area Processing
**Solutions:**
- Reduce batch size in prediction calls
- Process smaller spatial chunks
- Monitor memory usage and adjust chunk size dynamically

### Pipeline Validation Checklist
Before running predictions, verify:
- ✅ Correct model version (v2.2.2)
- ✅ Scaling parameters file exists and matches model version
- ✅ Label encoder file exists and loads correctly
- ✅ Feature order matches training script exactly
- ✅ Test predictions at known training locations
- ✅ Check prediction class distribution for reasonable diversity

## Related Documentation
- [Dataset Extraction](./DATASET_EXTRACTION.md)
- [Model Training Guide](./MODEL_TRAINING.md)
- [Data Processing and Alignment](./DATA_PROCESSING_ALIGNMENT.md)
