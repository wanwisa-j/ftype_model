# Model Prediction Pipeline Documentation

## Overview

This document describes the prediction pipeline for the forest type classification model, focusing on the current model version 2.2.0. The prediction process applies trained models to new areas or time periods using the same datacube structure as training.

## Prediction Pipeline Architecture

### Primary Tool
**Script**: `src/prediction_ftype_2_2_0.py`

### Model Version History
- **Version 1.x**: Initial random forest implementation
- **Version 2.0**: Enhanced feature engineering
- **Version 2.1**: Improved temporal processing
- **Version 2.2.0**: Current production version with optimized performance

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

### Model Assets
```
model/
├── forest_type_classifier_v2_2_0.pkl  # Trained model
├── feature_scaler_v2_2_0.pkl          # Feature scaling parameters
├── feature_selector_v2_2_0.pkl        # Feature selection
├── model_metadata_v2_2_0.json         # Model configuration
└── class_labels_v2_2_0.json           # Class mapping
```

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

### Step 2: Feature Extraction
```python
def extract_prediction_features(datacube):
    """Extract same features as training"""
    
    features = {}
    
    # Temporal features (statistical)
    for feature_name, data in datacube.items():
        if data.shape[0] == 21:  # Time series features
            features.update(extract_temporal_statistics(data, feature_name))
            features.update(extract_phenological_features(data, feature_name))
            features.update(extract_trend_features(data, feature_name))
        else:  # Static features
            features[feature_name] = data
            
    return features
```

### Step 3: Preprocessing
```python
def preprocess_features(features, scaler, selector):
    """Apply same preprocessing as training"""
    
    # Convert to feature matrix
    feature_matrix = create_feature_matrix(features)
    
    # Handle missing values
    feature_matrix = handle_missing_values(feature_matrix)
    
    # Apply scaling
    feature_matrix_scaled = scaler.transform(feature_matrix)
    
    # Apply feature selection
    feature_matrix_selected = selector.transform(feature_matrix_scaled)
    
    return feature_matrix_selected
```

### Step 4: Model Prediction
```python
def predict_forest_type(model, features):
    """Generate predictions"""
    
    # Probability predictions
    probabilities = model.predict_proba(features)
    
    # Class predictions
    predictions = model.predict(features)
    
    # Confidence scores
    confidence = np.max(probabilities, axis=1)
    
    return predictions, probabilities, confidence
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
    "mfl_cmf_area_project_2025_chr_zone_3"
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

## Related Documentation
- [Dataset Extraction](./DATASET_EXTRACTION.md)
- [Model Training Guide](./MODEL_TRAINING.md)
- [Data Processing and Alignment](./DATA_PROCESSING_ALIGNMENT.md)
