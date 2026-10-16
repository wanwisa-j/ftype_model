"""
Main Pipeline for Forest Type Classification Dataset Creation
Orchestrates the entire data extraction and preprocessing workflow
"""

import os
import sys
import argparse
import logging
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '.'))

from feature_extractor import ForestTypeFeatureExtractor
from data_preprocessor import ForestTypeDataPreprocessor

def setup_logging(log_file: str = None):
    """Setup logging configuration"""
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    if log_file:
        logging.basicConfig(
            level=logging.INFO,
            format=log_format,
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
    else:
        logging.basicConfig(level=logging.INFO, format=log_format)

def main():
    parser = argparse.ArgumentParser(description='Forest Type Classification Dataset Pipeline')
    parser.add_argument('--label-file', required=True, 
                       help='Path to label data CSV file')
    parser.add_argument('--output-dir', required=True,
                       help='Output directory for processed data')
    parser.add_argument('--max-workers', type=int, default=4,
                       help='Number of parallel workers for feature extraction')
    parser.add_argument('--batch-size', type=int, default=100,
                       help='Batch size for processing points')
    parser.add_argument('--test-size', type=float, default=0.2,
                       help='Proportion of data to use for testing')
    parser.add_argument('--years-start', type=int, default=2019,
                       help='Start year for data collection')
    parser.add_argument('--years-end', type=int, default=2024,
                       help='End year for data collection')
    parser.add_argument('--skip-extraction', action='store_true',
                       help='Skip feature extraction and use existing features')
    parser.add_argument('--features-file', 
                       help='Path to existing features file (if skip-extraction)')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Setup logging
    log_file = os.path.join(args.output_dir, f'pipeline_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    setup_logging(log_file)
    logger = logging.getLogger(__name__)
    
    logger.info("Starting Forest Type Classification Dataset Pipeline")
    logger.info(f"Arguments: {vars(args)}")
    
    # Configuration
    datacube_config = {
        'mfl_cmf_2024': '/fs2/ftype_mflf/shared_data/datacube/2024/1.2.0/mfl_cmf_2024',
        'mfl_cmf_area_project_2025_chr_zone_1': '/fs2/ftype_mflf/shared_data/datacube/2025/1.2.0/mfl_cmf_area_project_2025_chr_zone_1',
        'mfl_cmf_area_project_2025_chr_zone_2': '/fs2/ftype_mflf/shared_data/datacube/2025/1.2.0/mfl_cmf_area_project_2025_chr_zone_2',
        'zone_1': '/fs2/ftype_mflf/shared_data/datacube/2025/1.2.0/zone_1',
        'zone_2': '/fs2/ftype_mflf/shared_data/datacube/2025/1.2.0/zone_2',
        'zone_3': '/fs2/ftype_mflf/shared_data/datacube/2025/1.2.0/zone_3',
        'zone_4': '/fs2/ftype_mflf/shared_data/datacube/2025/1.2.0/zone_4',
        'zone_5': '/fs2/ftype_mflf/shared_data/datacube/2025/1.2.0/zone_5',
    }
    
    bbox_areas_config = {
        'mfl_cmf_2024': '/fs3/datacenter/ftype_mflf/shared_data/boundary/mfl_cmf_2024.parquet',
        'mfl_cmf_area_project_2025_chr_zone_1': '/fs3/datacenter/ftype_mflf/shared_data/boundary/mfl_cmf_area_project_2025_chr_zone_1-fit_areas.parquet',
        'mfl_cmf_area_project_2025_chr_zone_2': '/fs3/datacenter/ftype_mflf/shared_data/boundary/mfl_cmf_area_project_2025_chr_zone_2-fit_areas.parquet',
        'zone_1': '/fs3/datacenter/ftype_mflf/shared_data/boundary/mfl_cmf_project_2025_phase2/split/zone_1.parquet',
        'zone_2': '/fs3/datacenter/ftype_mflf/shared_data/boundary/mfl_cmf_project_2025_phase2/split/zone_2.parquet',
        'zone_3': '/fs3/datacenter/ftype_mflf/shared_data/boundary/mfl_cmf_project_2025_phase2/split/zone_3.parquet',
        'zone_4': '/fs3/datacenter/ftype_mflf/shared_data/boundary/mfl_cmf_project_2025_phase2/split/zone_4.parquet',
        'zone_5': '/fs3/datacenter/ftype_mflf/shared_data/boundary/mfl_cmf_project_2025_phase2/split/zone_5.parquet',
    }
    
    time_series_features = ['lai', 'fapar', 'fcover', 'gvmi', 'ndvi', 'slavi', 'vh', 'vv']
    static_features = ['elevation', 'slope']
    years_range = (args.years_start, args.years_end)
    
    try:
        # Step 1: Feature Extraction
        if args.skip_extraction and args.features_file:
            logger.info(f"Loading existing features from {args.features_file}")
            if args.features_file.endswith('.pkl'):
                features_df = pd.read_pickle(args.features_file)
            else:
                features_df = pd.read_csv(args.features_file)
        else:
            logger.info("Starting feature extraction")
            
            # Initialize feature extractor
            extractor = ForestTypeFeatureExtractor(
                datacube_config=datacube_config,
                bbox_areas_config=bbox_areas_config,
                time_series_features=time_series_features,
                static_features=static_features,
                years_range=years_range
            )
            
            # Load and prepare label data
            label_gdf = extractor.load_label_data(args.label_file)
            logger.info(f"Loaded {len(label_gdf)} label points")
            
            # Assign zones to points
            label_gdf = extractor.assign_zones_to_points(label_gdf)
            logger.info(f"Assigned zones to {len(label_gdf)} points")
            
            # Extract features
            features_df = extractor.extract_dataset(
                label_gdf=label_gdf,
                output_dir=args.output_dir,
                max_workers=args.max_workers,
                batch_size=args.batch_size
            )
            
            if features_df.empty:
                logger.error("Feature extraction failed")
                return
                
            # Generate feature statistics
            stats_df = extractor.get_feature_statistics(features_df)
            stats_file = os.path.join(args.output_dir, "feature_statistics.csv")
            stats_df.to_csv(stats_file, index=False)
            logger.info(f"Feature statistics saved to {stats_file}")
        
        # Step 2: Data Preprocessing and Dataset Creation
        logger.info("Starting data preprocessing")
        
        # Initialize preprocessor
        preprocessor = ForestTypeDataPreprocessor(
            time_series_features=time_series_features,
            static_features=static_features,
            years_range=years_range,
            imputation_strategy='median',
            scaling_method='standard'
        )
        
        from sklearn.model_selection import train_test_split
        
        # Process both target types
        for target_col in ['ftype_mix', 'ftype_nonmix']:
            logger.info(f"Processing target: {target_col}")
            
            # Filter data with valid target values
            target_df = features_df.dropna(subset=[target_col]).copy()
            
            if len(target_df) == 0:
                logger.warning(f"No valid data for target {target_col}")
                continue
                
            logger.info(f"Dataset size for {target_col}: {len(target_df)}")
            logger.info(f"Target distribution: {target_df[target_col].value_counts().to_dict()}")
            
            # Split into train/test
            train_df, test_df = train_test_split(
                target_df, 
                test_size=args.test_size, 
                random_state=42, 
                stratify=target_df[target_col]
            )
            
            # Fit and transform training data
            train_data = preprocessor.fit_transform_train(train_df, target_col)
            
            # Transform test data
            test_data = preprocessor.transform_test(test_df, target_col)
            
            # Save preprocessed datasets
            dataset_info = {
                'train': train_data,
                'test': test_data,
                'config': {
                    'target_col': target_col,
                    'n_train_samples': len(train_df),
                    'n_test_samples': len(test_df),
                    'feature_extraction_config': {
                        'time_series_features': time_series_features,
                        'static_features': static_features,
                        'years_range': years_range,
                        'datacube_paths': list(datacube_config.values()),
                        'boundary_paths': list(bbox_areas_config.values())
                    }
                }
            }
            
            # Save complete dataset
            dataset_file = os.path.join(args.output_dir, f"ml_dataset_{target_col}.pkl")
            pd.to_pickle(dataset_info, dataset_file)
            logger.info(f"ML dataset for {target_col} saved to {dataset_file}")
            
            # Save human-readable summary
            summary_file = os.path.join(args.output_dir, f"dataset_summary_{target_col}.txt")
            with open(summary_file, 'w') as f:
                f.write(f"Dataset Summary for {target_col}\n")
                f.write("=" * 50 + "\n\n")
                f.write(f"Training samples: {train_data['X_tabular'].shape[0]}\n")
                f.write(f"Test samples: {test_data['X_tabular'].shape[0]}\n")
                f.write(f"Tabular features: {train_data['X_tabular'].shape[1]}\n")
                f.write(f"Sequential features: {train_data['X_sequential'].shape[1] if train_data['X_sequential'].size > 0 else 0}\n")
                f.write(f"Target classes: {train_data['target_classes']}\n\n")
                
                f.write("Tabular Feature Names:\n")
                for i, name in enumerate(train_data['feature_names_tabular']):
                    f.write(f"  {i:3d}: {name}\n")
                    
            logger.info(f"Dataset summary saved to {summary_file}")
        
        # Save preprocessors
        preprocessor_file = os.path.join(args.output_dir, "preprocessors.pkl")
        preprocessor.save_preprocessors(preprocessor_file)
        
        logger.info("Pipeline completed successfully!")
        logger.info(f"Results saved to: {args.output_dir}")
        
    except Exception as e:
        logger.error(f"Pipeline failed with error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise

if __name__ == "__main__":
    main()
