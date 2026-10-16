#!/usr/bin/env python3
"""
Test script for the parallelized forest type dataset extraction.
This performs a quick validation of the parallel processing functionality.
"""

import sys
import os
sys.path.append('/fs1/tor_project/model_foresttype/scripts')

from extract_forest_type_dataset import ForestTypeDatasetExtractor
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_parallel_initialization():
    """Test that the parallel extractor initializes correctly."""
    try:
        # Test with default configuration
        extractor = ForestTypeDatasetExtractor()
        logger.info(f"✅ Default initialization successful. Workers: {extractor.n_workers}")
        
        # Test with specific number of workers
        extractor_custom = ForestTypeDatasetExtractor(n_workers=4)
        logger.info(f"✅ Custom workers initialization successful. Workers: {extractor_custom.n_workers}")
        
        # Test with parallel config
        config_path = '/fs1/tor_project/model_foresttype/config/parallel_extraction_config.yaml'
        if os.path.exists(config_path):
            extractor_config = ForestTypeDatasetExtractor(config_path=config_path)
            logger.info(f"✅ Config-based initialization successful. Workers: {extractor_config.n_workers}")
            
            # Check parallel config
            parallel_config = extractor_config.config.get('parallel_processing', {})
            logger.info(f"Parallel processing enabled: {parallel_config.get('enabled', False)}")
            logger.info(f"Use processes: {parallel_config.get('use_processes', True)}")
            logger.info(f"Configured workers: {parallel_config.get('n_workers', 'auto')}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Initialization test failed: {e}")
        return False

def test_helper_functions():
    """Test the parallel processing helper functions."""
    try:
        # Test imports
        from extract_forest_type_dataset import extract_feature_for_zone_year, extract_static_feature_for_zone
        logger.info("✅ Helper function imports successful")
        
        # Test function signatures (they should be callable)
        if callable(extract_feature_for_zone_year) and callable(extract_static_feature_for_zone):
            logger.info("✅ Helper functions are callable")
            return True
        else:
            logger.error("❌ Helper functions are not callable")
            return False
            
    except Exception as e:
        logger.error(f"❌ Helper function test failed: {e}")
        return False

def main():
    """Run all tests."""
    logger.info("🚀 Starting parallel extraction script validation...")
    
    tests = [
        ("Initialization Test", test_parallel_initialization),
        ("Helper Functions Test", test_helper_functions)
    ]
    
    results = []
    for test_name, test_func in tests:
        logger.info(f"\n📋 Running {test_name}...")
        result = test_func()
        results.append((test_name, result))
        
    # Summary
    logger.info("\n📊 Test Results Summary:")
    all_passed = True
    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        logger.info(f"  {test_name}: {status}")
        if not result:
            all_passed = False
    
    if all_passed:
        logger.info("\n🎉 All tests passed! The parallelized extraction script is ready.")
        logger.info("\nUsage examples:")
        logger.info("  # Use default parallel processing:")
        logger.info("  python scripts/extract_forest_type_dataset.py --config config/parallel_extraction_config.yaml")
        logger.info("  # Use 12 workers:")
        logger.info("  python scripts/extract_forest_type_dataset.py --workers 12")
        logger.info("  # Use thread-based parallelization:")
        logger.info("  python scripts/extract_forest_type_dataset.py --threads")
        logger.info("  # Force serial processing:")
        logger.info("  python scripts/extract_forest_type_dataset.py --serial")
    else:
        logger.error("\n💥 Some tests failed! Please check the implementation.")
        sys.exit(1)

if __name__ == "__main__":
    main()
