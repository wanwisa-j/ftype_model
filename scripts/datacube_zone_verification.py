#!/usr/bin/env python3
"""
Datacube Zone Verification Script
================================

Reusable verification script for individual zone directories in the forest type model pipeline.
This script verifies completeness, quality, and consistency of datacube features for a single zone.

Usage:
    python datacube_zone_verification.py <zone_directory> [options]

Example:
    python datacube_zone_verification.py /fs2/ftype_mflf/shared_data/datacube/2025/1.2.0/zone_1
    python datacube_zone_verification.py /fs2/ftype_mflf/shared_data/datacube/2025/1.2.0/zone_1 --output ./verification_output

Features verified:
- Directory structure and feature completeness
- Geometric consistency (CRS, transform, shape)
- Temporal dimension (21 bands for temporal features, 1 for static)
- Data quality (NoData patterns, value ranges, interpolation quality)
- Cross-feature alignment

Author: Generated for Forest Type Model Pipeline v1.2.0
Date: 2025-09-10
"""

import os
import sys
import rasterio
import numpy as np
import pandas as pd
from pathlib import Path
import logging
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import json
from typing import Dict, List, Optional, Tuple, Any
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ZoneDatacubeVerifier:
    """Comprehensive verification for a single zone datacube."""
    
    def __init__(self, zone_dir: str, output_dir: str = None):
        """
        Initialize the zone datacube verifier.
        
        Args:
            zone_dir: Path to zone datacube directory
            output_dir: Output directory for reports (default: zone_dir/verification_output)
        """
        self.zone_dir = Path(zone_dir)
        self.zone_name = self.zone_dir.name
        
        if output_dir:
            self.output_dir = Path(output_dir)
        else:
            self.output_dir = self.zone_dir / "verification_output"
        
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        # Expected features by type
        self.biophysical_features = ['fapar', 'fcover', 'lai']
        self.optical_features = ['gvmi', 'ndvi', 'slavi']
        self.sar_features = ['vh', 'vv']
        self.static_features = ['elevation', 'slope']
        self.all_features = self.biophysical_features + self.optical_features + self.sar_features + self.static_features
        
        # Expected temporal features (should have 21 bands)
        self.temporal_features = self.biophysical_features + self.optical_features + self.sar_features
        
        # NoData values to check
        self.nodata_values = [-9999, -32768, -3.4e38]
        
        # Results storage
        self.results = {
            'zone': self.zone_name,
            'zone_path': str(self.zone_dir),
            'verification_time': pd.Timestamp.now().isoformat(),
            'structure': {},
            'geometry': {},
            'temporal': {},
            'quality': {},
            'summary': {},
            'issues': [],
            'recommendations': [],
            'status': 'UNKNOWN'
        }
        
        logger.info(f"Initialized verifier for zone: {self.zone_name}")
        logger.info(f"Zone directory: {self.zone_dir}")
        logger.info(f"Output directory: {self.output_dir}")
    
    def verify_structure(self) -> Dict[str, Any]:
        """Verify directory structure and feature completeness."""
        logger.info("Verifying zone structure...")
        
        structure = {
            'zone_exists': self.zone_dir.exists(),
            'features_found': [],
            'features_missing': [],
            'file_counts': {},
            'total_files': 0,
            'file_details': {}
        }
        
        if not self.zone_dir.exists():
            logger.error(f"Zone directory not found: {self.zone_dir}")
            return structure
        
        # Check each expected feature
        for feature in self.all_features:
            feature_path = self.zone_dir / feature
            if feature_path.exists() and feature_path.is_dir():
                structure['features_found'].append(feature)
                
                # Count and list files
                tif_files = list(feature_path.glob("*.tif"))
                structure['file_counts'][feature] = len(tif_files)
                structure['total_files'] += len(tif_files)
                
                # Store file details
                structure['file_details'][feature] = [f.name for f in tif_files]
                
                logger.info(f"  {feature}: {len(tif_files)} files")
            else:
                structure['features_missing'].append(feature)
                structure['file_counts'][feature] = 0
                logger.warning(f"  {feature}: MISSING")
        
        logger.info(f"Structure check complete: {len(structure['features_found'])}/{len(self.all_features)} features found")
        return structure
    
    def verify_geometry(self) -> Dict[str, Any]:
        """Verify geometric consistency across all features."""
        logger.info("Verifying geometric consistency...")
        
        geometry = {
            'consistent': True,
            'reference_metadata': None,
            'feature_metadata': {},
            'inconsistencies': [],
            'crs_consistent': True,
            'transform_consistent': True,
            'shape_consistent': True
        }
        
        reference_metadata = None
        
        # Check each feature directory
        for feature in self.all_features:
            feature_path = self.zone_dir / feature
            if not feature_path.exists():
                continue
            
            # Get first raster file for geometry check
            raster_files = list(feature_path.glob("*.tif"))
            if not raster_files:
                continue
            
            sample_file = raster_files[0]
            try:
                with rasterio.open(sample_file) as src:
                    metadata = {
                        'crs': src.crs,
                        'transform': src.transform,
                        'shape': (src.height, src.width),
                        'bounds': src.bounds
                    }
                    
                    geometry['feature_metadata'][feature] = {
                        'crs': str(metadata['crs']),
                        'transform': list(metadata['transform']),
                        'shape': metadata['shape'],
                        'bounds': metadata['bounds'],
                        'sample_file': sample_file.name
                    }
                    
                    if reference_metadata is None:
                        reference_metadata = metadata
                        geometry['reference_metadata'] = {
                            'crs': str(metadata['crs']),
                            'transform': list(metadata['transform']),
                            'shape': metadata['shape'],
                            'bounds': metadata['bounds'],
                            'reference_feature': feature
                        }
                        logger.info(f"  Using {feature} as reference geometry")
                    else:
                        # Compare with reference
                        if metadata['crs'] != reference_metadata['crs']:
                            geometry['crs_consistent'] = False
                            geometry['inconsistencies'].append({
                                'feature': feature,
                                'issue': 'CRS mismatch',
                                'expected': str(reference_metadata['crs']),
                                'found': str(metadata['crs'])
                            })
                        
                        if metadata['transform'] != reference_metadata['transform']:
                            geometry['transform_consistent'] = False
                            geometry['inconsistencies'].append({
                                'feature': feature,
                                'issue': 'Transform mismatch',
                                'reference_transform': list(reference_metadata['transform']),
                                'found_transform': list(metadata['transform'])
                            })
                        
                        if metadata['shape'] != reference_metadata['shape']:
                            geometry['shape_consistent'] = False
                            geometry['inconsistencies'].append({
                                'feature': feature,
                                'issue': 'Shape mismatch',
                                'expected': reference_metadata['shape'],
                                'found': metadata['shape']
                            })
            
            except Exception as e:
                geometry['inconsistencies'].append({
                    'feature': feature,
                    'issue': 'Error reading file',
                    'error': str(e),
                    'file': sample_file.name
                })
        
        geometry['consistent'] = (geometry['crs_consistent'] and 
                                 geometry['transform_consistent'] and 
                                 geometry['shape_consistent'])
        
        if geometry['consistent']:
            logger.info("  Geometry verification: PASSED")
        else:
            logger.warning(f"  Geometry verification: FAILED ({len(geometry['inconsistencies'])} issues)")
        
        return geometry
    
    def verify_temporal_dimensions(self) -> Dict[str, Any]:
        """Verify temporal dimension consistency."""
        logger.info("Verifying temporal dimensions...")
        
        temporal = {
            'temporal_features_correct': [],
            'temporal_features_incorrect': [],
            'static_features_correct': [],
            'static_features_incorrect': [],
            'band_analysis': {},
            'all_temporal_correct': True,
            'all_static_correct': True
        }
        
        # Check temporal features (should have 21 bands)
        for feature in self.temporal_features:
            feature_path = self.zone_dir / feature
            if not feature_path.exists():
                continue
            
            raster_files = list(feature_path.glob("*.tif"))
            band_counts = []
            
            for raster_file in raster_files:
                try:
                    with rasterio.open(raster_file) as src:
                        band_counts.append(src.count)
                except Exception as e:
                    logger.error(f"Error reading {raster_file}: {e}")
            
            if band_counts:
                unique_counts = list(set(band_counts))
                all_correct = all(count == 21 for count in band_counts)
                
                temporal['band_analysis'][feature] = {
                    'files_checked': len(raster_files),
                    'band_counts_found': unique_counts,
                    'expected_bands': 21,
                    'all_correct': all_correct
                }
                
                if all_correct:
                    temporal['temporal_features_correct'].append(feature)
                    logger.info(f"  {feature}: OK (21 bands)")
                else:
                    temporal['temporal_features_incorrect'].append(feature)
                    temporal['all_temporal_correct'] = False
                    logger.warning(f"  {feature}: INCORRECT (found {unique_counts}, expected 21)")
        
        # Check static features (should have 1 band)
        for feature in self.static_features:
            feature_path = self.zone_dir / feature
            if not feature_path.exists():
                continue
            
            raster_files = list(feature_path.glob("*.tif"))
            band_counts = []
            
            for raster_file in raster_files:
                try:
                    with rasterio.open(raster_file) as src:
                        band_counts.append(src.count)
                except Exception as e:
                    logger.error(f"Error reading {raster_file}: {e}")
            
            if band_counts:
                unique_counts = list(set(band_counts))
                all_correct = all(count == 1 for count in band_counts)
                
                temporal['band_analysis'][feature] = {
                    'files_checked': len(raster_files),
                    'band_counts_found': unique_counts,
                    'expected_bands': 1,
                    'all_correct': all_correct
                }
                
                if all_correct:
                    temporal['static_features_correct'].append(feature)
                    logger.info(f"  {feature}: OK (1 band)")
                else:
                    temporal['static_features_incorrect'].append(feature)
                    temporal['all_static_correct'] = False
                    logger.warning(f"  {feature}: INCORRECT (found {unique_counts}, expected 1)")
        
        return temporal
    
    def analyze_data_quality(self, max_files_per_feature: int = 2) -> Dict[str, Any]:
        """Analyze data quality across features."""
        logger.info("Analyzing data quality...")
        
        quality = {
            'feature_analysis': {},
            'overall_stats': {
                'avg_valid_percentage': 0,
                'features_analyzed': 0,
                'problematic_features': []
            }
        }
        
        valid_percentages = []
        
        for feature in self.all_features:
            feature_path = self.zone_dir / feature
            if not feature_path.exists():
                continue
            
            logger.info(f"  Analyzing {feature}...")
            raster_files = list(feature_path.glob("*.tif"))[:max_files_per_feature]
            
            feature_quality = {
                'files_analyzed': len(raster_files),
                'total_files': len(list(feature_path.glob("*.tif"))),
                'file_results': [],
                'avg_valid_percentage': 0,
                'min_valid_percentage': 100,
                'max_valid_percentage': 0,
                'issues': []
            }
            
            file_valid_percentages = []
            
            for raster_file in raster_files:
                file_result = self._analyze_single_raster(raster_file)
                feature_quality['file_results'].append(file_result)
                
                if 'valid_percentage' in file_result:
                    valid_pct = file_result['valid_percentage']
                    file_valid_percentages.append(valid_pct)
                    
                    # Flag low quality files
                    if valid_pct < 80:
                        feature_quality['issues'].append({
                            'file': file_result['file'],
                            'issue': 'Low valid data percentage',
                            'valid_percentage': valid_pct
                        })
            
            if file_valid_percentages:
                feature_quality['avg_valid_percentage'] = np.mean(file_valid_percentages)
                feature_quality['min_valid_percentage'] = np.min(file_valid_percentages)
                feature_quality['max_valid_percentage'] = np.max(file_valid_percentages)
                valid_percentages.append(feature_quality['avg_valid_percentage'])
                
                if feature_quality['avg_valid_percentage'] < 85:
                    quality['overall_stats']['problematic_features'].append(feature)
            
            quality['feature_analysis'][feature] = feature_quality
            quality['overall_stats']['features_analyzed'] += 1
        
        if valid_percentages:
            quality['overall_stats']['avg_valid_percentage'] = np.mean(valid_percentages)
        
        logger.info(f"  Quality analysis complete: {quality['overall_stats']['avg_valid_percentage']:.1f}% average valid data")
        
        return quality
    
    def _analyze_single_raster(self, raster_path: Path) -> Dict[str, Any]:
        """Analyze quality metrics for a single raster file."""
        try:
            with rasterio.open(raster_path) as src:
                # Read data (sample if too large)
                if src.width * src.height > 10000000:  # 10M pixels
                    # Sample for large rasters
                    window = rasterio.windows.Window(0, 0, min(1000, src.width), min(1000, src.height))
                    data = src.read(window=window)
                    total_pixels = data[0].size
                else:
                    data = src.read()
                    total_pixels = data[0].size
                
                # Basic analysis
                result = {
                    'file': raster_path.name,
                    'bands': src.count,
                    'shape': (src.height, src.width),
                    'dtype': str(data.dtype),
                    'sampled': src.width * src.height > 10000000
                }
                
                # NoData analysis
                total_values = total_pixels * src.count
                nodata_count = 0
                
                for ndv in self.nodata_values:
                    nodata_count += np.sum(data == ndv)
                
                nodata_count += np.sum(np.isnan(data))
                nodata_count += np.sum(np.isinf(data))
                
                valid_pixels = total_values - nodata_count
                valid_percentage = (valid_pixels / total_values) * 100
                
                result.update({
                    'total_values': total_values,
                    'valid_pixels': valid_pixels,
                    'nodata_pixels': nodata_count,
                    'valid_percentage': valid_percentage
                })
                
                # Value range analysis for valid data
                if valid_pixels > 0:
                    valid_mask = np.ones_like(data, dtype=bool)
                    for ndv in self.nodata_values:
                        valid_mask &= (data != ndv)
                    valid_mask &= ~np.isnan(data)
                    valid_mask &= ~np.isinf(data)
                    
                    valid_data = data[valid_mask]
                    
                    result.update({
                        'min_value': float(valid_data.min()),
                        'max_value': float(valid_data.max()),
                        'mean_value': float(valid_data.mean()),
                        'std_value': float(valid_data.std())
                    })
                
                return result
                
        except Exception as e:
            return {
                'file': raster_path.name,
                'error': str(e),
                'analysis_failed': True
            }
    
    def generate_summary(self) -> None:
        """Generate verification summary and recommendations."""
        logger.info("Generating summary and recommendations...")
        
        summary = {
            'zone_status': 'PASS',
            'structure_status': 'PASS',
            'geometry_status': 'PASS',
            'temporal_status': 'PASS',
            'quality_status': 'PASS',
            'total_features_expected': len(self.all_features),
            'total_features_found': len(self.results['structure']['features_found']),
            'total_files': self.results['structure']['total_files'],
            'issues_count': 0
        }
        
        issues = []
        recommendations = []
        
        # Check structure
        missing_features = self.results['structure']['features_missing']
        if missing_features:
            summary['structure_status'] = 'FAIL'
            summary['zone_status'] = 'FAIL'
            issues.append(f"Missing features: {missing_features}")
            recommendations.append(f"Generate missing feature data: {missing_features}")
        
        # Check geometry
        if not self.results['geometry']['consistent']:
            summary['geometry_status'] = 'FAIL'
            summary['zone_status'] = 'FAIL'
            geom_issues = len(self.results['geometry']['inconsistencies'])
            issues.append(f"Geometry inconsistencies: {geom_issues} issues found")
            recommendations.append("Re-run raster_clip_align.py to ensure consistent geometry")
        
        # Check temporal
        temporal = self.results['temporal']
        if not (temporal['all_temporal_correct'] and temporal['all_static_correct']):
            summary['temporal_status'] = 'FAIL'
            summary['zone_status'] = 'FAIL'
            incorrect_features = temporal['temporal_features_incorrect'] + temporal['static_features_incorrect']
            issues.append(f"Incorrect band counts in features: {incorrect_features}")
            recommendations.append("Regenerate temporal datacubes with correct band structure")
        
        # Check quality
        quality = self.results['quality']
        if quality['overall_stats']['avg_valid_percentage'] < 90:
            summary['quality_status'] = 'FAIL'
            summary['zone_status'] = 'FAIL'
            avg_quality = quality['overall_stats']['avg_valid_percentage']
            issues.append(f"Low data quality: {avg_quality:.1f}% average valid data")
            recommendations.append("Re-run interpolation or investigate source data quality")
        
        if quality['overall_stats']['problematic_features']:
            prob_features = quality['overall_stats']['problematic_features']
            issues.append(f"Features with quality issues: {prob_features}")
            recommendations.append(f"Investigate and fix quality issues in: {prob_features}")
        
        summary['issues_count'] = len(issues)
        
        # Overall recommendations
        if summary['zone_status'] == 'PASS':
            recommendations.append("✅ Zone verification PASSED - ready for dataset extraction")
        else:
            recommendations.append("❌ Zone verification FAILED - requires fixes before use")
        
        self.results['summary'] = summary
        self.results['issues'] = issues
        self.results['recommendations'] = recommendations
        self.results['status'] = summary['zone_status']
    
    def run_verification(self) -> None:
        """Run complete verification workflow."""
        logger.info(f"Starting verification for zone: {self.zone_name}")
        
        try:
            # Run all verification steps
            self.results['structure'] = self.verify_structure()
            self.results['geometry'] = self.verify_geometry()
            self.results['temporal'] = self.verify_temporal_dimensions()
            self.results['quality'] = self.analyze_data_quality()
            
            # Generate summary
            self.generate_summary()
            
            logger.info(f"Verification complete for {self.zone_name}: {self.results['status']}")
            
        except Exception as e:
            logger.error(f"Verification failed for {self.zone_name}: {e}")
            self.results['status'] = 'ERROR'
            self.results['error'] = str(e)
    
    def save_results(self) -> None:
        """Save verification results to JSON and text report."""
        # Save JSON results
        json_path = self.output_dir / f"{self.zone_name}_verification_results.json"
        with open(json_path, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        
        # Save text report
        report_path = self.output_dir / f"{self.zone_name}_verification_report.txt"
        with open(report_path, 'w') as f:
            f.write("="*80 + "\n")
            f.write(f"DATACUBE VERIFICATION REPORT - {self.zone_name.upper()}\n")
            f.write(f"Generated: {self.results['verification_time']}\n")
            f.write("="*80 + "\n\n")
            
            # Summary
            f.write("SUMMARY\n")
            f.write("-"*40 + "\n")
            f.write(f"Zone: {self.zone_name}\n")
            f.write(f"Status: {self.results['status']}\n")
            f.write(f"Features found: {self.results['summary']['total_features_found']}/{self.results['summary']['total_features_expected']}\n")
            f.write(f"Total files: {self.results['summary']['total_files']}\n")
            f.write(f"Issues found: {self.results['summary']['issues_count']}\n\n")
            
            # Detailed results
            f.write("DETAILED RESULTS\n")
            f.write("-"*40 + "\n")
            
            # Structure
            struct = self.results['structure']
            f.write(f"Structure: {self.results['summary']['structure_status']}\n")
            f.write(f"  Features found: {struct['features_found']}\n")
            if struct['features_missing']:
                f.write(f"  Features missing: {struct['features_missing']}\n")
            f.write(f"  File counts: {struct['file_counts']}\n\n")
            
            # Geometry
            geom = self.results['geometry']
            f.write(f"Geometry: {self.results['summary']['geometry_status']}\n")
            f.write(f"  CRS consistent: {geom['crs_consistent']}\n")
            f.write(f"  Transform consistent: {geom['transform_consistent']}\n")
            f.write(f"  Shape consistent: {geom['shape_consistent']}\n")
            if geom['inconsistencies']:
                f.write(f"  Inconsistencies: {len(geom['inconsistencies'])}\n")
            f.write("\n")
            
            # Temporal
            temp = self.results['temporal']
            f.write(f"Temporal: {self.results['summary']['temporal_status']}\n")
            f.write(f"  Temporal features correct: {temp['temporal_features_correct']}\n")
            f.write(f"  Static features correct: {temp['static_features_correct']}\n")
            if temp['temporal_features_incorrect']:
                f.write(f"  Temporal features incorrect: {temp['temporal_features_incorrect']}\n")
            if temp['static_features_incorrect']:
                f.write(f"  Static features incorrect: {temp['static_features_incorrect']}\n")
            f.write("\n")
            
            # Quality
            qual = self.results['quality']
            f.write(f"Quality: {self.results['summary']['quality_status']}\n")
            f.write(f"  Average valid data: {qual['overall_stats']['avg_valid_percentage']:.1f}%\n")
            f.write(f"  Features analyzed: {qual['overall_stats']['features_analyzed']}\n")
            if qual['overall_stats']['problematic_features']:
                f.write(f"  Problematic features: {qual['overall_stats']['problematic_features']}\n")
            f.write("\n")
            
            # Issues and recommendations
            if self.results['issues']:
                f.write("ISSUES FOUND\n")
                f.write("-"*40 + "\n")
                for i, issue in enumerate(self.results['issues'], 1):
                    f.write(f"{i}. {issue}\n")
                f.write("\n")
            
            f.write("RECOMMENDATIONS\n")
            f.write("-"*40 + "\n")
            for i, rec in enumerate(self.results['recommendations'], 1):
                f.write(f"{i}. {rec}\n")
        
        logger.info(f"Results saved to: {json_path}")
        logger.info(f"Report saved to: {report_path}")
    
    def create_visualization(self) -> None:
        """Create verification summary visualization."""
        try:
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            
            # Feature completeness
            ax = axes[0, 0]
            features_found = len(self.results['structure']['features_found'])
            features_missing = len(self.results['structure']['features_missing'])
            ax.pie([features_found, features_missing], 
                   labels=['Found', 'Missing'], 
                   autopct='%1.1f%%',
                   colors=['green', 'red'])
            ax.set_title(f'Feature Completeness\n{self.zone_name}')
            
            # Quality by feature
            ax = axes[0, 1]
            quality_data = self.results['quality']['feature_analysis']
            features = list(quality_data.keys())
            qualities = [quality_data[f]['avg_valid_percentage'] for f in features]
            
            if features:
                bars = ax.bar(features, qualities, 
                             color=['red' if q < 90 else 'green' for q in qualities])
                ax.set_title('Data Quality by Feature')
                ax.set_ylabel('Valid Data %')
                ax.axhline(y=90, color='red', linestyle='--', alpha=0.7)
                plt.setp(ax.get_xticklabels(), rotation=45)
            
            # Verification status summary
            ax = axes[1, 0]
            statuses = ['Structure', 'Geometry', 'Temporal', 'Quality']
            status_values = [
                1 if self.results['summary']['structure_status'] == 'PASS' else 0,
                1 if self.results['summary']['geometry_status'] == 'PASS' else 0,
                1 if self.results['summary']['temporal_status'] == 'PASS' else 0,
                1 if self.results['summary']['quality_status'] == 'PASS' else 0
            ]
            colors = ['green' if v == 1 else 'red' for v in status_values]
            ax.bar(statuses, status_values, color=colors)
            ax.set_title('Verification Status')
            ax.set_ylabel('Pass (1) / Fail (0)')
            ax.set_ylim(0, 1.2)
            
            # File counts by feature
            ax = axes[1, 1]
            file_counts = self.results['structure']['file_counts']
            features = list(file_counts.keys())
            counts = list(file_counts.values())
            
            if features:
                ax.bar(features, counts, color='blue', alpha=0.7)
                ax.set_title('File Counts by Feature')
                ax.set_ylabel('Number of Files')
                plt.setp(ax.get_xticklabels(), rotation=45)
            
            plt.tight_layout()
            
            # Save plot
            plot_path = self.output_dir / f"{self.zone_name}_verification_summary.png"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Visualization saved to: {plot_path}")
            
        except Exception as e:
            logger.error(f"Failed to create visualization: {e}")

def main():
    """Main function with command line interface."""
    parser = argparse.ArgumentParser(
        description='Datacube Zone Verification Script',
        epilog='''
Examples:
  python datacube_zone_verification.py /fs2/ftype_mflf/shared_data/datacube/2025/1.2.0/zone_1
  python datacube_zone_verification.py /fs2/ftype_mflf/shared_data/datacube/2025/1.2.0/zone_1 --output ./verification_output
  python datacube_zone_verification.py /fs2/ftype_mflf/shared_data/datacube/2025/1.2.0/zone_1 --max-files 5 --verbose
        ''',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('zone_directory', 
                       help='Path to zone datacube directory')
    parser.add_argument('--output', '-o',
                       help='Output directory for verification results')
    parser.add_argument('--max-files', '-m', type=int, default=2,
                       help='Maximum files to analyze per feature (default: 2)')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose logging')
    parser.add_argument('--no-plots', action='store_true',
                       help='Skip visualization generation')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Validate zone directory
    zone_dir = Path(args.zone_directory)
    if not zone_dir.exists():
        logger.error(f"Zone directory not found: {zone_dir}")
        sys.exit(1)
    
    # Initialize verifier
    verifier = ZoneDatacubeVerifier(zone_dir, args.output)
    
    # Run verification
    verifier.run_verification()
    
    # Save results
    verifier.save_results()
    
    # Create visualization
    if not args.no_plots:
        verifier.create_visualization()
    
    # Print summary to console
    logger.info("="*60)
    logger.info("VERIFICATION SUMMARY")
    logger.info("="*60)
    logger.info(f"Zone: {verifier.zone_name}")
    logger.info(f"Status: {verifier.results['status']}")
    logger.info(f"Features: {verifier.results['summary']['total_features_found']}/{verifier.results['summary']['total_features_expected']}")
    logger.info(f"Files: {verifier.results['summary']['total_files']}")
    logger.info(f"Issues: {verifier.results['summary']['issues_count']}")
    
    if verifier.results['issues']:
        logger.info("\nIssues found:")
        for issue in verifier.results['issues']:
            logger.info(f"  - {issue}")
    
    if verifier.results['status'] == 'PASS':
        logger.info("✅ Zone verification PASSED")
        sys.exit(0)
    else:
        logger.info("❌ Zone verification FAILED")
        sys.exit(1)

if __name__ == "__main__":
    main()