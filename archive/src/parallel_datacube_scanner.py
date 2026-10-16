#!/usr/bin/env python3
"""
Parallel scanner for datacube directories to identify files needing interpolation.
Optimized for high-memory, multi-core systems.
"""

import os
import numpy as np
import rasterio
from pathlib import Path
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
from dataclasses import dataclass
from typing import List, Dict, Any
import time

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class FileAnalysisResult:
    file_path: str
    file_size_mb: float
    bands: int
    dimensions: tuple
    total_pixels: int
    avg_missing_pct: float
    estimated_total_missing: int
    needs_interpolation: bool
    processing_time: float
    status: str
    error: str = ""

def analyze_single_file(file_path_str: str, sample_bands: int = 3) -> FileAnalysisResult:
    """Analyze a single raster file for missing data."""
    start_time = time.time()
    file_path = Path(file_path_str)
    
    try:
        with rasterio.open(file_path) as src:
            total_bands = src.count
            height, width = src.height, src.width
            total_pixels_per_band = height * width
            nodata = src.nodata
            
            # Sample bands efficiently
            num_sample_bands = min(sample_bands, total_bands)
            if num_sample_bands == 1:
                bands_to_check = [0]
            else:
                bands_to_check = np.linspace(0, total_bands-1, num_sample_bands).astype(int).tolist()
            
            missing_percentages = []
            
            for band_idx in bands_to_check:
                # Read band with windowed reading for large files
                band_data = src.read(band_idx + 1, out_dtype=np.float32)
                
                # Count missing values efficiently
                nan_count = np.sum(np.isnan(band_data))
                nodata_count = 0
                if nodata is not None:
                    nodata_count = np.sum(np.isclose(band_data, nodata))
                
                missing_count = nan_count + nodata_count
                missing_pct = (missing_count / total_pixels_per_band) * 100
                missing_percentages.append(missing_pct)
                
                # Early exit if no missing data found
                if missing_pct == 0 and len(missing_percentages) >= 2:
                    break
            
            avg_missing_pct = np.mean(missing_percentages)
            estimated_total_missing = int((avg_missing_pct / 100) * total_pixels_per_band * total_bands)
            
            processing_time = time.time() - start_time
            
            return FileAnalysisResult(
                file_path=str(file_path),
                file_size_mb=file_path.stat().st_size / (1024*1024),
                bands=total_bands,
                dimensions=(height, width),
                total_pixels=total_pixels_per_band * total_bands,
                avg_missing_pct=avg_missing_pct,
                estimated_total_missing=estimated_total_missing,
                needs_interpolation=avg_missing_pct > 0.1,
                processing_time=processing_time,
                status='success'
            )
            
    except Exception as e:
        processing_time = time.time() - start_time
        return FileAnalysisResult(
            file_path=str(file_path),
            file_size_mb=file_path.stat().st_size / (1024*1024) if file_path.exists() else 0,
            bands=0,
            dimensions=(0, 0),
            total_pixels=0,
            avg_missing_pct=0,
            estimated_total_missing=0,
            needs_interpolation=False,
            processing_time=processing_time,
            status='error',
            error=str(e)
        )

def scan_directory_parallel(datacube_dirs: List[str], max_workers: int = None, 
                           output_file: str = None) -> Dict[str, Any]:
    """Scan multiple datacube directories in parallel."""
    
    if max_workers is None:
        max_workers = min(19, multiprocessing.cpu_count() - 1)  # Leave 1 CPU free
    
    logger.info(f"Starting parallel scan with {max_workers} workers")
    logger.info(f"Available memory: ~128GB, using efficient chunking")
    
    # Collect all TIF files from all directories
    all_tif_files = []
    for datacube_dir in datacube_dirs:
        datacube_path = Path(datacube_dir)
        if datacube_path.exists():
            tif_files = list(datacube_path.rglob("*.tif"))
            all_tif_files.extend(tif_files)
            logger.info(f"Found {len(tif_files)} TIF files in {datacube_dir}")
        else:
            logger.warning(f"Directory not found: {datacube_dir}")
    
    total_files = len(all_tif_files)
    logger.info(f"Total files to process: {total_files}")
    
    if total_files == 0:
        return {'total_files': 0, 'files_needing_interpolation': [], 'results': []}
    
    # Process files in parallel
    results = []
    files_needing_interpolation = []
    completed = 0
    
    print(f"\n=== PARALLEL SCANNING {total_files} FILES ===")
    print("File | Size (MB) | Bands | Dimensions | Missing % | Time (s) | Status")
    print("-" * 85)
    
    start_time = time.time()
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all jobs
        future_to_file = {
            executor.submit(analyze_single_file, str(tif_file), 3): tif_file 
            for tif_file in all_tif_files
        }
        
        # Process completed jobs
        for future in as_completed(future_to_file):
            tif_file = future_to_file[future]
            try:
                result = future.result()
                results.append(result)
                completed += 1
                
                # Print progress
                if result.status == 'success':
                    dims_str = f"{result.dimensions[0]}x{result.dimensions[1]}"
                    status_str = 'NEEDS INTERP' if result.needs_interpolation else 'OK'
                    print(f"{tif_file.name[:25]:25s} | {result.file_size_mb:7.1f} | {result.bands:5d} | {dims_str:11s} | {result.avg_missing_pct:7.1f} | {result.processing_time:6.2f} | {status_str}")
                    
                    if result.needs_interpolation:
                        files_needing_interpolation.append({
                            'file': str(tif_file),
                            'missing_pct': result.avg_missing_pct,
                            'estimated_missing': result.estimated_total_missing
                        })
                else:
                    print(f"{tif_file.name[:25]:25s} | {result.file_size_mb:7.1f} | ERROR | ERROR      | ERROR | {result.processing_time:6.2f} | {result.error[:20]}")
                
                # Progress update every 50 files
                if completed % 50 == 0:
                    elapsed = time.time() - start_time
                    rate = completed / elapsed
                    eta = (total_files - completed) / rate if rate > 0 else 0
                    logger.info(f"Progress: {completed}/{total_files} ({100*completed/total_files:.1f}%), Rate: {rate:.1f} files/s, ETA: {eta:.0f}s")
            
            except Exception as e:
                logger.error(f"Error processing {tif_file}: {e}")
                completed += 1
    
    total_time = time.time() - start_time
    
    # Summary
    success_count = sum(1 for r in results if r.status == 'success')
    error_count = sum(1 for r in results if r.status == 'error')
    
    print(f"\n=== SUMMARY ===")
    print(f"Total files processed: {total_files}")
    print(f"Successful analyses: {success_count}")
    print(f"Files with errors: {error_count}")
    print(f"Files needing interpolation: {len(files_needing_interpolation)}")
    print(f"Total processing time: {total_time:.1f}s")
    print(f"Average rate: {total_files/total_time:.1f} files/second")
    
    if files_needing_interpolation:
        print(f"\n=== TOP 10 FILES BY MISSING DATA % ===")
        sorted_files = sorted(files_needing_interpolation, key=lambda x: x['missing_pct'], reverse=True)
        for item in sorted_files[:10]:
            print(f"  {item['missing_pct']:5.1f}% - {Path(item['file']).name}")
    
    # Save results if output file specified
    if output_file:
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            f.write(f"# Files needing raster interpolation\n")
            f.write(f"# Generated by parallel scanning: {', '.join(datacube_dirs)}\n")
            f.write(f"# Found {len(files_needing_interpolation)} files needing interpolation\n")
            f.write(f"# Processing time: {total_time:.1f}s with {max_workers} workers\n\n")
            
            for item in sorted(files_needing_interpolation, key=lambda x: x['missing_pct'], reverse=True):
                f.write(f"# {item['missing_pct']:5.1f}% missing - {item['estimated_missing']:,} pixels\n")
                f.write(f"{item['file']}\n\n")
        
        logger.info(f"Results saved to: {output_path}")
    
    return {
        'total_files': total_files,
        'files_needing_interpolation': files_needing_interpolation,
        'results': results,
        'processing_time': total_time,
        'success_count': success_count,
        'error_count': error_count
    }

if __name__ == "__main__":
    # Define directories to scan
    datacube_dirs = [
        "/fs2/ftype_mflf/shared_data/datacube/2024/1.2.0",
        "/fs2/ftype_mflf/shared_data/datacube/2025/1.2.0"
    ]
    
    output_file = "/fs2/ftype_mflf/shared_data/interpolated/all_files_needing_interpolation.txt"
    
    # Use 16 workers (leaving 3 CPUs free for system)
    max_workers = 16
    
    results = scan_directory_parallel(datacube_dirs, max_workers, output_file)