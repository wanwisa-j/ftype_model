#!/usr/bin/env python3
"""
Ultra-fast datacube scanner using threading + memory optimization.
Designed to max out 128GB RAM and 19-thread CPU.
"""

import numpy as np
import rasterio
from pathlib import Path
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from dataclasses import dataclass
from typing import List, Dict, Any
import time
import gc

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class FileAnalysisResult:
    file_path: str
    file_size_mb: float
    bands: int
    dimensions: tuple
    avg_missing_pct: float
    needs_interpolation: bool
    processing_time: float
    status: str
    error: str = ""

class HighMemoryFileAnalyzer:
    def __init__(self, max_memory_gb: int = 120):
        self.max_memory_gb = max_memory_gb
        self.lock = threading.Lock()
        self.processed_count = 0
        self.start_time = time.time()
        
    def analyze_file_aggressive(self, file_path_str: str) -> FileAnalysisResult:
        """Aggressively analyze file using more memory for speed."""
        start_time = time.time()
        file_path = Path(file_path_str)
        
        try:
            with rasterio.open(file_path) as src:
                total_bands = src.count
                height, width = src.height, src.width
                total_pixels_per_band = height * width
                nodata = src.nodata
                
                # Use larger sample for better accuracy
                sample_bands = min(7, total_bands)  # More samples
                if sample_bands == 1:
                    bands_to_check = [0]
                else:
                    bands_to_check = np.linspace(0, total_bands-1, sample_bands).astype(int).tolist()
                
                # Pre-allocate arrays for speed
                missing_percentages = []
                
                # Read multiple bands at once if memory allows
                file_size_gb = file_path.stat().st_size / (1024**3)
                
                if file_size_gb < 2.0:  # Small files - read all at once
                    try:
                        all_data = src.read(out_dtype=np.float32)  # Read all bands
                        for band_idx in bands_to_check:
                            band_data = all_data[band_idx]
                            missing_pct = self._calculate_missing_percentage(band_data, nodata, total_pixels_per_band)
                            missing_percentages.append(missing_pct)
                        del all_data  # Free memory immediately
                    except MemoryError:
                        # Fallback to single band reading
                        for band_idx in bands_to_check:
                            band_data = src.read(band_idx + 1, out_dtype=np.float32)
                            missing_pct = self._calculate_missing_percentage(band_data, nodata, total_pixels_per_band)
                            missing_percentages.append(missing_pct)
                            del band_data
                else:
                    # Large files - read bands individually but with chunking
                    for band_idx in bands_to_check:
                        band_data = src.read(band_idx + 1, out_dtype=np.float32)
                        missing_pct = self._calculate_missing_percentage(band_data, nodata, total_pixels_per_band)
                        missing_percentages.append(missing_pct)
                        del band_data
                        gc.collect()  # Force garbage collection for large files
                
                avg_missing_pct = np.mean(missing_percentages)
                processing_time = time.time() - start_time
                
                # Thread-safe counter update
                with self.lock:
                    self.processed_count += 1
                
                return FileAnalysisResult(
                    file_path=str(file_path),
                    file_size_mb=file_path.stat().st_size / (1024*1024),
                    bands=total_bands,
                    dimensions=(height, width),
                    avg_missing_pct=avg_missing_pct,
                    needs_interpolation=avg_missing_pct > 0.1,
                    processing_time=processing_time,
                    status='success'
                )
                
        except Exception as e:
            processing_time = time.time() - start_time
            with self.lock:
                self.processed_count += 1
            
            return FileAnalysisResult(
                file_path=str(file_path),
                file_size_mb=file_path.stat().st_size / (1024*1024) if file_path.exists() else 0,
                bands=0,
                dimensions=(0, 0),
                avg_missing_pct=0,
                needs_interpolation=False,
                processing_time=processing_time,
                status='error',
                error=str(e)[:100]  # Truncate long errors
            )
    
    def _calculate_missing_percentage(self, band_data: np.ndarray, nodata: float, total_pixels: int) -> float:
        """Optimized missing data calculation."""
        # Vectorized operations for speed
        nan_mask = np.isnan(band_data)
        nan_count = np.sum(nan_mask)
        
        nodata_count = 0
        if nodata is not None:
            nodata_mask = np.isclose(band_data, nodata)
            nodata_count = np.sum(nodata_mask)
        
        missing_count = nan_count + nodata_count
        return (missing_count / total_pixels) * 100

def scan_with_max_resources(datacube_dirs: List[str], max_workers: int = 18, 
                           output_file: str = None) -> Dict[str, Any]:
    """Scan using maximum available resources."""
    
    logger.info(f"🚀 ULTRA-FAST MODE: {max_workers} threads, targeting 120GB RAM usage")
    
    # Collect all files
    all_tif_files = []
    for datacube_dir in datacube_dirs:
        datacube_path = Path(datacube_dir)
        if datacube_path.exists():
            tif_files = list(datacube_path.rglob("*.tif"))
            all_tif_files.extend(tif_files)
            logger.info(f"Found {len(tif_files)} TIF files in {datacube_dir}")
    
    total_files = len(all_tif_files)
    logger.info(f"Total files to process: {total_files}")
    
    # Sort by file size - process larger files first for better parallelization
    all_tif_files.sort(key=lambda x: x.stat().st_size, reverse=True)
    
    analyzer = HighMemoryFileAnalyzer(max_memory_gb=120)
    results = []
    files_needing_interpolation = []
    
    print(f"\n🔥 ULTRA-FAST SCANNING {total_files} FILES 🔥")
    print("File | Size (MB) | Bands | Dimensions | Missing % | Time (s) | Status")
    print("-" * 85)
    
    start_time = time.time()
    last_progress_time = start_time
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all jobs at once to maximize throughput
        future_to_file = {
            executor.submit(analyzer.analyze_file_aggressive, str(tif_file)): tif_file 
            for tif_file in all_tif_files
        }
        
        # Process results as they complete
        for future in as_completed(future_to_file):
            tif_file = future_to_file[future]
            try:
                result = future.result()
                results.append(result)
                
                # Print result
                if result.status == 'success':
                    dims_str = f"{result.dimensions[0]}x{result.dimensions[1]}"
                    status_str = '🔴 NEEDS INTERP' if result.needs_interpolation else '✅ OK'
                    print(f"{tif_file.name[:23]:23s} | {result.file_size_mb:7.1f} | {result.bands:5d} | {dims_str:11s} | {result.avg_missing_pct:7.1f} | {result.processing_time:6.2f} | {status_str}")
                    
                    if result.needs_interpolation:
                        files_needing_interpolation.append({
                            'file': str(tif_file),
                            'missing_pct': result.avg_missing_pct,
                            'file_size_mb': result.file_size_mb
                        })
                else:
                    print(f"{tif_file.name[:23]:23s} | {result.file_size_mb:7.1f} | ERROR | ERROR      | ERROR | {result.processing_time:6.2f} | ❌ {result.error[:15]}")
                
                # Progress updates every 5 seconds
                current_time = time.time()
                if current_time - last_progress_time > 5:
                    completed = analyzer.processed_count
                    elapsed = current_time - start_time
                    rate = completed / elapsed if elapsed > 0 else 0
                    eta = (total_files - completed) / rate if rate > 0 else 0
                    logger.info(f"⚡ Progress: {completed}/{total_files} ({100*completed/total_files:.1f}%), Rate: {rate:.1f} files/s, ETA: {eta:.0f}s")
                    last_progress_time = current_time
            
            except Exception as e:
                logger.error(f"❌ Error processing {tif_file}: {e}")
    
    total_time = time.time() - start_time
    success_count = sum(1 for r in results if r.status == 'success')
    error_count = len(results) - success_count
    
    # Final summary
    print(f"\n🎯 FINAL RESULTS")
    print(f"📁 Total files processed: {total_files}")
    print(f"✅ Successful analyses: {success_count}")
    print(f"❌ Files with errors: {error_count}")
    print(f"🔴 Files needing interpolation: {len(files_needing_interpolation)}")
    print(f"⏱️  Total time: {total_time:.1f}s")
    print(f"🚀 Average rate: {total_files/total_time:.1f} files/second")
    
    # Show directory structure
    print(f"\n📂 DIRECTORY STRUCTURE ANALYZED:")
    for datacube_dir in datacube_dirs:
        subdirs = [d.name for d in Path(datacube_dir).iterdir() if d.is_dir()]
        print(f"   {datacube_dir}:")
        for subdir in sorted(subdirs)[:10]:  # Show first 10
            print(f"     ├── {subdir}")
        if len(subdirs) > 10:
            print(f"     └── ... and {len(subdirs)-10} more")
    
    if files_needing_interpolation:
        print(f"\n🔴 TOP FILES BY MISSING DATA:")
        sorted_files = sorted(files_needing_interpolation, key=lambda x: x['missing_pct'], reverse=True)
        for i, item in enumerate(sorted_files[:15], 1):
            print(f"  {i:2d}. {item['missing_pct']:5.1f}% - {Path(item['file']).name} ({item['file_size_mb']:.1f}MB)")
    
    # Save results
    if output_file:
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            f.write(f"# 🔥 ULTRA-FAST DATACUBE SCAN RESULTS\n")
            f.write(f"# Scanned: {', '.join(datacube_dirs)}\n")
            f.write(f"# Files needing interpolation: {len(files_needing_interpolation)}/{total_files}\n")
            f.write(f"# Processing: {total_time:.1f}s with {max_workers} threads\n")
            f.write(f"# Rate: {total_files/total_time:.1f} files/second\n\n")
            
            for item in sorted(files_needing_interpolation, key=lambda x: x['missing_pct'], reverse=True):
                f.write(f"# {item['missing_pct']:5.1f}% missing ({item['file_size_mb']:.1f}MB)\n")
                f.write(f"{item['file']}\n\n")
        
        logger.info(f"💾 Results saved to: {output_path}")
    
    return {
        'total_files': total_files,
        'files_needing_interpolation': files_needing_interpolation,
        'results': results,
        'processing_time': total_time,
        'success_count': success_count,
        'error_count': error_count
    }

if __name__ == "__main__":
    datacube_dirs = [
        "/fs2/ftype_mflf/shared_data/datacube/2024/1.2.0",
        "/fs2/ftype_mflf/shared_data/datacube/2025/1.2.0"
    ]
    
    output_file = "/fs2/ftype_mflf/shared_data/interpolated/ultra_fast_scan_results.txt"
    
    # Use 18 threads (leave 1 for system)
    results = scan_with_max_resources(datacube_dirs, max_workers=18, output_file=output_file)