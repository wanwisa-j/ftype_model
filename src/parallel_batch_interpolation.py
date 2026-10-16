#!/usr/bin/env python3
"""
Parallel batch interpolation with verification and safe replacement.
Processes multiple TIF files simultaneously using all available CPU cores.
"""

import subprocess
import time
import shutil
import logging
from pathlib import Path
from typing import List, Dict
import numpy as np
import rasterio
from concurrent.futures import ProcessPoolExecutor, as_completed
import tempfile

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class BatchInterpolationManager:
    def __init__(self, interpolation_script_path: str, max_workers: int = 16):
        self.interpolation_script = Path(interpolation_script_path)
        self.max_workers = max_workers
        self.temp_dir = Path("/fs2/ftype_mflf/shared_data/interpolated/temp_processing")
        self.backup_dir = Path("/fs2/ftype_mflf/shared_data/interpolated/originals_backup")
        self.verification_log = Path("/fs2/ftype_mflf/shared_data/interpolated/verification_log.txt")
        
        # Create directories
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Batch manager initialized with {max_workers} workers")
        logger.info(f"Temp processing: {self.temp_dir}")
        logger.info(f"Backup directory: {self.backup_dir}")

    def verify_interpolation(self, original_path: str, interpolated_path: str) -> Dict:
        """
        Verify interpolated file against original.
        Returns verification results.
        """
        try:
            with rasterio.open(original_path) as orig_src, \
                 rasterio.open(interpolated_path) as interp_src:
                
                # Basic structure verification
                if orig_src.count != interp_src.count:
                    return {'status': 'FAILED', 'error': 'Band count mismatch'}
                if orig_src.shape != interp_src.shape:
                    return {'status': 'FAILED', 'error': 'Shape mismatch'}
                if orig_src.crs != interp_src.crs:
                    return {'status': 'FAILED', 'error': 'CRS mismatch'}
                
                # Sample a few bands for data verification
                total_bands = orig_src.count
                sample_bands = [0, total_bands//2, total_bands-1] if total_bands > 2 else [0]
                
                verification_results = []
                
                for band_idx in sample_bands:
                    orig_band = orig_src.read(band_idx + 1, out_dtype=np.float32)
                    interp_band = interp_src.read(band_idx + 1, out_dtype=np.float32)
                    
                    # Count valid/invalid pixels
                    orig_valid = np.sum(~np.isnan(orig_band))
                    interp_valid = np.sum(~np.isnan(interp_band))
                    
                    # Check if interpolation improved data coverage
                    pixels_filled = interp_valid - orig_valid
                    
                    # Verify that original valid pixels remain unchanged
                    orig_valid_mask = ~np.isnan(orig_band)
                    if np.any(orig_valid_mask):
                        orig_valid_data = orig_band[orig_valid_mask]
                        interp_valid_data = interp_band[orig_valid_mask]
                        
                        # Allow small floating point differences
                        max_diff = np.max(np.abs(orig_valid_data - interp_valid_data))
                        
                        verification_results.append({
                            'band': band_idx,
                            'orig_valid': int(orig_valid),
                            'interp_valid': int(interp_valid),
                            'pixels_filled': int(pixels_filled),
                            'max_difference': float(max_diff),
                            'valid_preserved': max_diff < 1e-5
                        })
                
                # Overall verification
                total_pixels_filled = sum(r['pixels_filled'] for r in verification_results)
                all_preserved = all(r['valid_preserved'] for r in verification_results)
                
                if total_pixels_filled > 0 and all_preserved:
                    return {
                        'status': 'PASSED',
                        'pixels_filled': total_pixels_filled,
                        'bands_checked': len(verification_results),
                        'details': verification_results
                    }
                else:
                    return {
                        'status': 'FAILED',
                        'error': f'Data integrity issue. Filled: {total_pixels_filled}, Preserved: {all_preserved}',
                        'details': verification_results
                    }
                    
        except Exception as e:
            return {'status': 'ERROR', 'error': str(e)}

    def interpolate_single_file(self, input_file: str, test_mode: bool = False) -> Dict:
        """
        Process a single file with interpolation, verification, and optional replacement.
        """
        start_time = time.time()
        input_path = Path(input_file)
        
        # Create temporary output path
        temp_output = self.temp_dir / f"{input_path.stem}_interpolated{input_path.suffix}"
        
        try:
            # Run interpolation
            cmd = [
                "eval", '"$(conda shell.bash hook)"', "&&",
                "conda", "activate", "gisworkspace-cupy", "&&",
                "python", str(self.interpolation_script),
                input_file, str(temp_output)
            ]
            
            cmd_str = " ".join(cmd)
            result = subprocess.run(cmd_str, shell=True, capture_output=True, text=True, timeout=1800)  # 30 min timeout
            
            if result.returncode != 0:
                return {
                    'file': input_file,
                    'status': 'INTERPOLATION_FAILED',
                    'error': result.stderr,
                    'processing_time': time.time() - start_time
                }
            
            # Verify interpolation
            if not temp_output.exists():
                return {
                    'file': input_file,
                    'status': 'OUTPUT_MISSING',
                    'error': 'Interpolated file not created',
                    'processing_time': time.time() - start_time
                }
            
            verification = self.verify_interpolation(input_file, str(temp_output))
            
            if verification['status'] != 'PASSED':
                return {
                    'file': input_file,
                    'status': 'VERIFICATION_FAILED',
                    'error': verification.get('error', 'Verification failed'),
                    'verification': verification,
                    'processing_time': time.time() - start_time
                }
            
            # If not in test mode, perform backup and replacement
            if not test_mode:
                # Create backup
                backup_path = self.backup_dir / input_path.name
                shutil.copy2(input_file, backup_path)
                
                # Replace original with interpolated
                shutil.move(str(temp_output), input_file)
                
                return {
                    'file': input_file,
                    'status': 'SUCCESS_REPLACED',
                    'backup': str(backup_path),
                    'pixels_filled': verification['pixels_filled'],
                    'processing_time': time.time() - start_time,
                    'verification': verification
                }
            else:
                # Test mode - keep temporary file for inspection
                final_test_path = self.temp_dir / f"TEST_{input_path.name}"
                shutil.move(str(temp_output), final_test_path)
                
                return {
                    'file': input_file,
                    'status': 'SUCCESS_TEST',
                    'output_file': str(final_test_path),
                    'pixels_filled': verification['pixels_filled'],
                    'processing_time': time.time() - start_time,
                    'verification': verification
                }
            
        except subprocess.TimeoutExpired:
            return {
                'file': input_file,
                'status': 'TIMEOUT',
                'error': 'Processing exceeded 30 minute timeout',
                'processing_time': time.time() - start_time
            }
        except Exception as e:
            return {
                'file': input_file,
                'status': 'ERROR',
                'error': str(e),
                'processing_time': time.time() - start_time
            }
        finally:
            # Clean up temporary file if it still exists
            if temp_output.exists():
                temp_output.unlink()

    def process_batch(self, file_list: List[str], test_mode: bool = False, max_files: int = None) -> Dict:
        """
        Process multiple files in parallel.
        """
        if max_files:
            file_list = file_list[:max_files]
        
        total_files = len(file_list)
        logger.info(f"🚀 Starting {'TEST' if test_mode else 'PRODUCTION'} batch processing")
        logger.info(f"📁 Files to process: {total_files}")
        logger.info(f"🔧 Workers: {self.max_workers}")
        logger.info(f"⚡ Max parallel processes: {self.max_workers}")
        
        results = []
        completed = 0
        start_time = time.time()
        
        print(f"\\n🔥 BATCH INTERPOLATION {'(TEST MODE)' if test_mode else '(PRODUCTION)'}")
        print("File | Status | Time (s) | Pixels Filled | Notes")
        print("-" * 80)
        
        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all jobs
            future_to_file = {
                executor.submit(self.interpolate_single_file, file_path, test_mode): file_path
                for file_path in file_list
            }
            
            # Process completed jobs
            for future in as_completed(future_to_file):
                file_path = future_to_file[future]
                try:
                    result = future.result()
                    results.append(result)
                    completed += 1
                    
                    # Print progress
                    filename = Path(file_path).name[:25]
                    status = result['status'][:15]
                    time_taken = result['processing_time']
                    pixels_filled = result.get('pixels_filled', 0)
                    
                    if result['status'].startswith('SUCCESS'):
                        status_icon = '✅'
                    elif result['status'].startswith('VERIFICATION'):
                        status_icon = '⚠️ '
                    else:
                        status_icon = '❌'
                    
                    print(f"{filename:25s} | {status_icon} {status:13s} | {time_taken:7.1f} | {pixels_filled:11,d} | {'OK' if result['status'].startswith('SUCCESS') else result.get('error', '')[:20]}")
                    
                    # Progress update every 10 files
                    if completed % 10 == 0:
                        elapsed = time.time() - start_time
                        rate = completed / elapsed
                        eta = (total_files - completed) / rate if rate > 0 else 0
                        logger.info(f"⚡ Progress: {completed}/{total_files} ({100*completed/total_files:.1f}%), Rate: {rate:.2f} files/s, ETA: {eta:.0f}s")
                
                except Exception as e:
                    logger.error(f"❌ Error processing {file_path}: {e}")
                    results.append({
                        'file': file_path,
                        'status': 'PROCESSING_ERROR',
                        'error': str(e),
                        'processing_time': 0
                    })
                    completed += 1
        
        # Summary
        total_time = time.time() - start_time
        success_count = sum(1 for r in results if r['status'].startswith('SUCCESS'))
        failed_count = len(results) - success_count
        
        print(f"\\n🎯 BATCH PROCESSING COMPLETE")
        print(f"📁 Total files: {total_files}")
        print(f"✅ Successful: {success_count}")
        print(f"❌ Failed: {failed_count}")
        print(f"⏱️  Total time: {total_time/60:.1f} minutes")
        print(f"🚀 Average rate: {total_files/total_time:.2f} files/second")
        
        # Save detailed log
        log_content = self._generate_detailed_log(results, test_mode, total_time)
        with open(self.verification_log, 'w') as f:
            f.write(log_content)
        
        logger.info(f"📊 Detailed log saved: {self.verification_log}")
        
        return {
            'total_files': total_files,
            'successful': success_count,
            'failed': failed_count,
            'processing_time': total_time,
            'results': results
        }

    def _generate_detailed_log(self, results: List[Dict], test_mode: bool, total_time: float) -> str:
        """Generate detailed processing log."""
        log_lines = [
            f"# 🔥 BATCH INTERPOLATION LOG",
            f"# Mode: {'TEST' if test_mode else 'PRODUCTION'}",
            f"# Processed: {len(results)} files in {total_time/60:.1f} minutes",
            f"# Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}",
            ""
        ]
        
        for result in results:
            log_lines.append(f"## {Path(result['file']).name}")
            log_lines.append(f"- Status: {result['status']}")
            log_lines.append(f"- Processing time: {result['processing_time']:.1f}s")
            
            if 'pixels_filled' in result:
                log_lines.append(f"- Pixels filled: {result['pixels_filled']:,}")
            
            if 'backup' in result:
                log_lines.append(f"- Backup: {result['backup']}")
            
            if 'error' in result:
                log_lines.append(f"- Error: {result['error']}")
            
            if 'verification' in result and 'details' in result['verification']:
                log_lines.append("- Verification details:")
                for detail in result['verification']['details']:
                    log_lines.append(f"  - Band {detail['band']}: {detail['pixels_filled']:,} pixels filled")
            
            log_lines.append("")
        
        return "\\n".join(log_lines)


def main():
    """Main function to run batch interpolation."""
    
    # Configuration
    interpolation_script = "/fs1/tor_project/model_foresttype/src/raster_interpolation_ultra_fast.py"
    file_list_path = "/fs2/ftype_mflf/shared_data/interpolated/CORRECTED_files_needing_interpolation.txt"
    max_workers = 16  # Use 16 workers for maximum throughput
    
    # Read file list
    with open(file_list_path, 'r') as f:
        files_to_process = [line.strip() for line in f if line.strip() and not line.startswith('#')]
    
    logger.info(f"📋 Loaded {len(files_to_process)} files from {file_list_path}")
    
    # Initialize batch manager
    manager = BatchInterpolationManager(interpolation_script, max_workers)
    
    # Ask user for test mode or production mode
    print(f"\\n🎯 BATCH INTERPOLATION READY")
    print(f"📁 Files to process: {len(files_to_process)}")
    print(f"🔧 Workers: {max_workers}")
    print()
    print("Options:")
    print("1. TEST MODE (3 files) - Process 3 files to verify everything works")
    print("2. PRODUCTION MODE - Process all 80 files with backup & replacement")
    print("3. CUSTOM TEST (N files) - Process N files in test mode")
    
    choice = input("Choose option (1/2/3): ").strip()
    
    if choice == "1":
        # Test mode with 3 files
        result = manager.process_batch(files_to_process, test_mode=True, max_files=3)
    elif choice == "2":
        # Production mode
        confirm = input(f"⚠️  This will process ALL {len(files_to_process)} files and REPLACE originals. Continue? (yes/no): ").strip().lower()
        if confirm == "yes":
            result = manager.process_batch(files_to_process, test_mode=False)
        else:
            print("❌ Cancelled.")
            return
    elif choice == "3":
        # Custom test
        try:
            n_files = int(input("Enter number of files to test: "))
            result = manager.process_batch(files_to_process, test_mode=True, max_files=n_files)
        except ValueError:
            print("❌ Invalid number.")
            return
    else:
        print("❌ Invalid choice.")
        return
    
    print(f"\\n✅ Batch processing completed!")
    print(f"📊 Check detailed log: {manager.verification_log}")

if __name__ == "__main__":
    main()