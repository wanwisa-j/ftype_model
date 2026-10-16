#!/usr/bin/env python3
"""
Multi-instance scanner - splits work and runs multiple scripts simultaneously
"""

import subprocess
import time
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_single_dir_scanner():
    """Create a simple single directory scanner script"""
    script_content = '''#!/usr/bin/env python3
import sys
import numpy as np
import rasterio
from pathlib import Path
import time

def scan_single_directory(datacube_dir, output_file):
    """Scan a single directory quickly"""
    datacube_path = Path(datacube_dir)
    if not datacube_path.exists():
        print(f"Directory not found: {datacube_dir}")
        return
    
    tif_files = list(datacube_path.rglob("*.tif"))
    print(f"Scanning {len(tif_files)} files in {datacube_dir}")
    
    files_needing_interpolation = []
    start_time = time.time()
    
    for i, tif_file in enumerate(tif_files):
        try:
            with rasterio.open(tif_file) as src:
                # Quick check - sample first and last bands
                total_bands = src.count
                height, width = src.height, src.width
                
                # Sample 2-3 bands for speed
                bands_to_check = [0, total_bands//2, total_bands-1] if total_bands > 2 else [0]
                missing_percentages = []
                
                for band_idx in bands_to_check:
                    band_data = src.read(band_idx + 1, out_dtype=np.float32)
                    nan_count = np.sum(np.isnan(band_data))
                    missing_pct = (nan_count / band_data.size) * 100
                    missing_percentages.append(missing_pct)
                
                avg_missing = np.mean(missing_percentages)
                
                if avg_missing > 0.1:  # Needs interpolation
                    files_needing_interpolation.append({
                        'file': str(tif_file),
                        'missing_pct': avg_missing,
                        'bands': total_bands,
                        'size_mb': tif_file.stat().st_size / (1024*1024)
                    })
                
                # Progress every 10 files
                if (i + 1) % 10 == 0:
                    elapsed = time.time() - start_time
                    rate = (i + 1) / elapsed
                    print(f"  Progress: {i+1}/{len(tif_files)} ({100*(i+1)/len(tif_files):.1f}%), {rate:.1f} files/s")
        
        except Exception as e:
            print(f"  Error: {tif_file.name} - {str(e)[:50]}")
    
    # Save results
    with open(output_file, 'w') as f:
        f.write(f"# Scanned: {datacube_dir}\\n")
        f.write(f"# Found {len(files_needing_interpolation)} files needing interpolation\\n\\n")
        for item in sorted(files_needing_interpolation, key=lambda x: x['missing_pct'], reverse=True):
            f.write(f"# {item['missing_pct']:.1f}% missing, {item['bands']} bands, {item['size_mb']:.1f}MB\\n")
            f.write(f"{item['file']}\\n\\n")
    
    total_time = time.time() - start_time
    print(f"Completed {datacube_dir}: {len(files_needing_interpolation)} files need interpolation")
    print(f"Time: {total_time:.1f}s, Rate: {len(tif_files)/total_time:.1f} files/s\\n")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python script.py <datacube_dir> <output_file>")
        sys.exit(1)
    
    scan_single_directory(sys.argv[1], sys.argv[2])
'''
    
    script_path = Path("/fs1/tor_project/model_foresttype/src/single_dir_scanner.py")
    with open(script_path, 'w') as f:
        f.write(script_content)
    
    return script_path

def run_multiple_instances():
    """Run multiple scanner instances simultaneously"""
    
    # Create the single directory scanner
    scanner_script = create_single_dir_scanner()
    logger.info(f"Created scanner script: {scanner_script}")
    
    # Define all directories to scan
    base_2024 = "/fs2/ftype_mflf/shared_data/datacube/2024/1.2.0"
    base_2025 = "/fs2/ftype_mflf/shared_data/datacube/2025/1.2.0"
    
    # Get all subdirectories
    directories_to_scan = []
    
    for base_dir in [base_2024, base_2025]:
        base_path = Path(base_dir)
        if base_path.exists():
            subdirs = [d for d in base_path.iterdir() if d.is_dir()]
            directories_to_scan.extend(subdirs)
    
    logger.info(f"Found {len(directories_to_scan)} directories to scan")
    
    # Print directory structure
    print("\\n📂 DIRECTORY STRUCTURE:")
    for dir_path in sorted(directories_to_scan):
        print(f"  {dir_path}")
    
    # Create commands for each directory
    commands = []
    output_files = []
    
    for i, dir_path in enumerate(directories_to_scan):
        output_file = f"/fs2/ftype_mflf/shared_data/interpolated/scan_results_{dir_path.name}.txt"
        output_files.append(output_file)
        
        cmd = [
            "eval", '"$(conda shell.bash hook)"', "&&", 
            "conda", "activate", "gisworkspace-cupy", "&&",
            "python", str(scanner_script), str(dir_path), output_file
        ]
        commands.append(cmd)
    
    print(f"\\n🚀 LAUNCHING {len(commands)} PARALLEL INSTANCES")
    
    # Launch all processes simultaneously
    processes = []
    start_time = time.time()
    
    for i, cmd in enumerate(commands):
        cmd_str = " ".join(cmd)
        print(f"  Instance {i+1}: {directories_to_scan[i].name}")
        
        # Use subprocess.Popen for true parallelism
        proc = subprocess.Popen(
            cmd_str,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        processes.append((proc, directories_to_scan[i].name))
    
    print(f"\\n⏳ Waiting for all {len(processes)} processes to complete...")
    
    # Monitor processes
    completed = []
    while len(completed) < len(processes):
        for i, (proc, name) in enumerate(processes):
            if i not in completed and proc.poll() is not None:
                stdout, stderr = proc.communicate()
                completed.append(i)
                elapsed = time.time() - start_time
                print(f"✅ Completed {len(completed)}/{len(processes)}: {name} (after {elapsed:.1f}s)")
                if stdout:
                    print(f"   Output: {stdout.strip()}")
                if stderr:
                    print(f"   Error: {stderr.strip()}")
        
        time.sleep(1)  # Check every second
    
    total_time = time.time() - start_time
    print(f"\\n🎯 ALL PROCESSES COMPLETED in {total_time:.1f}s")
    
    # Combine results
    all_files_needing_interpolation = []
    
    print("\\n📋 COMBINING RESULTS:")
    for output_file in output_files:
        if Path(output_file).exists():
            with open(output_file, 'r') as f:
                lines = f.readlines()
                file_lines = [line.strip() for line in lines if line.strip() and not line.startswith('#')]
                all_files_needing_interpolation.extend(file_lines)
                print(f"  {Path(output_file).name}: {len(file_lines)} files")
    
    # Save combined results
    combined_output = "/fs2/ftype_mflf/shared_data/interpolated/FINAL_files_needing_interpolation.txt"
    with open(combined_output, 'w') as f:
        f.write(f"# 🔥 MULTI-INSTANCE SCAN RESULTS\\n")
        f.write(f"# Total directories scanned: {len(directories_to_scan)}\\n")
        f.write(f"# Total files needing interpolation: {len(all_files_needing_interpolation)}\\n")
        f.write(f"# Total processing time: {total_time:.1f}s\\n")
        f.write(f"# Parallel instances: {len(processes)}\\n\\n")
        
        for file_path in all_files_needing_interpolation:
            f.write(f"{file_path}\\n")
    
    print(f"\\n💾 FINAL RESULTS:")
    print(f"  📁 Total files needing interpolation: {len(all_files_needing_interpolation)}")
    print(f"  📂 Directories scanned: {len(directories_to_scan)}")
    print(f"  ⏱️  Total time: {total_time:.1f}s")
    print(f"  💾 Combined results: {combined_output}")
    
    return all_files_needing_interpolation

if __name__ == "__main__":
    run_multiple_instances()