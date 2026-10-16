#!/usr/bin/env python3
"""
Distribute batch interpolation across multiple machines.
Creates file splits and setup instructions for parallel processing.
"""

import math
from pathlib import Path
import json

def distribute_files_across_machines(file_list_path: str, num_machines: int = 4):
    """
    Split files across multiple machines for parallel processing.
    """
    
    # Read all files
    with open(file_list_path, 'r') as f:
        all_files = [line.strip() for line in f if line.strip() and not line.startswith('#')]
    
    total_files = len(all_files)
    files_per_machine = math.ceil(total_files / num_machines)
    
    print(f"🔄 DISTRIBUTING {total_files} FILES ACROSS {num_machines} MACHINES")
    print(f"📊 Files per machine: ~{files_per_machine}")
    print()
    
    # Create distribution
    machine_distributions = []
    
    for machine_id in range(1, num_machines + 1):
        start_idx = (machine_id - 1) * files_per_machine
        end_idx = min(start_idx + files_per_machine, total_files)
        
        machine_files = all_files[start_idx:end_idx]
        
        if machine_files:  # Only if machine has files assigned
            distribution = {
                'machine_id': machine_id,
                'files': machine_files,
                'file_count': len(machine_files),
                'start_index': start_idx,
                'end_index': end_idx - 1
            }
            machine_distributions.append(distribution)
    
    return machine_distributions

def create_machine_specific_files(distributions, output_dir="/fs2/ftype_mflf/shared_data/interpolated/machine_splits"):
    """
    Create machine-specific file lists and scripts.
    """
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"📁 Creating machine-specific files in: {output_dir}")
    print()
    
    setup_instructions = []
    
    for dist in distributions:
        machine_id = dist['machine_id']
        
        # Create file list for this machine
        file_list_path = output_path / f"machine_{machine_id}_files.txt"
        with open(file_list_path, 'w') as f:
            f.write(f"# Machine {machine_id} file list\n")
            f.write(f"# Files: {dist['file_count']}\n")
            f.write(f"# Indices: {dist['start_index']}-{dist['end_index']}\n\n")
            for file_path in dist['files']:
                f.write(f"{file_path}\n")
        
        # Create run script for this machine
        run_script_path = output_path / f"run_machine_{machine_id}.py"
        run_script_content = f'''#!/usr/bin/env python3
"""
Machine {machine_id} batch interpolation script.
Auto-generated for distributed processing.
"""

import sys
import os
sys.path.append('/fs1/tor_project/model_foresttype/src')

from parallel_batch_interpolation import BatchInterpolationManager
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_machine_{machine_id}_batch():
    """Run batch processing for Machine {machine_id}."""
    
    # Configuration
    interpolation_script = "/fs1/tor_project/model_foresttype/src/raster_interpolation_ultra_fast.py"
    file_list_path = "{file_list_path}"
    max_workers = 16  # Adjust based on machine specs
    
    # Read assigned files
    with open(file_list_path, 'r') as f:
        files_to_process = [line.strip() for line in f if line.strip() and not line.startswith('#')]
    
    logger.info(f"🤖 MACHINE {machine_id} - BATCH INTERPOLATION")
    logger.info(f"📁 Files assigned: {{len(files_to_process)}}")
    logger.info(f"🔧 Workers: {{max_workers}}")
    
    # Show file assignments
    for i, file_path in enumerate(files_to_process, 1):
        logger.info(f"  {{i:2d}}. {{file_path}}")
    
    # Initialize batch manager
    manager = BatchInterpolationManager(interpolation_script, max_workers)
    
    # Customize output directories for this machine
    manager.temp_dir = Path("/fs2/ftype_mflf/shared_data/interpolated/machine_{machine_id}_temp")
    manager.backup_dir = Path("/fs2/ftype_mflf/shared_data/interpolated/machine_{machine_id}_backup")
    manager.verification_log = Path("/fs2/ftype_mflf/shared_data/interpolated/machine_{machine_id}_log.txt")
    
    # Create directories
    manager.temp_dir.mkdir(parents=True, exist_ok=True)
    manager.backup_dir.mkdir(parents=True, exist_ok=True)
    
    # Run production batch
    print(f"\\n🚀 Starting PRODUCTION batch for Machine {machine_id}")
    print(f"⚠️  This will process {{len(files_to_process)}} files and REPLACE originals")
    
    result = manager.process_batch(files_to_process, test_mode=False)
    
    logger.info(f"\\n✅ MACHINE {machine_id} COMPLETED!")
    logger.info(f"📊 Success rate: {{result['successful']}}/{{result['total_files']}}")
    logger.info(f"⏱️  Total time: {{result['processing_time']/3600:.1f}} hours")
    logger.info(f"📄 Log: {{manager.verification_log}}")
    
    return result

if __name__ == "__main__":
    run_machine_{machine_id}_batch()
'''
        
        with open(run_script_path, 'w') as f:
            f.write(run_script_content)
        
        run_script_path.chmod(0o755)  # Make executable
        
        # Create setup instructions for this machine
        setup_instruction = {
            'machine_id': machine_id,
            'files_assigned': dist['file_count'],
            'file_list': str(file_list_path),
            'run_script': str(run_script_path),
            'estimated_time_hours': round(dist['file_count'] * 5 / 60, 1),  # 5 min per file estimate
            'commands': [
                f"# Machine {machine_id} Setup & Run Commands",
                "",
                "# 1. Navigate to project directory",
                "cd /fs1/tor_project/model_foresttype",
                "",
                "# 2. Activate conda environment", 
                "eval \"$(conda shell.bash hook)\"",
                "conda activate gisworkspace-cupy",
                "",
                "# 3. Run machine-specific batch processing",
                f"python {run_script_path}",
                "",
                f"# Expected: {dist['file_count']} files, ~{round(dist['file_count'] * 5 / 60, 1)} hours",
                f"# Output: /fs2/ftype_mflf/shared_data/interpolated/machine_{machine_id}_log.txt"
            ]
        }
        
        setup_instructions.append(setup_instruction)
        
        print(f"🤖 Machine {machine_id}:")
        print(f"   📄 Files: {dist['file_count']} ({dist['start_index']}-{dist['end_index']})")
        print(f"   📝 List: {file_list_path}")
        print(f"   🚀 Script: {run_script_path}")
        print(f"   ⏱️  Est. time: ~{setup_instruction['estimated_time_hours']} hours")
        print()
    
    # Create master setup file
    master_setup_path = output_path / "SETUP_INSTRUCTIONS.md"
    with open(master_setup_path, 'w') as f:
        f.write("# 🚀 DISTRIBUTED BATCH INTERPOLATION SETUP\n\n")
        f.write(f"## 📊 DISTRIBUTION SUMMARY\n")
        f.write(f"- **Total files:** {sum(d['file_count'] for d in distributions)}\n")
        f.write(f"- **Machines:** {len(distributions)}\n")
        f.write(f"- **Estimated total time:** ~{max(s['estimated_time_hours'] for s in setup_instructions)} hours (parallel)\n")
        f.write(f"- **Sequential time saved:** ~{sum(s['estimated_time_hours'] for s in setup_instructions) - max(s['estimated_time_hours'] for s in setup_instructions):.1f} hours\n\n")
        
        for setup in setup_instructions:
            f.write(f"## 🤖 MACHINE {setup['machine_id']}\n")
            f.write(f"- **Files assigned:** {setup['files_assigned']}\n")
            f.write(f"- **Estimated time:** ~{setup['estimated_time_hours']} hours\n")
            f.write(f"- **File list:** `{setup['file_list']}`\n")
            f.write(f"- **Run script:** `{setup['run_script']}`\n\n")
            f.write("### Commands:\n")
            f.write("```bash\n")
            for cmd in setup['commands']:
                f.write(f"{cmd}\n")
            f.write("```\n\n")
            f.write("---\n\n")
        
        f.write("## 🔍 MONITORING PROGRESS\n\n")
        f.write("### Check Status:\n")
        f.write("```bash\n")
        for i in range(1, len(distributions) + 1):
            f.write(f"# Machine {i} log\n")
            f.write(f"tail -f /fs2/ftype_mflf/shared_data/interpolated/machine_{i}_log.txt\n\n")
        f.write("```\n\n")
        
        f.write("### Collect Results:\n")
        f.write("```bash\n")
        f.write("# Count completed files\n")
        for i in range(1, len(distributions) + 1):
            f.write(f"grep 'SUCCESS_REPLACED' /fs2/ftype_mflf/shared_data/interpolated/machine_{i}_log.txt | wc -l\n")
        f.write("```\n\n")
    
    print(f"📋 Master setup instructions: {master_setup_path}")
    
    return setup_instructions, master_setup_path

def main():
    """Main distribution function."""
    
    file_list_path = "/fs2/ftype_mflf/shared_data/interpolated/CORRECTED_files_needing_interpolation.txt"
    num_machines = 4
    
    # Distribute files
    distributions = distribute_files_across_machines(file_list_path, num_machines)
    
    # Create machine-specific files and scripts
    setup_instructions, master_file = create_machine_specific_files(distributions)
    
    print(f"✅ DISTRIBUTION COMPLETE!")
    print(f"📋 Setup guide: {master_file}")
    print()
    print(f"🎯 NEXT STEPS:")
    print(f"1. Copy setup instructions to other machines")
    print(f"2. Run each machine's script simultaneously")
    print(f"3. Monitor progress via log files")
    print(f"4. Collect results when all complete")
    
    return distributions, setup_instructions

if __name__ == "__main__":
    main()