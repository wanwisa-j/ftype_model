#!/bin/bash

# Forest Type Dataset Extraction Runner
# This script activates the conda environment and runs the dataset extraction

set -e  # Exit on any error

echo "🌲 Forest Type Classification Dataset Extraction"
echo "=============================================="

# Configuration
CONDA_ENV="gisworkspace-cupy"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
CONFIG_FILE="$PROJECT_DIR/config/extraction_config.yaml"
EXTRACTION_SCRIPT="$SCRIPT_DIR/extract_forest_type_dataset.py"

# Default output directory
OUTPUT_DIR="/fs2/ftype_mflf/shared_data/dataset/1.2.0"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --config)
            CONFIG_FILE="$2"
            shift 2
            ;;
        --output)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --env)
            CONDA_ENV="$2"
            shift 2
            ;;
        --verbose|-v)
            VERBOSE="--verbose"
            shift
            ;;
        --help|-h)
            echo "Usage: $0 [options]"
            echo ""
            echo "Options:"
            echo "  --config PATH    Configuration file (default: config/extraction_config.yaml)"
            echo "  --output PATH    Output directory (default: /fs2/ftype_mflf/shared_data/dataset/1.2.0)"
            echo "  --env NAME       Conda environment name (default: gisworkspace-cupy)"
            echo "  --verbose|-v     Enable verbose logging"
            echo "  --help|-h        Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0                                    # Use default configuration"
            echo "  $0 --config my_config.yaml          # Use custom configuration"
            echo "  $0 --output /tmp/dataset --verbose   # Custom output with verbose logging"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Validate files exist
if [[ ! -f "$CONFIG_FILE" ]]; then
    echo "❌ Error: Configuration file not found: $CONFIG_FILE"
    exit 1
fi

if [[ ! -f "$EXTRACTION_SCRIPT" ]]; then
    echo "❌ Error: Extraction script not found: $EXTRACTION_SCRIPT"
    exit 1
fi

# Check if conda is available
if ! command -v conda &> /dev/null; then
    echo "❌ Error: conda is not available. Please install conda or ensure it's in your PATH."
    exit 1
fi

# Initialize conda for bash
eval "$(conda shell.bash hook)"

# Check if environment exists
if ! conda env list | grep -q "^$CONDA_ENV "; then
    echo "❌ Error: Conda environment '$CONDA_ENV' not found."
    echo "Available environments:"
    conda env list
    exit 1
fi

echo "📋 Configuration:"
echo "  Environment: $CONDA_ENV"
echo "  Config file: $CONFIG_FILE"
echo "  Output dir:  $OUTPUT_DIR"
echo "  Script:      $EXTRACTION_SCRIPT"
echo ""

# Activate conda environment
echo "🔧 Activating conda environment: $CONDA_ENV"
conda activate "$CONDA_ENV"

# Verify Python environment
echo "🐍 Python environment info:"
echo "  Python: $(which python)"
echo "  Version: $(python --version)"

# Check required packages
echo "📦 Checking required packages..."
python -c "
import sys
required_packages = ['pandas', 'geopandas', 'rasterio', 'numpy', 'xarray', 'tqdm', 'yaml']
missing = []
for pkg in required_packages:
    try:
        __import__(pkg)
        print(f'  ✅ {pkg}')
    except ImportError:
        print(f'  ❌ {pkg} (missing)')
        missing.append(pkg)

if missing:
    print(f'\\n❌ Missing packages: {missing}')
    print('Please install them with: conda install -c conda-forge ' + ' '.join(missing))
    sys.exit(1)
else:
    print('\\n✅ All required packages are available')
"

if [[ $? -ne 0 ]]; then
    exit 1
fi

echo ""
echo "🚀 Starting dataset extraction..."
echo ""

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Run the extraction script
python "$EXTRACTION_SCRIPT" \
    --config "$CONFIG_FILE" \
    --output "$OUTPUT_DIR" \
    ${VERBOSE:-}

# Check if extraction was successful
if [[ $? -eq 0 ]]; then
    echo ""
    echo "✅ Dataset extraction completed successfully!"
    echo "📁 Results saved to: $OUTPUT_DIR"
    echo ""
    echo "📊 Generated files:"
    find "$OUTPUT_DIR" -name "*.parquet" -o -name "*.json" | sort | while read file; do
        size=$(du -h "$file" | cut -f1)
        echo "  $file ($size)"
    done
else
    echo ""
    echo "❌ Dataset extraction failed!"
    exit 1
fi
