#!/bin/bash

# Comprehensive Forest Type Dataset Extraction - Single Machine
# Extracts ALL zones and ALL indices for complete dataset

echo "🚀 Starting Comprehensive Forest Type Dataset Extraction"
echo "========================================================"
echo "Machine: Single machine comprehensive extraction"
echo "Config: config/comprehensive_single_config.yaml"
echo "Timestamp: $(date '+%Y%m%d_%H%M%S')"
echo "Zones: ALL (8 zones total)"
echo "Features: ALL (8 time series + 2 static = 10 features)"
echo "========================================================"

# Check environment
echo "🔍 Checking environment..."
if ! command -v conda &> /dev/null; then
    echo "❌ Conda not found"
    exit 1
fi

# Activate conda environment
echo "🔧 Activating conda environment..."
source $(conda info --base)/etc/profile.d/conda.sh
conda activate gisworkspace-cupy

if [ $? -ne 0 ]; then
    echo "❌ Failed to activate conda environment"
    exit 1
fi

# Verify Python packages
echo "📦 Verifying Python packages..."
python -c "import rasterio, geopandas, pandas, numpy, yaml; print('✅ All packages available')" || {
    echo "❌ Missing required packages"
    exit 1
}

# Check data directories
echo "📁 Checking data directories..."
if [ ! -d "/fs2/ftype_mflf/shared_data/datacube" ]; then
    echo "❌ Datacube directory not found"
    exit 1
fi

if [ ! -d "/fs1/tor_project/model_foresttype/data" ]; then
    echo "❌ Labels directory not found"
    exit 1
fi

echo "✅ Pre-flight checks completed successfully"
echo ""

# Create output directories
mkdir -p /fs2/ftype_mflf/shared_data/dataset/1.2.0/{raw,processed,metadata}
mkdir -p /fs1/tor_project/model_foresttype/logs

echo "🏗️  Starting comprehensive extraction process..."
echo "⏰ Start time: $(date)"
echo ""

# Run the extraction
python scripts/extract_forest_type_dataset.py config/comprehensive_single_config.yaml

# Check if extraction was successful
if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Comprehensive extraction completed successfully!"
    echo "⏰ End time: $(date)"
    echo "📊 Check output in: /fs2/ftype_mflf/shared_data/dataset/1.2.0/"
    echo "📝 Log file: /fs1/tor_project/model_foresttype/logs/extraction_comprehensive_*.log"
    echo ""
    echo "🎉 Complete forest type dataset with ALL indices ready!"
    echo "📋 Summary:"
    echo "   - All 8 zones processed"
    echo "   - All 10 features extracted (8 time series + 2 static)"
    echo "   - 7 years of data (2019-2025)"
    echo "   - Output location: /fs2/ftype_mflf/shared_data/dataset/1.2.0/"
else
    echo ""
    echo "❌ Extraction failed!"
    echo "📝 Check log file: /fs1/tor_project/model_foresttype/logs/extraction_comprehensive_*.log"
    exit 1
fi
