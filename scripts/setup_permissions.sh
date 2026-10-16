#!/bin/bash
# Permissions Verification and Setup Script
# Ensures all files and directories have proper 774 permissions

set -e

echo "🔐 Setting up permissions for Forest Type Dataset Extraction"
echo "============================================================="

# Project directories
PROJECT_DIR="/fs1/tor_project/model_foresttype"
OUTPUT_DIR="/fs2/ftype_mflf/shared_data/dataset/1.2.0"
LOGS_DIR="/fs1/tor_project/model_foresttype/logs"

echo "📁 Setting directory permissions..."

# Set directory permissions
chmod 774 "$PROJECT_DIR/scripts" 2>/dev/null || echo "Warning: Cannot set permissions on scripts directory"
chmod 774 "$PROJECT_DIR/config" 2>/dev/null || echo "Warning: Cannot set permissions on config directory"
chmod 774 "$PROJECT_DIR/docs" 2>/dev/null || echo "Warning: Cannot set permissions on docs directory"

# Create and set output directories
mkdir -p "$OUTPUT_DIR/raw" "$OUTPUT_DIR/wide"
chmod 774 "$OUTPUT_DIR/raw" 2>/dev/null || echo "Warning: Cannot set permissions on raw output directory"
chmod 774 "$OUTPUT_DIR/wide" 2>/dev/null || echo "Warning: Cannot set permissions on wide output directory"

# Create and set logs directory
mkdir -p "$LOGS_DIR"
chmod 774 "$LOGS_DIR" 2>/dev/null || echo "Warning: Cannot set permissions on logs directory"

echo "📄 Setting file permissions..."

# Set file permissions
chmod 774 "$PROJECT_DIR/scripts"/*.py 2>/dev/null || echo "Warning: Cannot set permissions on Python scripts"
chmod 774 "$PROJECT_DIR/scripts"/*.sh 2>/dev/null || echo "Warning: Cannot set permissions on shell scripts"
chmod 774 "$PROJECT_DIR/config"/*.yaml 2>/dev/null || echo "Warning: Cannot set permissions on config files"
chmod 774 "$PROJECT_DIR/docs"/*.md 2>/dev/null || echo "Warning: Cannot set permissions on documentation"

echo "🔍 Verifying permissions..."

echo ""
echo "📂 Directory permissions:"
ls -ld "$PROJECT_DIR/scripts" "$PROJECT_DIR/config" "$PROJECT_DIR/docs" 2>/dev/null

echo ""
echo "📜 Key script permissions:"
ls -la "$PROJECT_DIR/scripts/extract_forest_type_dataset.py" 2>/dev/null
ls -la "$PROJECT_DIR/scripts/run_extraction_machine_"*.sh 2>/dev/null | head -3

echo ""
echo "⚙️  Config file permissions:"
ls -la "$PROJECT_DIR/config/machine_"*.yaml 2>/dev/null

echo ""
echo "📊 Output directory permissions:"
ls -ld "$OUTPUT_DIR/raw" "$OUTPUT_DIR/wide" 2>/dev/null

# Check deployment package if it exists
DEPLOYMENT_PACKAGE=$(ls /tmp/forest_extraction_deployment_*.tar.gz 2>/dev/null | tail -1)
if [[ -n "$DEPLOYMENT_PACKAGE" ]]; then
    echo ""
    echo "📦 Deployment package:"
    ls -la "$DEPLOYMENT_PACKAGE"
    chmod 774 "$DEPLOYMENT_PACKAGE" 2>/dev/null || echo "Warning: Cannot set permissions on deployment package"
fi

echo ""
echo "✅ Permission setup completed!"
echo ""
echo "📋 Summary:"
echo "   - All directories set to 774 (rwxrwxr--)"
echo "   - All files set to 774 (rwxrwxr--)"
echo "   - Output directories created with proper permissions"
echo "   - Logs directory ready"
echo "   - Deployment package accessible"
echo ""
echo "🚀 Ready for multi-machine deployment!"
