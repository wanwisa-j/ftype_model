#!/bin/bash
# Deployment Package Creator for Multi-Machine Forest Type Extraction
# Creates a deployment package for easy distribution to 3 machines

set -e

PACKAGE_DIR="/tmp/forest_extraction_deployment"
PROJECT_DIR="/fs1/tor_project/model_foresttype"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

echo "🚀 Creating Deployment Package for Multi-Machine Extraction"
echo "============================================================"

# Create package directory
rm -rf $PACKAGE_DIR
mkdir -p $PACKAGE_DIR/{scripts,config,docs}

echo "📦 Copying core files..."

# Copy main extraction script
cp $PROJECT_DIR/scripts/extract_forest_type_dataset.py $PACKAGE_DIR/scripts/
cp $PROJECT_DIR/scripts/consolidate_datasets.py $PACKAGE_DIR/scripts/

# Copy machine-specific files
for i in {1..3}; do
    cp $PROJECT_DIR/config/machine_${i}_config.yaml $PACKAGE_DIR/config/
    cp $PROJECT_DIR/scripts/run_extraction_machine_${i}.sh $PACKAGE_DIR/scripts/
done

# Copy documentation
cp $PROJECT_DIR/docs/MULTI_MACHINE_EXTRACTION_GUIDE.md $PACKAGE_DIR/docs/
cp $PROJECT_DIR/docs/AGENTIC_AI_INSTRUCTIONS.md $PACKAGE_DIR/docs/
cp $PROJECT_DIR/docs/DEPLOYMENT_READY.md $PACKAGE_DIR/docs/

# Make scripts executable
chmod +x $PACKAGE_DIR/scripts/*.sh

# Set 774 permissions for all deployment files
chmod -R 774 $PACKAGE_DIR/

echo "📋 Creating deployment instructions..."

# Create machine-specific README files
for i in {1..3}; do
    cat > $PACKAGE_DIR/MACHINE_${i}_README.md << EOF
# Machine ${i} Deployment Instructions

## Quick Start
\`\`\`bash
# Copy files to machine
scp -r forest_extraction_deployment/ user@machine${i}:/fs1/tor_project/model_foresttype/

# On machine ${i}:
cd /fs1/tor_project/model_foresttype
conda activate gisworkspace-cupy
bash scripts/run_extraction_machine_${i}.sh
\`\`\`

## Files for Machine ${i}:
- \`scripts/extract_forest_type_dataset.py\` (core extraction)
- \`config/machine_${i}_config.yaml\` (machine-specific config)
- \`scripts/run_extraction_machine_${i}.sh\` (execution script)

## Expected Runtime: $(if [ $i -eq 3 ]; then echo "2-3 hours"; else echo "3-4 hours"; fi)

## Zones to Process:
EOF

    # Add zone info based on machine
    case $i in
        1) echo "- mfl_cmf_2024" >> $PACKAGE_DIR/MACHINE_${i}_README.md
           echo "- zone_1" >> $PACKAGE_DIR/MACHINE_${i}_README.md  
           echo "- zone_2" >> $PACKAGE_DIR/MACHINE_${i}_README.md ;;
        2) echo "- mfl_cmf_area_project_2025_chr_zone_1" >> $PACKAGE_DIR/MACHINE_${i}_README.md
           echo "- zone_3" >> $PACKAGE_DIR/MACHINE_${i}_README.md
           echo "- zone_4" >> $PACKAGE_DIR/MACHINE_${i}_README.md ;;
        3) echo "- mfl_cmf_area_project_2025_chr_zone_2" >> $PACKAGE_DIR/MACHINE_${i}_README.md
           echo "- zone_5" >> $PACKAGE_DIR/MACHINE_${i}_README.md ;;
    esac
done

# Create consolidated README
cat > $PACKAGE_DIR/README.md << 'EOF'
# Forest Type Dataset Extraction - Multi-Machine Deployment Package

This package contains everything needed to run distributed forest type dataset extraction across 3 machines.

## Package Contents:
- `scripts/` - Core extraction and consolidation scripts
- `config/` - Machine-specific configuration files  
- `docs/` - Comprehensive documentation
- `MACHINE_X_README.md` - Machine-specific instructions

## Quick Deployment:
1. Copy this entire package to each machine
2. Run the machine-specific script simultaneously on all 3 machines
3. After all complete, run consolidation on Machine 1

## Total Expected Output:
- 500,000+ raw records across all zones
- 50,000+ unique spatial points with time series
- Complete LAI/NDVI data for 2019-2025

## Support:
See `docs/AGENTIC_AI_INSTRUCTIONS.md` for detailed execution protocol.
EOF

# Create tarball
echo "📦 Creating deployment tarball..."
cd /tmp
tar -czf forest_extraction_deployment_${TIMESTAMP}.tar.gz forest_extraction_deployment/

# Calculate package size
PACKAGE_SIZE=$(du -sh $PACKAGE_DIR | cut -f1)
TARBALL_SIZE=$(du -sh forest_extraction_deployment_${TIMESTAMP}.tar.gz | cut -f1)

echo ""
echo "✅ DEPLOYMENT PACKAGE CREATED SUCCESSFULLY!"
echo "============================================="
echo "📁 Package directory: $PACKAGE_DIR"
echo "📦 Tarball: /tmp/forest_extraction_deployment_${TIMESTAMP}.tar.gz"
echo "📊 Package size: $PACKAGE_SIZE"
echo "📊 Tarball size: $TARBALL_SIZE"
echo ""
echo "📋 Package contents:"
find $PACKAGE_DIR -type f | sort
echo ""
echo "🚀 Ready for deployment to 3 machines!"
echo ""
echo "Next steps:"
echo "1. Copy package to each machine: scp forest_extraction_deployment_${TIMESTAMP}.tar.gz user@machine:/tmp/"
echo "2. Extract on each machine: tar -xzf /tmp/forest_extraction_deployment_${TIMESTAMP}.tar.gz"
echo "3. Follow machine-specific README instructions"
echo "4. Start extractions simultaneously"
echo "5. Run consolidation after all complete"
