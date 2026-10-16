#!/bin/bash
"""
Align Biophysical Datacubes Script
==================================

This script applies raster_clip_align.py to all biophysical datacubes 
for a specified zone to clip to AOI boundaries and align pixels.

Usage:
    ./align_biophys_datacubes.sh zone_name

Example:
    ./align_biophys_datacubes.sh mfl_cmf_area_project_2025_chr_zone_1
"""

if [ $# -ne 1 ]; then
    echo "Usage: $0 <zone_name>"
    echo "Example: $0 mfl_cmf_area_project_2025_chr_zone_1"
    exit 1
fi

ZONE_NAME="$1"

# Paths
DATACUBE_DIR="/fs2/ftype_mflf/shared_data/datacube/2025/1.2.0/$ZONE_NAME"
AOI_FILE="/fs3/datacenter/ftype_mflf/shared_data/boundary/$ZONE_NAME-fit_areas.parquet"
SCRIPT_PATH="/fs1/tor_project/model_foresttype/src/raster_clip_align.py"

echo "Aligning Biophysical Datacubes for $ZONE_NAME"
echo "============================================="
echo "Datacube directory: $DATACUBE_DIR"
echo "AOI file: $AOI_FILE"
echo ""

# Check if AOI file exists
if [ ! -f "$AOI_FILE" ]; then
    echo "Error: AOI file not found: $AOI_FILE"
    exit 1
fi

# Variables to process
VARIABLES=(lai fapar fcover)

# Activate conda environment
source /opt/miniconda3/etc/profile.d/conda.sh
conda activate gisworkspace-cupy

# Process each variable
for var in "${VARIABLES[@]}"; do
    echo "Processing $var..."
    
    INPUT_DIR="$DATACUBE_DIR/$var"
    OUTPUT_DIR="$DATACUBE_DIR/${var}_aligned"
    
    # Check if input directory exists
    if [ ! -d "$INPUT_DIR" ]; then
        echo "Warning: Input directory not found: $INPUT_DIR"
        continue
    fi
    
    # Create output directory
    mkdir -p "$OUTPUT_DIR"
    
    # Count files to process
    FILE_COUNT=$(find "$INPUT_DIR" -name "*.tif" | wc -l)
    echo "  Found $FILE_COUNT TIF files in $INPUT_DIR"
    
    if [ $FILE_COUNT -eq 0 ]; then
        echo "  No TIF files found, skipping"
        continue
    fi
    
    # Process each TIF file
    file_num=0
    for tif_file in "$INPUT_DIR"/*.tif; do
        if [ -f "$tif_file" ]; then
            file_num=$((file_num + 1))
            filename=$(basename "$tif_file")
            echo "    Processing $file_num/$FILE_COUNT: $filename"
            
            python "$SCRIPT_PATH" \
                --mode single \
                --raster "$tif_file" \
                --aoi "$AOI_FILE" \
                --output "$OUTPUT_DIR" \
                --target-crs EPSG:32647 \
                --pixel-size 10 \
                --resampling bilinear
            
            if [ $? -eq 0 ]; then
                echo "    ✓ Successfully processed $filename"
            else
                echo "    ✗ Failed to process $filename"
            fi
        fi
    done
    
    echo "  Completed $var processing"
    echo ""
done

echo "Alignment completed for $ZONE_NAME"
echo ""

# Summary
echo "Summary:"
echo "========"
for var in "${VARIABLES[@]}"; do
    OUTPUT_DIR="$DATACUBE_DIR/${var}_aligned"
    if [ -d "$OUTPUT_DIR" ]; then
        aligned_count=$(find "$OUTPUT_DIR" -name "*.tif" | wc -l)
        echo "$var: $aligned_count aligned files"
    else
        echo "$var: no output directory"
    fi
done