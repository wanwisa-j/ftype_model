#!/bin/bash
"""
Sequential Zone Verification Script
===================================

This script runs datacube verification for all zones sequentially (one after another).
Useful when system resources are limited or when you want to avoid parallel processing.

Features:
- Sequential execution (no parallel processing)
- Lower resource usage
- Detailed progress tracking
- Individual zone result collection
- Consolidated summary report

Usage:
    chmod +x verify_all_zones_sequential.sh
    ./verify_all_zones_sequential.sh

Options:
    Set STOP_ON_FAILURE=true to stop processing if any zone fails
    Set VERBOSE_OUTPUT=true for detailed logging

Author: Generated for Forest Type Model Pipeline v1.2.0
Date: 2025-09-10
"""

# Configuration
DATACUBE_BASE_PATH="/fs2/ftype_mflf/shared_data/datacube/2025/1.2.0"
SCRIPT_DIR="/fs1/tor_project/model_foresttype"
VERIFICATION_SCRIPT="datacube_zone_verification.py"
OUTPUT_BASE_DIR="/fs1/tor_project/model_foresttype/verification_results_sequential"

# Options
STOP_ON_FAILURE=false
VERBOSE_OUTPUT=false

# Zones to verify
ZONES=("zone_1" "zone_2" "zone_3" "zone_4" "zone_5")

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

echo "=============================================="
echo "Sequential Zone Verification for Datacube v1.2.0"
echo "=============================================="
echo "Datacube path: $DATACUBE_BASE_PATH"
echo "Output directory: $OUTPUT_BASE_DIR"
echo "Zones to verify: ${ZONES[*]}"
echo "Stop on failure: $STOP_ON_FAILURE"
echo "Verbose output: $VERBOSE_OUTPUT"
echo "Start time: $(date)"
echo "=============================================="

# Create directories
mkdir -p "$OUTPUT_BASE_DIR"

# Progress tracking
declare -A zone_results
declare -A zone_duration
total_zones=${#ZONES[@]}
current_zone=0
success_count=0
failure_count=0
skipped_count=0

# Function to run verification for a single zone
run_zone_verification() {
    local zone="$1"
    local zone_dir="$DATACUBE_BASE_PATH/$zone"
    local output_dir="$OUTPUT_BASE_DIR/$zone"
    
    echo ""
    echo -e "${CYAN}[$(date '+%H:%M:%S')] Processing $zone (${current_zone}/${total_zones})${NC}"
    echo "Zone directory: $zone_dir"
    echo "Output directory: $output_dir"
    
    # Check if zone directory exists
    if [ ! -d "$zone_dir" ]; then
        echo -e "${RED}❌ Zone directory not found: $zone_dir${NC}"
        zone_results["$zone"]="MISSING"
        return 1
    fi
    
    # Create output directory
    mkdir -p "$output_dir"
    
    # Record start time
    local start_time=$(date +%s)
    
    # Run verification
    echo -e "${BLUE}🔍 Starting verification...${NC}"
    
    cd "$SCRIPT_DIR"
    
    # Activate conda environment
    source /opt/miniconda3/etc/profile.d/conda.sh
    conda activate gisworkspace-cupy
    
    if [ "$VERBOSE_OUTPUT" = true ]; then
        echo "Running: python $VERIFICATION_SCRIPT $zone_dir --output $output_dir --verbose"
        python "$VERIFICATION_SCRIPT" "$zone_dir" --output "$output_dir" --verbose
    else
        echo "Running verification (use VERBOSE_OUTPUT=true for detailed logs)..."
        python "$VERIFICATION_SCRIPT" "$zone_dir" --output "$output_dir" > "$output_dir/${zone}_verification.log" 2>&1
    fi
    
    local exit_code=$?
    local end_time=$(date +%s)
    local duration=$((end_time - start_time))
    
    zone_duration["$zone"]=$duration
    
    # Determine result status
    if [ $exit_code -eq 0 ]; then
        zone_results["$zone"]="PASS"
        echo -e "${GREEN}✅ $zone verification completed successfully (${duration}s)${NC}"
        ((success_count++))
    else
        zone_results["$zone"]="FAIL"
        echo -e "${RED}❌ $zone verification failed (exit code: $exit_code, ${duration}s)${NC}"
        ((failure_count++))
        
        if [ "$STOP_ON_FAILURE" = true ]; then
            echo -e "${RED}🛑 Stopping due to failure (STOP_ON_FAILURE=true)${NC}"
            return $exit_code
        fi
    fi
    
    # Quick result summary if not verbose
    if [ "$VERBOSE_OUTPUT" = false ] && [ -f "$output_dir/${zone}_verification_results.json" ]; then
        echo "Quick summary:"
        python3 -c "
import json
try:
    with open('$output_dir/${zone}_verification_results.json', 'r') as f:
        data = json.load(f)
    summary = data.get('summary', {})
    print(f\"  Features: {summary.get('total_features_found', 'N/A')}/{summary.get('total_features_expected', 'N/A')}\")
    print(f\"  Files: {summary.get('total_files', 'N/A')}\")
    print(f\"  Issues: {summary.get('issues_count', 'N/A')}\")
except: pass
" 2>/dev/null
    fi
    
    return $exit_code
}

# Main processing loop
echo ""
echo -e "${BLUE}Starting sequential verification of all zones...${NC}"

overall_start_time=$(date +%s)

for zone in "${ZONES[@]}"; do
    ((current_zone++))
    
    # Progress indicator
    progress=$((current_zone * 100 / total_zones))
    echo ""
    echo "=========================================="
    echo -e "${CYAN}Progress: $progress% ($current_zone/$total_zones)${NC}"
    echo "=========================================="
    
    run_zone_verification "$zone"
    verification_exit_code=$?
    
    # Check if we should stop
    if [ "$STOP_ON_FAILURE" = true ] && [ $verification_exit_code -ne 0 ]; then
        echo ""
        echo -e "${RED}Verification stopped due to failure in $zone${NC}"
        # Mark remaining zones as skipped
        for ((i=current_zone; i<total_zones; i++)); do
            local remaining_zone="${ZONES[$i]}"
            zone_results["$remaining_zone"]="SKIPPED"
            ((skipped_count++))
        done
        break
    fi
done

overall_end_time=$(date +%s)
total_duration=$((overall_end_time - overall_start_time))

echo ""
echo "=============================================="
echo "Sequential Verification Complete"
echo "=============================================="
echo "Total time: ${total_duration}s ($(($total_duration / 60))m $(($total_duration % 60))s)"
echo ""

# Results summary
echo "RESULTS SUMMARY:"
echo "----------------"
echo -e "Total zones: $total_zones"
echo -e "${GREEN}Successful: $success_count${NC}"
echo -e "${RED}Failed: $failure_count${NC}"
if [ $skipped_count -gt 0 ]; then
    echo -e "${YELLOW}Skipped: $skipped_count${NC}"
fi
echo ""

echo "INDIVIDUAL ZONE RESULTS:"
echo "------------------------"
for zone in "${ZONES[@]}"; do
    local result=${zone_results["$zone"]}
    local duration=${zone_duration["$zone"]:-"N/A"}
    
    case $result in
        "PASS")
            echo -e "${GREEN}✅ $zone: PASS${NC} (${duration}s)"
            ;;
        "FAIL") 
            echo -e "${RED}❌ $zone: FAIL${NC} (${duration}s)"
            ;;
        "MISSING")
            echo -e "${RED}🚫 $zone: MISSING DIRECTORY${NC}"
            ;;
        "SKIPPED")
            echo -e "${YELLOW}⏭️  $zone: SKIPPED${NC}"
            ;;
        *)
            echo -e "${YELLOW}❓ $zone: UNKNOWN${NC}"
            ;;
    esac
done

# Generate consolidated report
SUMMARY_REPORT="$OUTPUT_BASE_DIR/sequential_verification_summary.txt"

echo ""
echo "Generating consolidated report..."

{
    echo "=============================================="
    echo "SEQUENTIAL DATACUBE VERIFICATION SUMMARY"
    echo "Generated: $(date)"
    echo "=============================================="
    echo ""
    echo "EXECUTION DETAILS"
    echo "-----------------"
    echo "Datacube path: $DATACUBE_BASE_PATH"
    echo "Execution mode: Sequential"
    echo "Stop on failure: $STOP_ON_FAILURE"
    echo "Total execution time: ${total_duration}s"
    echo "Start time: $(date -d @$overall_start_time)"
    echo "End time: $(date -d @$overall_end_time)"
    echo ""
    
    echo "OVERALL RESULTS"
    echo "---------------"
    echo "Total zones: $total_zones"
    echo "Successful: $success_count"
    echo "Failed: $failure_count"
    echo "Skipped: $skipped_count"
    echo "Success rate: $(( (success_count * 100) / total_zones ))%"
    echo ""
    
    echo "ZONE RESULTS"
    echo "------------"
    for zone in "${ZONES[@]}"; do
        local result=${zone_results["$zone"]}
        local duration=${zone_duration["$zone"]:-"N/A"}
        printf "%-10s: %-10s (%ss)\n" "$zone" "$result" "$duration"
    done
    echo ""
    
    echo "DETAILED ZONE INFORMATION"
    echo "-------------------------"
    for zone in "${ZONES[@]}"; do
        local result=${zone_results["$zone"]}
        if [ "$result" = "PASS" ] || [ "$result" = "FAIL" ]; then
            echo ""
            echo "$zone ($result):"
            echo "$(printf '=%.0s' {1..20})"
            
            local results_file="$OUTPUT_BASE_DIR/$zone/${zone}_verification_results.json"
            if [ -f "$results_file" ]; then
                echo "Results file: $results_file"
                
                # Extract key info
                python3 -c "
import json
try:
    with open('$results_file', 'r') as f:
        data = json.load(f)
    summary = data.get('summary', {})
    print(f\"Features: {summary.get('total_features_found', 'N/A')}/{summary.get('total_features_expected', 'N/A')}\")
    print(f\"Files: {summary.get('total_files', 'N/A')}\")
    print(f\"Issues: {summary.get('issues_count', 'N/A')}\")
    
    if data.get('issues'):
        print('Key issues:')
        for issue in data['issues'][:3]:  # Show first 3 issues
            print(f\"  - {issue}\")
except: 
    print('Could not extract details')
" 2>/dev/null || echo "Details not available"
            else
                echo "Results file not found"
            fi
        fi
    done
    
    echo ""
    echo "FILES GENERATED"
    echo "---------------"
    echo "Individual zone results: $OUTPUT_BASE_DIR/[zone_name]/"
    echo "This summary: $SUMMARY_REPORT"
    echo ""
    
    echo "NEXT STEPS"
    echo "----------"
    if [ $failure_count -eq 0 ]; then
        echo "✅ All zones passed verification successfully!"
        echo "✅ Datacube v1.2.0 is ready for dataset extraction and model training."
    else
        echo "❌ $failure_count zones failed verification."
        echo "📋 Review individual zone reports for detailed failure information."
        echo "🔧 Fix identified issues using the recommendations in zone reports."
        echo "🔄 Re-run verification for failed zones after applying fixes."
        
        echo ""
        echo "Failed zones to investigate:"
        for zone in "${ZONES[@]}"; do
            if [ "${zone_results["$zone"]}" = "FAIL" ]; then
                echo "  - $zone: $OUTPUT_BASE_DIR/$zone/${zone}_verification_report.txt"
            fi
        done
    fi
    
} > "$SUMMARY_REPORT"

echo "Consolidated summary saved to: $SUMMARY_REPORT"
echo ""

# Final status and recommendations
if [ $failure_count -eq 0 ]; then
    echo -e "${GREEN}🎉 ALL ZONES VERIFICATION COMPLETED SUCCESSFULLY!${NC}"
    echo -e "${GREEN}✅ Datacube v1.2.0 is ready for use${NC}"
    echo ""
    echo "Next steps:"
    echo "  1. Proceed with dataset extraction"
    echo "  2. Run model training pipeline"
    exit 0
else
    echo -e "${RED}⚠️  $failure_count out of $total_zones zones failed verification${NC}"
    echo ""
    echo "Recommended actions:"
    echo "  1. Review failed zone reports for specific issues"
    echo "  2. Fix identified problems (missing data, alignment issues, etc.)"
    echo "  3. Re-run verification for failed zones"
    echo "  4. Use single zone verification script for quick retesting:"
    echo "     ./run_single_zone_verification.sh <failed_zone>"
    echo ""
    echo "Failed zones:"
    for zone in "${ZONES[@]}"; do
        if [ "${zone_results["$zone"]}" = "FAIL" ]; then
            echo -e "  ${RED}- $zone${NC}"
        fi
    done
    exit 1
fi