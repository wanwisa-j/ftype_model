#!/usr/bin/env python3
"""
Pre-extraction validation script
Checks availability of all indices across all zones
"""

import os
from pathlib import Path

def check_datacube_availability():
    """Check which indices are available in each zone"""
    
    print("🔍 COMPREHENSIVE DATACUBE AVAILABILITY CHECK")
    print("=" * 60)
    
    # Define expected zones and features
    zones_2024 = ['mfl_cmf_2024']
    zones_2025 = [
        'mfl_cmf_area_project_2025_chr_zone_1',
        'mfl_cmf_area_project_2025_chr_zone_2',
        'zone_1', 'zone_2', 'zone_3', 'zone_4', 'zone_5'
    ]
    
    expected_features = ['lai', 'fapar', 'fcover', 'ndvi', 'gvmi', 'slavi', 'vv', 'vh', 'elevation', 'slope']
    
    datacube_base = Path("/fs2/ftype_mflf/shared_data/datacube")
    
    total_available = 0
    total_expected = 0
    
    # Check 2024 zones
    print("\n📅 2024 DATACUBE:")
    for zone in zones_2024:
        zone_path = datacube_base / "2024" / "1.2.0" / zone
        print(f"\n🗺️  Zone: {zone}")
        print(f"   Path: {zone_path}")
        
        if zone_path.exists():
            print("   ✅ Zone directory exists")
            available_features = []
            for feature in expected_features:
                feature_path = zone_path / feature
                if feature_path.exists():
                    # Check if there are files in the feature directory
                    files = list(feature_path.glob("*.tif"))
                    if files:
                        available_features.append(feature)
                        total_available += 1
                        print(f"   ✅ {feature}: {len(files)} files")
                    else:
                        print(f"   ⚠️  {feature}: directory empty")
                else:
                    print(f"   ❌ {feature}: not found")
                total_expected += 1
            
            print(f"   📊 Available: {len(available_features)}/{len(expected_features)} features")
        else:
            print("   ❌ Zone directory not found")
            total_expected += len(expected_features)
    
    # Check 2025 zones
    print("\n📅 2025 DATACUBE:")
    for zone in zones_2025:
        zone_path = datacube_base / "2025" / "1.2.0" / zone
        print(f"\n🗺️  Zone: {zone}")
        print(f"   Path: {zone_path}")
        
        if zone_path.exists():
            print("   ✅ Zone directory exists")
            available_features = []
            for feature in expected_features:
                feature_path = zone_path / feature
                if feature_path.exists():
                    # Check if there are files in the feature directory
                    files = list(feature_path.glob("*.tif"))
                    if files:
                        available_features.append(feature)
                        total_available += 1
                        print(f"   ✅ {feature}: {len(files)} files")
                    else:
                        print(f"   ⚠️  {feature}: directory empty")
                else:
                    print(f"   ❌ {feature}: not found")
                total_expected += 1
            
            print(f"   📊 Available: {len(available_features)}/{len(expected_features)} features")
        else:
            print("   ❌ Zone directory not found")
            total_expected += len(expected_features)
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 OVERALL SUMMARY")
    print("=" * 60)
    print(f"Total zones checked: {len(zones_2024) + len(zones_2025)}")
    print(f"Total features expected: {total_expected}")
    print(f"Total features available: {total_available}")
    print(f"Availability rate: {(total_available/total_expected)*100:.1f}%")
    
    if total_available == total_expected:
        print("✅ ALL FEATURES AVAILABLE - Ready for comprehensive extraction!")
    elif total_available > total_expected * 0.8:
        print("⚠️  MOST FEATURES AVAILABLE - Extraction possible with some missing data")
    else:
        print("❌ MANY FEATURES MISSING - Check datacube preparation")
    
    print("\n🚀 Recommended extraction features based on availability:")
    # Get most common available features
    feature_counts = {}
    for datacube_year in ["2024", "2025"]:
        zones = zones_2024 if datacube_year == "2024" else zones_2025
        for zone in zones:
            zone_path = datacube_base / datacube_year / "1.2.0" / zone
            if zone_path.exists():
                for feature in expected_features:
                    feature_path = zone_path / feature
                    if feature_path.exists() and list(feature_path.glob("*.tif")):
                        feature_counts[feature] = feature_counts.get(feature, 0) + 1
    
    # Sort by availability
    sorted_features = sorted(feature_counts.items(), key=lambda x: x[1], reverse=True)
    for feature, count in sorted_features:
        total_zones = len(zones_2024) + len(zones_2025)
        coverage = (count / total_zones) * 100
        print(f"   {feature}: {count}/{total_zones} zones ({coverage:.1f}% coverage)")

if __name__ == "__main__":
    check_datacube_availability()
