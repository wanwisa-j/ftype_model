#!/usr/bin/env python3
"""
Raster Analysis and Visualization Script
Plot LAI, FAPAR, FCOVER rasters for zones 1 and 2 across years to identify root causes
"""

import rasterio
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

def plot_raster_band(raster_path, band_num=1, title="", vmin=None, vmax=None, cmap='viridis'):
    """Plot a single band from a raster file."""
    try:
        with rasterio.open(raster_path) as src:
            data = src.read(band_num)
            
            # Handle no-data values for display
            display_data = data.copy().astype(float)
            nodata_mask = (data == -9999) | (data < -1000)  # Conservative no-data detection
            display_data[nodata_mask] = np.nan
            
            # Auto-scale if not provided
            if vmin is None or vmax is None:
                valid_data = display_data[~np.isnan(display_data)]
                if len(valid_data) > 0:
                    if vmin is None:
                        vmin = np.percentile(valid_data, 2)
                    if vmax is None:
                        vmax = np.percentile(valid_data, 98)
                else:
                    vmin, vmax = -1, 1  # Default range if no valid data
            
            # Create the plot
            plt.imshow(display_data, cmap=cmap, vmin=vmin, vmax=vmax, interpolation='nearest')
            plt.colorbar(label='Value', shrink=0.8)
            plt.title(f"{title}\nBand {band_num}")
            
            # Add statistics
            valid_pixels = np.sum(~nodata_mask)
            total_pixels = data.size
            valid_pct = valid_pixels / total_pixels * 100
            
            plt.text(0.02, 0.98, f"Valid: {valid_pct:.1f}%", 
                    transform=plt.gca().transAxes, 
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
                    verticalalignment='top', fontsize=8)
            
            if len(valid_data) > 0:
                plt.text(0.02, 0.88, f"Range: [{valid_data.min():.3f}, {valid_data.max():.3f}]", 
                        transform=plt.gca().transAxes,
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
                        verticalalignment='top', fontsize=8)
            
            plt.axis('off')
            
    except Exception as e:
        plt.text(0.5, 0.5, f"Error: {str(e)}", ha='center', va='center', 
                transform=plt.gca().transAxes, fontsize=10, color='red')
        plt.title(f"{title} - ERROR")
        plt.axis('off')

def create_zone_comparison_plots(base_dir, feature, years, bands_to_plot=[1, 11, 21]):
    """Create comparison plots for both zones across years."""
    zone1_dir = Path(base_dir) / "mfl_cmf_area_project_2025_chr_zone_1" / feature
    zone2_dir = Path(base_dir) / "mfl_cmf_area_project_2025_chr_zone_2" / feature
    
    n_years = len(years)
    n_bands = len(bands_to_plot)
    
    # Create large figure
    fig = plt.figure(figsize=(20, 6 * n_years))
    
    plot_idx = 1
    
    for year in years:
        # Find files for this year
        zone1_files = list(zone1_dir.glob(f"cube_biophys_{feature}_{year}_*_21bands.tif"))
        zone2_files = list(zone2_dir.glob(f"cube_biophys_{feature}_{year}_*_21bands.tif"))
        
        zone1_file = zone1_files[0] if zone1_files else None
        zone2_file = zone2_files[0] if zone2_files else None
        
        for band in bands_to_plot:
            # Zone 1 plot
            plt.subplot(n_years, n_bands * 2, plot_idx)
            if zone1_file and zone1_file.exists():
                plot_raster_band(zone1_file, band, f"Zone 1 - {feature.upper()} {year}", 
                               cmap='RdYlGn')
            else:
                plt.text(0.5, 0.5, "File Not Found", ha='center', va='center', 
                        transform=plt.gca().transAxes, fontsize=12, color='red')
                plt.title(f"Zone 1 - {feature.upper()} {year} Band {band}")
                plt.axis('off')
            
            # Zone 2 plot  
            plt.subplot(n_years, n_bands * 2, plot_idx + 1)
            if zone2_file and zone2_file.exists():
                plot_raster_band(zone2_file, band, f"Zone 2 - {feature.upper()} {year}",
                               cmap='RdYlGn')
            else:
                plt.text(0.5, 0.5, "File Not Found", ha='center', va='center',
                        transform=plt.gca().transAxes, fontsize=12, color='red')
                plt.title(f"Zone 2 - {feature.upper()} {year} Band {band}")
                plt.axis('off')
            
            plot_idx += 2
    
    plt.tight_layout()
    return fig

def create_temporal_analysis_plots(base_dir, feature, year=2024, zones=['mfl_cmf_area_project_2025_chr_zone_1', 'mfl_cmf_area_project_2025_chr_zone_2']):
    """Create temporal analysis plots showing all 21 bands for a specific year."""
    
    fig, axes = plt.subplots(len(zones), 21, figsize=(42, 4 * len(zones)))
    if len(zones) == 1:
        axes = axes.reshape(1, -1)
    
    for zone_idx, zone in enumerate(zones):
        zone_dir = Path(base_dir) / zone / feature
        files = list(zone_dir.glob(f"cube_biophys_{feature}_{year}_*_21bands.tif"))
        
        if not files:
            for band in range(21):
                axes[zone_idx, band].text(0.5, 0.5, "No File", ha='center', va='center', 
                                        transform=axes[zone_idx, band].transAxes, 
                                        fontsize=8, color='red')
                axes[zone_idx, band].set_title(f"{zone[-6:]}\nBand {band+1}", fontsize=8)
                axes[zone_idx, band].axis('off')
            continue
        
        raster_file = files[0]
        
        try:
            with rasterio.open(raster_file) as src:
                for band in range(21):
                    data = src.read(band + 1)
                    
                    # Handle no-data
                    display_data = data.copy().astype(float)
                    nodata_mask = (data == -9999) | (data < -1000)
                    display_data[nodata_mask] = np.nan
                    
                    # Plot
                    im = axes[zone_idx, band].imshow(display_data, cmap='RdYlGn', 
                                                   interpolation='nearest')
                    
                    # Statistics
                    valid_pixels = np.sum(~nodata_mask)
                    total_pixels = data.size
                    valid_pct = valid_pixels / total_pixels * 100
                    
                    axes[zone_idx, band].set_title(f"{zone[-6:]}\nBand {band+1}\n{valid_pct:.1f}% valid", 
                                                 fontsize=8)
                    axes[zone_idx, band].axis('off')
                    
                    # Add small colorbar
                    if valid_pct > 0:
                        valid_data = display_data[~np.isnan(display_data)]
                        if len(valid_data) > 10:  # Only if enough valid data
                            cbar = plt.colorbar(im, ax=axes[zone_idx, band], shrink=0.5)
                            cbar.ax.tick_params(labelsize=6)
        
        except Exception as e:
            for band in range(21):
                axes[zone_idx, band].text(0.5, 0.5, f"Error\n{str(e)[:20]}", 
                                        ha='center', va='center',
                                        transform=axes[zone_idx, band].transAxes, 
                                        fontsize=6, color='red')
                axes[zone_idx, band].set_title(f"{zone[-6:]}\nBand {band+1}", fontsize=8)
                axes[zone_idx, band].axis('off')
    
    plt.suptitle(f'{feature.upper()} - All 21 Bands - Year {year}', fontsize=16)
    plt.tight_layout()
    return fig

def create_nodata_pattern_analysis(base_dir, zones, features, years):
    """Create analysis of no-data patterns across zones, features, and years."""
    
    fig, axes = plt.subplots(len(features), len(zones), figsize=(12, 4 * len(features)))
    if len(features) == 1:
        axes = axes.reshape(1, -1)
    if len(zones) == 1:
        axes = axes.reshape(-1, 1)
    
    nodata_summary = {}
    
    for feat_idx, feature in enumerate(features):
        nodata_summary[feature] = {}
        
        for zone_idx, zone in enumerate(zones):
            zone_dir = Path(base_dir) / zone / feature
            
            year_data = []
            valid_percentages = []
            
            for year in years:
                files = list(zone_dir.glob(f"cube_biophys_{feature}_{year}_*_21bands.tif"))
                
                if files:
                    try:
                        with rasterio.open(files[0]) as src:
                            # Read first band to get overall pattern
                            data = src.read(1)
                            valid_pixels = np.sum((data != -9999) & (data > -1000))
                            total_pixels = data.size
                            valid_pct = valid_pixels / total_pixels * 100
                            
                            year_data.append(year)
                            valid_percentages.append(valid_pct)
                    except:
                        year_data.append(year)
                        valid_percentages.append(0)
                else:
                    year_data.append(year)
                    valid_percentages.append(0)
            
            nodata_summary[feature][zone] = {
                'years': year_data,
                'valid_percentages': valid_percentages
            }
            
            # Plot
            ax = axes[feat_idx, zone_idx] if len(features) > 1 and len(zones) > 1 else \
                 axes[feat_idx] if len(zones) == 1 else \
                 axes[zone_idx] if len(features) == 1 else axes
            
            bars = ax.bar(year_data, valid_percentages, 
                         color=['green' if x > 50 else 'orange' if x > 10 else 'red' 
                               for x in valid_percentages])
            
            ax.set_title(f'{feature.upper()}\n{zone.split("_")[-1]}')
            ax.set_xlabel('Year')
            ax.set_ylabel('Valid Data (%)')
            ax.set_ylim(0, 100)
            ax.grid(True, alpha=0.3)
            
            # Add percentage labels on bars
            for bar, pct in zip(bars, valid_percentages):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                       f'{pct:.1f}%', ha='center', va='bottom', fontsize=8)
    
    plt.suptitle('Data Quality Analysis: Valid Data Percentage by Zone and Year', 
                 fontsize=14)
    plt.tight_layout()
    
    return fig, nodata_summary

def main():
    """Main plotting function."""
    base_dir = "/fs2/ftype_mflf/shared_data/datacube/2025/1.2.0"
    output_dir = Path("/fs1/tor_project/model_foresttype/plots")
    output_dir.mkdir(exist_ok=True, parents=True)
    
    features = ['lai', 'fapar', 'fcover']
    zones = ['mfl_cmf_area_project_2025_chr_zone_1', 'mfl_cmf_area_project_2025_chr_zone_2']
    years = [2019, 2020, 2021, 2022, 2023, 2024, 2025]
    
    print("🎨 CREATING RASTER ANALYSIS PLOTS")
    print(f"📁 Output directory: {output_dir}")
    print(f"🎯 Features: {features}")
    print(f"🌍 Zones: {zones}")
    print(f"📅 Years: {years}")
    
    # 1. Create no-data pattern analysis (overview)
    print("\n📊 Creating no-data pattern analysis...")
    fig, summary = create_nodata_pattern_analysis(base_dir, zones, features, years)
    save_path = output_dir / 'nodata_pattern_analysis.png'
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"   ✅ Saved: {save_path}")
    plt.close(fig)
    
    # 2. Create temporal analysis for recent year (all 21 bands)
    print("\n⏱️ Creating temporal analysis plots (all bands)...")
    for feature in features:
        print(f"   Processing {feature.upper()}...")
        fig = create_temporal_analysis_plots(base_dir, feature, year=2024, zones=zones)
        save_path = output_dir / f'{feature}_temporal_analysis_2024_allbands.png'
        fig.savefig(save_path, dpi=200, bbox_inches='tight')  # Lower DPI for large plots
        print(f"   ✅ Saved: {save_path}")
        plt.close(fig)
    
    # 3. Create zone comparison plots (selected bands across years)
    print("\n🔄 Creating zone comparison plots...")
    bands_to_show = [1, 11, 21]  # Beginning, middle, end of season
    
    for feature in features:
        print(f"   Processing {feature.upper()} comparison...")
        fig = create_zone_comparison_plots(base_dir, feature, years[:4], bands_to_show)  # First 4 years
        save_path = output_dir / f'{feature}_zone_comparison_2019-2022.png'
        fig.savefig(save_path, dpi=200, bbox_inches='tight')
        print(f"   ✅ Saved: {save_path}")
        plt.close(fig)
        
        # Recent years
        fig = create_zone_comparison_plots(base_dir, feature, years[4:], bands_to_show)  # Last 3 years
        save_path = output_dir / f'{feature}_zone_comparison_2023-2025.png'
        fig.savefig(save_path, dpi=200, bbox_inches='tight')
        print(f"   ✅ Saved: {save_path}")
        plt.close(fig)
    
    # 4. Print summary statistics
    print("\n📈 DATA QUALITY SUMMARY:")
    print("="*60)
    
    for feature in features:
        print(f"\n{feature.upper()}:")
        for zone in zones:
            zone_short = zone.split('_')[-1]
            data = summary[feature][zone]
            avg_valid = np.mean(data['valid_percentages'])
            min_valid = min(data['valid_percentages'])
            max_valid = max(data['valid_percentages'])
            
            print(f"  {zone_short}: Avg={avg_valid:.1f}%, Range=[{min_valid:.1f}%-{max_valid:.1f}%]")
            
            # Identify problematic years
            bad_years = [year for year, pct in zip(data['years'], data['valid_percentages']) if pct < 10]
            if bad_years:
                print(f"    ⚠️ Problematic years (<10% valid): {bad_years}")
    
    print(f"\n✅ All plots saved to: {output_dir}")
    print("🔍 Key files to examine:")
    print(f"   - nodata_pattern_analysis.png: Overview of data quality issues")
    print(f"   - *_temporal_analysis_2024_allbands.png: Detailed band-by-band view")
    print(f"   - *_zone_comparison_*.png: Side-by-side zone comparisons")

if __name__ == "__main__":
    main()