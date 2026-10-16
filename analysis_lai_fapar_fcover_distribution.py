#!/usr/bin/env python3
"""
Data Distribution Analysis for LAI, FAPAR, FCOVER
Analysis of biophysical variables across timesteps and zones
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set style for better plots
plt.style.use('default')
sns.set_palette("husl")

def load_parquet_data(file_path):
    """Load and inspect the parquet file structure."""
    print("="*80)
    print("📂 LOADING PARQUET DATA")
    print("="*80)
    
    df = pd.read_parquet(file_path)
    print(f"✅ Loaded data shape: {df.shape}")
    print(f"📊 Columns: {list(df.columns)}")
    print(f"📅 Years: {df['year'].min()} to {df['year'].max()}")
    print(f"📃 Features: {df['feature'].unique()}")
    print(f"🌍 Unique zones: {df['zone'].nunique()}")
    print(f"   Zone list: {df['zone'].unique()}")
    print(f"🎯 Forest types: {df['ftype_nonmix'].unique()}")
    print(f"📊 Weeks: {df['week'].min()} to {df['week'].max()}")
    
    return df

def extract_timestep_data(df, feature_name):
    """Extract timestep data for a specific feature from long format."""
    # Filter for the specific feature
    feature_df = df[df['feature'] == feature_name].copy()
    
    if feature_df.empty:
        print(f"⚠️ No data found for feature: {feature_name}")
        available_features = df['feature'].unique()
        print(f"   Available features: {available_features}")
        return None, []
    
    print(f"📈 Found {len(feature_df)} records for {feature_name}")
    print(f"   Weeks range: {feature_df['week'].min()} to {feature_df['week'].max()}")
    print(f"   Years: {feature_df['year'].unique()}")
    
    # Pivot to get wide format: rows=point_id, columns=week, values=value
    pivot_df = feature_df.pivot_table(
        index=['point_id', 'zone', 'year'], 
        columns='week', 
        values='value', 
        aggfunc='first'
    ).reset_index()
    
    # Get timestep columns (weeks)
    week_cols = [col for col in pivot_df.columns if isinstance(col, (int, float))]
    week_cols.sort()
    
    print(f"   Pivoted to shape: {pivot_df.shape}")
    print(f"   Timestep columns (weeks): {week_cols[:5]}...{week_cols[-5:]}")
    
    # Extract just the timestep data
    timestep_data = pivot_df[week_cols].values
    
    # Handle NaN values
    nan_count = np.isnan(timestep_data).sum()
    if nan_count > 0:
        print(f"   ⚠️ Found {nan_count} NaN values, filling with feature median")
        feature_median = np.nanmedian(timestep_data)
        timestep_data = np.nan_to_num(timestep_data, nan=feature_median)
    
    return timestep_data, week_cols, pivot_df

def analyze_feature_distribution(df, feature_name):
    """Comprehensive analysis of a biophysical feature."""
    print(f"\n🔍 ANALYZING {feature_name.upper()} DISTRIBUTION")
    print("-" * 60)
    
    # Extract timestep data
    result = extract_timestep_data(df, feature_name)
    if result is None or result[0] is None:
        return None
    
    data, timestep_cols, pivot_df = result
    
    n_samples, n_timesteps = data.shape
    
    # Overall statistics
    print(f"📊 Overall Statistics for {feature_name.upper()}:")
    print(f"   Samples: {n_samples:,}")
    print(f"   Timesteps: {n_timesteps}")
    print(f"   Global min: {data.min():.6f}")
    print(f"   Global max: {data.max():.6f}")
    print(f"   Global mean: {data.mean():.6f}")
    print(f"   Global std: {data.std():.6f}")
    print(f"   Global range: {data.max() - data.min():.6f}")
    
    # Per-timestep statistics
    timestep_stats = []
    for t in range(n_timesteps):
        timestep_data = data[:, t]
        stats = {
            'timestep': t,
            'min': timestep_data.min(),
            'max': timestep_data.max(),
            'mean': timestep_data.mean(),
            'std': timestep_data.std(),
            'q25': np.percentile(timestep_data, 25),
            'q50': np.percentile(timestep_data, 50),
            'q75': np.percentile(timestep_data, 75),
            'range': timestep_data.max() - timestep_data.min(),
            'n_zeros': (timestep_data == 0).sum(),
            'n_nans': np.isnan(timestep_data).sum()
        }
        timestep_stats.append(stats)
    
    timestep_df = pd.DataFrame(timestep_stats)
    
    print(f"\n📈 Per-Timestep Summary:")
    print(f"   Min range across timesteps: {timestep_df['range'].min():.6f}")
    print(f"   Max range across timesteps: {timestep_df['range'].max():.6f}")
    print(f"   Timestep with smallest range: {timestep_df.loc[timestep_df['range'].idxmin(), 'timestep']}")
    print(f"   Timestep with largest range: {timestep_df.loc[timestep_df['range'].idxmax(), 'timestep']}")
    
    return {
        'data': data,
        'timestep_cols': timestep_cols,
        'timestep_stats': timestep_df,
        'overall_stats': {
            'min': data.min(),
            'max': data.max(),
            'mean': data.mean(),
            'std': data.std(),
            'range': data.max() - data.min()
        }
    }

def analyze_by_zone(df, feature_name):
    """Analyze feature distribution by zone."""
    print(f"\n🌍 ZONE-WISE ANALYSIS for {feature_name.upper()}")
    print("-" * 60)
    
    result = extract_timestep_data(df, feature_name)
    if result is None or result[0] is None:
        return None
    
    data, timestep_cols, pivot_df = result
    zones = pivot_df['zone'].unique()
    zone_stats = []
    
    for zone in zones:
        zone_mask = pivot_df['zone'] == zone
        zone_data = data[zone_mask]
        
        stats = {
            'zone': zone,
            'n_samples': zone_data.shape[0],
            'min': zone_data.min(),
            'max': zone_data.max(),
            'mean': zone_data.mean(),
            'std': zone_data.std(),
            'range': zone_data.max() - zone_data.min(),
            'q25': np.percentile(zone_data, 25),
            'q75': np.percentile(zone_data, 75)
        }
        zone_stats.append(stats)
        
        print(f"   {zone}: {stats['n_samples']:,} samples, range=[{stats['min']:.4f}, {stats['max']:.4f}], mean={stats['mean']:.4f}")
    
    return pd.DataFrame(zone_stats)

def create_distribution_plots(results, feature_name, save_dir):
    """Create comprehensive distribution plots."""
    print(f"\n📊 Creating distribution plots for {feature_name.upper()}...")
    
    data = results['data']
    timestep_stats = results['timestep_stats']
    
    # Create subplot grid
    fig = plt.figure(figsize=(20, 15))
    
    # Plot 1: Boxplot across timesteps
    plt.subplot(3, 3, 1)
    timestep_data_list = [data[:, t] for t in range(data.shape[1])]
    plt.boxplot(timestep_data_list, showfliers=False)
    plt.title(f'{feature_name.upper()} Distribution Across Timesteps')
    plt.xlabel('Timestep')
    plt.ylabel(feature_name.upper())
    plt.xticks(range(1, min(21, data.shape[1]+1)), range(0, min(20, data.shape[1])))
    
    # Plot 2: Mean and std across timesteps
    plt.subplot(3, 3, 2)
    plt.plot(timestep_stats['timestep'], timestep_stats['mean'], 'o-', label='Mean', linewidth=2)
    plt.fill_between(timestep_stats['timestep'], 
                     timestep_stats['mean'] - timestep_stats['std'],
                     timestep_stats['mean'] + timestep_stats['std'], 
                     alpha=0.3, label='±1 STD')
    plt.title(f'{feature_name.upper()} Mean ± STD Over Time')
    plt.xlabel('Timestep')
    plt.ylabel(feature_name.upper())
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 3: Range across timesteps
    plt.subplot(3, 3, 3)
    plt.plot(timestep_stats['timestep'], timestep_stats['range'], 'o-', color='red', linewidth=2)
    plt.title(f'{feature_name.upper()} Range Over Time')
    plt.xlabel('Timestep')
    plt.ylabel('Range (Max - Min)')
    plt.grid(True, alpha=0.3)
    
    # Plot 4: Histogram of all values
    plt.subplot(3, 3, 4)
    plt.hist(data.flatten(), bins=100, alpha=0.7, edgecolor='black')
    plt.title(f'{feature_name.upper()} Overall Distribution')
    plt.xlabel(feature_name.upper())
    plt.ylabel('Frequency')
    plt.yscale('log')
    
    # Plot 5: Quartiles over time
    plt.subplot(3, 3, 5)
    plt.plot(timestep_stats['timestep'], timestep_stats['q25'], 'o-', label='Q25', linewidth=2)
    plt.plot(timestep_stats['timestep'], timestep_stats['q50'], 'o-', label='Q50 (Median)', linewidth=2)
    plt.plot(timestep_stats['timestep'], timestep_stats['q75'], 'o-', label='Q75', linewidth=2)
    plt.title(f'{feature_name.upper()} Quartiles Over Time')
    plt.xlabel('Timestep')
    plt.ylabel(feature_name.upper())
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 6: Coefficient of Variation
    plt.subplot(3, 3, 6)
    cv = timestep_stats['std'] / timestep_stats['mean']
    plt.plot(timestep_stats['timestep'], cv, 'o-', color='purple', linewidth=2)
    plt.title(f'{feature_name.upper()} Coefficient of Variation')
    plt.xlabel('Timestep')
    plt.ylabel('CV (STD/Mean)')
    plt.grid(True, alpha=0.3)
    
    # Plot 7: Min/Max bounds over time
    plt.subplot(3, 3, 7)
    plt.plot(timestep_stats['timestep'], timestep_stats['min'], 'o-', label='Min', linewidth=2)
    plt.plot(timestep_stats['timestep'], timestep_stats['max'], 'o-', label='Max', linewidth=2)
    plt.fill_between(timestep_stats['timestep'], timestep_stats['min'], timestep_stats['max'], 
                     alpha=0.2, label='Range')
    plt.title(f'{feature_name.upper()} Min/Max Bounds')
    plt.xlabel('Timestep')
    plt.ylabel(feature_name.upper())
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 8: Sample with extreme small range (if any)
    plt.subplot(3, 3, 8)
    min_range_timestep = timestep_stats.loc[timestep_stats['range'].idxmin(), 'timestep']
    extreme_data = data[:, min_range_timestep]
    plt.hist(extreme_data, bins=50, alpha=0.7, color='orange', edgecolor='black')
    plt.title(f'{feature_name.upper()} at Timestep {min_range_timestep} (Min Range)')
    plt.xlabel(feature_name.upper())
    plt.ylabel('Frequency')
    
    # Plot 9: Density plot for first few timesteps
    plt.subplot(3, 3, 9)
    for t in range(min(5, data.shape[1])):
        plt.hist(data[:, t], bins=50, alpha=0.4, label=f't={t}', density=True)
    plt.title(f'{feature_name.upper()} Density - First 5 Timesteps')
    plt.xlabel(feature_name.upper())
    plt.ylabel('Density')
    plt.legend()
    
    plt.tight_layout()
    
    # Save plot
    save_path = save_dir / f'{feature_name}_distribution_analysis.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"   ✅ Saved plot: {save_path}")
    plt.show()

def create_comparison_plots(all_results, save_dir):
    """Create comparison plots between LAI, FAPAR, FCOVER."""
    print(f"\n📊 Creating comparison plots...")
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    features = ['lai', 'fapar', 'fcover']
    colors = ['blue', 'green', 'red']
    
    # Plot mean over time for all features
    ax = axes[0, 0]
    for feat, color in zip(features, colors):
        if feat in all_results:
            stats = all_results[feat]['timestep_stats']
            ax.plot(stats['timestep'], stats['mean'], 'o-', 
                   label=feat.upper(), color=color, linewidth=2)
    ax.set_title('Mean Values Over Time')
    ax.set_xlabel('Timestep')
    ax.set_ylabel('Value')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot std over time
    ax = axes[0, 1]
    for feat, color in zip(features, colors):
        if feat in all_results:
            stats = all_results[feat]['timestep_stats']
            ax.plot(stats['timestep'], stats['std'], 'o-', 
                   label=feat.upper(), color=color, linewidth=2)
    ax.set_title('Standard Deviation Over Time')
    ax.set_xlabel('Timestep')
    ax.set_ylabel('STD')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot range over time
    ax = axes[0, 2]
    for feat, color in zip(features, colors):
        if feat in all_results:
            stats = all_results[feat]['timestep_stats']
            ax.plot(stats['timestep'], stats['range'], 'o-', 
                   label=feat.upper(), color=color, linewidth=2)
    ax.set_title('Range Over Time')
    ax.set_xlabel('Timestep')
    ax.set_ylabel('Range (Max-Min)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot coefficient of variation
    ax = axes[1, 0]
    for feat, color in zip(features, colors):
        if feat in all_results:
            stats = all_results[feat]['timestep_stats']
            cv = stats['std'] / stats['mean']
            ax.plot(stats['timestep'], cv, 'o-', 
                   label=feat.upper(), color=color, linewidth=2)
    ax.set_title('Coefficient of Variation Over Time')
    ax.set_xlabel('Timestep')
    ax.set_ylabel('CV (STD/Mean)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Histogram comparison
    ax = axes[1, 1]
    for feat, color in zip(features, colors):
        if feat in all_results:
            data = all_results[feat]['data'].flatten()
            ax.hist(data, bins=100, alpha=0.5, label=feat.upper(), 
                   color=color, density=True)
    ax.set_title('Overall Distribution Comparison')
    ax.set_xlabel('Value')
    ax.set_ylabel('Density')
    ax.legend()
    ax.set_yscale('log')
    
    # Summary statistics comparison
    ax = axes[1, 2]
    summary_data = []
    for feat in features:
        if feat in all_results:
            overall = all_results[feat]['overall_stats']
            summary_data.append([overall['min'], overall['mean'], overall['max']])
    
    x_pos = np.arange(len(features))
    bar_width = 0.25
    
    mins = [d[0] for d in summary_data]
    means = [d[1] for d in summary_data]
    maxs = [d[2] for d in summary_data]
    
    ax.bar(x_pos - bar_width, mins, bar_width, label='Min', alpha=0.8)
    ax.bar(x_pos, means, bar_width, label='Mean', alpha=0.8)
    ax.bar(x_pos + bar_width, maxs, bar_width, label='Max', alpha=0.8)
    
    ax.set_title('Min/Mean/Max Comparison')
    ax.set_xlabel('Features')
    ax.set_ylabel('Value')
    ax.set_xticks(x_pos)
    ax.set_xticklabels([f.upper() for f in features])
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save comparison plot
    save_path = save_dir / 'biophysical_features_comparison.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"   ✅ Saved comparison plot: {save_path}")
    plt.show()

def generate_summary_report(all_results, zone_results, save_dir):
    """Generate a comprehensive summary report."""
    print(f"\n📝 Generating summary report...")
    
    report_path = save_dir / 'biophysical_analysis_report.txt'
    
    with open(report_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("BIOPHYSICAL FEATURES DISTRIBUTION ANALYSIS REPORT\n")
        f.write("LAI, FAPAR, FCOVER Analysis\n")
        f.write("="*80 + "\n\n")
        
        # Overall summary
        f.write("1. OVERALL SUMMARY\n")
        f.write("-" * 40 + "\n")
        
        for feature in ['lai', 'fapar', 'fcover']:
            if feature in all_results:
                stats = all_results[feature]['overall_stats']
                f.write(f"\n{feature.upper()}:\n")
                f.write(f"  Range: [{stats['min']:.6f}, {stats['max']:.6f}]\n")
                f.write(f"  Mean: {stats['mean']:.6f}\n")
                f.write(f"  STD: {stats['std']:.6f}\n")
                f.write(f"  Total Range: {stats['range']:.6f}\n")
                
                # Identify problematic features (very small range)
                if stats['range'] < 0.01:
                    f.write(f"  ⚠️  WARNING: Very small range detected!\n")
        
        # Timestep analysis
        f.write(f"\n\n2. TIMESTEP ANALYSIS\n")
        f.write("-" * 40 + "\n")
        
        for feature in ['lai', 'fapar', 'fcover']:
            if feature in all_results:
                timestep_stats = all_results[feature]['timestep_stats']
                f.write(f"\n{feature.upper()} Timestep Analysis:\n")
                f.write(f"  Min range across timesteps: {timestep_stats['range'].min():.6f}\n")
                f.write(f"  Max range across timesteps: {timestep_stats['range'].max():.6f}\n")
                f.write(f"  Most stable timestep: {timestep_stats.loc[timestep_stats['range'].idxmin(), 'timestep']}\n")
                f.write(f"  Most variable timestep: {timestep_stats.loc[timestep_stats['range'].idxmax(), 'timestep']}\n")
                
                # Identify flat timesteps
                flat_timesteps = timestep_stats[timestep_stats['range'] < 0.001]['timestep'].tolist()
                if flat_timesteps:
                    f.write(f"  🔴 Flat timesteps (range < 0.001): {flat_timesteps}\n")
        
        # Zone analysis
        f.write(f"\n\n3. ZONE ANALYSIS\n")
        f.write("-" * 40 + "\n")
        
        for feature in ['lai', 'fapar', 'fcover']:
            if feature in zone_results:
                zone_stats = zone_results[feature]
                f.write(f"\n{feature.upper()} by Zone:\n")
                for _, row in zone_stats.iterrows():
                    f.write(f"  {row['zone']}: range=[{row['min']:.4f}, {row['max']:.4f}], "
                           f"mean={row['mean']:.4f}, samples={row['n_samples']:,}\n")
        
        # Recommendations
        f.write(f"\n\n4. RECOMMENDATIONS\n")
        f.write("-" * 40 + "\n")
        
        for feature in ['lai', 'fapar', 'fcover']:
            if feature in all_results:
                stats = all_results[feature]['overall_stats']
                if stats['range'] < 0.01:
                    f.write(f"\n{feature.upper()}:\n")
                    f.write(f"  🔴 ISSUE: Extremely small range ({stats['range']:.6f})\n")
                    f.write(f"  💡 SOLUTIONS:\n")
                    f.write(f"     - Check data source and preprocessing\n")
                    f.write(f"     - Consider removing this feature if non-informative\n")
                    f.write(f"     - Use min-max scaling instead of z-score\n")
                    f.write(f"     - Investigate if data is already normalized\n")
                elif stats['range'] < 0.1:
                    f.write(f"\n{feature.upper()}:\n")
                    f.write(f"  ⚠️  WARNING: Small range ({stats['range']:.6f})\n")
                    f.write(f"  💡 RECOMMENDATION: Use min-max scaling for better model performance\n")
        
        f.write(f"\n\n5. SCALING RECOMMENDATIONS\n")
        f.write("-" * 40 + "\n")
        f.write("Based on the analysis:\n")
        f.write("- Features with small ranges benefit from min-max scaling\n")
        f.write("- Features with large differences in scale should be normalized\n")
        f.write("- Consider removing features with extremely small variance\n")
        f.write("- Apply outlier filtering before scaling\n")
    
    print(f"   ✅ Saved report: {report_path}")

def main():
    """Main analysis function."""
    # Configuration
    data_file = "/fs2/ftype_mflf/shared_data/dataset/1.2.0/raw/forest_type_dataset_raw_20250903_145707.cleaned.parquet"
    output_dir = Path("/fs1/tor_project/model_foresttype/analysis_output")
    output_dir.mkdir(exist_ok=True)
    
    features_to_analyze = ['lai', 'fapar', 'fcover']
    
    print("🚀 Starting Biophysical Features Distribution Analysis")
    print(f"📁 Data file: {data_file}")
    print(f"💾 Output directory: {output_dir}")
    
    # Load data
    df = load_parquet_data(data_file)
    
    # Analyze each feature
    all_results = {}
    zone_results = {}
    
    for feature in features_to_analyze:
        print(f"\n" + "="*80)
        print(f"🔬 ANALYZING {feature.upper()}")
        print("="*80)
        
        # Feature distribution analysis
        results = analyze_feature_distribution(df, feature)
        if results:
            all_results[feature] = results
            
            # Zone analysis
            zone_stats = analyze_by_zone(df, feature)
            if zone_stats is not None:
                zone_results[feature] = zone_stats
            
            # Create plots
            create_distribution_plots(results, feature, output_dir)
    
    # Create comparison plots
    if all_results:
        create_comparison_plots(all_results, output_dir)
        
        # Generate summary report
        generate_summary_report(all_results, zone_results, output_dir)
    
    print(f"\n" + "="*80)
    print("✅ ANALYSIS COMPLETE!")
    print("="*80)
    print(f"📊 Results saved to: {output_dir}")
    print("Generated files:")
    for file in output_dir.glob("*"):
        print(f"  - {file.name}")

if __name__ == "__main__":
    main()