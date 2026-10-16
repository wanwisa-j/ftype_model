#!/usr/bin/env python3
"""
Plot Test Dataset - LAI Time Series Visualization
Visualizes the extracted forest type dataset for analysis and validation
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse
import json

def load_test_data(data_path):
    """Load the test dataset from parquet file"""
    print(f"Loading data from: {data_path}")
    df = pd.read_parquet(data_path)
    print(f"Dataset shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    return df

def plot_lai_time_series_by_forest_type(df, output_dir=None, max_samples=100):
    """Plot LAI time series grouped by forest type"""
    
    # Filter for LAI data only
    lai_data = df[df['feature'] == 'lai'].copy()
    print(f"LAI data shape: {lai_data.shape}")
    
    if lai_data.empty:
        print("No LAI data found!")
        return
    
    # Sample data if too many points
    if len(lai_data['point_id'].unique()) > max_samples:
        sample_points = np.random.choice(
            lai_data['point_id'].unique(), 
            size=max_samples, 
            replace=False
        )
        lai_data = lai_data[lai_data['point_id'].isin(sample_points)]
        print(f"Sampled {max_samples} points for visualization")
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('LAI Time Series by Forest Type - Test Dataset', fontsize=16, fontweight='bold')
    
    # Plot 1: Individual time series by forest type (ftype_mix)
    ax1 = axes[0, 0]
    for ftype in lai_data['ftype_mix'].unique():
        if pd.notna(ftype):
            ftype_data = lai_data[lai_data['ftype_mix'] == ftype]
            for point_id in ftype_data['point_id'].unique()[:10]:  # Show first 10 points per type
                point_data = ftype_data[ftype_data['point_id'] == point_id]
                ax1.plot(point_data['year'], point_data['value'], alpha=0.6, linewidth=1)
    
    ax1.set_title('Individual LAI Time Series by Forest Mix Type')
    ax1.set_xlabel('Year')
    ax1.set_ylabel('LAI Value')
    ax1.grid(True, alpha=0.3)
    ax1.legend(lai_data['ftype_mix'].unique(), title='Forest Mix Type')
    
    # Plot 2: Mean LAI by forest type over time
    ax2 = axes[0, 1]
    lai_mean = lai_data.groupby(['ftype_mix', 'year'])['value'].agg(['mean', 'std']).reset_index()
    
    for ftype in lai_mean['ftype_mix'].unique():
        if pd.notna(ftype):
            ftype_data = lai_mean[lai_mean['ftype_mix'] == ftype]
            ax2.errorbar(ftype_data['year'], ftype_data['mean'], 
                        yerr=ftype_data['std'], label=ftype, 
                        marker='o', linewidth=2, capsize=5)
    
    ax2.set_title('Mean LAI by Forest Mix Type')
    ax2.set_xlabel('Year')
    ax2.set_ylabel('Mean LAI Value')
    ax2.grid(True, alpha=0.3)
    ax2.legend(title='Forest Mix Type')
    
    # Plot 3: LAI distribution by forest type (ftype_nonmix)
    ax3 = axes[1, 0]
    lai_data.boxplot(column='value', by='ftype_nonmix', ax=ax3)
    ax3.set_title('LAI Distribution by Forest Non-Mix Type')
    ax3.set_xlabel('Forest Non-Mix Type')
    ax3.set_ylabel('LAI Value')
    plt.suptitle('')  # Remove automatic title
    
    # Plot 4: LAI histogram by forest type
    ax4 = axes[1, 1]
    for ftype in lai_data['ftype_mix'].unique():
        if pd.notna(ftype):
            ftype_data = lai_data[lai_data['ftype_mix'] == ftype]['value']
            ax4.hist(ftype_data, alpha=0.6, bins=30, label=ftype, density=True)
    
    ax4.set_title('LAI Value Distribution by Forest Mix Type')
    ax4.set_xlabel('LAI Value')
    ax4.set_ylabel('Density')
    ax4.legend(title='Forest Mix Type')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot if output directory specified
    if output_dir:
        output_path = Path(output_dir) / 'lai_time_series_analysis.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {output_path}")
    
    plt.show()

def plot_feature_correlation(df, output_dir=None):
    """Plot correlation between features"""
    
    # Pivot to wide format for correlation analysis
    pivot_data = df.pivot_table(
        index='point_id', 
        columns='feature', 
        values='value', 
        aggfunc='mean'
    )
    
    # Calculate correlation matrix
    corr_matrix = pivot_data.corr()
    
    # Create correlation heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, 
                square=True, linewidths=0.5)
    plt.title('Feature Correlation Matrix - Test Dataset', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    # Save plot if output directory specified
    if output_dir:
        output_path = Path(output_dir) / 'feature_correlation.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Correlation plot saved to: {output_path}")
    
    plt.show()

def print_dataset_summary(df):
    """Print summary statistics of the dataset"""
    
    print("\n" + "="*60)
    print("DATASET SUMMARY")
    print("="*60)
    
    print(f"Total records: {len(df):,}")
    print(f"Unique points: {df['point_id'].nunique():,}")
    print(f"Features: {', '.join(df['feature'].unique())}")
    print(f"Years: {sorted(df['year'].unique())}")
    print(f"Zones: {', '.join(df['zone'].unique())}")
    
    print("\nForest Type Distribution:")
    print("Mix Types:")
    print(df['ftype_mix'].value_counts())
    print("\nNon-Mix Types:")
    print(df['ftype_nonmix'].value_counts())
    
    print("\nFeature Statistics:")
    feature_stats = df.groupby('feature')['value'].agg(['count', 'mean', 'std', 'min', 'max'])
    print(feature_stats.round(3))

def main():
    parser = argparse.ArgumentParser(description='Plot test dataset for forest type classification')
    parser.add_argument('--data-path', type=str, 
                       default='/fs2/ftype_mflf/shared_data/dataset/1.2.0/test/raw',
                       help='Path to the test dataset directory or parquet file')
    parser.add_argument('--output-dir', type=str, 
                       default='/fs1/tor_project/model_foresttype/plots',
                       help='Output directory for plots')
    parser.add_argument('--max-samples', type=int, default=100,
                       help='Maximum number of sample points to plot for time series')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find the most recent parquet file if directory is provided
    data_path = Path(args.data_path)
    if data_path.is_dir():
        parquet_files = list(data_path.glob('*.parquet'))
        if not parquet_files:
            print(f"No parquet files found in {data_path}")
            return
        # Get the most recent file
        data_file = max(parquet_files, key=lambda x: x.stat().st_mtime)
        print(f"Using most recent file: {data_file}")
    else:
        data_file = data_path
    
    # Load and analyze data
    df = load_test_data(data_file)
    
    # Print summary
    print_dataset_summary(df)
    
    # Create plots
    print("\nGenerating LAI time series plots...")
    plot_lai_time_series_by_forest_type(df, args.output_dir, args.max_samples)
    
    print("\nGenerating feature correlation plot...")
    plot_feature_correlation(df, args.output_dir)
    
    print(f"\nAll plots saved to: {args.output_dir}")

if __name__ == "__main__":
    main()
