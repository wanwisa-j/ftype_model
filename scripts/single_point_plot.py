#!/usr/bin/env python3
"""
Single Point Time Series Plot
Randomly selects one data point and plots its complete time series
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import random

def plot_single_point_timeseries():
    """Plot time series for a single random point"""
    
    # Load the test data
    data_dir = Path('/fs2/ftype_mflf/shared_data/dataset/1.2.0/test/raw')
    parquet_files = list(data_dir.glob('*.parquet'))
    latest_file = max(parquet_files, key=lambda x: x.stat().st_mtime)
    
    print(f"Loading data from: {latest_file.name}")
    df = pd.read_parquet(latest_file)
    
    # Filter for LAI data only
    lai_data = df[df['feature'] == 'lai'].copy()
    
    if lai_data.empty:
        print("No LAI data found!")
        return
    
    # Get all unique points
    unique_points = lai_data['point_id'].unique()
    print(f"Total points available: {len(unique_points)}")
    
    # Randomly select one point
    random_point = random.choice(unique_points)
    point_data = lai_data[lai_data['point_id'] == random_point].copy()
    
    # Sort by week
    point_data = point_data.sort_values('week')
    
    print(f"\n🎯 Selected Point: {random_point}")
    print(f"Forest Type (Mix): {point_data['ftype_mix'].iloc[0]}")
    print(f"Forest Type (Non-Mix): {point_data['ftype_nonmix'].iloc[0]}")
    print(f"Zone: {point_data['zone'].iloc[0]}")
    print(f"UTM Coordinates: ({point_data['utm_x'].iloc[0]:.1f}, {point_data['utm_y'].iloc[0]:.1f})")
    print(f"Number of timesteps: {len(point_data)}")
    
    # Create the plot
    plt.figure(figsize=(15, 8))
    
    # Main time series plot
    plt.subplot(2, 2, 1)
    plt.plot(point_data['week'], point_data['value'], 
             marker='o', linewidth=2, markersize=6, color='green')
    plt.title(f'LAI Time Series - Point {random_point}\n({point_data["ftype_mix"].iloc[0]} Forest)', 
              fontweight='bold')
    plt.xlabel('Week')
    plt.ylabel('LAI Value')
    plt.grid(True, alpha=0.3)
    plt.xticks(range(1, 22))
    
    # Add data table
    plt.subplot(2, 2, 2)
    plt.axis('off')
    table_data = []
    for _, row in point_data.iterrows():
        table_data.append([f"Week {int(row['week'])}", f"{row['value']:.3f}"])
    
    # Split into two columns for better display
    mid = len(table_data) // 2
    col1 = table_data[:mid]
    col2 = table_data[mid:]
    
    # Pad shorter column
    while len(col1) < len(col2):
        col1.append(['', ''])
    while len(col2) < len(col1):
        col2.append(['', ''])
    
    combined_data = []
    for i in range(len(col1)):
        combined_data.append([col1[i][0], col1[i][1], col2[i][0], col2[i][1]])
    
    table = plt.table(cellText=combined_data,
                     colLabels=['Week', 'LAI', 'Week', 'LAI'],
                     cellLoc='center',
                     loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.5)
    plt.title('LAI Values by Week', fontweight='bold', pad=20)
    
    # Statistics
    plt.subplot(2, 2, 3)
    plt.axis('off')
    stats_text = f"""
Point Statistics:
━━━━━━━━━━━━━━━━━━
Point ID: {random_point}
Forest Type: {point_data['ftype_mix'].iloc[0]}
Zone: {point_data['zone'].iloc[0]}

LAI Statistics:
━━━━━━━━━━━━━━━━━━
Mean LAI: {point_data['value'].mean():.3f}
Std Dev: {point_data['value'].std():.3f}
Min LAI: {point_data['value'].min():.3f}
Max LAI: {point_data['value'].max():.3f}
Range: {point_data['value'].max() - point_data['value'].min():.3f}

Temporal Coverage:
━━━━━━━━━━━━━━━━━━
Total weeks: {len(point_data)}
Week range: {int(point_data['week'].min())} - {int(point_data['week'].max())}
Year: {point_data['year'].iloc[0]:.0f}
    """
    plt.text(0.1, 0.9, stats_text, transform=plt.gca().transAxes, 
             fontfamily='monospace', fontsize=10, verticalalignment='top')
    
    # Bar chart of LAI values
    plt.subplot(2, 2, 4)
    colors = plt.cm.Greens(np.linspace(0.4, 0.9, len(point_data)))
    bars = plt.bar(point_data['week'], point_data['value'], color=colors, alpha=0.8)
    plt.title('LAI Values by Week (Bar Chart)', fontweight='bold')
    plt.xlabel('Week')
    plt.ylabel('LAI Value')
    plt.xticks(range(1, 22, 2))  # Show every other week to avoid crowding
    plt.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar, value in zip(bars, point_data['value']):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{value:.2f}', ha='center', va='bottom', fontsize=8, rotation=45)
    
    plt.tight_layout()
    
    # Save the plot
    output_dir = Path('/fs1/tor_project/model_foresttype/plots')
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / f'single_point_timeseries_{random_point}.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n📊 Plot saved to: {output_path}")
    
    plt.show()
    
    # Print the time series data
    print(f"\n📈 Complete Time Series Data:")
    print("=" * 40)
    print(f"{'Week':>6} {'LAI':>8} {'File':>15}")
    print("-" * 40)
    for _, row in point_data.iterrows():
        filename = Path(row['file_path']).name if pd.notna(row['file_path']) else 'N/A'
        print(f"{int(row['week']):6d} {row['value']:8.3f} {filename:>15}")
    
    print(f"\n✅ Single point analysis complete!")
    print(f"Selected point {random_point} has {len(point_data)} timesteps as expected.")

if __name__ == "__main__":
    plot_single_point_timeseries()
