#!/usr/bin/env python3
"""
Weekly Biophysical Data Aggregation Pipeline
============================================

This script processes raw daily biophysical data and creates weekly aggregated datacubes 
with 21 bands for any zone. It replaces the notebook-based approach with a controlled
pipeline for consistent processing.

Features:
- Processes any zone with prepared biophysical raster data
- Creates 21-week datacubes (4 weeks before + 17 weeks after target year)
- Handles missing data with proper interpolation
- ISO week-based aggregation (Monday start)
- Robust NaN handling and data quality checks

Input Structure:
    {input_dir}/
    ├── fapar/
    │   ├── s2_fapar_YYYY_MM_DD.tif
    │   └── ...
    ├── fcover/
    │   ├── s2_fcover_YYYY_MM_DD.tif  
    │   └── ...
    └── lai/
        ├── s2_lai_YYYY_MM_DD.tif
        └── ...

Output Structure:
    {output_dir}/
    ├── fapar/
    │   └── cube_biophys_fapar_{year}_W4before_to_W16after_21bands.tif
    ├── fcover/
    │   └── cube_biophys_fcover_{year}_W4before_to_W16after_21bands.tif
    └── lai/
        └── cube_biophys_lai_{year}_W4before_to_W16after_21bands.tif

Usage:
    conda activate gisworkspace-cupy
    python weekly_biophys_aggregation.py --input-dir /path/to/daily/data --output-dir /path/to/datacube --year 2022
    python weekly_biophys_aggregation.py --input-dir /fs2/ftype_mflf/shared_data/temp/biophys/2025/1.0.0/mfl_cmf_area_project_2025_chr_zone_1 --output-dir /fs2/ftype_mflf/shared_data/datacube/2025/1.2.0/mfl_cmf_area_project_2025_chr_zone_1 --year 2022
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import rasterio as rio
import xarray as xr
import rioxarray as rxr
from datetime import datetime
from pathlib import Path
import logging
from typing import Tuple, List, Dict, Optional
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class WeeklyBiophysicalAggregator:
    """Weekly aggregation pipeline for biophysical data"""
    
    def __init__(self, input_dir: str, output_dir: str):
        """
        Initialize the aggregator
        
        Args:
            input_dir: Path to input directory with daily biophysical data
            output_dir: Path to output datacube directory
        """
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        
        # Validate input directory structure
        self._validate_input_structure()
        
        # Ensure output directories exist
        for var in ['lai', 'fapar', 'fcover']:
            (self.output_dir / var).mkdir(parents=True, exist_ok=True)
    
    def _validate_input_structure(self):
        """Validate that input directory has expected structure"""
        if not self.input_dir.exists():
            raise FileNotFoundError(f"Input directory not found: {self.input_dir}")
        
        required_vars = ['fapar', 'lai', 'fcover']
        missing_vars = []
        
        for var in required_vars:
            var_dir = self.input_dir / var
            if not var_dir.exists():
                missing_vars.append(var)
            else:
                # Check for at least one file
                files = list(var_dir.glob(f's2_{var}_*.tif'))
                if not files:
                    missing_vars.append(f"{var} (no files)")
        
        if missing_vars:
            raise ValueError(f"Missing required subdirectories or files: {missing_vars}")
        
        logger.info(f"Input directory structure validated: {self.input_dir}")
    
    def create_biophys_df(self) -> pd.DataFrame:
        """
        Create DataFrame mapping dates to biophysical file paths
        
        Returns:
            DataFrame with columns: datetime, fapar, lai, fcover
        """
        logger.info(f"Scanning biophysical files in {self.input_dir}")
        
        # Collect files for each variable
        vars_data = {}
        for var in ['fapar', 'lai', 'fcover']:
            var_dir = self.input_dir / var
            files = sorted(var_dir.glob(f's2_{var}_*.tif'))
            dates = []
            filenames = []
            
            for file_path in files:
                # Extract date from s2_{var}_YYYY_MM_DD.tif pattern
                filename = file_path.name
                if filename.startswith(f's2_{var}_') and filename.endswith('.tif'):
                    date_part = filename[len(f's2_{var}_'):-4]  # Remove prefix and suffix
                    try:
                        date_obj = datetime.strptime(date_part, '%Y_%m_%d')
                        dates.append(date_obj)
                        filenames.append(str(file_path))
                    except ValueError as e:
                        logger.warning(f"Could not parse date from filename {filename}: {e}")
                        continue
            
            vars_data[var] = dict(zip(dates, filenames))
            logger.info(f"Found {len(vars_data[var])} {var} files")
        
        # Find common dates across all variables
        common_dates = set(vars_data['fapar'].keys()) & set(vars_data['lai'].keys()) & set(vars_data['fcover'].keys())
        logger.info(f"Found {len(common_dates)} dates with all three variables")
        
        if not common_dates:
            raise ValueError("No common dates found across all biophysical variables")
        
        # Create DataFrame
        records = []
        for date in sorted(common_dates):
            records.append({
                'datetime': date,
                'fapar': vars_data['fapar'][date],
                'lai': vars_data['lai'][date],
                'fcover': vars_data['fcover'][date]
            })
        
        df = pd.DataFrame(records)
        logger.info(f"Created DataFrame with {len(df)} records")
        logger.info(f"Date range: {df['datetime'].min().date()} to {df['datetime'].max().date()}")
        return df
    
    def filter_date_range(self, df: pd.DataFrame, year: int) -> pd.DataFrame:
        """
        Filter DataFrame to date range for given year
        
        Args:
            df: Input DataFrame with datetime column
            year: Target year
            
        Returns:
            Filtered DataFrame
        """
        start_date = pd.Timestamp(f'{year-1}-11-15')  # November 15th previous year
        end_date = pd.Timestamp(f'{year}-04-30')      # April 30th target year
        
        filtered_df = df[
            (df['datetime'] >= start_date) & 
            (df['datetime'] <= end_date)
        ].sort_values('datetime').reset_index(drop=True)
        
        logger.info(f"Filtered to {len(filtered_df)} records for year {year}")
        logger.info(f"Date range: {start_date.date()} to {end_date.date()}")
        
        if len(filtered_df) == 0:
            logger.warning(f"No data found for year {year} in range {start_date.date()} to {end_date.date()}")
        
        return filtered_df
    
    def load_daily_data(self, df: pd.DataFrame, variable: str) -> Tuple[np.ndarray, pd.DatetimeIndex, dict]:
        """
        Load daily raster data into memory
        
        Args:
            df: DataFrame with file paths
            variable: Variable name ('lai', 'fapar', 'fcover')
            
        Returns:
            data_stack: 3D array (time, height, width)
            time_index: DatetimeIndex for time axis
            metadata: Raster metadata from first file
        """
        logger.info(f"Loading daily {variable} data ({len(df)} files)")
        
        data_arrays = []
        time_coords = []
        metadata = None
        
        for _, row in df.iterrows():
            try:
                with rio.open(row[variable]) as src:
                    data = src.read(1)
                    if metadata is None:
                        metadata = {
                            'transform': src.transform,
                            'crs': src.crs,
                            'height': src.height,
                            'width': src.width,
                            'dtype': data.dtype,
                            'nodata': src.nodata
                        }
                        logger.info(f"Data shape per file: {data.shape}, dtype: {data.dtype}")
                        
                    data_arrays.append(data)
                    time_coords.append(row['datetime'])
                    
            except Exception as e:
                logger.error(f"Error loading {row[variable]}: {e}")
                raise
        
        data_stack = np.stack(data_arrays, axis=0)  # (time, height, width)
        time_index = pd.DatetimeIndex(time_coords)
        
        logger.info(f"Loaded {variable} stack shape: {data_stack.shape}")
        
        # Data quality check
        finite_pct = np.isfinite(data_stack).mean() * 100
        logger.info(f"Data quality: {finite_pct:.1f}% finite values")
        
        return data_stack, time_index, metadata
    
    def compute_weekly_aggregation(self, daily_data: np.ndarray, time_index: pd.DatetimeIndex) -> Tuple[np.ndarray, pd.DatetimeIndex]:
        """
        Aggregate daily data to weekly means using ISO weeks (Monday start)
        
        Args:
            daily_data: 3D array (time, height, width)
            time_index: DatetimeIndex for time axis
            
        Returns:
            weekly_data: 3D array (weeks, height, width)
            weekly_times: DatetimeIndex for week start dates
        """
        logger.info("Computing weekly aggregation using ISO weeks (Monday start)")
        
        # Group by ISO week (Monday start)
        week_starts = time_index.to_series().dt.to_period('W-MON').dt.start_time
        unique_weeks = week_starts.unique()
        
        logger.info(f"Found {len(unique_weeks)} unique weeks")
        
        weekly_data_list = []
        weekly_times = []
        
        for week_start in sorted(unique_weeks):
            # Find all daily observations in this week
            week_mask = (week_starts == week_start).values
            week_indices = np.where(week_mask)[0]
            
            if len(week_indices) == 0:
                continue
            
            # Compute weekly mean, handling NaN values properly
            week_data = daily_data[week_indices]  # (days_in_week, height, width)
            
            # Use nanmean to ignore NaN values in computation
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                week_mean = np.nanmean(week_data, axis=0)  # (height, width)
            
            weekly_data_list.append(week_mean)
            weekly_times.append(week_start)
            
            logger.debug(f"Week {week_start.date()}: {len(week_indices)} daily observations")
        
        weekly_data = np.stack(weekly_data_list, axis=0)  # (weeks, height, width)
        weekly_times = pd.DatetimeIndex(weekly_times)
        
        logger.info(f"Created {len(weekly_times)} weekly aggregates")
        logger.info(f"Weekly data shape: {weekly_data.shape}")
        
        return weekly_data, weekly_times
    
    def select_target_weeks(self, weekly_data: np.ndarray, weekly_times: pd.DatetimeIndex, year: int) -> Tuple[np.ndarray, pd.DatetimeIndex]:
        """
        Select 21 target weeks: last 4 weeks of previous year + first 17 weeks of target year
        
        Args:
            weekly_data: Weekly aggregated data
            weekly_times: Week start dates
            year: Target year
            
        Returns:
            selected_data: Data for selected weeks (shape: 21, height, width)
            selected_times: Times for selected weeks (length: 21)
        """
        logger.info(f"Selecting 21 target weeks for year {year}")
        
        year_start = pd.Timestamp(f'{year}-01-01')
        
        # Separate weeks before and after year start
        before_mask = weekly_times < year_start
        after_mask = weekly_times >= year_start
        
        weeks_before = weekly_times[before_mask]
        weeks_after = weekly_times[after_mask]
        
        logger.info(f"Available weeks before {year}: {len(weeks_before)}")
        logger.info(f"Available weeks after {year}: {len(weeks_after)}")
        
        # Select last 4 weeks before year start
        if len(weeks_before) >= 4:
            selected_before_indices = np.where(before_mask)[0][-4:]
            selected_before_times = weeks_before[-4:]
        else:
            selected_before_indices = np.where(before_mask)[0]
            selected_before_times = weeks_before
            logger.warning(f"Only {len(weeks_before)} weeks available before {year}, need 4")
        
        # Select first 17 weeks after year start
        # Filter to only weeks in target year (avoid next year spillover)
        year_end = pd.Timestamp(f'{year}-12-31')
        after_in_year_mask = (weekly_times >= year_start) & (weekly_times <= year_end)
        after_in_year = weekly_times[after_in_year_mask]
        
        if len(after_in_year) >= 17:
            selected_after_indices = np.where(after_in_year_mask)[0][:17]
            selected_after_times = after_in_year[:17]
        else:
            selected_after_indices = np.where(after_in_year_mask)[0]
            selected_after_times = after_in_year
            logger.warning(f"Only {len(after_in_year)} weeks available in {year}, need 17")
        
        # Combine selections
        selected_indices = np.concatenate([selected_before_indices, selected_after_indices])
        selected_times = pd.DatetimeIndex(list(selected_before_times) + list(selected_after_times))
        
        selected_data = weekly_data[selected_indices]
        
        logger.info(f"Selected {len(selected_indices)} weeks total")
        if len(selected_times) > 0:
            logger.info(f"Week range: {selected_times.min().date()} to {selected_times.max().date()}")
        
        # Handle missing weeks with interpolation if needed
        if len(selected_indices) < 21:
            logger.warning(f"Only {len(selected_indices)} weeks selected, target is 21")
            selected_data, selected_times = self._interpolate_missing_weeks(
                selected_data, selected_times, year, target_weeks=21
            )
        elif len(selected_indices) > 21:
            logger.warning(f"Too many weeks selected ({len(selected_indices)}), truncating to 21")
            selected_data = selected_data[:21]
            selected_times = selected_times[:21]
        
        return selected_data, selected_times
    
    def _interpolate_missing_weeks(self, data: np.ndarray, times: pd.DatetimeIndex, year: int, target_weeks: int = 21) -> Tuple[np.ndarray, pd.DatetimeIndex]:
        """
        Interpolate missing weeks to reach target number of weeks
        
        Args:
            data: Weekly data array
            times: Week times
            year: Target year
            target_weeks: Target number of weeks
            
        Returns:
            interpolated_data: Data with interpolated weeks
            interpolated_times: Times including interpolated weeks
        """
        logger.info(f"Interpolating missing weeks to reach {target_weeks} total")
        
        # Create ideal 21-week sequence
        # Week 1 is the first week of the year containing January 4th
        jan4 = pd.Timestamp(f'{year}-01-04')
        week1_monday = jan4 - pd.Timedelta(days=jan4.weekday())
        ideal_weeks = [week1_monday + pd.Timedelta(weeks=w) for w in range(-4, 17)]
        ideal_times = pd.DatetimeIndex(ideal_weeks)
        
        logger.info(f"Ideal week range: {ideal_times.min().date()} to {ideal_times.max().date()}")
        
        # Find which ideal weeks we have data for
        existing_set = set(times)
        missing_weeks = [w for w in ideal_times if w not in existing_set]
        
        if not missing_weeks:
            logger.info("No missing weeks, using existing data")
            return data[:target_weeks], times[:target_weeks]
        
        logger.info(f"Interpolating {len(missing_weeks)} missing weeks")
        
        # Simple nearest neighbor interpolation for missing weeks
        interpolated_data = []
        interpolated_times = []
        
        for target_time in ideal_times:
            if target_time in existing_set:
                # Use existing data
                idx = list(times).index(target_time)
                interpolated_data.append(data[idx])
            else:
                # Find nearest existing time
                if len(times) > 0:
                    time_diffs = np.abs((times - target_time).total_seconds())
                    nearest_idx = np.argmin(time_diffs)
                    interpolated_data.append(data[nearest_idx])
                    logger.debug(f"Interpolated {target_time.date()} using {times[nearest_idx].date()}")
                else:
                    # No existing data - create zeros
                    if len(interpolated_data) > 0:
                        interpolated_data.append(np.zeros_like(interpolated_data[0]))
                    else:
                        logger.error("No existing data to interpolate from")
                        raise ValueError("Cannot interpolate - no existing weekly data")
            
            interpolated_times.append(target_time)
        
        final_data = np.stack(interpolated_data, axis=0)
        final_times = pd.DatetimeIndex(interpolated_times)
        
        logger.info(f"Final interpolated data shape: {final_data.shape}")
        return final_data, final_times
    
    def handle_nodata_values(self, data: np.ndarray) -> np.ndarray:
        """
        Handle NaN values with pixel-wise linear interpolation
        
        Args:
            data: 3D array (time, height, width)
            
        Returns:
            cleaned_data: Data with interpolated NaN values
        """
        logger.info("Handling NaN values with linear interpolation")
        
        data_copy = data.copy().astype(np.float64)  # Use float64 for interpolation precision
        n_time, n_height, n_width = data_copy.shape
        
        # Count NaN pixels before processing
        nan_mask = np.isnan(data_copy)
        nan_pixels = nan_mask.any(axis=0).sum()
        total_pixels = n_height * n_width
        nan_values_total = nan_mask.sum()
        
        logger.info(f"Before interpolation:")
        logger.info(f"  Pixels with NaN values: {nan_pixels}/{total_pixels} ({100*nan_pixels/total_pixels:.1f}%)")
        logger.info(f"  Total NaN values: {nan_values_total}/{data_copy.size} ({100*nan_values_total/data_copy.size:.2f}%)")
        
        if nan_pixels == 0:
            logger.info("No NaN values found, returning original data")
            return data_copy.astype(np.float32)
        
        # Interpolate NaN values pixel by pixel
        time_axis = np.arange(n_time)
        interpolated_pixels = 0
        
        for h in range(n_height):
            for w in range(n_width):
                pixel_series = data_copy[:, h, w]
                
                if np.any(np.isnan(pixel_series)):
                    valid_mask = ~np.isnan(pixel_series)
                    
                    if np.any(valid_mask):
                        # Has some valid values - interpolate
                        valid_indices = time_axis[valid_mask]
                        valid_values = pixel_series[valid_mask]
                        
                        # Linear interpolation with edge extension
                        interpolated = np.interp(time_axis, valid_indices, valid_values)
                        data_copy[:, h, w] = interpolated
                        interpolated_pixels += 1
                    else:
                        # No valid values - leave as NaN (will become nodata in output)
                        pass
        
        # Count NaN pixels after interpolation
        remaining_nan_mask = np.isnan(data_copy)
        remaining_nan_pixels = remaining_nan_mask.any(axis=0).sum()
        remaining_nan_values = remaining_nan_mask.sum()
        
        logger.info(f"After interpolation:")
        logger.info(f"  Interpolated pixels: {interpolated_pixels}")
        logger.info(f"  Remaining NaN pixels: {remaining_nan_pixels}/{total_pixels} ({100*remaining_nan_pixels/total_pixels:.1f}%)")
        logger.info(f"  Remaining NaN values: {remaining_nan_values}/{data_copy.size} ({100*remaining_nan_values/data_copy.size:.2f}%)")
        
        return data_copy.astype(np.float32)
    
    def save_weekly_datacube(self, data: np.ndarray, variable: str, year: int, metadata: dict, times: pd.DatetimeIndex):
        """
        Save weekly datacube as GeoTIFF
        
        Args:
            data: 3D array (time, height, width)
            variable: Variable name
            year: Target year
            metadata: Raster metadata
            times: Week start times
        """
        output_path = self.output_dir / variable / f'cube_biophys_{variable}_{year}_W4before_to_W16after_21bands.tif'
        
        logger.info(f"Saving {variable} datacube: {output_path}")
        logger.info(f"Data shape: {data.shape}, dtype: {data.dtype}")
        
        # Update metadata
        profile = {
            'driver': 'GTiff',
            'height': data.shape[1],
            'width': data.shape[2], 
            'count': data.shape[0],
            'dtype': 'float32',
            'crs': metadata['crs'],
            'transform': metadata['transform'],
            'compress': 'LZW',
            'nodata': np.nan  # Use NaN as nodata value
        }
        
        # Convert to float32 to save space
        data_f32 = data.astype(np.float32)
        
        # Write datacube
        with rio.open(output_path, 'w', **profile) as dst:
            dst.write(data_f32)
            
            # Add band descriptions with week dates
            for i, week_date in enumerate(times):
                dst.set_band_description(i + 1, f'Week_{week_date.strftime("%Y_%m_%d")}')
        
        # Verify output
        with rio.open(output_path) as src:
            verify_shape = (src.count, src.height, src.width)
            logger.info(f"Verified output shape: {verify_shape}")
            
            # Quick data quality check
            sample_band = src.read(1)
            finite_pixels = np.isfinite(sample_band).sum()
            total_pixels = sample_band.size
            finite_pct = 100 * finite_pixels / total_pixels
            
            value_range = (np.nanmin(sample_band), np.nanmax(sample_band))
            
            logger.info(f"Output data quality:")
            logger.info(f"  Valid pixels: {finite_pixels}/{total_pixels} ({finite_pct:.1f}%)")
            logger.info(f"  Value range: [{value_range[0]:.4f}, {value_range[1]:.4f}]")
            
            if finite_pct < 10:
                logger.warning("Very low percentage of valid pixels - check input data quality")
            elif finite_pct > 90:
                logger.info("Good data quality - high percentage of valid pixels")
    
    def process_year(self, year: int):
        """
        Process one year of biophysical data
        
        Args:
            year: Target year to process
        """
        logger.info(f"{'='*60}")
        logger.info(f"Processing year {year}")
        logger.info(f"Input: {self.input_dir}")
        logger.info(f"Output: {self.output_dir}")
        logger.info(f"{'='*60}")
        
        # Create biophysical files DataFrame
        biophys_df = self.create_biophys_df()
        
        # Filter to date range
        filtered_df = self.filter_date_range(biophys_df, year)
        
        if len(filtered_df) == 0:
            logger.error(f"No data found for year {year}")
            return False
        
        # Process each variable
        success_count = 0
        for variable in ['lai', 'fapar', 'fcover']:
            logger.info(f"\n{'-'*50}")
            logger.info(f"Processing {variable.upper()} for {year}")
            logger.info(f"{'-'*50}")
            
            try:
                # Load daily data
                daily_data, time_index, metadata = self.load_daily_data(filtered_df, variable)
                
                # Compute weekly aggregation 
                weekly_data, weekly_times = self.compute_weekly_aggregation(daily_data, time_index)
                
                # Select target 21 weeks
                selected_data, selected_times = self.select_target_weeks(weekly_data, weekly_times, year)
                
                # Handle NaN values
                cleaned_data = self.handle_nodata_values(selected_data)
                
                # Save datacube
                self.save_weekly_datacube(cleaned_data, variable, year, metadata, selected_times)
                
                logger.info(f"✓ Successfully processed {variable} for {year}")
                success_count += 1
                
            except Exception as e:
                logger.error(f"✗ Error processing {variable} for {year}: {e}")
                logger.exception("Full error details:")
                
        logger.info(f"\nYear {year} summary: {success_count}/3 variables processed successfully")
        return success_count == 3


def main():
    parser = argparse.ArgumentParser(
        description='Weekly biophysical data aggregation pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process zone 1 for year 2022
  python weekly_biophys_aggregation.py \\
    --input-dir /fs2/ftype_mflf/shared_data/temp/biophys/2025/1.0.0/mfl_cmf_area_project_2025_chr_zone_1 \\
    --output-dir /fs2/ftype_mflf/shared_data/datacube/2025/1.2.0/mfl_cmf_area_project_2025_chr_zone_1 \\
    --year 2022
    
  # Process multiple years
  for year in 2019 2020 2022 2023 2024; do
    python weekly_biophys_aggregation.py \\
      --input-dir /fs2/ftype_mflf/shared_data/temp/biophys/2025/1.0.0/mfl_cmf_area_project_2025_chr_zone_2 \\
      --output-dir /fs2/ftype_mflf/shared_data/datacube/2025/1.2.0/mfl_cmf_area_project_2025_chr_zone_2 \\
      --year $year
  done
        """
    )
    
    parser.add_argument('--input-dir', required=True,
                      help='Input directory with daily biophysical data')
    parser.add_argument('--output-dir', required=True,
                      help='Output directory for weekly datacubes')  
    parser.add_argument('--year', type=int, required=True,
                      help='Year to process (e.g. 2022)')
    parser.add_argument('--verbose', '-v', action='store_true',
                      help='Enable verbose logging')
    
    args = parser.parse_args()
    
    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    logger.info("Weekly Biophysical Data Aggregation Pipeline")
    logger.info(f"Input directory: {args.input_dir}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Target year: {args.year}")
    
    try:
        # Create aggregator and process
        aggregator = WeeklyBiophysicalAggregator(args.input_dir, args.output_dir)
        success = aggregator.process_year(args.year)
        
        if success:
            logger.info(f"✓ Successfully completed processing for year {args.year}")
            sys.exit(0)
        else:
            logger.error(f"✗ Processing failed for year {args.year}")
            sys.exit(1)
        
    except KeyboardInterrupt:
        logger.info("Processing interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Processing failed: {e}")
        logger.exception("Full error details:")
        sys.exit(1)


if __name__ == '__main__':
    main()