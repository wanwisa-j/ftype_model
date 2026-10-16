import rioxarray as rxr
import numpy as np
import os
import rasterio as rio
import geopandas as gpd
import xarray as xr
import matplotlib.pyplot as plt
import pandas as pd

from datetime import datetime
from shapely.ops import transform

def create_biophys_df(biophys_dir):
    fapar_files = sorted(os.listdir(f'{biophys_dir}/fapar'))
    dates = []
    filenames = []
    for file in fapar_files:
        # Extract date from s2_fapar_YYYY_MM_DD.tif pattern
        if file.startswith('s2_fapar_') and file.endswith('.tif'):
            date_part = file[9:-4]  # Remove 's2_fapar_' prefix and '.tif' suffix
            try:
                date_obj = datetime.strptime(date_part, '%Y_%m_%d')
                dates.append(date_obj)
                filenames.append(f'{biophys_dir}/fapar/{file}')
            except ValueError:
                continue

    fapar_dict = dict(zip(dates, filenames))
    
    lai_files = sorted(os.listdir(f'{biophys_dir}/lai'))
    dates = []
    filenames = []
    for file in lai_files:
        # Extract date from s2_lai_YYYY_MM_DD.tif pattern
        if file.startswith('s2_lai_') and file.endswith('.tif'):
            date_part = file[7:-4]  # Remove 's2_lai_' prefix and '.tif' suffix
            try:
                date_obj = datetime.strptime(date_part, '%Y_%m_%d')
                dates.append(date_obj)
                filenames.append(f'{biophys_dir}/lai/{file}')
            except ValueError:
                continue

    lai_dict = dict(zip(dates, filenames))
    
    fcover_files = sorted(os.listdir(f'{biophys_dir}/fcover'))
    dates = []
    filenames = []
    for file in fcover_files:
        # Extract date from s2_fcover_YYYY_MM_DD.tif pattern
        if file.startswith('s2_fcover_') and file.endswith('.tif'):
            date_part = file[10:-4]  # Remove 's2_fcover_' prefix and '.tif' suffix
            try:
                date_obj = datetime.strptime(date_part, '%Y_%m_%d')
                dates.append(date_obj)
                filenames.append(f'{biophys_dir}/fcover/{file}')
            except ValueError:
                continue

    fcover_dict = dict(zip(dates, filenames))
    
    common_dates = set(fapar_dict.keys()) & set(lai_dict.keys()) & set(fcover_dict.keys())

    # Create lists for the dataframe
    dates_list = []
    fapar_paths = []
    lai_paths = []
    fcover_paths = []
    dhssm_paths = []

    for date in sorted(common_dates):
        dates_list.append(date)
        fapar_paths.append(fapar_dict[date])
        lai_paths.append(lai_dict[date])
        fcover_paths.append(fcover_dict[date])
        # dhssm_paths.append(dhssm_dict[date])

    # Create the dataframe
    biophys_df = pd.DataFrame({
        'datetime': dates_list,
        'fapar': fapar_paths,
        'lai': lai_paths,
        'fcover': fcover_paths,
        # 'dhssm': dhssm_paths
    })

    print(f"Created dataframe with {len(biophys_df)} records")
    return biophys_df

def gen_dataset(df, var_name):
    data_arrays = []
    dates = []

    for _, row in df.iterrows():
        da = rxr.open_rasterio(row[var_name], chunks=True).squeeze()
        da = da.expand_dims('time')
        da = da.assign_coords(time=[row['datetime']])
        data_arrays.append(da)
        dates.append(row['datetime'])
    
    # Concatenate along time dimension
    dataset = xr.concat(data_arrays, dim='time')
    dataset = dataset.sortby('time')
    return dataset

def to_2d(geom):
	# Only keep x, y
	return transform(lambda x, y, z=None: (x, y), geom)

def mask_dataset(dataset, mask : gpd.GeoDataFrame):
    """
    Apply a mask to the dataset.
    """
    mask_crs = mask.crs
    if mask_crs is None:
        raise ValueError("Mask GeoDataFrame must have a defined CRS.")
    mask = mask.geometry.union_all()
    mask = mask.buffer(0)  # Ensure the mask is valid
    mask = gpd.GeoDataFrame(geometry=[mask], crs=mask_crs)
    mask = mask.dissolve()  
    mask = mask.to_crs(dataset.rio.crs)

    if mask.geometry.iloc[0].has_z:
        mask['geometry'] = mask.geometry.apply(to_2d)

    # Check for valid geometry before clipping
    if mask.geometry.is_empty.any() or mask.geometry.isna().any():
        raise ValueError("No valid geometry objects found for rasterize (mask is empty or invalid).")

    dataset = dataset.rio.clip(mask.geometry, crs=dataset.rio.crs, drop=True, all_touched=True)
    return dataset

def gen_data_feature(dataset_dict, vars, timesteps):
    tuple_data = ()
    for feature in vars:
        if feature in dataset_dict:
            data = dataset_dict[feature].values.reshape(timesteps, -1).transpose()
            tuple_data += (data,)
        else:
            print(f"Feature {feature} not found in dataset_dict")
    return np.dstack(tuple_data)

def prepare_biophys_dataset(biophys_df, start_date, end_date, vars=None, mask=None):
    """
    Stack biophysical data from the dataframe into a single xarray Dataset and aggregate weekly
    to match the training extraction logic (weekly aggregation, select last 4 weeks of previous year + weeks 1..17 of project year).
    """
    year = end_date.year if isinstance(end_date, datetime) else pd.to_datetime(end_date).year
    # Convert dates to datetime if strings
    if isinstance(start_date, str):
        start_date = pd.to_datetime(start_date)
    if isinstance(end_date, str):
        end_date = pd.to_datetime(end_date)
    
    # Filter biophys_df by date range
    filtered_df = biophys_df[
        (biophys_df['datetime'] >= start_date) & 
        (biophys_df['datetime'] <= end_date)
    ].sort_values('datetime').reset_index(drop=True)
    
    if len(filtered_df) == 0:
        print("No data found for the specified date range")
        return None
    
    print(f"Found {len(filtered_df)} time steps from {filtered_df['datetime'].min()} to {filtered_df['datetime'].max()}")
    datasets = {}
    results = {}
    
    for var in vars:
        print(f'Processing variable: {var}')
        # Build time-stacked dataset from files listed in filtered_df
        data_arrays = []
        for _, row in filtered_df.iterrows():
            da = rxr.open_rasterio(row[var], chunks='auto').squeeze()
            da = da.expand_dims('time')
            da = da.assign_coords(time=[row['datetime']])
            data_arrays.append(da)
        dataset = xr.concat(data_arrays, dim='time', chunks='auto')
        dataset = dataset.sortby('time')
        print(f'Created {var} dataset with shape: {dataset.shape}')
        
        if mask is not None:
            print(f'Applying mask to {var} dataset')
            dataset = mask_dataset(dataset, mask)
            print(f'Applied mask to {var} dataset, new shape: {dataset.shape}')
        
        # Ensure dataset has time dim
        if 'time' not in dataset.dims:
            dataset = dataset.expand_dims('time')
        # Compute weekly aggregation anchored to Monday (W-MON)
        time_index = pd.to_datetime(dataset['time'].values)
        week_start = pd.Series(time_index).dt.to_period('W-MON').dt.start_time
        dataset = dataset.assign_coords(week_start=('time', week_start.values))
        weekly_mean = dataset.groupby('week_start').mean('time')
        
        # Determine desired week starts: last 4 weeks before year start + weeks 1..17 of project year
        year_start = pd.Timestamp(f'{year}-01-01')
        all_week_starts = pd.to_datetime(pd.Series(week_start.unique()).values)
        prev_year_weeks = sorted([w for w in all_week_starts if w < year_start])
        last4_prev_weeks = prev_year_weeks[-4:]
        weekly_times = pd.to_datetime(weekly_mean['week_start'].values)
        iso = weekly_times.isocalendar()
        mask_current = (weekly_times >= year_start) & (iso.year == year) & (iso.week <= 17)
        desired_weeks = set(last4_prev_weeks) | set(weekly_times[mask_current])
        available_weeks = set(weekly_times)
        selected_week_starts = sorted(desired_weeks & available_weeks)
        missing_weeks = sorted(desired_weeks - available_weeks)
        if missing_weeks:
            print(f"Warning: the following requested week_start values are not present in the data and will be skipped: {missing_weeks}")
        if len(selected_week_starts) == 0:
            raise ValueError("No weekly timesteps available after filtering for the requested weeks.")
        sampled_weekly = weekly_mean.sel(week_start=selected_week_starts)
        sampled_weekly = sampled_weekly.rename({'week_start': 'time'})
        agg_times = pd.to_datetime(sampled_weekly['time'].values)
        sampled_weekly = sampled_weekly.assign_coords(week_number=('time', agg_times.isocalendar().week.to_numpy()))
        
        # Align to canonical desired 21-week sequence (last 4 prev + weeks 1..17) using the existing weekday
        existing_weekdays = pd.Series(pd.to_datetime(sampled_weekly.time.values)).dt.weekday.unique()
        jan4 = pd.Timestamp(f'{year}-01-04')
        week1_monday = jan4 - pd.Timedelta(days=jan4.weekday())
        if len(existing_weekdays) == 1:
            day_shift = int(existing_weekdays[0])
        else:
            day_shift = 0
        desired_weeks_list = [week1_monday + pd.Timedelta(weeks=w) + pd.Timedelta(days=day_shift) for w in range(-4, 17)]
        desired_weeks_list = sorted(desired_weeks_list)
        existing_set = set(pd.to_datetime(sampled_weekly.time.values))
        missing_weeks_full = [d for d in desired_weeks_list if d not in existing_set]
        if missing_weeks_full:
            print(f'Info: inserting {len(missing_weeks_full)} missing weeks using nearest edge values: {missing_weeks_full}')
            sampled_weekly = sampled_weekly.interp(time=desired_weeks_list, method='nearest')
        else:
            sampled_weekly = sampled_weekly.sel(time=desired_weeks_list)
        if sampled_weekly.sizes.get('time', 0) != 21:
            print(f"Warning: expected 21 weekly timesteps, found {sampled_weekly.sizes.get('time')}")
            
        if sampled_weekly.isnull().any():
            # Rechunk time to a single chunk for interpolate_na to work with dask
            if hasattr(sampled_weekly, 'chunks') and sampled_weekly.chunks is not None:
                sampled_weekly = sampled_weekly.chunk({'time': -1})
            sampled_weekly = sampled_weekly.interpolate_na(dim='time', method='linear', fill_value="extrapolate")

        results[var] = sampled_weekly
    
    # Build return structure similar to training extraction: 'data' dict and 'dates' list
    dates = []
    if results:
        first_da = next(iter(results.values()))
        weekly_dates = pd.to_datetime(first_da['time'].values)
        weekly_dates = pd.Index(weekly_dates).sort_values().unique()
    else:
        weekly_dates = pd.date_range(start=f'{year-1}-12-01', end=f'{year}-04-30', freq='W-MON')
    weekly_dates = list(pd.to_datetime(weekly_dates))
    dates.extend(weekly_dates)
    result_dict = {
        'data': results,
        'dates': dates,
    }
    return result_dict


for zone in ['zone_1', 'zone_2', 'zone_3', 'zone_4', 'zone_5']:
    try:
        out_dir_cube = f'/fs2/ftype_mflf/shared_data/datacube/2025/1.1.1/chm_project_2025/{zone}/'
        os.makedirs(out_dir_cube, exist_ok=True)
        biophys_df = create_biophys_df(f'/fs2/ftype_mflf/shared_data/temp/biophys/2025/1.0.0/chm_project_2025/{zone}/')
        
        for year in range(2025, 2026):
            
            if os.path.exists(f'{out_dir_cube}/biophys_data_{year}.npy'):
                print(f"Output file for zone {zone} year {year} already exists, skipping processing.")
                continue

            result_dict = prepare_biophys_dataset(
                biophys_df,
                start_date=f'{year-1}-11-15',
                end_date=f'{year}-04-30',
                vars=['lai', 'fapar', 'fcover'],
                )
            mask_processing_refer_path = f'/fs2/ftype_mflf/shared_data/datacube/2025/1.0.0/chm_project_2025/{zone}/mask_processing_{year}.tif'
            if os.path.exists(mask_processing_refer_path):
                mask_processing_refer = rxr.open_rasterio(mask_processing_refer_path)
                for keys, value in result_dict['data'].items():
                    if len(mask_processing_refer.y) == (len(value.y)) and len(mask_processing_refer.x) == (len(value.x)):
                        print(f"Shapes match for {keys}: {mask_processing_refer.shape}")
                    else:
                        print(f"Shapes do not match for {keys}: {mask_processing_refer.shape} vs {value.shape}")
                        # Reproject each variable dataset to match mask_processing_refer x/y grid
                        target = mask_processing_refer.squeeze()
                        try:
                            # Reproject/resample to the mask grid (use bilinear for continuous biophysical layers)
                            reprojected = value.rio.reproject_match(target, resampling=rio.enums.Resampling.bilinear)
                            result_dict['data'][keys] = reprojected
                            if len(target.y) == len(reprojected.y) and len(target.x) == len(reprojected.x):
                                print(f"Reprojected {keys} -> shape {reprojected.shape}")
                            else:
                                print(f"Reprojected {keys} but shapes differ: target {target.shape} vs reprojected {reprojected.shape}")
                        except Exception as e:
                            print(f"Error reprojecting {keys}: {e}")

            mask_processing = result_dict['data']['lai'].isnull().all(dim='time')
            mask_processing_array = mask_processing.values
            mask_processing.rio.to_raster(f'{out_dir_cube}/mask_processing_{year}.tif',
                                        dtype='uint8',
                                        nodata=False,
                                        driver='GTiff',
                                        compress='LZW')
            features_py = gen_data_feature(result_dict['data'], vars=['lai', 'fapar', 'fcover'], timesteps=21)
            with open(f'{out_dir_cube}/biophys_data_{year}.npy', 'wb') as f:
                np.save(f, features_py)
    except Exception as e:
        print(f"Error processing zone {zone} year {year}: {e}")
        continue
