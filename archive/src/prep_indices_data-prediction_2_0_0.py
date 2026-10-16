import rioxarray as rxr
import numpy as np
import os
import rasterio as rio
import geopandas as gpd
import xarray as xr
import matplotlib.pyplot as plt
import pandas as pd
from rasterio.enums import Resampling
from datetime import datetime
from shapely.ops import transform


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

def downsample_raster(raster, downscale_factor=0.5, resampling=Resampling.bilinear):
    """
    Downsample a raster by a given factor using bilinear resampling.
    
    Parameters:
    - raster: xarray DataArray or Dataset with raster data.
    - downscale_factor: Factor by which to downsample the raster.
    
    Returns:
    - Downsampled raster as an xarray DataArray.
    """
    if not isinstance(raster, (xr.DataArray, xr.Dataset)):
        raise ValueError("Input must be an xarray DataArray or Dataset.")
    
    # Calculate new dimensions
    new_width = raster.rio.width * downscale_factor
    new_height = raster.rio.height * downscale_factor

    # Downsample raster
    down_sampled = raster.rio.reproject(raster.rio.crs, shape=(int(new_height), int(new_width)), resampling=resampling)

    return down_sampled

def prepare_indices_monthly(year, zone, indices_paths, vars=None, mask=None):
    datasets = {}

    for var in vars:
        data_arrays = []
        dates = []
        if var in ['VV', 'VH']:
            satellite = 's1'
        else:
            satellite = 's2'

        for month in range(1, 5):
            month_str = f'{month:02d}'
            index_file = os.path.join(indices_paths, f'max_{satellite}_{var}_{year}_{month_str}_mfl_chm_{zone}.tif')
            if not os.path.exists(index_file):
                print(f"Index file {index_file} does not exist, skipping...")
                continue
            
            da = rxr.open_rasterio(index_file, chunks=True).squeeze()
            da = da.expand_dims('time')
            da = da.assign_coords(time=[f'{year}-{month_str}'])
            data_arrays.append(da)
            dates.append(f'{year}-{month_str}')

        # Concatenate along time dimension
        dataset = xr.concat(data_arrays, dim='time')
        dataset = dataset.sortby('time')

        if mask is not None:
            dataset = mask_dataset(dataset, mask)
        datasets[var] = dataset
        print(f'Created {var} dataset with shape: {dataset.shape}')
        
    return datasets

zone = 'zone_3'
# year = 2019
# for zone in [f'zone_{i}' for i in range(2, 3)]:
for year in [2019, 2020, 2021, 2022, 2023, 2024, 2025]:
    out_dir_cube = f'/fs2/ftype_mflf/shared_data/datacube/2025/1.0.0/chm_project_2025/{zone}/'
    os.makedirs(out_dir_cube, exist_ok=True)

    indices_paths = f'/fs2/ftype_mflf/shared_data/temp/chm/chm_{zone}/{year}'
    datasets_indices = prepare_indices_monthly(year, zone, indices_paths, vars=['VV', 'VH', 'gvmi', 'ndvi', 'slavi'])
    features_py = gen_data_feature(datasets_indices, vars=['VV', 'VH', 'gvmi', 'ndvi', 'slavi'], timesteps=4)

    with open(f'{out_dir_cube}/indices_data_{year}.npy', 'wb') as f:
        np.save(f, features_py)

