import rasterio
import rioxarray
import xarray as xr
import numpy as np
import tensorflow as tf
import os
import pandas as pd

from rasterio.merge import merge
from rasterio.mask import mask
from rasterio.features import geometry_mask
from scipy.stats import mode
from tensorflow.keras.models import load_model
import argparse
import yaml
import glob

gpus = tf.config.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
logical_gpus = tf.config.list_logical_devices('GPU')

with open('/fs1/tor_project/model_foresttype/config/config.yaml', 'r') as f:
    config = yaml.safe_load(f)

cal_majority = config.get('cal_majority', True)
retroactive_years = config.get('retroactive_years', 5)
model_version = config['model']['model_version']
model_file = config['model']['model_file']
model = load_model(f'{config["model"]["path"]}/{model_file}')

def open_raster_and_meta(raster_path, **kwargs):
    band_number = kwargs.get('band_number', 1)
    with rasterio.open(raster_path, 'r') as src:
        raster_array = src.read(indexes=band_number)
        metadata = src.meta.copy()
    del src
    
    return raster_array, metadata

def matches_pattern_name(pattern):
    matched_files = glob.glob(pattern)
    if not matched_files:
        raise FileNotFoundError(f"No file matches pattern: {pattern}")
    return matched_files[0]

parser = argparse.ArgumentParser(description='Prediction of forest type')
parser.add_argument('feature_path', help='Path to the feature data')
parser.add_argument('prediction_year', help='Year for which to make predictions', type=int)
parser.add_argument('--outpath', help='Output path for the predictions', type=str, nargs='?', default=None)


args = parser.parse_args()
feature_path = args.feature_path
prediction_year = args.prediction_year

outpath = args.outpath
if outpath is None:
    outpath = f'{feature_path}/predictions'
outpath = f'{outpath}/{model_version}'

if not os.path.isdir(outpath):
    os.makedirs(outpath, exist_ok=True)

years = range(prediction_year, prediction_year - retroactive_years, -1)

for year_predict in years:
    data_array = []
    for vi in ['ndvi', 'gvmi', 'slavi', 's1']:
        data_stack = []
        for mm in range(1, 5):
            if vi == 's1':
                pattern = f'{feature_path}/{year_predict}/max_s1_VV_{year_predict}_{mm:02}*.tif'
                raster_array_vv, metadata = open_raster_and_meta(matches_pattern_name(pattern))

                pattern = f'{feature_path}/{year_predict}/max_s1_VH_{year_predict}_{mm:02}*.tif'
                raster_array_vh, metadata = open_raster_and_meta(matches_pattern_name(pattern))
                raster_array = raster_array_vv / raster_array_vh
            else:
                pattern = f'{feature_path}/{year_predict}/max_s2_{vi}_{year_predict}_{mm:02}*.tif'
                raster_array, metadata = open_raster_and_meta(matches_pattern_name(pattern))
            data_stack.append(raster_array)
        data_array.append(np.array(data_stack))
        
    tuple_data = ()
    for feature in range(len(['ndvi', 'gvmi', 'slavi', 's1'])):
        tuple_data += (data_array[feature].reshape(4, -1).transpose(),)
    
    features = np.dstack(tuple_data)
    prediction_result = model.predict(features)
    
    prediction_result_array = np.argmax(prediction_result, axis=1).reshape(1, metadata['height'], metadata['width'])
    prediction_result_array = prediction_result_array.astype(metadata['dtype'])
    
    
    output_path = f'{outpath}/pred_{year_predict}_{model_version}.tif'

    with rasterio.open(output_path, 'w', **metadata) as dst:
        dst.write(prediction_result_array)
    
    del prediction_result_array

if cal_majority:
    prediction_array = []
    for yy in years:
        raster_array, metadata = open_raster_and_meta(f'{outpath}/pred_{yy}_{model_version}.tif')
        prediction_array.append(raster_array)

    tuple_data = ()
    for prediction in range(len(prediction_array)):
        tuple_data += (prediction_array[prediction],)
        
    prediction_array = np.dstack(tuple_data)
    reshaped_data = prediction_array.reshape((-1, prediction_array.shape[-1]))
    mode_values, mode_counts = mode(reshaped_data, axis=1)
    mode_values_reshape = mode_values.reshape((prediction_array.shape[0], prediction_array.shape[1]))

    output_path = f'{outpath}/pred_mode_{model_version}.tif'

    with rasterio.open(output_path, 'w', **metadata) as dst:
        dst.write(mode_values_reshape, 1)