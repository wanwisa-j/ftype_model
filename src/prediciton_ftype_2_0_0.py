import numpy as np
import tensorflow as tf
from tensorflow.keras import models
from scipy.stats import mode

import xarray as xr
import rioxarray as rxr
import os
# Documentation for data preparation for prediction

# The model requires three inputs:
# 1. Biophysical data: time series of biophysical measurements
# 2. Indices data: time series of remote sensing indices 
# 3. Static features: elevation and slope values

# Input shapes
# - Biophysical data: (samples, 105, 3) - 105 time steps with 3 features per step
# - Indices data: (samples, 4, 6) - 4 time steps with 6 features per step
# - Static features: (samples, 2) - 2 features (elevation and slope)

def prepare_data_for_prediction(biophys_data, indices_data, elev_values, slope_values):
    """
    Prepare data for prediction with the forest type classification model
    
    Parameters:
    -----------
    biophys_data : numpy.ndarray
        Raw biophysical time series data with shape (samples, 151, 3)
    indices_data : numpy.ndarray
        Raw indices data with shape (samples, 4, 5) - before adding VV/VH channel
    elev_values : numpy.ndarray
        Elevation values with shape (samples,)
    slope_values : numpy.ndarray
        Slope values with shape (samples,)
        
    Returns:
    --------
    list of numpy arrays ready for model prediction
    """
    # 1. Process biophysical data - extract specific time steps
    biophys_data = biophys_data[:, 31:-15, :]  # Keep time steps 31 to (end-15)
    
    # 2. Process indices data - add VV/VH ratio channel
    vv_vh_ratio = indices_data[:, :, 0] / np.where(indices_data[:, :, 1] == 0, 1e-6, indices_data[:, :, 1])
    indices_data_with_ratio = np.concatenate((indices_data, vv_vh_ratio[..., np.newaxis]), axis=-1)
    
    # 3. Combine static features
    static_features = np.vstack((elev_values, slope_values)).T
    
    return [biophys_data, indices_data_with_ratio, static_features]

    # Example usage:
    # prediction = model.predict(prepare_data_for_prediction(
    #     new_biophys_data, new_indices_data, new_elev_values, new_slope_values
    # ))

    # Feature reference:
    # Biophysical data (3 channels): [lai, fapar, fcover]
    # Indices data (6 channels): [VV, VH, gvmi, ndvi, slavi, VV/VH_ratio]
    # Static features (2 channels): [elevation, slope]

def check_mismatch_shapes(biophys_data, indices_data, elev_data, slope_data, mask_array):
    mismatch = False
    if biophys_data.shape[0] != mask_array.shape[0]:
        print('biophys data shape mismatch')
        mismatch = True
    if indices_data.shape[0] != mask_array.shape[0]:
        print('indices data shape mismatch')
        mismatch = True
    if elev_data.shape[0] != mask_array.shape[0]:
        print('elev data shape mismatch')
        mismatch = True
    if slope_data.shape[0] != mask_array.shape[0]:
        print('slope data shape mismatch')
        mismatch = True
    return mismatch

aoi_name = 'zone_3'
model = models.load_model('/fs1/tor_project/model_foresttype/model/mfl_v2_0_0/hybrid_lstm_cnn_model-20250806.keras',
                          compile=False,)
data_set_path = f'/fs2/ftype_mflf/shared_data/datacube/2025/1.0.0/chm_project_2025/{aoi_name}/'
out_path = f'/fs2/ftype_mflf/result/2025/1.0.0/chm_project_2025/{aoi_name}/model_version_2.0.0'
os.makedirs(out_path, exist_ok=True)

prediction_class_tuple = ()
for year in range(2019, 2026):
    out_file = f'{out_path}/pred_{year}_2.0.0-class.tif'


    biophys_data_all = np.load(f'{data_set_path}/biophys_data_{year}.npy')
    indices_data_all = np.load(f'{data_set_path}/indices_data_{year}.npy')
    elev_data_all = np.load(f'{data_set_path}/elev_data_all.npy')
    slope_data_all = np.load(f'{data_set_path}/slope_data_all.npy')

    metadata = rxr.open_rasterio(f'{data_set_path}/mask_processing_{year}.tif')
    mask_processing_array = metadata.squeeze().values.astype(bool)
    mask_processing_array_reshaped = mask_processing_array.reshape(-1)

    if elev_data_all.shape[0] > mask_processing_array_reshaped.shape[0] or slope_data_all.shape[0] > mask_processing_array_reshaped.shape[0]:
        elev_data_all = elev_data_all[:mask_processing_array_reshaped.shape[0]]
        slope_data_all = slope_data_all[:mask_processing_array_reshaped.shape[0]]

    if check_mismatch_shapes(biophys_data_all, indices_data_all, elev_data_all, slope_data_all, mask_processing_array_reshaped):
        continue

    biophys_data_all, indices_data_all, elev_data_all, slope_data_all = biophys_data_all[~mask_processing_array_reshaped], indices_data_all[~mask_processing_array_reshaped], elev_data_all[~mask_processing_array_reshaped], slope_data_all[~mask_processing_array_reshaped]
    
    prediction_proba = model.predict(prepare_data_for_prediction(biophys_data_all, indices_data_all, elev_data_all, slope_data_all), batch_size=32*4)
    prediction_class = np.argmax(prediction_proba, axis=1)
    
    prediction_data = np.full(mask_processing_array_reshaped.shape, -999, dtype=np.int32)
    prediction_data[~mask_processing_array_reshaped] = prediction_class
    prediction_data = prediction_data.reshape(mask_processing_array.shape)

    # prediction_class_tuple += (prediction_data,)

    prediction_data = xr.DataArray(
                        prediction_data,
                        dims=('y', 'x'),
                        coords={'y': metadata.coords['y'], 'x': metadata.coords['x']},
                        attrs=metadata.attrs
                    )
    
    prediction_data = prediction_data.rio.write_nodata(-999)
    prediction_data = prediction_data.rio.write_crs(metadata.rio.crs)
    prediction_data.rio.to_raster(out_file, dtype='int32', 
                                    driver='GTiff', crs=metadata.rio.crs, transform=metadata.rio.transform(), nodata=-999)
    
    prediction_data_proba = np.full((mask_processing_array_reshaped.shape[0], 5), -999, dtype=np.float32)
    prediction_data_proba[~mask_processing_array_reshaped] = prediction_proba
    prediction_data_proba = prediction_data_proba.T.reshape(5, *mask_processing_array.shape)
    prediction_data_proba = xr.DataArray(
                                prediction_data_proba,
                                dims=('class_prob', 'y', 'x'),
                                coords={'class_prob': np.arange(5), 'y': metadata.coords['y'], 'x': metadata.coords['x']},
                                attrs=metadata.attrs
                            )
    prediction_data_proba = prediction_data_proba.rio.write_nodata(-999)
    prediction_data_proba = prediction_data_proba.rio.write_crs(metadata.rio.crs)
    prediction_data_proba.rio.to_raster(out_file.replace('-class.tif', '-proba.tif'), dtype='float32', 
                                        driver='GTiff', crs=metadata.rio.crs, transform=metadata.rio.transform(), nodata=-999)
    
prediction_class_tuple = np.dstack(prediction_class_tuple)

height, width, num_years = prediction_class_tuple.shape
prediction_class_tuple = prediction_class_tuple.reshape((-1, prediction_class_tuple.shape[-1]))
mode_values, mode_counts = mode(prediction_class_tuple, axis=1)
mode_values_reshape = mode_values.reshape((height, width))

mode_values_reshape = xr.DataArray(
    mode_values_reshape,
    dims=('y', 'x'),
    coords={'y': metadata.coords['y'], 'x': metadata.coords['x']},
    attrs=metadata.attrs
)

mode_values_reshape = mode_values_reshape.rio.write_nodata(-999)
mode_values_reshape = mode_values_reshape.rio.write_crs(4326)

mode_values_reshape.rio.to_raster(f'{out_path}/pred_mode_2.0.0.tif', dtype='int32', 
                                    driver='GTiff', transform=metadata.rio.transform())