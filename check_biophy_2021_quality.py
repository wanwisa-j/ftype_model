import rasterio
import numpy as np
import os

# Define zones and features
zones = [
    "mfl_cmf_area_project_2025_chr_zone_1",
    "mfl_cmf_area_project_2025_chr_zone_2"
]

biophy_features = ["lai", "fapar", "fcover"]
gee_features = ["ndvi", "gvmi", "slavi", "vh", "vv"]
terrain_features = ["elevation", "slope"]

base_dir = "/fs2/ftype_mflf/shared_data/datacube/2025/1.2.0"

# Helper to print raster info and nodata stats
def check_raster(path, feature_name):
    if not os.path.exists(path):
        print(f"❌ {feature_name}: File not found: {path}")
        return
    with rasterio.open(path) as src:
        arr = src.read()
        nodata = src.nodata
        arr_flat = arr.flatten()
        if nodata is not None:
            nodata_mask = (arr_flat == nodata)
        else:
            nodata_mask = np.isnan(arr_flat)
        nodata_pct = 100 * np.sum(nodata_mask) / arr_flat.size
        print(f"{feature_name}: shape={src.shape}, bands={src.count}, CRS={src.crs},\n  affine={src.transform}")
        print(f"  NoData value: {nodata}, % NoData/NaN: {nodata_pct:.2f}%\n")

for zone in zones:
    print(f"\n=== Zone: {zone} (2021) ===")
    # Biophy features
    for feat in biophy_features:
        path = f"{base_dir}/{zone}/{feat}/cube_biophys_{feat}_2021_W4before_to_W16after_21bands.tif"
        check_raster(path, f"biophy_{feat}")
    # GEE features
    for feat in gee_features:
        path = f"{base_dir}/{zone}/{feat}/cube_s2_{feat}_2021_W4before_to_W16after_21bands.tif"
        check_raster(path, f"gee_{feat}")
    # Terrain features (static)
    for feat in terrain_features:
        path = f"{base_dir}/{zone}/{feat}/{feat}_data.tif"
        check_raster(path, f"terrain_{feat}")
