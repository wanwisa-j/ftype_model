import rasterio
from rasterio.plot import show
import matplotlib.pyplot as plt
import os

def compare_rasters(zone, year, lai_file, ndvi_file, vv_file, elevation_file):
    """
    Compare LAI, NDVI, VV, and elevation rasters for a given zone and year.

    Parameters:
        zone (str): Zone identifier.
        year (int): Year of the data.
        lai_file (str): Path to the LAI raster file.
        ndvi_file (str): Path to the NDVI raster file.
        vv_file (str): Path to the VV raster file.
        elevation_file (str): Path to the elevation raster file.
    """
    datasets = {
        "LAI": lai_file,
        "NDVI": ndvi_file,
        "VV": vv_file,
        "Elevation": elevation_file
    }

    metadata = {}

    # Load and compare metadata
    for name, file in datasets.items():
        if not os.path.exists(file):
            print(f"Error: {name} file not found at {file}")
            return

        with rasterio.open(file) as src:
            metadata[name] = {
                "shape": src.shape,
                "crs": src.crs,
                "transform": src.transform
            }

    # Print metadata comparison
    print("\nMetadata Comparison:")
    for name, meta in metadata.items():
        print(f"\n{name}:")
        print(f"  Shape: {meta['shape']}")
        print(f"  CRS: {meta['crs']}")
        print(f"  Transform: {meta['transform']}")

    # Visualize rasters
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    for idx, (name, file) in enumerate(datasets.items()):
        with rasterio.open(file) as src:
            show(src, ax=axes[idx], title=name)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Define file paths for Zone 1 and Zone 2, Year 2023
    year = 2023

    # Zone 1
    zone1 = "mfl_cmf_area_project_2025_chr_zone_1"
    lai_file1 = f"/fs2/ftype_mflf/shared_data/datacube/2025/1.2.0/{zone1}/lai/cube_biophys_lai_{year}_W4before_to_W16after_21bands.tif"
    ndvi_file1 = f"/fs2/ftype_mflf/shared_data/datacube/2025/1.2.0/{zone1}/ndvi/cube_s2_ndvi_{year}_W4before_to_W16after_21bands.tif"
    vv_file1 = f"/fs2/ftype_mflf/shared_data/datacube/2025/1.2.0/{zone1}/vv/cube_s1_vv_{year}_W4before_to_W16after_21bands.tif"
    elevation_file1 = f"/fs2/ftype_mflf/shared_data/datacube/2025/1.2.0/{zone1}/elevation/elevation_data.tif"
    print(f"\n--- Checking {zone1} ---")
    compare_rasters(zone1, year, lai_file1, ndvi_file1, vv_file1, elevation_file1)

    # Zone 2
    zone2 = "mfl_cmf_area_project_2025_chr_zone_2"
    lai_file2 = f"/fs2/ftype_mflf/shared_data/datacube/2025/1.2.0/{zone2}/lai/cube_biophys_lai_{year}_W4before_to_W16after_21bands.tif"
    ndvi_file2 = f"/fs2/ftype_mflf/shared_data/datacube/2025/1.2.0/{zone2}/ndvi/cube_s2_ndvi_{year}_W4before_to_W16after_21bands.tif"
    vv_file2 = f"/fs2/ftype_mflf/shared_data/datacube/2025/1.2.0/{zone2}/vv/cube_s1_vv_{year}_W4before_to_W16after_21bands.tif"
    elevation_file2 = f"/fs2/ftype_mflf/shared_data/datacube/2025/1.2.0/{zone2}/elevation/elevation_data.tif"
    print(f"\n--- Checking {zone2} ---")
    compare_rasters(zone2, year, lai_file2, ndvi_file2, vv_file2, elevation_file2)
