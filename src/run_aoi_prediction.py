#!/usr/bin/env python3
"""
Run forest type prediction (v2.2.x) for an AOI and summarise it per polygon.

All paths come from config/config.yaml (`prediction` + `aoi` sections):

  1. Find which datacube zones the AOI polygons fall in.
  2. Predict every year, reading only the window around the AOI polygons and
     predicting only pixels inside them (skips zone-years whose class raster
     already exists unless --force):
       outputs/aoi/<aoi stem>/prediction/<zone>/model_version_<v>/pred_<year>_*.tif
     With --full-zone, predict whole zones into `prediction.output_dir` instead
     (shared by all AOIs, much slower).
  3. Clip the yearly class rasters to each polygon, compute a per-pixel
     majority across years, and write:
       outputs/aoi/<aoi stem>/summary_long.csv      polygon x year x class areas
       outputs/aoi/<aoi stem>/summary_majority.csv  one row per polygon
       outputs/aoi/<aoi stem>/<aoi stem>_ftype.geojson
       outputs/aoi/<aoi stem>/rasters/<zone>_majority.tif  (AOI-masked)

Usage:
    python src/run_aoi_prediction.py [--aoi path.geojson] [--force] [--skip-predict] [--full-zone]
"""
import argparse
import glob
import math
import os
import subprocess
import sys

import geopandas as gpd
import joblib
import numpy as np
import pandas as pd
import rasterio
from rasterio.features import geometry_mask
from rasterio.windows import Window, from_bounds
from shapely.geometry import box
import yaml

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SRC_DIR)
NODATA = -999
M2_PER_RAI = 1600.0


def repo_path(p):
    return p if os.path.isabs(p) else os.path.join(REPO_ROOT, p)


def zone_footprints(datacube_dir):
    """Footprint polygon + CRS of every zone, from its slope raster."""
    rows = []
    for zone_dir in sorted(glob.glob(os.path.join(datacube_dir, '*'))):
        zone = os.path.basename(zone_dir)
        refs = sorted(glob.glob(os.path.join(zone_dir, 'slope', '*.tif')))
        if '_backup_' in zone or not refs:
            continue
        with rasterio.open(refs[0]) as src:
            rows.append({'zone': zone, 'crs': src.crs, 'geometry': box(*src.bounds)})
    return rows


def assign_zones(aoi, footprints):
    """Zone each polygon falls in (largest overlap); None if outside all zones."""
    best = pd.Series([None] * len(aoi), index=aoi.index, dtype=object)
    best_area = pd.Series(0.0, index=aoi.index)
    for fp in footprints:
        geoms = aoi.geometry.to_crs(fp['crs'])
        inter = geoms.intersection(fp['geometry']).area
        take = inter > best_area
        best[take] = fp['zone']
        best_area[take] = inter[take]
    return best


def class_names(cfg):
    version_dir = f"mfl_v{cfg['model_version'].replace('.', '_')}"
    found = glob.glob(os.path.join(repo_path(cfg['model_root']), version_dir, 'config', 'label_encoder_*.joblib'))
    if not found:
        raise FileNotFoundError(f'No label encoder for model {version_dir}')
    return [str(c) for c in joblib.load(found[0]).classes_]


def pred_file(out_dir, zone, year, version):
    return os.path.join(out_dir, zone, f'model_version_{version}',
                        f"pred_{year}_{version.replace('.', '_')}-class.tif")


def run_prediction_full_zone(zones, cfg, out_dir, force):
    years = cfg['years']
    todo = [z for z in zones
            if force or not all(os.path.exists(pred_file(out_dir, z, y, cfg['model_version'])) for y in years)]
    if not todo:
        print('All zone predictions already exist, skipping prediction (use --force to rerun).')
        return
    cmd = [sys.executable, os.path.join(SRC_DIR, 'prediction_ftype_2_2_1.py'),
           '--zones', ','.join(todo), '--output_dir', out_dir]
    print('Running:', ' '.join(cmd))
    subprocess.run(cmd, check=True)


def aoi_window(part, transform, shape):
    """Pixel window (snapped outwards to whole pixels) covering `part`, clipped to the raster."""
    w = from_bounds(*part.total_bounds, transform=transform)
    col0, row0 = math.floor(w.col_off), math.floor(w.row_off)
    col1, row1 = math.ceil(w.col_off + w.width), math.ceil(w.row_off + w.height)
    return Window(col0, row0, col1 - col0, row1 - row0).intersection(Window(0, 0, shape[1], shape[0]))


def run_prediction_aoi(aoi, zones, cfg, pred_dir, force):
    """Predict only the pixels inside the AOI polygons, zone by zone, in this process."""
    import prediction_ftype_2_2_1 as P
    from tensorflow.keras import models

    version, years = cfg['model_version'], cfg['years']
    datacube_dir = repo_path(cfg['datacube_dir'])
    model = scaling = None

    for zone in zones:
        todo = [y for y in years if force or not os.path.exists(pred_file(pred_dir, zone, y, version))]
        if not todo:
            print(f'{zone}: all years already predicted, skipping (use --force to rerun)')
            continue
        if model is None:
            artifacts = P.detect_model_artifacts(version)
            model = models.load_model(artifacts['model_path'], compile=False)
            scaling = P.load_scaling_params(artifacts['config_dir'], model_version=version)
            print(f"Loaded model: {artifacts['model_path']}")

        ref = sorted(glob.glob(os.path.join(datacube_dir, zone, 'slope', '*.tif')))[0]
        with rasterio.open(ref) as src:
            transform, shape, crs = src.transform, src.shape, src.crs
        part = aoi[aoi['zone'] == zone].to_crs(crs)
        win = aoi_window(part, transform, shape)
        wshape = (int(win.height), int(win.width))
        outside = geometry_mask(part.geometry, out_shape=wshape,
                                transform=rasterio.windows.transform(win, transform))
        print(f'{zone}: window {wshape[1]}x{wshape[0]} px, {int((~outside).sum()):,} px inside AOI '
              f'(zone {shape[1]}x{shape[0]})')

        for year in todo:
            P.process_zone_year(datacube_dir, zone, year, model, scaling, pred_dir, version,
                                window=win, extra_mask=outside, batch_size=cfg.get('batch_size', 128))


def majority(stack):
    """Per-pixel mode over axis 0 ignoring NODATA; ties -> smallest class (as scipy.stats.mode)."""
    n_cls = int(stack[stack != NODATA].max()) + 1 if (stack != NODATA).any() else 1
    counts = np.stack([(stack == c).sum(axis=0) for c in range(n_cls)])
    out = counts.argmax(axis=0).astype(np.int16)
    out[counts.sum(axis=0) == 0] = NODATA
    return out


def summarise(aoi, cfg, cfg_aoi, pred_dir, aoi_out, classes):
    version, years = cfg['model_version'], cfg['years']
    id_fields = [f for f in cfg_aoi['id_fields'] if f in aoi.columns]
    long_rows, maj_rows = [], []
    os.makedirs(os.path.join(aoi_out, 'rasters'), exist_ok=True)

    for zone, part in aoi.groupby('zone'):
        files = {y: pred_file(pred_dir, zone, y, version) for y in years}
        missing = [y for y, f in files.items() if not os.path.exists(f)]
        if missing:
            print(f'WARNING: {zone} missing predictions for {missing}, skipped those years')
        files = {y: f for y, f in files.items() if y not in missing}
        if not files:
            continue

        with rasterio.open(next(iter(files.values()))) as ref:
            crs, transform, shape, profile = ref.crs, ref.transform, ref.shape, ref.profile
        part = part.to_crs(crs)
        px_rai = abs(transform.a * transform.e) / M2_PER_RAI

        # Read only the window covering this zone's AOI polygons.
        win = from_bounds(*part.total_bounds, transform=transform).round_offsets().round_lengths()
        win = win.intersection(rasterio.windows.Window(0, 0, shape[1], shape[0]))
        wtr = rasterio.windows.transform(win, transform)
        wshape = (int(win.height), int(win.width))
        stack = []
        for y in files:
            with rasterio.open(files[y]) as src:
                stack.append(src.read(1, window=win))
        stack = np.stack(stack)
        maj = majority(stack)
        layers = dict(zip(files, stack)); layers['majority'] = maj

        aoi_mask = geometry_mask(part.geometry, out_shape=wshape, transform=wtr, invert=True)
        masked = np.where(aoi_mask, maj, NODATA).astype(np.int16)
        profile.update(height=wshape[0], width=wshape[1], transform=wtr, count=1, dtype='int16', nodata=NODATA)
        with rasterio.open(os.path.join(aoi_out, 'rasters', f'{zone}_majority.tif'), 'w', **profile) as dst:
            dst.write(masked, 1)

        for idx, row in part.iterrows():
            inside = geometry_mask([row.geometry], out_shape=wshape, transform=wtr, invert=True)
            ids = {f: row[f] for f in id_fields}
            for period, arr in layers.items():
                vals = arr[inside]
                n_valid = int((vals != NODATA).sum())
                for ci, cname in enumerate(classes):
                    n = int((vals == ci).sum())
                    long_rows.append({'aoi_idx': idx, **ids, 'zone': zone, 'period': str(period),
                                      'class': cname, 'pixels': n, 'area_rai': round(n * px_rai, 2),
                                      'pct': round(100 * n / n_valid, 2) if n_valid else np.nan})
            vals = maj[inside]; n_valid = int((vals != NODATA).sum())
            rec = {'aoi_idx': idx, **ids, 'zone': zone,
                   'polygon_rai': round(row.geometry.area / M2_PER_RAI, 2),
                   'pixels_total': int(inside.sum()), 'pixels_predicted': n_valid}
            for ci, cname in enumerate(classes):
                rec[f'{cname}_rai'] = round(int((vals == ci).sum()) * px_rai, 2)
                rec[f'{cname}_pct'] = round(100 * (vals == ci).sum() / n_valid, 2) if n_valid else np.nan
            pcts = [rec[f'{c}_pct'] for c in classes]
            rec['dominant_class'] = classes[int(np.nanargmax(pcts))] if n_valid else None
            maj_rows.append(rec)

    long_df = pd.DataFrame(long_rows)
    maj_df = pd.DataFrame(maj_rows)
    return long_df, maj_df


if __name__ == '__main__':
    with open(os.path.join(REPO_ROOT, 'config', 'config.yaml')) as f:
        full_cfg = yaml.safe_load(f)
    cfg, cfg_aoi = full_cfg['prediction'], full_cfg['aoi']

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--aoi', default=cfg_aoi['path'], help='AOI vector file (default from config)')
    ap.add_argument('--force', action='store_true', help='Rerun prediction even if outputs exist')
    ap.add_argument('--skip-predict', action='store_true', help='Only summarise existing predictions')
    ap.add_argument('--full-zone', action='store_true',
                    help='Predict whole zones into prediction.output_dir (slow) instead of AOI pixels only')
    args = ap.parse_args()

    aoi_path = repo_path(args.aoi)
    stem = os.path.splitext(os.path.basename(aoi_path))[0]
    aoi_out = os.path.join(repo_path(cfg_aoi['output_dir']), stem)
    pred_dir = repo_path(cfg['output_dir']) if args.full_zone else os.path.join(aoi_out, 'prediction')
    os.makedirs(aoi_out, exist_ok=True)

    aoi = gpd.read_file(aoi_path)
    aoi['geometry'] = aoi.geometry.force_2d()
    footprints = zone_footprints(repo_path(cfg['datacube_dir']))
    aoi['zone'] = assign_zones(aoi, footprints)
    outside = aoi['zone'].isna()
    print(f'AOI: {aoi_path} ({len(aoi)} polygons)')
    print(aoi['zone'].value_counts(dropna=False).to_string())
    if outside.any():
        print(f'WARNING: {outside.sum()} polygons are outside every datacube zone and will be skipped')
    zones = sorted(aoi.loc[~outside, 'zone'].unique())

    if not args.skip_predict:
        if args.full_zone:
            run_prediction_full_zone(zones, cfg, pred_dir, args.force)
        else:
            run_prediction_aoi(aoi[~outside], zones, cfg, pred_dir, args.force)

    classes = class_names(cfg)
    long_df, maj_df = summarise(aoi[~outside], cfg, cfg_aoi, pred_dir, aoi_out, classes)
    long_df.to_csv(os.path.join(aoi_out, 'summary_long.csv'), index=False, encoding='utf-8-sig')
    maj_df.to_csv(os.path.join(aoi_out, 'summary_majority.csv'), index=False, encoding='utf-8-sig')

    gdf = aoi.join(maj_df.set_index('aoi_idx').drop(columns=[c for c in maj_df.columns if c in aoi.columns]))
    gdf.to_file(os.path.join(aoi_out, f'{stem}_ftype.geojson'), driver='GeoJSON')

    print(f'\nClasses: {classes}')
    print(maj_df['dominant_class'].value_counts(dropna=False).to_string())
    print(f'\nOutputs written to {aoi_out}')
