#!/usr/bin/env python3
"""
Compare two sets of class predictions pixel by pixel (regression check).

Each set is a directory laid out as <zone>/model_version_<v>/pred_<year>_<v>-class.tif.
Rasters may cover different extents (e.g. AOI window vs whole zone): set B is read
over set A's footprint, and only pixels predicted in A are compared.

Usage:
    python scripts/compare_predictions.py A_DIR B_DIR [--b-version-sep .] [--min-match 99.9]

Exit code 1 if any zone-year matches less than --min-match percent.
"""
import argparse
import glob
import os
import sys

import numpy as np
import rasterio
from rasterio.windows import from_bounds

NODATA = -999


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('a_dir')
    ap.add_argument('b_dir')
    ap.add_argument('--b-version-sep', default='_',
                    help="Version separator in B's file names ('.' for the original /fs2/ftype_mflf/result files)")
    ap.add_argument('--min-match', type=float, default=99.9)
    args = ap.parse_args()

    failed = 0
    for a_path in sorted(glob.glob(os.path.join(args.a_dir, '*', 'model_version_*', 'pred_*-class.tif'))):
        rel = os.path.relpath(a_path, args.a_dir)
        zone, vdir, name = rel.split(os.sep)
        year, ver = name.split('_', 2)[1], vdir.replace('model_version_', '')
        b_path = os.path.join(args.b_dir, zone, vdir,
                              f"pred_{year}_{ver.replace('.', args.b_version_sep)}-class.tif")
        if not os.path.exists(b_path):
            print(f'{zone} {year}: no file in B')
            continue

        with rasterio.open(a_path) as a_src:
            a = a_src.read(1)
            bounds = a_src.bounds
        with rasterio.open(b_path) as b_src:
            win = from_bounds(*bounds, transform=b_src.transform).round_offsets().round_lengths()
            b = b_src.read(1, window=win, boundless=True, fill_value=NODATA)

        if a.shape != b.shape:
            print(f'{zone} {year}: shape mismatch {a.shape} vs {b.shape}')
            failed += 1
            continue
        valid = a != NODATA
        n = int(valid.sum())
        match = 100 * float((a[valid] == b[valid]).mean()) if n else float('nan')
        only_b = int(((a == NODATA) & (b != NODATA)).sum())
        flag = '' if match >= args.min_match else '  <-- below threshold'
        print(f'{zone} {year}: match={match:.3f}%  pixels={n:,}  only-in-B={only_b:,}{flag}')
        failed += match < args.min_match

    sys.exit(1 if failed else 0)


if __name__ == '__main__':
    main()
