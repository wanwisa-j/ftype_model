#!/bin/bash

for xxxx in 1 2 3 4 5; do
    python /fs1/tor_project/model_foresttype/src/prediction_ftype.py /fs2/ftype_mflf/shared_data/temp/chm/chm_zone_${xxxx} 2020
done