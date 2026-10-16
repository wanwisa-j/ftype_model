# Forest Type Prediction ตาม AOI (v2.2.2)

เอกสารนี้อธิบายการทำนายประเภทป่า (DDF / DEF / MDF) ให้แปลง AOI ด้วยโมเดล v2.2.2
ครอบคลุมตั้งแต่ตั้งค่า environment, config, ขั้นตอนการทำงานของโค้ด, output ไปจนถึงข้อควรระวัง

> ที่มาของข้อมูลและการสร้าง datacube ดูใน [docs/final_pipline/](final_pipline/README.md)
> เอกสารนี้เริ่มจาก datacube ที่มีอยู่แล้ว

---

## 1. ภาพรวม

```
data/aoi/<aoi>.geojson
        │
        ▼
[1] หาว่าแต่ละแปลงอยู่ zone ไหน        (เทียบกับขอบเขต datacube ของแต่ละ zone)
        │
        ▼
[2] predict เฉพาะพิกเซลในแปลง × ทุกปี   functions ใน src/prediction_ftype_2_2_1.py
        │   input : datacube/<zone>/{vv,...,elevation} อ่านเฉพาะ window ที่ครอบแปลง
        │   output: outputs/aoi/<aoi>/prediction/<zone>/model_version_2.2.2/pred_<year>_2_2_2-{class,proba}.tif
        ▼
[3] clip ตามแปลง + majority ข้ามปี      src/run_aoi_prediction.py
        │
        ▼
outputs/aoi/<aoi>/  (CSV, GeoJSON, raster)
```

ทั้งหมดรันได้ด้วยคำสั่งเดียว:

```bash
docker compose run --rm dev python src/run_aoi_prediction.py
```

---

## 2. Environment (Docker)

| ไฟล์ | หน้าที่ |
|---|---|
| `Dockerfile` | image `ftype-dev:wanwisa` ขนาดเล็ก (~90 MB) มีแค่ micromamba |
| `docker-compose.yml` | service `dev` และ `jupyter` (port 8899) เปิด GPU, รันเป็น uid ของเรา, mount ข้อมูลกลางแบบ read-only |
| `scripts/docker_setup_env.sh` | สร้าง env ครั้งแรก |
| `environment.yml` | รายการแพ็กเกจ (conda-forge อย่างเดียว) |

**ทำไม env ไม่อยู่ใน image:** ดิสก์ `/` ของ Docker บนเครื่องนี้เกือบเต็ม
env (~10 GB: Python 3.12, GDAL, TF 2.21 + CUDA) จึงเก็บไว้ที่ `.venv-docker/` ใน repo บน `/fs2`
ส่วน cache และ tmp ทั้งหมดอยู่ที่ `.cache/` ทั้งสองโฟลเดอร์อยู่ใน `.gitignore`

```bash
docker compose build dev                                       # ครั้งแรก / เมื่อแก้ Dockerfile
docker compose run --rm dev bash scripts/docker_setup_env.sh   # สร้าง env (ครั้งแรก)
docker compose run --rm dev bash                               # เข้า shell
docker compose up -d jupyter                                   # JupyterLab ที่ http://localhost:8899
```

**`LD_LIBRARY_PATH` ใน Dockerfile จำเป็น ห้ามลบ** มันแก้ปัญหาสองอย่าง:
- TF ไปโหลด `libstdc++` เก่าของระบบ ทำให้ import rasterio/PROJ ไม่ได้ (`CXXABI_1.3.15 not found`)
- TF หา CUDA lib จาก pip ไม่เจอ (เช่น `libcusolver.so.11`) จึงมองไม่เห็น GPU

**Mount แบบ read-only** ทุก path ข้อมูลกลางจะเขียนทับไม่ได้ ผลทั้งหมดต้องลงที่ `outputs/` เท่านั้น:
`/fs2/ftype_mflf/shared_data`, `/fs2/ftype_mflf/result`, `/fs1/tor_project/model_foresttype`, `/fs3/datacenter/ftype_mflf`

---

## 3. Config (`config/config.yaml`)

path แบบ relative จะนับจาก root ของ repo

```yaml
prediction:
  model_version: "2.2.2"
  model_root: "model"                    # -> model/mfl_v2_2_2/
  datacube_dir: "/fs2/ftype_mflf/shared_data/datacube/2025/1.2.0"
  output_dir: "outputs/prediction/1.2.0"
  years: [2019, 2020, 2021, 2022, 2023, 2024, 2025]
  batch_size: 128

aoi:
  path: "data/aoi/ftype_lu_addarea_dissolved_cmf_moo.geojson"
  id_fields: ["CMF_Name", "Moo", "Tambon", "Amphoe", "Province"]
  output_dir: "outputs/aoi"
```

**เพิ่ม AOI ใหม่:** วางไฟล์ vector (GeoJSON/GPKG/SHP มี CRS) ไว้ที่ `data/aoi/`
แล้วแก้ `aoi.path` หรือส่งเป็น argument: `--aoi data/aoi/<file>.geojson`
ถ้าชื่อ field ต่างจากเดิม ให้แก้ `id_fields` ด้วย (field ที่ไม่มีในไฟล์จะถูกข้ามไป)

ส่วน `model:` / `retroactive_years` / `cal_majority` ด้านบนเป็นของ `src/prediction_ftype.py` (v1.0.0) ไม่เกี่ยวกับ flow นี้

### Model artifacts (`model/mfl_v2_2_2/`)

| ไฟล์ | ใช้ทำอะไร |
|---|---|
| `hybrid_lstm_cnn_model-20250910-no-leakage.keras` | โมเดล (เลือกอัตโนมัติ: ไฟล์ `.keras`/`.h5` ที่ชื่อเรียงท้ายสุด) |
| `config/scaling_params_v2_2_2_no_leakage-20250910.npz` | mean/std สำหรับ z-score (ต้องเป็นชุดเดียวกับตอน train) |
| `config/label_encoder_v2_2_2.joblib` | แปลง index เป็นชื่อ class: `0=DDF, 1=DEF, 2=MDF` |

copy มาจาก `/fs1/tor_project/model_foresttype/model/mfl_v2_2_2` แล้ว ไม่ได้อ่านจากโฟลเดอร์ของพี่ตรงๆ

---

## 4. การทำงานของ `src/prediction_ftype_2_2_1.py`

รันทีละ zone × ปี (`process_zone_year`):

1. **โหลด raster** (`load_raster_data`) จาก `datacube_dir/<zone>/<feature>/`
   - time series 21 band (สัปดาห์ W4before → W16after)
     - `cube_s1_{vv,vh}_<year>_W4before_to_W16after_21bands.tif`
     - `cube_s2_{gvmi,ndvi,slavi}_<year>_...tif`
     - `cube_biophys_{lai,fapar,fcover}_<year>_...tif`
   - static: `slope/slope_data.tif`, `elevation/elevation_data.tif` (ใช้ทุกปีเหมือนกัน)
   - ค่า nodata ของไฟล์จะถูกแปลงเป็น NaN
2. **Mask** (`create_union_mask`): ตัดพิกเซลที่ feature ใดก็ตามเป็น NaN **แม้แค่สัปดาห์เดียว** หรือเป็น -9999 ครบทุก band
   (โค้ดเดิมตัดเฉพาะที่ NaN ทุกสัปดาห์ NaN จึงหลุดเข้าโมเดลแล้วถูกทายเป็น DDF ทั้งหมด ดูข้อ 7)
   ถ้ารันโหมด AOI จะ mask พิกเซลนอกแปลงทิ้งด้วย (`extra_mask`)
3. **Stack** (`reshape_and_stack_features`) ได้ `(N, 21, 8)` + `(N, 2)`
   ลำดับ channel **ต้องตรงกับตอน train** ไม่อย่างนั้นผลจะเอียงไป class เดียว (ดู `docs/MEMORY/Prediction_Pipeline_Mismatch_Issue.md`)
   - time series: `vv, vh, gvmi, ndvi, slavi, lai, fapar, fcover`
   - static: `slope, elevation`
4. **เพิ่ม ratio** (`prepare_prediction_data`): `VH/VV` และ `VV/VH` (ถ้าตัวหารเป็น 0 ใช้ 1e-6 แทน) ได้ `(N, 21, 10)`
5. **Scaling**: z-score ด้วย mean/std จาก `.npz` ของตอน train
6. **Predict**: `model.predict([X_ts, X_static])` ได้ probability 3 class แล้วใช้ `argmax` เป็น class index
7. **บันทึก** ไปที่ `<output_dir>/<zone>/model_version_2.2.2/`
   - `pred_<year>_2_2_2-class.tif`: int16, ค่า `0/1/2`, nodata `-999`
   - `pred_<year>_2_2_2-proba.tif`: float32, 3 band (DDF, DEF, MDF), nodata `-999`

argument ทั้งหมด (ค่า default มาจาก config): `--model_version --data_dir --zones --years --output_dir --batch_size --majority_vote`
ถ้ามี zone-ปี ไหนพัง จะรัน zone-ปี อื่นต่อจนจบ แล้วสรุปรายการที่พังท้ายรันและ exit 1
ไฟล์โมเดลเลือกจาก `prediction.model_file` (glob) ใน config

```bash
# รันเฉพาะ zone/ปี ที่ต้องการ
docker compose run --rm dev python src/prediction_ftype_2_2_1.py --zones zone_1 --years 2024,2025
```

---

## 5. การทำงานของ `src/run_aoi_prediction.py`

1. **อ่าน AOI** แล้วตัดค่า Z ทิ้ง (`force_2d`)
2. **จับคู่แปลงกับ zone**: อ่านขอบเขตของ `slope/*.tif` ทุก zone (ข้าม `*_backup_*`)
   แต่ละแปลงจะได้ zone ที่ทับซ้อนกันมากที่สุด ถ้าแปลงไม่อยู่ใน zone ไหนเลยจะขึ้น WARNING แล้วข้ามไป
3. **Predict (ค่า default: โหมด AOI)** โหลดโมเดลครั้งเดียว แล้วต่อ zone: อ่านเฉพาะ window ที่ครอบแปลงของ zone นั้น
   และ predict เฉพาะพิกเซลที่อยู่ในแปลง ผลเก็บที่ `outputs/aoi/<aoi>/prediction/`
   ข้าม zone-ปี ที่มีไฟล์อยู่แล้ว (ใช้ `--force` ถ้าต้องการรันใหม่) ใช้ `--skip-predict` ถ้าต้องการสรุปจากผลที่มีอยู่เท่านั้น
   `--full-zone` = แบบเดิม predict ทั้ง zone ลง `prediction.output_dir` (ช้า แต่ AOI อื่นใน zone เดียวกันใช้ผลซ้ำได้)
4. **Majority ข้ามปี** แบบ vectorized อ่านเฉพาะ window ที่ครอบ AOI ใน zone นั้น
   - ทุกพิกเซล นับว่าแต่ละ class ชนะกี่ปี (ไม่นับ nodata) แล้วเลือก class ที่ชนะบ่อยที่สุด
   - ถ้าเสมอกันจะเลือก class ที่ index น้อยกว่า (เหมือน `scipy.stats.mode` ที่สคริปต์เดิมใช้)
   - ไม่ได้ใช้ `--majority_vote` ของสคริปต์เดิม เพราะมันวนทีละพิกเซลทั้ง zone ซึ่งช้ามาก
5. **สรุปรายแปลง**: นับพิกเซลแต่ละ class ในแปลง แล้วคูณด้วยพื้นที่ของพิกเซลจาก transform (1 ไร่ = 1,600 m²)

### Output (`outputs/aoi/<aoi stem>/`)

| ไฟล์ | เนื้อหา |
|---|---|
| `summary_majority.csv` | 1 แถวต่อแปลง: `<class>_rai`, `<class>_pct`, `dominant_class`, `polygon_rai`, `pixels_total`, `pixels_predicted` |
| `summary_long.csv` | แปลง × period (แต่ละปี + `majority`) × class: `pixels`, `area_rai`, `pct` |
| `<aoi stem>_ftype.geojson` | geometry เดิม + คอลัมน์จาก `summary_majority` (เปิดใน QGIS ได้เลย) |
| `rasters/<zone>_majority.tif` | majority class เฉพาะพิกเซลใน AOI (นอก AOI เป็น -999) |

CSV ใช้ encoding `utf-8-sig` เพื่อให้เปิดภาษาไทยใน Excel ได้
`pct` คำนวณจากพิกเซลที่ predict ได้เท่านั้น ส่วน `polygon_rai` เป็นพื้นที่จาก geometry
ถ้า `pixels_predicted` น้อยกว่า `pixels_total` มาก แปลว่าแปลงนั้นมี nodata เยอะ

---

## 6. AOI ปัจจุบัน: `ftype_lu_addarea_dissolved_cmf_moo.geojson`

- 53 แปลงป่าชุมชน รวม ~32,000 ไร่ (เชียงใหม่ + เชียงราย)
- ทุกแปลงอยู่ใน 7 zone ที่มี datacube ปี 2019-2025 ครบ จึงไม่ต้องดึง GEE ใหม่
  - เชียงใหม่: `zone_1`-`zone_5`
  - เชียงราย: `mfl_cmf_area_project_2025_chr_zone_1`, `mfl_cmf_area_project_2025_chr_zone_2`
- เวลา (RTX 3080, 49 zone-ปี): โหมด `--full-zone` ~1 ชม. 45 นาที, **โหมด AOI 21 นาที** (predict ~517k พิกเซล)
  ผลโหมด AOI ตรงกับโหมดทั้ง zone **100%** ทุกพิกเซลในแปลง (`scripts/compare_predictions.py`)
  ส่วนที่ยังกินเวลาคือการอ่าน raster เพราะแปลงเชียงรายกระจายเกือบเต็ม zone window จึงใหญ่เกือบเท่า zone

**ตรวจผลเทียบกับของเดิม:** ผล `chr_zone_1` ปี 2021-2025 เทียบกับ `/fs2/ftype_mflf/result/2025/1.2.0/.../model_version_2.2.2/`
มีพิกเซลที่ได้ class เดียวกัน **99.99%** (~4.35 ล้านพิกเซลต่อปี)
ส่วนที่ต่างราว 0.01% น่าจะมาจาก TF/CUDA คนละเวอร์ชัน แต่ยังไม่ได้ตรวจยืนยัน
หมายเหตุ: ไฟล์ของเดิมตั้งชื่อเป็น `pred_<year>_2.2.2-class.tif` (ใช้จุด) และมีแค่ปี 2021-2025

---

## 7. ข้อควรระวัง

- **Zone ที่ใช้ train:** ใน `scaling_params` ระบุ `training_zones` ไว้ว่ามี `chr_zone_2` (และ `mfl_cmf_2024`) ส่วน `chr_zone_1` เป็น zone ที่กันไว้ไม่ให้ใช้ train
  แปลงใน `chr_zone_2` จึงอาจมีจุดสำรวจที่ใช้ train อยู่ด้วย ความแม่นยำที่ได้ตรงนั้นจะดูดีเกินจริง
- **NaN ใน datacube (แก้แล้ว):** biophys (lai/fapar/fcover) ของ `zone_3` ปี 2025 หายในสัปดาห์ที่ 13-18
  โค้ดเดิมส่ง NaN เข้าโมเดล ทำให้ **28,462 พิกเซลในแปลง (31% ของ zone_3) ถูกทายเป็น DDF ทั้งหมด**
  (ปี 2024 พิกเซลเดียวกันเป็น DDF 57% / MDF 43%) ผลใน `/fs2/ftype_mflf/result` ก็ผิดแบบเดียวกัน
  ตอนนี้พิกเซลพวกนี้เป็น nodata ในปีนั้น ส่วน majority ใช้ปีที่เหลือ (ยอดรวมตาม majority เปลี่ยนแค่ ~1 ไร่)
  ถ้าต้องการให้ predict ได้ครบ ต้อง interpolate ช่องว่างใน datacube ก่อน
- **Static mask:** เงื่อนไข `slope == 0` ใน `create_union_mask` ไม่ทำงานจริง เพราะ static ถูกอ่านเป็น 3D `(1, H, W)` จึงเข้าเงื่อนไขแบบ time series แทน
- **Predict ทั้ง zone:** ถึงแปลง AOI จะเล็ก ก็ยังเสียเวลาเท่ากับรันทั้ง zone ถ้าต้องการให้เร็วขึ้น ควรเพิ่มตัวเลือกให้ crop ตามกรอบ AOI ก่อน predict
- **Driver 545 (CUDA 12.3) + CUDA 12.9 wheels:** ตอนนี้ใช้ได้ แต่ถ้าอัปเกรด TF แล้ว GPU หาย ให้เช็คเรื่องนี้ก่อน

---

## 8. สิ่งที่เปลี่ยนจาก repo ต้นทาง (`/fs2/thanakit/ftype_model_development`)

- เริ่ม git history ใหม่ (repo: `github.com/wanwisa-j/ftype_model`) ไม่ผูกกับ repo ต้นทาง
- `src/prediction_ftype_2_2_1.py`: path โมเดล, datacube, output, ปี และ batch size อ่านจาก config, import `dataset_builder` จาก `src/` ของ repo นี้ (เดิมใช้ `/fs1/tor_project/...`)
- `environment.yml`: แก้ `cftime=1.6.4.post1` เป็น `1.6.4` และเอา channel `defaults` ออก
- เพิ่ม: `Dockerfile`, `docker-compose.yml`, `.dockerignore`, `scripts/docker_setup_env.sh`, `src/run_aoi_prediction.py`, section `prediction`/`aoi` ใน config
- ไม่ได้ copy venv `tf/` ของเดิมมา เพราะมี path ของพี่ฝังอยู่และใช้กับเครื่องนี้ไม่ได้
