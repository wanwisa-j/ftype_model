# ftype_model: Forest Type Classification

จำแนกประเภทป่า **DDF** (ป่าเต็งรัง) / **DEF** (ป่าดิบแล้ง) / **MDF** (ป่าเบญจพรรณ)
จากข้อมูลรายสัปดาห์ของ Sentinel-1, Sentinel-2 และ biophysical (21 สัปดาห์) ร่วมกับ slope และ elevation
ด้วยโมเดล hybrid LSTM-CNN **v2.2.2** รันตาม AOI (แปลงพื้นที่) ผ่าน Docker

```
AOI (.geojson) ──► หาว่าอยู่ zone ไหน ──► predict เฉพาะพิกเซลในแปลง (2019–2025) ──► majority ข้ามปี ──► สรุปไร่/% รายแปลง
```

---

## Quick start

ต้องมี Docker, NVIDIA container runtime และสิทธิ์อ่าน datacube ที่ `/fs2/ftype_mflf/shared_data/datacube/2025/1.2.0`

```bash
git clone git@github.com:wanwisa-j/ftype_model.git
cd ftype_model

docker compose build dev                                       # image ขนาดเล็ก (~90 MB)
docker compose run --rm dev bash scripts/docker_setup_env.sh   # สร้าง env ครั้งแรก (~10 GB ที่ .venv-docker/)

docker compose run --rm dev python src/run_aoi_prediction.py   # รัน AOI ตาม config
```

คำสั่งอื่นที่ใช้บ่อย

```bash
docker compose run --rm dev python src/run_aoi_prediction.py --aoi data/aoi/<file>.geojson   # AOI อื่น
docker compose run --rm dev python src/run_aoi_prediction.py --buffer-km 1                   # ขยายแปลงออก 1 กม.
docker compose run --rm dev python src/run_aoi_prediction.py --output-dir outputs/xxx        # เปลี่ยนโฟลเดอร์ output
docker compose run --rm dev python src/run_aoi_prediction.py --skip-predict                  # สรุปใหม่จากผลที่มีอยู่
docker compose run --rm dev python src/run_aoi_prediction.py --force                         # predict ใหม่ทั้งหมด
docker compose run --rm dev bash                                                             # shell ใน container
docker compose up -d jupyter                                                                 # JupyterLab :8899
```

> env (Python 3.12, GDAL, TF 2.21 + CUDA) ไม่ได้อยู่ใน image แต่เก็บไว้ที่ `.venv-docker/` ใน repo
> เพราะดิสก์ของ Docker บนเครื่อง dev เกือบเต็ม รายละเอียดอยู่ใน [docs/AOI_PREDICTION.md](docs/AOI_PREDICTION.md#2-environment-docker)

## AOI ใหม่

1. วางไฟล์ vector ที่มี CRS (GeoJSON/GPKG/SHP) ไว้ที่ `data/aoi/` **ไม่บังคับชื่อคอลัมน์** ใช้แค่ geometry
2. แก้ `aoi.path` ใน `config/config.yaml` หรือส่งผ่าน `--aoi`
3. ถ้าอยากให้ชื่อแปลงติดไปใน output ให้ใส่ชื่อคอลัมน์ไว้ที่ `aoi.id_fields`
4. ถ้าต้องการขยายแปลงออกจากขอบ ให้ใส่ `--buffer-km N` (หรือ `aoi.buffer_km`) ค่า default คือ 0
   ระบบจะ predict และสรุปบนแปลงที่ขยายแล้ว output อยู่ที่ `<ชื่อ AOI>_buf<N>km/` (แปลงที่ขยายจนทับกัน พื้นที่ส่วนที่ทับจะถูกนับในทั้งสองแปลง)

ถ้าใช้ชื่อไฟล์ AOI เดิมแต่แก้ geometry ต้องใส่ `--force` ไม่อย่างนั้นจะใช้ผล predict ของเดิม

แปลงต้องอยู่ใน zone ที่มี datacube อยู่แล้ว คือ `zone_1`–`zone_5` (เชียงใหม่) และ `mfl_cmf_area_project_2025_chr_zone_1/2` (เชียงราย)
แปลงที่อยู่นอก zone จะขึ้น WARNING แล้วถูกข้ามไป

## Output

`outputs/aoi/<ชื่อไฟล์ AOI>/`

| ไฟล์ | เนื้อหา |
|---|---|
| `<aoi>_ftype.geojson` | แปลงพร้อม `dominant_class`, `<class>_rai`, `<class>_pct` เปิดใน QGIS ได้เลย |
| `summary_majority.csv` | 1 แถวต่อแปลง (majority 2019–2025) |
| `summary_long.csv` | แปลง × ปี × class |
| `rasters/<zone>_majority.tif` | raster ประเภทป่าในแปลง `0=DDF 1=DEF 2=MDF`, nodata `-999` |
| `prediction/<zone>/model_version_2.2.2/pred_<year>_2_2_2-{class,proba}.tif` | ผลรายปี |

## Config

ทุก path อยู่ใน `config/config.yaml` (path แบบ relative นับจาก root ของ repo)

| key | ค่า |
|---|---|
| `prediction.model_version` / `model_file` | `2.2.2` / `hybrid_lstm_cnn_model-*-no-leakage.keras` |
| `prediction.datacube_dir` | `/fs2/ftype_mflf/shared_data/datacube/2025/1.2.0` (mount แบบ read-only) |
| `prediction.years` | 2019–2025 |
| `aoi.path`, `aoi.id_fields`, `aoi.output_dir` | ไฟล์ AOI, คอลัมน์ที่จะใส่ใน output, โฟลเดอร์ output (`--aoi`, `--output-dir`) |
| `aoi.buffer_km` | ระยะขยายแปลง (กม.), 0 = ไม่ขยาย (`--buffer-km`) |
| `paths.training_dataset`, `paths.logs_dir` | dataset สำหรับ train (`dataset_builder`), โฟลเดอร์ log |

ข้อมูลกลางทั้งหมดถูก mount แบบ **read-only** ใน container ผลทุกอย่างจะเขียนได้เฉพาะใน `outputs/` ของ repo

## Model v2.2.2

| | |
|---|---|
| ไฟล์ | `model/mfl_v2_2_2/` (มีใน git: โมเดล, scaling params, label encoder) |
| Input | time series `(21 สัปดาห์, 10 channels)`: vv, vh, gvmi, ndvi, slavi, lai, fapar, fcover, VH/VV, VV/VH + static `(slope, elevation)` |
| Output | probability 3 class → `0=DDF, 1=DEF, 2=MDF` |
| Train zones | `mfl_cmf_2024`, `chr_zone_2` (กัน `chr_zone_1` ไว้ไม่ให้ใช้ train) |

ลำดับ channel **ต้องตรงกับตอน train** ไม่อย่างนั้นผลจะเอียงไป class เดียว
รายละเอียดโมเดลแต่ละเวอร์ชันอยู่ที่ [docs/models/](docs/models/README.md)

## Validation

```bash
docker compose run --rm dev python scripts/compare_predictions.py <dir A> <dir B> [--b-version-sep .]
```

- โหมด AOI เทียบกับการ predict ทั้ง zone: ตรงกัน **100%** ทุกพิกเซลในแปลง (49 zone-ปี)
- เทียบกับผลเดิมที่ `/fs2/ftype_mflf/result/2025/1.2.0` (v2.2.2): ตรงกัน **99.98–99.99%**
  - ยกเว้น `zone_1` ปี 2023 เพราะ datacube SAR ถูกสร้างใหม่หลังจากรันผลเดิมไปแล้ว

## Known issues

- **ช่องว่างใน datacube:** biophys ของ `zone_3` ปี 2025 หายไปในสัปดาห์ที่ 13–18 ตอนนี้พิกเซลพวกนี้ถูก mask เป็น nodata
  (โค้ดเดิมทายเป็น DDF ทั้งหมด และผลใน `/fs2/ftype_mflf/result` ก็โดนแบบเดียวกัน) ถ้าต้องการให้ predict ได้ครบ ต้อง interpolate ก่อน
- **DEF ออกมาน้อยมาก** (AOI ป่าชุมชน 53 แปลง: DEF 0.3%) ควรเทียบกับข้อมูลภาคสนามก่อนนำไปใช้
- **AOI นอก 7 zone** ยังรันไม่ได้ ต้องสร้าง datacube ใหม่ก่อน (GEE export → biophys → datacube) ขั้นตอนนี้ยังเป็น notebook และสคริปต์ที่ฝังชื่อ zone ไว้
- **ขั้นสร้าง datacube / extraction / interpolation** ยังเขียนผลลงข้อมูลกลางโดยตรง ใน Docker จะล้มเพราะข้อมูลกลาง mount แบบ read-only (ตั้งใจ)

## Repo structure

```
├── src/
│   ├── run_aoi_prediction.py           ★ entry point: AOI → predict → summary
│   ├── prediction_ftype_2_2_1.py       ★ โหลด datacube, mask, scaling, predict (v2.2.x)
│   ├── dataset_builder.py                สร้าง dataset สำหรับ train
│   ├── weekly_biophys_aggregation.py     datacube: biophys รายสัปดาห์
│   ├── raster_clip_align.py              datacube: clip/align
│   └── batch_interpolate.py, raster_interpolation_ultra_fast.py, ...   datacube: เติมช่องว่าง
├── scripts/
│   ├── docker_setup_env.sh             ★ สร้าง env
│   ├── compare_predictions.py          ★ regression check
│   ├── train_model_v2_2_1.py             train (เซฟโมเดลลง model/)
│   ├── extract_forest_type_dataset.py    ดึงค่าที่จุดสำรวจ (+ run_extraction.sh)
│   └── datacube_zone_verification.py     ตรวจ datacube (+ verify_all_zones_sequential.sh)
├── config/config.yaml                  ★ path และค่าตั้งค่าหลัก (config อื่นใน config/ ใช้กับ extraction)
├── model/mfl_v2_2_2/                   ★ โมเดลที่ใช้
├── docs/                                 เริ่มที่ docs/README.md
├── archive/                              สคริปต์เก่า/ใช้ครั้งเดียวจาก repo ต้นทาง (เก็บไว้อ้างอิง ไม่ได้ดูแลต่อ)
├── Dockerfile, docker-compose.yml, environment.yml
├── data/aoi/                             ไฟล์ AOI (ไม่อยู่ใน git)
└── outputs/                              ผลลัพธ์ (ไม่อยู่ใน git)
```

★ = ส่วนที่ใช้ในการ predict ตาม AOI
