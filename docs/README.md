# Documentation

## เริ่มที่นี่

| เอกสาร | เนื้อหา |
|---|---|
| [AOI_PREDICTION.md](AOI_PREDICTION.md) | **flow ที่ใช้งานจริง:** predict ตาม AOI ด้วย Docker, config, หลักการทำงาน, output, ข้อควรระวัง |
| [models/](models/README.md) | รายละเอียดโมเดลแต่ละเวอร์ชัน (v1.0.0 – v2.2.x) |

## ที่มาของข้อมูลและการสร้าง datacube

[final_pipline/](final_pipline/README.md): GEE → biophysical → alignment → interpolation → verification

| ขั้น | เอกสาร | สคริปต์ |
|---|---|---|
| ข้อมูล GEE (S1/S2, SRTM) | [GEE_DATA_SOURCE.md](final_pipline/GEE_DATA_SOURCE.md), [GEE_DATA_PREPARATION_v1_2_0.md](GEE_DATA_PREPARATION_v1_2_0.md) | notebook (ไม่อยู่ใน git), `scripts/sync_gdrive.sh` |
| Biophysical (LAI/FAPAR/FCOVER) | [BIOPHYSICAL_DATA_SOURCE.md](final_pipline/BIOPHYSICAL_DATA_SOURCE.md) | `src/weekly_biophys_aggregation.py` |
| Alignment / clip | [DATA_PROCESSING_ALIGNMENT.md](final_pipline/DATA_PROCESSING_ALIGNMENT.md) | `src/raster_clip_align.py`, `scripts/align_biophys_datacubes.sh` |
| Interpolation | | `src/batch_interpolate.py` → `src/raster_interpolation_ultra_fast.py` (parallel: `parallel_batch_interpolation.py`, `distribute_batch_processing.py`) |
| Verification | [DATACUBE_VERIFICATION.md](final_pipline/DATACUBE_VERIFICATION.md), [VERIFICATION_QUICK_REFERENCE.md](VERIFICATION_QUICK_REFERENCE.md) | `scripts/datacube_zone_verification.py`, `scripts/verify_all_zones_sequential.sh` |

## Dataset และการ train

| เอกสาร | เนื้อหา |
|---|---|
| [final_pipline/DATASET_EXTRACTION.md](final_pipline/DATASET_EXTRACTION.md) | ดึงค่าพิกเซลที่จุดสำรวจ (`scripts/extract_forest_type_dataset.py`, `scripts/run_extraction.sh`) |
| [DATASET_BUILDER.md](DATASET_BUILDER.md) | `src/dataset_builder.py` สร้าง dataset จาก parquet |
| [DATASET_DOCUMENTATION_SUMMARY.md](DATASET_DOCUMENTATION_SUMMARY.md), [DATASET_QUERY_AND_SELECTION_GUIDE.md](DATASET_QUERY_AND_SELECTION_GUIDE.md), [data_storage_and_query.md](data_storage_and_query.md) | โครงสร้างและการ query dataset |
| [TRAINING_INTEGRATION_GUIDE.md](TRAINING_INTEGRATION_GUIDE.md), [feature_engineering_and_models.md](feature_engineering_and_models.md) | การ train (`scripts/train_model_v2_2_1.py`) |

## บันทึกการสืบสวนปัญหา

[MEMORY/](MEMORY/README.md): prediction bias (ลำดับ channel ผิด), data leakage, nodata ฯลฯ ช่วง ก.ย. 2025

## Archive

[archive/](archive/): เอกสารรุ่นเก่าที่มีฉบับใหม่กว่าอยู่ใน `final_pipline/` แล้ว รวมถึง index เดิม (`README_old_index.md`)
เก็บไว้อ้างอิงเท่านั้น เนื้อหาอาจไม่ตรงกับโค้ดปัจจุบัน

> หมายเหตุ: ขั้นสร้าง datacube, extraction และ interpolation **เขียนลงข้อมูลกลาง** (`/fs2/ftype_mflf/shared_data/...`) โดยตรง
> ใน Docker ข้อมูลกลาง mount แบบ read-only สคริปต์เหล่านี้จึงจะล้มถ้าไม่เปลี่ยนปลายทางไปที่ `outputs/` ก่อน
> ซึ่งเป็นการกันไว้โดยตั้งใจ
