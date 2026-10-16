# รายละเอียดแบบจำลอง (Model Versions) - ภาษาไทย

## 📋 สรุปเปรียบเทียบทุกเวอร์ชัน

| เวอร์ชัน | สถาปัตยกรรม | คุณลักษณะอินพุต | จำนวนคลาส | สถานะ |
|---------|-----------|---------|----------|------|
| **v1.0.0** | LSTM-FCN | 4 ช่วงเวลา, ไม่มีแบบ static | ? | หยุดแล้ว (เก่า) |
| **v2.0.0** | Hybrid LSTM-CNN (3 input) | 60 biophys + 4 indices + 2 static | 5 | ทดลอง (พบปัญหา drift) |
| **v2.1.0** | Hybrid LSTM-CNN (3 input) | เหมือน v2.0.0 | 5 | ทดลอง (focal loss) |
| **v2.1.1** | Hybrid LSTM-CNN (3 input) | เหมือน v2.0.0 | 4 | ทดลอง (dataset เล็ก) |
| **v2.2.0** | Hybrid CNN (2 input) | 21 timesteps รวม (9-10 ch) | 3-5 | ใช้ได้ (spatial validation) |
| **v2.2.1** | Hybrid CNN (2 input) | 21 timesteps รวม (9-10 ch) | 5 | ✅ **ปัจจุบัน** (auto-detect, label encoder) |

---

## v1.0.0 — ยุคแรก (Basic LSTM-FCN)

### ลักษณะเด่น
- **สถาปัตยกรรม**: LSTM-FCN (Long Short-Term Memory + Fully Convolutional Network)
- **อินพุต**: ช่วงเวลา 4 เดือน × 4 ช่วงเวลา (ไม่มี static features)
- **ช่องข้อมูล**: NDVI, GVMI, SLAVI, S1 VV/VH ratio
- **โอเวอร์เฮด**: ไม่มีการ scale, ไม่มี biophysical data

### ปัญหา
- ช่วงเวลาน้อยเกินไป (4 เดือน)
- ไม่มีข้อมูล topographic (elevation/slope)
- ไม่มีความน่าจะเป็นของคำทำนาย (prediction probability)

---

## v2.0.0 — ก้าวกระโดด (Hybrid Multi-Input)

### สิ่งที่เปลี่ยน
- **สถาปัตยกรรม**: 3 input streams (biophys + indices + static)
- **อินพุต 1 - Biophysical**: 60 timesteps × 3 ช่อง (LAI, FAPAR, FCOVER)
  - เดิม 105 timesteps → ตัด `[:, 31:-15, :]` → เหลือ 60
- **อินพุต 2 - Indices**: 4 timesteps × 6 ช่อง (VV, VH, GVMI, NDVI, SLAVI, VV/VH ratio)
- **อินพุต 3 - Static**: elevation, slope

### ปัญหา
- **Scaling parameters ไม่เก็บ** → ความเสี่ยง train/inference drift
- ไม่มี hold-out test set (ใช้ 70/30 split)
- ใช้ sparse categorical loss → ไม่ดีกับ imbalanced classes
- Hardcoded เฉพาะ zone_3

---

## v2.1.0 — โฟกัสความมั่นคง (Focal Loss + Hold-out)

### สิ่งที่เปลี่ยน
- **Loss**: sparse categorical → **CategoricalFocalCrossentropy** (γ=2.0)
- **Model size**: 126,789 → **70,309 parameters** (-45%)
- **Validation**: เพิ่ม **stratified split** + **explicit hold-out test set** (102,489 samples)
- **Callbacks**: เพิ่ม `DistributionMonitor` (ดูการกระจายคลาส)
- **Dataset**: เหมือน v2.0.0 (341,630 samples)

### ปัญหา
- Scaling parameters ยังไม่เก็บ
- คลาส `delh` ยังไม่มั่นคง

---

## v2.1.1 — คุณภาพข้อมูล (Clean Dataset + Persist Scaling)

### สิ่งที่เปลี่ยน
- **Dataset**: 341,630 → **6,836 samples** (↓ 50×, แต่คุณภาพดี)
  - อัปเกรด เวอร์ชัน `1.1.0` → `1.1.1` (hand-curated)
- **Classes**: 5 → **4** (ลบ `delh`)
- **Epochs**: 60 → **120** (เพราะ dataset เล็ก converge ช้า)
- **Scaling params**: ✅ **เก็บใน `.npz`** (ไม่มี train/inference drift)

### ประโยชน์
- Scaling params อยู่ในไฟล์ `scaling_params_from_training.npz`
- ลดปัญหา train/inference drift

---

## v2.2.0 — ปรับโครงสร้างใหญ่ (Combined Timeseries + Spatial Validation)

### สิ่งที่เปลี่ยน
- **สถาปัตยกรรม**: 3 streams → **2 inputs** (รวม timeseries)
- **Timeseries**: 21 timesteps = **4 ISO weeks ก่อนปี + 17 ISO weeks ปีเป้าหมาย**
- **Channels**: รวม 9 ช่อง (VV, VH, GVMI, NDVI, SLAVI, LAI, FAPAR, FCOVER, ratios)
- **Outlier filtering**: IQR method (threshold=1.5) ✅ **ใหม่**
- **Validation**: เพิ่ม **spatial validation** (leakage-robust)
- **Classes**: 4 → **3** (ลบ `MIX`) + โปรแกรมอนุมาน 5 classes
- **Scaling `.npz` keys**: อัปเกรด (`biophys_*` → `timeseries_*`, `indices_*` → รวมอยู่ใน `timeseries_*`)

### ข้อมูล Input
```
21 timesteps × 10 channels:
  [VV, VH, GVMI, NDVI, SLAVI, LAI, FAPAR, FCOVER, VH/VV, VV/VH]
  + 2 static: [slope, elevation]
```

---

## v2.2.1 — ปรับแต่งสุดท้าย (Auto-detection + Label Encoder)

### สิ่งที่เปลี่ยน
- **Label encoder**: ✅ **ใหม่** เพิ่มเข้า → ได้ชื่อคลาส (DDF, DEF, MDF ฯลฯ) ไม่ใช่ index
- **Auto-detection**: ตรวจหา model/scaling/label encoder อัตโนมัติตามเวอร์ชัน
- **Fallback chain**: ยืดหยุ่นสำหรับชื่อไฟล์ที่ต่างกัน
- **Channel order enforcement**: ✅ **ยาก** (v2.2.0 บางครั้งผิด)
- **Static feature order**: แก้ไข (v2.0.0 กลับด้าน elevation/slope)

### สิ่งที่ยังเหมือน
- สถาปัตยกรรมเดียวกัน v2.2.0
- 21 timesteps, 10 channels

### Multi-version support
- รันได้ v2.2.0, v2.2.1, v2.2.2 เพราะตรวจหาเวอร์ชันอัตโนมัติ

---

## 📊 ตารางเปรียบเทียบคุณลักษณะหลัก

### อินพุท / ข้อมูล

| ลักษณะ | v1.0.0 | v2.0.0/2.1.0/2.1.1 | v2.2.0/2.2.1 |
|--------|--------|----------|-----------|
| **Biophysical TS** | ❌ | 60 timesteps × 3 | 21 timesteps × 3 |
| **Indices TS** | 4 months | 4 timesteps × 6 | 21 timesteps × 5-6 |
| **Static** | ❌ | ✓ (elevation, slope) | ✓ (slope, elevation) |
| **Total channels** | 4 | biophys(3) + indices(6) + static(2) | 10 (timeseries + 2 ratios) |

### Loss / Optimization

| ลักษณะ | v1.0.0 | v2.0.0 | v2.1.0+ | v2.2.0+ |
|--------|--------|---------|---------|---------|
| **Loss** | ❓ | Sparse categorical | **Focal (γ=2)** | **Focal (γ=2)** |
| **Class weights** | ❌ | ✓ balanced | ✓ balanced + Focal | ✓ balanced + Focal |
| **Hold-out test** | ❌ | ❌ | ✓ 30% | ✓ spatial validation |

### ข้อมูล / คุณภาพ

| ลักษณะ | v1.0.0 | v2.0.0 | v2.1.0 | v2.1.1 | v2.2.0+ |
|--------|--------|---------|---------|---------|---------|
| **Dataset version** | - | 1.1.0 | 1.1.0 | **1.1.1** | 1.1.1 (post-filter) |
| **Samples** | - | 341,630 | 341,630 | **6,836** | 6,836 |
| **Outlier filter** | ❌ | ❌ | ❌ | ❌ | **IQR (thr=1.5)** |
| **Scaling persisted** | ❌ | ❌ | ❌ | ✓ | ✓ |
| **Spatial validation** | - | - | - | - | ✓ |

### Inference / Output

| ลักษณะ | v1.0.0 | v2.0.0 | v2.1.0/2.1.1 | v2.2.0 | v2.2.1 |
|--------|--------|---------|---------|---------|---------|
| **Classes** | ? | 5 | 5/4 | 3/5 | 5 |
| **Output class** | index | index | index | index | **name (via encoder)** |
| **Probability output** | ❌ | ✓ | ✓ | ✓ | ✓ |
| **Majority vote** | ✓ | commented | - | ✓ | ✓ |
| **Auto-detect paths** | - | - | - | ❌ | ✅ |

---

## 🔑 ประเด็นสำคัญต่อเวอร์ชัน

### v1.0.0 → v2.0.0
- **เหตุ**: Sentinel-2 spectral + biophysical เพิ่มเติม
- **ผล**: ช่องข้อมูลเพิ่มขึ้น 4 → 11, model เพิ่มซับซ้อน

### v2.0.0 → v2.1.0
- **เหตุ**: Class imbalance รุนแรง (minority ~0.3%)
- **ผล**: Focal loss + stratified split + hold-out test

### v2.1.0 → v2.1.1
- **เหตุ**: Dataset ใหญ่เกินไป (341K samples → overfit)
- **ผล**: Hand-curated dataset (6.8K), scaling persisted

### v2.1.1 → v2.2.0
- **เหตุ**: Time series separate หลายส่วน → ซับซ้อน, 105 timesteps ≠ 21 weeks
- **ผล**: Unified 21-week timeseries, spatial validation, outlier filtering

### v2.2.0 → v2.2.1
- **เหตุ**: Channel order ผิดบ่อยๆ → 99.9% single class
- **ผล**: Label encoder + auto-detect + strict channel ordering

---

## ⚡ โฟลว์การใช้งาน

### v1.0.0
```
Load 4 monthly rasters (NDVI, GVMI, SLAVI, S1 ratio)
  ↓
Reshape → (H*W, 4, 4)
  ↓
model.predict() → argmax → class index
```

### v2.0.0/2.1.0/2.1.1
```
Load biophys (105 timesteps) → slice [31:-15]
Load indices (4 months, 5 ch) → add ratio
Load elevation + slope
  ↓
[biophys (60,3), indices (4,6), static (2,)]
  ↓
3-stream model → concatenate → Dense → Softmax
  ↓
Probabilities + argmax → class index
```

### v2.2.0/v2.2.1
```
Load 21 datacube rasters per feature (VV, VH, GVMI, NDVI, SLAVI, LAI, FAPAR, FCOVER)
  ↓
Union mask → valid pixels
  ↓
Stack → (n_valid, 21, 8) + add ratios → (n_valid, 21, 10)
  ↓
Z-score scale (training stats)
  ↓
model.predict([timeseries (21,10), static (2,)]) → [proba (n_valid, 5)]
  ↓
argmax → class index → [label encoder] → class name → GeoTIFF
```

---

## 📁 Path Convention

### Training Artifact
```
model/mfl_v{major}_{minor}_{patch}/
  ├── hybrid_lstm_cnn_model-{date}.keras    (model)
  └── config/
      ├── scaling_params_*.npz               (params)
      └── label_encoder_v*.joblib            (v2.2.1+)
```

### Prediction Output
```
/fs2/ftype_mflf/result/2025/1.2.0/{zone}/model_version_{ver}/
  ├── pred_{year}_{ver}-class.tif            (class map)
  ├── pred_{year}_{ver}-proba.tif            (probabilities)
  └── pred_majority_{ver}.tif                (majority vote)
```

---

## ⚠️ ข้อระวัง / Common Issues

| ปัญหา | สาเหตุ | แก้ไข |
|--------|--------|------|
| **99.9% single class** | Channel order ผิด | ตรวจสอบลำดับใน `.npy` / `.tif` |
| **Train/inference drift** | Scaling params คำนวณใหม่ | ใช้ scaling `.npz` จาก training |
| **Low accuracy on new zone** | Spatial leakage ใน v<2.2.0 | ใช้ v2.2.0+ (spatial validation) |
| **Scaling not loading** | Path hardcoded | ใช้ v2.2.1+ (auto-detect) |
| **Raw indices output** | Label encoder ไม่โหลด | ใช้ v2.2.1 (auto-detect) |

---

## 📈 Timeline

```
May 2024: v1.0.0 (LSTM-FCN, 4 months, no biophys)
Aug 2024: v2.0.0 (Hybrid, 60+4 timesteps, 3-stream, 341K samples)
Aug 2024: v2.1.0 (Focal loss, stratified, hold-out, 341K)
Aug 2024: v2.1.1 (Clean dataset 1.1.1, 6.8K samples, persist scaling)
Sep 2024: v2.2.0 (21-week unified, 2-stream, spatial validation, outlier filter)
Sep 2024: v2.2.1 (Label encoder, auto-detect, strict channel order) ← **CURRENT**
```

---

## 🎯 การเลือกเวอร์ชัน

**ใช้ v2.2.1** ในกรณี:
- ✅ การใช้งานใหม่ / Production
- ✅ ต้องการผลลัพธ์ชื่อคลาส (ไม่ใช่ index)
- ✅ ต้องการความแม่นยำสูง

**ใช้ v2.2.0** ในกรณี:
- ต้องการสถาปัตยกรรมแบบต้นฉบับ
- ยอมรับผลลัพธ์ class index (ต้องแปลงเอง)

**ไม่ใช้ v2.1.x/v2.0.0/v1.0.0**:
- ❌ Legacy, มีปัญหาด้าน stability/drift
- ❌ ไม่มี spatial validation
- ❌ Dataset เก่า / ทะยาบาย
