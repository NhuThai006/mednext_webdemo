# 🧠 Vibe Code Prompt — MedNeXt Backend API + Frontend Integration + Visualization

## 🎯 Mục tiêu dự án

Xây dựng **Python FastAPI backend** để làm trung gian giữa **Next.js 16 frontend** (đã có sẵn, đang chạy ổn định) và **model MedNeXt (nnUNet-based)** phục vụ bài toán **phân đoạn tạng bụng từ ảnh CT dạng NIfTI** (`.nii` / `.nii.gz`).

Dataset: **BTCV (Beyond The Cranial Vault)** — **14 classes** (background + 13 organs).

Ngoài inference, backend còn phải **render visualization**: overlay màu segmentation lên slice CT, kèm legend tên + màu từng tạng — giống hình mẫu `[CT Input | Prediction]`.

---

## 📁 Cấu trúc dự án hiện tại

```
MedNeXt-webdemo/
├── frontend/                        ← Next.js 16 App Router (đã hoạt động)
│   ├── app/
│   │   ├── page.tsx                 ← UI upload NIfTI, hiển thị danh sách file
│   │   └── layout.tsx
│   └── package.json
│
└── model/                           ← Model MedNeXt (nnUNet backbone)
    ├── nnunet_mednext/
    │   ├── inference/
    │   │   ├── predict.py           ← hàm predict_from_folder()
    │   │   └── predict_simple.py    ← CLI entry point
    │   └── paths.py
    ├── data/
    │   └── nnUNet_trained_models/
    │       └── nnUNet/
    │           └── 3d_fullres/
    │               └── Task017_AbdominalOrganSegmentation/
    │                   └── nnUNetTrainerV2_MedNeXt_S_kernel3__nnUNetPlansv2.1_trgSp_1x1x1/
    │                       ├── fold_0/   ← CHỈ DÙNG CÁI NÀY (model_best.model)
    │                       └── fold_1..4/ ← bỏ qua hết
    └── setup.py
```

---

## 📊 Label Map — BTCV Dataset (14 classes)

> **Dataset: BTCV (Beyond The Cranial Vault)** — 14 classes: label 0 (background) + label 1–13 (13 organs).

| Label ID | Organ | Màu RGB | Màu hex |
|----------|-------|---------|---------|
| 0 | Background | (0, 0, 0) | `#000000` |
| 1 | Spleen | (76, 175, 80) | `#4CAF50` — Xanh lá |
| 2 | Right Kidney | (126, 184, 247) | `#7EB8F7` — Xanh dương nhạt |
| 3 | Left Kidney | (91, 140, 255) | `#5B8CFF` — Xanh dương đậm |
| 4 | Gallbladder | (255, 140, 66) | `#FF8C42` — Cam |
| 5 | Esophagus | (255, 107, 157) | `#FF6B9D` — Hồng |
| 6 | Liver | (232, 212, 77) | `#E8D44D` — Vàng |
| 7 | Stomach | (30, 144, 255) | `#1E90FF` — Xanh biển |
| 8 | Aorta | (255, 68, 68) | `#FF4444` — Đỏ tươi |
| 9 | Inferior Vena Cava (IVC) | (255, 159, 67) | `#FF9F43` — Cam đậm |
| 10 | Portal & Splenic Veins | (162, 155, 254) | `#A29BFE` — Tím nhạt |
| 11 | Pancreas | (181, 123, 255) | `#B57BFF` — Tím |
| 12 | Right Adrenal Gland | (0, 206, 201) | `#00CEC9` — Xanh ngọc |
| 13 | Left Adrenal Gland | (85, 239, 196) | `#55EFC4` — Xanh mint |

---

## 🏗️ Yêu cầu xây dựng Backend

### Tech Stack
- **Python 3.10**
- **FastAPI** + **Uvicorn**
- **python-multipart** (upload file)
- **nibabel** (đọc NIfTI)
- **numpy** (xử lý array)
- **Pillow (PIL)** (vẽ visualization → PNG)
- Tích hợp trực tiếp với code nnUNet trong thư mục `model/`

---

### 📌 Cấu trúc thư mục backend cần tạo

```
MedNeXt-webdemo/
└── backend/
    ├── main.py            ← FastAPI app chính
    ├── inference.py       ← logic gọi MedNeXt model
    ├── visualize.py       ← render slice overlay PNG
    ├── config.py          ← đường dẫn model, label map BTCV
    ├── requirements.txt
    └── temp/
        ├── input/
        └── output/
```

---

### 📡 API Endpoints

#### 1. `POST /api/segment`
**Request:** `multipart/form-data`, field `file` (.nii hoặc .nii.gz)

**Response thành công (200):**
```json
{
  "status": "success",
  "job_id": "abc123",
  "download_url": "/api/result/abc123",
  "preview_url": "/api/preview/abc123",
  "total_slices": 85,
  "organs_detected": [
    { "id": 1,  "name": "Spleen",                 "color": "#4CAF50", "present": true  },
    { "id": 2,  "name": "Right Kidney",            "color": "#7EB8F7", "present": true  },
    { "id": 3,  "name": "Left Kidney",             "color": "#5B8CFF", "present": true  },
    { "id": 4,  "name": "Gallbladder",             "color": "#FF8C42", "present": false },
    { "id": 5,  "name": "Esophagus",               "color": "#FF6B9D", "present": true  },
    { "id": 6,  "name": "Liver",                   "color": "#E8D44D", "present": true  },
    { "id": 7,  "name": "Stomach",                 "color": "#1E90FF", "present": true  },
    { "id": 8,  "name": "Aorta",                   "color": "#FF4444", "present": true  },
    { "id": 9,  "name": "Inferior Vena Cava",      "color": "#FF9F43", "present": true  },
    { "id": 10, "name": "Portal & Splenic Veins",  "color": "#A29BFE", "present": false },
    { "id": 11, "name": "Pancreas",                "color": "#B57BFF", "present": true  },
    { "id": 12, "name": "Right Adrenal Gland",     "color": "#00CEC9", "present": false },
    { "id": 13, "name": "Left Adrenal Gland",      "color": "#55EFC4", "present": false }
  ],
  "processing_time_seconds": 42.5
}
```

`present: true` khi số voxel của label đó > 100 trong toàn volume.

---

#### 2. `GET /api/preview/{job_id}?slice={n}&alpha={alpha}`
Trả về **PNG** 1 slice axial: `[CT grayscale | Segmentation overlay màu]` + legend tên organ.

Query params:
- `slice` (int, default = giữa volume)
- `alpha` (float 0.0–1.0, default = 0.5)

---

#### 3. `GET /api/preview_full/{job_id}`
Trả về **PNG** tổng hợp 6 slices đại diện stacked dọc (thumbnail overview).

---

#### 4. `GET /api/result/{job_id}`
Download file `.nii.gz` segmentation thô.

---

#### 5. `GET /api/health`
```json
{ "status": "ok", "device": "cuda" }
```

---

### ⚙️ File `config.py`

```python
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_BASE_PATH = os.path.join(
    BASE_DIR, "model", "data", "nnUNet_trained_models",
    "nnUNet", "3d_fullres",
    "Task017_AbdominalOrganSegmentation",
    "nnUNetTrainerV2_MedNeXt_S_kernel3__nnUNetPlansv2.1_trgSp_1x1x1"
)

# nnUNet env vars — PHẢI set TRƯỚC KHI import nnunet_mednext
os.environ["nnUNet_raw_data_base"] = os.path.join(BASE_DIR, "model", "data", "nnUNet_raw_data_base")
os.environ["nnUNet_preprocessed"]  = os.path.join(BASE_DIR, "model", "data", "nnUNet_preprocessed")
os.environ["RESULTS_FOLDER"]       = os.path.join(BASE_DIR, "model", "data", "nnUNet_trained_models")

# Chỉ dùng fold_0 (demo mode)
FOLDS      = [0]
CHECKPOINT = "model_best"

TEMP_INPUT_DIR  = os.path.join(BASE_DIR, "backend", "temp", "input")
TEMP_OUTPUT_DIR = os.path.join(BASE_DIR, "backend", "temp", "output")

# ─── BTCV Label Map — 14 classes ───────────────────────────────────────────
LABEL_MAP = {
    0:  {"name": "Background",            "color": (0,   0,   0),   "hex": "#000000"},
    1:  {"name": "Spleen",                "color": (76,  175, 80),  "hex": "#4CAF50"},
    2:  {"name": "Right Kidney",          "color": (126, 184, 247), "hex": "#7EB8F7"},
    3:  {"name": "Left Kidney",           "color": (91,  140, 255), "hex": "#5B8CFF"},
    4:  {"name": "Gallbladder",           "color": (255, 140, 66),  "hex": "#FF8C42"},
    5:  {"name": "Esophagus",             "color": (255, 107, 157), "hex": "#FF6B9D"},
    6:  {"name": "Liver",                 "color": (232, 212, 77),  "hex": "#E8D44D"},
    7:  {"name": "Stomach",               "color": (30,  144, 255), "hex": "#1E90FF"},
    8:  {"name": "Aorta",                 "color": (255, 68,  68),  "hex": "#FF4444"},
    9:  {"name": "Inferior Vena Cava",    "color": (255, 159, 67),  "hex": "#FF9F43"},
    10: {"name": "Portal & Splenic Veins","color": (162, 155, 254), "hex": "#A29BFE"},
    11: {"name": "Pancreas",              "color": (181, 123, 255), "hex": "#B57BFF"},
    12: {"name": "Right Adrenal Gland",   "color": (0,   206, 201), "hex": "#00CEC9"},
    13: {"name": "Left Adrenal Gland",    "color": (85,  239, 196), "hex": "#55EFC4"},
}
```

---

### ⚙️ File `inference.py`

```python
"""
QUAN TRỌNG: import config (set env vars) TRƯỚC khi import bất cứ thứ gì từ nnunet_mednext
"""
import sys, os, uuid, shutil, time, json
from pathlib import Path

BACKEND_DIR = Path(__file__).parent
ROOT_DIR    = BACKEND_DIR.parent
MODEL_DIR   = ROOT_DIR / "model"
if str(MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(MODEL_DIR))

import config          # ← set env vars TRƯỚC — KHÔNG được đổi thứ tự
from config import *   # import constants

from nnunet_mednext.inference.predict import predict_from_folder
import nibabel as nib
import numpy as np


def run_inference(input_nifti_path: str) -> dict:
    """
    Args:
        input_nifti_path: đường dẫn tuyệt đối tới file .nii.gz do user upload
    Returns:
        dict với keys: job_id, seg_path, ct_path, processing_time, total_slices, organs_detected
    """
    job_id = str(uuid.uuid4())[:8]
    job_input_dir  = os.path.join(TEMP_INPUT_DIR,  job_id)
    job_output_dir = os.path.join(TEMP_OUTPUT_DIR, job_id)
    os.makedirs(job_input_dir,  exist_ok=True)
    os.makedirs(job_output_dir, exist_ok=True)

    # nnUNet BẮT BUỘC: tên file phải theo format CASENAME_0000.nii.gz
    dest_path = os.path.join(job_input_dir, f"case_{job_id}_0000.nii.gz")
    shutil.copy2(input_nifti_path, dest_path)

    start = time.time()

    predict_from_folder(
        model                     = MODEL_BASE_PATH,
        input_folder              = job_input_dir,
        output_folder             = job_output_dir,
        folds                     = FOLDS,           # [0] — fold_0 only
        save_npz                  = False,
        num_threads_preprocessing = 1,               # giảm cho web demo
        num_threads_nifti_save    = 1,
        lowres_segmentations      = None,
        part_id                   = 0,
        num_parts                 = 1,
        tta                       = False,            # TTA off cho nhanh
        mixed_precision           = True,
        overwrite_existing        = True,
        mode                      = "normal",
        overwrite_all_in_gpu      = None,
        step_size                 = 0.5,
        checkpoint_name           = CHECKPOINT,      # "model_best"
    )

    elapsed = time.time() - start

    output_files = list(Path(job_output_dir).glob("*.nii.gz"))
    if not output_files:
        raise RuntimeError("Model did not generate output segmentation file.")

    seg_path = str(output_files[0])
    seg_data = nib.load(seg_path).get_fdata().astype(int)
    unique_labels = set(np.unique(seg_data).tolist())

    # Kiểm tra từng label BTCV (1–13) có xuất hiện không
    organs_detected = []
    for label_id in range(1, 14):
        info = LABEL_MAP[label_id]
        organs_detected.append({
            "id":      label_id,
            "name":    info["name"],
            "color":   info["hex"],
            "present": (label_id in unique_labels)
        })

    # Lưu metadata để visualize.py dùng sau
    meta = {
        "ct_path":     dest_path,
        "seg_path":    seg_path,
        "total_slices": int(seg_data.shape[2]),   # dim Z = axial
        "organs":      organs_detected,
    }
    with open(os.path.join(job_output_dir, "meta.json"), "w") as f:
        json.dump(meta, f)

    return {
        "job_id":          job_id,
        "seg_path":        seg_path,
        "ct_path":         dest_path,
        "processing_time": round(elapsed, 2),
        "total_slices":    int(seg_data.shape[2]),
        "organs_detected": organs_detected,
    }
```

---

### ⚙️ File `visualize.py`

```python
"""
Render visualization PNG:
  - Bên trái: CT slice grayscale (windowed HU soft tissue)
  - Bên phải: CT slice + segmentation overlay màu theo BTCV label map
  - Bên dưới: legend tên + màu từng organ (chỉ hiện organ present=True)
"""
import io, json, os
import numpy as np
import nibabel as nib
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path
from config import LABEL_MAP, TEMP_OUTPUT_DIR


def _load_meta(job_id: str) -> dict:
    meta_path = os.path.join(TEMP_OUTPUT_DIR, job_id, "meta.json")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"Job {job_id} not found.")
    with open(meta_path) as f:
        return json.load(f)


def _window_ct(ct_slice: np.ndarray) -> np.ndarray:
    """
    Window/level CT về [0, 255].
    Soft tissue abdomen: WL=50, WW=500 → HU range [-200, 300]
    """
    lo, hi = -200, 300
    clipped = np.clip(ct_slice, lo, hi)
    return ((clipped - lo) / (hi - lo) * 255).astype(np.uint8)


def _get_slice(volume: np.ndarray, idx: int) -> np.ndarray:
    """Lấy slice axial (axis Z) và rotate về đúng hướng giải phẫu."""
    slc = volume[:, :, idx]
    return np.rot90(slc)


def render_slice_png(job_id: str, slice_idx: int, alpha: float = 0.5) -> bytes:
    """
    Render 1 slice axial thành PNG:
      [CT grayscale]  |  [Segmentation overlay]
      + legend màu organ ở dưới cùng
    """
    meta = _load_meta(job_id)

    ct_data  = nib.load(meta["ct_path"]).get_fdata()
    seg_data = nib.load(meta["seg_path"]).get_fdata().astype(int)

    total_z   = ct_data.shape[2]
    slice_idx = max(0, min(slice_idx, total_z - 1))

    ct_slice  = _get_slice(ct_data,  slice_idx)
    seg_slice = _get_slice(seg_data, slice_idx)

    H, W = ct_slice.shape

    # ── Panel trái: CT grayscale ─────────────────────────────────────────────
    ct_gray = _window_ct(ct_slice)
    ct_rgb  = np.stack([ct_gray] * 3, axis=-1).astype(np.float32)

    # ── Panel phải: CT + overlay màu BTCV ───────────────────────────────────
    overlay = ct_rgb.copy()
    for label_id, info in LABEL_MAP.items():
        if label_id == 0:
            continue                          # skip background
        mask = seg_slice == label_id
        if not mask.any():
            continue
        color = np.array(info["color"], dtype=np.float32)
        for c in range(3):
            overlay[:, :, c] = np.where(
                mask,
                overlay[:, :, c] * (1 - alpha) + color[c] * alpha,
                overlay[:, :, c]
            )
    overlay = np.clip(overlay, 0, 255).astype(np.uint8)

    # ── Layout canvas ────────────────────────────────────────────────────────
    TITLE_H  = 24
    LEGEND_H = 70
    PAD      = 12
    canvas_w = W * 2 + PAD * 3
    canvas_h = TITLE_H + H + LEGEND_H + PAD * 2
    canvas   = Image.new("RGB", (canvas_w, canvas_h), (15, 17, 21))

    # Paste panels
    canvas.paste(Image.fromarray(ct_rgb.astype(np.uint8)), (PAD, TITLE_H + PAD))
    canvas.paste(Image.fromarray(overlay), (PAD * 2 + W, TITLE_H + PAD))

    draw = ImageDraw.Draw(canvas)

    # Tiêu đề mỗi panel
    try:
        font_title  = ImageFont.truetype("arial.ttf", 13)
        font_legend = ImageFont.truetype("arial.ttf", 11)
    except Exception:
        font_title  = ImageFont.load_default()
        font_legend = font_title

    draw.text((PAD + W // 2 - 28, 5),          "CT Input",   fill=(200, 200, 200), font=font_title)
    draw.text((PAD * 2 + W + W // 2 - 35, 5),  "Prediction", fill=(200, 200, 200), font=font_title)

    # Đường ngăn cách giữa 2 panel (vertical)
    mid_x = PAD + W + PAD // 2
    draw.line([(mid_x, PAD), (mid_x, TITLE_H + H + PAD)], fill=(50, 55, 65), width=1)

    # ── Legend: chỉ hiện organ present=True ─────────────────────────────────
    present_organs = [o for o in meta["organs"] if o["present"]]
    legend_y = TITLE_H + H + PAD + 8
    BOX      = 12
    ITEM_W   = 115
    per_row  = max(1, canvas_w // ITEM_W)

    for i, organ in enumerate(present_organs):
        col = i % per_row
        row = i // per_row
        x   = PAD + col * ITEM_W
        y   = legend_y + row * 20

        hex_str = organ["color"].lstrip("#")
        rgb     = tuple(int(hex_str[j:j+2], 16) for j in (0, 2, 4))
        draw.rectangle([x, y, x + BOX, y + BOX], fill=rgb, outline=(80, 85, 95))
        draw.text((x + BOX + 4, y + 1), organ["name"], fill=(190, 195, 205), font=font_legend)

    buf = io.BytesIO()
    canvas.save(buf, format="PNG")
    buf.seek(0)
    return buf.read()


def render_overview_png(job_id: str, n_slices: int = 6, alpha: float = 0.5) -> bytes:
    """
    Render n_slices đại diện (bỏ 10% đầu/cuối), stack dọc thành 1 ảnh PNG lớn.
    """
    meta    = _load_meta(job_id)
    total_z = meta["total_slices"]
    start_z = int(total_z * 0.10)
    end_z   = int(total_z * 0.90)
    indices = np.linspace(start_z, end_z, n_slices, dtype=int).tolist()

    panels = [Image.open(io.BytesIO(render_slice_png(job_id, idx, alpha))) for idx in indices]
    max_w  = max(p.width for p in panels)
    total_h = sum(p.height for p in panels)

    full  = Image.new("RGB", (max_w, total_h), (15, 17, 21))
    y_off = 0
    for p in panels:
        full.paste(p, (0, y_off))
        y_off += p.height

    buf = io.BytesIO()
    full.save(buf, format="PNG")
    buf.seek(0)
    return buf.read()
```

---

### ⚙️ File `main.py`

```python
import os, uuid, shutil, json, asyncio
from pathlib import Path
from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response

import config   # set env vars TRƯỚC mọi thứ
from config import TEMP_INPUT_DIR, TEMP_OUTPUT_DIR
from inference import run_inference
from visualize import render_slice_png, render_overview_png

app = FastAPI(title="MedNeXt BTCV Segmentation API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)

os.makedirs(TEMP_INPUT_DIR,  exist_ok=True)
os.makedirs(TEMP_OUTPUT_DIR, exist_ok=True)


@app.get("/api/health")
def health():
    import torch
    return {
        "status": "ok",
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "model":  "MedNeXt-S fold_0",
        "dataset": "BTCV (14 classes)"
    }


@app.post("/api/segment")
async def segment(file: UploadFile = File(...)):
    if not (file.filename.endswith(".nii") or file.filename.endswith(".nii.gz")):
        raise HTTPException(400, "Only .nii and .nii.gz files are accepted.")

    tmp_path = os.path.join(TEMP_INPUT_DIR, f"up_{uuid.uuid4().hex}_{file.filename}")
    with open(tmp_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    try:
        loop   = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, run_inference, tmp_path)
    except Exception as e:
        raise HTTPException(500, f"Segmentation failed: {e}")
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

    return JSONResponse({
        "status":                  "success",
        "job_id":                  result["job_id"],
        "message":                 "Segmentation complete",
        "download_url":            f"/api/result/{result['job_id']}",
        "preview_url":             f"/api/preview/{result['job_id']}",
        "total_slices":            result["total_slices"],
        "organs_detected":         result["organs_detected"],
        "processing_time_seconds": result["processing_time"],
    })


@app.get("/api/preview/{job_id}")
def preview_slice(
    job_id: str,
    slice: int   = Query(default=-1),
    alpha: float = Query(default=0.5, ge=0.0, le=1.0),
):
    try:
        meta_path = os.path.join(TEMP_OUTPUT_DIR, job_id, "meta.json")
        if not os.path.exists(meta_path):
            raise HTTPException(404, f"Job {job_id} not found.")
        with open(meta_path) as f:
            meta = json.load(f)

        idx       = slice if slice >= 0 else meta["total_slices"] // 2
        png_bytes = render_slice_png(job_id, idx, alpha)
        return Response(content=png_bytes, media_type="image/png")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, str(e))


@app.get("/api/preview_full/{job_id}")
def preview_full(job_id: str, alpha: float = Query(default=0.5)):
    try:
        png_bytes = render_overview_png(job_id, n_slices=6, alpha=alpha)
        return Response(content=png_bytes, media_type="image/png")
    except Exception as e:
        raise HTTPException(500, str(e))


@app.get("/api/result/{job_id}")
def get_result(job_id: str):
    job_dir = Path(TEMP_OUTPUT_DIR) / job_id
    files   = list(job_dir.glob("*.nii.gz"))
    if not files:
        raise HTTPException(404, "Result not found.")
    return FileResponse(
        str(files[0]),
        filename   = f"segmentation_{job_id}.nii.gz",
        media_type = "application/octet-stream"
    )
```

---

### 📦 `requirements.txt`

```
fastapi>=0.111.0
uvicorn[standard]>=0.29.0
python-multipart>=0.0.9
torch>=2.0.0
nibabel>=5.2.0
numpy>=1.24.0
Pillow>=10.0.0
batchgenerators
```

---

## 🖥️ Tích hợp Frontend (Next.js) — Cập nhật `page.tsx`

### Types

```typescript
type OrganInfo = {
  id: number
  name: string
  color: string   // hex vd "#4CAF50"
  present: boolean
}

type CaseFile = {
  id: string
  file: File           // giữ reference để upload
  name: string
  size: string
  status: 'uploading' | 'processing' | 'done' | 'error'
  jobId?: string
  downloadUrl?: string
  totalSlices?: number
  currentSlice?: number
  organs?: OrganInfo[]
  processingTime?: number
  errorMessage?: string
}
```

### Hàm `uploadAndSegment()`

```typescript
const BACKEND = 'http://localhost:8000'

async function uploadAndSegment(file: File, caseId: string) {
  const formData = new FormData()
  formData.append('file', file)

  setCases(prev => prev.map(c =>
    c.id === caseId ? { ...c, status: 'processing' } : c
  ))

  try {
    const res  = await fetch(`${BACKEND}/api/segment`, { method: 'POST', body: formData })
    const data = await res.json()
    if (!res.ok) throw new Error(data.detail || 'Segmentation failed')

    setCases(prev => prev.map(c =>
      c.id === caseId ? {
        ...c,
        status:         'done',
        jobId:          data.job_id,
        downloadUrl:    `${BACKEND}/api/result/${data.job_id}`,
        totalSlices:    data.total_slices,
        currentSlice:   Math.floor(data.total_slices / 2),
        organs:         data.organs_detected,
        processingTime: data.processing_time_seconds,
      } : c
    ))
  } catch (err: unknown) {
    const msg = err instanceof Error ? err.message : 'Unknown error'
    setCases(prev => prev.map(c =>
      c.id === caseId ? { ...c, status: 'error', errorMessage: msg } : c
    ))
  }
}
```

### Viewer (thay `"Viewer panels will appear after NIfTI parsing."`)

```tsx
{selectedCase?.status === 'done' && selectedCase.jobId && (
  <div className="flex flex-1 flex-col gap-4 p-6 overflow-auto">

    {/* Preview image — reload khi slice thay đổi */}
    <div className="rounded-xl overflow-hidden border border-[#dbe3e8] bg-black">
      <img
        key={`${selectedCase.jobId}-${selectedCase.currentSlice}`}
        src={`${BACKEND}/api/preview/${selectedCase.jobId}?slice=${selectedCase.currentSlice ?? 0}&alpha=0.5`}
        alt="Segmentation preview"
        className="w-full object-contain"
        style={{ maxHeight: '480px' }}
      />
    </div>

    {/* Slider */}
    <div className="flex items-center gap-3">
      <span className="text-xs text-[#7d8996] w-16 shrink-0">Axial</span>
      <input
        type="range"
        min={0}
        max={(selectedCase.totalSlices ?? 1) - 1}
        value={selectedCase.currentSlice ?? 0}
        onChange={e => {
          const val = Number(e.target.value)
          setCases(prev => prev.map(c =>
            c.id === selectedCase.id ? { ...c, currentSlice: val } : c
          ))
        }}
        className="flex-1 accent-cyan-400"
      />
      <span className="text-xs text-[#7d8996] w-20 text-right shrink-0">
        {(selectedCase.currentSlice ?? 0) + 1} / {selectedCase.totalSlices}
      </span>
    </div>

    {/* Color Legend — chỉ hiện organ present=true */}
    <div className="rounded-lg border border-[#dbe3e8] bg-[#f9fbfc] p-3">
      <div className="mb-2 text-[10px] font-semibold uppercase tracking-[0.16em] text-[#7d8996]">
        Detected Organs (BTCV)
      </div>
      <div className="flex flex-wrap gap-x-4 gap-y-2">
        {selectedCase.organs?.map(organ => (
          <div key={organ.id} className="flex items-center gap-1.5">
            <span
              className="inline-block size-3 rounded-sm flex-shrink-0 border border-white/10"
              style={{ backgroundColor: organ.color, opacity: organ.present ? 1 : 0.25 }}
            />
            <span className={`text-[11px] ${organ.present ? 'text-[#b0b8c4]' : 'text-[#5a6270] line-through'}`}>
              {organ.name}
            </span>
          </div>
        ))}
      </div>
    </div>

    {/* Footer: time + download */}
    <div className="flex items-center justify-between text-xs text-[#7d8996]">
      <span>⏱ {selectedCase.processingTime?.toFixed(1)}s · MedNeXt-S fold_0 · BTCV</span>
      <a
        href={selectedCase.downloadUrl}
        download
        className="flex items-center gap-1.5 rounded-md bg-[#647184] px-3 py-1.5 text-white hover:bg-[#566276]"
      >
        ⬇ Download .nii.gz
      </a>
    </div>
  </div>
)}
```

---

## 🖥️ Môi trường thực tế

> **Quan trọng**: Backend KHÔNG chạy trên máy local. Toàn bộ model + backend chạy trên **server của trường**.

| | Chi tiết |
|---|---|
| **Nơi chạy** | Server GPU của trường (Linux) |
| **Conda env** | Một env riêng đã cài nnUNet/MedNeXt theo `pip install -e .` từ repo |
| **Package** | `nnunet_mednext` được cài dạng editable (`pip install -e model/`) — **KHÔNG phải từ PyPI** |
| **Frontend** | Vẫn có thể chạy local (localhost:3000), gọi API qua IP/domain của server |
| **Backend port** | `0.0.0.0:8000` — expose ra ngoài để frontend local gọi được |

---

## 🚀 Hướng dẫn chạy

### Trên server trường (nơi có model + GPU)

```bash
# 1. Activate conda env đã cài nnunet_mednext
conda activate <tên_env_của_bạn>

# 2. Kiểm tra nnunet_mednext đã import được chưa
python -c "import nnunet_mednext; print('OK')"

# 3. Cài thêm các package web (FastAPI, nibabel, Pillow...)
#    nnunet_mednext và torch đã có sẵn rồi, KHÔNG cài lại
pip install fastapi "uvicorn[standard]" python-multipart nibabel Pillow

# 4. Tạo thư mục backend trong project (nếu chưa có)
cd /path/to/MedNeXt-webdemo
mkdir -p backend/temp/input backend/temp/output

# 5. Chạy FastAPI server
cd backend
uvicorn main:app --host 0.0.0.0 --port 8000
```

### Trên máy local (frontend)

```bash
# Frontend gọi API lên server — cần đổi BACKEND URL
# Trong page.tsx: const BACKEND = 'http://<IP_SERVER_TRUONG>:8000'

cd frontend
pnpm dev   # http://localhost:3000
```

> **Lưu ý port**: Nếu server trường có firewall, cần mở port 8000 hoặc dùng SSH tunnel:
> ```bash
> ssh -L 8000:localhost:8000 <user>@<server_ip>
> ```
> Khi đó frontend vẫn dùng `BACKEND = 'http://localhost:8000'` là được.

---

## ⚠️ Lưu ý quan trọng cho AI

1. **`nnunet_mednext` đã được cài bằng `pip install -e .`** từ thư mục `model/` — không install lại từ PyPI. Khi viết `sys.path`, không cần thêm thủ công nếu package đã install editable, nhưng vẫn nên để phòng hờ.

2. **`os.environ` nnUNet phải set TRƯỚC khi import `nnunet_mednext`** — sai thứ tự là crash ngay lập tức. `import config` phải là dòng đầu tiên trong `inference.py`.

3. **`sys.path`**: Nếu chạy trên server mà không chắc editable install hoạt động, thêm `model/` vào `sys.path[0]` như sau:
   ```python
   import sys
   sys.path.insert(0, "/path/to/MedNeXt-webdemo/model")
   ```
   Đường dẫn trên **server** có thể khác đường dẫn Windows — hãy dùng `Path(__file__).parent.parent / "model"` để tự suy ra.

4. **Input filename**: nnUNet STRICT về format `CASENAME_0000.nii.gz` — nếu không đúng sẽ báo lỗi `"did not find any valid input files"`.

5. **BTCV có 14 classes (0–13)** — code inference, visualize, và frontend đều phải dùng đúng bộ label map này, KHÔNG phải Synapse 8 class.

6. **Blocking inference**: `predict_from_folder()` chạy ~2-5 phút. Bắt buộc dùng `await loop.run_in_executor(None, run_inference, path)` để FastAPI không bị freeze.

7. **Fold đơn**: `FOLDS = [0]`, `CHECKPOINT = "model_best"` — không ensemble.

8. **CT Windowing**: clip HU về [-200, 300] (soft tissue abdomen) để ảnh CT trông rõ tạng.

9. **Font trên Linux server**: Không có `arial.ttf`. Dùng `ImageFont.load_default()` hoặc cài font:
   ```bash
   apt-get install -y fonts-liberation
   # rồi dùng: ImageFont.truetype("/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf", 13)
   ```

10. **CORS**: `allow_origins` trong FastAPI phải include địa chỉ frontend. Nếu frontend chạy local và backend trên server:
    ```python
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"]
    ```
    Hoặc dùng `allow_origins=["*"]` cho demo.

11. **`organs` trong legend**: hiển thị tất cả 13 organs nhưng `line-through` + mờ với organ `present=false`. Đừng ẩn hoàn toàn.

---

## 🔄 Luồng hoạt động

```
User upload .nii.gz  (browser — máy local)
        ↓  POST http://<SERVER_IP>:8000/api/segment  multipart
FastAPI backend (chạy trên server trường, port 8000)
        ↓  lưu → backend/temp/input/{job_id}/case_{job_id}_0000.nii.gz
inference.py: predict_from_folder(folds=[0], chk="model_best")
        ↓  ~2-5 phút (GPU server)
        ↓  output → backend/temp/output/{job_id}/case_{job_id}.nii.gz
        ↓  đọc mask → detect 13 BTCV organs → save meta.json
JSON: { job_id, total_slices, organs_detected[13], preview_url }
        ↓
Next.js frontend (local)
        ↓  <img src="http://<SERVER_IP>:8000/api/preview/{job_id}?slice=42">
        ↓  Slider axial → cập nhật ?slice= trong URL → ảnh reload
        ↓  Legend 13 organs BTCV có màu (present ✓ / not present ╌)
        ↓  Nút Download .nii.gz
```
