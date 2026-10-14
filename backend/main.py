import asyncio
import json
import logging
import os
import re
import shutil
import uuid
from pathlib import Path
from typing import Annotated

logger = logging.getLogger("uvicorn.error")

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response

import config  # set env vars TRƯỚC mọi thứ
from config import TEMP_CACHE_DIR, TEMP_INPUT_DIR, TEMP_OUTPUT_DIR
from inference import attach_ground_truth, get_volume_gzip, run_inference
from visualize import prediction_file, render_overview_png, render_slice_png, resolve_job_dir

app = FastAPI(title="MedNeXt BTCV Segmentation API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-Volume-Shape", "X-Volume-Dtype"],
)

os.makedirs(TEMP_INPUT_DIR, exist_ok=True)
os.makedirs(TEMP_OUTPUT_DIR, exist_ok=True)
os.makedirs(TEMP_CACHE_DIR, exist_ok=True)

_JOB_ID = re.compile(r"^[a-f0-9]{8}(?:[a-f0-9]{8})?$")


def _require_job(job_id: str) -> str:
    if not _JOB_ID.fullmatch(job_id):
        raise HTTPException(400, "Invalid job id.")
    return job_id


def _is_nifti(filename: str | None) -> bool:
    if not filename:
        return False
    lower = filename.lower()
    return lower.endswith(".nii") or lower.endswith(".nii.gz")


def _has_upload(upload: UploadFile | None) -> bool:
    return bool(upload and (upload.filename or "").strip())


def _save_upload(upload: UploadFile) -> str:
    safe_name = Path(upload.filename or "upload.nii.gz").name
    dest = os.path.join(TEMP_INPUT_DIR, f"up_{uuid.uuid4().hex}_{safe_name}")
    with open(dest, "wb") as handle:
        shutil.copyfileobj(upload.file, handle)
    return dest


def _payload(result: dict) -> dict:
    cached = bool(result["cached"])
    return {
        "status": "success",
        "job_id": result["job_id"],
        "cached": cached,
        "has_gt": bool(result["has_gt"]),
        "mean_dice": result.get("mean_dice"),
        "organ_dice": result.get("organ_dice") or {},
        "message": "Loaded from cache" if cached else "Segmentation complete",
        "download_url": f"/api/result/{result['job_id']}",
        "preview_url": f"/api/preview/{result['job_id']}",
        "total_slices": result["total_slices"],
        "volume_shape": result.get("volume_shape"),
        "organs_detected": result["organs_detected"],
        "processing_time_seconds": result["processing_time"],
    }


@app.get("/api/health")
def health():
    import torch

    return {
        "status": "ok",
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "model": "MedNeXt-S fold_0",
        "dataset": "BTCV (14 classes)",
    }


@app.post("/api/segment")
async def segment(
    file: UploadFile = File(...),
    gt_file: Annotated[UploadFile | None, File()] = None,
):
    if not _is_nifti(file.filename):
        raise HTTPException(400, "Only .nii and .nii.gz files are accepted.")
    if _has_upload(gt_file) and not _is_nifti(gt_file.filename):
        raise HTTPException(400, "Ground truth must be a .nii or .nii.gz file.")

    tmp_path = _save_upload(file)
    gt_tmp = _save_upload(gt_file) if _has_upload(gt_file) else None
    try:
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(None, run_inference, tmp_path, gt_tmp)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(404, str(exc)) from exc
    except Exception as exc:
        logger.exception("Segmentation failed")
        raise HTTPException(500, f"Segmentation failed: {exc}") from exc
    finally:
        for path in (tmp_path, gt_tmp):
            if path and os.path.exists(path):
                os.remove(path)

    return JSONResponse(_payload(result))


@app.post("/api/upload_gt/{job_id}")
async def upload_gt(job_id: str, gt_file: UploadFile = File(...)):
    _require_job(job_id)
    if not _is_nifti(gt_file.filename):
        raise HTTPException(400, "Ground truth must be a .nii or .nii.gz file.")

    gt_tmp = _save_upload(gt_file)
    try:
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(None, attach_ground_truth, job_id, gt_tmp)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(404, str(exc)) from exc
    except Exception as exc:
        logger.exception("Ground truth upload failed")
        raise HTTPException(500, f"Ground truth upload failed: {exc}") from exc
    finally:
        if os.path.exists(gt_tmp):
            os.remove(gt_tmp)

    return JSONResponse(_payload(result))


@app.get("/api/volume/{job_id}/{kind}")
async def volume(job_id: str, kind: str):
    _require_job(job_id)
    if kind not in {"ct", "pred", "gt"}:
        raise HTTPException(404, "Volume kind must be ct, pred, or gt.")
    try:
        loop = asyncio.get_running_loop()
        payload, shape = await loop.run_in_executor(None, get_volume_gzip, job_id, kind)
    except FileNotFoundError as exc:
        raise HTTPException(404, str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    except Exception as exc:
        logger.exception("Volume export failed")
        raise HTTPException(500, str(exc)) from exc

    return Response(
        content=payload,
        media_type="application/octet-stream",
        headers={
            "Content-Encoding": "gzip",
            "X-Volume-Shape": ",".join(str(int(v)) for v in shape),
            "X-Volume-Dtype": "uint8",
            "Cache-Control": "no-store",
        },
    )


@app.get("/api/preview/{job_id}")
def preview_slice(
    job_id: str,
    slice: int = Query(default=-1),
    alpha: float = Query(default=0.5, ge=0.0, le=1.0),
):
    _require_job(job_id)
    try:
        job_dir = resolve_job_dir(job_id)
        meta_path = os.path.join(job_dir, "meta.json")
        if not os.path.exists(meta_path):
            raise HTTPException(404, f"Job {job_id} not found.")
        with open(meta_path, encoding="utf-8") as handle:
            meta = json.load(handle)

        idx = slice if slice >= 0 else int(meta["total_slices"]) // 2
        png_bytes = render_slice_png(job_id, idx, alpha)
        return Response(content=png_bytes, media_type="image/png")
    except HTTPException:
        raise
    except FileNotFoundError as exc:
        raise HTTPException(404, str(exc)) from exc
    except Exception as exc:
        raise HTTPException(500, str(exc)) from exc


@app.get("/api/preview_full/{job_id}")
def preview_full(job_id: str, alpha: float = Query(default=0.5, ge=0.0, le=1.0)):
    _require_job(job_id)
    try:
        png_bytes = render_overview_png(job_id, n_slices=6, alpha=alpha)
        return Response(content=png_bytes, media_type="image/png")
    except FileNotFoundError as exc:
        raise HTTPException(404, str(exc)) from exc
    except Exception as exc:
        raise HTTPException(500, str(exc)) from exc


@app.get("/api/result/{job_id}")
def get_result(job_id: str):
    _require_job(job_id)
    try:
        job_dir = resolve_job_dir(job_id)
    except FileNotFoundError as exc:
        raise HTTPException(404, str(exc)) from exc
    path = prediction_file(job_dir)
    if not path:
        raise HTTPException(404, "Result not found.")
    return FileResponse(
        path,
        filename=f"segmentation_{job_id}.nii.gz",
        media_type="application/octet-stream",
    )
