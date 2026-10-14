import asyncio
import json
import logging
import os
import re
import shutil
import uuid
from pathlib import Path

logger = logging.getLogger("uvicorn.error")

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response

import config  # set env vars TRƯỚC mọi thứ
from config import TEMP_INPUT_DIR, TEMP_OUTPUT_DIR
from inference import run_inference
from visualize import render_overview_png, render_slice_png

app = FastAPI(title="MedNeXt BTCV Segmentation API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

os.makedirs(TEMP_INPUT_DIR, exist_ok=True)
os.makedirs(TEMP_OUTPUT_DIR, exist_ok=True)

_JOB_ID = re.compile(r"^[a-f0-9]{8}$")


def _require_job(job_id: str) -> str:
    if not _JOB_ID.fullmatch(job_id):
        raise HTTPException(400, "Invalid job id.")
    return job_id


def _is_nifti(filename: str | None) -> bool:
    if not filename:
        return False
    lower = filename.lower()
    return lower.endswith(".nii") or lower.endswith(".nii.gz")


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
async def segment(file: UploadFile = File(...)):
    if not _is_nifti(file.filename):
        raise HTTPException(400, "Only .nii and .nii.gz files are accepted.")

    safe_name = Path(file.filename or "upload.nii.gz").name
    tmp_path = os.path.join(TEMP_INPUT_DIR, f"up_{uuid.uuid4().hex}_{safe_name}")
    with open(tmp_path, "wb") as handle:
        shutil.copyfileobj(file.file, handle)

    try:
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(None, run_inference, tmp_path)
    except Exception as exc:
        logger.exception("Segmentation failed")
        raise HTTPException(500, f"Segmentation failed: {exc}") from exc
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

    return JSONResponse({
        "status": "success",
        "job_id": result["job_id"],
        "message": "Segmentation complete",
        "download_url": f"/api/result/{result['job_id']}",
        "preview_url": f"/api/preview/{result['job_id']}",
        "total_slices": result["total_slices"],
        "organs_detected": result["organs_detected"],
        "processing_time_seconds": result["processing_time"],
    })


@app.get("/api/preview/{job_id}")
def preview_slice(
    job_id: str,
    slice: int = Query(default=-1),
    alpha: float = Query(default=0.5, ge=0.0, le=1.0),
):
    _require_job(job_id)
    try:
        meta_path = os.path.join(TEMP_OUTPUT_DIR, job_id, "meta.json")
        if not os.path.exists(meta_path):
            raise HTTPException(404, f"Job {job_id} not found.")
        with open(meta_path, encoding="utf-8") as handle:
            meta = json.load(handle)

        idx = slice if slice >= 0 else meta["total_slices"] // 2
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
    job_dir = Path(TEMP_OUTPUT_DIR) / job_id
    files = list(job_dir.glob("*.nii.gz"))
    if not files:
        raise HTTPException(404, "Result not found.")
    return FileResponse(
        str(files[0]),
        filename=f"segmentation_{job_id}.nii.gz",
        media_type="application/octet-stream",
    )
