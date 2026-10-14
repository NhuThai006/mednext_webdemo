"""
QUAN TRỌNG: import config (set env vars) TRƯỚC khi import bất cứ thứ gì từ nnunet_mednext.

Kết quả segmentation được cache theo SHA-256 của file CT upload.
Upload lại cùng nội dung sẽ không gọi model.
"""
import gzip
import hashlib
import logging
import os
import shutil
import sys
import threading
import time
from pathlib import Path

BACKEND_DIR = Path(__file__).parent
ROOT_DIR = BACKEND_DIR.parent
MODEL_DIR = ROOT_DIR / "model"
if str(MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(MODEL_DIR))

import config  # set env vars TRƯỚC — KHÔNG được đổi thứ tự
from config import (
    CHECKPOINT,
    FOLDS,
    MODEL_BASE_PATH,
    TEMP_CACHE_DIR,
)

from visualize import (
    ct_file,
    display_shape,
    ensure_volume_files,
    load_meta,
    merge_dice_into_organs,
    prediction_file,
    resolve_job_dir,
    save_meta,
    score_ground_truth,
    summarize_prediction,
)

from nnunet_mednext.inference.predict import predict_from_folder
import nibabel as nib

logger = logging.getLogger("uvicorn.error")

_LOCKS: dict[str, threading.Lock] = {}
_LOCKS_GUARD = threading.Lock()


def job_lock(job_id: str) -> threading.Lock:
    with _LOCKS_GUARD:
        lock = _LOCKS.get(job_id)
        if lock is None:
            lock = threading.Lock()
            _LOCKS[job_id] = lock
        return lock


def sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()[:16]


def _stage_nifti(src: str, dest: str) -> None:
    """nnUNet chỉ nhận CASENAME_0000.nii.gz. File .nii thuần được nén gzip."""
    lower = src.lower()
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    if lower.endswith(".nii.gz"):
        shutil.copy2(src, dest)
        return
    if lower.endswith(".nii"):
        with open(src, "rb") as raw, gzip.open(dest, "wb") as gz:
            shutil.copyfileobj(raw, gz)
        return
    raise ValueError("Only .nii and .nii.gz files are accepted.")


def _result_dict(job_id: str, meta: dict) -> dict:
    return {
        "job_id": job_id,
        "seg_path": meta.get("seg_path"),
        "ct_path": meta.get("ct_path"),
        "processing_time": meta.get("processing_time") or 0,
        "total_slices": int(meta["total_slices"]),
        "organs_detected": meta.get("organs") or [],
        "cached": bool(meta.get("cached")),
        "has_gt": bool(meta.get("has_gt")),
        "mean_dice": meta.get("mean_dice"),
        "organ_dice": meta.get("organ_dice") or {},
        "volume_shape": meta.get("volume_shape"),
    }


def _empty_gt_fields(meta: dict) -> None:
    meta["has_gt"] = False
    meta["mean_dice"] = None
    meta["organ_dice"] = {}
    meta.pop("gt_path", None)
    meta.pop("gt_hash", None)


def _apply_ground_truth(job_dir: str, meta: dict, gt_src: str) -> dict:
    pred_path = prediction_file(job_dir, meta)
    if not pred_path:
        raise FileNotFoundError("Prediction is not available for this case.")
    meta["seg_path"] = os.path.abspath(pred_path)
    gt_hash = sha256_file(gt_src)
    dest = os.path.join(job_dir, "gt.nii.gz")
    if meta.get("gt_hash") == gt_hash and os.path.exists(dest) and meta.get("organ_dice"):
        meta["has_gt"] = True
        meta["gt_path"] = os.path.abspath(dest)
        meta["organs"] = merge_dice_into_organs(meta.get("organs") or [], meta["organ_dice"])
        return meta

    staged = os.path.join(job_dir, "_gt_incoming.nii.gz")
    _stage_nifti(gt_src, staged)
    try:
        organ_dice, mean_dice = score_ground_truth(pred_path, staged)
    except Exception:
        if os.path.exists(staged):
            os.remove(staged)
        raise
    os.replace(staged, dest)
    gz = os.path.join(job_dir, "gt.bin.gz")
    if os.path.exists(gz):
        os.remove(gz)

    meta["gt_path"] = os.path.abspath(dest)
    meta["gt_hash"] = gt_hash
    meta["has_gt"] = True
    meta["organ_dice"] = organ_dice
    meta["mean_dice"] = mean_dice
    if not meta.get("organs"):
        organs, total, shape = summarize_prediction(pred_path, organ_dice)
        meta["organs"] = organs
        meta["total_slices"] = total
        meta["volume_shape"] = shape
    else:
        meta["organs"] = merge_dice_into_organs(meta["organs"], organ_dice)
    return meta


def _meta_from_existing(job_dir: str, job_id: str, pred_path: str, ct_path: str) -> dict:
    organs, total, shape = summarize_prediction(pred_path)
    meta = {
        "file_hash": job_id,
        "ct_path": os.path.abspath(ct_path),
        "seg_path": os.path.abspath(pred_path),
        "total_slices": total,
        "volume_shape": shape,
        "organs": organs,
        "processing_time": 0,
        "cached": True,
    }
    _empty_gt_fields(meta)
    gt = os.path.join(job_dir, "gt.nii.gz")
    if os.path.exists(gt):
        _apply_ground_truth(job_dir, meta, gt)
    return meta


def _load_cached(job_dir: str, job_id: str, gt_src: str | None) -> dict | None:
    ct_path = ct_file(job_dir)
    pred_path = prediction_file(job_dir)
    if not ct_path or not pred_path:
        return None
    meta_path = os.path.join(job_dir, "meta.json")
    if os.path.exists(meta_path):
        meta = load_meta(job_dir)
    else:
        meta = _meta_from_existing(job_dir, job_id, pred_path, ct_path)
    meta["ct_path"] = os.path.abspath(ct_path)
    meta["seg_path"] = os.path.abspath(pred_path)
    meta["file_hash"] = job_id
    meta["cached"] = True
    if "processing_time" not in meta:
        meta["processing_time"] = 0
    gt_on_disk = os.path.join(job_dir, "gt.nii.gz")
    if gt_src:
        _apply_ground_truth(job_dir, meta, gt_src)
    elif meta.get("has_gt") and not os.path.exists(meta.get("gt_path") or "") and not os.path.exists(gt_on_disk):
        meta["organs"] = [
            {key: value for key, value in organ.items() if key != "dice"}
            for organ in (meta.get("organs") or [])
        ]
        _empty_gt_fields(meta)
    elif not meta.get("has_gt"):
        _empty_gt_fields(meta)
    save_meta(job_dir, meta)
    return meta


def _execute_model(job_dir: str, job_id: str, input_nifti_path: str, gt_src: str | None) -> dict:
    ct_path = os.path.join(job_dir, "ct.nii.gz")
    if not os.path.exists(ct_path):
        _stage_nifti(input_nifti_path, ct_path)

    nn_in = os.path.join(job_dir, "nn_in")
    nn_out = os.path.join(job_dir, "nn_out")
    shutil.rmtree(nn_in, ignore_errors=True)
    shutil.rmtree(nn_out, ignore_errors=True)
    os.makedirs(nn_in, exist_ok=True)
    os.makedirs(nn_out, exist_ok=True)
    staged_input = os.path.join(nn_in, f"case_{job_id}_0000.nii.gz")
    shutil.copy2(ct_path, staged_input)

    start = time.time()
    predict_from_folder(
        model=MODEL_BASE_PATH,
        input_folder=nn_in,
        output_folder=nn_out,
        folds=FOLDS,
        save_npz=False,
        num_threads_preprocessing=1,
        num_threads_nifti_save=1,
        lowres_segmentations=None,
        part_id=0,
        num_parts=1,
        tta=False,
        mixed_precision=True,
        overwrite_existing=True,
        mode="normal",
        overwrite_all_in_gpu=None,
        step_size=0.5,
        checkpoint_name=CHECKPOINT,
    )
    elapsed = time.time() - start

    outputs = [
        p for p in Path(nn_out).glob("*.nii.gz") if not p.name.endswith("_0000.nii.gz")
    ]
    if not outputs:
        raise RuntimeError("Model did not generate output segmentation file.")

    pred_path = os.path.join(job_dir, "pred.nii.gz")
    shutil.copy2(outputs[0], pred_path)
    organs, total, shape = summarize_prediction(pred_path)
    meta = {
        "file_hash": job_id,
        "ct_path": os.path.abspath(ct_path),
        "seg_path": os.path.abspath(pred_path),
        "total_slices": total,
        "volume_shape": shape,
        "organs": organs,
        "processing_time": round(elapsed, 2),
        "cached": False,
    }
    _empty_gt_fields(meta)
    if gt_src:
        _apply_ground_truth(job_dir, meta, gt_src)
    save_meta(job_dir, meta)
    shutil.rmtree(nn_in, ignore_errors=True)
    shutil.rmtree(nn_out, ignore_errors=True)
    return meta


def run_inference(input_nifti_path: str, gt_nifti_path: str | None = None) -> dict:
    """
    Segment a CT volume. job_id is sha256(file)[:16].
    A complete cache hit returns without calling the model.
    """
    file_hash = sha256_file(input_nifti_path)
    job_id = file_hash
    job_dir = os.path.join(TEMP_CACHE_DIR, job_id)
    os.makedirs(job_dir, exist_ok=True)

    with job_lock(job_id):
        cached = _load_cached(job_dir, job_id, gt_nifti_path)
        if cached is not None:
            # Touch the header so volume_shape is present even before the bin cache exists.
            if not cached.get("volume_shape"):
                try:
                    image = nib.load(cached["seg_path"])
                    cached["volume_shape"] = display_shape(image.shape)
                    cached["total_slices"] = int(image.shape[2])
                    save_meta(job_dir, cached)
                except Exception:
                    logger.exception("Could not read cached prediction header")
            logger.info("Cache hit for job %s", job_id)
            return _result_dict(job_id, cached)

        logger.info("Cache miss for job %s — running MedNeXt", job_id)
        meta = _execute_model(job_dir, job_id, input_nifti_path, gt_nifti_path)
        return _result_dict(job_id, meta)


def attach_ground_truth(job_id: str, gt_nifti_path: str) -> dict:
    """Score a ground-truth mask against a finished prediction. Does not rerun the model."""
    with job_lock(job_id):
        job_dir = resolve_job_dir(job_id)
        meta_path = os.path.join(job_dir, "meta.json")
        if not os.path.exists(meta_path):
            raise FileNotFoundError(f"Job {job_id} not found.")
        meta = load_meta(job_dir)
        _apply_ground_truth(job_dir, meta, gt_nifti_path)
        save_meta(job_dir, meta)
        return _result_dict(job_id, meta)


def get_volume_gzip(job_id: str, kind: str) -> tuple[bytes, list[int]]:
    if kind not in {"ct", "pred", "gt"}:
        raise ValueError("Volume kind must be ct, pred, or gt.")
    with job_lock(job_id):
        job_dir = resolve_job_dir(job_id)
        if not os.path.exists(os.path.join(job_dir, "meta.json")):
            raise FileNotFoundError(f"Job {job_id} not found.")
        meta = load_meta(job_dir)
        if kind == "gt" and not meta.get("has_gt"):
            raise FileNotFoundError("Ground truth is not available for this case.")
        shape = ensure_volume_files(job_dir, meta)
        path = os.path.join(job_dir, f"{kind}.bin.gz")
        if not os.path.exists(path):
            raise FileNotFoundError(f"{kind} volume is not available for this case.")
        with open(path, "rb") as handle:
            payload = handle.read()
        return payload, shape
