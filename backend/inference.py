"""
QUAN TRỌNG: import config (set env vars) TRƯỚC khi import bất cứ thứ gì từ nnunet_mednext.
"""
import gzip
import json
import os
import shutil
import sys
import time
import uuid
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
    LABEL_MAP,
    MODEL_BASE_PATH,
    PRESENT_VOXEL_MIN,
    TEMP_INPUT_DIR,
    TEMP_OUTPUT_DIR,
)

from nnunet_mednext.inference.predict import predict_from_folder
import nibabel as nib
import numpy as np


def _stage_nifti(src: str, dest: str) -> None:
    """nnUNet chỉ nhận CASENAME_0000.nii.gz. File .nii thuần được nén gzip."""
    lower = src.lower()
    if lower.endswith(".nii.gz"):
        shutil.copy2(src, dest)
        return
    if lower.endswith(".nii"):
        with open(src, "rb") as raw, gzip.open(dest, "wb") as gz:
            shutil.copyfileobj(raw, gz)
        return
    raise ValueError("Only .nii and .nii.gz files are accepted.")


def run_inference(input_nifti_path: str) -> dict:
    """
    Args:
        input_nifti_path: đường dẫn tuyệt đối tới file .nii / .nii.gz do user upload
    Returns:
        dict với keys: job_id, seg_path, ct_path, processing_time, total_slices, organs_detected
    """
    job_id = str(uuid.uuid4())[:8]
    job_input_dir = os.path.join(TEMP_INPUT_DIR, job_id)
    job_output_dir = os.path.join(TEMP_OUTPUT_DIR, job_id)
    os.makedirs(job_input_dir, exist_ok=True)
    os.makedirs(job_output_dir, exist_ok=True)

    # nnUNet BẮT BUỘC: tên file phải theo format CASENAME_0000.nii.gz
    dest_path = os.path.join(job_input_dir, f"case_{job_id}_0000.nii.gz")
    _stage_nifti(input_nifti_path, dest_path)

    start = time.time()

    predict_from_folder(
        model=MODEL_BASE_PATH,
        input_folder=job_input_dir,
        output_folder=job_output_dir,
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

    output_files = list(Path(job_output_dir).glob("*.nii.gz"))
    if not output_files:
        raise RuntimeError("Model did not generate output segmentation file.")

    seg_path = str(output_files[0])
    seg_data = nib.load(seg_path).get_fdata().astype(np.int32)
    labels, counts = np.unique(seg_data, return_counts=True)
    count_by_label = {int(label): int(count) for label, count in zip(labels, counts)}

    organs_detected = []
    for label_id in range(1, 14):
        info = LABEL_MAP[label_id]
        organs_detected.append({
            "id": label_id,
            "name": info["name"],
            "color": info["hex"],
            "present": count_by_label.get(label_id, 0) > PRESENT_VOXEL_MIN,
        })

    meta = {
        "ct_path": dest_path,
        "seg_path": seg_path,
        "total_slices": int(seg_data.shape[2]),
        "organs": organs_detected,
    }
    with open(os.path.join(job_output_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f)

    return {
        "job_id": job_id,
        "seg_path": seg_path,
        "ct_path": dest_path,
        "processing_time": round(elapsed, 2),
        "total_slices": int(seg_data.shape[2]),
        "organs_detected": organs_detected,
    }
