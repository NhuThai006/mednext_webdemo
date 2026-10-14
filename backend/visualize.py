"""
PNG preview (legacy) and uint8 volume export for client-side canvas rendering.

Display orientation matches the original axial preview: each slice is np.rot90
of volume[:, :, z], then stacked as (D, H, W) = (Z, Y, X).
"""
import gzip
import io
import json
import logging
import os
from pathlib import Path

import nibabel as nib
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from config import LABEL_MAP, PRESENT_VOXEL_MIN, TEMP_CACHE_DIR, TEMP_OUTPUT_DIR

logger = logging.getLogger("uvicorn.error")

HU_LO = -200.0
HU_HI = 300.0

_FONT_CANDIDATES = (
    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    r"C:\Windows\Fonts\arial.ttf",
    "arial.ttf",
)


def resolve_job_dir(job_id: str) -> str:
    candidates = [
        os.path.join(TEMP_CACHE_DIR, job_id),
        os.path.join(TEMP_OUTPUT_DIR, job_id),
    ]
    for path in candidates:
        if os.path.exists(os.path.join(path, "meta.json")):
            return path
    for path in candidates:
        if os.path.isdir(path):
            return path
    raise FileNotFoundError(f"Job {job_id} not found.")


def rebind_meta_paths(job_dir: str, meta: dict) -> dict:
    """Prefer the stable filenames inside the job folder when they exist."""
    ct = os.path.join(job_dir, "ct.nii.gz")
    pred = os.path.join(job_dir, "pred.nii.gz")
    gt = os.path.join(job_dir, "gt.nii.gz")
    if os.path.exists(ct):
        meta["ct_path"] = os.path.abspath(ct)
    if os.path.exists(pred):
        meta["seg_path"] = os.path.abspath(pred)
    if os.path.exists(gt):
        meta["gt_path"] = os.path.abspath(gt)
    return meta


def load_meta(job_dir: str) -> dict:
    meta_path = os.path.join(job_dir, "meta.json")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"Job metadata not found in {job_dir}.")
    with open(meta_path, encoding="utf-8") as handle:
        meta = json.load(handle)
    return rebind_meta_paths(job_dir, meta)


def save_meta(job_dir: str, meta: dict) -> None:
    path = os.path.join(job_dir, "meta.json")
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(meta, handle, indent=2)
    os.replace(tmp, path)


def prediction_file(job_dir: str, meta: dict | None = None) -> str | None:
    pred = os.path.join(job_dir, "pred.nii.gz")
    if os.path.exists(pred):
        return pred
    if meta and meta.get("seg_path") and os.path.exists(meta["seg_path"]):
        return meta["seg_path"]
    matches = [
        p for p in Path(job_dir).glob("*.nii.gz")
        if p.name not in {"ct.nii.gz", "gt.nii.gz"} and not p.name.endswith("_0000.nii.gz")
    ]
    nn_out = Path(job_dir) / "nn_out"
    if nn_out.is_dir():
        matches.extend(
            p for p in nn_out.glob("*.nii.gz") if not p.name.endswith("_0000.nii.gz")
        )
    return str(matches[0]) if matches else None


def ct_file(job_dir: str, meta: dict | None = None) -> str | None:
    ct = os.path.join(job_dir, "ct.nii.gz")
    if os.path.exists(ct):
        return ct
    if meta and meta.get("ct_path") and os.path.exists(meta["ct_path"]):
        return meta["ct_path"]
    staged = list((Path(job_dir) / "nn_in").glob("*_0000.nii.gz"))
    return str(staged[0]) if staged else None


def display_shape(image_shape: tuple) -> list[int]:
    if len(image_shape) < 3:
        raise ValueError(f"Expected a 3D NIfTI volume, got shape {image_shape}.")
    x, y, z = int(image_shape[0]), int(image_shape[1]), int(image_shape[2])
    return [z, y, x]


def to_display_volume(data: np.ndarray) -> np.ndarray:
    """(X, Y, Z) -> (Z, Y, X), matching np.rot90 on each axial slice."""
    if data.ndim != 3:
        raise ValueError(f"Expected a 3D volume, got shape {data.shape}.")
    rotated = np.rot90(data, k=1, axes=(0, 1))
    return np.ascontiguousarray(np.moveaxis(rotated, -1, 0))


def window_ct_uint8(ct: np.ndarray) -> np.ndarray:
    """Soft-tissue window HU [-200, 300] mapped linearly to uint8."""
    lo = np.float32(HU_LO)
    hi = np.float32(HU_HI)
    scaled = (np.clip(ct, lo, hi) - lo) * (np.float32(255.0) / (hi - lo))
    return scaled.astype(np.uint8)


def labels_uint8(data: np.ndarray) -> np.ndarray:
    rounded = np.rint(np.asarray(data))
    return np.clip(rounded, 0, 13).astype(np.uint8)


def _squeezed_fdata(img) -> np.ndarray:
    data = np.squeeze(np.asarray(img.get_fdata(dtype=np.float32)))
    if data.ndim != 3:
        raise ValueError(f"Expected a 3D NIfTI volume, got shape {tuple(int(v) for v in data.shape)}.")
    return data


def align_gt_labels(pred_img, gt_img, pred_labels: np.ndarray | None = None) -> np.ndarray:
    """Nearest-neighbour resample onto the prediction grid when shapes differ."""
    gt_data = _squeezed_fdata(gt_img)
    ref_shape = pred_labels.shape if pred_labels is not None else _squeezed_fdata(pred_img).shape
    if gt_data.shape == ref_shape:
        return labels_uint8(gt_data)
    try:
        from nibabel.processing import resample_from_to

        resampled = resample_from_to(gt_img, pred_img, order=0)
        aligned = labels_uint8(_squeezed_fdata(resampled))
    except Exception as exc:
        raise ValueError(
            "Ground truth shape "
            f"{tuple(int(v) for v in gt_img.shape)} does not match prediction shape "
            f"{tuple(int(v) for v in pred_img.shape)} and could not be resampled onto the CT grid."
        ) from exc
    if aligned.shape != ref_shape:
        raise ValueError(
            f"Resampled ground truth shape {aligned.shape} does not match prediction shape {ref_shape}."
        )
    return aligned


def compute_dice(pred: np.ndarray, gt: np.ndarray) -> tuple[dict, float | None]:
    """
    Dice percentage for BTCV labels 1..13.
    Labels absent from both volumes are stored as 100 and excluded from the mean.
    """
    if pred.shape != gt.shape:
        raise ValueError(f"Shape mismatch between prediction {pred.shape} and ground truth {gt.shape}.")
    pred_f = np.ascontiguousarray(pred, dtype=np.uint8).ravel()
    gt_f = np.ascontiguousarray(gt, dtype=np.uint8).ravel()
    pred_counts = np.bincount(pred_f, minlength=14)
    gt_counts = np.bincount(gt_f, minlength=14)
    inter = np.bincount(pred_f[pred_f == gt_f], minlength=14)
    organ_dice: dict[str, float] = {}
    scores: list[float] = []
    for label in range(1, 14):
        denom = int(pred_counts[label] + gt_counts[label])
        if denom == 0:
            organ_dice[str(label)] = 100.0
            continue
        dsc = 200.0 * int(inter[label]) / denom
        organ_dice[str(label)] = round(dsc, 2)
        scores.append(dsc)
    mean = round(float(np.mean(scores)), 2) if scores else None
    return organ_dice, mean


def organ_inventory(pred: np.ndarray, organ_dice: dict | None = None) -> list:
    counts = np.bincount(np.ascontiguousarray(pred, dtype=np.uint8).ravel(), minlength=14)
    organs = []
    for label_id in range(1, 14):
        info = LABEL_MAP[label_id]
        item = {
            "id": label_id,
            "name": info["name"],
            "color": info["hex"],
            "present": int(counts[label_id]) > PRESENT_VOXEL_MIN,
        }
        if organ_dice and str(label_id) in organ_dice:
            item["dice"] = organ_dice[str(label_id)]
        organs.append(item)
    return organs


def merge_dice_into_organs(organs: list, organ_dice: dict) -> list:
    merged = []
    for organ in organs:
        item = dict(organ)
        key = str(organ["id"])
        if key in organ_dice:
            item["dice"] = organ_dice[key]
        merged.append(item)
    return merged


def summarize_prediction(pred_path: str, organ_dice: dict | None = None) -> tuple[list, int, list[int]]:
    pred_img = nib.load(pred_path)
    pred = labels_uint8(_squeezed_fdata(pred_img))
    return organ_inventory(pred, organ_dice), int(pred.shape[2]), display_shape(pred.shape)


def score_ground_truth(pred_path: str, gt_path: str) -> tuple[dict, float | None]:
    pred_img = nib.load(pred_path)
    pred = labels_uint8(_squeezed_fdata(pred_img))
    gt = align_gt_labels(pred_img, nib.load(gt_path), pred)
    return compute_dice(pred, gt)


def _resample_continuous(src_img, ref_img) -> np.ndarray:
    from nibabel.processing import resample_from_to

    resampled = resample_from_to(src_img, ref_img, order=1)
    return _squeezed_fdata(resampled)


def _write_uint8_gzip(path: str, arr: np.ndarray) -> None:
    raw = np.ascontiguousarray(arr, dtype=np.uint8).tobytes()
    tmp = path + ".tmp"
    with gzip.open(tmp, "wb", compresslevel=6) as handle:
        handle.write(raw)
    os.replace(tmp, path)


def ensure_volume_files(job_dir: str, meta: dict) -> list[int]:
    """Write ct.bin.gz, pred.bin.gz, and gt.bin.gz when missing. Returns [D, H, W]."""
    ct_gz = os.path.join(job_dir, "ct.bin.gz")
    pred_gz = os.path.join(job_dir, "pred.bin.gz")
    gt_gz = os.path.join(job_dir, "gt.bin.gz")
    gt_path = meta.get("gt_path")
    has_gt = bool(meta.get("has_gt") and gt_path and os.path.exists(gt_path))
    need_ct = not os.path.exists(ct_gz)
    need_pred = not os.path.exists(pred_gz)
    need_gt = has_gt and not os.path.exists(gt_gz)
    shape = meta.get("volume_shape")
    if shape and not need_ct and not need_pred and not need_gt:
        return [int(v) for v in shape]

    ct_path = ct_file(job_dir, meta)
    pred_path = prediction_file(job_dir, meta)
    if not ct_path:
        raise FileNotFoundError("CT volume for this case is missing from cache.")
    if not pred_path:
        raise FileNotFoundError("Prediction volume for this case is missing from cache.")
    meta["ct_path"] = os.path.abspath(ct_path)
    meta["seg_path"] = os.path.abspath(pred_path)

    ct_img = nib.load(ct_path)
    pred_img = nib.load(pred_path)
    pred = labels_uint8(_squeezed_fdata(pred_img))

    if need_ct or not shape:
        ct = _squeezed_fdata(ct_img)
        if ct.shape != pred.shape:
            ct = _resample_continuous(ct_img, pred_img)
        ct_u8 = window_ct_uint8(ct)
        del ct
        ct_disp = to_display_volume(ct_u8)
        del ct_u8
        _write_uint8_gzip(ct_gz, ct_disp)
        shape = [int(v) for v in ct_disp.shape]
        del ct_disp
        meta["volume_shape"] = shape

    if need_pred or not os.path.exists(pred_gz):
        pred_disp = to_display_volume(pred)
        _write_uint8_gzip(pred_gz, pred_disp)
        pred_shape = [int(v) for v in pred_disp.shape]
        if shape and pred_shape != [int(v) for v in shape]:
            raise ValueError(
                f"Prediction display shape {pred_shape} does not match CT display shape {shape}."
            )
        shape = pred_shape
        meta["volume_shape"] = shape
        del pred_disp

    if has_gt and (need_gt or not os.path.exists(gt_gz)):
        gt = align_gt_labels(pred_img, nib.load(gt_path), pred)
        gt_disp = to_display_volume(gt)
        _write_uint8_gzip(gt_gz, gt_disp)
        if gt_disp.shape[0] != shape[0] or gt_disp.shape[1] != shape[1] or gt_disp.shape[2] != shape[2]:
            raise ValueError("Ground truth display volume does not match the CT volume.")
        del gt, gt_disp

    if not has_gt and os.path.exists(gt_gz):
        os.remove(gt_gz)

    save_meta(job_dir, meta)
    if not shape:
        raise RuntimeError("Failed to determine volume shape.")
    logger.info("Volume cache ready at %s shape=%s", job_dir, shape)
    return [int(v) for v in shape]


def _load_meta(job_id: str) -> dict:
    return load_meta(resolve_job_dir(job_id))


def _window_ct(ct_slice: np.ndarray) -> np.ndarray:
    lo, hi = HU_LO, HU_HI
    clipped = np.clip(ct_slice, lo, hi)
    return ((clipped - lo) / (hi - lo) * 255).astype(np.uint8)


def _get_slice(volume: np.ndarray, idx: int) -> np.ndarray:
    slc = volume[:, :, idx]
    return np.rot90(slc)


def _fonts():
    for path in _FONT_CANDIDATES:
        try:
            return ImageFont.truetype(path, 13), ImageFont.truetype(path, 11)
        except Exception:
            continue
    fallback = ImageFont.load_default()
    return fallback, fallback


def _compose_slice(
    ct_data: np.ndarray,
    seg_data: np.ndarray,
    organs: list,
    slice_idx: int,
    alpha: float,
) -> Image.Image:
    total_z = ct_data.shape[2]
    slice_idx = max(0, min(slice_idx, total_z - 1))

    ct_slice = _get_slice(ct_data, slice_idx)
    seg_slice = _get_slice(seg_data, slice_idx)
    height, width = ct_slice.shape

    ct_gray = _window_ct(ct_slice)
    ct_rgb = np.stack([ct_gray] * 3, axis=-1).astype(np.float32)

    overlay = ct_rgb.copy()
    for label_id, info in LABEL_MAP.items():
        if label_id == 0:
            continue
        mask = seg_slice == label_id
        if not mask.any():
            continue
        color = np.array(info["color"], dtype=np.float32)
        for channel in range(3):
            overlay[:, :, channel] = np.where(
                mask,
                overlay[:, :, channel] * (1 - alpha) + color[channel] * alpha,
                overlay[:, :, channel],
            )
    overlay = np.clip(overlay, 0, 255).astype(np.uint8)

    present_organs = [organ for organ in organs if organ.get("present")]
    title_h = 24
    pad = 12
    box = 12
    item_w = 168
    canvas_w = width * 2 + pad * 3
    per_row = max(1, canvas_w // item_w)
    legend_rows = max(1, (len(present_organs) + per_row - 1) // per_row) if present_organs else 1
    legend_h = 16 + legend_rows * 20
    canvas_h = title_h + height + legend_h + pad * 2
    canvas = Image.new("RGB", (canvas_w, canvas_h), (15, 17, 21))

    canvas.paste(Image.fromarray(ct_rgb.astype(np.uint8)), (pad, title_h + pad))
    canvas.paste(Image.fromarray(overlay), (pad * 2 + width, title_h + pad))

    draw = ImageDraw.Draw(canvas)
    font_title, font_legend = _fonts()
    draw.text((pad + width // 2 - 28, 5), "CT Input", fill=(200, 200, 200), font=font_title)
    draw.text((pad * 2 + width + width // 2 - 35, 5), "Prediction", fill=(200, 200, 200), font=font_title)

    mid_x = pad + width + pad // 2
    draw.line([(mid_x, pad), (mid_x, title_h + height + pad)], fill=(50, 55, 65), width=1)

    legend_y = title_h + height + pad + 8
    for index, organ in enumerate(present_organs):
        col = index % per_row
        row = index // per_row
        x = pad + col * item_w
        y = legend_y + row * 20
        hex_str = organ["color"].lstrip("#")
        rgb = tuple(int(hex_str[j:j + 2], 16) for j in (0, 2, 4))
        draw.rectangle([x, y, x + box, y + box], fill=rgb, outline=(80, 85, 95))
        draw.text((x + box + 4, y), organ["name"], fill=(190, 195, 205), font=font_legend)

    return canvas


def _load_volumes(meta: dict):
    ct_data = nib.load(meta["ct_path"]).get_fdata()
    seg_data = nib.load(meta["seg_path"]).get_fdata().astype(np.int32)
    return ct_data, seg_data


def render_slice_png(job_id: str, slice_idx: int, alpha: float = 0.5) -> bytes:
    meta = _load_meta(job_id)
    ct_data, seg_data = _load_volumes(meta)
    canvas = _compose_slice(ct_data, seg_data, meta.get("organs", []), slice_idx, alpha)
    buf = io.BytesIO()
    canvas.save(buf, format="PNG")
    buf.seek(0)
    return buf.read()


def render_overview_png(job_id: str, n_slices: int = 6, alpha: float = 0.5) -> bytes:
    meta = _load_meta(job_id)
    ct_data, seg_data = _load_volumes(meta)
    total_z = int(meta["total_slices"])
    start_z = int(total_z * 0.10)
    end_z = max(start_z, int(total_z * 0.90) - 1)
    indices = np.linspace(start_z, end_z, n_slices, dtype=int).tolist()
    organs = meta.get("organs", [])

    panels = [_compose_slice(ct_data, seg_data, organs, idx, alpha) for idx in indices]
    max_w = max(panel.width for panel in panels)
    total_h = sum(panel.height for panel in panels)

    full = Image.new("RGB", (max_w, total_h), (15, 17, 21))
    y_off = 0
    for panel in panels:
        full.paste(panel, (0, y_off))
        y_off += panel.height

    buf = io.BytesIO()
    full.save(buf, format="PNG")
    buf.seek(0)
    return buf.read()
