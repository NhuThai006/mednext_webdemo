"""
Render visualization PNG:
  - Bên trái: CT slice grayscale (windowed HU soft tissue)
  - Bên phải: CT slice + segmentation overlay màu theo BTCV label map
  - Bên dưới: legend tên + màu từng organ (chỉ hiện organ present=True)
"""
import io
import json
import os

import nibabel as nib
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from config import LABEL_MAP, TEMP_OUTPUT_DIR

_FONT_CANDIDATES = (
    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    r"C:\Windows\Fonts\arial.ttf",
    "arial.ttf",
)


def _load_meta(job_id: str) -> dict:
    meta_path = os.path.join(TEMP_OUTPUT_DIR, job_id, "meta.json")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"Job {job_id} not found.")
    with open(meta_path, encoding="utf-8") as f:
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
    """
    Render 1 slice axial thành PNG:
      [CT grayscale]  |  [Segmentation overlay]
      + legend màu organ ở dưới cùng
    """
    meta = _load_meta(job_id)
    ct_data, seg_data = _load_volumes(meta)
    canvas = _compose_slice(ct_data, seg_data, meta.get("organs", []), slice_idx, alpha)
    buf = io.BytesIO()
    canvas.save(buf, format="PNG")
    buf.seek(0)
    return buf.read()


def render_overview_png(job_id: str, n_slices: int = 6, alpha: float = 0.5) -> bytes:
    """
    Render n_slices đại diện (bỏ 10% đầu/cuối), stack dọc thành 1 ảnh PNG lớn.
    """
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
