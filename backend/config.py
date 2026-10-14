import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_BASE_PATH = os.path.join(
    BASE_DIR, "model", "data", "nnUNet_trained_models",
    "nnUNet", "3d_fullres",
    "Task017_AbdominalOrganSegmentation",
    "nnUNetTrainerV2_MedNeXt_S_kernel3__nnUNetPlansv2.1_trgSp_1x1x1",
)

# nnUNet env vars — PHẢI set TRƯỚC KHI import nnunet_mednext
os.environ["nnUNet_raw_data_base"] = os.path.join(BASE_DIR, "model", "data", "nnUNet_raw_data_base")
os.environ["nnUNet_preprocessed"] = os.path.join(BASE_DIR, "model", "data", "nnUNet_preprocessed")
os.environ["RESULTS_FOLDER"] = os.path.join(BASE_DIR, "model", "data", "nnUNet_trained_models")

# Chỉ dùng fold_0 (demo mode)
FOLDS = [0]
CHECKPOINT = "model_best"

TEMP_INPUT_DIR = os.path.join(BASE_DIR, "backend", "temp", "input")
TEMP_OUTPUT_DIR = os.path.join(BASE_DIR, "backend", "temp", "output")

# Organ is "present" only when it occupies more than this many voxels.
PRESENT_VOXEL_MIN = 100

# BTCV Label Map — 14 classes (background + 13 organs)
LABEL_MAP = {
    0:  {"name": "Background",             "color": (0,   0,   0),   "hex": "#000000"},
    1:  {"name": "Spleen",                 "color": (76,  175, 80),  "hex": "#4CAF50"},
    2:  {"name": "Right Kidney",           "color": (126, 184, 247), "hex": "#7EB8F7"},
    3:  {"name": "Left Kidney",            "color": (91,  140, 255), "hex": "#5B8CFF"},
    4:  {"name": "Gallbladder",            "color": (255, 140, 66),  "hex": "#FF8C42"},
    5:  {"name": "Esophagus",              "color": (255, 107, 157), "hex": "#FF6B9D"},
    6:  {"name": "Liver",                  "color": (232, 212, 77),  "hex": "#E8D44D"},
    7:  {"name": "Stomach",                "color": (30,  144, 255), "hex": "#1E90FF"},
    8:  {"name": "Aorta",                  "color": (255, 68,  68),  "hex": "#FF4444"},
    9:  {"name": "Inferior Vena Cava",     "color": (255, 159, 67),  "hex": "#FF9F43"},
    10: {"name": "Portal & Splenic Veins", "color": (162, 155, 254), "hex": "#A29BFE"},
    11: {"name": "Pancreas",               "color": (181, 123, 255), "hex": "#B57BFF"},
    12: {"name": "Right Adrenal Gland",    "color": (0,   206, 201), "hex": "#00CEC9"},
    13: {"name": "Left Adrenal Gland",     "color": (85,  239, 196), "hex": "#55EFC4"},
}
