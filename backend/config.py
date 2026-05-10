import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "best_weights" / "yolo26L_best.pt"
ULTRALYTICS_CONFIG_DIR = PROJECT_ROOT / ".cache" / "ultralytics"


SPEED_LIMIT_CLASSES = {
    "forb_speed_over_5": 5,
    "forb_speed_over_10": 10,
    "forb_speed_over_20": 20,
    "forb_speed_over_30": 30,
    "forb_speed_over_40": 40,
    "forb_speed_over_50": 50,
    "forb_speed_over_60": 60,
    "forb_speed_over_70": 70,
    "forb_speed_over_80": 80,
    "forb_speed_over_90": 90,
    "forb_speed_over_100": 100,
    "forb_speed_over_130": 130,
}


def get_confidence_threshold() -> float:
    raw_value = os.getenv("CONFIDENCE_THRESHOLD", "0.3")
    try:
        value = float(raw_value)
    except ValueError as exc:
        raise ValueError("CONFIDENCE_THRESHOLD должен быть числом") from exc

    if not 0 <= value <= 1:
        raise ValueError("CONFIDENCE_THRESHOLD должен быть в диапазоне от 0 до 1")

    return value


def get_required_confirmations() -> int:
    raw_value = os.getenv("REQUIRED_CONFIRMATIONS", "2")
    try:
        value = int(raw_value)
    except ValueError as exc:
        raise ValueError("REQUIRED_CONFIRMATIONS должен быть целым числом") from exc

    if value < 1:
        raise ValueError("REQUIRED_CONFIRMATIONS должен быть не меньше 1")

    return value


def get_min_confidence_for_state_update() -> float:
    raw_value = os.getenv("MIN_CONFIDENCE_FOR_STATE_UPDATE", "0.7")
    try:
        value = float(raw_value)
    except ValueError as exc:
        raise ValueError("MIN_CONFIDENCE_FOR_STATE_UPDATE должен быть числом") from exc

    if not 0 <= value <= 1:
        raise ValueError("MIN_CONFIDENCE_FOR_STATE_UPDATE должен быть в диапазоне от 0 до 1")

    return value
