"""Project paths and typed access to hyperparameters stored in ``params.yaml``."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

ROOT_DIR = Path(__file__).resolve().parents[1]
PARAMS_PATH = ROOT_DIR / "params.yaml"

RAW_DIR = ROOT_DIR / "data" / "raw"
RAW_ZIP_PATH = RAW_DIR / "rtsd-dataset.zip"
KAGGLE_DATASET = "watchman/rtsd-dataset"

DATA_DIR = ROOT_DIR / "data" / "processed"
TRAIN_DIR = DATA_DIR / "train"
VAL_DIR = DATA_DIR / "val"

MODELS_DIR = ROOT_DIR / "models"
MODEL_PATH = MODELS_DIR / "best_model.pth"

OUTPUT_DIR = ROOT_DIR / "outputs"
PLOTS_DIR = OUTPUT_DIR / "plots"
TRAIN_PLOTS_DIR = PLOTS_DIR / "train"
EVAL_PLOTS_DIR = PLOTS_DIR / "eval"
METRICS_DIR = OUTPUT_DIR / "metrics"

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass(frozen=True)
class DataParams:
    min_size: int
    pad_ratio: float


@dataclass(frozen=True)
class TrainParams:
    model: str
    img_size: int
    batch_size: int
    epochs: int
    lr: float
    weight_decay: float
    num_workers: int
    seed: int
    class_weighting: str
    effective_number_beta: float
    sampler: str
    loss: str
    focal_gamma: float
    label_smoothing: float
    augmentation: str


def _read_params(path: Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as file:
        loaded: dict[str, Any] = yaml.safe_load(file)
    return loaded


def load_data_params(path: Path = PARAMS_PATH) -> DataParams:
    return DataParams(**_read_params(path)["data"])


def load_train_params(path: Path = PARAMS_PATH) -> TrainParams:
    return TrainParams(**_read_params(path)["train"])
