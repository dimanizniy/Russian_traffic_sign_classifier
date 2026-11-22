"""Loading a trained checkpoint and running single-image predictions."""

from dataclasses import dataclass
from pathlib import Path

import torch
from PIL import Image
from torch import nn

from config import MODEL_PATH
from dataset import build_eval_transform
from models import build_model_for_checkpoint

DEFAULT_IMG_SIZE = 64


@dataclass(frozen=True)
class LoadedModel:
    model: nn.Module
    device: torch.device
    idx_to_class: dict[int, str]
    img_size: int
    model_name: str


@dataclass(frozen=True)
class Prediction:
    class_name: str
    probability: float


def pick_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_trained_model(path: Path = MODEL_PATH) -> LoadedModel:
    device = pick_device()
    checkpoint = torch.load(path, map_location=device, weights_only=False)

    model = build_model_for_checkpoint(checkpoint["model_name"], checkpoint["num_classes"])
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device).eval()

    idx_to_class = {idx: name for name, idx in checkpoint["class_to_idx"].items()}
    return LoadedModel(
        model=model,
        device=device,
        idx_to_class=idx_to_class,
        img_size=checkpoint.get("img_size", DEFAULT_IMG_SIZE),
        model_name=checkpoint["model_name"],
    )


@torch.no_grad()
def predict_top_k(loaded: LoadedModel, image: Image.Image, top_k: int = 5) -> list[Prediction]:
    batch = build_eval_transform(loaded.img_size)(image.convert("RGB")).unsqueeze(0)
    probabilities = torch.softmax(loaded.model(batch.to(loaded.device)), dim=1)[0]
    top_probs, top_indices = torch.topk(probabilities, k=min(top_k, probabilities.numel()))
    return [
        Prediction(loaded.idx_to_class[int(idx)], float(prob))
        for prob, idx in zip(top_probs.cpu().tolist(), top_indices.cpu().tolist(), strict=True)
    ]
