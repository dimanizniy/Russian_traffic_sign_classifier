"""Train a sign classifier with hyperparameters from ``params.yaml``."""

import json
import random
import time
from dataclasses import asdict, dataclass

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from class_balance import compute_class_weights
from config import METRICS_DIR, MODEL_PATH, PLOTS_DIR, TrainParams, load_train_params
from dataset import build_data
from inference import pick_device
from losses import build_criterion
from models import build_model

# Collation turns the int targets into a tensor, the declared item type stays (Tensor, int).
Batch = tuple[torch.Tensor, int]


@dataclass(frozen=True)
class EpochResult:
    loss: float
    accuracy: float
    macro_f1: float


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _summarise(total_loss: float, count: int, targets: list[int], preds: list[int]) -> EpochResult:
    return EpochResult(
        loss=total_loss / count,
        accuracy=float(accuracy_score(targets, preds)),
        macro_f1=float(f1_score(targets, preds, average="macro", zero_division=0)),
    )


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader[Batch],
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    device: torch.device,
) -> EpochResult:
    model.train()
    total_loss, count = 0.0, 0
    all_preds: list[int] = []
    all_targets: list[int] = []

    for images, targets in tqdm(loader, desc="train", leave=False):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
            outputs = model(images)
            loss = criterion(outputs, targets)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item() * images.size(0)
        count += images.size(0)
        all_preds.extend(outputs.argmax(dim=1).tolist())
        all_targets.extend(targets.tolist())

    return _summarise(total_loss, count, all_targets, all_preds)


@torch.no_grad()
def evaluate_epoch(
    model: nn.Module,
    loader: DataLoader[Batch],
    criterion: nn.Module,
    device: torch.device,
) -> EpochResult:
    model.eval()
    total_loss, count = 0.0, 0
    all_preds: list[int] = []
    all_targets: list[int] = []

    for images, targets in tqdm(loader, desc="val", leave=False):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
            outputs = model(images)
            loss = criterion(outputs, targets)

        total_loss += loss.item() * images.size(0)
        count += images.size(0)
        all_preds.extend(outputs.argmax(dim=1).tolist())
        all_targets.extend(targets.tolist())

    return _summarise(total_loss, count, all_targets, all_preds)


def plot_history(history: list[dict[str, float]]) -> None:
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    for metric, ylabel in (("loss", "Loss"), ("accuracy", "Accuracy"), ("macro_f1", "Macro F1")):
        plt.figure()
        plt.plot([row[f"train_{metric}"] for row in history], label=f"train {ylabel}")
        plt.plot([row[f"val_{metric}"] for row in history], label=f"val {ylabel}")
        plt.xlabel("Epoch")
        plt.ylabel(ylabel)
        plt.legend()
        plt.grid()
        plt.savefig(PLOTS_DIR / f"{metric}.png", dpi=150)
        plt.close()


def build_class_weights(
    class_counts: list[int], params: TrainParams, device: torch.device
) -> torch.Tensor | None:
    if params.class_weighting == "none":
        return None
    weights = compute_class_weights(
        class_counts, params.class_weighting, params.effective_number_beta
    )
    return torch.tensor(weights, dtype=torch.float32, device=device)


def main() -> None:
    params = load_train_params()
    set_seed(params.seed)
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    device = pick_device()
    data = build_data(params)
    num_classes = len(data.train_dataset.classes)
    print(f"Device: {device} | params: {asdict(params)}")
    print(
        f"Classes: {num_classes} | train: {len(data.train_dataset)} | val: {len(data.val_dataset)}"
    )

    model = build_model(params.model, num_classes).to(device)
    class_weights = build_class_weights(data.train_dataset.class_counts(), params, device)
    criterion = build_criterion(
        params.loss, class_weights, params.label_smoothing, params.focal_gamma
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=params.lr, weight_decay=params.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=params.epochs)
    scaler = torch.amp.GradScaler(enabled=device.type == "cuda")

    history: list[dict[str, float]] = []
    best_val_f1, best_epoch = -1.0, 0
    start_time = time.time()

    for epoch in range(1, params.epochs + 1):
        train_result = train_one_epoch(
            model, data.train_loader, criterion, optimizer, scaler, device
        )
        val_result = evaluate_epoch(model, data.val_loader, criterion, device)
        scheduler.step()

        row = {"epoch": epoch, "lr": optimizer.param_groups[0]["lr"]}
        row |= {f"train_{k}": v for k, v in asdict(train_result).items()}
        row |= {f"val_{k}": v for k, v in asdict(val_result).items()}
        history.append(row)
        print(
            f"epoch {epoch}/{params.epochs} "
            + " ".join(f"{k}={v:.4f}" for k, v in row.items() if k != "epoch")
        )

        if val_result.macro_f1 > best_val_f1:
            best_val_f1, best_epoch = val_result.macro_f1, epoch
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "model_name": params.model,
                    "num_classes": num_classes,
                    "img_size": params.img_size,
                    "class_to_idx": data.train_dataset.class_to_idx,
                    "params": asdict(params),
                    "epoch": epoch,
                    "val_f1": val_result.macro_f1,
                    "val_acc": val_result.accuracy,
                },
                MODEL_PATH,
            )

    training_time = time.time() - start_time
    with open(METRICS_DIR / "history.json", "w", encoding="utf-8") as file:
        json.dump(history, file, indent=2)
    with open(METRICS_DIR / "train_metrics.json", "w", encoding="utf-8") as file:
        json.dump(
            {
                "best_val_macro_f1": best_val_f1,
                "best_epoch": best_epoch,
                "training_time_seconds": round(training_time, 1),
            },
            file,
            indent=2,
        )
    plot_history(history)
    print(f"Best val macro F1 {best_val_f1:.4f} at epoch {best_epoch}, {training_time:.0f}s")


if __name__ == "__main__":
    main()
