import json
import random
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
from collections import Counter
from sklearn.metrics import accuracy_score, f1_score
from torch import nn
from tqdm import tqdm

from config import (
    CHECKPOINT_DIR,
    METRICS_DIR,
    PLOTS_DIR,
    MODEL_NAME,
    NUM_EPOCHS,
    LEARNING_RATE,
    SEED,
)
from dataset import get_dataloaders
from models import build_model


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()

    total_loss = 0.0
    all_preds = []
    all_targets = []

    for images, targets in tqdm(loader, desc="train", leave=False):
        images = images.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, targets)

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)

        preds = outputs.argmax(dim=1)
        all_preds.extend(preds.detach().cpu().numpy())
        all_targets.extend(targets.detach().cpu().numpy())

    avg_loss = total_loss / len(loader.dataset)
    acc = accuracy_score(all_targets, all_preds)
    f1 = f1_score(all_targets, all_preds, average="macro", zero_division=0)

    return avg_loss, acc, f1


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()

    total_loss = 0.0
    all_preds = []
    all_targets = []

    for images, targets in tqdm(loader, desc="val", leave=False):
        images = images.to(device)
        targets = targets.to(device)

        outputs = model(images)
        loss = criterion(outputs, targets)

        total_loss += loss.item() * images.size(0)

        preds = outputs.argmax(dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_targets.extend(targets.cpu().numpy())

    avg_loss = total_loss / len(loader.dataset)
    acc = accuracy_score(all_targets, all_preds)
    f1 = f1_score(all_targets, all_preds, average="macro", zero_division=0)

    return avg_loss, acc, f1


def plot_history(history):
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    plt.figure()
    plt.plot(history["train_loss"], label="train loss")
    plt.plot(history["val_loss"], label="val loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid()
    plt.savefig(PLOTS_DIR / "loss.png", dpi=200)
    plt.close()

    plt.figure()
    plt.plot(history["train_acc"], label="train accuracy")
    plt.plot(history["val_acc"], label="val accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid()
    plt.savefig(PLOTS_DIR / "accuracy.png", dpi=200)
    plt.close()

    plt.figure()
    plt.plot(history["train_f1"], label="train macro F1")
    plt.plot(history["val_f1"], label="val macro F1")
    plt.xlabel("Epoch")
    plt.ylabel("Macro F1")
    plt.legend()
    plt.grid()
    plt.savefig(PLOTS_DIR / "f1.png", dpi=200)
    plt.close()


def main():
    set_seed(SEED)

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    train_loader, val_loader, train_dataset, val_dataset = get_dataloaders()

    num_classes = len(train_dataset.classes)
    print("Classes:", num_classes)
    print("Train images:", len(train_dataset))
    print("Val images:", len(val_dataset))

    with open(METRICS_DIR / "class_to_idx.json", "w", encoding="utf-8") as f:
        json.dump(train_dataset.class_to_idx, f, ensure_ascii=False, indent=2)

    model = build_model(MODEL_NAME, num_classes).to(device)

    targets = [target for _, target in train_dataset.samples]
    class_counts = Counter(targets)

    class_weights = []
    for i in range(num_classes):
        class_weights.append(len(targets) / class_counts[i])

    class_weights = torch.tensor(
        class_weights,
        dtype=torch.float32,
        device=device,
    )

    class_weights = class_weights / class_weights.mean()

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)

    history = {
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": [],
        "train_f1": [],
        "val_f1": [],
    }

    best_val_f1 = 0.0
    start_time = time.time()

    for epoch in range(1, NUM_EPOCHS + 1):
        print(f"\nEpoch {epoch}/{NUM_EPOCHS}")

        train_loss, train_acc, train_f1 = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )

        val_loss, val_acc, val_f1 = evaluate(
            model, val_loader, criterion, device
        )

        scheduler.step()

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)
        history["train_f1"].append(train_f1)
        history["val_f1"].append(val_f1)

        print(
            f"train_loss={train_loss:.4f} "
            f"train_acc={train_acc:.4f} "
            f"train_f1={train_f1:.4f} | "
            f"val_loss={val_loss:.4f} "
            f"val_acc={val_acc:.4f} "
            f"val_f1={val_f1:.4f}"
        )

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "model_name": MODEL_NAME,
                    "num_classes": num_classes,
                    "class_to_idx": train_dataset.class_to_idx,
                    "val_f1": val_f1,
                    "val_acc": val_acc,
                },
                CHECKPOINT_DIR / "best_model.pth",
            )
            print("Saved best model.")

    total_time = time.time() - start_time

    history["training_time_seconds"] = total_time
    history["best_val_f1"] = best_val_f1

    with open(METRICS_DIR / "history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)

    plot_history(history)

    print("\nTraining finished.")
    print(f"Best val macro F1: {best_val_f1:.4f}")
    print(f"Training time: {total_time:.1f} sec")


if __name__ == "__main__":
    main()