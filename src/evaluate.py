"""Evaluate the trained checkpoint on the validation split."""

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import METRICS_DIR, PLOTS_DIR, VAL_DIR
from dataset import SignDataset, build_eval_transform
from inference import LoadedModel, load_trained_model
from plot_normalized_confusion import plot_normalized_confusion


@torch.no_grad()
def predict_dataset(loaded: LoadedModel, loader: DataLoader) -> tuple[np.ndarray, np.ndarray]:
    all_preds: list[int] = []
    all_targets: list[int] = []
    for images, targets in tqdm(loader, desc="evaluate"):
        outputs = loaded.model(images.to(loaded.device))
        all_preds.extend(outputs.argmax(dim=1).cpu().tolist())
        all_targets.extend(targets.tolist())
    return np.asarray(all_targets), np.asarray(all_preds)


def save_top_errors(targets: np.ndarray, preds: np.ndarray, idx_to_class: dict[int, str]) -> None:
    errors = pd.DataFrame(
        {
            "true_class": [idx_to_class[int(t)] for t in targets[targets != preds]],
            "predicted_class": [idx_to_class[int(p)] for p in preds[targets != preds]],
        }
    )
    top_errors = (
        errors.value_counts(["true_class", "predicted_class"]).reset_index(name="count")
        if not errors.empty
        else pd.DataFrame(columns=["true_class", "predicted_class", "count"])
    )
    top_errors.to_csv(METRICS_DIR / "top_errors.csv", index=False, encoding="utf-8")


def plot_confusion(cm: np.ndarray) -> None:
    plt.figure(figsize=(18, 18))
    plt.imshow(cm)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted class")
    plt.ylabel("True class")
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "confusion_matrix.png", dpi=150)
    plt.close()


def summarize_metrics(
    targets: np.ndarray, preds: np.ndarray, num_classes: int
) -> tuple[dict[str, float | int], dict[str, dict[str, float]]]:
    """Headline metrics plus a per-class report restricted to classes present in val.

    Classes absent from val have no support: sklearn scores them as F1 = 0, which
    silently drags macro F1 down, so they are excluded from the headline macro F1.
    """
    val_labels = sorted(set(targets.tolist()))
    report = classification_report(
        targets, preds, labels=val_labels, output_dict=True, zero_division=0
    )
    summary: dict[str, float | int] = {
        "num_classes": num_classes,
        "val_classes_present": len(val_labels),
        "val_accuracy": float(accuracy_score(targets, preds)),
        "macro_precision": float(report["macro avg"]["precision"]),
        "macro_recall": float(report["macro avg"]["recall"]),
        "macro_f1": float(report["macro avg"]["f1-score"]),
        "macro_f1_all_train_classes": float(
            f1_score(
                targets, preds, labels=list(range(num_classes)), average="macro", zero_division=0
            )
        ),
        "weighted_f1": float(report["weighted avg"]["f1-score"]),
        "total_val_samples": int(len(targets)),
        "total_errors": int((targets != preds).sum()),
    }
    return summary, report


def main() -> None:
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    loaded = load_trained_model()
    class_to_idx = {name: idx for idx, name in loaded.idx_to_class.items()}
    val_dataset = SignDataset(VAL_DIR, class_to_idx, build_eval_transform(loaded.img_size))
    loader = DataLoader(val_dataset, batch_size=256, num_workers=8, pin_memory=True)
    targets, preds = predict_dataset(loaded, loader)

    num_classes = len(class_to_idx)
    summary, report = summarize_metrics(targets, preds, num_classes)
    per_class = pd.DataFrame(report).transpose()
    per_class.index = [
        loaded.idx_to_class[int(label)] if label.isdigit() else label for label in per_class.index
    ]
    per_class.to_csv(METRICS_DIR / "classification_report.csv", encoding="utf-8")

    cm = confusion_matrix(targets, preds, labels=list(range(num_classes)))
    np.save(METRICS_DIR / "confusion_matrix.npy", cm)
    plot_confusion(cm)
    plot_normalized_confusion(cm, PLOTS_DIR / "confusion_matrix_normalized.png")
    save_top_errors(targets, preds, loaded.idx_to_class)

    report_json = {"model_name": loaded.model_name, **summary}
    with open(METRICS_DIR / "eval_summary.json", "w", encoding="utf-8") as file:
        json.dump(report_json, file, ensure_ascii=False, indent=2)
    print(json.dumps(report_json, indent=2))


if __name__ == "__main__":
    main()
