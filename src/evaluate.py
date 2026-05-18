import json
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm

from config import CHECKPOINT_DIR, METRICS_DIR, PLOTS_DIR, MODEL_NAME
from dataset import get_dataloaders
from models import build_model


@torch.no_grad()
def main():
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    _, val_loader, _, val_dataset = get_dataloaders()

    checkpoint = torch.load(
        CHECKPOINT_DIR / "best_model.pth",
        map_location=device,
        weights_only=False,
    )

    num_classes = checkpoint["num_classes"]
    model_name = checkpoint.get("model_name", MODEL_NAME)

    model = build_model(model_name, num_classes).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    all_preds = []
    all_targets = []

    for images, targets in tqdm(val_loader, desc="evaluate"):
        images = images.to(device)

        outputs = model(images)
        preds = outputs.argmax(dim=1).cpu().numpy()

        all_preds.extend(preds)
        all_targets.extend(targets.numpy())

    idx_to_class = {v: k for k, v in val_dataset.class_to_idx.items()}
    target_names = [idx_to_class[i] for i in range(num_classes)]

    labels = list(range(num_classes))

    report = classification_report(
        all_targets,
        all_preds,
        labels=labels,
        target_names=target_names,
        output_dict=True,
        zero_division=0,
    )

    report_df = pd.DataFrame(report).transpose()
    report_df.to_csv(METRICS_DIR / "classification_report.csv", encoding="utf-8")

    cm = confusion_matrix(all_targets, all_preds, labels=labels)
    np.save(METRICS_DIR / "confusion_matrix.npy", cm)

    plt.figure(figsize=(18, 18))
    plt.imshow(cm)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted class")
    plt.ylabel("True class")
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "confusion_matrix.png", dpi=200)
    plt.close()

    errors = []

    for true_idx, pred_idx in zip(all_targets, all_preds):
        if true_idx != pred_idx:
            errors.append({
                "true_class": idx_to_class[true_idx],
                "predicted_class": idx_to_class[pred_idx],
            })

    errors_df = pd.DataFrame(errors)

    if not errors_df.empty:
        top_errors = (
            errors_df
            .value_counts(["true_class", "predicted_class"])
            .reset_index(name="count")
            .sort_values("count", ascending=False)
        )
        top_errors.to_csv(METRICS_DIR / "top_errors.csv", index=False, encoding="utf-8")
    else:
        pd.DataFrame(columns=["true_class", "predicted_class", "count"]).to_csv(
            METRICS_DIR / "top_errors.csv",
            index=False,
            encoding="utf-8",
        )

    summary = {
        "model_name": model_name,
        "num_classes": num_classes,
        "val_accuracy": float(report["accuracy"]),
        "macro_precision": float(report["macro avg"]["precision"]),
        "macro_recall": float(report["macro avg"]["recall"]),
        "macro_f1": float(report["macro avg"]["f1-score"]),
        "weighted_f1": float(report["weighted avg"]["f1-score"]),
        "total_val_samples": len(all_targets),
        "total_errors": int(sum(np.array(all_targets) != np.array(all_preds))),
    }

    with open(METRICS_DIR / "eval_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\nEvaluation summary:")
    for k, v in summary.items():
        print(f"{k}: {v}")

    print("\nSaved:")
    print(METRICS_DIR / "classification_report.csv")
    print(METRICS_DIR / "top_errors.csv")
    print(METRICS_DIR / "eval_summary.json")
    print(PLOTS_DIR / "confusion_matrix.png")


if __name__ == "__main__":
    main()