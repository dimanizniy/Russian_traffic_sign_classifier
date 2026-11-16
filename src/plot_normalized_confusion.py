"""Row-normalised confusion matrix without the diagonal, to make confusions visible."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from config import METRICS_DIR, PLOTS_DIR


def normalize_without_diagonal(cm: np.ndarray) -> np.ndarray:
    row_sums = cm.sum(axis=1, keepdims=True)
    normalized = np.divide(cm, row_sums, out=np.zeros_like(cm, dtype=float), where=row_sums != 0)
    np.fill_diagonal(normalized, 0)
    return normalized


def plot_normalized_confusion(cm: np.ndarray, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(14, 12))
    plt.imshow(normalize_without_diagonal(cm), interpolation="nearest")
    plt.title("Normalized Confusion Matrix without diagonal")
    plt.xlabel("Predicted class")
    plt.ylabel("True class")
    plt.colorbar(label="Share of class samples")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def main() -> None:
    cm_path = METRICS_DIR / "confusion_matrix.npy"
    if not cm_path.exists():
        raise FileNotFoundError(f"{cm_path} not found, run `make evaluate` first")
    out_path = PLOTS_DIR / "confusion_matrix_normalized.png"
    plot_normalized_confusion(np.load(cm_path), out_path)
    print(f"Saved to: {out_path}")


if __name__ == "__main__":
    main()
