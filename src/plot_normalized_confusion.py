import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


CM_PATH = Path("outputs/metrics/confusion_matrix.npy")
OUT_PATH = Path("outputs/plots/confusion_matrix_normalized.png")


def main():
    if not CM_PATH.exists():
        raise FileNotFoundError(
            f"Не найден файл {CM_PATH}. "
            f"Сначала запусти: python src/evaluate.py"
        )

    cm = np.load(CM_PATH)

    row_sums = cm.sum(axis=1, keepdims=True)

    cm_normalized = np.divide(
        cm,
        row_sums,
        out=np.zeros_like(cm, dtype=float),
        where=row_sums != 0,
    )

    np.fill_diagonal(cm_normalized, 0)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(14, 12))
    plt.imshow(cm_normalized, interpolation="nearest")
    plt.title("Normalized Confusion Matrix without diagonal")
    plt.xlabel("Predicted class")
    plt.ylabel("True class")
    plt.colorbar(label="Share of class samples")

    plt.tight_layout()
    plt.savefig(OUT_PATH, dpi=300)
    plt.close()

    print(f"Saved to: {OUT_PATH}")


if __name__ == "__main__":
    main()