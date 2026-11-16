"""Print top-k predictions for a single image."""

import argparse
from pathlib import Path

from PIL import Image

from config import MODEL_PATH
from inference import load_trained_model, predict_top_k


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, type=Path, help="Path to image")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--model", type=Path, default=MODEL_PATH)
    args = parser.parse_args()

    loaded = load_trained_model(args.model)
    print("Prediction results:")
    for prediction in predict_top_k(loaded, Image.open(args.image), args.top_k):
        print(f"{prediction.class_name}: {prediction.probability:.4f}")


if __name__ == "__main__":
    main()
