import argparse
import json

import torch
from PIL import Image
from torchvision import transforms

from config import CHECKPOINT_DIR, IMG_SIZE, MODEL_NAME
from models import build_model


def get_transform():
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, help="Path to image")
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

    class_to_idx = checkpoint["class_to_idx"]
    idx_to_class = {v: k for k, v in class_to_idx.items()}

    image = Image.open(args.image).convert("RGB")
    x = get_transform()(image).unsqueeze(0).to(device)

    logits = model(x)
    probs = torch.softmax(logits, dim=1)[0]

    top_probs, top_indices = torch.topk(probs, k=args.top_k)

    print("Prediction results:")
    for prob, idx in zip(top_probs.cpu().numpy(), top_indices.cpu().numpy()):
        print(f"{idx_to_class[int(idx)]}: {float(prob):.4f}")


if __name__ == "__main__":
    main()