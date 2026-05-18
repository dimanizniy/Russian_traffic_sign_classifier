import random
from pathlib import Path

import streamlit as st
import torch
from PIL import Image
from torchvision import transforms

from config import CHECKPOINT_DIR, IMG_SIZE, VAL_DIR, TRAIN_DIR, MODEL_NAME
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


@st.cache_resource
def load_model():
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

    return model, device, idx_to_class


def find_example_image(class_name):
    possible_dirs = [
        VAL_DIR / class_name,
        TRAIN_DIR / class_name,
    ]

    for class_dir in possible_dirs:
        if class_dir.exists():
            images = list(class_dir.glob("*.jpg"))
            if images:
                return random.choice(images)

    return None


@torch.no_grad()
def predict(image, model, device, idx_to_class, top_k=5):
    transform = get_transform()

    x = transform(image).unsqueeze(0).to(device)

    logits = model(x)
    probs = torch.softmax(logits, dim=1)[0]

    top_probs, top_indices = torch.topk(probs, k=top_k)

    results = []

    for prob, idx in zip(top_probs.cpu().numpy(), top_indices.cpu().numpy()):
        class_name = idx_to_class[int(idx)]
        results.append({
            "class": class_name,
            "probability": float(prob),
        })

    return results


def main():
    st.set_page_config(
        page_title="Traffic Sign Classifier",
        page_icon="🚦",
        layout="centered",
    )

    st.title("🚦 Классификатор российских дорожных знаков")
    st.write(
        "Загрузите изображение дорожного знака, и модель определит его класс."
    )

    model, device, idx_to_class = load_model()

    st.sidebar.header("Информация")
    st.sidebar.write(f"Устройство: `{device}`")
    st.sidebar.write(f"Количество классов: `{len(idx_to_class)}`")
    st.sidebar.write(f"Размер входа модели: `{IMG_SIZE}×{IMG_SIZE}`")

    uploaded_file = st.file_uploader(
        "Выберите изображение",
        type=["jpg", "jpeg", "png", "bmp", "webp"],
    )

    if uploaded_file is None:
        st.info("Загрузите изображение для классификации.")
        return

    image = Image.open(uploaded_file).convert("RGB")

    st.subheader("Загруженное изображение")
    st.image(image, use_container_width=True)

    results = predict(
        image=image,
        model=model,
        device=device,
        idx_to_class=idx_to_class,
        top_k=5,
    )

    best = results[0]
    best_class = best["class"]
    best_prob = best["probability"]

    st.subheader("Результат классификации")

    st.success(
        f"Модель считает, что это знак класса: **{best_class}** "
        f"с вероятностью **{best_prob:.4f}**"
    )

    example_path = find_example_image(best_class)

    if example_path is not None:
        st.subheader("Пример знака этого класса из датасета")
        example_image = Image.open(example_path).convert("RGB")
        st.image(
            example_image,
            caption=f"Эталонный пример класса {best_class}",
            width=250,
        )

    st.subheader("Top-5 предсказаний")

    for item in results:
        st.write(
            f"**{item['class']}** — {item['probability']:.4f}"
        )
        st.progress(min(item["probability"], 1.0))


if __name__ == "__main__":
    main()