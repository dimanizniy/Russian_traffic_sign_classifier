"""Streamlit demo for the traffic sign classifier."""

import random
from pathlib import Path

import streamlit as st
from PIL import Image

from config import TRAIN_DIR, VAL_DIR
from inference import LoadedModel, load_trained_model, predict_top_k


@st.cache_resource
def load_model() -> LoadedModel:
    return load_trained_model()


def find_example_image(class_name: str) -> Path | None:
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


def main() -> None:
    st.set_page_config(
        page_title="Traffic Sign Classifier",
        page_icon="🚦",
        layout="centered",
    )

    st.title("🚦 Классификатор российских дорожных знаков")
    st.write("Загрузите изображение дорожного знака, и модель определит его класс.")

    loaded = load_model()

    st.sidebar.header("Информация")
    st.sidebar.write(f"Устройство: `{loaded.device}`")
    st.sidebar.write(f"Количество классов: `{len(loaded.idx_to_class)}`")
    st.sidebar.write(f"Размер входа модели: `{loaded.img_size}×{loaded.img_size}`")

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

    results = predict_top_k(loaded, image, top_k=5)

    best_class = results[0].class_name
    best_prob = results[0].probability

    st.subheader("Результат классификации")

    st.success(
        f"Модель считает, что это знак класса: **{best_class}** с вероятностью **{best_prob:.4f}**"
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
        st.write(f"**{item.class_name}** — {item.probability:.4f}")
        st.progress(min(item.probability, 1.0))


if __name__ == "__main__":
    main()
