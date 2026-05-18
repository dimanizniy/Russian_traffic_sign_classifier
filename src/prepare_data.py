import json
import shutil
import zipfile
from collections import Counter
from pathlib import Path

import pandas as pd
from PIL import Image
from tqdm import tqdm


ZIP_PATH = Path("data/raw/rtsd-dataset.zip")
OUT_DIR = Path("data/processed")
MIN_SIZE = 12
PAD_RATIO = 0.08


def load_json_from_zip(z, name):
    return json.loads(z.read(name).decode("utf-8"))


def prepare_split(z, anno_name, split_name, id_to_label):
    print(f"\nPreparing {split_name} from {anno_name}")

    anno = load_json_from_zip(z, anno_name)

    images = {img["id"]: img for img in anno["images"]}
    annotations = anno["annotations"]

    split_dir = OUT_DIR / split_name
    split_dir.mkdir(parents=True, exist_ok=True)

    class_counter = Counter()
    skipped = 0

    for item in tqdm(annotations):
        image_id = item["image_id"]
        category_id = item["category_id"]
        x, y, w, h = item["bbox"]

        if w < MIN_SIZE or h < MIN_SIZE:
            skipped += 1
            continue

        img_info = images[image_id]
        file_name = img_info["file_name"]

        zip_img_path = f"rtsd-frames/rtsd-frames/{Path(file_name).name}"

        if zip_img_path not in z.namelist():
            skipped += 1
            continue

        label_name = id_to_label.get(category_id, str(category_id))
        class_dir = split_dir / label_name
        class_dir.mkdir(parents=True, exist_ok=True)

        with z.open(zip_img_path) as img_file:
            image = Image.open(img_file).convert("RGB")

            img_w, img_h = image.size

            pad_x = int(w * PAD_RATIO)
            pad_y = int(h * PAD_RATIO)

            left = max(0, int(x - pad_x))
            top = max(0, int(y - pad_y))
            right = min(img_w, int(x + w + pad_x))
            bottom = min(img_h, int(y + h + pad_y))

            crop = image.crop((left, top, right, bottom))

            out_name = f"{item['id']}_{image_id}_{category_id}.jpg"
            crop.save(class_dir / out_name, quality=95)

            class_counter[label_name] += 1

    return class_counter, skipped


def main():
    if not ZIP_PATH.exists():
        raise FileNotFoundError(f"Archive not found: {ZIP_PATH}")

    if OUT_DIR.exists():
        shutil.rmtree(OUT_DIR)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(ZIP_PATH, "r") as z:
        label_map = load_json_from_zip(z, "label_map.json")

        # label_map имеет вид: {"2_1": 1, "1_23": 2, ...}
        id_to_label = {v: k for k, v in label_map.items()}

        with open(OUT_DIR / "label_map.json", "w", encoding="utf-8") as f:
            json.dump(label_map, f, ensure_ascii=False, indent=2)

        train_counter, train_skipped = prepare_split(
            z=z,
            anno_name="train_anno.json",
            split_name="train",
            id_to_label=id_to_label,
        )

        val_counter, val_skipped = prepare_split(
            z=z,
            anno_name="val_anno.json",
            split_name="val",
            id_to_label=id_to_label,
        )

    all_classes = sorted(set(train_counter) | set(val_counter))

    stats = []
    for cls in all_classes:
        stats.append(
            {
                "class": cls,
                "train_count": train_counter.get(cls, 0),
                "val_count": val_counter.get(cls, 0),
                "total": train_counter.get(cls, 0) + val_counter.get(cls, 0),
            }
        )

    stats_df = pd.DataFrame(stats).sort_values("total", ascending=False)
    stats_df.to_csv(OUT_DIR / "dataset_stats.csv", index=False, encoding="utf-8")

    print("\nDone.")
    print(f"Train images: {sum(train_counter.values())}")
    print(f"Val images: {sum(val_counter.values())}")
    print(f"Skipped train objects: {train_skipped}")
    print(f"Skipped val objects: {val_skipped}")
    print(f"Classes: {len(all_classes)}")
    print(f"Stats saved to: {OUT_DIR / 'dataset_stats.csv'}")


if __name__ == "__main__":
    main()