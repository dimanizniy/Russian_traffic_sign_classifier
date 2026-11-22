"""Crop annotated signs out of RTSD frames into an ImageFolder layout."""

import json
import os
import shutil
import zipfile
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
from PIL import Image
from tqdm import tqdm

from config import DATA_DIR, RAW_ZIP_PATH, DataParams, load_data_params
from splitting import choose_val_sequences, group_into_sequences

FRAMES_PREFIX = "rtsd-frames/rtsd-frames/"
FRAMES_PER_TASK = 500
SOURCE_ANNOTATIONS = {"train": "train_anno.json", "val": "val_anno.json"}


@dataclass(frozen=True)
class SignBox:
    source: str
    annotation_id: int
    image_id: int
    category_id: int
    label: str
    bbox: tuple[float, float, float, float]


@dataclass(frozen=True)
class FrameTask:
    zip_member: str
    boxes: tuple[SignBox, ...]


def padded_crop_box(
    bbox: tuple[float, float, float, float],
    image_size: tuple[int, int],
    pad_ratio: float,
) -> tuple[int, int, int, int]:
    """Expand a COCO ``(x, y, w, h)`` box by ``pad_ratio`` and clip it to the image."""
    x, y, w, h = bbox
    img_w, img_h = image_size
    pad_x, pad_y = int(w * pad_ratio), int(h * pad_ratio)
    return (
        max(0, int(x - pad_x)),
        max(0, int(y - pad_y)),
        min(img_w, int(x + w + pad_x)),
        min(img_h, int(y + h + pad_y)),
    )


def _crop_frames(
    zip_path: Path, tasks: list[FrameTask], split_dir: Path, pad_ratio: float
) -> Counter[str]:
    counter: Counter[str] = Counter()
    with zipfile.ZipFile(zip_path) as archive:
        for task in tasks:
            with archive.open(task.zip_member) as img_file:
                image = Image.open(img_file).convert("RGB")
            for box in task.boxes:
                crop = image.crop(padded_crop_box(box.bbox, image.size, pad_ratio))
                # Annotation ids restart in each source json, the prefix keeps names unique.
                out_name = f"{box.source}_{box.annotation_id}_{box.image_id}_{box.category_id}.jpg"
                crop.save(split_dir / box.label / out_name, quality=95)
                counter[box.label] += 1
    return counter


def build_frame_tasks(
    source: str,
    annotations: dict[str, Any],
    id_to_label: dict[int, str],
    zip_members: set[str],
    min_size: int,
) -> tuple[list[FrameTask], int]:
    """Group sign boxes by source frame so that each frame is decoded only once."""
    file_names = {img["id"]: Path(img["file_name"]).name for img in annotations["images"]}
    boxes_by_member: dict[str, list[SignBox]] = defaultdict(list)
    skipped = 0

    for item in annotations["annotations"]:
        x, y, w, h = item["bbox"]
        member = FRAMES_PREFIX + file_names[item["image_id"]]
        if w < min_size or h < min_size or member not in zip_members:
            skipped += 1
            continue
        boxes_by_member[member].append(
            SignBox(
                source=source,
                annotation_id=item["id"],
                image_id=item["image_id"],
                category_id=item["category_id"],
                label=id_to_label.get(item["category_id"], str(item["category_id"])),
                bbox=(x, y, w, h),
            )
        )

    tasks = [FrameTask(member, tuple(boxes)) for member, boxes in boxes_by_member.items()]
    return tasks, skipped


def split_tasks(
    tasks_by_source: dict[str, list[FrameTask]], params: DataParams
) -> dict[str, list[FrameTask]]:
    if params.split == "official":
        return tasks_by_source
    if params.split != "sequence":
        raise ValueError(f"Unknown split '{params.split}', expected 'official' or 'sequence'")

    all_tasks = [task for tasks in tasks_by_source.values() for task in tasks]
    sequence_of = group_into_sequences(
        (Path(task.zip_member).name for task in all_tasks), params.sequence_gap_seconds
    )
    val_sequences = choose_val_sequences(sequence_of, params.val_fraction, params.split_seed)
    split: dict[str, list[FrameTask]] = {"train": [], "val": []}
    for task in all_tasks:
        in_val = sequence_of[Path(task.zip_member).name] in val_sequences
        split["val" if in_val else "train"].append(task)
    return split


def crop_split(tasks: list[FrameTask], split_name: str, pad_ratio: float) -> Counter[str]:
    split_dir = DATA_DIR / split_name
    for label in {box.label for task in tasks for box in task.boxes}:
        (split_dir / label).mkdir(parents=True, exist_ok=True)

    chunks = [tasks[i : i + FRAMES_PER_TASK] for i in range(0, len(tasks), FRAMES_PER_TASK)]
    counter: Counter[str] = Counter()
    with ProcessPoolExecutor(max_workers=os.cpu_count()) as pool:
        futures = [
            pool.submit(_crop_frames, RAW_ZIP_PATH, chunk, split_dir, pad_ratio) for chunk in chunks
        ]
        for future in tqdm(futures, desc=split_name):
            counter.update(future.result())
    return counter


def save_stats(train_counter: Counter[str], val_counter: Counter[str]) -> pd.DataFrame:
    classes = sorted(set(train_counter) | set(val_counter))
    stats = pd.DataFrame(
        {
            "class": classes,
            "train_count": [train_counter.get(c, 0) for c in classes],
            "val_count": [val_counter.get(c, 0) for c in classes],
        }
    )
    stats["total"] = stats["train_count"] + stats["val_count"]
    stats = stats.sort_values("total", ascending=False)
    stats.to_csv(DATA_DIR / "dataset_stats.csv", index=False, encoding="utf-8")
    return stats


def main() -> None:
    if not RAW_ZIP_PATH.exists():
        raise FileNotFoundError(f"Archive not found: {RAW_ZIP_PATH}, run `make data`")
    params = load_data_params()

    if DATA_DIR.exists():
        shutil.rmtree(DATA_DIR)
    DATA_DIR.mkdir(parents=True)

    with zipfile.ZipFile(RAW_ZIP_PATH) as archive:
        label_map: dict[str, int] = json.loads(archive.read("label_map.json").decode("utf-8"))
        id_to_label = {category_id: label for label, category_id in label_map.items()}
        members = set(archive.namelist())
        tasks_by_source: dict[str, list[FrameTask]] = {}
        skipped = 0
        for source, anno_name in SOURCE_ANNOTATIONS.items():
            annotations = json.loads(archive.read(anno_name).decode("utf-8"))
            tasks, source_skipped = build_frame_tasks(
                source, annotations, id_to_label, members, params.min_size
            )
            tasks_by_source[source] = tasks
            skipped += source_skipped

    with open(DATA_DIR / "label_map.json", "w", encoding="utf-8") as file:
        json.dump(label_map, file, ensure_ascii=False, indent=2)

    split = split_tasks(tasks_by_source, params)
    train_counter = crop_split(split["train"], "train", params.pad_ratio)
    val_counter = crop_split(split["val"], "val", params.pad_ratio)

    stats = save_stats(train_counter, val_counter)
    print(f"Split: {params.split}, skipped small/missing boxes: {skipped}")
    print(f"Train: {len(split['train'])} frames, {sum(train_counter.values())} crops")
    print(f"Val: {len(split['val'])} frames, {sum(val_counter.values())} crops")
    print(
        f"Classes: {len(stats)}, val-only: {int((stats['train_count'] == 0).sum())}, "
        f"train-only: {int((stats['val_count'] == 0).sum())}"
    )


if __name__ == "__main__":
    main()
