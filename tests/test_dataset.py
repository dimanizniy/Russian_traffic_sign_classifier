from pathlib import Path

import pytest
import torch

from dataset import (
    SignDataset,
    build_eval_transform,
    build_train_transform,
    get_class_to_idx,
)


def test_class_to_idx_is_sorted(image_folder: Path) -> None:
    assert get_class_to_idx(image_folder) == {"a": 0, "b": 1}


def test_dataset_counts_and_items(image_folder: Path) -> None:
    dataset = SignDataset(image_folder, {"a": 0, "b": 1}, build_eval_transform(32))
    image, target = dataset[0]
    assert len(dataset) == 4
    assert dataset.class_counts() == [3, 1]
    assert image.shape == (3, 32, 32)
    assert target == 0


def test_dataset_without_images_raises(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError):
        SignDataset(tmp_path, {"a": 0})


@pytest.mark.parametrize("augmentation", ["baseline", "light"])
def test_train_transform_keeps_shape(image_folder: Path, augmentation: str) -> None:
    dataset = SignDataset(image_folder, {"a": 0, "b": 1}, build_train_transform(48, augmentation))
    image, _ = dataset[1]
    assert isinstance(image, torch.Tensor)
    assert image.shape == (3, 48, 48)


def test_unknown_augmentation_raises() -> None:
    with pytest.raises(ValueError):
        build_train_transform(64, "heavy")
