"""ImageFolder-like dataset of cropped signs, transforms and data loaders."""

from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms

from class_balance import compute_sample_weights
from config import IMAGENET_MEAN, IMAGENET_STD, TRAIN_DIR, VAL_DIR, TrainParams


class SignDataset(Dataset[tuple[torch.Tensor, int]]):
    def __init__(
        self,
        root_dir: Path,
        class_to_idx: dict[str, int],
        transform: transforms.Compose | None = None,
    ) -> None:
        self.root_dir = Path(root_dir)
        self.class_to_idx = class_to_idx
        self.transform = transform
        self.samples: list[tuple[Path, int]] = []

        for class_name, class_idx in self.class_to_idx.items():
            class_dir = self.root_dir / class_name
            if not class_dir.exists():
                continue
            for img_path in sorted(class_dir.glob("*.jpg")):
                self.samples.append((img_path, class_idx))

        if not self.samples:
            raise RuntimeError(f"No images found in {self.root_dir}")

        self.classes = list(class_to_idx.keys())

    @property
    def targets(self) -> list[int]:
        return [target for _, target in self.samples]

    def class_counts(self) -> list[int]:
        counter = Counter(self.targets)
        return [counter.get(idx, 0) for idx in range(len(self.class_to_idx))]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        img_path, target = self.samples[index]
        image = Image.open(img_path).convert("RGB")
        if self.transform is None:
            return transforms.functional.to_tensor(image), target
        return self.transform(image), target


def build_eval_transform(img_size: int) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def _baseline_augmentations() -> list[torch.nn.Module]:
    """Augmentations of the first project version, kept for comparability."""
    return [
        transforms.RandomApply(
            [transforms.ColorJitter(brightness=0.35, contrast=0.35, saturation=0.25, hue=0.03)],
            p=0.8,
        ),
        transforms.RandomRotation(15),
        transforms.RandomAffine(degrees=0, translate=(0.08, 0.08), scale=(0.9, 1.1), shear=8),
        transforms.RandomPerspective(distortion_scale=0.15, p=0.3),
        transforms.RandomApply([transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.5))], p=0.25),
    ]


def _light_augmentations() -> list[torch.nn.Module]:
    """Milder geometry: crops are already tight, and rotation + shear + perspective
    stacked on 64px images pad black corners and blur the pictograms that tell
    similar classes apart. Flips are excluded on purpose: they change sign meaning.
    """
    return [
        transforms.RandomApply(
            [transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.02)],
            p=0.8,
        ),
        transforms.RandomApply(
            [transforms.RandomAffine(degrees=8, translate=(0.05, 0.05), scale=(0.92, 1.08))],
            p=0.6,
        ),
        transforms.RandomApply([transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0))], p=0.15),
    ]


AUGMENTATIONS = {
    "baseline": _baseline_augmentations,
    "light": _light_augmentations,
}


def build_train_transform(img_size: int, augmentation: str) -> transforms.Compose:
    if augmentation not in AUGMENTATIONS:
        raise ValueError(f"Unknown augmentation '{augmentation}', expected {list(AUGMENTATIONS)}")
    return transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            *AUGMENTATIONS[augmentation](),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def get_class_to_idx(train_dir: Path = TRAIN_DIR) -> dict[str, int]:
    classes = sorted(p.name for p in train_dir.iterdir() if p.is_dir())
    return {class_name: idx for idx, class_name in enumerate(classes)}


@dataclass(frozen=True)
class DataBundle:
    train_loader: DataLoader[tuple[torch.Tensor, int]]
    val_loader: DataLoader[tuple[torch.Tensor, int]]
    train_dataset: SignDataset
    val_dataset: SignDataset


def _build_sampler(dataset: SignDataset, sampler: str) -> WeightedRandomSampler | None:
    if sampler == "none":
        return None
    if sampler == "weighted":
        weights = compute_sample_weights(dataset.targets, dataset.class_counts())
        return WeightedRandomSampler(
            weights=weights.tolist(),
            num_samples=len(dataset),
            replacement=True,
        )
    raise ValueError(f"Unknown sampler: {sampler}")


def build_data(params: TrainParams) -> DataBundle:
    class_to_idx = get_class_to_idx()
    train_dataset = SignDataset(
        TRAIN_DIR, class_to_idx, build_train_transform(params.img_size, params.augmentation)
    )
    val_dataset = SignDataset(VAL_DIR, class_to_idx, build_eval_transform(params.img_size))

    sampler = _build_sampler(train_dataset, params.sampler)
    train_loader = DataLoader(
        train_dataset,
        batch_size=params.batch_size,
        shuffle=sampler is None,
        sampler=sampler,
        drop_last=True,
        num_workers=params.num_workers,
        pin_memory=True,
        persistent_workers=params.num_workers > 0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=params.batch_size,
        shuffle=False,
        num_workers=params.num_workers,
        pin_memory=True,
        persistent_workers=params.num_workers > 0,
    )
    return DataBundle(train_loader, val_loader, train_dataset, val_dataset)
