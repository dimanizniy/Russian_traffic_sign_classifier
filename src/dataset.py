from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from config import TRAIN_DIR, VAL_DIR, IMG_SIZE, BATCH_SIZE, NUM_WORKERS


class SignDataset(Dataset):
    def __init__(self, root_dir, class_to_idx, transform=None):
        self.root_dir = Path(root_dir)
        self.class_to_idx = class_to_idx
        self.transform = transform
        self.samples = []

        for class_name, class_idx in self.class_to_idx.items():
            class_dir = self.root_dir / class_name

            if not class_dir.exists():
                continue

            for img_path in class_dir.glob("*.jpg"):
                self.samples.append((img_path, class_idx))

        if len(self.samples) == 0:
            raise RuntimeError(f"No images found in {self.root_dir}")

        self.classes = list(class_to_idx.keys())

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        img_path, target = self.samples[index]
        image = Image.open(img_path).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        return image, target


def get_transforms():
    train_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),

        transforms.RandomApply([
            transforms.ColorJitter(
                brightness=0.35,
                contrast=0.35,
                saturation=0.25,
                hue=0.03,
            )
        ], p=0.8),

        transforms.RandomRotation(15),

        transforms.RandomAffine(
            degrees=0,
            translate=(0.08, 0.08),
            scale=(0.9, 1.1),
            shear=8,
        ),

        transforms.RandomPerspective(
            distortion_scale=0.15,
            p=0.3,
        ),

        transforms.RandomApply([
            transforms.GaussianBlur(
                kernel_size=3,
                sigma=(0.1, 1.5),
            )
        ], p=0.25),

        transforms.ToTensor(),

        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    val_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    return train_transform, val_transform


def get_class_to_idx():
    classes = sorted([
        p.name for p in Path(TRAIN_DIR).iterdir()
        if p.is_dir()
    ])

    return {class_name: idx for idx, class_name in enumerate(classes)}


def get_dataloaders():
    train_transform, val_transform = get_transforms()
    class_to_idx = get_class_to_idx()

    train_dataset = SignDataset(
        root_dir=TRAIN_DIR,
        class_to_idx=class_to_idx,
        transform=train_transform,
    )

    val_dataset = SignDataset(
        root_dir=VAL_DIR,
        class_to_idx=class_to_idx,
        transform=val_transform,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    return train_loader, val_loader, train_dataset, val_dataset