from pathlib import Path

import pytest
from PIL import Image


@pytest.fixture
def image_folder(tmp_path: Path) -> Path:
    """Tiny ImageFolder: class 'a' with 3 images, class 'b' with 1."""
    for class_name, count in (("a", 3), ("b", 1)):
        class_dir = tmp_path / class_name
        class_dir.mkdir()
        for i in range(count):
            Image.new("RGB", (20, 30), color=(i * 40, 10, 10)).save(class_dir / f"{i}.jpg")
    return tmp_path
