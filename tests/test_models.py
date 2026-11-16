import pytest
import torch

from models import build_model, build_model_for_checkpoint


@pytest.mark.parametrize("model_name", ["custom_cnn", "residual_cnn"])
def test_models_output_logits_per_class(model_name: str) -> None:
    model = build_model(model_name, num_classes=7).eval()
    with torch.no_grad():
        logits = model(torch.randn(2, 3, 64, 64))
    assert logits.shape == (2, 7)


def test_resnet18_architecture_without_download() -> None:
    model = build_model_for_checkpoint("resnet18", num_classes=5).eval()
    with torch.no_grad():
        assert model(torch.randn(1, 3, 96, 96)).shape == (1, 5)


def test_unknown_model_raises() -> None:
    with pytest.raises(ValueError):
        build_model("vit_huge", 3)
