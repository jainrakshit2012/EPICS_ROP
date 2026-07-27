import torch

from model import get_rop_model


def test_model_output_shape():
    model = get_rop_model(num_classes=2, pretrained=False)
    model.eval()

    dummy_input = torch.randn(1, 3, 64, 64)
    with torch.no_grad():
        output = model(dummy_input)

    assert output.shape == (1, 2)
