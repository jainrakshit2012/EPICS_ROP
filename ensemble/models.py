import timm
import torch.nn as nn
from torchvision import models

MODEL_NAMES = ["inception_v3", "densenet201", "mobilenet", "xception"]


def build_model(name, pretrained=True):
    """
    Binary-classification (single sigmoid logit) variant of each base model,
    matching the paper's transfer-learning setup.
    """
    if name == "inception_v3":
        weights = models.Inception_V3_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.inception_v3(weights=weights, aux_logits=True, transform_input=False)
        model.aux_logits = False  # forward() then always returns just the main logits
        model.AuxLogits = None
        model.fc = nn.Linear(model.fc.in_features, 1)

    elif name == "densenet201":
        weights = models.DenseNet201_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.densenet201(weights=weights)
        model.classifier = nn.Linear(model.classifier.in_features, 1)

    elif name == "mobilenet":
        weights = models.MobileNet_V2_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.mobilenet_v2(weights=weights)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, 1)

    elif name == "xception":
        # Not in torchvision; pulled from timm, which builds the binary head directly.
        model = timm.create_model("xception", pretrained=pretrained, num_classes=1)

    else:
        raise ValueError(f"Unknown model: {name}")

    return model
