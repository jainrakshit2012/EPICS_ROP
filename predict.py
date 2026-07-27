import argparse
import os

import matplotlib.pyplot as plt
import torch
from PIL import Image

from config import Config
from dataset import get_transforms
from model import get_rop_model


def predict_single_image(img_path, checkpoint_path, img_size, show=True):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = get_rop_model(num_classes=2, pretrained=False)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)
    model.eval()

    transform = get_transforms("val", img_size)
    image = Image.open(img_path).convert("RGB")
    input_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(input_tensor)
        probs = torch.softmax(output, dim=1)

        prob_rop = probs[0][1].item()
        prob_normal = probs[0][0].item()

        _, predicted = output.max(1)
        class_name = "ROP" if predicted.item() == 1 else "Normal"
        confidence = prob_rop if predicted.item() == 1 else prob_normal

    print(f"Image: {img_path}")
    print(f"Result: {class_name} | Confidence: {confidence * 100:.2f}% | Probability of ROP: {prob_rop:.4f}")

    if show:
        plt.imshow(image)
        plt.title(f"Prediction: {class_name} ({confidence * 100:.2f}% confidence)")
        plt.axis("off")
        plt.show()

    return class_name, confidence


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("image", help="Path to the retinal image")
    parser.add_argument("--checkpoint", default=None, help="Path to model checkpoint (.pth)")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--no-show", action="store_true", help="Skip the matplotlib preview window")
    args = parser.parse_args()

    cfg = Config.from_yaml(args.config)
    checkpoint_path = args.checkpoint or os.path.join(cfg.output_dir, "best_rop_model.pth")

    predict_single_image(args.image, checkpoint_path, cfg.image_size, show=not args.no_show)
