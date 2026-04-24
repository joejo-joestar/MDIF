"""
Inference Script for the MDIF Model

This script allows users to input an image path and receive a prediction on whether the image is an authentic photograph, fully AI-generated, or partially AI-inpainted. It performs the following steps:
1. Loads the trained Spatial Stream and Fusion Classifier models.
2. Preprocesses the input image to extract both spatial and mathematical features.
3. Uses the MiDaS model to compute depth maps for depth feature extraction.
4. Combines the features and feeds them into the fusion classifier to get the final prediction and confidence score.
5. Provides a user-friendly command-line interface for continuous image analysis until the user decides to exit.
"""

import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Union, cast
from torchvision import transforms
from PIL import Image

from mdif.preprocessing.signal_proc import (
    extract_spectral_features,
    extract_depth_features,
)
from mdif.models.spatial_stream import SpatialStream
from mdif.models.fusion_classifier import MDIFFusionClassifier

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

PROJECT_ROOT = Path(__file__).resolve().parents[1]

WEIGHTS_DIR = PROJECT_ROOT / "weights"

SPATIAL_WEIGHTS = WEIGHTS_DIR / "spatial_backbone_best.pth"
FUSION_WEIGHTS = WEIGHTS_DIR / "fusion_head_best.pth"


def load_models():
    """Load all three models. Call once and cache the result (e.g. with st.cache_resource)."""
    midas = (
        cast(torch.nn.Module, torch.hub.load("intel-isl/MiDaS", "MiDaS_small", trust_repo=True)) # type: ignore
        .to(DEVICE)
        .eval()
    )

    spatial_model = SpatialStream(num_classes=3)
    spatial_model.load_state_dict(torch.load(SPATIAL_WEIGHTS, map_location=DEVICE))
    spatial_model.to(DEVICE)
    spatial_model.eval()

    fusion_model = MDIFFusionClassifier(input_dim=777)
    fusion_model.load_state_dict(torch.load(FUSION_WEIGHTS, map_location=DEVICE))
    fusion_model.to(DEVICE)
    fusion_model.eval()

    return midas, spatial_model, fusion_model


def _display_resized_image(img_res):
    """Display the resized (224x224) RGB image used for inference."""
    plt.figure(figsize=(4, 4))
    plt.imshow(img_res)
    plt.title("Resized Input (224x224)")
    plt.axis("off")
    plt.tight_layout()
    plt.show()


def analyze_image(
    img_source: Union[str, Path, Image.Image],
    models: tuple | None = None,
    show_resized: bool = False,
):
    """
    Analyze an image for forgery.

    Args:
        img_source: A file path (str/Path) or a PIL Image (e.g. from st.file_uploader).
        models: Optional tuple of (midas, spatial_model, fusion_model). If None, models
                are loaded fresh — pass pre-loaded models to avoid reloading on every call.
        show_resized: If True, display the 224x224 resized image (CLI use only).
    """
    if models is not None:
        midas, spatial_model, fusion_model = models
    else:
        midas, spatial_model, fusion_model = load_models()

    # region Image Preprocessing
    if isinstance(img_source, Image.Image):
        pil_img = img_source.convert("RGB")
        img_rgb = np.array(pil_img)
    else:
        image = Path(img_source)
        if not image.exists():
            return "File not found.", 0.0, {}
        print(f"Predicting for: {image.name}")
        img_bgr = cv2.imread(str(image))
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    img_res = cv2.resize(img_rgb, (224, 224))
    if show_resized:
        _display_resized_image(img_res)
    # endregion

    # region Extract Math Features (201-dim)
    # Spectral (Stream B)
    spec_feat = extract_spectral_features(img_res)

    # Depth (Stream C)
    img_t = (
        torch.from_numpy(img_res)
        .permute(2, 0, 1)
        .float()
        .unsqueeze(0)
        .to(DEVICE)
        / 255.0
    )
    with torch.no_grad():
        depth = midas(img_t).squeeze().cpu().numpy()
    depth_feat = extract_depth_features(img_res, depth)

    # Concatenate features (192 from spectral + 9 from depth = 201 total)
    feat_vector = np.concatenate([spec_feat, depth_feat])

    math_vector = (
        torch.from_numpy(feat_vector)
        .float()
        .unsqueeze(0)
        .to(DEVICE)
    )
    # endregion

    # region Extract Spatial Features (576-dim)
    data_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    # Apply the transform
    img_tensor = data_transform(Image.fromarray(img_res))

    # Explicitly cast to Tensor so the IDE recognizes 'unsqueeze'
    if not isinstance(img_tensor, torch.Tensor):
        img_tensor = torch.tensor(img_tensor)

    img_tensor = img_tensor.unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        spatial_feat = spatial_model(img_tensor, return_features=True)
    # endregion

    # region MDIF Fusion Decision
    with torch.no_grad():
        mdif_vector = torch.cat((spatial_feat, math_vector), dim=1)
        output = fusion_model(mdif_vector)
        probs = torch.softmax(output, dim=1)
        conf, pred = torch.max(probs, 1)
    # endregion

    classes = {
        0: "Authentic Photograph",
        1: "Fully AI-Generated",
        2: "Partially AI-Inpainted",
    }
    class_probabilities = {
        classes[i]: float(probs[0, i].item())
        for i in sorted(classes)
    }
    # Explicitly cast pred.item() to int
    predicted_class_id = int(pred.item())
    return classes[predicted_class_id], float(conf.item()), class_probabilities

def infer():
    print("--- MDIF Image Forgery Detector ---")
    while True:
        user_input = input("\nEnter image path (or 'exit'): ")
        if user_input.lower() == "exit":
            break

        try:
            print("\n")
            label, confidence, class_probabilities = analyze_image(
                user_input,
                show_resized=True,
            )
            print(f"\nResult: {label}")
            print(f"Confidence: {confidence * 100:.2f}%")
            print("\nScores:")
            for class_name, probability in class_probabilities.items():
                print(f"- {class_name}: {probability * 100:.2f}%")
        except Exception as e:
            print(f"Error: {e}")


if __name__ == "__main__":
    infer()
