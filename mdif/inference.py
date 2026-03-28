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
from pathlib import Path
from typing import cast
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


midas = cast(torch.nn.Module, torch.hub.load("intel-isl/MiDaS", "MiDaS_small"))
midas.to(DEVICE)
midas.eval()

spatial_model = SpatialStream(num_classes=3)
spatial_model.load_state_dict(torch.load(SPATIAL_WEIGHTS, map_location=DEVICE))
spatial_model.to(DEVICE)
spatial_model.eval()

fusion_model = MDIFFusionClassifier(input_dim=777)
fusion_model.load_state_dict(torch.load(FUSION_WEIGHTS, map_location=DEVICE))
fusion_model.to(DEVICE)
fusion_model.eval()


def analyze_image(img_path):
    p = Path(img_path)
    if not p.exists():
        return "File not found."

    # 1. Image Preprocessing
    img_bgr = cv2.imread(str(p))
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_res = cv2.resize(img_rgb, (224, 224))

    # 2. Extract Math Features (201-dim)
    # Spectral (Includes the np.log1p scaling from signal_proc.py)
    spec_feat = extract_spectral_features(img_res)

    # Depth (Stream C)
    img_t = (
        torch.from_numpy(img_res).permute(2, 0, 1).float().unsqueeze(0).to(DEVICE)
        / 255.0
    )
    with torch.no_grad():
        depth = midas(img_t).squeeze().cpu().numpy()
    depth_feat = extract_depth_features(img_res, depth)

    math_vector = (
        torch.from_numpy(np.concatenate([spec_feat, depth_feat]))
        .float()
        .unsqueeze(0)
        .to(DEVICE)
    )

    # 3. Extract Spatial Features (576-dim)
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

    # 4. MDIF Fusion Decision
    with torch.no_grad():
        mdif_vector = torch.cat((spatial_feat, math_vector), dim=1)
        output = fusion_model(mdif_vector)
        probs = torch.softmax(output, dim=1)
        conf, pred = torch.max(probs, 1)

    classes = {
        0: "Authentic Photograph",
        1: "Fully AI-Generated",
        2: "Partially AI-Inpainted",
    }
    # Explicitly cast pred.item() to int
    predicted_class_id = int(pred.item())
    return classes[predicted_class_id], float(conf.item())


if __name__ == "__main__":
    print("--- MDIF Image Forgery Detector ---")
    while True:
        user_input = input("\nEnter image path (or 'exit'): ")
        if user_input.lower() == "exit":
            break

        try:
            label, confidence = analyze_image(user_input)
            print(f"\nResult: {label}")
            print(f"Confidence: {confidence * 100:.2f}%")
        except Exception as e:
            print(f"Error: {e}")
