"""
Evaluation Script for the MDIF Model

This script performs the following steps:
1. Loads the trained Spatial Stream and Fusion Classifier models.
2. Prepares the test dataset
3. Evaluates the models on the test set, computing predictions and probabilities.
4. Calculates overall accuracy and AUC-ROC, ensuring the target AUC-ROC > 0.93.
5. Generates a detailed classification report and confusion matrix visualization.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.metrics import confusion_matrix, roc_auc_score, classification_report
from tqdm import tqdm

from mdif.models.spatial_stream import SpatialStream
from mdif.models.fusion_classifier import MDIFFusionClassifier
from mdif.preprocessing.dataset_loader import MDIFDataset

# MARK: Setup
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
WEIGHTS_DIR = PROJECT_ROOT / "weights"

BASE_RAW = DATA_DIR / "raw"
BASE_PROC = DATA_DIR / "processed"
SPATIAL_WEIGHTS = WEIGHTS_DIR / "spatial_backbone_best.pth"
FUSION_WEIGHTS = WEIGHTS_DIR / "fusion_head_best.pth"


# MARK: Evaluation Function
def evaluate():
    test_path = BASE_PROC / "test"
    if not test_path.exists() or len(list(test_path.glob("*.npy"))) == 0:
        print("Test data not found. Did you run the preprocessing step?")

    spatial_model = SpatialStream(num_classes=3).to(DEVICE)
    spatial_model.load_state_dict(torch.load(SPATIAL_WEIGHTS, map_location=DEVICE))
    spatial_model.eval()

    fusion_model = MDIFFusionClassifier(input_dim=777).to(DEVICE)
    fusion_model.load_state_dict(torch.load(FUSION_WEIGHTS, map_location=DEVICE))
    fusion_model.eval()

    data_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    dataset = MDIFDataset(root_dir=test_path, transform=data_transform)
    loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=0)

    all_preds, all_labels, all_probs = [], [], []

    print(f"Evaluating {len(dataset)} images...")
    with torch.no_grad():
        for batch in tqdm(loader):
            imgs = batch["image"].to(DEVICE)
            math_feats = batch["features"].to(DEVICE)
            labels = batch["label"].to(DEVICE)

            spatial_feats = spatial_model(imgs, return_features=True)
            mdif_vector = torch.cat((spatial_feats, math_feats), dim=1)

            outputs = fusion_model(mdif_vector)
            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    acc = np.mean(np.array(all_preds) == np.array(all_labels))
    auc = roc_auc_score(all_labels, all_probs, multi_class="ovr")

    print("\n[RESULTS]")
    print(f"Overall Accuracy: {acc * 100:.2f}%")
    print(f"Target AUC-ROC (>0.93): {auc:.4f}")
    print("\nDetailed Classification Report:")
    print(
        classification_report(
            all_labels, all_preds, target_names=["Authentic", "Generated", "Inpainted"]
        )
    )

    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Auth", "Gen", "Inp"],
        yticklabels=["Auth", "Gen", "Inp"],
    )
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("MDIF Framework Confusion Matrix")
    plt.show()


if __name__ == "__main__":
    evaluate()
