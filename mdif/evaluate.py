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
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve, classification_report, precision_recall_curve, average_precision_score
from sklearn.preprocessing import label_binarize
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
        xticklabels=["Authentic", "Generated", "Inpainted"],
        yticklabels=["Authentic", "Generated", "Inpainted"],
    )
    plt.xlabel("Predicted", size=14)
    plt.ylabel("Actual", size=14)
    plt.title("MDIF Framework Confusion Matrix")
    plt.show()

    class_names = ["Authentic", "Generated", "Inpainted"]
    all_probs_arr = np.array(all_probs)
    all_labels_bin = np.array(label_binarize(all_labels, classes=[0, 1, 2]))

    plt.figure(figsize=(8, 6))
    for i, name in enumerate(class_names):
        fpr, tpr, _ = roc_curve(all_labels_bin[:, i], all_probs_arr[:, i])
        class_auc = roc_auc_score(all_labels_bin[:, i], all_probs_arr[:, i])
        plt.plot(fpr, tpr, label=f"{name} (AUC = {class_auc:.3f})")

    # plt.plot([0, 1], [0, 1], "k--", label="Chance")
    plt.xlabel("False Positive Rate", size=12)
    plt.ylabel("True Positive Rate", size=12)
    plt.title("MDIF Framework ROC Curves")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(8, 6))
    f1_values = [0.2, 0.4, 0.6, 0.8]
    for f1 in f1_values:
        r = np.linspace(0.01, 1.0, 500)
        p = f1 * r / (2 * r - f1)
        valid = (p > 0) & (p <= 1.0)
        plt.plot(r[valid], p[valid], color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
        idx = np.argmin(np.abs(r[valid] - p[valid]))
        plt.annotate(f"F1={f1}", xy=(r[valid][idx], p[valid][idx]), fontsize=7, color="gray", ha="center")

    for i, name in enumerate(class_names):
        precision, recall, _ = precision_recall_curve(all_labels_bin[:, i], all_probs_arr[:, i])
        ap = average_precision_score(all_labels_bin[:, i], all_probs_arr[:, i])
        plt.plot(recall, precision, label=f"{name} (AP = {ap:.3f})")

    plt.xlabel("Recall", size=12)
    plt.ylabel("Precision", size=12)
    plt.title("MDIF Framework Precision-Recall Curves")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    evaluate()
