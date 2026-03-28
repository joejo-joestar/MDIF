"""
This script trains the fusion head of the MDIF model, which takes the concatenated features from the spatial backbone and mathematical features to perform final classification.
The spatial backbone is frozen during this training, and only the fusion head's weights are updated.

The trained weights will be saved and later loaded for evaluation.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import transforms
from pathlib import Path
from tqdm import tqdm
import numpy as np
import gc

from mdif.models.spatial_stream import SpatialStream
from mdif.models.fusion_classifier import MDIFFusionClassifier
from mdif.preprocessing.dataset_loader import MDIFDataset

# MARK: Hyperparameters & Setup
BATCH_SIZE = 64
EPOCHS = 20
LEARNING_RATE = 1e-3
HIDDEN_DIM = 256

# 576 (Spatial) + 192 (Spectral) + 9 (Depth)
INPUT_DIM = 777

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
WEIGHTS_DIR = PROJECT_ROOT / "weights"

PROC_DATA_DIR = DATA_DIR / "processed" / "train"
SPATIAL_WEIGHTS = WEIGHTS_DIR / "spatial_backbone_best.pth"
FUSION_WEIGHTS = WEIGHTS_DIR / "fusion_head_best.pth"

transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


def train():
    print(f"Starting Fusion Training on: {DEVICE}")

    # Dataset Initialization
    full_dataset = MDIFDataset(root_dir=PROC_DATA_DIR, transform=transform)

    # Split into Train/Val (80/20)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size]
    )

    # Handle Class Imbalance
    all_labels = [int(p.name.split("_")[0]) for p in full_dataset.image_paths]
    class_counts = np.bincount(all_labels)
    class_weights = 1.0 / torch.tensor(class_counts, dtype=torch.float)
    all_sample_weights = [class_weights[label].item() for label in all_labels]
    train_sample_weights = [all_sample_weights[i] for i in train_dataset.indices]

    train_sampler = WeightedRandomSampler(
        weights=train_sample_weights,
        num_samples=len(train_sample_weights),
        replacement=True,
    )

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, sampler=train_sampler, num_workers=0
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0
    )

    spatial_model = SpatialStream(num_classes=3).to(DEVICE)
    if not SPATIAL_WEIGHTS.exists():
        raise FileNotFoundError(
            f"Spatial weights not found at {SPATIAL_WEIGHTS}. Did you train the spatial model yet?"
        )
    spatial_model.load_state_dict(torch.load(SPATIAL_WEIGHTS, map_location=DEVICE))
    spatial_model.eval()

    # Initialize Fusion Classifier
    model = MDIFFusionClassifier(input_dim=INPUT_DIM, hidden_dim=HIDDEN_DIM).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # MARK: Training Loop
    best_acc = 0.0
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0
        train_correct = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{EPOCHS} [Train]")
        for batch in pbar:
            images = batch["image"].to(DEVICE)
            math_feats = batch["features"].to(DEVICE)
            labels = batch["label"].to(DEVICE)

            # Feature Extraction from Spatial Backbone (Frozen)
            with torch.no_grad():
                spatial_feats = spatial_model(images, return_features=True)

            # MDIF Vector Concatenation
            mdif_vector = torch.cat((spatial_feats, math_feats), dim=1)

            optimizer.zero_grad()
            outputs = model(mdif_vector)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_correct += predicted.eq(labels).sum().item()
            pbar.set_postfix({"Loss": f"{loss.item():.4f}"})

        # MARK: Validation
        model.eval()
        val_loss, val_correct = 0.0, 0
        with torch.no_grad():
            for batch in val_loader:
                images = batch["image"].to(DEVICE)
                math_feats = batch["features"].to(DEVICE)
                labels = batch["label"].to(DEVICE)

                spatial_feats = spatial_model(images, return_features=True)
                mdif_vector = torch.cat((spatial_feats, math_feats), dim=1)

                outputs = model(mdif_vector)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_correct += predicted.eq(labels).sum().item()

        train_acc = 100.0 * train_correct / train_size
        val_acc = 100.0 * val_correct / val_size
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)

        print(
            f"Epoch {epoch + 1}: T-Loss: {avg_train_loss:.4f}, T-Acc: {train_acc:.2f}%, "
            f"V-Loss: {avg_val_loss:.4f}, V-Acc: {val_acc:.2f}%"
        )

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), FUSION_WEIGHTS)
            print(f"--- Best Model Saved (Acc: {val_acc:.2f}%) ---")

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    train()
