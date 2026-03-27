"""
This script trains the Spatial Stream (Stream A) backbone of the MDIF model using only the image data.
The trained weights will be saved and later loaded by the fusion training script.

The trained weights will be saved and later loaded to be used to train the fusion head.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import transforms
from pathlib import Path
from tqdm import tqdm
import numpy as np

from models.spatial_stream import SpatialStream
from src.preprocessing.dataset_loader import (
    MDIFDataset,
)

# MARK: Hyperparameters & Setup
BATCH_SIZE = 64
EPOCHS = 15
LEARNING_RATE = 1e-4

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROC_DATA_DIR = PROJECT_ROOT / "data" / "processed" / "train"
WEIGHTS_DIR = PROJECT_ROOT / "weights"
SPATIAL_WEIGHTS = WEIGHTS_DIR / "spatial_backbone_best.pth"

WEIGHTS_DIR.mkdir(exist_ok=True)

transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


def train():
    print(f"Starting Spatial Training on: {DEVICE}")

    # Dataset Initialization
    full_dataset = MDIFDataset(root_dir=PROC_DATA_DIR, transform=transform)

    # Split into Train/Val (80/20)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size]
    )

    # Handle Class Imbalance (Crucial for Objective #4)
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
        train_dataset, batch_size=BATCH_SIZE, sampler=train_sampler
    )
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = SpatialStream(num_classes=3).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    # Training Loop
    best_acc = 0.0
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0
        train_correct = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{EPOCHS}")
        for batch in pbar:
            images = batch["image"].to(DEVICE)
            labels = batch["label"].to(DEVICE)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_correct += predicted.eq(labels).sum().item()
            pbar.set_postfix({"Loss": f"{loss.item():.4f}"})

        scheduler.step()

        # Validation
        model.eval()
        val_correct = 0
        with torch.no_grad():
            for batch in val_loader:
                images = batch["image"].to(DEVICE)
                labels = batch["label"].to(DEVICE)
                outputs = model(images)
                _, predicted = outputs.max(1)
                val_correct += predicted.eq(labels).sum().item()

        val_acc = val_correct / val_size
        print(
            f"Epoch {epoch + 1}: Train Acc: {100.0 * train_correct / train_size:.2f}%, Val Acc: {100.0 * val_acc:.2f}%"
        )

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), SPATIAL_WEIGHTS)
            print(f"--- Best Model Saved (Acc: {100.0 * val_acc:.2f}%) ---")


if __name__ == "__main__":
    train()
