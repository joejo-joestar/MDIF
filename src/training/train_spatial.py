import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import transforms
from pathlib import Path
from tqdm import tqdm
import numpy as np

# Internal imports
from models.spatial_stream import SpatialStream
from src.preprocessing.dataset_loader import (
    MDIFDataset,
)  # Ensure this points to processed data

# Hyperparameters & Setup
BATCH_SIZE = 64
EPOCHS = 15
LEARNING_RATE = 1e-4  # Low LR for fine-tuning
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PROC_DATA_DIR = Path("data/processed/train")
WEIGHTS_DIR = Path("weights")
WEIGHTS_DIR.mkdir(exist_ok=True)

# Data Augmentation (Standardization for MobileNetV3)
transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


def train():
    # Dataset Initialization
    # We use a simplified version of MDIFDataset that only loads .jpg from processed/
    full_dataset = MDIFDataset(root_dir=PROC_DATA_DIR, transform=transform)

    # Split into Train/Val (80/20)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size]
    )

    # Handle Class Imbalance (Crucial for Objective #4)
    # Calculate weights: 1 / frequency of class
    all_labels = [int(p.name.split("_")[0]) for p in full_dataset.image_paths]
    class_counts = np.bincount(all_labels)
    # Calculate weight per class: 1 / frequency
    class_weights = 1.0 / torch.tensor(class_counts, dtype=torch.float)

    # Map class weights to EVERY sample in the dataset as plain floats
    # Use .item() to convert 0-dim Tensor to float to solve the Type Error
    all_sample_weights = [class_weights[label].item() for label in all_labels]

    # Filter weights to only include those in the train_dataset indices
    # This solves the "sample_weights assigned but never used" error
    train_sample_weights = [all_sample_weights[i] for i in train_dataset.indices]

    train_sampler = WeightedRandomSampler(
        weights=train_sample_weights,  # This is now a list[float]
        num_samples=len(train_sample_weights),
        replacement=True,
    )

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, sampler=train_sampler
    )
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Model, Loss, Optimizer
    model = SpatialStream(num_classes=3).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    # Training Loop
    best_acc = 0.0
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0
        correct = 0

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
            correct += predicted.eq(labels).sum().item()
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
            f"Epoch {epoch + 1}: Train Acc: {100.0 * correct / train_size:.2f}%, Val Acc: {100.0 * val_acc:.2f}%"
        )

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), WEIGHTS_DIR / "spatial_backbone_best.pth")
            print("--- Best Model Saved ---")


if __name__ == "__main__":
    train()
