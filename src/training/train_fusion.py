import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from models.spatial_stream import SpatialStream
from models.fusion_classifier import MDIFFusionClassifier
from src.preprocessing.dataset_loader import MDIFDataset


def train_fusion():
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the FROZEN Spatial Backbone
    spatial_model = SpatialStream(num_classes=3).to(DEVICE)
    spatial_model.load_state_dict(torch.load("weights/spatial_backbone_best.pth"))
    spatial_model.eval()  # Freeze weights and dropout

    # Initialize the Fusion Classifier
    fusion_model = MDIFFusionClassifier(input_dim=777).to(DEVICE)
    optimizer = optim.Adam(fusion_model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    # Loader (Use the same MDIFDataset)
    dataset = MDIFDataset(root_dir="data/processed/train")
    loader = DataLoader(dataset, batch_size=128, shuffle=True)

    # Fusion Training Loop
    for epoch in range(20):
        total_loss = 0
        for batch in loader:
            imgs = batch["image"].to(DEVICE)
            math_feats = batch["features"].to(DEVICE)  # The 201-dim vector
            labels = batch["label"].to(DEVICE)

            # Extract 576-dim Spatial Features
            with torch.no_grad():
                spatial_feats = spatial_model(imgs, return_features=True)

            # Concatenate to 777-dim
            # (Batch, 576) + (Batch, 201) -> (Batch, 777)
            mdif_vector = torch.cat((spatial_feats, math_feats), dim=1)

            # Train Fusion Head
            optimizer.zero_grad()
            outputs = fusion_model(mdif_vector)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch} Fusion Loss: {total_loss / len(loader):.4f}")

    torch.save(fusion_model.state_dict(), "weights/fusion_head_final.pth")
