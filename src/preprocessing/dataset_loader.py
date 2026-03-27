"""
MDIFDataset: A PyTorch Dataset class for loading the Multi-Domain Inconsistency Fusion (MDIF) dataset.

Dataset loader for the Multi-Domain Inconsistency Fusion (MDIF) framework.
Expects data in 'data/processed' standardized by the batch processor.
"""

import torch
import numpy as np
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import Compose


class MDIFDataset(Dataset):
    def __init__(
        self,
        root_dir: Path,
        transform: Compose | None = None,
    ):
        """
        Initialize the dataset.

        :param Path root_dir: Path to 'data/processed/train' or 'data/processed/test'
        :param Compose transform: torchvision transforms for the RGB image
        """

        self.root = Path(root_dir)
        self.transform = transform

        self.image_paths = sorted(list(self.root.glob("*.jpg")))

        if len(self.image_paths) == 0:
            raise RuntimeError(
                f"No images found in {root_dir}. Did you run the batch processor?"
            )

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx) -> dict:
        img_path = self.image_paths[idx]

        # Extract label and stem from filename (e.g., "2_73_forged.jpg" -> label=2, stem="73_forged")
        filename = img_path.name
        label = int(filename.split("_")[0])
        original_stem = "_".join(filename.split("_")[1:]).replace(".jpg", "")

        # Load RGB Image (Spatial Stream Input)
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        # Load Pre-computed Features (Spectral + Depth Stream Input)
        # Expected: "2_73_forged.npy" (201-dimensional vector)
        feat_path = img_path.with_suffix(".npy")
        if feat_path.exists():
            # Load the 201-dim vector (192 Spectral + 9 Depth)
            features = torch.from_numpy(np.load(feat_path)).float()
        else:
            # Fallback if .npy is missing
            features = torch.zeros(201)

        return {
            "image": image,  # For Stream A (Spatial CNN)
            "features": features,  # For Stream B & C (Spectral/Depth)
            "label": label,  # 0: Real, 1: Fake, 2: Inpainted
            "stem": original_stem,
        }
