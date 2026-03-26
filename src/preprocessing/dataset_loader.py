import torch
import numpy as np
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset


class MDIFDataset(Dataset):
    """
    Dataset loader for the Multi-Domain Inconsistency Fusion (MDIF) framework.
    Expects data in 'data/processed' standardized by the batch processor.
    """

    def __init__(self, root_dir, transform=None, mask_dir=None):
        """
        Args:
            root_dir: Path to 'data/processed/train' or 'data/processed/test'
            transform: torchvision transforms for the RGB image
            mask_dir: Path to 'data/raw/AutoSplice/Mask' (optional, for F1-Score evaluation)
        """
        self.root = Path(root_dir)
        self.mask_dir = Path(mask_dir) if mask_dir else None
        self.transform = transform

        # Collect all processed image paths
        # Naming convention: {label}_{original_stem}.jpg
        self.image_paths = sorted(list(self.root.glob("*.jpg")))

        if len(self.image_paths) == 0:
            raise RuntimeError(
                f"No images found in {root_dir}. Did you run the batch processor?"
            )

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]

        # Extract label and stem from filename
        # Example: "2_73_forged.jpg" -> label=2, stem="73_forged"
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

        # Load Mask (Optional, for Inpainting Evaluation)
        mask = torch.zeros((1, 224, 224))
        if label == 2 and self.mask_dir:
            # Look for mask in AutoSplice/Mask using the original stem
            # AutoSplice masks are often .jpg or .png
            mask_file = self.mask_dir / f"{original_stem}.jpg"
            if not mask_file.exists():
                mask_file = self.mask_dir / f"{original_stem}.png"

            if mask_file.exists():
                mask_img = Image.open(mask_file).convert("L").resize((224, 224))
                mask = torch.from_numpy(np.array(mask_img)).unsqueeze(0) / 255.0
                mask = (mask > 0.5).float()  # Binarize

        return {
            "image": image,  # For Stream A (Spatial CNN)
            "features": features,  # For Stream B & C (Spectral/Depth)
            "label": label,  # 0: Real, 1: Fake, 2: Inpainted
            "mask": mask,  # For localization metrics
            "stem": original_stem,
        }
