"""
This script processes the raw datasets (CIFAKE, AutoSplice, CocoGlide) to extract and save features for training/testing.
For each image:
1. Load and resize to 224x224.
2. Extract spectral features (192-dim).
3. Extract depth features using MiDaS (9-dim).
4. Concatenate features and save as .npy.
Naming convention for saved files: {label}_{original_filename}.npy
Example: 1_0001.npy, 1_0001.jpg
Labels:
- 0: Real (CIFAKE REAL, AutoSplice Authentic, CocoGlide real)
- 1: Fake (CIFAKE FAKE)
- 2: Forged/Inpainted (AutoSplice Forged_JPEG90, CocoGlide fake)
"""

import cv2
import torch
import numpy as np
from typing import cast
from pathlib import Path
from tqdm import tqdm
from src.preprocessing.signal_proc import (
    extract_spectral_features,
    extract_depth_features,
)

# Global: Load MiDaS and move to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
midas = (
    cast(torch.nn.Module, torch.hub.load("intel-isl/MiDaS", "MiDaS_small"))
    .to(device)
    .eval()
)


def process_dataset_batch(input_root: Path, output_root: Path, folder_label_map: dict):
    """
    input_root: Path object to the raw dataset directory
    output_root: Path object to the destination (data/processed/train or /test)
    folder_label_map: Dictionary mapping subfolder names to integer labels (e.g., {"REAL": 0, "FAKE": 1})
    """

    output_root.mkdir(parents=True, exist_ok=True)

    for folder_name, label in folder_label_map.items():
        input_dir = input_root / folder_name

        if not input_dir.exists():
            print(f"Warning: Directory {input_dir} not found. Skipping.")
            continue

        # Using rglob to find all jpg/png images recursively if needed
        image_paths = list(input_dir.glob("*.jpg")) + list(input_dir.glob("*.png"))

        for img_p in tqdm(
            image_paths, desc=f"Processing {folder_name} (Label {label})"
        ):
            try:
                img_bgr = cv2.imread(str(img_p))
                if img_bgr is None:
                    continue

                img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                img_res = cv2.resize(img_rgb, (224, 224))

                # Extract Stream B (Spectral) - 192-dim
                spec_feat = extract_spectral_features(img_res)

                # Extract Stream C (Geometric) - 9-dim
                img_input = (
                    torch.from_numpy(img_res)
                    .permute(2, 0, 1)
                    .float()
                    .unsqueeze(0)
                    .to(device)
                    / 255.0
                )
                with torch.no_grad():
                    depth = midas(img_input).squeeze().cpu().numpy()
                depth_feat = extract_depth_features(img_res, depth)

                # Feature Vector: 192 (Spectral) + 9 (Depth) = 201-dim (Offline component)
                feat_vector = np.concatenate([spec_feat, depth_feat])

                target_stem = f"{label}_{img_p.stem}"

                # Save feature vector
                np.save(output_root / f"{target_stem}.npy", feat_vector)

                # Save standardized 224x224 image for Stream A (Spatial CNN)
                cv2.imwrite(
                    str(output_root / f"{target_stem}.jpg"),
                    cv2.cvtColor(img_res, cv2.COLOR_RGB2BGR),
                )

            except Exception as e:
                print(f"Error processing {img_p.name}: {e}")


if __name__ == "__main__":
    BASE_RAW = Path("data/raw")
    BASE_PROC = Path("data/processed")

    # Process CIFAKE (Train Split)
    cifake_map = {"REAL": 0, "FAKE": 1}
    process_dataset_batch(BASE_RAW / "CIFAKE/train", BASE_PROC / "train", cifake_map)

    # Process AutoSplice
    autosplice_map = {"Authentic": 0, "Forged_JPEG90": 2}
    process_dataset_batch(BASE_RAW / "AutoSplice", BASE_PROC / "train", autosplice_map)

    # Process CocoGlide
    cocoglide_map = {"real": 0, "fake": 2}
    process_dataset_batch(BASE_RAW / "CocoGlide", BASE_PROC / "train", cocoglide_map)
