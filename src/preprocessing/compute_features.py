"""
Processes the raw datasets (CIFAKE, AutoSplice, CocoGlide) to extract and save features for training/testing.
For each image:
1. Load and resize to 224x224.
2. Extract spectral features (192-dim).
3. Extract depth features using MiDaS (9-dim).
4. Concatenate features and save as .npy.

Labels:
- 0: Real (CIFAKE REAL, AutoSplice Authentic, CocoGlide real, GenImage Nature)
- 1: Fake (CIFAKE FAKE, GenImage Midjourney/glide)
- 2: Forged/Inpainted (AutoSplice Forged_JPEG90, CocoGlide fake)
"""

import cv2
import torch
import numpy as np
import random
from typing import cast
from pathlib import Path
from tqdm import tqdm
from src.preprocessing.signal_proc import (
    extract_spectral_features,
    extract_depth_features,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
midas = (
    cast(torch.nn.Module, torch.hub.load("intel-isl/MiDaS", "MiDaS_small"))
    .to(device)
    .eval()
)


def process_dataset_with_split(
    input_root: Path, output_base: Path, folder_label_map: dict, split_ratio=0.8
) -> None:
    """
    Processes a dataset with an optional train/test split.

    :param Path input_root: Path to raw dataset
    :param Path output_base: Path to data/processed/ (will create /train and /test subfolders)
    :param dict folder_label_map: {folder_name: label_id}
    :param float split_ratio: Percentage of data for training (e.g., 0.8)
    """

    # Create split subdirectories
    train_path = output_base / "train"
    test_path = output_base / "test"
    train_path.mkdir(parents=True, exist_ok=True)
    test_path.mkdir(parents=True, exist_ok=True)

    random.seed(42)

    for folder_name, label in folder_label_map.items():
        input_dir = input_root / folder_name
        if not input_dir.exists():
            print(f"Skipping {input_dir}: Not found.")
            continue

        # Collect all images
        images = (
            list(input_dir.glob("*.jpg"))
            + list(input_dir.glob("*.png"))
            + list(input_dir.glob("*.JPEG"))
        )
        random.shuffle(images)

        # Determine split index
        split_idx = int(len(images) * split_ratio)
        train_images = images[:split_idx]
        test_images = images[split_idx:]

        def run_extraction(img_list, target_dir):
            for img_p in tqdm(
                img_list, desc=f"Processing {folder_name} -> {target_dir.name}"
            ):
                try:
                    img_bgr = cv2.imread(str(img_p))
                    if img_bgr is None:
                        continue
                    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                    img_res = cv2.resize(img_rgb, (224, 224))

                    spec_feat = extract_spectral_features(img_res)
                    img_t = (
                        torch.from_numpy(img_res)
                        .permute(2, 0, 1)
                        .float()
                        .unsqueeze(0)
                        .to(device)
                        / 255.0
                    )
                    with torch.no_grad():
                        depth = midas(img_t).squeeze().cpu().numpy()
                    depth_feat = extract_depth_features(img_res, depth)

                    # Concatenate features (192 from spectral + 9 from depth = 201 total)
                    feat_vector = np.concatenate([spec_feat, depth_feat])

                    target_stem = f"{label}_{img_p.stem}"
                    np.save(target_dir / f"{target_stem}.npy", feat_vector)
                    cv2.imwrite(
                        str(target_dir / f"{target_stem}.jpg"),
                        cv2.cvtColor(img_res, cv2.COLOR_RGB2BGR),
                    )
                except Exception as e:
                    print(f"Error on {img_p.name}: {e}")

        run_extraction(train_images, train_path)
        run_extraction(test_images, test_path)


if __name__ == "__main__":
    BASE_RAW = Path("data/raw")
    BASE_PROC = Path("data/processed")

    # CIFAKE has its own test/train folders, so we process them separately with ratio 1.0
    process_dataset_with_split(
        BASE_RAW / "CIFAKE/train", BASE_PROC, {"REAL": 0, "FAKE": 1}, split_ratio=1.0
    )
    process_dataset_with_split(
        BASE_RAW / "CIFAKE/test", BASE_PROC, {"REAL": 0, "FAKE": 1}, split_ratio=0.0
    )

    # AutoSplice is often one big folder, so we split it 80/20
    process_dataset_with_split(
        BASE_RAW / "AutoSplice",
        BASE_PROC,
        {"Authentic": 0, "Forged_JPEG90": 2},
        split_ratio=0.8,
    )

    # CocoGlide split
    process_dataset_with_split(
        BASE_RAW / "CocoGlide", BASE_PROC, {"real": 0, "fake": 2}, split_ratio=0.8
    )

    # GenImage split
    process_dataset_with_split(
        BASE_RAW / "GenImage",
        BASE_PROC,
        {"Nature": 0, "Midjourney": 1, "glide": 1},
        split_ratio=0.8,
    )
