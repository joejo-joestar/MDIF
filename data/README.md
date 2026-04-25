# Datasets

As mentioned in [`README.md`](../README.md#datasets-used), this directory has to be set up to store the datasets used by `mdif`.

There are two directories and the following sections explain how to set them up!

> [!NOTE]
> Most of the datasets are split by **_80% for training and validation_** and **_20% for testing_**!
> The training and validation dataset is then split again by **_80% for training_** and **_20% for validation_** in the `training/*` modules

---

## `raw`

The `raw` directory holds the datasets we manually download and extract. The data in this directory is used by the [`preprocessing/compute_features`](../mdif/preprocessing/compute_features.py) module for preprocessing.

### Setup

Download and extract the datasets from the following sources:

| Dataset                | Source                                                                                           | Remarks                                                                                                                                                              |
| :--------------------- | ------------------------------------------------------------------------------------------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| CIFAKE                 | [Kaggle](https://www.kaggle.com/datasets/birdy654/cifake-real-and-ai-generated-synthetic-images) | This dataset only contains the `REAL/*` (_real_) and `FAKE/*` (_fake_)                                                                                               |
| Unbiased Tiny GenImage | [Kaggle](https://www.kaggle.com/datasets/cartografia/unbiased-tiny-genimage)                     | We only use the `Nature/*` (_real_), `Midjourney/*` (_fake_), and `glide/*` (_fake_) for training and testing                                                        |
| AutoSplice             | [Github](https://github.com/shanface33/AutoSplice_Dataset/tree/main)                             | We only use the `Authentic/*` (_real_) and `Forged_JPEG90/*` (_inpainted_) for training and testing                                                                  |
| CocoGlide              | [Github](https://github.com/grip-unina/TruFor)                                                   | This dataset contains `real/*` (_real_) and `fake/*` (_inpainted_) for training and testing                                                                          |
| SAGI                   | [Kaggle](https://www.kaggle.com/datasets/giakop/sagi-d/data)                                     | This dataset contains `original/*` (_real_), `brushnet/*` (_inpainted_), `controlnet/*` (_inpainted_), and `removeanything/*` (_inpainted_) for training and testing |

> [!NOTE]
> The `CIFAKE` and `SAGI` dataset, by default, comes split as `train/*` and `test/*` datasets but the others do not.
> To address this, the [`preprocessing/compute_features`](../mdif/preprocessing/compute_features.py) module splits the datasets into training and testing datasets while preprocessing!
>
> The `SAGI` Dataset comes with two folders, `/coco/*` and `/raise/*`. The model has only been trained on the `/raise/*` split of the dataset.

Once the datasets have been stored in the `raw` directory, head on over to the [mdif notebook](../mdif.ipynb) to see how to perform the preprocessing steps!

---

## `processed`

The `processed` directory is further split into a `train` and `test` directories.

The files in these directories are _resized image (`.jpg`) files_ as well as their corresponding _extracted feature (`.npy`) files_ by the [`preprocessing/compute_features`](../mdif/preprocessing/compute_features.py) module.

These files have a consistent naming scheme `<label>_<original_file_name>.*` where the labels are as follows:

| Label | Description      | Dataset                                                                                     |
| ----- | ---------------- | ------------------------------------------------------------------------------------------- |
| 0     | Real             | CIFAKE `REAL`, AutoSplice `Authentic`, CocoGlide `real`, GenImage `Nature`, SAGI `original` |
| 1     | Fake             | CIFAKE `FAKE`, GenImage `Midjourney`/`glide`                                                |
| 2     | Forged/Inpainted | AutoSplice `Forged_JPEG90`, CocoGlide `fake`, SAGI `brushnet`/`controlnet`/`removeanything` |

---
