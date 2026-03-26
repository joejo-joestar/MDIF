<h1 align="center">🖼️ MDIF: A Multi-Domain Inconsistency Framework for Image Forgery Detection</h1>

<p align="center">
    <img src="" alt="Image Proc" title="Image Proc" width="128" >
</p>

---

## Getting Started

Firstly, clone this repo,
<!--
Firstly, clone this repo using

```bash
git clone https://github.com/joejo-joestar/Agentic-AI-Notebooks.git
```
-->

Then install Miniconda from [the Anaconda Website](https://docs.anaconda.com/miniconda/install).

> [!NOTE]
> All the notebooks **running locally** assumes you are using a conda environment!

Then open a command prompt, and run the following. This will create and activate a `python 3.11` environment called `mdif`. The [environment.yml](./environment.yml) will be used to create the environment and install all needed dependencies.

```bash
conda env create
```

```bash
conda activate mdif
```

After running this, your CMD prompt should have a "`(mdif)`" prefixed at the start.

> [!TIP]
> To **deactivate** the environment, simply run:
>
> ```bash
> conda deactivate
> ```
>
> To **remove** the environment completely, run:
>
> ```bash
> conda env remove -n mdif
> ```

<br/>

> [!NOTE]
> Remember to select `mdif` as the kernel for all the notebooks!

---

## Architecture

---

## Datasets

| Dataset                                                                                          | Type               | Usage in MDIF                         |
| :----------------------------------------------------------------------------------------------- | :----------------- | :------------------------------------ |
| [CIFAKE](https://www.kaggle.com/datasets/birdy654/cifake-real-and-ai-generated-synthetic-images) | SD v1.4 + CIFAR-10 | Initial pipeline testing              |
| [GenImage](https://genimage-dataset.github.io/)                                                  | MJ, DALL-E 3, Flux | Training the spatial/spectral streams |
| [AutoSplice](https://github.com/shanface33/AutoSplice_Dataset/tree/main)                         | DALL-E 2 Inpainted | Training inpainting localization      |
| [CocoGlide](https://github.com/grip-unina/TruFor)                                                | GLIDE Inpainted    | Testing boundary inconsistencies      |
| [Raise-1k](https://loki.disi.unitn.it/RAISE/confirm.php?package=1k)                              | High-Res RAW       | Frequency domain baseline             |

---

## Acknowledgments

---
