# 3D Glioma Segmentation with U-Net

A lightweight 3D U-Net web app for automated brain tumor (glioma) segmentation from multi-modal MRI scans, optimized for CPU-only inference.

**Live app:** https://yargamji-brain-tumor-segmentation-app.streamlit.app

## Overview

This project implements an end-to-end pipeline for segmenting glioma sub-regions (necrotic/tumor core, edema, enhancing tumor) from 3D brain MRI scans, using a lightweight 3D U-Net architecture designed to train and run on standard CPUs rather than requiring a GPU. The trained model is deployed as an interactive Streamlit app where users can upload MRI scans and receive a segmentation prediction with visualization and a downloadable result.

The project was built as part of a transition from clinical medicine into data science, and follows the [Lightweight Brain Tumor Segmentation on Low-Resource Systems](https://dx.doi.org/10.17504/protocols.io.dm6gpdwmdgzp) tutorial developed by the Medical Artificial Intelligence (MAI) Lab, Lagos, and SPARK, with several adaptations described below.

## Dataset

[BraTS-Africa 2024](https://doi.org/10.7937/v8h6-867) — 95 glioma cases from The Cancer Imaging Archive (TCIA), each with four MRI modalities (T1n, T1c, T2f, T2w) and expert-annotated segmentation masks for three tumor sub-regions.

## Model & Training

- **Architecture:** Lightweight 3D U-Net (fewer layers and filters than standard 3D U-Net implementations, to reduce memory footprint)
- **Input:** 96×96×96×4 patches (4 stacked MRI modalities)
- **Loss:** Combined Dice Loss + Categorical Focal Loss
- **Optimizer:** Nadam, learning rate 0.001, with gradient clipping
- **Hardware:** Trained entirely on CPU
- **Training time:** ~2.1 hours (26 epochs, early stopping on validation loss)
- **Validation Dice score:** ~0.54

## Live Demo

The deployed app accepts four NIfTI files per case (T1n, T1c, T2f, T2w) and optionally a segmentation mask for side-by-side comparison. It outputs:
- A slice-by-slice visualization of the input scan, predicted segmentation, and ground truth (if provided)
- A downloadable NIfTI file (`.nii.gz`) containing the predicted segmentation

## Known Limitations

- **Center-patch inference:** the deployed app extracts and predicts on a single 96³ center patch of the cropped 128³ volume, rather than processing the full volume. Tumors located far from the image center may not be captured by the current deployed version, even though the underlying model performs patch-based prediction correctly during training/evaluation.
- **CPU-only inference:** prediction can take 30 seconds to a few minutes depending on the host machine, since no GPU acceleration is used.
- **Single-case inference:** the app processes one patient case at a time.

## Tech Stack

- TensorFlow / Keras — model architecture and training
- `segmentation_models_3D` — Dice and Focal loss implementations
- NiBabel — NIfTI file I/O
- scikit-learn — intensity scaling
- Streamlit — web app deployment
- Streamlit Community Cloud — hosting

## Running Locally

```bash
git clone https://github.com/Yargamji/brain-tumor-segmentation-app.git
cd brain-tumor-segmentation-app
pip install -r requirements.txt
streamlit run app.py
```

The trained model (`saved_model/3D_unet_100_epochs_2_batch_patch_training.keras`) is included directly in this repository.

## Acknowledgments

- Tutorial and reference implementation: Oladele, A., Confidence, R., Zhang, D., Umoren, C., Iorumbur, A. M., Gbadamosi, A., Dako, F., Adewole, M., & Anazodo, U. (2025). *Lightweight Brain Tumor Segmentation on Low-Resource Systems: A Step-by-Step Guide with 3D U-Net.* protocols.io. https://dx.doi.org/10.17504/protocols.io.dm6gpdwmdgzp
- Base 3D U-Net implementation reference: Bhattiprolu, S. — [python_for_microscopists](https://github.com/bnsreenu/python_for_microscopists/tree/master/231_234_BraTa2020_Unet_segmentation)
- Dataset: Adewole, M. et al. (2024). *Expanding the Brain Tumor Segmentation (BraTS) data to include African Populations (BraTS-Africa)* [Dataset]. The Cancer Imaging Archive. https://doi.org/10.7937/v8h6-867

## License

This project is for educational and research purposes only. It has not undergone clinical validation and is not intended for diagnostic or clinical use.
