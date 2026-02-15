# Stress-Testing of Convolutional Neural Networks (DL Assignment 1)

This repository contains the complete implementation for the "Stress-Testing of CNNs" assignment. It includes a baseline ResNet-18 model trained from scratch on CIFAR-10, a systematic failure analysis using Grad-CAM, and a constrained improvement using Data Augmentation.

## 👥 Group Members
* **Member 1:** Satendra Singh (M25AIR006)
* **Member 2:** Shashank Kumar Gautam (M25AIR007)
* **Member 3:** Shivansh Yadav (M25AIR008)
* **Member 4:** Sunil Pargi (M25AIR010)

## 📌 Project Overview
The goal of this assignment is to move beyond simple accuracy metrics and understand the "why" behind model predictions.

* **Dataset:** CIFAR-10 (10 classes, $32 \times 32$ images)
* **Architecture:** ResNet-18 (Modified for $32 \times 32$ input, trained from scratch)
* **Framework:** PyTorch
* **Seed:** 42 (Fixed for reproducibility)
* **Optimizer:** SGD with Momentum and Cosine Annealing

## 📂 File Structure
```text
.
├── DL_Assignment_1_.ipynb  # Main Jupyter Notebook containing all code
├── README.md                             # This file
├── data/                                 # CIFAR-10 dataset (downloaded automatically)
└── reports/                              # (Optional) Folder for saved plots/reports
```
🚀 How to Run the Code
Open the Notebook: Upload the .ipynb file to Google Colab or run it locally with Jupyter Lab.

Install Dependencies: The code requires standard PyTorch libraries.

Bash
```text
pip install torch torchvision matplotlib numpy opencv-python
```
Execute Cells in Order:
```
Cells 1-2: Setup environment, fix random seed (42), and load data.

Cells 3-5: Initialize the Baseline ResNet-18 and train it for 35-50 epochs.

Cell 6: Detect "High-Confidence Failure" cases in the test set.

Cell 7: Generate Grad-CAM heatmaps to visualize model attention.

Cell 8: Train the "Augmented Model" (Constrained Improvement) using RandomCrop/Flip.

Cell 9: Re-evaluate the specific failure cases on the improved model.
```
## 📊 Experimental Results
1. Baseline Performance
Architecture: ResNet-18 (No pretraining)

Test Accuracy: ~85.4% (Typical baseline)

Observation: The model trains smoothly but shows signs of overfitting (Training Acc > Test Acc).

2. Failure Analysis
We identified distinct failure modes where the model was confidently wrong (Confidence > 0.90).

Common Errors: Background confusion (e.g., Blue background = Plane), Texture bias (e.g., Cat vs. Dog).

Explainability: Grad-CAM visualizations confirmed that in many failure cases, the model focused on irrelevant background features (e.g., clouds, grass) rather than the object itself.

3. Improvement Strategy (Data Augmentation)
We applied a single constrained improvement: Standard Data Augmentation.

Technique: RandomCrop(32, padding=4) + RandomHorizontalFlip().

Result: Test Accuracy improved to ~88-90%.

Impact: The gap between Training and Validation accuracy narrowed, indicating better generalization. Several failure cases (e.g., off-center objects) were corrected by the augmented model.

## ⚠️ Reproducibility Note
All experiments utilize a fixed random seed (SEED = 42) for:

Python random

numpy

torch CPU and CUDA

torch.backends.cudnn (Deterministic mode)

This ensures that the Training Curves, Failure Cases, and Grad-CAM heatmaps generated are identical every time the notebook is run.
