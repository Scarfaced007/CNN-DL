# Stress-Testing of Convolutional Neural Networks (DL Assignment 1)

This repository contains the complete implementation for the "Stress-Testing of CNNs" assignment. It focuses on understanding CNN behavior through systematic experimentation, failure analysis, and explainability techniques, rather than just maximizing accuracy.

## 👥 Team Members
* **Member 1:** Satendra Singh (M25AIR006)
* **Member 2:** Shashank Kumar Gautam (M25AIR007)
* **Member 3:** Shivansh Yadav (M25AIR008)
* **Member 4:** Sunil Pargi (M25AIR010)

## 📌 Project Overview
The objective of this assignment is to develop a deep and critical understanding of how Convolutional Neural Networks (CNNs) behave in practice. We move beyond simple accuracy metrics to investigate why models fail and how to improve them.

* **Dataset:** CIFAR-10 (10 classes, $32 \times 32$ RGB images)
* **Architecture:** ResNet-18 (Modified for $32 \times 32$ input, trained from scratch)
* **Framework:** PyTorch
* **Seed:** 42 (Fixed for reproducibility)
* **Optimizer:** SGD with Momentum (0.9) and Weight Decay (5e-4)
* **Scheduler:** Cosine Annealing

## 📂 File Structure
```text
.
├── DL_Assignment_1.ipynb               # Main Jupyter Notebook containing all code
├── training_curves.png                 # Training/validation loss and accuracy curves
├── model_comparison.png                # Baseline vs improved model comparison
├── failure_case.png                    # Grad-CAM analysis of failure case 1
├── analysis_report.txt                 # Comprehensive analysis report
├── README.md                           # This file (Project Report/README)
├── data/                               # CIFAR-10 dataset (downloaded automatically)
```
## 🚀 How to Run the Code
1. Open the Notebook: Upload the .ipynb file to Google Colab or run it locally with Jupyter Lab.

2. Install Dependencies: The code requires standard PyTorch libraries.
```bash
pip install torch torchvision matplotlib numpy opencv-python
```
3. Execute Cells in Order:
```
Cells 1-2: Setup environment, fix random seed (42), and load data.

Cells 3-5: Initialize the Baseline ResNet-18 and train it for 35-50 epochs.

Cell 6: Detect "High-Confidence Failure" cases in the test set.

Cell 7: Generate Grad-CAM heatmaps to visualize model attention.

Cell 8: Train the "Augmented Model" (Constrained Improvement) using RandomCrop/Flip.

Cell 9: Re-evaluate the specific failure cases on the improved model.
```
## 📊 Experimental Results
1. ### Baseline Performance

- Test Accuracy: 87.42%
- **Training Behavior:** The model achieved high training accuracy but plateaued on validation, indicating clear overfitting.
- **High-Confidence Failures:** We identified numerous cases where the model was confident (>90%) but incorrect.

2. ### Failure Analysis
- We analyzed distinct failure modes using Grad-CAM:
- **Background Confusion:** In several cases (e.g., Planes vs. Ships), the model focused on the blue background (sky/water) rather than the object.
- **Visual Similarity:** Systematic confusion between semantically similar classes like Cat ↔ Dog and Automobile ↔ Truck.
- **Attention Patterns:** Grad-CAM heatmaps revealed that for misclassified images, the model often attended to "spurious cues" (like grass texture for deer/frogs) rather than defining features.

3. ### Improvement Strategy (Data Augmentation)

- We applied a single constrained improvement to address the overfitting and brittleness.
- **Technique:** RandomCrop(32, padding=4) + RandomHorizontalFlip().
- **Result:** Test Accuracy improved to ~90.5%.
- **Impact:**
  - **Failure Reduction:** Significant decrease in high-confidence failures.
  - **Better Generalization:** The gap between Training and Validation accuracy narrowed significantly.
  - **Robustness:** The improved model successfully corrected several "off-center" or "reversed" objects that failed in the baseline.

4. ### Key Findings

- **Data Augmentation Effectiveness:** A simple augmentation strategy yielded a significant accuracy improvement.
- **Model Confidence is Deceptive:** High confidence does not imply correctness; the baseline model was frequently "confidently wrong."
- **Attention Matters:** Correct predictions usually correlated with Grad-CAM heatmaps focusing on the object, while failures often focused on the background.

## ⚠️ Reproducibility Note
### All experiments utilize a fixed random seed (SEED = 42) for:

- Python random
- numpy
- torch CPU and CUDA

**This ensures that the Training Curves, Failure Cases, and Grad-CAM heatmaps generated are identical every time the notebook is run.**
