# Image Processing and Classification Pipeline

## Overview
This project was developed as part of an **Image Processing course**.  
It presents a complete workflow that combines **classical image processing**, **deep learning–based image classification**, and a **graphical user interface (GUI)** for interactive classification.

The project uses the **Brain Tumor MRI Dataset** from Kaggle and compares the performance of a **custom CNN model** with a **pretrained EfficientNet-B0 model** to evaluate the effect of advanced architectures on classification accuracy after image enhancement.

**Dataset:**  
https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset

---

## Objectives
- Enhance MRI images using classical image processing techniques
- Build a robust preprocessing and augmentation pipeline
- Address dataset class imbalance
- Train and evaluate a custom CNN model
- Fine-tune a pretrained **EfficientNet-B0** model and compare results
- Implement a **Tkinter GUI** for practical image classification

---

## Image Processing Techniques
- **CLAHE (Contrast Limited Adaptive Histogram Equalization)**
  - Applied to grayscale MRI images
- Image resizing and normalization
- Conversion between PIL, NumPy, and OpenCV formats

---

## Dataset Handling
- Dataset loading using `torchvision.datasets.ImageFolder`
- Dataset balancing to reduce class bias
- Train / validation / test splitting
- Custom PyTorch `Dataset` and `DataLoader`

---

## Data Augmentation
- Random horizontal flipping
- Random rotation
- On-the-fly augmentation using `torchvision.transforms`

---

## Deep Learning Models

### 1. Custom CNN
- Implemented using **PyTorch**
- Architecture includes convolutional, pooling, and fully connected layers
- Optimized with **Adam**
- Learning rate scheduling via **ReduceLROnPlateau**

### 2. EfficientNet-B0
- Pretrained **EfficientNet-B0** from `torchvision.models`
- Fine-tuned on the brain MRI dataset
- Performance compared to the custom CNN

---

## Model Evaluation
- Accuracy
- Confusion Matrix
- Classification Report (Precision, Recall, F1-score)

---

## Graphical User Interface (GUI)
- Built using **Tkinter**
- Allows users to:
  - Load an MRI image
  - Apply preprocessing
  - Run classification using trained models
  - View predicted class results

---

## Libraries and Tools
- Python
- PyTorch
- Torchvision
- OpenCV
- NumPy
- Matplotlib
- Scikit-learn
- PIL
- Tkinter

---
