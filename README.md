# Image Processing and Classification Pipeline

## Overview
This project was developed as part of an **Image Processing course**.  
It presents a complete workflow that combines **classical image processing**, **deep learning–based image classification**, and a **graphical user interface (GUI)** for user interaction.

The project compares the performance of a **custom-built CNN model** with a **pretrained EfficientNet-B0 model** to evaluate the effect of advanced architectures on classification accuracy after image enhancement.

---

## Objectives
- Enhance images using classical image processing techniques
- Build a robust preprocessing and data augmentation pipeline
- Handle dataset class imbalance
- Train and evaluate a custom CNN model
- Compare CNN performance with a pretrained **EfficientNet-B0**
- Provide a user-friendly **Tkinter-based GUI** for image classification

---

## Image Processing Techniques
- **CLAHE (Contrast Limited Adaptive Histogram Equalization)**
  - Applied to grayscale images
  - Applied to the L-channel in LAB color space for RGB images
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
- Dynamic augmentation using `torchvision.transforms`

---

## Deep Learning Models

### 1. Custom CNN
- Implemented from scratch using **PyTorch**
- Consists of:
  - Convolutional layers
  - Activation functions
  - Pooling layers
  - Fully connected layers
- Optimized using **Adam optimizer**
- Learning rate scheduling with **ReduceLROnPlateau**

### 2. EfficientNet-B0
- Pretrained **EfficientNet-B0** model
- Used transfer learning for classification
- Fine-tuned on the project dataset
- Performance compared directly with the custom CNN

---

## Model Evaluation
- Accuracy
- Confusion Matrix
- Classification Report:
  - Precision
  - Recall
  - F1-score

---

## Graphical User Interface (GUI)
- Implemented using **Tkinter**
- Allows users to:
  - Load an image
  - Apply preprocessing
  - Run classification using the trained model
  - View predicted class results
- Designed to demonstrate practical deployment of the trained models

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

## Project Structure
