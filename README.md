# Brain Tumor Detection using Deep Learning (VGG16 + Transfer Learning)

This project uses **Deep Learning** with a **VGG16 pre-trained model** and **Transfer Learning** to classify MRI brain scans as tumor or non-tumor.  
The goal is to provide an accurate, automated approach for early tumor detection, which can assist radiologists in diagnosis.

Tumor types Prediction through MRI Images:
Glioma
Meningioma
NoTumor
Pituitary


---

## Overview

Brain tumors can be life-threatening if not detected early.  
Manual detection from MRI scans is time-consuming and prone to human error.  
This project leverages **Convolutional Neural Networks (CNNs)** and **Transfer Learning** to automatically classify MRI images with high accuracy.

**Key Steps:**
1. **Data Preprocessing** – Resize, normalize, and split MRI images into train/test sets.
2. **Model** – Use **VGG16** pre-trained on ImageNet, freeze base layers, and add custom dense layers.
3. **Training** – Fine-tune upper layers to adapt to MRI features.
4. **Evaluation** – Assess performance using accuracy, precision, recall, and F1-score.
5. **Prediction** – Classify new MRI scans in real-time.

---

## Dataset

- **Source:** [https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset] (e.g., Kaggle)
- **Classes:** Tumor / Non-tumor (or multiple tumor types)
- **Format:** JPEG/PNG MRI scans
- **Preprocessing:**
  - Resized to **224×224 pixels** (VGG16 input size)
  - Normalized pixel values to [0,1]
  - Augmentation for better generalization

---

##  Model Architecture

- **Base Model:** VGG16 (pre-trained on ImageNet)
- **Custom Layers:** Flatten → Dense → Dropout → Output Layer (Softmax/Sigmoid)
- **Loss Function:** Binary Crossentropy / Categorical Crossentropy
- **Optimizer:** Adam / SGD
- **Metrics:** Accuracy, Precision, Recall, F1-Score

---

## Installation

Before running the project, install the dependencies:

```bash
# Create virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate   # Mac/Linux
venv\Scripts\activate      # Windows

# Upgrade pip
pip install --upgrade pip

# Install core libraries
pip install numpy pandas matplotlib seaborn
pip install tensorflow keras
pip install scikit-learn
pip install opencv-python
