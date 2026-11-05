# CNN-Based Helmet and Mask Detection System

## 📌 Project Overview

This project implements a Convolutional Neural Network (CNN) in Python and TensorFlow/Keras to determine whether a person is:
* **Wearing a helmet/mask**
* **Not wearing a helmet/mask**

It specifically demonstrates the mathematical concept of Convolutional Neural Networks and visualizes how convolutional layers progressively extract visual features from raw pixels to classification.

---

## 🎯 Objectives 

1. **Binary CNN Classification:** Classify images into *Wearing helmet/mask* vs *Not wearing helmet/mask*.
2. **Visual Progression Demonstration:** Explicitly visualize the progression:
   $$\text{Original Image} \rightarrow \text{Kernel} \rightarrow \text{Convolution} \rightarrow \text{Feature Map} \rightarrow \text{ReLU} \rightarrow \text{Pooling} \rightarrow \text{Classification}$$
3. **Progressive Feature Extraction:** Demonstrate how convolutional layers progressively extract features:
   * **Early Layers (Conv 1):** Low-level edges, borders, and line textures.
   * **Middle Layers (Conv 2):** Mid-level curves, corners, and contours.
   * **Deep Layers (Conv 3):** High-level abstract patterns and headgear/helmet regions.
4. **Assignment & Video Presentation:** Clear results and concise write-up ready for the 5–8 minute YouTube presentation and Assignment Book submission.

---

## 📂 Project Structure

```text
helmet_mask_cnn_project/
├── dataset/
│   ├── train/               # 226 images (113 with helmet, 113 without)
│   ├── validation/          # 50 images (25 with helmet, 25 without)
│   └── test/                # 50 images (25 with helmet, 25 without)
├── helmet_mask_cnn.keras    # Trained CNN model
├── train.py                 # Training script (epochs, curves, metrics)
├── predict.py               # Fast inference on any test image
├── visualize.py             # Visual progression & feature extraction demonstration
├── prepare_dataset.py       # Dataset distribution script
├── accuracy_curve.png       # Training vs validation accuracy plot
├── loss_curve.png           # Training vs validation loss plot
├── visual_progression.png   # 7-stage visual progression plot (for assignment screenshot)
├── progressive_features.png # Multi-layer feature maps across Conv1, Conv2, Conv3
├── results.txt              # Test evaluation metrics & confusion matrix
├── writeup.txt              # One-paragraph summary write-up for Assignment Book
├── presentation/            # 12-slide presentation (.pptx) for video
└── requirements.txt         # Dependencies
```

---

## 🚀 Quick Start & How to Run

### 1. Install Dependencies
```bash
python -m pip install -r requirements.txt
```

### 2. Run the Visual Progression Demonstration
Generates both `visual_progression.png` and `progressive_features.png`:
```bash
python visualize.py
```
Or test on any specific image:
```bash
python visualize.py dataset/test/without_helmet_mask/without_helmet_mask_12.jpg
```

### 3. Test a Prediction
```bash
python predict.py
```
Or test a custom image:
```bash
python predict.py dataset/test/with_helmet_mask/with_helmet_mask_11.jpg
```

### 4. Train the Model (Optional)
```bash
python train.py
```

---

## 🔍 Visual Progression Explained (For Presentation & Viva)

When explaining during your video presentation, walk through each stage:

1. **Original Image ($128 \times 128 \times 3$):** Input RGB photograph of the person.
2. **Kernel ($3 \times 3$):** A learned matrix of weights that slides across the image calculating spatial dot products.
3. **Convolution ($I * K + b$):** Linear response of the image convolved with the kernel, highlighting areas where patterns match.
4. **Feature Map:** Spatial representation of detected pattern intensities across the image.
5. **ReLU Activation ($f(x) = \max(0, x)$):** Sets all negative values to zero (black), keeping only positive activations and introducing non-linearity.
6. **Max Pooling ($2 \times 2$):** Takes the maximum value in each $2 \times 2$ window, downsampling the feature map from $128 \times 128$ to $64 \times 64$ to achieve translation invariance and lower computation.
7. **Classification (Dense + Sigmoid):** Flattened feature maps pass through dense layers and a sigmoid function, outputting probability:
   * Score $< 0.5 \rightarrow$ **Wearing helmet/mask**
   * Score $\ge 0.5 \rightarrow$ **Not wearing helmet/mask**

---

## 📈 Experimental Results

* **Test Accuracy:** 78.00%
* **Test Loss:** 0.5148
* **Test Dataset:** 50 held-out images (25 with helmet, 25 without helmet)
* **Confusion Matrix:**
  $$\begin{bmatrix} 19 & 6 \\ 5 & 20 \end{bmatrix}$$

---
