# Skin Cancer Classification using Deep Learning

## 📌 Overview
This repository presents a **deep learning-based multi-class skin cancer classification system**. The aim is to leverage **state-of-the-art deep learning models** to detect and classify different types of skin cancer from dermatoscopic images. The project utilizes **transfer learning and fine-tuning techniques** on **MobileNet** and **EfficientNet** architectures, achieving an optimal recall score of approximately **75%**.

---

## 🎯 Motivation
Skin cancer is one of the most prevalent cancers worldwide, and early detection is crucial for **improving patient survival rates**. Traditional diagnostic methods rely heavily on **dermatologists' expertise**, which may not always be **accessible or consistent**. This project aims to **automate and enhance** the detection process using deep learning techniques, aiding **early diagnosis** and **reducing misclassification risks**.

---

## 📂 Dataset Information
- The dataset consists of **dermatoscopic images** of various skin cancer types.
- **Classes:** Multiple categories representing different types of skin cancer.
- **Preprocessing Steps:**
  - Resizing images to a fixed input dimension (**224x224** for MobileNet and **300x300** for EfficientNet).
  - Normalization to **scale pixel values** between 0 and 1.
  - Data Augmentation techniques:
    - **Rotation & Flipping:** To increase dataset variability.
    - **Contrast Adjustment:** To improve feature visibility.
    - **Zoom & Cropping:** To simulate real-world image variations.

---

## 🔬 Model Architectures Used

### 1️⃣ MobileNet
✔ **Lightweight and optimized for mobile devices**  
✔ **Achieved ~70% recall score**  
✔ Training time: **1 hour 34 minutes (10 epochs)**  
❌ **Overfitting observed with additional training**  

Implementation: **SkinCancerClassification2.ipynb**

---

### 2️⃣ EfficientNet-B3
✔ **High-performance model with better feature extraction**  
✔ **Achieved ~75% recall score**  
✔ Training time: **20 minutes (2 epochs)**  
❌ **Overfitting observed after further training**  

Implementation: **SkinCancerClassification1.ipynb**

---

## 🛠️ Methodology & Approach

### 1️⃣ Data Preprocessing  
- **Resizing:** Standardizing image dimensions for deep learning models.  
- **Normalization:** Scaling pixel values to a range between 0-1 for stable training.  
- **Data Augmentation:** Enhancing dataset variability to improve model robustness.  

### 2️⃣ Transfer Learning & Fine-Tuning  
- Used **pre-trained MobileNet and EfficientNet models**.  
- **Replaced top layers** with custom classification layers (Fully Connected + Softmax).  
- **Fine-tuned** selected layers for enhanced feature extraction.  

### 3️⃣ Model Training & Optimization  
- **Loss Function:** CrossEntropyLoss (suited for multi-class classification).  
- **Optimizer:** Adam with learning rate decay for stable convergence.  
- **Batch Size:** Tuned to balance memory usage and performance.  
- **Learning Rate Scheduling:** Prevents overfitting and accelerates convergence.  

### 4️⃣ Evaluation Metrics  
- **Recall Score:** **Primary metric** (important for medical applications to reduce false negatives).  
- **Accuracy & Loss Curves:** Used to detect overfitting trends.  
- **Confusion Matrix:** Visualized misclassification patterns.  

---

## 📊 Performance Summary

| Model        | Recall Score | Training Time | Overfitting Observed? |
|-------------|-------------|--------------|-------------------|
| MobileNet   | ~70%        | 1 hr 34 min (10 epochs)  | Yes |
| EfficientNet-B3 | ~75%    | 20 min (2 epochs) | Yes |

- **EfficientNet-B3 performed better** in terms of recall score and training efficiency.  
- **Both models suffered from overfitting**, requiring **further regularization techniques**.  

---

## 🔥 Future Improvements
- **Experiment with additional architectures** (e.g., **ResNet, DenseNet**).  
- **Ensemble Learning:** Combining predictions from multiple models.  
- **Advanced Regularization Techniques:** Dropout, L1/L2 Regularization.  
- **Hyperparameter Tuning:** Further optimize learning rate, batch size, and dropout rates.  
- **Self-Supervised Learning:** Leverage semi-supervised techniques for improved feature extraction.  

---
