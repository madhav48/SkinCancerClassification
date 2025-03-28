# Skin Cancer Classification using Deep Learning

## Overview
This project contains a deep learning-based skin cancer classification system. The objective is to classify skin cancer images into multiple categories using state-of-the-art deep learning models with transfer learning and fine-tuning techniques. The models trained in this project include MobileNet and EfficientNet, achieving a maximum recall score of approximately 75%.

## Motivation
Skin cancer is a major health concern worldwide, and early detection plays a crucial role in improving treatment outcomes. This project aims to leverage deep learning techniques to assist in the automated classification of skin cancer types, which can support dermatologists in diagnosis and decision-making.

## Dataset
The dataset consists of images of different types of skin cancer. The images were preprocessed and augmented to enhance the generalizability of the model. Standard image preprocessing techniques such as normalization, resizing, and augmentation (flipping, rotation, and brightness adjustments) were applied to improve model robustness.

## Models Used
### 1. **MobileNet**
- Achieved a recall score of about **70%**.
- Took approximately **1 hour 34 minutes for 10 epochs**.
- Further training led to overfitting, causing a decline in accuracy.
- Model implemented in **SkinCancerClassification2.ipynb**.

### 2. **EfficientNet**
- Achieved a recall score of about **75%**.
- Trained using **EfficientNet-B3**, which is well-suited for the given dataset size and complexity.
- Took around **20 minutes for 2 epochs**.
- Similar to MobileNet, excessive training led to overfitting.
- Model implemented in **SkinCancerClassification1.ipynb**.

## Methodology
### 1. **Data Preprocessing**
   - Resizing images to a fixed dimension suitable for the models.
   - Normalization to scale pixel values between 0 and 1.
   - Data Augmentation: Rotation, flipping, contrast enhancement.

### 2. **Transfer Learning & Fine-Tuning**
   - Pretrained MobileNet and EfficientNet models were used.
   - The top layers were replaced with custom fully connected layers.
   - Fine-tuning was performed on selected layers to improve feature extraction.

### 3. **Loss Function & Optimization**
   - **Loss Function:** CrossEntropyLoss.
   - **Optimizer:** Adam optimizer with learning rate scheduling.
   - Batch size and learning rate were tuned for best performance.

### 4. **Evaluation Metrics**
   - **Recall Score:** Used as the primary evaluation metric due to the importance of minimizing false negatives in medical diagnosis.
   - **Accuracy & Loss Curves:** Monitored to detect overfitting and underfitting trends.

## Results
| Model        | Recall Score | Training Time |
|-------------|-------------|--------------|
| MobileNet   | ~70%        | 1 hr 34 min  |
| EfficientNet-B3 | ~75%    | 20 min (2 epochs) |

- Both models showed signs of overfitting after further training.
- EfficientNet-B3 outperformed MobileNet in recall and training efficiency.

## Future Improvements
- Experimenting with additional architectures such as ResNet and DenseNet.
- Implementing **ensemble learning** to combine predictions from multiple models.
- Applying techniques such as **dropout, batch normalization, and advanced augmentation** to mitigate overfitting.
- Exploring self-supervised learning approaches for improved feature extraction.

