# 🧠 OCR Digit Recognition App

### **M.Sc. Computer Science – Final Year Project**

![Status](https://img.shields.io/badge/Project_Status-Under_Development-yellow)
![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-critical)
![NumPy](https://img.shields.io/badge/NumPy-1.x-blueviolet)
![Matplotlib](https://img.shields.io/badge/Matplotlib-3.x-green)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

---

## 📌 Overview

This repository presents a deep learning–based **Optical Character Recognition (OCR)** system developed as part of my **M.Sc. Computer Science Final Year Project**.  
The academic objective is to design, train, and deploy an efficient handwritten digit recognition model that demonstrates conceptual understanding, practical machine learning competency, and deployment readiness.  
The project integrates a trained Convolutional Neural Network (CNN) with a user-friendly Streamlit interface, enabling real-time prediction from user-uploaded digit images.

The broader aim of the work is to explore essential components of applied computer vision—preprocessing methods, optimized CNN architectures, model generalization strategies, and interactive model deployment. This makes the project both academically rigorous and practically valuable.

> **Note:**  
> This project is **still under active development**, and further enhancements, UI improvements, and performance optimizations will be added.

---

### 🔎 Application Summary

The system implements end-to-end **handwritten digit recognition (0–9)** based on the MNIST dataset.  
Users can upload digit images, which are preprocessed through:

- Grayscale conversion
- Resizing to 28×28
- Gaussian Blur
- Otsu Thresholding
- Normalization

The cleaned image is then passed to the CNN for classification.  
This workflow ensures robustness across a range of handwritten styles.

---

## 📁 Project Structure

```
📦 OCR-Digit-Recognition
├── app.ipynb          # Model training, evaluation, and saving
├── app.py             # Streamlit interface for real-time prediction
├── app.keras          # Saved CNN model
├── requirements.txt   # Dependencies
└── README.md          # Project documentation
```

---

## 🧠 Model Architecture

The CNN used in this project consists of:

- **Input Layer:** shape (28, 28, 1)
- **Data Augmentation Block:** rotation, translation, zoom, contrast
- **Two Convolution Blocks:** Conv2D + MaxPooling
- **Flatten + Dense Layers**
- **Softmax Output Layer** (10 digits)

### Key Training Details

- Dataset: **MNIST**
- Epochs: **7**
- Loss Function: `sparse_categorical_crossentropy`
- Optimizer: `adam`
- Expected Accuracy: **~98%** depending on hardware
- Final Model Saved As: **app.keras**

---

## 🧪 Code Summary

### Model Training (`app.ipynb`)

Data loading and normalization:

```python
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

x_train = x_train.reshape((60000, 28, 28, 1)) / 255.0
x_test  = x_test.reshape((10000, 28, 28, 1)) / 255.0
```

Data augmentation:

```python
data_augmentation = models.Sequential([
    layers.RandomRotation(0.1),
    layers.RandomTranslation(0.1, 0.1),
    layers.RandomZoom(0.1),
    layers.RandomContrast(0.1)
])
```

Model:

```python
model = models.Sequential([
    Input(shape=(28,28,1)),
    data_augmentation,
    layers.Conv2D(32, (3,3), activation='relu', padding='same'),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(32, (3,3), activation='relu', padding='same'),
    layers.MaxPooling2D((2,2)),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])
```

---

## 🌐 Streamlit App (`app.py`)

The app allows users to upload images and receive predictions.  
Preprocessing steps include:

- RGB → Grayscale
- Resize to 28×28
- Gaussian Blur
- Otsu Thresholding
- Normalization

Prediction:

```python
prediction = model.predict(final_img)
digit = np.argmax(prediction)
st.write("### Predicted Digit:", digit)
```

---

## 🚀 How to Run the Application

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/itspriyambhattacharya/OCR-Digit-Recognition.git
cd OCR-Digit-Recognition
```

### 2️⃣ Install Dependencies

```bash
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Mac / Linux

pip install -r requirements.txt
```

### 3️⃣ Start the Streamlit App

```bash
streamlit run app.py
```

Open in browser:

```
http://localhost:8501/
```

---

## 🖼 Uploading Images

** Will do this later**

---

## 🔧 Requirements

```
numpy
opencv-python
tensorflow
streamlit
Pillow
matplotlib
```

---

## 🛠 Planned Enhancements

- Multi-digit recognition
- Bounding-box auto-detection
- Grad-CAM interpretability visualizations
- Export to TensorFlow Lite for mobile deployment
- UI enhancements in Streamlit

---

## 📝 License

This project is licensed under the **MIT License**.  
You are free to use, modify, and redistribute it with attribution.
