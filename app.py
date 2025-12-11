import streamlit as st
import numpy as np
import cv2
import tensorflow as tf
from PIL import Image

# Load the trained model
model = tf.keras.models.load_model("app.keras")

st.title("OCR Digit Recognition App")
st.write("Upload an image containing a single handwritten digit (0–9).")

# File uploader
uploaded_file = st.file_uploader(
    "Upload an image", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # Display uploaded image
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", use_container_width=True)

    # Convert to numpy array
    img = np.array(img)

    # Convert image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Resize to 28x28
    resized = cv2.resize(gray, (28, 28))

    # Gaussian Blur
    gb = cv2.GaussianBlur(src=resized, ksize=(5, 5), sigmaX=0)

    _, otsu = cv2.threshold(src=gb, thresh=0, maxval=255,
                            type=cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

    # Normalize pixel values
    normalized = otsu.astype("float32") / 255.0

    # Reshape for model input (1, 28, 28, 1)
    final_img = normalized.reshape(1, 28, 28, 1)

    # Prediction
    prediction = model.predict(final_img)
    digit = np.argmax(prediction)

    st.write("### Predicted Digit:", digit)
