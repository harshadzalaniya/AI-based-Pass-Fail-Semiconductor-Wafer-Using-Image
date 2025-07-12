import streamlit as st
import tensorflow as tf
import cv2
import numpy as np
from PIL import Image
import os

# Load the trained model
model_path = "saved_model/wafer_cnn_model.h5"
try:
    if not os.path.exists(model_path):
        st.error(f"Model file not found at {model_path}")
        st.stop()
    st.write("Loading model...")
    model = tf.keras.models.load_model(model_path)
    st.write("Model loaded successfully.")
except Exception as e:
    st.error(f"Error loading model: {str(e)}")
    st.stop()

def is_valid_wafer_image(image):
    img_array = np.array(image)
    if len(img_array.shape) == 3:
        # Check if image is nearly grayscale (R, G, B channels similar)
        r, g, b = img_array[:, :, 0], img_array[:, :, 1], img_array[:, :, 2]
        color_diff = np.mean(np.abs(r - g)) + np.mean(np.abs(g - b))
        return color_diff < 20  # Relaxed threshold for grayscale-like images
    return len(img_array.shape) == 2  # Grayscale is valid

def preprocess_image(image, img_size=(64, 64)):
    # Convert to grayscale
    img = image.convert('L')  # Convert to grayscale using PIL
    img = np.array(img)
    # Stack to 3 channels for model compatibility
    img = np.stack([img] * 3, axis=-1)
    # Resize and normalize
    img = cv2.resize(img, img_size, interpolation=cv2.INTER_NEAREST)
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    return img

def predict_wafer(image):
    processed_img = preprocess_image(image)
    prediction = model
