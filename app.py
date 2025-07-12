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
    prediction = model.predict(processed_img)[0][0]
    if 0.2 < prediction < 0.8:  # Threshold for ambiguous predictions
        return "Uncertain", 0.0
    label = "Pass" if prediction >= 0.5 else "Fail"
    confidence = prediction if label == "Pass" else 1 - prediction
    return label, confidence

st.title("Pass/Fail Semiconductor Wafer Classifier")
st.write("Capture a wafer map image with your camera or upload one to classify it as Pass or Fail.")
st.write("**Tip**: Use clear wafer map images (grayscale, defect patterns). Images are converted to grayscale for classification.")

# Camera input
camera_image = st.camera_input("Take a photo of the wafer map")
# File uploader
uploaded_file = st.file_uploader("Or upload a wafer map image...", type=["png", "jpg", "jpeg"])

if camera_image is not None:
    image = Image.open(camera_image)
    if not is_valid_wafer_image(image):
        st.warning("Image converted to grayscale for processing, but ensure it’s a clear wafer map for best results.")
    st.image(image.convert('L'), caption="Camera-Captured Wafer Map (Grayscale)", use_column_width=True)
    st.write(f"Image shape: {np.array(image.convert('L')).shape}")
    with st.spinner("Classifying..."):
        label, confidence = predict_wafer(image)
        if label == "Uncertain":
            st.warning("Prediction uncertain. Try a clearer wafer map image.")
        else:
            st.write(f"**Prediction**: {label}")
            st.write(f"**Confidence**: {confidence:.2%}")
elif uploaded_file is not None:
    image = Image.open(uploaded_file)
    if not is_valid_wafer_image(image):
        st.warning("Image converted to grayscale for processing, but ensure it’s a clear wafer map for best results.")
    st.image(image.convert('L'), caption="Uploaded Wafer Map (Grayscale)", use_column_width=True)
    st.write(f"Image shape: {np.array(image.convert('L')).shape}")
    with st.spinner("Classifying..."):
        label, confidence = predict_wafer(image)
        if label == "Uncertain":
            st.warning("Prediction uncertain. Try a clearer wafer map image.")
        else:
            st.write(f"**Prediction**: {label}")
            st.write(f"**Confidence**: {confidence:.2%}")
else:
    st.info("Please capture a photo or upload an image to get started.")

st.write("**Note**: This model was trained on the WM-811K dataset. 'Pass' indicates no defect pattern, while 'Fail' indicates a defect.")
