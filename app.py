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

def preprocess_image(image, img_size=(64, 64)):
    img = np.array(image)
    if len(img.shape) == 2:
        img = np.stack([img] * 3, axis=-1)
    elif img.shape[2] == 4:
        img = img[:, :, :3]
    img = cv2.resize(img, img_size, interpolation=cv2.INTER_NEAREST)
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    return img

def predict_wafer(image):
    processed_img = preprocess_image(image)
    prediction = model.predict(processed_img)[0][0]
    label = "Pass" if prediction >= 0.5 else "Fail"
    confidence = prediction if label == "Pass" else 1 - prediction
    return label, confidence

st.title("Pass/Fail Semiconductor Wafer Classifier")
st.write("Capture a wafer map image with your camera or upload one to classify it as Pass or Fail.")

# Camera input
camera_image = st.camera_input("Take a photo of the wafer map")
# File uploader
uploaded_file = st.file_uploader("Or upload a wafer map image...", type=["png", "jpg", "jpeg"])

if camera_image is not None:
    image = Image.open(camera_image)
    st.image(image, caption="Camera-Captured Wafer Map", use_column_width=True)
    with st.spinner("Classifying..."):
        label, confidence = predict_wafer(image)
        st.write(f"**Prediction**: {label}")
        st.write(f"**Confidence**: {confidence:.2%}")
elif uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Wafer Map", use_column_width=True)
    with st.spinner("Classifying..."):
        label, confidence = predict_wafer(image)
        st.write(f"**Prediction**: {label}")
        st.write(f"**Confidence**: {confidence:.2%}")
else:
    st.info("Please capture a photo or upload an image to get started.")

st.write("**Note**: This model was trained on the WM-811K dataset. 'Pass' indicates no defect pattern, while 'Fail' indicates a defect.")
