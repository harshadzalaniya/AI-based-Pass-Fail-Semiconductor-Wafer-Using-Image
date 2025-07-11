import streamlit as st
       import tensorflow as tf
       import cv2
       import numpy as np
       from PIL import Image
       import gdown
       import os

       # Download model from Google Drive
       model_url = "YOUR_GOOGLE_DRIVE_SHAREABLE_LINK"  # Replace with your link
       model_path = "saved_model/wafer_cnn_model.h5"
       if not os.path.exists(model_path):
           os.makedirs("saved_model", exist_ok=True)
           gdown.download(model_url, model_path, quiet=False)

       # Load the trained model
       model = tf.keras.models.load_model(model_path)

       def preprocess_image(image, img_size=(64, 64)):
           # Convert PIL image to numpy array
           img = np.array(image)
           
           # Ensure image has 3 channels
           if len(img.shape) == 2:
               img = np.stack([img] * 3, axis=-1)
           elif img.shape[2] == 4:
               img = img[:, :, :3]  # Remove alpha channel
           
           # Resize image
           img = cv2.resize(img, img_size, interpolation=cv2.INTER_NEAREST)
           
           # Normalize to [0, 1]
           img = img / 255.0
           
           # Add batch dimension
           img = np.expand_dims(img, axis=0)
           return img

       def predict_wafer(image):
           # Preprocess the image
           processed_img = preprocess_image(image)
           
           # Make prediction
           prediction = model.predict(processed_img)[0][0]
           
           # Determine label and confidence
           label = "Pass" if prediction >= 0.5 else "Fail"
           confidence = prediction if label == "Pass" else 1 - prediction
           return label, confidence

       # Streamlit app
       st.title("Pass/Fail Semiconductor Wafer Classifier")
       st.write("Upload a wafer map image to classify it as Pass or Fail.")

       # File uploader
       uploaded_file = st.file_uploader("Choose a wafer map image...", type=["png", "jpg", "jpeg"])

       if uploaded_file is not None:
           # Display uploaded image
           image = Image.open(uploaded_file)
           st.image(image, caption="Uploaded Wafer Map", use_column_width=True)
           
           # Predict and display result
           with st.spinner("Classifying..."):
               label, confidence = predict_wafer(image)
               st.write(f"**Prediction**: {label}")
               st.write(f"**Confidence**: {confidence:.2%}")

       st.write("Note: This model was trained on the WM-811K dataset. 'Pass' indicates no defect pattern, while 'Fail' indicates a defect.")