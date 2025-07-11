# AI-based-Pass-Fail-Semiconductor-Wafer-Using-Image
Pass/Fail Semiconductor Wafer Classifier
This project is an AI-based web application that classifies semiconductor wafer map images as Pass (no defects) or Fail (defective) using a Convolutional Neural Network (CNN). Built with TensorFlow and deployed on Streamlit Community Cloud, it processes wafer map images from the WM-811K dataset to detect defect patterns.
Project Overview

Objective: Automatically classify semiconductor wafer maps as Pass or Fail based on defect patterns.
Dataset: WM-811K Wafer Map Dataset (CC0-1.0 license).
Model: A CNN trained on 64x64 grayscale wafer map images, stacked to 3 channels, achieving binary classification (Pass/Fail).
Deployment: Hosted on Streamlit Community Cloud for easy access and testing.
Use Case: Demonstrates AI-driven defect detection for semiconductor manufacturing, suitable for educational demos or portfolio showcases.

Setup
To run this project locally or modify it:

Clone the Repository:git clone https://github.com/YOUR_USERNAME/ai-based-pass-fail-semiconductor-wafer-using-image.git
cd ai-based-pass-fail-semiconductor-wafer-using-image


Install Dependencies:Ensure Python 3.10 is installed, then:pip install -r requirements.txt

Dependencies include:
streamlit==1.39.0
tensorflow==2.15.0
opencv-python-headless==4.11.0.86
numpy==1.26.4
pillow==10.4.0
h5py==3.10.0


Run the App Locally:streamlit run app.py


Model: The trained model (wafer_cnn_model.h5) is stored in saved_model/.

Usage

Access the Live App:
Visit Streamlit App.


Upload an Image:
Upload a wafer map image (PNG/JPG/JPEG).
The app resizes the image to 64x64, processes it, and predicts Pass (no defects) or Fail (defects present) with confidence.


Demo Images:
Use sample images from the WM-811K dataset or synthetic wafer maps (see Training and Demo).



Training and Demo

Training:
The CNN was trained on the WM-811K dataset in Google Colab using tensorflow==2.15.0.
Images are preprocessed to 64x64 grayscale, stacked to 3 channels, and labeled as Pass (failureType='none') or Fail (any defect).
Model architecture: Sequential CNN with 2 Conv2D layers, MaxPooling, Dense layers, and Dropout (see app.py for details).


Demo Images:
Sample images generated from WM-811K for testing:
 Prediction: Pass (95% confidence)
 Prediction: Fail (98% confidence)




Repository Structure
ai-based-pass-fail-semiconductor-wafer-using-image/
├── app.py                  # Streamlit app script
├── requirements.txt        # Dependencies for Python 3.10
├── saved_model/
│   └── wafer_cnn_model.h5  # Trained CNN model (~7.9 MB)
├── sample_images/          # Sample wafer map images for demo
│   ├── wafer_Pass_0.png
│   ├── wafer_Fail_0.png
│   └── ...

Credits

Dataset: WM-811K Wafer Map (CC0-1.0 license).
Tools: TensorFlow, Streamlit, OpenCV, NumPy, Pillow.
Author: Harshad Zalaniya And AI

Contributing
Feel free to fork this repository, submit issues, or create pull requests to enhance the app’s functionality or UI.


Built with 💻 and ☕ by Harshad for Learning Purpose
