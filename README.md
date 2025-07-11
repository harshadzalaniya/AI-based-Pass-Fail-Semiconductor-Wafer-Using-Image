# AI-BASED-PASS-FAIL-SEMICONDUCTOR-WAFER-USING-IMAGE

This project is a machine learning web application that classifies semiconductor wafer map images as **Pass** (no defects) or **Fail** (defective) using a Convolutional Neural Network (CNN). Built with TensorFlow and deployed on Streamlit Community Cloud, it leverages the WM-811K dataset to detect defect patterns in semiconductor wafers.

## Project Overview
- **Objective**: Automate the classification of semiconductor wafer maps to identify defect-free (Pass) or defective (Fail) wafers.
- **Dataset**: [WM-811K Wafer Map Dataset](https://www.kaggle.com/datasets/qingyi/wm811k-wafer-map) (CC0-1.0 license).
- **Model**: CNN trained on 64x64 grayscale wafer map images (stacked to 3 channels) for binary classification (Pass/Fail).
- **Deployment**: Hosted on Streamlit Community Cloud for seamless interaction.
- **Use Case**: Ideal for demonstrating AI-driven defect detection in semiconductor manufacturing, suitable for portfolios or educational content.

## Setup
To run this project locally or contribute:
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/YOUR_USERNAME/ai-based-pass-fail-semiconductor-wafer-using-image.git
   cd ai-based-pass-fail-semiconductor-wafer-using-image
  
  2. **Download Requirements**
  ```bash
   pip install -r requirements.txt
  ```
  3. **Run Locally**
  ```bash
   streamlit run app.py
  ```
  ## Streamlit Link For This Application
  https://ai-based-pass-fail-semiconductor-wafer-using-image-9jefucbzzeu.streamlit.app/
  ## File Structure
  ai-based-pass-fail-semiconductor-wafer-using-image/
├── app.py                  # Streamlit app script
├── requirements.txt        # Python 3.10 dependencies
├── saved_model/
│   └── wafer_cnn_model.h5  # Trained CNN model (~7.9 MB)
├── sample_images/          # Sample wafer maps for demo
│   ├── wafer_Pass_0.png
│   ├── wafer_Fail_0.png
│   └── ...

## Credits
Dataset: WM-811K Wafer Map (CC0-1.0 license).
Tools: TensorFlow, Streamlit, OpenCV, NumPy, Pillow.
Author: HARSHAD ZALANIYA

## Contributing
Feel free to fork this repository, submit issues, or create pull requests to enhance the app’s functionality or UI.

Built with 💻 and ☕ by ME for the Learning An Fun
  
   
