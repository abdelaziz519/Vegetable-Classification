import streamlit as st
import numpy as np
import cv2
import joblib
import os
import gdown

# 1. Page Configuration
st.set_page_config(page_title="Vegetable Classifier", page_icon="🥦", layout="centered")

# Custom CSS for a better look
st.markdown("""
    <style>
    .main {
        background-color: #f5f7f9;
    }
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        height: 3em;
        background-color: #2E7D32;
        color: white;
        font-weight: bold;
    }
    </style>
    """, unsafe_allow_html=True)

st.title("🥦 Smart Vegetable Classification System")
st.write("Upload a vegetable image, and the SVM model will identify its type.")
st.markdown("---")

# 2. Function to Download/Load Model from Google Drive
@st.cache_resource
def load_model():
    file_id = '1Qwx2ZcqAfv4HxaUYkd_aA-tJvgorWJoH'
    url = f'https://drive.google.com/uc?id={file_id}'
    output = 'best_model.pkl'
    
    if not os.path.exists(output):
        with st.spinner('Downloading model from Google Drive... This may take a moment the first time.'):
            try:
                gdown.download(url, output, quiet=False)
            except Exception as e:
                st.error(f"Error downloading from Drive: {e}")
    
    return joblib.load(output)

# 3. Function to Load Class Names
@st.cache_data
def load_classes():
    if os.path.exists('classes.npy'):
        return np.load('classes.npy')
    else:
        # Fallback list if classes.npy is missing
        return ["Beans", "Bitter_Gourd", "Bottle_Gourd", "Brinjal", "Broccoli", 
                "Cabbage", "Capsicum", "Carrot", "Cauliflower", "Cucumber", 
                "Papaya", "Potato", "Pumpkin", "Radish", "Tomato"]

# Attempt to load the model and classes
try:
    model = load_model()
    class_names = load_classes()
except Exception as e:
    st.error(f"App initialization failed: {e}")

# 4. Image Upload Section
uploaded_file = st.file_uploader("Choose an image (JPG, PNG, JPEG)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Convert uploaded file to OpenCV format
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)
    
    # Display the uploaded image
    st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), caption='Uploaded Image', use_container_width=True)

    # 5. Prediction Logic
    if st.button('Classify Image 🚀'):
        with st.spinner('Analyzing the image...'):
            # --- Preprocessing Steps (Mirroring Training Phase) ---
            # A. Convert to Grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            # B. Resize to 64x64
            resized = cv2.resize(gray, (64, 64))
            # C. Flatten and Normalization (/255.0)
            flattened = resized.flatten().astype('float32') / 255.0
            # D. Reshape for the model (single sample)
            features = flattened.reshape(1, -1)

            # E. Perform Prediction
            prediction = model.predict(features)
            label = class_names[prediction[0]]
            
            # Display Final Result
            st.markdown("---")
            st.balloons()
            st.success(f"### Predicted Category: **{label}**")
            st.info("This prediction is based on pixel-pattern recognition learned by the SVM model.")

st.markdown("---")
st.caption("AI-Powered Application | Developed with OpenCV, SVM, and Streamlit")