import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow import keras

# Load model function with caching
@st.cache_resource
def load_teachable_model():
    try:
        model = keras.models.load_model('keras_model.h5', compile=False)
        class_names = open('labels.txt', 'r').readlines()
        return model, class_names
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None

# Load modelimport streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow import keras
import cv2
import time

# Load model function
load_model = keras.models.load_model

# Set page config
st.set_page_config(
    page_title="Product Anomaly Inspector",
    page_icon=" ",
    layout="wide"
)

# Load the model with caching
@st.cache_resource
def load_teachable_model():
    try:
        model = load_model('keras_model.h5', compile=False)
        class_names = open('labels.txt', 'r').readlines()
        return model, class_names
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None

model, class_names = load_teachable_model()

# Preprocess function
def preprocess_image(image):
    image = image.resize((224, 224))
    image_array = np.asarray(image)
    normalized_image = (image_array.astype(np.float32) / 127.5) - 1
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    data[0] = normalized_image
    return data

# Prediction function
def make_prediction(image_data):
    if model is None:
        return None, None, None  # Always return three values
    prediction = model.predict(image_data)
    index = np.argmax(prediction)
    confidence = prediction[0][index]
    return index, confidence, prediction

# Main app function
def main():
    st.title("Product Anomaly Detection System")
    st.markdown("""
        This system detects anomalies in manufactured products using a deep learning model trained with Teachable Machine.
    """)

    # Create tabs for different detection methods
    tab1, tab2 = st.tabs(["Image Upload", "Real-Time Camera"])

    # Image Upload tab
    with tab1:
        st.header("Image Upload Detection")
        st.write("Upload an image of the product to check for defects")

        uploaded_file = st.file_uploader(
            "Choose a product image...",
            type=["jpg", "png", "jpeg"],
            key="file_uploader"
        )

        if uploaded_file is not None:
            try:
                image = Image.open(uploaded_file).convert('RGB')
                st.image(image, caption='Uploaded Product Image', use_container_width=True)

                with st.spinner('Analyzing the product...'):
                    image_data = preprocess_image(image)
                    index, confidence, prediction = make_prediction(image_data)

                    if index is not None:
                        col1, col2 = st.columns(2)

                        with col1:
                            st.subheader("Detection Results")
                            status = class_names[index].strip()
                            if "normal" in status.lower():
                                st.success(f"Status: {status}")
                            else:
                                st.error(f"Status: {status}")
                            st.metric("Confidence", f"{confidence:.2%}")

                        with col2:
                            st.subheader("Class Probabilities")
                            for i, class_name in enumerate(class_names):
                                prob = prediction[0][i]
                                progress = int(prob * 100)
                                st.write(f"{class_name.strip()}")
                                st.progress(progress, text=f"{prob:.2%}")
            except Exception as e:
                st.error(f"Error processing image: {e}")

    # Real-Time Camera tab
    with tab2:
        st.header("Real-Time Camera Detection")
        st.write("Use your camera to detect anomalies in real-time")

        picture = st.camera_input("Take a picture of the product", key="camera")

        if picture:
            try:
                image = Image.open(picture).convert('RGB')

                with st.spinner('Analyzing in real-time...'):
                    image_data = preprocess_image(image)
                    index, confidence, prediction = make_prediction(image_data)

                    if index is not None:
                        st.image(image, caption='Captured Product Image', use_container_width=True)
                        col1, col2 = st.columns(2)

                        with col1:
                            st.subheader("Real-Time Results")
                            status = class_names[index].strip()
                            if "normal" in status.lower():
                                st.success(f"Status: {status}")
                            else:
                                st.error(f"Status: {status}")
                            st.metric("Confidence", f"{confidence:.2%}")

                        with col2:
                            st.subheader("Detection Probabilities")
                            for i, class_name in enumerate(class_names):
                                prob = prediction[0][i]
                                progress = int(prob * 100)
                                st.write(f"{class_name.strip()}")
                                st.progress(progress, text=f"{prob:.2%}")
            except Exception as e:
                st.error(f"Error processing camera image: {e}")

    # Sidebar with additional info
    with st.sidebar:
        st.header("About This System")
        st.markdown("""
        - Built with Teachable Machine and TensorFlow  
        - Deployed using Streamlit  
        - Detects anomalies in manufactured products  
        """)

        if model is not None:
            st.success("Model loaded successfully")
        else:
            st.error("Model failed to load")

        st.header("Instructions")
        st.markdown("""
        1. Use the Image Upload tab to analyze existing images  
        2. Use the Real-Time Camera tab for live detection  
        3. Wait for the analysis results  
        """)

# Entry point
if __name__ == "__main__":
    main()

model, class_names = load_teachable_model()

# Preprocessing function
def preprocess_image(image):
    image = image.resize((224, 224))
    image_array = np.asarray(image)
    normalized_image = (image_array.astype(np.float32) / 127.5) - 1
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    data[0] = normalized_image
    return data

# Prediction function
def make_prediction(image_data):
    if model is None:
        return None, None, None
    prediction = model.predict(image_data)
    index = np.argmax(prediction)
    confidence = prediction[0][index]
    return index, confidence, prediction

# Streamlit app config
st.set_page_config(
    page_title="Product Anomaly Inspector",
    layout="wide"
)

def main():
    st.title("🧠 Product Anomaly Detection System")
    st.markdown("""
    This system detects anomalies in manufactured products using a deep learning model trained with Teachable Machine.
    """)

    tab1, tab2 = st.tabs(["📁 Image Upload", "📷 Real-Time Camera"])

    # Tab 1 - Upload
    with tab1:
        st.header("Image Upload Detection")
        uploaded_file = st.file_uploader("Choose a product image...", type=["jpg", "png", "jpeg"])
        
        if uploaded_file is not None:
            try:
                image = Image.open(uploaded_file).convert('RGB')
                st.image(image, caption='Uploaded Product Image', use_container_width=True)
                with st.spinner('Analyzing the product...'):
                    image_data = preprocess_image(image)
                    index, confidence, prediction = make_prediction(image_data)
                    
                    if index is not None:
                        col1, col2 = st.columns(2)
                        with col1:
                            st.subheader("Detection Results")
                            status = class_names[index].strip()
                            if "normal" in status.lower():
                                st.success(f"Status: {status}")
                            else:
                                st.error(f"Status: {status}")
                            st.metric("Confidence", f"{confidence:.2%}")

                        with col2:
                            st.subheader("Class Probabilities")
                            for i, class_name in enumerate(class_names):
                                prob = prediction[0][i]
                                st.write(f"{class_name.strip()}")
                                st.progress(int(prob * 100), text=f"{prob:.2%}")
            except Exception as e:
                st.error(f"Error processing image: {e}")

    # Tab 2 - Camera
    with tab2:
        st.header("Real-Time Camera Detection")
        picture = st.camera_input("Take a picture of the product")

        if picture:
            try:
                image = Image.open(picture).convert('RGB')
                with st.spinner('Analyzing in real-time...'):
                    image_data = preprocess_image(image)
                    index, confidence, prediction = make_prediction(image_data)

                    if index is not None:
                        st.image(image, caption='Captured Product Image', use_container_width=True)
                        col1, col2 = st.columns(2)
                        with col1:
                            st.subheader("Real-Time Results")
                            status = class_names[index].strip()
                            if "normal" in status.lower():
                                st.success(f"Status: {status}")
                            else:
                                st.error(f"Status: {status}")
                            st.metric("Confidence", f"{confidence:.2%}")

                        with col2:
                            st.subheader("Detection Probabilities")
                            for i, class_name in enumerate(class_names):
                                prob = prediction[0][i]
                                st.write(f"{class_name.strip()}")
                                st.progress(int(prob * 100), text=f"{prob:.2%}")
            except Exception as e:
                st.error(f"Error processing camera image: {e}")

    # Sidebar Info
    with st.sidebar:
        st.header("About This System")
        st.markdown("""
        - Built with Teachable Machine and TensorFlow  
        - Deployed using Streamlit  
        - Detects anomalies in manufactured products
        """)
        if model:
            st.success("✅ Model loaded successfully")
        else:
            st.error("❌ Model failed to load")

        st.header("Instructions")
        st.markdown("""
        1. Use the **Image Upload** tab to analyze existing images  
        2. Use the **Real-Time Camera** tab for live detection  
        3. Wait for analysis and view confidence scores
        """)

if __name__ == "__main__":
    main()
