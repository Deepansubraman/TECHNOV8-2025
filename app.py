import streamlit as st
import numpy as np
import cv2
import tensorflow as tf
from PIL import Image

# Load trained model
MODEL_PATH = "soil_model.h5"  # Ensure this model file exists in the correct location
model = tf.keras.models.load_model(MODEL_PATH)

# Soil type labels
SOIL_LABELS = {
    0: "Alluvial Soil",
    1: "black Soil",
    2: "clay Soil",
    3: "red Soil",
}

def predict_soil(uploaded_file):
    """Process the image and predict soil type."""
    if uploaded_file is not None:
        try:
            image = Image.open(uploaded_file)  # Open image
            img = np.array(image)  # Convert PIL image to NumPy array
            
            if img is None or img.size == 0:
                st.error("Invalid image file. Please upload a valid image.")
                return None
            
            # Convert to OpenCV format
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            img = cv2.resize(img, (128, 128))  # Resize to match model input
            img = img / 255.0  # Normalize pixel values
            img = np.expand_dims(img, axis=0)  # Add batch dimension

            # Predict soil type
            predictions = model.predict(img)
            predicted_class = np.argmax(predictions)  # Get highest probability class
            
            return SOIL_LABELS.get(predicted_class, "Unknown Soil Type")
        
        except Exception as e:
            st.error(f"Error processing image: {e}")
            return None
    else:
        st.error("No file uploaded!")
        return None

# Streamlit UI
st.title("🌱 Soil Type Prediction")
st.write("Upload an image of soil, and the model will predict its type.")

uploaded_file = st.file_uploader("Choose a soil image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
    
    soil_type = predict_soil(uploaded_file)
    
    if soil_type:
        st.success(f"**Predicted Soil Type:** {soil_type}")
