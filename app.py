import streamlit as st
import cv2
import numpy as np
from PIL import Image
import tensorflow as tf

# Load the trained soil model
model = tf.keras.models.load_model("soil_model.h5")

# Soil Type Mapping
soil_types= {
    0: "Alluvial Soil",
    1: "black Soil",
    2: "clay Soil",
    3: "red Soil",
}

# Crop Recommendations (You can expand this based on real data)
crop_recommendations = {
    "Black Soil": {
        "Summer": ["Cotton", "Soybean", "Groundnut"],
        "Winter": ["Wheat", "Mustard"],
        "Rainy": ["Rice", "Millets", "Sugarcane"]
    },
    "Red Soil": {
        "Summer": ["Millets", "Pulses"],
        "Winter": ["Wheat", "Barley"],
        "Rainy": ["Rice", "Maize"]
    },
    "Sandy Soil": {
        "Summer": ["Watermelon", "Groundnut"],
        "Winter": ["Mustard", "Carrot"],
        "Rainy": ["Bajra", "Jowar"]
    },
    "Clay Soil": {
        "Summer": ["Paddy", "Sugarcane"],
        "Winter": ["Potato", "Peas"],
        "Rainy": ["Rice", "Jute"]
    },
    "Loamy Soil": {
        "Summer": ["Tomato", "Onion"],
        "Winter": ["Cabbage", "Cauliflower"],
        "Rainy": ["Paddy", "Banana"]
    }
}

# Function to predict soil type
def predict_soil(image):
    img = np.array(image)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    img = cv2.resize(img, (128, 128))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    prediction = model.predict(img)
    predicted_class = np.argmax(prediction)
    confidence = np.max(prediction) * 100

    return soil_types[predicted_class], confidence

# ---- UI Design ----
st.set_page_config(page_title="Soil & Crop Recommendation", page_icon="🌱", layout="wide")

st.sidebar.title("🌾 Crop Recommendation System")
st.sidebar.write("Upload a soil image and select season & water availability.")

st.title("🌿 Advanced Soil & Crop Recommendation System")
st.write("Upload a soil image to get the predicted soil type and suitable crops.")

uploaded_file = st.file_uploader("Choose a soil image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    if st.button("Predict Soil Type"):
        with st.spinner("Analyzing soil..."):
            soil_type, confidence = predict_soil(image)
        
        st.success(f"**Predicted Soil Type:** {soil_type} ({confidence:.2f}% confidence)")

        # User Inputs for Crop Recommendation
        season = st.selectbox("Select Season", ["Summer", "Winter", "Rainy"])
        water_capacity = st.slider("Water Availability (Liters per square meter)", 10, 500, 100)

        # Recommend Crops
        recommended_crops = crop_recommendations.get(soil_type, {}).get(season, ["No Data Available"])
        st.subheader("🌾 Recommended Crops")
        st.write(f"For **{soil_type}** in **{season}**, best crops are:")
        
        for crop in recommended_crops:
            st.markdown(f"- ✅ **{crop}**")

        # Show Water Requirement
        if water_capacity < 50:
            st.warning("🚱 Water availability is low! Consider drought-resistant crops.")
        elif water_capacity > 300:
            st.info("💧 Water availability is high! You can grow water-intensive crops.")

