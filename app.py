import streamlit as st
import cv2
import numpy as np
from PIL import Image
import tensorflow as tf
import random
import pandas as pd

# Load Pre-trained Model
model = tf.keras.models.load_model(r"C:\Users\Pretish S.S(Deepan)\OneDrive\Desktop\crop2\models\soil_model.h5")

# Soil Type Mapping
soil_types = {
    0: "Alluvial Soil",
    1: "Black Soil",
    2: "Clay Soil",
    3: "Red Soil",
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

# Function to simulate soil moisture
def get_soil_moisture():
    return random.randint(30, 80)

# Function for smart irrigation
def smart_irrigation(moisture_level, water_capacity):
    if moisture_level < 40:
        irrigation_needed = "Increase water supply"
    elif moisture_level > 70:
        irrigation_needed = "Reduce water supply"
    else:
        irrigation_needed = "Water level optimal"
    
    if water_capacity < 50:
        irrigation_needed += " (Low water availability detected)"
    
    return irrigation_needed

# Function to simulate crop disease prediction
def predict_disease(image):
    disease_prob = random.random()
    if disease_prob > 0.8:
        return "Leaf Blight (High Confidence)"
    elif disease_prob > 0.5:
        return "Mild Infection (Moderate Confidence)"
    else:
        return "No Disease Detected"

# ---- UI Design ----
st.set_page_config(page_title="Soil & Crop Recommendation", page_icon="🌱", layout="wide")
st.sidebar.title("🌾 Crop Recommendation System")
st.sidebar.write("Upload a soil image and select season & water availability.")

st.title("🌿 Advanced Soil & Crop Recommendation System")
st.write("Upload a soil image to get the predicted soil type and suitable crops.")

# User uploads image
uploaded_file = st.file_uploader("Choose a soil image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    if st.button("Predict Soil Type"):
        with st.spinner("Analyzing soil..."):
            soil_type, confidence = predict_soil(image)
        
        st.success(f"**Predicted Soil Type:** {soil_type} ({confidence:.2f}% confidence)")

        # Display soil moisture level
        moisture = get_soil_moisture()
        st.write(f"Current Soil Moisture Level: {moisture}%")
        
        # Irrigation advice based on moisture level and water capacity
        water_capacity = st.slider("Water Availability (Liters per square meter)", 10, 500, 100)
        irrigation_advice = smart_irrigation(moisture, water_capacity)
        st.write(f"Smart Irrigation Suggestion: {irrigation_advice}")

        # User Inputs for Crop Recommendation
        season = st.selectbox("Select Season", ["Summer", "Winter", "Rainy"])

        # Recommend Crops
        recommended_crops = crop_recommendations.get(soil_type, {}).get(season, ["No Data Available"])
        st.subheader("🌾 Recommended Crops")
        st.write(f"For **{soil_type}** in **{season}**, best crops are:")
        for crop in recommended_crops:
            st.markdown(f"- ✅ **{crop}**")

        # Disease Prediction
        disease_prediction = predict_disease(uploaded_file)
        st.write(f"Disease Prediction: {disease_prediction}")

        # Community Section for Sharing Experiences
        st.title("🌾 Community Crop Sharing")
        st.write("Share your experience with growing crops. Upload images and tell others about your success!")

        uploaded_crop_image = st.file_uploader("Upload a Crop Photo", type=["jpg", "png", "jpeg"])
        crop_experience = st.text_area("Share your experience or challenges")

        if uploaded_crop_image and crop_experience:
            st.image(uploaded_crop_image, caption="Shared Crop Image", use_column_width=True)
            st.write(f"User Experience: {crop_experience}")
            st.write("Your experience has been shared with the community.")

        # Feedback from users
        feedback = st.selectbox("Was this recommendation helpful?", ["Yes", "No", "Somewhat"])
        comments = st.text_area("Any additional comments or suggestions?")
        if feedback:
            st.write(f"Feedback: {feedback}")
        if comments:
            st.write(f"Comments: {comments}")

# Analytics Dashboard
st.write("📊 Crop Growth & Soil Analytics Dashboard")
data = {
    'Date': ["2023-03-01", "2023-03-15", "2023-03-30"],
    'Soil Type': ["Red Soil", "Black Soil", "Clay Soil"],
    'Crop': ["Rice", "Wheat", "Paddy"],
    'Growth Stage': ["Germination", "Vegetative", "Flowering"],
    'Moisture Level': [65, 55, 75],
}
df = pd.DataFrame(data)
st.dataframe(df)

# Plotting Growth Stage Over Time
st.line_chart(df['Moisture Level'])
