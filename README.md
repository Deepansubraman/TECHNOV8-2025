
# 🌱 Soil & Crop Recommendation System

## Overview

This project is a **Soil & Crop Recommendation System** that uses machine learning to predict soil types from images and recommend suitable crops based on the predicted soil type, season, and water availability. Additionally, it provides smart irrigation suggestions and basic crop disease detection.

## Features

- **Soil Type Prediction**: Uses a pre-trained deep learning model to classify soil images into different soil types.
- **Crop Recommendations**: Suggests suitable crops based on soil type and season.
- **Soil Moisture Simulation**: Provides simulated moisture levels to guide irrigation decisions.
- **Smart Irrigation System**: Recommends whether to increase or decrease water supply based on soil moisture and water availability.
- **Crop Disease Detection (Simulated)**: Provides a basic assessment of potential crop diseases.
- **Community Crop Sharing**: Users can upload crop images and share their farming experiences.
- **Analytics Dashboard**: Displays historical soil and crop data, along with a moisture level trend chart.

## Technologies Used

- **Python** 
- **Streamlit** (for UI development)
- **TensorFlow/Keras** (for soil type prediction)
- **OpenCV & PIL** (for image processing)
- **NumPy & Pandas** (for data handling)
- **Matplotlib & Streamlit Charts** (for visualization)

## Installation

### Prerequisites

Ensure you have Python installed (>=3.7) along with the necessary dependencies.

### Steps

1. Clone the repository:

    ```bash
    git clone https://github.com/your-repo/soil-crop-recommendation.git
    cd soil-crop-recommendation
    ```

2. Install dependencies:

    ```bash
    pip install -r requirements.txt
    ```

3. Run the application:

    ```bash
    streamlit run app.py
    ```

## Usage

1. **Upload a soil image**: Upload a soil image to classify it.
2. **Click "Predict Soil Type"**: Classify the uploaded soil image into one
