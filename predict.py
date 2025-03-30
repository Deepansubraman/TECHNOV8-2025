import numpy as np
import cv2
from PIL import Image

def predict_soil(uploaded_file):
    if uploaded_file is not None:
        image = Image.open(uploaded_file)  # Open image with PIL
        image = image.convert("RGB")  # Ensure it's RGB format

        # Convert to OpenCV format
        img = np.array(image)  # Convert PIL Image to NumPy array
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)  # Convert RGB to BGR (OpenCV format)

        # Ensure img is in correct format (Check dimensions)
        if img is None or img.size == 0:
            raise ValueError("Invalid image format. Please upload a valid image.")

        # Continue with your prediction logic
        # Example: resized = cv2.resize(img, (128, 128))
        
        return "Predicted Soil Type: Example_Type"  # Replace with actual model prediction

    return "No image uploaded"
