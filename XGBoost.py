import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

# 1️⃣ Load the dataset
df = pd.read_csv("soil_data.csv")  # Replace with your dataset

# 2️⃣ Encode categorical target variable (soil type)
label_encoder = LabelEncoder()
df["soil_type"] = label_encoder.fit_transform(df["soil_type"])  

# 3️⃣ Define features and target
X = df.drop(columns=["soil_type"])  # Features: pH, moisture, nutrients, etc.
y = df["soil_type"]  # Target: Soil classification

# 4️⃣ Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5️⃣ Train XGBoost model
model = xgb.XGBClassifier(n_estimators=100, max_depth=5, learning_rate=0.1)
model.fit(X_train, y_train)

# 6️⃣ Make predictions
y_pred = model.predict(X_test)

# 7️⃣ Evaluate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2f}")

# 8️⃣ Save model for later use
import joblib
joblib.dump(model, "soil_model.pkl")
