"""
Car Price Prediction using CSV Dataset
CodeAlpha Data Science Internship - Task 3
This script loads a car price dataset from a CSV file, cleans the data,
preprocesses it, trains a regression model, evaluates its performance,
"""

# 1. IMPORT LIBRARIES

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# 2. LOAD DATASET

df = pd.read_csv("car data.csv")

print("Dataset preview:")
print(df.head())

print("\nDataset info:")
print(df.info())


# 3. DATA CLEANING

df.dropna(inplace=True)

# Encode categorical columns
label_encoders = {}
for col in df.columns:
    if df[col].dtype == "object":
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le

# 4. FEATURES & TARGET

target_column = "Selling_Price"

X = df.drop(target_column, axis=1)
y = df[target_column]



# 5. TRAIN-TEST SPLIT

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# 6. MODEL PIPELINE


model = Pipeline([
    ("scaler", StandardScaler()),
    ("regressor", LinearRegression())
])


# 7. TRAIN MODEL

model.fit(X_train, y_train)


# 8. PREDICTION & EVALUATION

y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\nModel Performance:")
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"R² Score: {r2:.4f}")


# 9. VISUALIZATION

plt.figure(figsize=(7,5))
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs Predicted Car Prices")
plt.show()

print("\nTask-3 Car Price Prediction completed successfully!")
