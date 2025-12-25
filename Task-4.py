"""
TASK 4: Sales Prediction using Python
CodeAlpha Data Science Internship

Objective:
Predict future sales based on advertising spend across
different platforms and analyze their impact on sales.
This script loads an advertising dataset from a CSV file,
cleans the data, preprocesses it, trains a regression model,
evaluates its performance, and analyzes the impact of each
advertising platform on sales.

"""

# 1. IMPORT LIBRARIES

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# 2. LOAD DATASET

df = pd.read_csv("Advertising.csv")

print("Dataset Preview:")
print(df.head())

print("\nDataset Info:")
print(df.info())


# 3. DATA CLEANING & TRANSFORMATION

# Remove unnecessary index column if present
if "Unnamed: 0" in df.columns:
    df.drop(columns=["Unnamed: 0"], inplace=True)

# Handle missing values
df.dropna(inplace=True)

# 4. FEATURE SELECTION

features = ["TV", "Radio", "Newspaper"]
target = "Sales"

X = df[features]
y = df[target]


# 5. EXPLORATORY ANALYSIS

plt.figure(figsize=(7,5))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Between Advertising Platforms and Sales")
plt.show()


# 6. TRAIN-TEST SPLIT

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# 7. REGRESSION MODEL

model = LinearRegression()
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


# 9. FEATURE IMPACT ANALYSIS

impact = pd.DataFrame({
    "Platform": features,
    "Impact_on_Sales": model.coef_
})

print("\nImpact of Advertising Platforms on Sales:")
print(impact)

impact.plot(x="Platform", y="Impact_on_Sales", kind="bar", legend=False)
plt.title("Impact of Advertising Platforms on Sales")
plt.ylabel("Sales Impact")
plt.show()


# 10. ACTUAL vs PREDICTED SALES

plt.figure(figsize=(6,4))
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Sales")
plt.ylabel("Predicted Sales")
plt.title("Actual vs Predicted Sales")
plt.show()

print("\nTASK-4 Sales Prediction completed successfully!")
