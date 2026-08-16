import pandas as pd
import numpy as np

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load the dataset
df = pd.read_csv("houses.csv")

# Display first 5 rows
print("===== FIRST 5 ROWS =====")
print(df.head())

# Dataset shape
print("\n===== DATASET SHAPE =====")
print(df.shape)

# Column names
print("\n===== COLUMN NAMES =====")
print(df.columns.tolist())

# Basic statistics
print("\n===== DESCRIPTIVE STATISTICS =====")
print(df.describe())

# Check missing values
print("\n===== MISSING VALUES =====")
print(df.isnull().sum())

# Check duplicate rows
print("\n===== DUPLICATE ROWS =====")
print(df.duplicated().sum())


# ==========================================
# EXPLORATORY VISUALIZATIONS
# ==========================================

# Size vs Price
plt.figure(figsize=(7, 5))
plt.scatter(df["size"], df["price"])
plt.xlabel("House Size")
plt.ylabel("House Price")
plt.title("House Size vs Price")
plt.grid(True)
plt.savefig("size_vs_price.png")
plt.close()


# Rooms vs Price
plt.figure(figsize=(7, 5))
plt.scatter(df["rooms"], df["price"])
plt.xlabel("Number of Rooms")
plt.ylabel("House Price")
plt.title("Number of Rooms vs Price")
plt.grid(True)
plt.savefig("rooms_vs_price.png")
plt.close()


# Age vs Price
plt.figure(figsize=(7, 5))
plt.scatter(df["age"], df["price"])
plt.xlabel("House Age")
plt.ylabel("House Price")
plt.title("House Age vs Price")
plt.grid(True)
plt.savefig("age_vs_price.png")
plt.close()


# Boxplot
plt.figure(figsize=(8, 5))
df.boxplot()
plt.title("Boxplot of House Dataset")
plt.ylabel("Value")
plt.savefig("house_boxplot.png")
plt.close()

print("\n===== GRAPHS CREATED =====")
print("size_vs_price.png")
print("rooms_vs_price.png")
print("age_vs_price.png")
print("house_boxplot.png")


# ==========================================
# 3. SELECT FEATURES AND TARGET
# ==========================================

features = ["size", "rooms", "age"]

X = df[features]
y = df["price"]

print("\n===== FEATURES AND TARGET =====")
print("Features:", features)
print("Target: price")

from sklearn.model_selection import train_test_split

# ==========================================
# 4. TRAIN / TEST SPLIT
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=1
)

print("\n===== TRAIN / TEST SPLIT =====")
print("Training feature rows:", X_train.shape[0])
print("Testing feature rows:", X_test.shape[0])
print("Training target rows:", y_train.shape[0])
print("Testing target rows:", y_test.shape[0])

# ==========================================
# 5. CREATE AND TRAIN REGRESSION MODEL
# ==========================================

model = LinearRegression()

model.fit(X_train, y_train)

print("\n===== MODEL TRAINED =====")
print("Linear Regression model trained successfully!")

# ==========================================
# 6. MAKE TEST PREDICTIONS
# ==========================================

predictions = model.predict(X_test)

print("\n===== PREDICTIONS =====")
print("First 10 actual prices:")
print(y_test.head(10).to_numpy())

print("\nFirst 10 predicted prices:")
print(predictions[:10])

# ==========================================
# 7. EVALUATE - R²
# ==========================================

r2 = r2_score(y_test, predictions)

print("\n===== MODEL EVALUATION =====")
print("Test R²:", r2)

rmse = np.sqrt(mean_squared_error(y_test, predictions))

print("Test RMSE:", rmse)

# ==========================================
# 8. MODEL COEFFICIENTS
# ==========================================

print("\n===== MODEL COEFFICIENTS =====")

for feature, coefficient in zip(features, model.coef_):
    print(f"{feature}: {coefficient:.4f}")

print(f"Intercept: {model.intercept_:.4f}")

# ==========================================
# 9. PREDICTED VS ACTUAL PLOT
# ==========================================

plt.figure(figsize=(7, 5))

plt.scatter(y_test, predictions)

plt.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()]
)

plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Predicted vs Actual House Prices")
plt.grid(True)

plt.savefig("predicted_vs_actual.png")
plt.close()

print("\n===== FINAL GRAPH =====")
print("predicted_vs_actual.png created successfully!")