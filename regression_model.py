import pandas as pd
import numpy as np

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


df = pd.read_csv("houses.csv")

print("===== DATASET LOADED =====")
print("Dataset loaded successfully!")


print("\n===== FIRST 5 ROWS =====")
print(df.head())

print("\n===== DATASET SHAPE =====")
print(df.shape)

print("\n===== COLUMN NAMES =====")
print(df.columns.tolist())

print("\n===== DESCRIPTIVE STATISTICS =====")
print(df.describe())

print("\n===== MISSING VALUES =====")
print(df.isnull().sum())

print("\n===== DUPLICATE ROWS =====")
print(df.duplicated().sum())


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


features = ["size", "rooms", "age"]

X = df[features]
y = df["price"]

print("\n===== FEATURES AND TARGET =====")
print("Features:", features)
print("Target: price")


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


model = LinearRegression()

model.fit(X_train, y_train)

print("\n===== MODEL TRAINED =====")
print("Linear Regression model trained successfully!")


predictions = model.predict(X_test)

print("\n===== PREDICTIONS =====")
print("First 10 actual prices:")
print(y_test.head(10).to_numpy())

print("\nFirst 10 predicted prices:")
print(predictions[:10])


r2 = r2_score(y_test, predictions)

rmse = np.sqrt(mean_squared_error(y_test, predictions))

print("\n===== MODEL EVALUATION =====")
print("Test R²:", r2)
print("Test RMSE:", rmse)

print("\n===== MODEL COEFFICIENTS =====")

for feature, coefficient in zip(features, model.coef_):
    print(f"{feature}: {coefficient:.4f}")

print(f"Intercept: {model.intercept_:.4f}")


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


print("\n===== FINAL SUMMARY =====")
print(f"Number of observations: {len(df)}")
print(f"Number of training observations: {len(X_train)}")
print(f"Number of testing observations: {len(X_test)}")
print(f"Test R²: {r2:.4f}")
print(f"Test RMSE: {rmse:.4f}")