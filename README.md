# Day 3 - House Price Regression Model

## Project Objective

The objective of this project is to build a regression model that predicts house prices using house size, number of rooms, and house age.

## Dataset

A synthetic dataset containing 200 house observations was created because a dataset was not provided for the exercise.

The dataset contains four variables:

- `size` - House size
- `rooms` - Number of rooms
- `age` - House age
- `price` - House price (target variable)

## Data Exploration

The dataset was checked for:

- Missing values
- Duplicate rows
- Basic descriptive statistics
- Relationships between features and house price
- Potential outliers

There were no missing values and no duplicate rows.

## Features and Target

### Features

- Size
- Rooms
- Age

### Target

- Price

## Train/Test Split

The dataset was divided into:

- 75% training data - 150 observations
- 25% testing data - 50 observations

The test data was kept separate and was only used for final evaluation.

## Regression Model

A Multiple Linear Regression model from scikit-learn was used.

The model was trained using:

- House size
- Number of rooms
- House age

## Results

### Test Performance

- **R²: 0.9338**
- **RMSE: 25.86**

The model explains approximately 93.38% of the variation in house prices on the unseen test data.

### Model Coefficients

| Feature | Coefficient |
|---|---:|
| Size | +0.1205 |
| Rooms | +18.2684 |
| Age | -1.8393 |

The positive coefficients for size and rooms indicate that larger houses and houses with more rooms tend to have higher predicted prices. The negative coefficient for age indicates that older houses tend to have lower predicted prices. The predicted-versus-actual plot shows that most predictions are reasonably close to the ideal diagonal line. Overall, the model performs well on the held-back test data, although additional relevant features could potentially improve the model.

## Project Files

- `regression_model.py` - Complete Python regression program
- `houses.csv` - Dataset
- `size_vs_price.png` - Size versus price plot
- `rooms_vs_price.png` - Rooms versus price plot
- `age_vs_price.png` - Age versus price plot
- `house_boxplot.png` - Boxplot for outlier inspection
- `predicted_vs_actual.png` - Final predicted versus actual plot