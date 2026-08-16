# House Price Regression Model

## 1. Project Objective

The objective of this project is to build and evaluate a regression model that predicts house prices using house size, number of rooms, and house age.

The project follows the six required steps:

1. Pick and explore the dataset
2. Prepare and split the data
3. Train a regression model
4. Evaluate the model
5. Interpret the results
6. Present the findings

---

## 2. Dataset

The project uses the `houses.csv` dataset provided for the workshop.

The dataset contains 200 house observations and four variables:

| Variable | Description | Role |
|---|---|---|
| `size` | House size | Feature |
| `rooms` | Number of rooms | Feature |
| `age` | House age | Feature |
| `price` | House price | Target |

---

## 3. Data Exploration

The dataset was explored using Python and Pandas.

The following checks were performed:

- Dataset shape
- First five observations
- Column names
- Descriptive statistics
- Missing values
- Duplicate rows
- Feature versus target relationships
- Potential outliers

The dataset contains:

- **200 observations**
- **4 variables**
- **No missing values**
- **No duplicate rows**

Exploratory plots were created for:

- House size vs price
- Number of rooms vs price
- House age vs price
- Boxplots for outlier inspection

---

## 4. Features and Target

### Features

The following three variables were selected as predictors:

- `size`
- `rooms`
- `age`

### Target

The target variable is:

- `price`

The model therefore predicts house price based on the size, number of rooms, and age of the house.

---

## 5. Train/Test Split

The dataset was divided into training and testing sets using a 75/25 split.

- **Training observations:** 150
- **Testing observations:** 50

The model was trained only on the 150 training observations.

The 50 testing observations were held back and used only for final evaluation.

This ensures that the model is evaluated on data that it did not use during training.

---

## 6. Regression Model

A Multiple Linear Regression model from scikit-learn was used.

The model was trained using:

- House size
- Number of rooms
- House age

The general regression equation is:

`price = intercept + (size × coefficient) + (rooms × coefficient) + (age × coefficient)`

---

## 7. Model Results

The model was evaluated using the held-back test dataset.

### Test Performance

| Metric | Result |
|---|---:|
| **R²** | **0.8854** |
| **RMSE** | **10.1496** |

The test R² of **0.8854** means that the model explains approximately **88.54% of the variation in house prices** on the unseen test data.

The RMSE is **10.1496**, meaning the model's prediction error is approximately 10.15 price units in RMSE terms.

---

## 8. Model Coefficients

| Feature | Coefficient |
|---|---:|
| `size` | **+3.1743** |
| `rooms` | **+15.3605** |
| `age` | **-1.4052** |
| Intercept | **23.4462** |

### Interpretation

The coefficient for `size` is positive, meaning that larger houses are associated with higher predicted prices, while keeping the other variables constant.

The coefficient for `rooms` is also positive, indicating that houses with more rooms tend to have higher predicted prices.

The coefficient for `age` is negative, meaning that older houses tend to have lower predicted prices when the other features are held constant.

All three coefficient signs are consistent with reasonable expectations for house prices.

---

## 9. Predicted vs Actual Results

A predicted-versus-actual plot was created using the 50 unseen test observations.

The closer the predicted values are to the diagonal reference line, the more accurate the predictions are.

The plot provides a visual check of the model's performance in addition to the numerical R² and RMSE metrics.

---

## 10. Conclusion

The multiple linear regression model performed well on the held-back test data, achieving a test R² of **0.8854** and an RMSE of **10.1496**. The model explains approximately 88.54% of the variation in house prices. The positive coefficients for size and rooms and the negative coefficient for age are sensible and align with expected relationships between these variables and house price. The model could potentially be improved by engineering additional useful features or testing other regression approaches.

---

## 11. Project Files

- `regression_model.py` - Complete Python regression program
- `houses.csv` - Lecturer-provided dataset
- `size_vs_price.png` - House size versus price plot
- `rooms_vs_price.png` - Number of rooms versus price plot
- `age_vs_price.png` - House age versus price plot
- `house_boxplot.png` - Boxplot used for outlier inspection
- `predicted_vs_actual.png` - Predicted versus actual test results plot
- `README.md` - Project documentation

---

## 12. Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Git
- GitHub