# 🏠 Ames Housing Price Prediction: A Comparative Study
**Predicting Residential Home Values through Advanced Feature Engineering and Regularized Models**

## 📌 Project Overview
The objective of this project is to accurately predict the sale price for residential homes in Ames, Iowa. I developed a pipeline to explore high-dimensional data, engineer meaningful features, and compare the performance of standard linear models against LASSO regression.

## 🛠️ Tech Stack
* **Language:** Python
* **Environment:** Google Colab
* **Libraries:** `Pandas`, `NumPy`, `Scikit-Learn`, `Matplotlib`, `Seaborn`

## 📊 The Data
Using the Ames Housing dataset, I performed the following steps:
* **Cleaning:** Handled missing values and outliers in the `SalePrice` target variable.
* **Feature Engineering:** Introduced non-linearities and scaled features using `StandardScaler`.
* **Preprocessing:** Split data into training and testing sets using a 70/30 split.

## 💻 Core Code Snippet
```python
from sklearn.linear_model import LinearRegression, LassoCV
from sklearn.metrics import root_mean_squared_error

# Initializing models
ols_model = LinearRegression()
lasso_cv = LassoCV(cv=5, random_state=42)
