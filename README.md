# Bengaluru House Price Prediction

This repository contains a machine learning project that predicts real estate prices in Bengaluru, India. It takes various features into account, such as the location, total square footage, number of bedrooms (BHK), and bathrooms, to provide an estimated property price.

## Overview

The project walks through a complete end-to-end data science workflow:

1. **Data Loading & Inspection:** Importing the Bengaluru house prices dataset and analyzing its structure and missing values.
2. **Data Cleaning:** Handling null values and formatting specific columns (e.g., extracting numeric values from the `size` column to create a `bhk` column).
3. **Feature Engineering:** * Resolving range anomalies in the `total_sqft` column by taking the average.
* Creating a new `price_per_sqft` feature to help detect outliers.
* Grouping rare, low-frequency locations (less than 10 data points) into a general `'other'` category to reduce dimensionality.


4. **Outlier Removal:** * Filtering out properties where the price per square foot falls outside of one standard deviation from the mean for that specific location.
* Removing logical inconsistencies (e.g., homes where the number of bathrooms exceeds the number of bedrooms + 2).


5. **Model Building & Evaluation:** Creating a robust regression model to predict the final price.

## Technologies Used

* **Python 3**
* **Pandas & NumPy:** Data manipulation and numerical operations.
* **Matplotlib & Seaborn:** Data visualization.
* **Scikit-Learn:** Machine learning modeling, preprocessing, and evaluation.
* **Pickle:** Model serialization.

## Models Evaluated

The dataset was split into an 80/20 train-test ratio. Several regression algorithms were evaluated using R² Score, Mean Absolute Error (MAE), and Root Mean Squared Error (RMSE).

* **Linear Regression:** Achieved the best baseline performance ($R^2 \approx 0.6841$).
* **Ridge Regression:** $R^2 \approx 0.6825$
* **Lasso Regression:** $R^2 \approx 0.6158$
* **Random Forest Regressor:** $R^2 \approx 0.6069$
* **Gradient Boosting Regressor:** $R^2 \approx 0.5928$
* **Decision Tree Regressor:** $R^2 \approx 0.4262$

## Exported Files

To make the model usable in a production environment or web application, the following files are exported at the end of the notebook:

* `house_price_model.pkl`: The serialized Linear Regression model.
* `columns.pkl`: The list of feature columns (including the One-Hot Encoded locations) required for making predictions.

## How to Use

You can use the saved model and column list to predict prices in a separate Python script or backend server:

```python
import pickle
import numpy as np

# Load the model
with open('house_price_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Load the columns
with open('columns.pkl', 'rb') as f:
    columns = pickle.load(f)

# Example usage using the predict_price logic
# location = 'Whitefield', total_sqft = 1500, bath = 2, balcony = 1, bhk = 3

```
