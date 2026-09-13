# House Price Prediction

Machine Learning project for predicting residential house prices in Hanoi and Ho Chi Minh City using real-estate listing data collected through web scraping.

## Project Overview

This project builds an end-to-end Machine Learning pipeline to predict house prices based on property characteristics such as:

- Area
- Number of floors
- Number of bedrooms
- Property type
- District / City
- Front street width
- House frontage
- Available amenities

The project also analyzes feature importance to identify the factors that contribute most to house prices.

## Dataset

- Location: Hanoi and Ho Chi Minh City
- Property type: Residential houses
- Price range: Under 15 billion VND
- Source: Public real-estate listings collected through web scraping
- Task: Regression

## Machine Learning Pipeline

1. Data Cleaning
2. Exploratory Data Analysis (EDA)
3. Missing Data Handling
4. Feature Engineering
5. Stratified Sampling
6. Train/Test Split
7. Data Preprocessing
8. Model Training
9. Hyperparameter Optimization
10. Model Evaluation
11. Feature Importance Analysis
12. Model Persistence & Inference

### Data Processing

Key preprocessing techniques:

- Duplicate removal
- Missing-value analysis and hybrid imputation
- Feature engineering
- Log transformation for skewed variables
- RobustScaler for numerical features
- Stratified sampling based on house-price groups

The original dataset contained duplicated listings, which were removed to reduce the risk of data leakage between training and testing data.

## Models

Three regression models were evaluated:

- Ridge Regression
- Random Forest Regressor
- XGBoost Regressor

Hyperparameter optimization was performed using:

- `RandomizedSearchCV`
- 5-fold Cross-Validation

The main evaluation metrics are:

- MAE
- RMSE
- R² Score

## Results

| Model | R² | MAE | RMSE |
|---|---:|---:|---:|
| XGBoost | **0.6082** | **0.1984** | **0.2761** |
| Random Forest | 0.6006 | 0.2003 | 0.2788 |
| Ridge | 0.4731 | 0.2285 | 0.3203 |

### Best Model

**XGBoost Regressor**

- R²: **0.6082**
- MAE: **0.1984**
- RMSE: **0.2761**
- CV MAE: **0.1920**

XGBoost achieved the best overall performance among the evaluated models and was selected as the final model.

## Feature Importance

Feature importance analysis with XGBoost showed that the most influential features include:

1. District
2. Area
3. Front street width
4. Number of floors
5. Property type
6. Number of bedrooms

This indicates that **location, property size and accessibility** play a major role in house-price prediction.

## Inference

The final XGBoost model is saved using `joblib` as:

```text
house_price_xgb_package.pkl
