# Fraud Detection using Machine Learning

An end-to-end fraud detection project using machine learning to identify fraudulent financial transactions under severe class imbalance.

## Overview

This project covers the complete ML workflow:

- Exploratory Data Analysis (EDA)
- Feature preprocessing and categorical encoding
- Stratified train/test splitting
- Imbalance handling using SMOTE and class weighting
- Model comparison
- Stratified cross-validation
- Randomized hyperparameter tuning
- Final model evaluation using Precision, Recall, F1 and PR-AUC
- Streamlit deployment

## Models

| Model | Precision | Recall | F1 | PR-AUC |
|---|---:|---:|---:|---:|
| Logistic Regression + SMOTE | 2.36% | 96.35% | 4.61% | 0.576 |
| Random Forest + Class Weight | 98.09% | 78.03% | 86.92% | 0.941 |
| Tuned XGBoost | **96.98%** | **83.99%** | **90.02%** | **0.966** |

### Final XGBoost Parameters

```python
n_estimators = 300
max_depth = 6
learning_rate = 0.05
