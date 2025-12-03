House Loan Data Analysis & Default Prediction

A complete data analytics and machine learning project focused on predicting whether a loan applicant will default based on historical loan data.

Project Overview

This project analyzes the loan_data.csv dataset and builds predictive models to determine loan default risk.
It includes:

Data cleaning & preprocessing

Exploratory data analysis (EDA)

Handling missing values

Addressing class imbalance with SMOTE

Machine learning model training

Deep learning model using TensorFlow

Model evaluation & comparison

Key Objectives

Identify patterns in loan defaults

Understand key drivers (income, credit score, loan purpose, etc.)

Build predictive models for loan risk classification

Improve performance using oversampling (SMOTE)

Compare ML & Deep Learning accuracy

Tech Stack

Python

Pandas, NumPy

Matplotlib, Seaborn

Scikit-learn

SMOTE (Imbalanced-Learn)

TensorFlow / Keras

Project Structure
HouseLoanDataAnalysis.ipynb
loan_data.csv
README.md
assets/
   └── charts/   (optional)

Detailed Workflow
Data Loading & Cleaning

Load dataset (loan_data.csv)

Check and fill missing values

Numerical → mean

Categorical → mode

Remove duplicates

Convert required columns to numeric

Exploratory Data Analysis

Distribution of numerical features

Categorical variable analysis

Loan status vs features

Correlation heatmap

Handling Class Imbalance

Applied SMOTE to ensure balanced representation of default vs non-default cases.

Preprocessing

Train–test split

One-hot encoding for categorical features

Scaling numeric values

Machine Learning Models Trained

Logistic Regression

Random Forest

Gradient Boosting / XGBoost (if included)

SVM (if included)

Evaluated using:

Accuracy

Precision, Recall, F1-score

Confusion Matrix

Deep Learning (ANN)

Built a neural network using TensorFlow/Keras with:

Dense layers

Dropout for regularization

Binary classification output

Evaluated using accuracy & loss curves.

Results Summary

SMOTE improved minority-class prediction

ML models performed well for simple patterns

ANN provided strong predictive capability for complex relationships

Key features impacting loan default typically included:

Credit score

Income level

Debt-to-income ratio

Loan purpose

Interest rate

How to Run the Notebook
Install required libraries:
pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn tensorflow

Open and run the notebook:
jupyter notebook HouseLoanDataAnalysis.ipynb

Dataset

loan_data.csv contains fields like:

Credit score

Income

Age

Loan amount

Interest rate

Employment data

Loan status (target variable)

And other financial indicators

Future Enhancements

Hyperparameter tuning with GridSearchCV

Add SHAP explanations for model interpretability

Build Streamlit dashboard for loan officers

Try LSTM for time-based loan repayment patterns


Author


Pavan Chinta
GitHub: https://github.com/pavan123chinta
