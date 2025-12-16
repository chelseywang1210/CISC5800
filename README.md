# CISC5800
final project 
# Predicting Broadway Weekly Box Office Revenue Using Machine Learning

## Overview
This project develops a machine learning framework to forecast one-week-ahead Broadway box office revenue.
The task is formulated as a supervised time-series regression problem, where historical weekly grosses
and engineered temporal features are used to predict future revenue.

The primary goal is to build a stable and interpretable forecasting model under limited data and strong
temporal constraints. An ensemble of XGBoost models is employed to capture non-linear demand dynamics
while avoiding temporal leakage.

## Data
The project uses real-world, publicly available data sources:

- **Playbill**: Weekly Broadway box office gross data by show
- **U.S. Office of Personnel Management (OPM)**: Federal holiday calendar
- **Internet Broadway Database (IBDB)**: Original cast return and star participation information

Due to licensing and size considerations, raw data files are not included in the repository.
All data sources are publicly accessible.

## Methods
The forecasting task is addressed using a feature-engineered XGBoost ensemble consisting of three models:
a conservative model, a balanced model, and a trend-oriented model. These models are combined via
weighted averaging to improve robustness and generalization.

Key methodological components include:
- Lag features and moving averages to capture short-term momentum
- Exponential moving averages (EMA) to model demand decay
- Lifecycle normalization features (e.g., percentage of first-week gross)
- TimeSeriesSplit cross-validation to prevent temporal leakage

## Results
The model achieves strong predictive performance across multiple Broadway productions.
Evaluation metrics include RMSE, MAE, R², and MAPE.

Results show that internal temporal dynamics explain most revenue variation, while sparse external
signals (e.g., cast returns) are difficult for tree-based models to learn under limited sample sizes.

## Repository Structure
.
├── data/
│   └── README.md          # Description of data sources
├── src/
│   └── xgboost_model.py   # Model training and evaluation code
├── notebooks/
│   └── exploration.ipynb  # Data exploration and visualization
├── report/
│   └── Final_Report.pdf   # IEEE-style project report
├── requirements.txt
└── README.md
