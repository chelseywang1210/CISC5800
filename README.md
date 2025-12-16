# Predicting Broadway Weekly Box Office Revenue  
### A Course Project on Time-Series Forecasting with XGBoost


This project implements a generalizable machine learning framework for forecasting Broadway weekly box office revenue. The model automatically engineers time-series features, incorporates cast-return effects through a continuous decay mechanism, and produces next-week predictions along with confidence intervals.

This repository was developed as an individual course project for a graduate-level machine learning course.


---

## 1. Project Structure

```
project_root/
│
├── main.py                 # Main prediction script
├── README.md               # Project documentation
├── sample_input.xlsx       # Example input file
├── figures/                # Auto-generated plots
├── results/                # Saved text summaries
└── requirements.txt        # Python dependencies
```

---

## 2. Environment Requirements

### Python Version  
Python 3.8+ is recommended.

### Installation  
Install required packages:

```bash
pip install -r requirements.txt
```

Main dependencies:

* pandas
* numpy
* scikit-learn
* xgboost
* matplotlib

---

## 3. Input File Format

Your Excel file must contain **exactly these four columns**:

| week ending | This Week Gross | cast | holiday |
| ----------- | --------------- | ---- | ------- |
| 21-Nov-21   | 2715955         | 0    | 0       |
| 28-Nov-21   | 3170432         | 0    | 1       |

### Column descriptions

* **week ending** — Date of the week ending (format: dd-mmm-yy)
* **This Week Gross** — Weekly gross revenue (integer or float)
* **cast** — 1 if an original cast member returned this week; else 0
* **holiday** — 1 if the week includes a U.S. federal holiday; else 0

Rows should be sorted oldest → newest (the script will auto-sort if needed).

---

## 4. How to Run the Script

### Step 1 — Place your dataset in the project folder

Example names:

* Book
* `Book1.xlsx`
* `Book2.xlsx`
* `Book3.xlsx`
* `Book4.xlsx`

### Step 2 — Modify one line in `sample_eng.py`

Find the following line in `sample_eng.py`:

import pandas as pd
```python
df = pd.read_excel('Book1.xlsx')
```

Replace with your filename:

```python
df = pd.read_excel('Hamilton.xlsx')  # Replace with your file name
```

### Step 3 — Run the script:

```bash
python main.py
```

The script performs:

1. Data validation and cleaning
2. Automatic feature engineering (17 total features)
3. TimeSeriesSplit (5-fold) cross-validation
4. Training of a three-model XGBoost ensemble
5. Cast-boost modeling (continuous decay)
6. Prediction of next week's revenue + 95% CI
7. Auto-generation of plots and summary tables

---

## 5. Output Files

### 1. Console Summary

Displays in terminal:

* Data overview
* Cross-validation metrics
* Feature importances
* Cast effect contribution
* Next-week prediction with confidence interval

### 2. `results/summary.txt`

Full detailed report including fold-by-fold RMSE/MAE/MAPE metrics.

### 3. `figures/` Directory

Auto-generated figures:

* `actual_vs_predicted.png` — Actual vs predicted plot
* `feature_importance.png` — Feature importance chart
* `cast_decay_visualization.png` — Cast-decay visualization (if applicable)

---

## 6. Feature Engineering Overview

The script automatically creates the following 17 features:

### Temporal & Lag Features
* **Lagged values:** `gross_lag_1`, `gross_lag_2`
* **Moving averages:** `gross_ma_2`, `gross_ma_3`
* **Exponential moving average:** `ema_3`
* **Growth rate:** `gross_change_rate_1`
* **Normalized lifecycle:** `pct_of_first_week`
* **Temporal index:** `week_number`

### Holiday Features
* `holiday`
* `holiday_lag_1`

### Cast Effect Features
* `cast_boost` — Continuous decay function
* `weeks_since_cast` — Time since last cast event
* `cast_cumulative` — Cumulative cast events
* `cast × pct_of_first_week` — Interaction term
* `cast × ma2` — Interaction term
* `cast × week` — Interaction term

Total features: **17**

---

## 7. Notes on Cast Modeling

* Binary cast signals (`0` or `1`) are too sparse for XGBoost to learn reliably.
* This model replaces the binary indicator with a **continuous decay function**, enabling the learner to capture residual cast effects over multiple weeks.
* If your dataset has *no* cast events, the model automatically bypasses cast-based features.

---

## 8. Data Sources

This project uses publicly available Broadway information from:

* **Playbill** — [https://playbill.com](https://playbill.com)
* **Playbill News** — [https://playbill.com/news](https://playbill.com/news)
* **U.S. Office of Personnel Management (Federal Holidays)** — [https://opm.gov](https://opm.gov)
* **IBDB (Internet Broadway Database)** — [https://ibdb.com](https://ibdb.com)

Please cite these sources when reusing or publishing results.

---

## 9. License

This project is intended for academic and research use only.

---

## 10. Contact

**Author:** Chelsey (Xiqiao) Wang  
**Institution:** Fordham University — Graduate School of Arts and Sciences  
**Email:** [xw22@fordham.edu](mailto:xw22@fordham.edu)

For questions, issues, or collaboration inquiries, please contact via email.
