"""
XGBoost Gross Prediction - Simplified Optimized Version
Using only 10 core features for efficiency and robustness
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
import matplotlib.pyplot as plt
import warnings


warnings.filterwarnings('ignore')

df = pd.read_excel('Book3.xlsx')

# 2. Core Feature Engineering
def create_core_features(df):
    """
    Create core features (importance > 1%)
    """
    df_features = df.copy()

    # 1-3. Moving averages (most important features)
    df_features['gross_ma_2'] = df_features['This Week Gross'].rolling(window=2).mean()
    df_features['gross_ma_3'] = df_features['This Week Gross'].rolling(window=3).mean()

    # 4-5. Lag features
    df_features['gross_lag_1'] = df_features['This Week Gross'].shift(1)
    df_features['gross_lag_2'] = df_features['This Week Gross'].shift(2)

    # 6. Exponential moving average
    df_features['ema_3'] = df_features['This Week Gross'].ewm(span=3, adjust=False).mean()

    # 7. Relative to first week
    first_week_gross = df_features['This Week Gross'].iloc[0]
    df_features['pct_of_first_week'] = df_features['This Week Gross'] / first_week_gross

    # 8. Week index
    df_features['week_number'] = range(1, len(df_features) + 1)

    # 9. Weekly percentage change
    df_features['gross_change_rate_1'] = df_features['This Week Gross'].pct_change(1)

    # 10. Cast (external factor)
    if "cast" in df_features.columns:
        df_features["cast"] = df_features["cast"].astype(float)
        df_features["cast_lag_1"] = df_features["cast"].shift(1)
    else:
        df_features["cast"] = 0
        df_features["cast_lag_1"] = 0

    # 11. Holiday indicator
    if "holiday" in df_features.columns:
        df_features["holiday"] = df_features["holiday"].astype(float)
        df_features["holiday_lag_1"] = df_features["holiday"].shift(1)
    else:
        df_features["holiday"] = 0
        df_features["holiday_lag_1"] = 0

    return df_features


# 3. Train Ensemble Models
def train_ensemble_models(X_train, y_train):
    """
    Train 3 complementary XGBoost models
    """
    models = []

    # Model 1: Conservative (avoid overfitting)
    model1 = xgb.XGBRegressor(
        n_estimators=150,
        learning_rate=0.03,
        max_depth=3,
        min_child_weight=5,
        subsample=0.7,
        colsample_bytree=0.8,
        gamma=1,
        reg_alpha=0.1,
        reg_lambda=1,
        random_state=42,
        objective='reg:squarederror'
    )

    # Model 2: Balanced
    model2 = xgb.XGBRegressor(
        n_estimators=200,
        learning_rate=0.02,
        max_depth=4,
        min_child_weight=3,
        subsample=0.8,
        colsample_bytree=0.8,
        gamma=0.5,
        reg_alpha=0.05,
        reg_lambda=0.5,
        random_state=123,
        objective='reg:squarederror'
    )

    # Model 3: Trend-focused
    model3 = xgb.XGBRegressor(
        n_estimators=100,
        learning_rate=0.05,
        max_depth=2,
        min_child_weight=7,
        subsample=0.6,
        colsample_bytree=0.7,
        gamma=2,
        reg_alpha=0.2,
        reg_lambda=2,
        random_state=456,
        objective='reg:squarederror'
    )

    for model in [model1, model2, model3]:
        model.fit(X_train, y_train)
        models.append(model)

    return models


def ensemble_predict(models, X):
    """
    Weighted ensemble prediction
    Weights: 50% conservative, 30% balanced, 20% trend
    """
    predictions = np.column_stack([model.predict(X) for model in models])
    weights = np.array([0.5, 0.3, 0.2])
    return np.average(predictions, axis=1, weights=weights)


def predict_with_confidence(models, X):
    """
    Prediction with 95% confidence interval
    """
    predictions = np.column_stack([model.predict(X) for model in models])
    mean_pred = predictions.mean(axis=1)
    std_pred = predictions.std(axis=1)

    lower_bound = mean_pred - 1.96 * std_pred
    upper_bound = mean_pred + 1.96 * std_pred

    return mean_pred, lower_bound, upper_bound


#4. Main Program
print("=" * 60)
print("XGBoost Gross Prediction - Simplified Optimized Version")
print("=" * 60)

df_with_features = create_core_features(df)
df_model = df_with_features.dropna().reset_index(drop=True)

core_feature_cols = [
    'gross_ma_3',
    'gross_ma_2',
    'gross_lag_1',
    'pct_of_first_week',
    'ema_3',
    'gross_lag_2',
    'cast',
    'week_number',
    'gross_change_rate_1',
    'holiday',
]

X = df_model[core_feature_cols]
y = df_model['This Week Gross']

print("\n Data Overview:")
print(f"   Samples: {len(X)}")
print(f"   Features: {len(core_feature_cols)}")
print(f"   Gross range: {y.min():,.0f} - {y.max():,.0f}")
print(f"   Mean gross: {y.mean():,.0f}")

# 5. Time-Series Cross-Validation
print("\n" + "=" * 60)
print("Time-Series Cross-Validation (5-Fold)")
print("=" * 60)

tscv = TimeSeriesSplit(n_splits=5)

fold_results = []
all_predictions = []
all_actuals = []

for fold, (train_index, test_index) in enumerate(tscv.split(X)):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    models = train_ensemble_models(X_train, y_train)

    y_train_pred = ensemble_predict(models, X_train)
    y_test_pred = ensemble_predict(models, X_test)

    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    test_mae = mean_absolute_error(y_test, y_test_pred)
    test_mape = np.mean(np.abs((y_test - y_test_pred) / y_test)) * 100

    fold_results.append({
        'fold': fold + 1,
        'train_rmse': train_rmse,
        'test_rmse': test_rmse,
        'test_mae': test_mae,
        'test_mape': test_mape,
        'test_mean': y_test.mean(),
        'test_std': y_test.std()
    })

    all_predictions.extend(y_test_pred)
    all_actuals.extend(y_test)

    print(f"\nFold {fold + 1}:")
    print(f"  Train RMSE: {train_rmse:>10,.0f}")
    print(f"  Test RMSE:  {test_rmse:>10,.0f}")
    print(f"  Test MAE:   {test_mae:>10,.0f}")
    print(f"  Test MAPE:  {test_mape:>10.2f}%")
    print(f"  Samples: {len(y_test)}, Mean: {y_test.mean():,.0f}, Std: {y_test.std():,.0f}")

# 6. Outlier Fold Detection
print("\n" + "=" * 60)
print("Outlier Fold Detection")
print("=" * 60)

test_rmses = [r['test_rmse'] for r in fold_results]
Q1 = np.percentile(test_rmses, 25)
Q3 = np.percentile(test_rmses, 75)
IQR = Q3 - Q1
outlier_threshold = Q3 + 1.5 * IQR

outlier_folds = []
normal_rmses = []

for result in fold_results:
    if result['test_rmse'] > outlier_threshold:
        outlier_folds.append(result['fold'])
        print(f"  Fold {result['fold']} is an outlier")
        print(f"    RMSE: {result['test_rmse']:,.0f} (threshold: {outlier_threshold:,.0f})")
        print(f"    Reason: Std {result['test_std']:,.0f} significantly higher than others")
    else:
        normal_rmses.append(result['test_rmse'])

if len(normal_rmses) > 0:
    print(f"\n Average Test RMSE after removing outliers: {np.mean(normal_rmses):,.0f}")
else:
    print("\n No outliers detected")


# 7. Overall Evaluation
print("\n" + "=" * 60)
print("Overall Model Performance")
print("=" * 60)

overall_rmse = np.sqrt(mean_squared_error(all_actuals, all_predictions))
overall_mae = mean_absolute_error(all_actuals, all_predictions)
overall_r2 = r2_score(all_actuals, all_predictions)
overall_mape = np.mean(np.abs((np.array(all_actuals) - np.array(all_predictions)) / np.array(all_actuals))) * 100

print(f"\nAggregated Performance:")
print(f"  RMSE:  {overall_rmse:>10,.0f}")
print(f"  MAE:   {overall_mae:>10,.0f}")
print(f"  R²:    {overall_r2:>10.4f}")
print(f"  MAPE:  {overall_mape:>10.2f}%")

if overall_mape < 5:
    rating = "Excellent"
elif overall_mape < 8:
    rating = "Good"
elif overall_mape < 12:
    rating = "Acceptable "
else:
    rating = "Needs Improvement"

print(f"\nModel Rating: {rating}")

# 8. Feature Importance
print("\n" + "=" * 60)
print("Feature Importance")
print("=" * 60)

final_models = train_ensemble_models(X, y)
final_model = final_models[0]

feature_importance = pd.DataFrame({
    'feature': core_feature_cols,
    'importance': final_model.feature_importances_
}).sort_values('importance', ascending=False)

print("\nCore Feature Importances:")
for idx, row in feature_importance.iterrows():
    print(f"  {row['feature']:20s} {row['importance']:6.2%}")

# (bar display removed)



# 9. Predict Next Week
print("Next Week Gross Prediction")


df_pred = create_core_features(df)
last_row = df_pred.iloc[[-1]][core_feature_cols]

next_mean, next_lower, next_upper = predict_with_confidence(final_models, last_row)

uncertainty_pct = ((next_upper[0] - next_lower[0]) / (2 * next_mean[0])) * 100

print("\nPrediction:")
print(f"  Forecast:      {next_mean[0]:>12,.0f}")
print(f"  95% CI:        [{next_lower[0]:>10,.0f}, {next_upper[0]:>10,.0f}]")
print(f"  Uncertainty:   ±{(next_upper[0] - next_lower[0]) / 2:>10,.0f} ({uncertainty_pct:.1f}%)")

if uncertainty_pct < 5:
    confidence = "High Confidence"
elif uncertainty_pct < 10:
    confidence = "Medium Confidence"
else:
    confidence = "Low Confidence"

print(f"\nPrediction Confidence: {confidence}")





# Convert to numpy arrays for safety
actual = np.array(all_actuals)
pred = np.array(all_predictions)

plt.figure(figsize=(14, 6))
plt.plot(actual, label='Actual Gross', linewidth=2)
plt.plot(pred, label='Predicted Gross', linewidth=2, linestyle='--')

plt.title('Actual vs Predicted Weekly Gross', fontsize=16, fontweight='bold')
plt.xlabel('Week Index', fontsize=12)
plt.ylabel('Gross ($)', fontsize=12)
plt.legend()
plt.grid(True, alpha=0.3)

plt.show()
