"""
train_model.py
---------------
Trains the AC Capacity prediction model, evaluates it, and persists
everything the Streamlit app needs (model, metrics, feature stats).

Fixes applied vs. the original notebook:
  - y_pred is computed BEFORE it's used for mae/r2 (was reversed)
  - model + metrics are saved to disk so the app doesn't retrain every run
  - adds a second model (Random Forest) for comparison
  - reports 5-fold cross-validated R^2, not just a single train/test split

Run:  python train_model.py
"""

import json
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

FEATURES = [
    "Room_Area",
    "Room_Height",
    "Occupancy",
    "Outdoor_Temperature",
    "Window_Area",
    "Equipment_Load",
    "Insulation_Level",
    "Sun_Exposure",
]
TARGET = "AC_Capacity_Ton"

df = pd.read_csv("data/AC_Capacity_Dataset_2000.csv")
X = df[FEATURES]
y = df[TARGET]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ---- Model 1: Linear Regression (primary / interpretable model) --------
lin_model = LinearRegression()
lin_model.fit(X_train, y_train)
y_pred_lin = lin_model.predict(X_test)          # computed BEFORE use (bug fix)

lin_metrics = {
    "MAE": float(mean_absolute_error(y_test, y_pred_lin)),
    "RMSE": float(np.sqrt(mean_squared_error(y_test, y_pred_lin))),
    "R2": float(r2_score(y_test, y_pred_lin)),
    "CV_R2_mean": float(cross_val_score(lin_model, X, y, cv=5, scoring="r2").mean()),
}

# ---- Model 2: Random Forest (comparison / non-linear benchmark) --------
rf_model = RandomForestRegressor(n_estimators=300, max_depth=8, random_state=42)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)

rf_metrics = {
    "MAE": float(mean_absolute_error(y_test, y_pred_rf)),
    "RMSE": float(np.sqrt(mean_squared_error(y_test, y_pred_rf))),
    "R2": float(r2_score(y_test, y_pred_rf)),
    "CV_R2_mean": float(cross_val_score(rf_model, X, y, cv=5, scoring="r2").mean()),
}

print("=" * 50)
print("Linear Regression")
for k, v in lin_metrics.items():
    print(f"  {k:12s}: {v:.4f}")
print("-" * 50)
print("Random Forest (benchmark)")
for k, v in rf_metrics.items():
    print(f"  {k:12s}: {v:.4f}")
print("=" * 50)

# ---- Persist everything the app needs -----------------------------------
joblib.dump(lin_model, "model/linear_model.pkl")
joblib.dump(rf_model, "model/rf_model.pkl")

feature_coefs = dict(zip(FEATURES, lin_model.coef_.tolist()))

artifact = {
    "features": FEATURES,
    "target": TARGET,
    "linear_metrics": lin_metrics,
    "rf_metrics": rf_metrics,
    "linear_intercept": float(lin_model.intercept_),
    "linear_coefficients": feature_coefs,
    "feature_ranges": {
        f: {"min": float(df[f].min()), "max": float(df[f].max()), "mean": float(df[f].mean())}
        for f in FEATURES
    },
    "y_test": y_test.tolist(),
    "y_pred_linear": y_pred_lin.tolist(),
    "y_pred_rf": y_pred_rf.tolist(),
}

with open("model/artifact.json", "w") as f:
    json.dump(artifact, f, indent=2)

print("\nSaved: model/linear_model.pkl, model/rf_model.pkl, model/artifact.json")
