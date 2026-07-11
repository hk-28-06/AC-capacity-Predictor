# ❄️ AC Capacity Predictor

A machine-learning web application that predicts the required Air Conditioner
capacity (in Tons of Refrigeration) for a room, based on its physical and
environmental characteristics.

Built for a college ML project — clean UI, transparent model, real dataset.

---

## 🗂️ Project Structure

```
ac_app/
├── data/
│   └── AC_Capacity_Dataset_2000.csv   # dataset (2000 rows)
├── model/
│   ├── linear_model.pkl               # trained Linear Regression model
│   ├── rf_model.pkl                   # trained Random Forest (benchmark)
│   └── artifact.json                  # metrics, coefficients, test predictions
├── generate_dataset.py                # (optional) synthetic data generator/fallback
├── train_model.py                     # trains & saves both models + metrics
├── app.py                             # Streamlit web application
├── requirements.txt
└── README.md
```

## ▶️ How to Run

**1. Install dependencies** (Python 3.9+ recommended):

```bash
pip install -r requirements.txt
```

**2. (Already done for you) Train the model** — only needed if you replace the
dataset or want to retrain:

```bash
python train_model.py
```

This regenerates `model/linear_model.pkl`, `model/rf_model.pkl`, and
`model/artifact.json`.

**3. Launch the app:**

```bash
streamlit run app.py
```

Streamlit will open the app automatically at `http://localhost:8501`.

---

## 🧠 What's Inside

| Tab | What it shows |
|---|---|
| 🔮 Predict | Interactive sliders for all 8 room parameters, live prediction with a gauge, recommended tonnage, and a per-feature contribution breakdown |
| 📊 Model Performance | MAE / RMSE / R² / 5-fold CV R², Actual-vs-Predicted plot, residual plot, feature coefficients |
| 🔍 Dataset Explorer | Raw data preview, correlation heatmap, target distribution |
| ℹ️ About | Methodology, modeling rationale, and disclaimers |

## 🛠️ Bugs Fixed vs. the Original Notebook

- `y_pred` was referenced (for `mean_absolute_error` / `r2_score`) **before**
  it was ever computed — reordered so predictions are generated first.
- Model was retrained from scratch every run with no persistence — now
  trained once via `train_model.py` and loaded instantly by the app
  (`joblib` + `st.cache_resource`).
- `Room_Volume` (present in the raw CSV) is intentionally excluded from
  training since it's a deterministic product of `Room_Area × Room_Height`
  and would introduce multicollinearity.

## 📈 Model Performance (on held-out 20% test set)

Run `python train_model.py` to reproduce — typical results:

- **MAE:** ~0.14 Ton
- **R²:** ~0.95
- **5-fold CV R²:** ~0.95

## ⚠️ Disclaimer

This tool gives an ML-based **engineering estimate**, not a certified HVAC
load calculation (e.g., Manual J). Always confirm final AC sizing with a
qualified HVAC professional.
