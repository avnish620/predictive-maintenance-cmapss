# 🔧 Predictive Maintenance — NASA C-MAPSS Turbofan RUL Prediction

> Predicting how many flight cycles a jet engine has left before it needs maintenance — before it tells you itself.

[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/built%20with-Streamlit-FF4B4B)](https://streamlit.io/)
[![scikit-learn](https://img.shields.io/badge/model-Random%20Forest-informational)](https://scikit-learn.org/)
[![SHAP](https://img.shields.io/badge/explainability-SHAP-purple)](https://github.com/shap/shap)

---

## ✈️ The problem

Every jet engine degrades a little more with every flight. Wait too long to service one and you risk an in-flight failure; service it too early and you're throwing away perfectly good engine life. **Predictive maintenance** tries to answer one question precisely: *given how an engine is behaving right now, how many cycles does it realistically have left?*

This project answers that question using **NASA's C-MAPSS FD001 dataset** — simulated turbofan engine degradation data — and turns the resulting model into an interactive dashboard anyone on a maintenance team could actually use, not just a notebook full of metrics.

---

## 📊 Results

| Metric | Value |
|---|---|
| **MAE** | 21.52 cycles |
| **RMSE** | 30.20 cycles |
| **R²** | 0.472 |
| Model | Random Forest Regressor (200 trees) |
| Engineered features | 64 (raw sensors + rolling mean / std / delta) |

On average, the model's Remaining Useful Life estimate lands within **~21 cycles** of the true failure point — enough signal to flag an engine for closer inspection well before it becomes an emergency.

---

## 🖥️ The dashboard

A multi-page Streamlit app, not just a single script:

| Page | What it does |
|---|---|
| **🏠 Home** | Project overview, dataset stats, navigation |
| **🔮 Predict RUL** | Three ways to get a prediction: pick a known test engine, upload your own sensor log for batch predictions, or manually enter sensor readings for a quick what-if estimate |
| **📈 Model Performance** | MAE / RMSE / R², actual-vs-predicted curves, per-engine error breakdown |
| **🔬 Explainability** | SHAP feature importance — which sensors actually drive the model's predictions |
| **🚨 Drift Monitoring** | Simulated sensor drift detection and retraining-trigger demo |

---

## 🧠 How it works

**Data → Features → Model → Explanation**, in that order:

1. **Raw signal**: 21 sensors + 3 operational settings, logged every cycle, for 100 engines run to failure.
2. **Feature engineering**: for each sensor, a rolling 5-cycle mean, rolling std, and cycle-over-cycle delta — turning raw readings into *trend* signals, since a sensor's trajectory matters more than any single value.
3. **Model**: a Random Forest Regressor trained to predict RUL directly from the latest engineered feature vector.
4. **Explainability**: SHAP's `TreeExplainer` quantifies exactly which features push a prediction up or down — because "the model says 43 cycles" isn't useful to a maintenance engineer without knowing *why*.

---

## 📂 Project structure

```
predictive-maintenance-cmapss/
├── app.py                    # Entrypoint — defines page navigation
├── utils.py                  # Shared: model loading, feature engineering, prediction logic
├── precompute_shap.py        # One-time script to cache SHAP values (see below)
├── requirements.txt
├── views/
│   ├── home.py
│   ├── predict.py            # The 3-tab prediction page
│   ├── performance.py
│   ├── explainability.py
│   └── drift.py
├── CMAPSSData/                # NASA C-MAPSS FD001 train/test/RUL files
├── rf_rul_model_joblib.pkl    # Trained model
├── rul_imputer_joblib.pkl     # Fitted imputer
├── X_val_imputed.csv          # Validation sample used for SHAP
└── 01–12_*.ipynb              # EDA → feature engineering → modeling → SHAP → drift → final eval
```

The 12 notebooks are the full research trail — from initial EDA through sensor degradation analysis, RUL regression, classification baselines, SHAP explainability, drift simulation, and final held-out evaluation. `app.py` is where that research becomes something usable.

---

## 🚀 Running it locally

```bash
git clone https://github.com/avnish620/predictive-maintenance-cmapss.git
cd predictive-maintenance-cmapss

python -m venv venv
venv\Scripts\Activate.ps1        # Windows PowerShell
# source venv/bin/activate       # macOS/Linux

pip install -r requirements.txt

# optional but recommended — precomputes SHAP values so the
# Explainability page loads instantly instead of recomputing
# (this Random Forest has 200 trees, ~9k leaves each — SHAP is slow live)
python precompute_shap.py

streamlit run app.py
```

The app opens at `http://localhost:8501`.

---

## 🌐 Live demo

*(add your Streamlit Community Cloud link here once deployed)*

```
🔗 https://your-app-name.streamlit.app
```

---

## 🛠️ Tech stack

- **Modeling**: scikit-learn (Random Forest), pandas, NumPy
- **Explainability**: SHAP
- **Dashboard**: Streamlit
- **Data**: [NASA C-MAPSS Turbofan Engine Degradation Simulation](https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/) (FD001 subset)

---

## 📈 Possible next steps

- [ ] Swap Random Forest for a gradient-boosted model (XGBoost/LightGBM) and compare
- [ ] Extend to the FD002–FD004 subsets (multiple operating conditions / fault modes)
- [ ] Real drift detection against live incoming data instead of a simulated demo
- [ ] FastAPI backend + React frontend for a production-grade deployment

---

## 🙏 Acknowledgments

Dataset courtesy of NASA's Prognostics Center of Excellence (PCoE) Data Repository. This project is for educational/portfolio purposes.

---

## 📄 License

*(add your license here — e.g. MIT)*
