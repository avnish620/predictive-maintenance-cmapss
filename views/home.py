import streamlit as st

from utils import ensure_model_loaded, load_test_data

# ============================================================
# TITLE
# ============================================================

st.title("🔧 Predictive Maintenance Dashboard")
st.subheader("C-MAPSS FD001 — Turbofan Engine RUL Prediction")

st.write(
    "Predictive maintenance dashboard for turbofan engine health "
    "and Remaining Useful Life (RUL) prediction."
)

model, imputer, model_loaded = ensure_model_loaded()

if model_loaded:
    st.success("✅ Trained RUL model and preprocessing pipeline loaded successfully.")

st.divider()

# ============================================================
# PROJECT OVERVIEW
# ============================================================

st.header("📊 Project Overview")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Dataset", "NASA C-MAPSS")
with col2:
    st.metric("Subset", "FD001")
with col3:
    st.metric("RUL Model", "Random Forest")
with col4:
    st.metric("Features", "64")

try:
    test_data = load_test_data()
    n_engines = test_data["engine_id"].nunique()
    st.caption(f"FD001 test set loaded — {n_engines} engines available for prediction.")
except Exception:
    pass

st.divider()

# ============================================================
# NAVIGATION
# ============================================================

st.header("🧭 Explore the Dashboard")

st.write("Use the sidebar (or the cards below) to navigate between pages.")

nav_col1, nav_col2 = st.columns(2)

with nav_col1:
    with st.container(border=True):
        st.subheader("🔮 Predict RUL")
        st.write(
            "Predict Remaining Useful Life for a test engine, your own "
            "uploaded sensor data, or a manual what-if input."
        )
        st.page_link("views/predict.py", label="Go to Prediction", icon="🔮")

    with st.container(border=True):
        st.subheader("🔬 Explainability")
        st.write(
            "See which sensors and features drive the model's RUL "
            "predictions, using SHAP."
        )
        st.page_link("views/explainability.py", label="Go to Explainability", icon="🔬")

with nav_col2:
    with st.container(border=True):
        st.subheader("📈 Model Performance")
        st.write(
            "Review MAE / RMSE / R² and actual-vs-predicted RUL on the "
            "held-out test engines."
        )
        st.page_link("views/performance.py", label="Go to Performance", icon="📈")

    with st.container(border=True):
        st.subheader("🚨 Drift Monitoring")
        st.write(
            "Review the simulated sensor drift experiment and retraining "
            "trigger from the monitoring notebook."
        )
        st.page_link("views/drift.py", label="Go to Drift Monitoring", icon="🚨")

st.divider()

st.info("Project: NASA C-MAPSS FD001 Predictive Maintenance")
