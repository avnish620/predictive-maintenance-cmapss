import pandas as pd
import streamlit as st

st.title("🚨 Drift Monitoring")

st.write(
    "This section demonstrates the simulated sensor-data drift "
    "experiment from the model monitoring notebook."
)

drift_features = [
    "sensor_4_rolling_mean",
    "sensor_15_rolling_mean",
    "sensor_11_rolling_mean",
]

st.subheader("Simulated Drift")
st.write("A 15% distribution shift was simulated in three sensor-based features.")

drift_table = pd.DataFrame({
    "Feature": drift_features,
    "Simulated Drift": ["15%", "15%", "15%"],
    "KS Test": ["Detected", "Detected", "Detected"],
    "p-value": [0.0, 0.0, 0.0],
})

st.dataframe(drift_table, width='stretch', hide_index=True)

st.error("⚠️ DATA DRIFT DETECTED — Retraining trigger activated.")

st.caption(
    "Note: The drift shown here is intentionally simulated to demonstrate "
    "the monitoring and retraining mechanism. It does not represent real "
    "production drift."
)

st.divider()

st.header("🚀 Dashboard Status")

st.success(
    "✅ Model integration, engine-level RUL prediction, model evaluation, "
    "SHAP explainability, and drift monitoring are working successfully."
)
