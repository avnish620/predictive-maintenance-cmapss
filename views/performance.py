import pandas as pd
import streamlit as st

from utils import (
    ensure_model_loaded,
    load_test_data,
    load_rul_ground_truth,
    engineer_features,
    prepare_model_input,
    predict_rul,
)

st.title("📈 Model Performance")

model, imputer, model_loaded = ensure_model_loaded()

if not model_loaded:
    st.stop()

st.subheader("Final Test Performance")

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("MAE", "21.52")
with col2:
    st.metric("RMSE", "30.20")
with col3:
    st.metric("R²", "0.472")

try:
    test_data = load_test_data()
    actual_rul = load_rul_ground_truth()

    test_features = engineer_features(test_data)

    last_rows = (
        test_features.sort_values(["engine_id", "cycle"])
        .groupby("engine_id")
        .tail(1)
        .copy()
    )

    X_test_final = prepare_model_input(last_rows, imputer)
    predicted_rul_test = predict_rul(model, imputer, X_test_final)

    st.subheader("Actual vs Predicted RUL")

    comparison = pd.DataFrame({
        "Actual RUL": actual_rul,
        "Predicted RUL": predicted_rul_test,
    })
    st.line_chart(comparison)

    comparison["Error"] = comparison["Predicted RUL"] - comparison["Actual RUL"]

    st.subheader("Prediction Error")
    st.bar_chart(comparison["Error"])

    st.subheader("Engine-wise Evaluation")

    evaluation = pd.DataFrame({
        "engine_id": last_rows["engine_id"].values,
        "actual_RUL": actual_rul,
        "predicted_RUL": predicted_rul_test,
    })
    evaluation["error"] = evaluation["predicted_RUL"] - evaluation["actual_RUL"]
    evaluation["absolute_error"] = evaluation["error"].abs()

    st.dataframe(evaluation, width='stretch')

except Exception as e:
    st.warning("Final evaluation visualization could not be generated.")
    st.code(str(e))
