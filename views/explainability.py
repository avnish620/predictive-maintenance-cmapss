import matplotlib.pyplot as plt
import streamlit as st

from utils import ensure_model_loaded, calculate_shap_values

st.title("🔬 Explainability")

model, imputer, model_loaded = ensure_model_loaded()

if not model_loaded:
    st.stop()

st.write(
    "SHAP (SHapley Additive exPlanations) helps explain which "
    "features contribute most strongly to the Random Forest "
    "RUL predictions."
)

try:
    with st.spinner("Calculating SHAP feature importance..."):
        X_shap, shap_values, importance = calculate_shap_values()

    st.subheader("Global Feature Importance")
    st.write(
        "The chart below shows the average absolute SHAP value for the "
        "sampled validation observations. Higher values indicate greater "
        "contribution to the model's predictions."
    )

    top_features = importance.head(15).copy()
    top_features = top_features.sort_values("mean_abs_shap", ascending=True)

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh(top_features["feature"], top_features["mean_abs_shap"])
    ax.set_xlabel("Mean Absolute SHAP Value")
    ax.set_ylabel("Feature")
    ax.set_title("Top 15 Features Influencing RUL Prediction")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    st.subheader("Top SHAP Features")

    display_importance = importance.head(15).copy()
    display_importance["mean_abs_shap"] = display_importance["mean_abs_shap"].round(4)

    st.dataframe(display_importance, width='stretch', hide_index=True)

    top_feature = importance.iloc[0]["feature"]
    top_value = importance.iloc[0]["mean_abs_shap"]

    st.info(
        f"🔎 The most influential feature in the sampled validation data "
        f"is **{top_feature}**, with a mean absolute SHAP value of "
        f"approximately **{top_value:.2f}**."
    )

except Exception as e:
    st.error("❌ SHAP analysis could not be loaded.")
    st.code(str(e))
