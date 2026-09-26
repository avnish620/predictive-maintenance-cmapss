"""
Shared utilities for the Predictive Maintenance Streamlit dashboard.

Every page in `pages/` imports from this module so that data loading,
feature engineering and prediction logic live in exactly one place.
"""

import os

import joblib
import numpy as np
import pandas as pd
import streamlit as st

# ============================================================
# PROJECT PATHS
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "rf_rul_model_joblib.pkl")
IMPUTER_PATH = os.path.join(BASE_DIR, "rul_imputer_joblib.pkl")
X_VAL_PATH = os.path.join(BASE_DIR, "X_val_imputed.csv")
DATA_DIR = os.path.join(BASE_DIR, "CMAPSSData")

TEST_DATA_PATH = os.path.join(DATA_DIR, "test_FD001.txt")
RUL_DATA_PATH = os.path.join(DATA_DIR, "RUL_FD001.txt")

# Columns dropped during training (near-constant / uninformative sensors)
DROPPED_SENSOR_COLS = [
    "sensor_1", "sensor_5", "sensor_10",
    "sensor_16", "sensor_18", "sensor_19",
]

RAW_COLUMN_NAMES = (
    ["engine_id", "cycle", "setting_1", "setting_2", "setting_3"]
    + [f"sensor_{i}" for i in range(1, 22)]
)

ROLLING_WINDOW = 5


# ============================================================
# MODEL / IMPUTER LOADING
# ============================================================

@st.cache_resource(show_spinner="Loading trained model...")
def load_model():
    """Load the trained Random Forest RUL model and its imputer."""
    model = joblib.load(MODEL_PATH)
    imputer = joblib.load(IMPUTER_PATH)
    return model, imputer


def get_expected_features(imputer):
    """Return the ordered list of feature names the model/imputer expect."""
    return list(imputer.feature_names_in_)


# ============================================================
# RAW DATA LOADING
# ============================================================

@st.cache_data(show_spinner="Loading FD001 test data...")
def load_test_data():
    """Load and lightly clean the FD001 test set."""
    df = pd.read_csv(
        TEST_DATA_PATH,
        sep=r"\s+",
        header=None,
        names=RAW_COLUMN_NAMES,
    )
    df = df.dropna(axis=1, how="all")
    df = df.drop(columns=DROPPED_SENSOR_COLS, errors="ignore")
    return df


@st.cache_data(show_spinner=False)
def load_rul_ground_truth():
    """Load the true RUL at the final cycle of every FD001 test engine."""
    rul_values = pd.read_csv(RUL_DATA_PATH, sep=r"\s+", header=None)
    return rul_values.iloc[:, 0].values


@st.cache_data(show_spinner=False)
def load_shap_background():
    """Load the validation sample used as a SHAP background/reference set."""
    return pd.read_csv(X_VAL_PATH)


# ============================================================
# FEATURE ENGINEERING
# ============================================================

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add rolling mean, rolling std and cycle-over-cycle delta features
    for every sensor column, grouped per engine_id.

    Mirrors the feature engineering used to train the model
    (see notebooks 03 / 09). Expects an `engine_id` column and, ideally,
    several rows (cycles) per engine so the rolling stats are meaningful.
    """
    feature_data = df.copy()

    sensor_columns = [c for c in feature_data.columns if c.startswith("sensor_")]

    for sensor in sensor_columns:
        grouped = feature_data.groupby("engine_id")[sensor]

        feature_data[f"{sensor}_rolling_mean"] = grouped.transform(
            lambda x: x.rolling(window=ROLLING_WINDOW, min_periods=1).mean()
        )
        feature_data[f"{sensor}_rolling_std"] = grouped.transform(
            lambda x: x.rolling(window=ROLLING_WINDOW, min_periods=1).std()
        )
        feature_data[f"{sensor}_delta"] = grouped.diff()

    return feature_data


def prepare_model_input(row_df: pd.DataFrame, imputer) -> pd.DataFrame:
    """
    Reindex a single-row (or multi-row) feature dataframe to exactly the
    columns the imputer/model expect, in the right order. Missing columns
    are created as NaN so the imputer can fill them.
    """
    expected_features = get_expected_features(imputer)

    X = row_df.copy()
    X = X.drop(columns=["engine_id", "RUL"], errors="ignore")

    # Add any missing expected columns as NaN, then select/order them.
    for col in expected_features:
        if col not in X.columns:
            X[col] = np.nan

    return X[expected_features]


def predict_rul(model, imputer, X: pd.DataFrame) -> np.ndarray:
    """Impute missing values and run the RUL model. Clips at 0."""
    X_imputed = imputer.transform(X)
    preds = model.predict(X_imputed)
    return np.clip(preds, 0, None)


# ============================================================
# SHAP
# ============================================================

@st.cache_data(show_spinner="Calculating SHAP feature importance...")
def calculate_shap_values():
    import shap

    X_val = load_shap_background()
    model, _ = load_model()

    X_shap = X_val.sample(n=min(200, len(X_val)), random_state=42)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_shap)

    if isinstance(shap_values, list):
        shap_values = shap_values[0]

    mean_abs_shap = np.abs(shap_values).mean(axis=0)

    importance = pd.DataFrame({
        "feature": X_shap.columns,
        "mean_abs_shap": mean_abs_shap,
    }).sort_values("mean_abs_shap", ascending=False)

    return X_shap, shap_values, importance


# ============================================================
# UI HELPERS
# ============================================================

def rul_status(predicted_rul: float):
    """
    Return (emoji_label, severity) for a predicted RUL value, used to
    keep the health messaging consistent across every page.
    severity is one of 'good', 'warning', 'critical'.
    """
    if predicted_rul > 100:
        return "✅ Healthy", "good"
    elif predicted_rul > 50:
        return "⚠️ Moderate", "warning"
    else:
        return "🚨 Needs attention", "critical"


def render_rul_message(predicted_rul: float):
    """Render the standard success/warning/error banner for a RUL value."""
    if predicted_rul > 100:
        st.success("✅ Engine currently shows relatively higher remaining useful life.")
    elif predicted_rul > 50:
        st.warning("⚠️ Engine shows moderate remaining useful life.")
    else:
        st.error("🚨 Engine may require closer maintenance attention.")


def ensure_model_loaded():
    """
    Load the model/imputer and surface a consistent error banner on
    failure. Returns (model, imputer, loaded: bool).
    """
    try:
        model, imputer = load_model()
        return model, imputer, True
    except Exception as e:
        st.error("❌ Error while loading the trained model.")
        st.code(str(e))
        return None, None, False
