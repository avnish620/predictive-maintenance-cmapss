import io

import pandas as pd
import streamlit as st

from utils import (
    RAW_COLUMN_NAMES,
    DROPPED_SENSOR_COLS,
    TEST_DATA_PATH,
    ensure_model_loaded,
    load_test_data,
    engineer_features,
    prepare_model_input,
    predict_rul,
    rul_status,
    render_rul_message,
)

st.title("🔮 Predict Remaining Useful Life")
st.write(
    "Estimate how many operating cycles an engine has left before "
    "maintenance is likely to be required."
)

model, imputer, model_loaded = ensure_model_loaded()

if not model_loaded:
    st.stop()

tab_engine, tab_upload, tab_manual = st.tabs(
    ["🛠 Select Test Engine", "📁 Upload Your Own Data", "🎛 Manual What-If Input"]
)

# ============================================================
# TAB 1 — SELECT A KNOWN TEST ENGINE
# ============================================================

with tab_engine:
    try:
        test_data = load_test_data()
        test_engines = sorted(test_data["engine_id"].unique())
    except Exception as e:
        test_data = None
        test_engines = []
        st.error("❌ Unable to load FD001 test data.")
        st.code(str(e))

    if test_engines:
        selected_engine = st.selectbox("Select Engine", test_engines, key="engine_select")

        engine_data = (
            test_data[test_data["engine_id"] == selected_engine]
            .copy()
            .sort_values("cycle")
        )

        feature_data = engineer_features(engine_data)
        latest_row = feature_data.iloc[[-1]].copy()
        current_cycle = int(latest_row["cycle"].iloc[0])

        X_engine = prepare_model_input(latest_row, imputer)
        predicted_rul = predict_rul(model, imputer, X_engine)[0]

        st.subheader("Engine Health Summary")

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Selected Engine", selected_engine)
        with col2:
            st.metric("Current Cycle", current_cycle)
        with col3:
            st.metric("Predicted RUL", f"{predicted_rul:.1f} cycles")

        st.progress(min(int(predicted_rul), 200) / 200)
        render_rul_message(predicted_rul)

        st.subheader("📈 Engine Sensor History")

        sensor_columns = [c for c in engine_data.columns if c.startswith("sensor_")]
        selected_sensor = st.selectbox(
            "Select sensor to visualize", sensor_columns, key="engine_sensor_select"
        )

        sensor_history = engine_data[["cycle", selected_sensor]].set_index("cycle")
        st.line_chart(sensor_history)
    else:
        st.warning("No test engines available.")

# ============================================================
# TAB 2 — UPLOAD YOUR OWN DATA (BATCH PREDICTION)
# ============================================================

with tab_upload:
    st.write(
        "Upload a CMAPSS-format sensor log — whitespace-separated, **no header "
        "row**, one row per engine per cycle — to get a Remaining Useful Life "
        "prediction for every engine's most recent cycle."
    )

    expected_cols_str = ", ".join(RAW_COLUMN_NAMES)
    with st.expander("Expected file format"):
        st.write(f"**Columns (in order, {len(RAW_COLUMN_NAMES)} total):**")
        st.code(expected_cols_str, language="text")
        st.write(
            "Each row is one engine at one cycle. Include the full cycle "
            "history for every engine you want predictions for — the model "
            "uses a rolling 5-cycle window, so more history per engine gives "
            "a more accurate prediction for its latest cycle."
        )

        try:
            sample_df = pd.read_csv(
                TEST_DATA_PATH,
                sep=r"\s+",
                header=None,
                names=RAW_COLUMN_NAMES,
            ).head(20)
            sample_csv = sample_df.to_csv(sep=" ", header=False, index=False).encode("utf-8")
            st.download_button(
                "⬇️ Download sample template (.txt)",
                data=sample_csv,
                file_name="sample_engine_data.txt",
                mime="text/plain",
            )
        except Exception:
            pass

    uploaded_file = st.file_uploader(
        "Upload sensor log", type=["txt", "csv"], key="batch_upload"
    )

    if uploaded_file is not None:
        try:
            raw_bytes = uploaded_file.read()
            uploaded_df = pd.read_csv(
                io.BytesIO(raw_bytes),
                sep=r"\s+",
                header=None,
                names=RAW_COLUMN_NAMES,
                usecols=range(len(RAW_COLUMN_NAMES)),
            )
            uploaded_df = uploaded_df.dropna(axis=1, how="all")
            uploaded_df = uploaded_df.drop(columns=DROPPED_SENSOR_COLS, errors="ignore")

            uploaded_features = engineer_features(uploaded_df)

            last_rows = (
                uploaded_features.sort_values(["engine_id", "cycle"])
                .groupby("engine_id")
                .tail(1)
                .copy()
            )

            X_batch = prepare_model_input(last_rows, imputer)
            predictions = predict_rul(model, imputer, X_batch)

            results = pd.DataFrame({
                "engine_id": last_rows["engine_id"].values,
                "last_cycle": last_rows["cycle"].values,
                "predicted_RUL": predictions.round(1),
            })
            results["status"] = results["predicted_RUL"].apply(
                lambda r: rul_status(r)[0]
            )

            st.success(f"✅ Generated predictions for {len(results)} engine(s).")
            st.dataframe(results, width='stretch', hide_index=True)

            csv_bytes = results.to_csv(index=False).encode("utf-8")
            st.download_button(
                "⬇️ Download predictions as CSV",
                data=csv_bytes,
                file_name="rul_predictions.csv",
                mime="text/csv",
            )

        except Exception as e:
            st.error(
                "❌ Could not parse this file. Please check that it matches "
                "the expected format above."
            )
            st.code(str(e))

# ============================================================
# TAB 3 — MANUAL WHAT-IF INPUT
# ============================================================

with tab_manual:
    st.write(
        "Enter sensor readings for a single cycle to get a quick RUL "
        "estimate — useful for exploring 'what-if' scenarios without a "
        "full cycle history."
    )
    st.caption(
        "Note: rolling-window trend features (mean / std / delta over the "
        "last 5 cycles) can't be computed from a single reading, so this "
        "estimate is approximate — it relies on the model's typical values "
        "for those trend features."
    )

    try:
        reference_data = load_test_data()
        used_sensors = [
            c for c in reference_data.columns if c.startswith("sensor_")
        ]
        sensor_stats = reference_data[used_sensors].agg(["min", "max", "median"])
        setting_stats = reference_data[["setting_1", "setting_2", "setting_3"]].agg(
            ["min", "max", "median"]
        )
        cycle_max = int(reference_data["cycle"].max())
    except Exception:
        reference_data = None
        used_sensors = []
        sensor_stats = None
        setting_stats = None
        cycle_max = 300

    with st.form("manual_prediction_form"):
        st.subheader("Operating Settings")
        s_col1, s_col2, s_col3, s_col4 = st.columns(4)

        with s_col1:
            manual_cycle = st.number_input(
                "Cycle", min_value=1, max_value=cycle_max * 2, value=100
            )

        setting_inputs = {}
        for i, col in enumerate(["setting_1", "setting_2", "setting_3"], start=2):
            with [s_col2, s_col3, s_col4][i - 2]:
                default_val = (
                    float(setting_stats.loc["median", col]) if setting_stats is not None else 0.0
                )
                setting_inputs[col] = st.number_input(
                    col.replace("_", " ").title(), value=round(default_val, 3), format="%.3f"
                )

        st.subheader("Sensor Readings")
        st.caption("Pre-filled with typical (median) values from the FD001 test set — adjust as needed.")

        sensor_inputs = {}
        sensor_grid = st.columns(3)
        for idx, sensor in enumerate(used_sensors):
            default_val = (
                float(sensor_stats.loc["median", sensor]) if sensor_stats is not None else 0.0
            )
            with sensor_grid[idx % 3]:
                sensor_inputs[sensor] = st.number_input(
                    sensor.replace("_", " ").title(),
                    value=round(default_val, 3),
                    format="%.3f",
                    key=f"manual_{sensor}",
                )

        submitted = st.form_submit_button("🔮 Predict RUL", width='stretch')

    if submitted:
        manual_row = {"cycle": manual_cycle, **setting_inputs, **sensor_inputs}
        manual_df = pd.DataFrame([manual_row])

        X_manual = prepare_model_input(manual_df, imputer)
        predicted_rul_manual = predict_rul(model, imputer, X_manual)[0]

        st.divider()
        result_col1, result_col2 = st.columns([1, 2])

        with result_col1:
            st.metric("Predicted RUL", f"{predicted_rul_manual:.1f} cycles")

        with result_col2:
            render_rul_message(predicted_rul_manual)
