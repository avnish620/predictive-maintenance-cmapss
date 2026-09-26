import streamlit as st

# ============================================================
# PAGE CONFIGURATION (must be called once, here, in the entrypoint)
# ============================================================

st.set_page_config(
    page_title="Predictive Maintenance Dashboard",
    page_icon="🔧",
    layout="wide",
)

# ============================================================
# NAVIGATION
# ============================================================
# NOTE: file paths below are plain ASCII on purpose — emoji in
# filenames can get mangled by zip/OS handling (this is what caused
# the "StreamlitPageNotFoundError" you saw). Icons are set here in
# code instead, which works reliably everywhere.

home_page = st.Page("views/home.py", title="Home", icon="🏠", default=True)
predict_page = st.Page("views/predict.py", title="Predict RUL", icon="🔮")
performance_page = st.Page("views/performance.py", title="Model Performance", icon="📈")
explainability_page = st.Page("views/explainability.py", title="Explainability", icon="🔬")
drift_page = st.Page("views/drift.py", title="Drift Monitoring", icon="🚨")

pg = st.navigation(
    [home_page, predict_page, performance_page, explainability_page, drift_page]
)
pg.run()
