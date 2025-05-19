import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import lime.lime_tabular
import matplotlib.pyplot as plt
import streamlit.components.v1 as components
import os

# -------------------------------
# ✅ PAGE CONFIG
# -------------------------------
st.set_page_config(
    page_title="🎬 Netflix Revenue Forecast & ROI App",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------------------
# ✅ DARK THEME FIXED
# -------------------------------
st.markdown("""
<style>
body, .main {
    background-color: #0e0e0e;
    color: #ffffff;
}
h1, h2, h3, h4 {
    color: #e50914;
}
.stButton>button {
    background-color: #e50914;
    color: white;
    border: none;
}
section[data-testid="stSidebar"] {
    background-color: #f9f9f9;
    color: #000;
}
</style>
""", unsafe_allow_html=True)

# -------------------------------
# ✅ LOAD MODEL + DATA
# -------------------------------
@st.cache_resource
def load_model():
    return joblib.load("model_xgb.pkl")

@st.cache_resource
def load_scaler():
    return joblib.load("scaler.pkl")

@st.cache_resource
def load_data():
    return pd.read_csv("X_test.csv")

@st.cache_resource
def load_columns():
    return pd.read_csv("X_train_columns.csv").iloc[:, 0].tolist()

model = load_model()
scaler = load_scaler()
X_test = load_data()
xtrain_columns = load_columns()

# -------------------------------
# ✅ HEADER
# -------------------------------
st.title("🎬 Netflix Revenue Prediction & ROI Intelligence App")
st.markdown("""
Welcome to the **Netflix AI Revenue Optimizer**  
Use this tool to:
- 📈 Predict expected revenue for new or upcoming movies  
- 💰 Evaluate ROI instantly  
- 🔍 Explore explainability with **SHAP** & **LIME**
""")

# -------------------------------
# ✅ SIDEBAR INPUT MODE
# -------------------------------
st.sidebar.header("🎛️ Input Options")
input_mode = st.sidebar.radio("Select input method:", ["Manual Entry", "Use Sample Data"])
user_input_df = None

# -------------------------------
# ✅ MANUAL ENTRY MODE
# -------------------------------
if input_mode == "Manual Entry":
    st.sidebar.subheader("🖊️ Manual Input")
    avg_rating = st.sidebar.slider("Average Rating", 1.0, 10.0, 7.0)
    runtime = st.sidebar.slider("Runtime (minutes)", 60, 200, 110)
    budget = st.sidebar.number_input("🎬 Budget (USD)", min_value=1000000, max_value=500000000, value=30000000)
    release_month = st.sidebar.selectbox("Release Month", list(range(1, 13)))
    release_quarter = st.sidebar.selectbox("Release Quarter", [1, 2, 3, 4])
    release_year = st.sidebar.selectbox("Release Year", list(range(2001, 2021)))

    if st.sidebar.button("🔍 Predict Revenue & ROI"):
        input_dict = {
            'averageRating': avg_rating,
            'budget': budget,
            'run_time (minutes)': runtime,
            'release_month': release_month,
            'release_quarter': release_quarter,
            'release_year': release_year
        }
        user_input_df = pd.DataFrame([input_dict], columns=xtrain_columns).fillna(0)

# -------------------------------
# ✅ SAMPLE DATA MODE
# -------------------------------
elif input_mode == "Use Sample Data":
    st.sidebar.success("✅ Using test data")
    selected_index = st.sidebar.slider("Select test sample", 0, len(X_test)-1, 0)
    user_input_df = X_test.iloc[[selected_index]]

    st.subheader("📥 Sample Input Features")
    cols_to_display = ['run_time (minutes)', 'budget', 'domestic_revenue', 'international_revenue', 'averageRating']
    sample_display_df = user_input_df[cols_to_display]
    st.dataframe(sample_display_df, use_container_width=True)

# -------------------------------
# ✅ PREDICTION + ROI
# -------------------------------
if user_input_df is not None:
    st.subheader("🎯 Predicted Worldwide Revenue & ROI")
    log_pred = model.predict(user_input_df)[0]
    predicted_revenue = np.expm1(log_pred)
    used_budget = user_input_df['budget'].values[0]
    roi = (predicted_revenue - used_budget) / used_budget

    st.markdown(f"""
    - 💵 **Predicted Revenue:** ${predicted_revenue:,.0f}  
    - 📈 **Estimated ROI (Return on Investment):** {roi:.2f}x ({'✅ Profitable' if roi > 0 else '❌ Loss-Making'})  
    """)

    # -------------------------------
    # ✅ INTERPRETATION
    # -------------------------------
    st.subheader("📌 Prediction Interpretation")
    st.markdown(f"""
**Worldwide Revenue** represents the total projected box office earnings across global markets.  
Based on the inputs provided, the model forecasts:

- **Total Revenue**: ${predicted_revenue:,.0f}  
- **Budget Entered**: ${used_budget:,.0f}  
- **Return on Investment (ROI)**: {roi:.2f}x

This means for every $1 spent, Netflix is expected to earn **${roi+1:.2f}** in return.  
This investment is considered **{"profitable ✅" if roi > 0 else "loss-making ❌"}**.

#### 📈 Business Impact:
- A **profitable** movie with 2.5x ROI could generate **${(roi * used_budget):,.0f} in earnings**, boosting quarterly targets.  
- A **loss-making** movie with -0.5x ROI would return only **${predicted_revenue:,.0f}**, losing nearly **{abs(roi)*100:.1f}%** of the investment.

Data-backed decisions like these improve content ROI, budget planning, and viewer satisfaction.
""")

    # -------------------------------
    # ✅ SHAP / LIME EXPLAINABILITY
    # -------------------------------
    if input_mode == "Use Sample Data":
        st.subheader("🧠 Model Explainability (SHAP / LIME)")
        explain_mode = st.radio("Choose Method:", ["SHAP", "LIME"], horizontal=True)

        if explain_mode == "SHAP":
            st.markdown("#### 🔍 SHAP Force Plot")
            html_file = f"shap_force_plot_{selected_index}.html"
            if os.path.exists(html_file):
                with open(html_file, "r", encoding="utf-8") as f:
                    components.html(f.read(), height=400, scrolling=True)
            else:
                st.info("ℹ️ SHAP plot not found for this sample index.")

        elif explain_mode == "LIME":
            st.markdown("#### 🧪 LIME Explanation")
            html_file = "lime_explanation_2.html"
            if os.path.exists(html_file):
                with open(html_file, "r", encoding="utf-8") as f:
                    components.html(f.read(), height=600, scrolling=True)
            else:
                st.error("❌ LIME explanation file not found.")

# -------------------------------
# ✅ FINAL RECOMMENDATIONS
# -------------------------------
if user_input_df is not None:
    st.subheader("📋 Business Recommendations")
    st.markdown("""
- 🌍 Focus on boosting **international appeal** — it’s a major driver of revenue.  
- 🎯 Target **optimal release windows** (quarters/months) to maximize impact.  
- 🧠 Use model-powered predictions to **greenlight content** with high forecasted ROI.  
- 💡 Avoid high-budget projects with low expected returns — let data guide investment.
""")

# -------------------------------
# ✅ FOOTER
# -------------------------------
st.markdown("---")
st.caption("© 2025 • Built by Sweety Seelam | Advanced ML + ROI Forecast + SHAP + LIME | Netflix Dark Mode Edition")