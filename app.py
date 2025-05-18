import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import lime.lime_tabular
import matplotlib.pyplot as plt
import streamlit.components.v1 as components
from sklearn.preprocessing import StandardScaler

# -------------------------------
# ✅ PAGE CONFIG
# -------------------------------
st.set_page_config(
    page_title="🎬 Netflix Revenue Forecasting & ROI App",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------------------
# ✅ CACHED MODEL + DATA LOADING
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

model = load_model()
scaler = load_scaler()
X_test = load_data()

# Load column structure (for manual input alignment)
xtrain_columns = pd.read_csv("X_train_columns.csv", index_col=0).columns.tolist()

# -------------------------------
# ✅ HEADER
# -------------------------------
st.title("🎬 Netflix Revenue Prediction & ROI Intelligence App")
st.markdown("""
Welcome to the **Netflix AI Revenue Optimizer**.  
Use this tool to:
- 📈 Predict expected revenue for a new or existing movie
- 💰 Evaluate expected ROI
- 🔎 Explore model explainability via SHAP & LIME
""")

# -------------------------------
# ✅ SIDEBAR: INPUT MODE
# -------------------------------
st.sidebar.header("🎛️ Input Options")
input_mode = st.sidebar.radio("Select input method:", ["Manual Entry", "Use Sample Data"])

# -------------------------------
# ✅ USER INPUT: MANUAL / SAMPLE
# -------------------------------
if input_mode == "Manual Entry":
    st.sidebar.subheader("📥 Manual Input")
    avg_rating = st.sidebar.slider("Average Rating", 1.0, 10.0, 7.0)
    budget = st.sidebar.number_input("Budget (USD)", min_value=1000000, max_value=500000000, value=30000000)
    runtime = st.sidebar.slider("Runtime (min)", 60, 200, 110)
    release_month = st.sidebar.selectbox("Release Month", list(range(1, 13)))
    release_quarter = st.sidebar.selectbox("Release Quarter", [1, 2, 3, 4])

    input_dict = {
    'averageRating': avg_rating,
    'budget': budget,
    'run_time (minutes)': runtime,
    'release_month': release_month,
    'release_quarter': release_quarter
}
    
    user_input_df = pd.DataFrame([input_dict])
    user_input_df = user_input_df.reindex(columns=xtrain_columns, fill_value=0)
    user_input_scaled = scaler.transform(user_input_df)

else:
    st.sidebar.success("✅ Using sample data")
    selected_index = 2  # Fixed index to match lime_explanation_2.html
    user_input_df = X_test.iloc[[selected_index]]
    user_input_scaled = scaler.transform(user_input_df)

# -------------------------------
# ✅ REVENUE PREDICTION
# -------------------------------
st.subheader("🎯 Predicted Worldwide Revenue")
log_pred = model.predict(user_input_df)[0]
predicted_revenue = np.expm1(log_pred)
st.metric("💵 Revenue Prediction", f"${predicted_revenue:,.0f}")

# -------------------------------
# ✅ ROI ESTIMATION
# -------------------------------
st.subheader("📊 Estimated Return on Investment (ROI)")
used_budget = user_input_df['budget'].values[0]
roi = (predicted_revenue - used_budget) / used_budget
st.metric("📈 ROI", f"{roi * 100:.2f}%")

st.success(f"✅ Predicted Revenue: ${predicted_revenue:,.0f}")
st.markdown(f"💰 Estimated ROI: {roi:.2f}x")

# -------------------------------
# ✅ EXPLAINABILITY SECTION 
# -------------------------------
st.subheader("🧠 Model Explainability (SHAP or LIME)")

explain_mode = st.radio("Choose Explainability Method:", ["SHAP", "LIME"])

if explain_mode == "SHAP":
    if input_mode == "Use Sample Data":
        st.markdown("#### 🔍 SHAP Force Plot")
        html_file = "shap_force_plot_2.html"
        try:
            with open(html_file, "r", encoding="utf-8") as f:
                st.components.v1.html(f.read(), height=400, scrolling=True)
        except FileNotFoundError:
            st.error(f"❌ File not found: {html_file}")
    else:
        st.info("ℹ️ SHAP visualizations are available only for sample data.")

elif explain_mode == "LIME":
    if input_mode == "Use Sample Data":
        st.markdown("#### 🧪 LIME Explanation")
        html_file = "lime_explanation_2.html"
        try:
            with open(html_file, "r", encoding="utf-8") as f:
                st.components.v1.html(f.read(), height=600, scrolling=True)
        except FileNotFoundError:
            st.error(f"❌ File not found: {html_file}")
    else:
        st.info("ℹ️ LIME visualizations are available only for sample data.")

# -------------------------------
# ✅ FOOTER / ATTRIBUTION
# -------------------------------
st.markdown("---")
st.caption("Built by Sweety Seelam • Advanced ML + ROI + SHAP + LIME")

# -------------------------------
# 🔒 (Optional) Google Analytics Placeholder
# -------------------------------
# st.markdown(\"\"\"<script async src="https://www.googletagmanager.com/gtag/js?id=UA-XXXXXXXXX-X"></script>\"\"\", unsafe_allow_html=True)
# st.markdown(\"\"\"
# <script>
#   window.dataLayer = window.dataLayer || [];
#   function gtag(){dataLayer.push(arguments);}
#   gtag('js', new Date());
#   gtag('config', 'UA-XXXXXXXXX-X');
# </script>
# \"\"\", unsafe_allow_html=True)