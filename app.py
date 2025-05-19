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

@st.cache_resource
def load_columns():
    return pd.read_csv("X_train_columns.csv")['0'].tolist()

model = load_model()
scaler = load_scaler()
X_test = load_data()
xtrain_columns = load_columns()

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
user_input_df = None
if input_mode == "Manual Entry":
    st.sidebar.subheader("📥 Manual Input")
    avg_rating = st.sidebar.slider("Average Rating", 1.0, 10.0, 7.0)
    budget = st.sidebar.number_input("Budget (USD)", min_value=1000000, max_value=500000000, value=30000000)
    runtime = st.sidebar.slider("Runtime (min)", 60, 200, 110)
    release_month = st.sidebar.selectbox("Release Month", list(range(1, 13)))
    release_quarter = st.sidebar.selectbox("Release Quarter", [1, 2, 3, 4])
    release_year = st.sidebar.selectbox("Release Year", list(range(2001, 2021)))

    input_dict = {
        'averageRating': avg_rating,
        'budget': budget,
        'run_time': runtime,
        'release_month': release_month,
        'release_quarter': release_quarter,
        'release_year': release_year
    }

    if st.sidebar.button("Submit"):
        empty_df = pd.DataFrame(columns=xtrain_columns)
        user_input_df = pd.concat([empty_df, pd.DataFrame([input_dict])], ignore_index=True).fillna(0)
else:
    st.sidebar.success("✅ Using sample data")
    selected_index = st.sidebar.slider("Choose test sample index", 0, len(X_test)-1, 0)
    user_input_df = X_test.iloc[[selected_index]]
    st.markdown("### 🎬 Sample Input Features")
    st.dataframe(user_input_df)

# -------------------------------
# ✅ REVENUE PREDICTION
# -------------------------------
if user_input_df is not None:
    user_input_scaled = scaler.transform(user_input_df)
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
    # ✅ Prediction Interpretation
    # -------------------------------
    st.subheader("📌 Prediction Interpretation")
    st.markdown(f"- Based on the input features, the projected revenue is **${predicted_revenue:,.0f}**.")
    st.markdown(f"- The ROI of **{roi:.2f}x** indicates a {'profitable' if roi > 0 else 'loss-making'} investment.")
    st.markdown("- This can help Netflix decide which genres, budgets, and release windows are more lucrative.")

    # -------------------------------
    # ✅ EXPLAINABILITY SECTION 
    # -------------------------------
    st.subheader("🧠 Model Explainability (SHAP or LIME)")
    explain_mode = st.radio("Choose Explainability Method:", ["SHAP", "LIME"])

    if explain_mode == "SHAP":
        if input_mode == "Use Sample Data":
            st.markdown("#### 🔍 SHAP Force Plot")
            html_file = f"shap_force_plot_{selected_index}.html"
            try:
                with open(html_file, "r", encoding="utf-8") as f:
                    components.html(f.read(), height=400, scrolling=True)
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
                    components.html(f.read(), height=600, scrolling=True)
            except FileNotFoundError:
                st.error(f"❌ File not found: {html_file}")
        else:
            st.info("ℹ️ LIME visualizations are available only for sample data.")

    st.markdown("---")

    # -------------------------------
    # ✅ FINAL RECOMMENDATIONS
    # -------------------------------
    st.subheader("📋 Business Recommendations")
    st.markdown("""
- 🌍 Focus on movies with **higher international revenue drivers**, as they contribute the most.  
- 🧾 Optimize **budget allocation** to balance investment and ROI potential.  
- 📅 Release strategies in **specific quarters/months** with better historical ROI can boost success.  
- 🧠 Adopt **model-guided greenlighting** of new content to **maximize revenue forecasts**.
""")

# -------------------------------
# ✅ FOOTER / ATTRIBUTION
# -------------------------------
st.markdown("---")
st.caption("Built by Sweety Seelam • Advanced ML + ROI + SHAP + LIME")