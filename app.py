import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import lime.lime_tabular
import matplotlib.pyplot as plt
import streamlit.components.v1 as components

# -------------------------------
# ✅ PAGE CONFIG: Dark Netflix Theme
# -------------------------------
st.set_page_config(
    page_title="🎬 Netflix Revenue Forecast & ROI App",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Dark theme styling (custom CSS)
st.markdown("""
    <style>
        body, .reportview-container, .main {
            background-color: #0e0e0e;
            color: #ffffff;
        }
        .st-bb, .st-at, .st-cf, .st-cg, .st-ci {
            background-color: #0e0e0e;
        }
        h1, h2, h3, h4 {
            color: #e50914;
        }
        .stButton>button {
            color: white;
            background-color: #e50914;
        }
        .stSlider > div {
            color: white;
        }
        .stSelectbox > div {
            color: black;
        }
        .metric-label, .metric-value {
            font-weight: bold;
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
# ✅ MANUAL ENTRY PAGE
# -------------------------------
if input_mode == "Manual Entry":
    st.sidebar.subheader("🖊️ Manual Input")
    col1, col2 = st.sidebar.columns(2)
    avg_rating = col1.slider("Avg. Rating", 1.0, 10.0, 7.0)
    runtime = col2.slider("Runtime (min)", 60, 200, 110)
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
# ✅ SAMPLE DATA PAGE
# -------------------------------
elif input_mode == "Use Sample Data":
    st.sidebar.success("✅ Using test data")
    selected_index = st.sidebar.slider("Select test sample", 0, len(X_test)-1, 0)
    user_input_df = X_test.iloc[[selected_index]]

    st.subheader("📥 Sample Input Features")
    st.dataframe(user_input_df, use_container_width=True)

# -------------------------------
# ✅ PREDICTION + ROI
# -------------------------------
if user_input_df is not None:
    st.subheader("🎯 Predicted Worldwide Revenue & ROI")
    user_input_scaled = scaler.transform(user_input_df)
    log_pred = model.predict(user_input_df)[0]
    predicted_revenue = np.expm1(log_pred)

    col1, col2 = st.columns(2)
    col1.metric("💵 Revenue Prediction", f"${predicted_revenue:,.0f}")
    used_budget = user_input_df['budget'].values[0]
    roi = (predicted_revenue - used_budget) / used_budget
    col2.metric("📈 ROI", f"{roi * 100:.2f}%")

    st.markdown(f"""
        - 🎯 **Predicted Revenue:** ${predicted_revenue:,.0f}  
        - 💰 **Estimated ROI:** {roi:.2f}x ({'Profitable ✅' if roi > 0 else 'Loss ❌'})
    """)

    # -------------------------------
    # ✅ INTERPRETATION
    # -------------------------------
    st.subheader("📌 Prediction Interpretation")
    st.markdown(f"""
    - Based on the input features, the projected revenue is **${predicted_revenue:,.0f}**.
    - The ROI of **{roi:.2f}x** indicates a {'profitable' if roi > 0 else 'loss-making'} investment.
    - Use these insights to guide budget allocation and release strategy.
    """)

    # -------------------------------
    # ✅ EXPLAINABILITY SECTION
    # -------------------------------
    if input_mode == "Use Sample Data":
        st.subheader("🧠 Model Explainability (SHAP / LIME)")
        explain_mode = st.radio("Choose Method:", ["SHAP", "LIME"], horizontal=True)

        if explain_mode == "SHAP":
            st.markdown("#### 🔍 SHAP Force Plot")
            html_file = f"shap_force_plot_{selected_index}.html"
            try:
                with open(html_file, "r", encoding="utf-8") as f:
                    components.html(f.read(), height=400, scrolling=True)
            except FileNotFoundError:
                st.error(f"❌ File not found: {html_file}")

        elif explain_mode == "LIME":
            st.markdown("#### 🧪 LIME Explanation")
            html_file = "lime_explanation_2.html"
            try:
                with open(html_file, "r", encoding="utf-8") as f:
                    components.html(f.read(), height=600, scrolling=True)
            except FileNotFoundError:
                st.error(f"❌ File not found: {html_file}")

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
st.caption("© 2025 • Built by Sweety Seelam | Advanced ML + ROI Forecast + Explainability | Netflix Dark Mode Edition")