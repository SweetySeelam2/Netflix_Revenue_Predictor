import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import streamlit.components.v1 as components

# -------------------------------
# ✅ PAGE CONFIG
# -------------------------------
st.set_page_config(
    page_title="🎬 Netflix Revenue Forecast & ROI App",
    layout="wide",
    initial_sidebar_state="expanded"
)

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
X_test_full = X_test.copy()  # include movie titles

# -------------------------------
# ✅ HEADER
# -------------------------------
st.title("🎬 Netflix Revenue Prediction & ROI Intelligence App")
st.markdown("""
Welcome to the **Netflix AI Revenue Optimizer**.  
Use this tool to:
- 📈 Predict expected revenue for new or upcoming movies  
- 💰 Evaluate ROI instantly  
- 🔍 Explore explainability with SHAP & LIME
""")

# -------------------------------
# ✅ SIDEBAR INPUT MODE
# -------------------------------
st.sidebar.header("🎛️ Input Options")
input_mode = st.sidebar.radio("Select input method:", ["Manual Entry", "Use Sample Data"])
user_input_df = None
selected_index = 0

# -------------------------------
# ✅ MANUAL ENTRY MODE
# -------------------------------
if input_mode == "Manual Entry":
    st.sidebar.subheader("Manual Input")
    avg_rating = st.sidebar.slider("Average Rating", 1.0, 10.0, 7.0)
    runtime = st.sidebar.slider("Runtime (minutes)", 60, 200, 110)
    budget = st.sidebar.number_input("Budget (USD)", min_value=1000000, max_value=500000000, value=30000000)
    release_month = st.sidebar.selectbox("Release Month", list(range(1, 13)))
    release_quarter = st.sidebar.selectbox("Release Quarter", [1, 2, 3, 4])
    release_year = st.sidebar.selectbox("Release Year", list(range(2001, 2021)))

    if st.sidebar.button("Predict Revenue & ROI"):
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
    st.sidebar.success("Using test data")
    if 'movie_title' in X_test_full.columns:
        movie_titles = X_test_full['movie_title'].dropna().tolist()
        selected_title = st.sidebar.selectbox("🎬 Select Movie Title", movie_titles)
        selected_index = X_test_full[X_test_full['movie_title'] == selected_title].index[0]
    else:
        selected_index = st.sidebar.slider("Select test sample", 0, len(X_test) - 1, 0)

    user_input_df = X_test.iloc[[selected_index]]
    st.subheader("📥 Sample Input Features")
    cols_to_display = ['run_time (minutes)', 'budget', 'domestic_revenue', 'international_revenue', 'averageRating']
    st.dataframe(user_input_df[cols_to_display], use_container_width=True)

# -------------------------------
# ✅ PREDICTION + ROI
# -------------------------------
if user_input_df is not None:
    st.subheader("📊 Predicted Worldwide Revenue & ROI")
    scaled_input = scaler.transform(user_input_df)
    log_pred = model.predict(scaled_input)[0]
    predicted_revenue = np.expm1(log_pred)  # ✅ FIXED: no * 1_000_000
    used_budget = user_input_df['budget'].values[0]
    roi = (predicted_revenue - used_budget) / used_budget

    st.markdown(f"""
    - 💵 **Predicted Revenue:** ${predicted_revenue:,.0f}  
    - 📈 **Estimated ROI (Return on Investment):** {roi:.2f}x ({'✅ Profitable' if roi > 0 else '❌ Loss-Making'})  
    """)

    st.subheader("🧠 Prediction Interpretation")
    st.markdown(f"""
    **Worldwide Revenue** is the total forecasted income across global markets. 

    - **Predicted Revenue**: ${predicted_revenue:,.0f}  
    - **Budget Entered**: ${used_budget:,.0f}  
    - **Estimated ROI**: {roi:.2f}x  

    For every $1 spent, Netflix expects to return ${((roi+1)*1):.2f}.  
    This investment is considered **{'Profitable ✅' if roi > 0 else 'Loss-Making ❌'}**.

    ### 📈 Business Impact:
    - Estimated gain/loss: ${roi * used_budget:,.0f}  
    - ROI translates to a {abs(roi)*100:.2f}% {'gain' if roi > 0 else 'loss'} on investment.
    """)

    # -------------------------------
    # ✅ SHAP / LIME (Sample Data Only)
    # -------------------------------
    if input_mode == "Use Sample Data":
        st.subheader("📌 Model Explainability (SHAP / LIME)")
        explain_mode = st.radio("Choose Method:", ["SHAP", "LIME"], horizontal=True)

        if explain_mode == "SHAP":
            st.markdown("#### SHAP Force Plot")
            html_file = f"shap_force_plot_{selected_index}.html"
            if os.path.exists(html_file):
                with open(html_file, "r", encoding="utf-8") as f:
                    components.html(f.read(), height=400, scrolling=True)
                st.markdown("""
                🔍 **Interpretation:** SHAP shows how each input feature contributes to the final prediction.
                - Red: Positive influence
                - Blue: Negative influence
                """)
            else:
                st.info("ℹ️ SHAP plot not available for this sample.")

        elif explain_mode == "LIME":
            st.markdown("#### LIME Explanation")
            html_file = "lime_explanation_2.html"
            if os.path.exists(html_file):
                with open(html_file, "r", encoding="utf-8") as f:
                    components.html(f.read(), height=600, scrolling=True)
                st.markdown("""
                🧪 **Interpretation:** LIME identifies top influencing features.
                - Green: Increases prediction
                - Red: Decreases prediction
                """)
            else:
                st.error("❌ LIME explanation file not found.")

# -------------------------------
# ✅ FINAL RECOMMENDATIONS
# -------------------------------
if user_input_df is not None:
    st.subheader("🔎 Business Recommendations")
    st.markdown("""
- 🌍 Focus on **international appeal** to boost global revenue.
- 🗓️ Optimize **release month/quarter** based on ROI trends.
- 🧠 Use prediction insights to greenlight high-return content.
- 💡 Avoid high-budget projects with low predicted ROI.
""")

st.markdown("---")
st.caption("© 2025 • Built by Sweety Seelam • Netflix ROI Forecast App with SHAP & LIME")
