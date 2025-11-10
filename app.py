import os
import numpy as np
import pandas as pd
import joblib
import streamlit as st
import streamlit.components.v1 as components

# -------------------------------
# PAGE CONFIG
# -------------------------------
st.set_page_config(
    page_title="🎬 Netflix Revenue Forecast & ROI App",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -------------------------------
# LOAD ARTEFACTS (robustly)
# -------------------------------
TARGET_WAS_LOG1P = True  # <- set True if you trained on log1p(y); set False if plain log(y)

@st.cache_resource(show_spinner=False)
def _load_pipeline_if_any():
    # Prefer a single sklearn Pipeline that encapsulates preprocessing + model
    cand = ["pipeline.pkl", "model_pipeline.pkl", "clf_pipeline.pkl"]
    for c in cand:
        if os.path.exists(c):
            return joblib.load(c)
    return None

@st.cache_resource(show_spinner=False)
def _load_model():
    if os.path.exists("model_xgb.pkl"):
        return joblib.load("model_xgb.pkl")
    return None

@st.cache_resource(show_spinner=False)
def _load_scaler():
    if os.path.exists("scaler.pkl"):
        return joblib.load("scaler.pkl")
    return None

@st.cache_resource(show_spinner=False)
def _load_X_test():
    if os.path.exists("X_test.csv"):
        return pd.read_csv("X_test.csv")
    return pd.DataFrame()

@st.cache_resource(show_spinner=False)
def _load_feature_list():
    # Single-column CSV with the training feature names
    if os.path.exists("X_train_columns.csv"):
        return pd.read_csv("X_train_columns.csv").iloc[:, 0].tolist()
    return []

pipe = _load_pipeline_if_any()
model = _load_model()
scaler = _load_scaler()
X_test_full = _load_X_test()
xtrain_columns = _load_feature_list()

# -------------------------------
# HEADER
# -------------------------------
st.title("🎬 Netflix Revenue Prediction & ROI Intelligence App")
st.markdown(
    """
Welcome to the **Netflix AI Revenue Optimizer**. Use this tool to:
- 📈 Predict expected *worldwide revenue* for new or upcoming movies  
- 💰 Evaluate ROI instantly  
- 🔍 Explore explainability with SHAP & LIME
"""
)

# -------------------------------
# SIDEBAR INPUT MODE
# -------------------------------
st.sidebar.header("🎛️ Input Options")
input_mode = st.sidebar.radio("Select input method:", ["Manual Entry", "Use Sample Data"])

user_input_df = None
selected_index = 0

# -------------------------------
# MANUAL INPUTS (simplified)
# -------------------------------
if input_mode == "Manual Entry":
    st.sidebar.subheader("Manual Input")
    avg_rating = st.sidebar.slider("Average Rating", 1.0, 10.0, 7.0, step=0.1)
    runtime = st.sidebar.slider("Runtime (minutes)", 60, 240, 120, step=5)
    budget = st.sidebar.number_input("Budget (USD)", min_value=100_000, max_value=1_000_000_000, value=30_000_000, step=500_000)
    release_month = st.sidebar.selectbox("Release Month (1–12)", list(range(1, 13)), index=0)
    release_year = st.sidebar.selectbox("Release Year", list(range(2001, 2021)), index=0)

    if st.sidebar.button("Predict Revenue & ROI", use_container_width=True):
        # Build a one-row frame with EXACT training column names (others → 0)
        base = {col: 0 for col in xtrain_columns} if xtrain_columns else {}
        # Common names seen in your training
        rename_map = {
            "averageRating": "averageRating",
            "budget": "budget",
            "run_time (minutes)": "run_time (minutes)",
            "release_month": "release_month",
            "release_year": "release_year",
        }
        row = {
            "averageRating": avg_rating,
            "budget": float(budget),
            "run_time (minutes)": int(runtime),
            "release_month": int(release_month),
            "release_year": int(release_year),
        }
        base.update({rename_map[k]: v for k, v in row.items() if rename_map.get(k) in base or not base})
        user_input_df = pd.DataFrame([base]) if base else pd.DataFrame([row])

# -------------------------------
# SAMPLE DATA MODE
# -------------------------------
elif input_mode == "Use Sample Data":
    if X_test_full.empty:
        st.warning("No X_test.csv found. Switch to Manual Entry.")
    else:
        st.sidebar.success("Using test data")
        if "movie_title" in X_test_full.columns:
            titles = X_test_full["movie_title"].dropna().tolist()
            selected_title = st.sidebar.selectbox("🎬 Select Movie Title", titles)
            selected_index = X_test_full.index[X_test_full["movie_title"] == selected_title][0]
        else:
            selected_index = st.sidebar.slider("Select test sample", 0, len(X_test_full) - 1, 0)

        # Restrict to feature columns if available
        raw_row = X_test_full.iloc[[selected_index]].copy()
        user_input_df = raw_row[xtrain_columns] if xtrain_columns else raw_row

        st.subheader("📥 Sample Input (key fields)")
        cols_to_show = [c for c in ['run_time (minutes)', 'budget', 'averageRating', 'release_month', 'release_year'] if c in user_input_df.columns]
        st.dataframe(user_input_df[cols_to_show], use_container_width=True)

# -------------------------------
# PREDICT & ROI
# -------------------------------
def _inverse_log(y_pred):
    if TARGET_WAS_LOG1P:
        return np.maximum(0.0, np.expm1(y_pred))
    return np.maximum(0.0, np.exp(y_pred))

def _preprocess_for_model(df: pd.DataFrame) -> np.ndarray:
    """Ensure df has right columns and transform if needed."""
    # Align schema
    if xtrain_columns:
        missing = [c for c in xtrain_columns if c not in df.columns]
        if missing:
            # Create missing engineered columns as zeros
            for c in missing:
                df[c] = 0
        df = df[xtrain_columns]

    # Pipeline → no manual scaling
    if pipe is not None:
        return df

    # If no pipeline, but scaler exists → scale
    if scaler is not None:
        return scaler.transform(df)

    # Else, return numeric values as-is
    return df.values

if user_input_df is not None and (pipe is not None or model is not None):
    st.subheader("📊 Predicted Worldwide Revenue & ROI")

    try:
        Z = _preprocess_for_model(user_input_df.copy())

        # Predict
        if pipe is not None:
            raw_pred = pipe.predict(Z)[0]
        else:
            raw_pred = model.predict(Z)[0]

        # Inverse log if needed
        predicted_revenue = float(_inverse_log(raw_pred))

        # ROI math with safety
        used_budget = float(user_input_df.get("budget", pd.Series([np.nan])).values[0])
        eps = 1e-9
        roi = (predicted_revenue - used_budget) / max(used_budget, eps)

        # Cards (no broken markdown)
        c1, c2, c3 = st.columns(3)
        c1.metric(label="💵 Predicted Worldwide Revenue", value=f"${predicted_revenue:,.0f}")
        c2.metric(label="📈 Estimated ROI (x)", value=f"{roi:.2f}x", delta="Profitable" if roi > 0 else "Loss-Making", delta_color="normal")
        c3.metric(label="🏦 Budget (input)", value=f"${used_budget:,.0f}")

        # Sanity banner if way out-of-scope
        if used_budget > 0 and (predicted_revenue < 0.05 * used_budget or predicted_revenue > 10 * used_budget):
            st.info(
                "ℹ️ **Sanity check:** This prediction is far from the provided budget. "
                "If your training data didn’t include similar budgets or dates, consider retraining with more coverage."
            )

        # Interpretation (plain text, no funky wrapping)
        st.subheader("🧠 Prediction Interpretation")
        st.write(
            f"**Predicted Revenue:** ${predicted_revenue:,.0f}  \n"
            f"**Budget Entered:** ${used_budget:,.0f}  \n"
            f"**Estimated ROI:** {roi:.2f}x"
        )
        per_dollar_return = max(0.0, roi + 1.0)
        st.write(f"For every **$1** spent, the expected return is **${per_dollar_return:.2f}**.")
        st.write(f"This investment is **{'Profitable ✅' if roi > 0 else 'Loss-Making ❌'}**.")

        # Business Impact
        st.subheader("📈 Business Impact")
        gain_loss = predicted_revenue - used_budget
        gl_word = "gain" if gain_loss > 0 else "loss"
        st.write(f"Estimated {gl_word}: **${gain_loss:,.0f}**.")
        st.write(f"That is a **{abs(roi)*100:.2f}% {gl_word}** relative to budget.")

        # Explainability (best-effort)
        if input_mode == "Use Sample Data":
            st.subheader("📌 Model Explainability (SHAP / LIME)")
            explain_mode = st.radio("Choose Method:", ["SHAP", "LIME"], horizontal=True)
            if explain_mode == "SHAP":
                html_file = f"shap_force_plot_{selected_index}.html"
                if os.path.exists(html_file):
                    with open(html_file, "r", encoding="utf-8") as f:
                        components.html(f.read(), height=420, scrolling=True)
                else:
                    st.info("SHAP plot not available for this sample.")
            else:
                html_file = "lime_explanation_2.html"
                if os.path.exists(html_file):
                    with open(html_file, "r", encoding="utf-8") as f:
                        components.html(f.read(), height=600, scrolling=True)
                else:
                    st.info("LIME explanation file not available.")

    except Exception as e:
        st.error(f"🚨 Prediction failed: {e}")

elif user_input_df is not None:
    st.error("No model found. Please include a Pipeline (pipeline.pkl) or model_xgb.pkl (+ scaler.pkl).")

# -------------------------------
# FINAL RECOMMENDATIONS
# -------------------------------
if user_input_df is not None:
    st.subheader("🔎 Business Recommendations")
    st.markdown(
        """
- 🌍 Improve **international appeal** (cast, language, distribution windows).
- 🗓️ Choose **strong seasonal months** based on historical uplift.
- 🎯 If ROI < 0, re-scope: reduce budget, adjust runtime, or shift release window.
- 🧪 Use A/B test learnings (trailers, thumbnails, targeting) before greenlighting.
        """
    )

st.markdown("---")
st.caption("© 2025 • Built by Sweety Seelam • Netflix ROI Forecast App with SHAP & LIME")