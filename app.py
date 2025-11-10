import os
import numpy as np
import pandas as pd
import joblib
import streamlit as st
import streamlit.components.v1 as components

# ───────────────────────────
# Page config
# ───────────────────────────
st.set_page_config(
    page_title="🎬 Netflix Revenue Forecast & ROI App",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ───────────────────────────
# Robust loaders
# ───────────────────────────
@st.cache_resource(show_spinner=False)
def load_pipeline():
    for name in ["pipeline.pkl", "model_pipeline.pkl", "clf_pipeline.pkl"]:
        if os.path.exists(name):
            return joblib.load(name)
    return None

@st.cache_resource(show_spinner=False)
def load_model():
    if os.path.exists("model_xgb.pkl"):
        return joblib.load("model_xgb.pkl")
    return None

@st.cache_resource(show_spinner=False)
def load_scaler():
    if os.path.exists("scaler.pkl"):
        return joblib.load("scaler.pkl")
    return None

@st.cache_resource(show_spinner=False)
def load_X_test():
    if os.path.exists("X_test.csv"):
        return pd.read_csv("X_test.csv")
    return pd.DataFrame()

@st.cache_resource(show_spinner=False)
def load_feature_list():
    if os.path.exists("X_train_columns.csv"):
        cols = pd.read_csv("X_train_columns.csv").iloc[:, 0].tolist()
        return [str(c) for c in cols]
    return []

pipe = load_pipeline()
model = load_model()
scaler = load_scaler()
X_test_full = load_X_test()
xtrain_columns = load_feature_list()

# ───────────────────────────
# Helpers
# ───────────────────────────
def coerce_numeric(df: pd.DataFrame) -> pd.DataFrame:
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)
    return df

def align_to_training(df: pd.DataFrame) -> pd.DataFrame:
    """Return df with EXACT training columns; unseen -> dropped, missing -> 0."""
    if not xtrain_columns:
        return coerce_numeric(df.copy())
    aligned = pd.DataFrame({c: 0 for c in xtrain_columns}, index=df.index)
    for c in xtrain_columns:
        if c in df.columns:
            aligned[c] = df[c]
    return coerce_numeric(aligned)

@st.cache_resource(show_spinner=False)
def detect_target_is_loglike() -> bool:
    """
    Peek at a few predictions (without inverse) to decide if the target is log-scale.
    Heuristic: median raw prediction in [6, 30] => looks like ln(revenue).
    """
    try:
        if (pipe is None and model is None) or X_test_full.empty:
            return True  # default to loglike if unsure (safer)
        # Use 20 rows max
        Z = X_test_full.head(20).copy()
        Z = align_to_training(Z)
        if pipe is not None:
            raw = pipe.predict(Z)
        else:
            Zt = scaler.transform(Z) if scaler is not None else Z.values
            raw = model.predict(Zt)
        med = float(np.median(raw))
        return 6.0 <= med <= 30.0
    except Exception:
        return True

TARGET_IS_LOGLIKE = detect_target_is_loglike()

def invert_target(y):
    """Invert based on detected target scaling."""
    y = float(y)
    if TARGET_IS_LOGLIKE:
        # try expm1 first (harmless for big values), fall back to exp if negative
        val = np.expm1(y)
        return float(np.maximum(val, np.exp(y)))
    return y

def preprocess_for_model(df: pd.DataFrame):
    """Return array/DataFrame suitable for predict(), avoiding double scaling."""
    Z = align_to_training(df)
    if pipe is not None:
        return Z                      # pipeline handles preprocessing
    if scaler is not None:
        return scaler.transform(Z)    # external scaler only if no pipeline
    return Z.values                   # raw

def out_of_domain_flag(pred_rev: float, budget: float) -> bool:
    """Flag predictions that are very inconsistent with budget."""
    if budget <= 0:
        return False
    return (pred_rev < 0.05 * budget) or (pred_rev > 10.0 * budget)

# ───────────────────────────
# Header
# ───────────────────────────
st.title("🎬 Netflix Revenue Prediction & ROI Intelligence App")
st.markdown(
    """
Welcome to the **Netflix AI Revenue Optimizer**. Use this tool to:
- 📈 Predict expected *worldwide revenue* for new or upcoming movies  
- 💰 Evaluate ROI instantly  
- 🔍 Explore explainability with SHAP & LIME
"""
)

# ───────────────────────────
# Sidebar input mode
# ───────────────────────────
st.sidebar.header("🎛️ Input Options")
input_mode = st.sidebar.radio("Select input method:", ["Manual Entry", "Use Sample Data"])

user_input_df = None
selected_index = 0

# ───────────────────────────
# Manual inputs (minimal & necessary)
# ───────────────────────────
if input_mode == "Manual Entry":
    st.sidebar.subheader("Manual Input")
    avg_rating = st.sidebar.slider("Average Rating", 1.0, 10.0, 7.0, step=0.1)
    runtime = st.sidebar.slider("Runtime (minutes)", 60, 240, 120, step=5)
    budget = st.sidebar.number_input(
        "Budget (USD)", min_value=100_000, max_value=1_000_000_000, value=30_000_000, step=500_000
    )
    release_month = st.sidebar.selectbox("Release Month (1–12)", list(range(1, 13)), index=0)
    release_year = st.sidebar.selectbox("Release Year", list(range(2001, 2021)), index=0)

    if st.sidebar.button("Predict Revenue & ROI", use_container_width=True):
        # Build a row; align_to_training() will add any missing engineered cols as 0
        raw_row = pd.DataFrame([{
            "averageRating": avg_rating,
            "budget": float(budget),
            "run_time (minutes)": int(runtime),
            "release_month": int(release_month),
            "release_year": int(release_year),
        }])
        user_input_df = raw_row

# ───────────────────────────
# Sample data mode
# ───────────────────────────
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

        # Restrict to feature columns; align later
        raw_row = X_test_full.iloc[[selected_index]].copy()
        user_input_df = raw_row

        # Show key fields if present
        st.subheader("📥 Sample Input (key fields)")
        key_cols = [c for c in ['run_time (minutes)', 'budget', 'averageRating', 'release_month', 'release_year'] if c in raw_row.columns]
        if key_cols:
            st.dataframe(raw_row[key_cols], use_container_width=True)

# ───────────────────────────
# Predict & ROI
# ───────────────────────────
if user_input_df is not None and (pipe is not None or model is not None):
    st.subheader("📊 Predicted Worldwide Revenue & ROI")

    try:
        Z = preprocess_for_model(user_input_df.copy())
        raw_pred = pipe.predict(Z)[0] if pipe is not None else model.predict(Z)[0]
        predicted_revenue = max(0.0, invert_target(raw_pred))

        # Budget & ROI
        used_budget = float(user_input_df.get("budget", pd.Series([np.nan])).values[0])
        eps = 1e-9
        roi = (predicted_revenue - used_budget) / max(used_budget, eps) if used_budget > 0 else np.nan

        # Metrics row
        c1, c2, c3 = st.columns(3)
        c1.metric("💵 Predicted Worldwide Revenue", f"${predicted_revenue:,.0f}")
        if np.isfinite(roi):
            c2.metric(
                "📈 Estimated ROI (x)",
                f"{roi:.2f}x",
                delta=("Profitable" if roi > 0 else "Loss-Making"),
                delta_color="normal",
            )
        else:
            c2.metric("📈 Estimated ROI (x)", "—", delta="Budget not provided", delta_color="off")
        c3.metric("🏦 Budget (input)", f"${used_budget:,.0f}" if used_budget > 0 else "—")

        # Sanity banner + ROI reliability
        flagged = out_of_domain_flag(predicted_revenue, used_budget)
        if flagged:
            st.info(
                "ℹ️ **Sanity check:** This prediction appears far from the provided budget. "
                "If your training data didn’t include similar budgets or dates, consider retraining with more coverage. "
                "ROI is marked **Unreliable** below."
            )

        # Interpretation
        st.subheader("🧠 Prediction Interpretation")
        st.write(
            f"**Predicted Revenue:** ${predicted_revenue:,.0f}  \n"
            f"**Budget Entered:** ${used_budget:,.0f}  \n"
            f"**Estimated ROI:** {(f'{roi:.2f}x' if np.isfinite(roi) else '—')}"
        )
        if np.isfinite(roi):
            per_dollar = max(0.0, roi + 1.0)
            tag = "Profitable ✅" if roi > 0 else "Loss-Making ❌"
            reliab = " — **Unreliable (out of range)**" if flagged else ""
            st.write(f"For every **$1** spent, the expected return is **${per_dollar:.2f}**.  \nThis investment is **{tag}{reliab}**.")
        else:
            st.write("ROI cannot be computed without a positive budget.")

        # Business Impact
        st.subheader("📈 Business Impact")
        if np.isfinite(roi):
            gain_loss = predicted_revenue - used_budget
            gl_word = "gain" if gain_loss > 0 else "loss"
            pct = abs(roi) * 100.0
            rel = " (Unreliable)" if flagged else ""
            st.write(f"Estimated {gl_word}: **${gain_loss:,.0f}**.")
            st.write(f"That is a **{pct:.2f}% {gl_word}** relative to budget{rel}.")
        else:
            st.write("Provide a valid budget to compute a gain/loss estimate.")

        # Explainability (best-effort with pre-rendered HTMLs)
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

# ───────────────────────────
# Final recommendations
# ───────────────────────────
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