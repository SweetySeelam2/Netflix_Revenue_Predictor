import os
import numpy as np
import pandas as pd
import joblib
import streamlit as st

st.set_page_config(
    page_title="🎬 Netflix Revenue Forecast & ROI App",
    layout="wide",
    initial_sidebar_state="expanded",
)

@st.cache_resource(show_spinner=False)
def load_pipeline_bundle():
    if not os.path.exists("pipeline.pkl"):
        raise FileNotFoundError(
            "Missing pipeline.pkl. Train it with: python train_clean_ohe.py --csv training_engineered.csv"
        )
    bundle = joblib.load("pipeline.pkl")
    pipe = bundle["pipe"]
    target_is_log1p = bool(bundle.get("target_is_log1p", True))
    feature_cols = bundle.get("feature_columns", [])
    numeric_cols = bundle.get("numeric_columns", [])
    onehot_groups = bundle.get("onehot_groups", {})
    return pipe, target_is_log1p, feature_cols, numeric_cols, onehot_groups

try:
    pipe, TARGET_IS_LOGLIKE, FEATURE_COLS, NUMERIC_COLS, ONEHOT_GROUPS = load_pipeline_bundle()
except Exception as e:
    st.error(f"🚨 Could not load model pipeline: {e}")
    st.stop()

def invert_target(y: float) -> float:
    return float(np.expm1(y)) if TARGET_IS_LOGLIKE else float(y)

def out_of_domain(pred_rev: float, budget: float) -> bool:
    if budget <= 0:
        return False
    return (pred_rev < 0.05 * budget) or (pred_rev > 10.0 * budget)

def choices_for(prefix: str) -> list[str]:
    cols = ONEHOT_GROUPS.get(prefix, [])
    if not cols:
        return ["Unknown"]
    labels = [c.replace(prefix, "") for c in cols]
    return ["Unknown"] + labels

def one_hot_from_choice(prefix: str, choice: str, group_cols: list[str]) -> dict:
    row = {c: 0 for c in group_cols}
    if choice and choice != "Unknown":
        col_name = f"{prefix}{choice}"
        if col_name in row:
            row[col_name] = 1
    return row

st.title("🎬 Netflix Revenue Prediction & ROI Intelligence App")
st.markdown("""
Use this tool to:
- 📈 Predict expected *worldwide revenue* for new or upcoming movies  
- 💰 Evaluate ROI instantly  
""")

st.sidebar.header("🎛️ Input Options")
st.sidebar.subheader("Manual Input")

avg_rating = st.sidebar.slider("Average Rating", 1.0, 10.0, 7.0, step=0.1)
runtime = st.sidebar.slider("Runtime (minutes)", 60, 240, 120, step=5)
budget = st.sidebar.number_input("Budget (USD)", min_value=100_000, max_value=1_000_000_000, value=30_000_000, step=500_000)
release_month = st.sidebar.selectbox("Release Month (1–12)", list(range(1, 13)), index=0)
release_year = st.sidebar.selectbox("Release Year", list(range(2001, 2021)), index=0)

genre_choice   = st.sidebar.selectbox("Main Genre",          choices_for("main_genre_"))
mpaa_choice    = st.sidebar.selectbox("MPAA Rating",         choices_for("mpaa_"))
lang_choice    = st.sidebar.selectbox("Original Language",   choices_for("original_language_"))
country_choice = st.sidebar.selectbox("Country",             choices_for("country_"))
dist_choice    = st.sidebar.selectbox("Distributor",         choices_for("distributor_"))

predict_clicked = st.sidebar.button("Predict Revenue & ROI", use_container_width=True)

if predict_clicked:
    row = {}
    if "averageRating" in FEATURE_COLS:         row["averageRating"] = float(avg_rating)
    if "budget" in FEATURE_COLS:                row["budget"] = float(budget)
    if "run_time (minutes)" in FEATURE_COLS:    row["run_time (minutes)"] = int(runtime)
    if "release_month" in FEATURE_COLS:         row["release_month"] = int(release_month)
    if "release_year" in FEATURE_COLS:          row["release_year"] = int(release_year)
    if "release_quarter" in FEATURE_COLS:
        rq = (int(release_month) - 1) // 3 + 1
        row["release_quarter"] = rq

    row.update(one_hot_from_choice("main_genre_",        genre_choice,   ONEHOT_GROUPS.get("main_genre_", [])))
    row.update(one_hot_from_choice("mpaa_",              mpaa_choice,    ONEHOT_GROUPS.get("mpaa_", [])))
    row.update(one_hot_from_choice("original_language_", lang_choice,    ONEHOT_GROUPS.get("original_language_", [])))
    row.update(one_hot_from_choice("country_",           country_choice, ONEHOT_GROUPS.get("country_", [])))
    row.update(one_hot_from_choice("distributor_",       dist_choice,    ONEHOT_GROUPS.get("distributor_", [])))

    aligned = {c: 0 for c in FEATURE_COLS}
    for k, v in row.items():
        if k in aligned:
            aligned[k] = v
    X = pd.DataFrame([aligned])

    try:
        raw_pred = pipe.predict(X)[0]
        predicted_revenue = max(0.0, invert_target(raw_pred))
    except Exception as e:
        st.error(f"🚨 Prediction failed: {e}")
        st.stop()

    eps = 1e-9
    roi = (predicted_revenue - budget) / max(budget, eps) if budget > 0 else np.nan

    c1, c2, c3 = st.columns(3)
    c1.metric("💵 Predicted Worldwide Revenue", f"${predicted_revenue:,.0f}")
    if np.isfinite(roi):
        c2.metric("📈 Estimated ROI (x)", f"{roi:.2f}x",
                  delta=("Profitable" if roi > 0 else "Loss-Making"), delta_color="normal")
    else:
        c2.metric("📈 Estimated ROI (x)", "—", delta="Budget not provided", delta_color="off")
    c3.metric("🏦 Budget (input)", f"${budget:,.0f}" if budget > 0 else "—")

    flagged = (budget > 0) and ((predicted_revenue < 0.05 * budget) or (predicted_revenue > 10.0 * budget))
    if flagged:
        st.info(
            "ℹ️ **Sanity check:** Prediction appears far from the provided budget. "
            "Consider retraining with more coverage for these budgets/dates."
        )

    st.subheader("🧠 Prediction Interpretation")
    st.write(
        f"**Predicted Revenue:** ${predicted_revenue:,.0f}  \n\n"
        f"**Budget Entered:** ${budget:,.0f}  \n\n"
        f"**Estimated ROI:** {(f'{roi:.2f}x' if np.isfinite(roi) else '—')}"
    )
    if np.isfinite(roi):
        per_dollar = max(0.0, roi + 1.0)
        tag = "Profitable ✅" if roi > 0 else "Loss-Making ❌"
        reliab = " — **Unreliable (out of range)**" if flagged else ""
        st.write(
            f"For every **1 dollar** spent, expected return is **{per_dollar:.2f} dollar**.  \n\n"
            f"This investment is **{tag}{reliab}**."
        )
    else:
        st.write("ROI cannot be computed without a positive budget.")

    st.subheader("📈 Business Impact")
    if np.isfinite(roi):
        gain_loss = predicted_revenue - budget
        gl_word = "gain" if gain_loss > 0 else "loss"
        pct = abs(roi) * 100.0
        rel = " (Unreliable)" if flagged else ""
        st.write(f"Estimated {gl_word}: **${gain_loss:,.0f}**.")
        st.write(f"That is a **{pct:.2f}% {gl_word}** relative to budget{rel}.")
    else:
        st.write("Provide a valid budget to compute a gain/loss estimate.")

st.markdown("---")
st.caption("© 2025 • Clean one-hot pipeline (no leakage) with schema alignment")