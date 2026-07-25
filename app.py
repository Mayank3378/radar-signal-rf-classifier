import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

st.set_page_config(page_title="Radar Signal Classifier — Random Forest", page_icon="📡", layout="wide")

# ---------- Load model & reference data ----------
@st.cache_resource
def load_model():
    return joblib.load("rf_model.pkl")

@st.cache_data
def load_reference():
    return pd.read_csv("feature_reference.csv")

model = load_model()
ref = load_reference()
feature_cols = list(ref.columns)

# ---------- Header ----------
st.title("📡 Radar Signal Classifier")
st.caption("Random Forest model classifying ionospheric radar returns as **good** (structure detected) or **bad** (signal passed through) — Johns Hopkins APL / UCI Ionosphere dataset, 34 signal features per return.")

tab1, tab2, tab3 = st.tabs(["🔮 Live Prediction", "📊 Model Performance", "ℹ️ About"])

# ---------- TAB 1: Live prediction ----------
with tab1:
    st.subheader("Try a prediction")
    st.write("Load a random real sample, tweak a few key signal features, and get a live Random Forest prediction.")

    col_a, col_b = st.columns([1, 2])
    with col_a:
        if st.button("🎲 Load random sample"):
            st.session_state["sample"] = ref.sample(1).iloc[0]
        if "sample" not in st.session_state:
            st.session_state["sample"] = ref.sample(1, random_state=1).iloc[0]

    sample = st.session_state["sample"]

    # Show top 6 most important features as sliders (rest stay at sampled values)
    importances = pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=False)
    top_features = importances.head(6).index.tolist()

    st.markdown("**Adjust top signal features:**")
    cols = st.columns(3)
    input_vals = sample.copy()
    for i, feat in enumerate(top_features):
        with cols[i % 3]:
            input_vals[feat] = st.slider(
                feat, float(ref[feat].min()), float(ref[feat].max()), float(sample[feat]), key=feat
            )

    X_input = pd.DataFrame([input_vals])[feature_cols]
    pred = model.predict(X_input)[0]
    proba = model.predict_proba(X_input)[0]

    st.divider()
    result_col, proba_col = st.columns(2)
    with result_col:
        if pred == 1:
            st.success("### Prediction: **GOOD** return ✅\nStructure detected in ionosphere")
        else:
            st.error("### Prediction: **BAD** return ❌\nSignal passed through, no structure detected")
    with proba_col:
        st.metric("Confidence (good)", f"{proba[1]*100:.1f}%")
        st.progress(float(proba[1]))

# ---------- TAB 2: Model performance ----------
with tab2:
    st.subheader("Model Performance on Held-out Test Set")
    c1, c2, c3 = st.columns(3)
    c1.image("confusion_matrix.png", use_container_width=True)
    c2.image("roc_curve.png", use_container_width=True)
    c3.image("feature_importance.png", use_container_width=True)
    st.info("Model: RandomForestClassifier(n_estimators=200) — trained on 75% of 351 samples, evaluated on the remaining 25%. Test accuracy ≈ 93%.")

# ---------- TAB 3: About ----------
with tab3:
    st.subheader("About this project")
    st.markdown("""
**Goal:** Apply Random Forest to a real classification problem as a practical implementation task
during the DRDO AI Engineer internship (June 17 – July 31, 2026).

**Dataset:** Johns Hopkins University Ionosphere Database — radar returns collected by a phased-array
system in Goose Bay, Labrador, targeting free electrons in the ionosphere. Each sample has 34 continuous
signal features (autocorrelation function values across 17 pulse numbers) and a label: **good** (evidence
of structure) or **bad** (signal passed straight through).

**Why Random Forest:**
- Handles high-dimensional continuous features well without heavy preprocessing
- Naturally ranks feature importance — useful to see which pulse signals matter most
- Robust to overfitting compared to a single decision tree, via bagging + feature randomness

**Pipeline:**
1. Load & split data (75/25 train/test, stratified)
2. Train `RandomForestClassifier` (200 trees)
3. Evaluate — accuracy, confusion matrix, ROC-AUC, feature importance
4. Wrap in this live Streamlit app for interactive predictions

**Result:** ~93% test accuracy, ROC-AUC well above baseline.
""")
