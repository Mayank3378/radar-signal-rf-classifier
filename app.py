import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(page_title="Radar Signal Classifier — Random Forest", page_icon="📡", layout="wide")

# ---------- Load model & reference data ----------
@st.cache_resource
def load_model():
    return joblib.load("rf_model.pkl")

@st.cache_data
def load_reference():
    return pd.read_csv("feature_reference.csv")

@st.cache_data
def load_metrics():
    try:
        with open("metrics.txt") as f:
            return f.read()
    except FileNotFoundError:
        return None

model = load_model()
ref = load_reference()
metrics_text = load_metrics()
feature_cols = list(ref.columns)

# ---------- Sidebar ----------
with st.sidebar:
    st.header("📡 About")
    st.markdown(
        "Random Forest classifier for **ionospheric radar returns** — "
        "trained on the Johns Hopkins APL / UCI Ionosphere dataset (351 real radar samples, "
        "34 signal features across 17 pulses)."
    )
    st.divider()
    st.subheader("Model summary")
    st.markdown(
        "- **Algorithm:** Random Forest (200 trees)\n"
        "- **Test accuracy:** ~93%\n"
        "- **5-fold CV accuracy:** ~93%\n"
        "- **vs single Decision Tree:** 93.2% (RF) vs 87.5% (single tree) — Random Forest generalizes better\n"
    )
    st.divider()
    st.caption("Built as a practical Random Forest implementation during the DRDO AI Engineer internship.")

# ---------- Header ----------
st.title("📡 Radar Signal Classifier")
st.caption(
    "Classifies ionospheric radar returns as **good** (structure detected) or **bad** "
    "(signal passed through), using a Random Forest trained on real radar signal data."
)

tab1, tab2, tab3 = st.tabs(["🔮 Live Prediction", "📊 Model Performance", "ℹ️ About the Project"])

# ---------- TAB 1: Live prediction ----------
with tab1:
    st.subheader("Try a live prediction")
    st.write(
        "Load a real radar sample from the dataset, adjust its strongest signal features, "
        "and see the Random Forest's live prediction."
    )

    if st.button("🎲 Load random real sample", type="primary"):
        st.session_state["sample"] = ref.sample(1).iloc[0]
    if "sample" not in st.session_state:
        st.session_state["sample"] = ref.sample(1, random_state=1).iloc[0]

    sample = st.session_state["sample"]

    importances = pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=False)
    top_features = importances.head(6).index.tolist()

    st.markdown("**Adjust the top 6 most influential signal features:**")
    cols = st.columns(3)
    input_vals = sample.copy()
    for i, feat in enumerate(top_features):
        with cols[i % 3]:
            label = feat.replace("_", " ").title()
            input_vals[feat] = st.slider(
                label, float(ref[feat].min()), float(ref[feat].max()), float(sample[feat]), key=feat
            )

    X_input = pd.DataFrame([input_vals])[feature_cols]
    pred = model.predict(X_input)[0]
    proba = model.predict_proba(X_input)[0]

    st.divider()
    result_col, proba_col = st.columns(2)
    with result_col:
        if pred == 1:
            st.success("### Prediction: GOOD return ✅\nStructure detected in the ionosphere")
        else:
            st.error("### Prediction: BAD return ❌\nSignal passed through — no structure detected")
    with proba_col:
        st.metric("Model confidence (good)", f"{proba[1]*100:.1f}%")
        st.progress(float(proba[1]))

    with st.expander("See full feature vector sent to the model"):
        st.dataframe(X_input.T.rename(columns={X_input.index[0]: "value"}), use_container_width=True)

# ---------- TAB 2: Model performance ----------
with tab2:
    st.subheader("Model Performance on Held-out Test Set")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Test Accuracy", "93.2%")
    m2.metric("5-Fold CV Accuracy", "93.2%")
    m3.metric("ROC-AUC", "0.97+")
    m4.metric("vs Single Tree", "+5.7 pts")

    st.divider()
    c1, c2, c3 = st.columns(3)
    c1.image("confusion_matrix.png", use_container_width=True, caption="Confusion Matrix")
    c2.image("roc_curve.png", use_container_width=True, caption="ROC Curve")
    c3.image("feature_importance.png", use_container_width=True, caption="Top Feature Importances")

    if metrics_text:
        with st.expander("Raw metrics output"):
            st.code(metrics_text)

# ---------- TAB 3: About ----------
with tab3:
    st.subheader("About this project")
    st.markdown("""
**Goal:** Apply Random Forest to a real classification problem as a practical implementation
task during the DRDO AI Engineer internship (June 17 – July 31, 2026).

**Dataset:** Johns Hopkins University Ionosphere Database — 351 radar returns collected by a
phased-array system (16 antennas, ~6.4 kW) in Goose Bay, Labrador, targeting free electrons in
the ionosphere.

- Each return was probed with **17 radar pulses**
- Each pulse produced a **complex-valued autocorrelation signal** → 2 features per pulse
  (real + imaginary component) → **34 features total**
- Label: **good** (structure detected in the ionosphere) or **bad** (signal passed straight through)

**Why Random Forest:**
- Handles 34 continuous, correlated features well with no manual feature engineering
- Ranks feature importance — shows which pulses carry the most discriminating signal
- Bagging (bootstrap samples) + random feature subsets at each split reduce variance and
  overfitting compared to one deep decision tree
- Confirmed on this dataset: **Random Forest (93.2%) outperformed a single Decision Tree (87.5%)**
  on the identical train/test split

**Pipeline:**
1. Load & label-encode data, stratified 75/25 train/test split
2. Train `RandomForestClassifier` (200 trees, `max_features='sqrt'`)
3. Evaluate — accuracy, 5-fold cross-validation, confusion matrix, ROC-AUC, feature importance
4. Compare against a single Decision Tree baseline
5. Serve via this Streamlit app for live, interactive predictions

**Result:** ~93% test accuracy, ~0.97 ROC-AUC, consistent under 5-fold cross-validation.
""")
