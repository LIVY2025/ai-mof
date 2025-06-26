import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
import joblib
from scipy.stats import boxcox

# ========== Page Setup ==========
st.set_page_config(layout="wide", page_title="Stacking Model Prediction & SHAP Analysis", page_icon="📊")

st.markdown("""
<style>
body {
  background-color: #F9FAFB;
  font-family: 'Segoe UI', sans-serif;
}

h1, h2, h3 {
  color: #333;
  border-bottom: 2px solid #e1e4e8;
  padding-bottom: 6px;
}

section.main > div {
  max-width: 1000px;
  margin: auto;
}

.card {
  background-color: #fff;
  padding: 1.5rem;
  margin-top: 1.5rem;
  border-radius: 0.5rem;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05);
}

.card table {
  width: 100%;
  border-collapse: collapse;
  text-align: center;
}

.card th, .card td {
  padding: 0.5rem 1rem;
  border: 1px solid #eee;
}
</style>
""", unsafe_allow_html=True)

st.title("📊 Stacking Model Prediction & SHAP Analysis")
st.markdown("""
This web application predicts toluene adsorption capacity (TSN) of MOF materials based on input features, and visualizes feature contributions using SHAP.

**Model**: Stacking Regressor with 8 base learners and MLP meta-learner, achieving test R² ≈ 0.882.

---
""")

# ========== Load Models ==========
@st.cache_resource
def load_models():
    return {
        "stack": joblib.load("stacking_model.pkl"),
        "qt_lcd": joblib.load("qt_lcd.pkl"),
        "qt_gsa": joblib.load("qt_GSA.pkl"),
        "qt_density": joblib.load("qt_Density.pkl"),
        "lambda_kt": joblib.load("lambda_Ktoluene.pkl"),
        "lambda_vf": joblib.load("lambda_vf.pkl"),
        "qt_TSN": joblib.load("qt_TSN.pkl"),
    }

models = load_models()

# ========== Sidebar Input ==========
st.sidebar.header("🔧 Input Features")
LCD = st.sidebar.number_input("LCD (6.03 – 39.11)", min_value=6.03338, max_value=39.1106, value=8.33, format="%g")
Vf = st.sidebar.number_input("Vf (0.257 – 0.918)", min_value=0.2574, max_value=0.9182, value=0.5726, format="%g")
GSA = st.sidebar.number_input("GSA (204.91 – 7061.42)", min_value=204.912, max_value=7061.42, value=701.884, format="%g")
Density = st.sidebar.number_input("Density (0.237 – 2.865)", min_value=0.237838, max_value=2.86501, value=1.51454, format="%g")
Ktoluene = st.sidebar.number_input("Ktoluene (2.7e-5 – 28527.4)", min_value=0.000027383, max_value=28527.4, value=0.013545, format="%g")
predict_btn = st.sidebar.button("🔍 Predict")

# ========== Prediction Logic ==========
if predict_btn:
    try:
        lcd_q = models["qt_lcd"].transform([[LCD]])[0, 0]
        gsa_q = models["qt_gsa"].transform([[GSA]])[0, 0]
        density_q = models["qt_density"].transform([[Density]])[0, 0]
        vf_bc = boxcox(np.array([Vf]), lmbda=models["lambda_vf"])[0]
        ktol_bc = boxcox(np.array([Ktoluene]), lmbda=models["lambda_kt"])[0]

        df_trans = pd.DataFrame({
            "Feature": ["LCD", "Vf", "GSA", "Density", "Ktoluene"],
            "Original Value": [LCD, Vf, GSA, Density, Ktoluene],
            "Transformed Value": [lcd_q, vf_bc, gsa_q, density_q, ktol_bc]
        })

        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("🔄 Feature Transformation")
        st.dataframe(df_trans.style.format(precision=6), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

        X_user = df_trans["Transformed Value"].to_numpy().reshape(1, -1)
        pred_trans = models["stack"].predict(X_user)[0]
        pred_orig = models["qt_TSN"].inverse_transform([[pred_trans]])[0, 0]

        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("📈 Prediction Result")
        st.write(f"**Transformed TSN**: {pred_trans:.6f}")
        st.success(f"**Original TSN**: {pred_orig:.6f}")
        st.markdown('</div>', unsafe_allow_html=True)

    except Exception as e:
        st.error(f"Prediction error: {e}")

# ========== SHAP Visualization ==========
st.header("🔬 SHAP Feature Contribution")

shap_sections = [
    ("1. Base Learners SHAP", "SHAP summary from base models (RandomForest, XGB, etc.)", "summary_plot.png"),
    ("2. Meta Learner SHAP", "SHAP for the meta-learner (Linear Regression)", "SHAP Contribution Analysis for the Meta-Learner in the Second Layer of Stacking Regressor.png"),
    ("3. Overall SHAP", "Combined SHAP analysis for the whole stacking model", "Based on the overall feature contribution analysis of SHAP to the stacking model.png")
]

for title, desc, path in shap_sections:
    st.subheader(title)
    st.markdown(desc)
    try:
        st.image(Image.open(path), use_column_width=True)
    except FileNotFoundError:
        st.warning(f"Image not found: {path}")

# ========== Footer ==========
st.markdown("""
---
### 📝 About This App
Built for academic presentation of MOF material modeling and interpretation using stacking regression + SHAP. 
For questions or collaborations, contact: `m202311485@xs.ustb.edu.cn`
""")
