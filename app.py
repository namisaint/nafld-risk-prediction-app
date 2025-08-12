# app.py — NAFLD Risk Self-Screening (uses saved scikit-learn Pipeline in repo root)

import streamlit as st
import pandas as pd
import joblib
from pathlib import Path
import sys

# ===== EXACT feature schema (must match training) =====
FEATURES = [
    # Sociodemographic
    "RIAGENDR","RIDRETH3","RIDAGEYR","INDFMPIR",
    # Alcohol & Smoking
    "ALQ111","ALQ142","Is_Smoker_Cat","ALQ121","ALQ170","ALQ151",
    # Sleep
    "SLQ050","SLD012","SLQ120",
    # Diet (24h)
    "DR1TKCAL","DR1TPROT","DR1TCARB","DR1TSUGR","DR1TFIBE","DR1TTFAT",
    # Physical & Anthropometric
    "PAQ620","BMXBMI"
]

CHOICES = {
    "RIAGENDR": ["Male","Female"],
    "RIDRETH3": [
        "Mexican American","Other Hispanic","Non-Hispanic White",
        "Non-Hispanic Black","Non-Hispanic Asian","Other/Multi"
    ],
    "ALQ111": ["Yes","No"],
    "ALQ151": ["Yes","No"],
    "SLQ120": ["Yes","No"],
    "SLQ050": ["Never","Rarely","Sometimes","Often","Almost always"],
    "Is_Smoker_Cat": ["Never","Former","Current"],
}

st.set_page_config(page_title="NAFLD Risk Self-Screening Tool", page_icon="🧪", layout="wide")
st.title("NAFLD Risk Self-Screening Tool")
st.write("Enter your data below to receive a non-invasive risk assessment.")

# Show environment versions (helps verify)
try:
    import sklearn, numpy
    st.caption(f"Python {sys.version.split()[0]} • scikit-learn {sklearn.__version__} • numpy {numpy.__version__}")
except Exception:
    pass

@st.cache_resource
def load_pipeline():
    model_path = Path(__file__).parent / "nafld_pipeline.pkl"  # <-- repo root
    if not model_path.exists():
        st.error(f"Model file not found at: {model_path}\nMake sure 'nafld_pipeline.pkl' is committed to the repo root.")
        st.stop()
    try:
        return joblib.load(model_path)
    except Exception as e:
        import sklearn, numpy
        st.error(
            "Failed to load model pickle (likely version mismatch).\n\n"
            f"Runtime → Python {sys.version.split()[0]}, sklearn {sklearn.__version__}, numpy {numpy.__version__}\n\n"
            f"Raw error: {type(e).__name__}: {e}"
        )
        st.stop()

pipe = load_pipeline()

with st.form("risk_assessment_form"):
    st.subheader("Sociodemographic & Lifestyle Data")

    # Sociodemographic
    st.markdown("### Sociodemographic Data")
    col1, col2 = st.columns(2)
    with col1:
        RIAGENDR = st.selectbox("Gender (RIAGENDR)", CHOICES["RIAGENDR"])
        RIDRETH3 = st.selectbox("Race/Ethnicity (RIDRETH3)", CHOICES["RIDRETH3"])
    with col2:
        RIDAGEYR = st.slider("Age in Years (RIDAGEYR)", 18, 99, 45)
        INDFMPIR = st.number_input("Family Income-to-Poverty Ratio (INDFMPIR)", min_value=0.0, value=1.5, step=0.1)

    # Alcohol & Smoking
    st.markdown("### Alcohol and Smoking Data")
    col1, col2 = st.columns(2)
    with col1:
        ALQ111 = st.selectbox("Had at least 12 alcohol drinks/1 yr? (ALQ111)", CHOICES["ALQ111"])
        ALQ142 = st.number_input("Average number of drinks on days consumed (ALQ142)", min_value=0.0, value=2.0, step=0.5)
        Is_Smoker_Cat = st.selectbox("Smoking Status (Is_Smoker_Cat)", CHOICES["Is_Smoker_Cat"])
    with col2:
        ALQ121 = st.number_input("How often do you drink in the last year? (ALQ12_
