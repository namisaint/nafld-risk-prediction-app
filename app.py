# app.py — NAFLD Risk Self-Screening (uses saved pipeline)
import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

# ==== EXACT feature schema (must match what you trained) ====
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

# dropdown choices (strings must match training after code->label mapping)
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

@st.cache_resource
def load_pipeline():
    # your model file is in the repo ROOT as 'nafld_pipeline.pkl'
    model_path = Path(__file__).parent / "nafld_pipeline.pkl"
    if not model_path.exists():
        st.error(
            f"Model file not found at: {model_path}\n\n"
            "Make sure 'nafld_pipeline.pkl' is committed to the repo root."
        )
        st.stop()
    return joblib.load(model_path)

pipe = load_pipeline()

with st.form("risk_assessment_form"):
    st.subheader("Sociodemographic & Lifestyle Data")

    # --- Sociodemographic ---
    st.markdown("### Sociodemographic Data")
    col1, col2 = st.columns(2)
    with col1:
        RIAGENDR = st.selectbox("Gender (RIAGENDR)", CHOICES["RIAGENDR"])
        RIDRETH3 = st.selectbox("Race/Ethnicity (RIDRETH3)", CHOICES["RIDRETH3"])
    with col2:
        RIDAGEYR = st.slider("Age in Years (RIDAGEYR)", 18, 99, 45)
        INDFMPIR = st.number_input("Family Income-to-Poverty Ratio (INDFMPIR)", min_value=0.0, value=1.5, step=0.1)

    # --- Alcohol & Smoking ---
    st.markdown("### Alcohol and Smoking Data")
    col1, col2 = st.columns(2)
    with col1:
        ALQ111 = st.selectbox("Had at least 12 alcohol drinks/1 yr? (ALQ111)", CHOICES["ALQ111"])
        ALQ142 = st.number_input("Average number of drinks on days consumed (ALQ142)", min_value=0.0, value=2.0, step=0.5)
        Is_Smoker_Cat = st.selectbox("Smoking Status (Is_Smoker_Cat)", CHOICES["Is_Smoker_Cat"])
    with col2:
        ALQ121 = st.number_input("How often do you drink in the last year? (ALQ121, days)", min_value=0.0, value=100.0, step=1.0)
        ALQ170 = st.number_input("Number of days had 5+/4+ drinks? (ALQ170)", min_value=0.0, value=0.0, step=1.0)
        ALQ151 = st.selectbox("Ever had 5+/4+ drinks in a day? (ALQ151)", CHOICES["ALQ151"])

    # --- Sleep ---
    st.markdown("### Sleep Data")
    col1, col2 = st.columns(2)
    with col1:
        SLQ050 = st.selectbox("How often have trouble sleeping? (SLQ050)", CHOICES["SLQ050"])
    with col2:
        SLD012 = st.slider("Average sleep hours per day (SLD012)", 1, 12, 7)
        SLQ120 = st.selectbox("Had a medical sleep diagnosis? (SLQ120)", CHOICES["SLQ120"])

    # --- Diet ---
    st.markdown("### Dietary Intake (Last 24 Hours)")
    col1, col2, col3 = st.columns(3)
    with col1:
        DR1TKCAL = st.number_input("Total Kilocalories (DR1TKCAL)", min_value=0.0, value=2000.0, step=50.0)
        DR1TPROT = st.number_input("Total Protein (DR1TPROT)", min_value=0.0, value=75.0, step=5.0)
    with col2:
        DR1TCARB = st.number_input("Total Carbohydrates (DR1TCARB)", min_value=0.0, value=250.0, step=5.0)
        DR1TSUGR = st.number_input("Total Sugar (DR1TSUGR)", min_value=0.0, value=90.0, step=5.0)
    with col3:
        DR1TFIBE = st.number_input("Total Fiber (DR1TFIBE)", min_value=0.0, value=25.0, step=1.0)
        DR1TTFAT = st.number_input("Total Fat (DR1TTFAT)", min_value=0.0, value=65.0, step=2.0)

    # --- Physical & Anthropometric ---
    st.markdown("### Physical & Anthropometric Data")
    col1, col2 = st.columns(2)
    with col1:
        PAQ620 = st.slider("Days of moderate activity per week (PAQ620)", 0, 7, 3)
    with col2:
        BMXBMI = st.number_input("BMI (BMXBMI)", min_value=10.0, max_value=80.0, value=28.0, step=0.1)

    submit = st.form_submit_button("Get Risk Assessment")

if submit:
    # build the single-row DataFrame in the exact training order
    row = {
        "RIAGENDR": RIAGENDR,
        "RIDRETH3": RIDRETH3,
        "RIDAGEYR": RIDAGEYR,
        "INDFMPIR": INDFMPIR,
        "ALQ111": ALQ111,
        "ALQ142": ALQ142,
        "Is_Smoker_Cat": Is_Smoker_Cat,
        "ALQ121": ALQ121,
        "ALQ170": ALQ170,
        "ALQ151": ALQ151,
        "SLQ050": SLQ050,
        "SLD012": SLD012,
        "SLQ120": SLQ120,
        "DR1TKCAL": DR1TKCAL,
        "DR1TPROT": DR1TPROT,
        "DR1TCARB": DR1TCARB,
        "DR1TSUGR": DR1TSUGR,
        "DR1TFIBE": DR1TFIBE,
        "DR1TTFAT": DR1TTFAT,
        "PAQ620": PAQ620,
        "BMXBMI": BMXBMI,
    }
    X = pd.DataFrame([row], columns=FEATURES)

    proba = float(pipe.predict_proba(X)[0, 1])
    pred = int(proba >= 0.5)

    st.subheader("Your Results")
    st.metric("Predicted probability", f"{proba:.3f}")
    if pred == 1:
        st.error("Based on your data, you are at higher risk for NAFLD.")
    else:
        st.success("Based on your data, you are likely at lower risk (threshold 0.5).")

    st.caption("This is a screening tool and not a diagnosis.")
