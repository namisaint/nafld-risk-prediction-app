import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# ===== exact feature schema (must match training) =====
FEATURES = [
    'Gender', 'Age in years', 'Race/Ethnicity', 'Family income ratio',
    'Smoking status', 'Sleep Disorder Status', 'Sleep duration (hours/day)',
    'Work schedule duration (hours)', 'Physical activity (minutes/day)', 'BMI',
    'Alcohol consumption (days/week)', 'Alcohol drinks per day',
    'Number of days drank in the past year', 'Max number of drinks on any single day',
    'Alcohol intake frequency (drinks/day)',
    'Total calorie intake (kcal)', 'Total protein intake (grams)', 'Total carbohydrate intake (grams)',
    'Total sugar intake (grams)', 'Total fiber intake (grams)', 'Total fat intake (grams)'
]

# dropdown choices — strings must match what the pipeline saw during training
CHOICES = {
    "RIAGENDR": ["Male", "Female"],
    "RIDRETH3": [
        "Mexican American", "Other Hispanic", "Non-Hispanic White",
        "Non-Hispanic Black", "Non-Hispanic Asian", "Other/Multi"
    ],
    "ALQ111": ["Yes", "No"],
    "ALQ151": ["Yes", "No"],
    "SLQ120": ["Yes", "No"],
    "SLQ050": ["Never", "Rarely", "Sometimes", "Often", "Almost always"],
    "Is_Smoker_Cat": ["Never", "Former", "Current"],
}

st.set_page_config(page_title="NAFLD Risk Self-Screening Tool", page_icon="🧪", layout="wide")
st.title("NAFLD Risk Self-Screening Tool")
st.write("Enter your data below to receive a non-invasive risk assessment.")

# show env versions (helps catch version mismatches)
try:
    import sklearn, numpy
    st.caption(f"Python {sys.version.split()[0]} • scikit-learn {sklearn.__version__} • numpy {numpy.__version__}")
except Exception:
    pass

# --- Pipeline and Data Loading ---
@st.cache_data
def get_local_data():
    try:
        df = pd.read_csv(Path('preprocessed_data_sample.csv'))
        return df
    except FileNotFoundError:
        st.error("The data file 'preprocessed_data_sample.csv' was not found. Please upload it to GitHub.")
        st.stop()

@st.cache_resource
def load_pipeline():
    pipeline_path = Path('nafld_pipeline.pkl')
    try:
        with open(pipeline_path, 'rb') as f:
            return joblib.load(f)
    except FileNotFoundError:
        st.error(f"The pipeline file '{pipeline_path}' was not found. Please ensure it is uploaded to GitHub.")
        st.stop()
    except Exception as e:
        st.error(f"The pipeline file could not be loaded. This often happens due to a Python version mismatch. Details: {e}")
        st.stop()

# Load the data and model pipeline
df_sample = get_local_data()
pipeline = load_pipeline()

# --- Main Streamlit App Logic ---
with st.form("risk_assessment_form"):
    st.subheader("Sociodemographic & Lifestyle Data")

    # === Sociodemographic ===
    st.markdown("### Sociodemographic Data")
    col1, col2 = st.columns(2)
    with col1:
        RIAGENDR = st.selectbox("Gender (RIAGENDR)", CHOICES["RIAGENDR"])
        RIDRETH3 = st.selectbox("Race/Ethnicity (RIDRETH3)", CHOICES["RIDRETH3"])
    with col2:
        RIDAGEYR = st.slider("Age in Years (RIDAGEYR)", 18, 99, 45)
        INDFMPIR = st.number_input("Family Income-to-Poverty Ratio (INDFMPIR)", min_value=0.0, value=1.5, step=0.1)

    # === Alcohol & Smoking ===
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

    # === Sleep ===
    st.markdown("### Sleep Data")
    col1, col2 = st.columns(2)
    with col1:
        SLQ050 = st.selectbox("How often have trouble sleeping? (SLQ050)", CHOICES["SLQ050"])
    with col2:
        SLD012 = st.slider("Average sleep hours per day (SLD012)", 1, 12, 7)
        SLQ120 = st.selectbox("Had a medical sleep diagnosis? (SLQ120)", CHOICES["SLQ120"])

    # === Diet (24h) ===
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

    # === Physical & Anthropometric ===
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
        "RIAGENDR": RIAGENDR, "RIDRETH3": RIDRETH3, "RIDAGEYR": RIDAGEYR, "INDFMPIR": INDFMPIR,
        "ALQ111": ALQ111, "ALQ142": ALQ142, "Is_Smoker_Cat": Is_Smoker_Cat, "ALQ121": ALQ121, "ALQ170": ALQ170, "ALQ151": ALQ151,
        "SLQ050": SLQ050, "SLD012": SLD012, "SLQ120": SLQ120,
        "DR1TKCAL": DR1TKCAL, "DR1TPROT": DR1TPROT, "DR1TCARB": DR1TCARB, "DR1TSUGR": DR1TSUGR, "DR1TFIBE": DR1TFIBE, "DR1TTFAT": DR1TTFAT,
        "PAQ620": PAQ620, "BMXBMI": BMXBMI,
    }
    X = pd.DataFrame([row], columns=FEATURES)

    # predict
    proba = float(pipeline.predict_proba(X)[0, 1])
    pred = int(proba >= 0.5)

    st.subheader("Your Results")
    st.metric("Predicted probability", f"{proba:.3f}")
    if pred == 1:
        st.error("Based on your data, you are at higher risk for NAFLD.")
    else:
        st.success("Based on your data, you are likely at lower risk (threshold 0.5).")

    st.caption("This is a screening tool, not a diagnosis.")

    st.subheader("Explanation of the Prediction")
    explainer = shap.TreeExplainer(pipeline.named_steps['model'])
    preprocessed_user_data = pipeline.named_steps['preprocess'].transform(X)
    feature_names = pipeline.named_steps['preprocess'].get_feature_names_out()
    shap_values = explainer.shap_values(preprocessed_user_data)
    
    st.write("This chart shows how each factor contributed to your risk score:")
    shap.initjs()
    plt.figure()
    shap.force_plot(
        explainer.expected_value[1], 
        shap_values[1], 
        preprocessed_user_data, 
        feature_names=feature_names,
        show=False
    )
    st.pyplot(plt.gcf())
