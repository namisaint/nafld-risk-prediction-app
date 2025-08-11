import streamlit as st
import pandas as pd
import pickle
import numpy as np
import shap
import matplotlib.pyplot as plt
import seaborn as sns

# --- Data and Model Loading ---

@st.cache_data
def get_local_data():
    try:
        # Load a small sample of the pre-processed data
        df = pd.read_csv('preprocessed_data_sample.csv') 
        return df
    except FileNotFoundError:
        st.error("The data file 'preprocessed_data_sample.csv' was not found. Please upload it to GitHub.")
        st.stop()


@st.cache_resource
def load_model():
    model_path = 'rf_lifestyle_model (1).pkl'
    try:
        with open(model_path, 'rb') as f:
            return pickle.load(f)
    except FileNotFoundError:
        st.error(f"The model file '{model_path}' was not found. Please ensure it is uploaded to GitHub.")
        st.stop()
    except pickle.UnpicklingError:
        st.error(f"The model file '{model_path}' could not be loaded. This often happens due to a Python version mismatch.")
        st.stop()

# Load the data and model
df_sample = get_local_data()
model = load_model()

# --- Preprocessing Pipeline Function ---

def preprocessing_pipeline(user_inputs, df_sample_data):
    # This function ensures the user's input matches the model's expectations
    
    # Create a DataFrame with the user's input
    user_df = pd.DataFrame(user_inputs)
    
    # Reindex the user data to match the feature order of the original data
    # This is the crucial step that fixes the static prediction issue
    expected_columns = df_sample_data.drop(columns=['Fatty_Liver']).columns
    user_df = user_df.reindex(columns=expected_columns, fill_value=0)

    return user_df


# --- Main Streamlit App Logic ---

st.title("NAFLD Risk Self-Screening Tool")
st.write("Enter your data below to receive a non-invasive risk assessment.")

with st.form("risk_assessment_form"):
    st.subheader("Sociodemographic & Lifestyle Data")

    # Define the inputs based on the provided NHANES codes
    st.markdown("### Sociodemographic Data")
    col1, col2 = st.columns(2)
    with col1:
        gender_input = st.selectbox('Gender (RIAGENDR)', options=['Male', 'Female'])
        race_ethnicity_input = st.selectbox('Race/Ethnicity (RIDRETH3)', options=['Mexican American', 'Other Hispanic', 'Non-Hispanic White', 'Non-Hispanic Black', 'Other Race - Including Multi-Racial'])
    with col2:
        age = st.slider('Age in Years (RIDAGEYR)', min_value=18, max_value=80, value=45)
        income_ratio = st.slider('Family Income-to-Poverty Ratio (INDFMPIR)', min_value=0.0, max_value=5.0, value=1.5, step=0.01)

    st.markdown("### Alcohol and Smoking Data")
    col1, col2 = st.columns(2)
    with col1:
        has_drank_12_input = st.selectbox('Had at least 12 alcohol drinks/1 yr? (ALQ111)', options=['Yes', 'No'])
        drinks_per_day = st.slider('Average number of drinks on days consumed (ALQ142)', min_value=0, max_value=20, value=2)
        smoker_status_input = st.selectbox('Smoking Status (Is_Smoker_Cat)', options=['Never', 'Former', 'Current'])
    with col2:
        how_often_drink = st.slider('How often do you drink in the last year? (ALQ121)', min_value=0, max_value=365, value=100)
        num_heavy_drink_days = st.slider('Number of days had 5+/4+ drinks? (ALQ170)', min_value=0, max_value=365, value=0)
        has_heavy_drank_input = st.selectbox('Ever had 5+/4+ drinks in a day? (ALQ151)', options=['Yes', 'No'])

    st.markdown("### Sleep Data")
    col1, col2 = st.columns(2)
    with col1:
        sleep_trouble_input = st.selectbox('How often have trouble sleeping? (SLQ050)', options=['Never', 'Rarely', 'Sometimes', 'Often'])
    with col2:
        sleep_hours = st.slider('Average sleep hours per day (SLD012)', min_value=1, max_value=12, value=7)
        sleep_diagnosis_input = st.selectbox('Had a medical sleep diagnosis? (SLQ120)', options=['Yes', 'No'])

    st.markdown("### Dietary Intake (Last 24 Hours)")
    col1, col2, col3 = st.columns(3)
    with col1:
        calories = st.number_input('Total Kilocalories (DR1TKCAL)', min_value=0, value=2000)
        protein = st.number_input('Total Protein (DR1TPROT)', min_value=0, value=75)
    with col2:
        carbs = st.number_input('Total Carbohydrates (DR1TCARB)', min_value=0, value=250)
        sugar = st.number_input('Total Sugar (DR1TSUGR)', min_value=0, value=90)
    with col3:
        fiber = st.number_input('Total Fiber (DR1TFIBE)', min_value=0, value=25)
        total_fat = st.number_input('Total Fat (DR1TTFAT)', min_value=0, value=65)
        
    st.markdown("### Physical & Anthropometric Data")
    col1, col2 = st.columns(2)
    with col1:
        paq620 = st.slider('Days of moderate activity per week (PAQ620)', min_value=0, max_value=7, value=3)
    with col2:
        bmi = st.number_input('BMI (BMXBMI)', min_value=15.0, max_value=50.0, value=25.0)

    submitted = st.form_submit_button("Get Risk Assessment")

    if submitted:
        # --- Data Encoding and Prediction Logic ---
        
        # Mapping for categorical variables
        gender_map = {'Male': 1, 'Female': 2}
        race_ethnicity_map = {
            'Mexican American': 1, 'Other Hispanic': 2,
            'Non-Hispanic White': 3, 'Non-Hispanic Black': 4,
            'Other Race - Including Multi-Racial': 6
        }
        has_drank_map = {'Yes': 1, 'No': 2}
        smoker_map = {'Never': 0, 'Former': 1, 'Current': 2}
        sleep_trouble_map = {'Never': 1, 'Rarely': 2, 'Sometimes': 3, 'Often': 4}
        sleep_diagnosis_map = {'Yes': 1, 'No': 2}
        
        # Collect raw inputs from the user
        user_inputs = {
            'RIAGENDR': [gender_map.get(gender_input)],
            'RIDAGEYR': [age],
            'RIDRETH3': [race_ethnicity_map.get(race_ethnicity_input)],
            'INDFMPIR': [income_ratio],
            'ALQ111': [has_drank_map.get(has_drank_12_input)],
            'ALQ121': [how_often_drink],
            'ALQ142': [drinks_per_day],
            'ALQ151': [has_drank_map.get(has_heavy_drank_input)],
            'ALQ170': [num_heavy_drink_days],
            'Is_Smoker_Cat': [smoker_map.get(smoker_status_input)],
            'SLQ050': [sleep_trouble_map.get(sleep_trouble_input)],
            'SLQ120': [sleep_diagnosis_map.get(sleep_diagnosis_input)],
            'SLD012': [sleep_hours],
            'DR1TKCAL': [calories],
            'DR1TPROT': [protein],
            'DR1TCARB': [carbs],
            'DR1TSUGR': [sugar],
            'DR1TFIBE': [fiber],
            'DR1TTFAT': [total_fat],
            'PAQ620': [paq620],
            'BMXBMI': [bmi],
        }
        
        # Create a DataFrame from the user's input and ensure column order matches
        user_data_processed = preprocessing_pipeline(user_inputs, df_sample)

        # Make the prediction
        prediction = model.predict(user_data_processed)[0]

        st.subheader("Your Results")
        if prediction == 1:
            st.error("Based on your data, you are at risk for NAFLD.")
        else:
            st.success("Based on your data, you are likely not at risk for NAFLD.")

        # --- SHAP Explainability ---
        st.subheader("Explanation of the Prediction")
        
        explainer = shap.TreeExplainer(model)
        # The SHAP values will be a list of two arrays for a binary classifier
        # We need to get the values for class 1, which represents 'at risk'
        shap_values_to_plot = explainer.shap_values(user_data_processed)[1]
        
        st.write("This chart shows how each factor contributed to your risk score:")
        shap.initjs()
        plt.figure()
        shap.force_plot(explainer.expected_value[1], shap_values_to_plot, user_data_processed, show=False)
        st.pyplot(plt.gcf())
