import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_auc_score, accuracy_score, precision_score, recall_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
import numpy as np

# --- Pipeline and Data Loading ---

@st.cache_data
def get_local_data():
    try:
        df = pd.read_csv('preprocessed_data_sample.csv') 
        return df
    except FileNotFoundError:
        st.error("The data file 'preprocessed_data_sample.csv' was not found. Please upload it to GitHub.")
        st.stop()


@st.cache_resource
def load_pipeline():
    pipeline_path = 'models.pkl'
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

st.title("NAFLD Risk Self-Screening Tool")
st.write("Enter your data below to receive a non-invasive risk assessment.")

# Define the exact list of features your model expects
EXPECTED_FEATURES = [
    'Gender','Age in years','Race/Ethnicity','Family income ratio',
    'Smoking status','Sleep Disorder Status','Sleep duration (hours/day)',
    'Work schedule duration (hours)','Physical activity (minutes/day)','BMI',
    'Alcohol consumption (days/week)','Alcohol drinks per day',
    'Number of days drank in the past year','Max number of drinks on any single day',
    'Alcohol intake frequency (drinks/day)',
    'Total calorie intake (kcal)','Total protein intake (grams)','Total carbohydrate intake (grams)',
    'Total sugar intake (grams)','Total fiber intake (grams)','Total fat intake (grams)'
]


# Create tabs for the app
tab1, tab2 = st.tabs(["Prediction Tool", "Model Comparison"])

with tab1:
    # Use a form to group inputs and prevent re-runs
    with st.form("risk_assessment_form"):
        st.subheader("Sociodemographic & Lifestyle Data")

        # Define the inputs based on the provided NHANES codes
        st.markdown("### Sociodemographic Data")
        col1, col2 = st.columns(2)
        with col1:
            gender_input = st.selectbox('Gender', options=['Male', 'Female'])
            race_ethnicity_input = st.selectbox('Race/Ethnicity', options=['Mexican American', 'Other Hispanic', 'Non-Hispanic White', 'Non-Hispanic Black', 'Other Race - Including Multi-Racial'])
        with col2:
            age = st.slider('Age in years', min_value=18, max_value=80, value=45)
            income_ratio = st.slider('Family income ratio', min_value=0.0, max_value=5.0, value=1.5, step=0.01)

        st.markdown("### Alcohol and Smoking Data")
        col1, col2 = st.columns(2)
        with col1:
            smoker_status_input = st.selectbox('Smoking status', options=['Never', 'Former', 'Current'])
            sleep_disorder_status_input = st.selectbox('Sleep Disorder Status', options=['Yes', 'No'])
        with col2:
            sleep_duration = st.slider('Sleep duration (hours/day)', min_value=1, max_value=12, value=7)
            work_schedule_duration = st.slider('Work schedule duration (hours)', min_value=0, max_value=24, value=8)

        st.markdown("### Physical Activity & Diet Data")
        col1, col2, col3 = st.columns(3)
        with col1:
            physical_activity = st.slider('Physical activity (minutes/day)', min_value=0, max_value=300, value=30)
            bmi = st.number_input('BMI', min_value=15.0, max_value=50.0, value=25.0)
        with col2:
            total_calories = st.number_input('Total calorie intake (kcal)', min_value=0, value=2000)
            total_protein = st.number_input('Total protein intake (grams)', min_value=0, value=75)
        with col3:
            total_carbs = st.number_input('Total carbohydrate intake (grams)', min_value=0, value=250)
            total_sugar = st.number_input('Total sugar intake (grams)', min_value=0, value=90)
            total_fiber = st.number_input('Total fiber intake (grams)', min_value=0, value=25)
            total_fat = st.number_input('Total fat intake (grams)', min_value=0, value=65)
        
        st.markdown("### Alcohol Data")
        col1, col2 = st.columns(2)
        with col1:
            alcohol_days = st.slider('Alcohol consumption (days/week)', min_value=0, max_value=7, value=0)
            alcohol_drinks_per_day = st.slider('Alcohol drinks per day', min_value=0, max_value=20, value=0)
        with col2:
            alcohol_freq = st.slider('Number of days drank in the past year', min_value=0, max_value=365, value=0)
            max_drinks = st.slider('Max number of drinks on any single day', min_value=0, max_value=20, value=0)


        submitted = st.form_submit_button("Get Risk Assessment")

        if submitted:
            # Collect raw inputs from the user
            user_inputs = {
                'Gender': gender_input,
                'Age in years': age,
                'Race/Ethnicity': race_ethnicity_input,
                'Family income ratio': income_ratio,
                'Smoking status': smoker_status_input,
                'Sleep Disorder Status': sleep_disorder_status_input,
                'Sleep duration (hours/day)': sleep_duration,
                'Work schedule duration (hours)': work_schedule_duration,
                'Physical activity (minutes/day)': physical_activity,
                'BMI': bmi,
                'Alcohol consumption (days/week)': alcohol_days,
                'Alcohol drinks per day': alcohol_drinks_per_day,
                'Number of days drank in the past year': alcohol_freq,
                'Max number of drinks on any single day': max_drinks,
                'Alcohol intake frequency (drinks/day)': alcohol_freq,
                'Total calorie intake (kcal)': total_calories,
                'Total protein intake (grams)': total_protein,
                'Total carbohydrate intake (grams)': total_carbs,
                'Total sugar intake (grams)': total_sugar,
                'Total fiber intake (grams)': total_fiber,
                'Total fat intake (grams)': total_fat,
            }

            # Create a DataFrame from the user's input
            user_df = pd.DataFrame([user_inputs])

            # Make the prediction using the pipeline
            prediction = pipeline.predict(user_df)[0]
            prediction_proba = pipeline.predict_proba(user_df)[:, 1][0]
            
            st.subheader("Your Results")
            if prediction == 1:
                st.error("Based on your data, you are at risk for NAFLD.")
            else:
                st.success("Based on your data, you are likely at lower risk (threshold 0.5).")

            st.info(f"Model Confidence: {prediction_proba:.2%}")

            # SHAP Explainability
            st.subheader("Explanation of the Prediction")
            
            explainer = shap.TreeExplainer(pipeline.named_steps['model'])
            preprocessed_user_data = pipeline.named_steps['preprocess'].transform(user_df)

            # Need to get feature names from the preprocessor to display correctly
            feature_names = pipeline.named_steps['preprocess'].get_feature_names_out()

            # The SHAP values will be a list of two arrays for a binary classifier
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
            
with tab2:
    st.header("Model Performance and Comparison")
    st.info("This section proves the front-end and back-end are working together. The data for this analysis is a sample from your dissertation's dataset.")
    
    if not df_sample.empty:
        try:
            # We assume your sample data has the target variable 'Fatty_Liver'
            X_all = df_sample.drop(columns=['Fatty_Liver'])
            y_true = df_sample['Fatty_Liver']
            
            # Predict on the sample data to get a confusion matrix and metrics
            y_pred = pipeline.predict(X_all)
            y_proba = pipeline.predict_proba(X_all)[:, 1]
            
            # --- Display Metrics ---
            st.subheader("Random Forest Model Metrics")
            
            accuracy = accuracy_score(y_true, y_pred)
            precision = precision_score(y_true, y_pred)
            recall = recall_score(y_true, y_pred)
            roc_auc = roc_auc_score(y_true, y_proba)
            
            metrics_df = pd.DataFrame({
                "Metric": ["Accuracy", "Precision", "Recall", "ROC AUC"],
                "Value": [accuracy, precision, recall, roc_auc]
            })
            st.table(metrics_df.set_index("Metric"))

            # --- Display Confusion Matrix ---
            st.subheader("Confusion Matrix")
            cm = confusion_matrix(y_true, y_pred)
            fig, ax = plt.subplots()
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
            ax.set_xlabel("Predicted")
            ax.set_ylabel("Actual")
            st.pyplot(fig)
            
            st.write("These metrics and confusion matrix are calculated by running the pipeline on a sample of your dataset.")
            
        except KeyError:
            st.error("The 'Fatty_Liver' column was not found in the sample data.")
        except Exception as e:
            st.error(f"An error occurred during model analysis: {e}")
    else:
        st.warning("No data available to perform analysis.")

