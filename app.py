import os
import streamlit as st
import joblib
import numpy as np
import pandas as pd
import pickle

st.title("Employee Attrition Prediction (Single + Batch)")

# Try to load the trained model (supports joblib or pickle files)
# Prefer the exact model filename you used when training, fall back to common names
MODEL_CANDIDATES = ["random_forest_model.pkl", "model.pkl", "random_forest_model.joblib", "model.joblib"]
model = None
found_model = None
for mpath in MODEL_CANDIDATES:
    if os.path.exists(mpath):
        try:
            model = joblib.load(mpath)
            found_model = mpath
            break
        except Exception:
            try:
                with open(mpath, "rb") as f:
                    model = pickle.load(f)
                    found_model = mpath
                    break
            except Exception:
                model = None

if model is None:
    st.error(f"Model file not found or failed to load. Checked: {MODEL_CANDIDATES}")
else:
    st.info(f"Loaded model from: {found_model}")

# Optionally load a scaler if present (used at training time)
scaler = None
SCALER_CANDIDATES = ["standard_scaler.pkl", "standard_scaler.joblib", "scaler.pkl", "scaler.joblib"]
for candidate in SCALER_CANDIDATES:
    if os.path.exists(candidate):
        try:
            scaler = joblib.load(candidate)
            break
        except Exception:
            try:
                with open(candidate, "rb") as f:
                    scaler = pickle.load(f)
                    break
            except Exception:
                scaler = None

if scaler is not None:
    st.info("Scaler loaded for feature scaling")
else:
    st.info("No scaler found; predictions will proceed without scaling")

# CSV path used earlier for schema and defaults
DF_PATH = "EmployeeData_preprocessed.csv"
df = pd.read_csv(DF_PATH)

# --- Feature definitions (must match training) ---
num_inputs = [
    'Age', 'DailyRate', 'DistanceFromHome', 'Education', 'EnvironmentSatisfaction', 'HourlyRate',
    'JobInvolvement', 'JobLevel', 'JobSatisfaction', 'MonthlyIncome', 'MonthlyRate',
    'NumCompaniesWorked', 'PercentSalaryHike', 'PerformanceRating', 'RelationshipSatisfaction',
    'StockOptionLevel', 'TotalWorkingYears', 'TrainingTimesLastYear', 'WorkLifeBalance',
    'YearsAtCompany', 'YearsInCurrentRole', 'YearsSinceLastPromotion', 'YearsWithCurrManager'
]

cat_inputs = {
    'BusinessTravel': ['Travel_Frequently', 'Travel_Rarely', 'Non-Travel'],
    'Department': ['Research & Development', 'Sales', 'Human Resources'],
    'EducationField': ['Life Sciences', 'Marketing', 'Medical', 'Other', 'Technical Degree', 'Human Resources'],
    'Gender': ['Male', 'Female'],
    'JobRole': [
        'Human Resources', 'Laboratory Technician', 'Manager', 'Manufacturing Director',
        'Research Director', 'Research Scientist', 'Sales Executive', 'Sales Representative'
    ],
    'MaritalStatus': ['Married', 'Single', 'Divorced'],
    'OverTime': ['Yes', 'No']
}

onehot_cols = [
    'BusinessTravel_Travel_Frequently', 'BusinessTravel_Travel_Rarely',
    'Department_Research & Development', 'Department_Sales',
    'EducationField_Life Sciences', 'EducationField_Marketing', 'EducationField_Medical',
    'EducationField_Other', 'EducationField_Technical Degree', 'Gender_Male',
    'JobRole_Human Resources', 'JobRole_Laboratory Technician', 'JobRole_Manager',
    'JobRole_Manufacturing Director', 'JobRole_Research Director', 'JobRole_Research Scientist',
    'JobRole_Sales Executive', 'JobRole_Sales Representative', 'MaritalStatus_Married',
    'MaritalStatus_Single', 'OverTime_Yes'
]

engineered_cols = [
    'Multi_Age_JobLevel', 'Ratio_JobLevel_Age', 'Ratio_YearsAtCompany_Age', 'Multi_Age_StockOptionLevel',
    'Ratio_StockOptionLevel_Age', 'Minus_YearsAtCompany_YearsInCurrentRole',
    'Minus_YearsAtCompany_YearsSinceLastPromotion', 'Ratio_YearsSinceLastPromotion_YearsAtCompany',
    'Ratio_YearsInCurrentRole_YearsAtCompany', 'Multi_DistanceFromHome_OverTime', 'SalesExecutive_OverTime',
    'Scientist_Young', 'TravelFreq_OverTime', 'R&D_StockOptions', 'Income_per_JobLevel',
    'YearsWithCurrManager_per_YearsAtCompany', 'PromotionRate'
]

feature_columns = [
    'Age', 'DailyRate', 'DistanceFromHome', 'Education', 'EnvironmentSatisfaction', 'HourlyRate',
    'JobInvolvement', 'JobLevel', 'JobSatisfaction', 'MonthlyIncome', 'MonthlyRate',
    'NumCompaniesWorked', 'PercentSalaryHike', 'PerformanceRating', 'RelationshipSatisfaction',
    'StockOptionLevel', 'TotalWorkingYears', 'TrainingTimesLastYear', 'WorkLifeBalance',
    'YearsAtCompany', 'YearsInCurrentRole', 'YearsSinceLastPromotion', 'YearsWithCurrManager',
    *onehot_cols,
    *engineered_cols
]

def add_engineered_features(df_):
    df = df_.copy()
    df['Multi_Age_JobLevel'] = df['Age'] * df['JobLevel']
    df['Ratio_JobLevel_Age'] = df['JobLevel'] / (df['Age'] + 1)
    df['Ratio_YearsAtCompany_Age'] = df['YearsAtCompany'] / (df['Age'] + 1)
    df['Multi_Age_StockOptionLevel'] = df['Age'] * df['StockOptionLevel']
    df['Ratio_StockOptionLevel_Age'] = df['StockOptionLevel'] / (df['Age'] + 1)
    df['Minus_YearsAtCompany_YearsInCurrentRole'] = df['YearsAtCompany'] - df['YearsInCurrentRole']
    df['Minus_YearsAtCompany_YearsSinceLastPromotion'] = df['YearsAtCompany'] - df['YearsSinceLastPromotion']
    df['Ratio_YearsSinceLastPromotion_YearsAtCompany'] = df['YearsSinceLastPromotion'] / (df['YearsAtCompany'] + 1)
    df['Ratio_YearsInCurrentRole_YearsAtCompany'] = df['YearsInCurrentRole'] / (df['YearsAtCompany'] + 1)
    df['Multi_DistanceFromHome_OverTime'] = df['DistanceFromHome'] * df.get('OverTime_Yes', 0)
    df['SalesExecutive_OverTime'] = df.get('JobRole_Sales Executive', 0) * df.get('OverTime_Yes', 0)
    df['Scientist_Young'] = df.get('JobRole_Research Scientist', 0) * (df['Age'] < 30).astype(int)
    df['TravelFreq_OverTime'] = df.get('BusinessTravel_Travel_Frequently', 0) * df.get('OverTime_Yes', 0)
    df['R&D_StockOptions'] = df.get('Department_Research & Development', 0) * df['StockOptionLevel']
    df['Income_per_JobLevel'] = df['MonthlyIncome'] / (df['JobLevel'] + 1)
    df['YearsWithCurrManager_per_YearsAtCompany'] = df['YearsWithCurrManager'] / (df['YearsAtCompany'] + 1)
    df['PromotionRate'] = df['YearsSinceLastPromotion'] / (df['YearsAtCompany'] + 1)
    return df


st.write("Using CSV schema from:", DF_PATH)

# --- UI: Single prediction inputs ---
st.header("Upload CSV for Prediction (replaces manual inputs)")

# Allow the user to upload a CSV with one or more employees; this replaces manual fields input
upload_file = st.file_uploader("Upload CSV for prediction (single or multiple rows)", type=["csv"], key="single_csv")

if upload_file is not None:
    try:
        uploaded_df = pd.read_csv(upload_file)
    except Exception as e:
        st.error(f"Failed to read uploaded CSV: {e}")
        uploaded_df = None

    if uploaded_df is not None:
        # Use your provided prediction flow: drop, engineered features, reorder, scale, predict
        drop_cols = ['Attrition', 'Over18', 'EmployeeCount', 'EmployeeNumber', 'StandardHours']
        uploaded_df = uploaded_df.drop(columns=[col for col in drop_cols if col in uploaded_df.columns], errors='ignore')

        # Add engineered features (function defined above)
        uploaded_df = add_engineered_features(uploaded_df)

        # Ensure the exact feature columns (add missing with 0)
        for col in feature_columns:
            if col not in uploaded_df.columns:
                uploaded_df[col] = 0

        # Arrange columns in the exact order expected by the model
        df_pred = uploaded_df[feature_columns]

        # Scale features if scaler present
        if scaler is not None:
            try:
                X_scaled = scaler.transform(df_pred)
            except Exception as e:
                st.warning(f"Scaler transform failed: {e} — proceeding without scaling")
                X_scaled = df_pred.values
        else:
            X_scaled = df_pred.values

        # Predict using the loaded model
        if model is None:
            st.error('Model is not loaded; cannot predict.')
        else:
            try:
                preds = model.predict(X_scaled)
                probs = model.predict_proba(X_scaled)[:, 1] if hasattr(model, 'predict_proba') else None

                out = uploaded_df.copy()
                out['Predicted_Attrition'] = preds
                if probs is not None:
                    out['Attrition_Probability'] = probs

                st.success(f"Prediction complete for {len(out)} rows. Showing top 5 rows:")
                show_cols = ['Predicted_Attrition'] + (['Attrition_Probability'] if probs is not None else [])
                st.dataframe(out[show_cols].head())

                csv_out = out.to_csv(index=False).encode('utf-8')
                st.download_button("Download prediction CSV", csv_out, "attrition_predictions.csv")
            except Exception as e:
                st.error(f"Model prediction failed: {e}")

