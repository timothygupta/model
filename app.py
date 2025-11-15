import os
import streamlit as st
import joblib
import numpy as np
import pandas as pd
import pickle

st.title("Employee Attrition Prediction (Single + Batch)")

# Try to load the trained model (supports joblib or pickle files)
MODEL_PATH = "model.pkl"
model = None
if os.path.exists(MODEL_PATH):
    try:
        model = joblib.load(MODEL_PATH)
    except Exception:
        try:
            with open(MODEL_PATH, "rb") as f:
                model = pickle.load(f)
        except Exception as e:
            st.error(f"Failed to load model from {MODEL_PATH}: {e}")
else:
    st.error(f"Model file not found at {MODEL_PATH}")

# Optionally load a scaler if present (used at training time)
scaler = None
for candidate in ["standard_scaler.pkl", "standard_scaler.joblib", "scaler.pkl", "scaler.joblib", "standard_scaler"]:
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
st.header("Single Employee Prediction")
user_input = {}

# numeric inputs: use CSV medians as defaults and shift slightly
for f in num_inputs:
    default = 1
    if f in df.columns and not df[f].dropna().empty:
        med = df[f].median()
        # choose a default slightly different from the median
        default = int(med) + (1 if med >= 0 else 0)
    # step and format heuristic
    step = 1
    user_input[f] = st.number_input(f, value=default, step=step, key=f"inp_{f}")

# categorical inputs: choose options and default slightly different from mode when possible
for f, options in cat_inputs.items():
    default_idx = 0
    if f in df.columns:
        mode = df[f].mode().iat[0] if not df[f].mode().empty else None
        # pick first option different from mode
        for i, opt in enumerate(options):
            if str(opt) != str(mode):
                default_idx = i
                break
    user_input[f] = st.selectbox(f, options, index=default_idx, key=f"inp_{f}")


# Prepare a row dict and one-hot encode as in training
row = dict(user_input)
for col in onehot_cols:
    row[col] = 0

# Map selected user inputs to one-hot columns exactly like training code
if user_input.get('BusinessTravel') and user_input['BusinessTravel'] != 'Non-Travel':
    row[f"BusinessTravel_{user_input['BusinessTravel']}"] = 1
if user_input.get('Department') and user_input['Department'] != 'Human Resources':
    row[f"Department_{user_input['Department']}"] = 1
if user_input.get('EducationField') and user_input['EducationField'] != 'Human Resources':
    row[f"EducationField_{user_input['EducationField']}"] = 1
if user_input.get('Gender') and user_input['Gender'] == 'Male':
    row['Gender_Male'] = 1
if user_input.get('JobRole') and user_input['JobRole'] != 'Manager':
    row[f"JobRole_{user_input['JobRole']}"] = 1
if user_input.get('MaritalStatus') and user_input['MaritalStatus'] != 'Divorced':
    row[f"MaritalStatus_{user_input['MaritalStatus']}"] = 1
if user_input.get('OverTime') and user_input['OverTime'] == 'Yes':
    row['OverTime_Yes'] = 1

# Ensure numeric types for numeric inputs
for f in num_inputs:
    row[f] = float(row.get(f, 0))

# Ensure all feature columns are present (fill missing with 0)
for col in feature_columns:
    if col not in row:
        row[col] = 0

# Construct DataFrame, add engineered features, reorder to match training
single_df = pd.DataFrame([row])
single_df = add_engineered_features(single_df)
for col in feature_columns:
    if col not in single_df.columns:
        single_df[col] = 0
single_df = single_df[feature_columns]

if st.button('Predict for This Employee'):
    if model is None:
        st.error('Model is not loaded; cannot predict.')
    else:
        X = single_df.values
        # If a scaler was available, apply it (training used scaling)
        if scaler is not None:
            try:
                X = scaler.transform(single_df)
            except Exception as e:
                st.warning(f"Scaler transform failed: {e} — proceeding without scaling")
                X = single_df.values

        # Validate shape matches model expectation where possible
        try:
            pred = model.predict(X)[0]
            prob = None
            if hasattr(model, 'predict_proba'):
                try:
                    prob = model.predict_proba(X)[0][1]
                except Exception:
                    prob = None
            st.subheader("Result")
            st.write(f"*Predicted Attrition:* {'Yes' if pred == 1 else 'No'}")
            if prob is not None:
                st.write(f"*Probability of Attrition:* {prob:.2%}")
        except Exception as e:
            st.error(f"Model prediction failed: {e}")

