import streamlit as st
import joblib
import numpy as np
import pandas as pd

st.title("Simple ML Prediction App")

# Load model
model = joblib.load("model.pkl")

# Load the preprocessed CSV to know the expected input fields and to compute realistic defaults
DF_PATH = "EmployeeData_preprocessed.csv"
df = pd.read_csv(DF_PATH)

st.write("Using CSV schema from:", DF_PATH)

# Build input widgets dynamically in the same order as CSV columns
inputs = []
st.header("Input features (prefilled with realistic defaults)")
for col in df.columns:
    series = df[col]
    # Numeric columns
    if pd.api.types.is_numeric_dtype(series):
        uniq = np.unique(series.dropna())
        # Treat binary/dummy columns specially
        if set(uniq).issubset({0, 1}):
            # choose default opposite of the mode to vary from common rows
            mode_val = int(series.mode().iat[0]) if not series.mode().empty else 0
            default_val = 1 - mode_val
            # present as selectbox for clarity
            idx = 0 if default_val == 0 else 1
            val = st.selectbox(col, options=[0, 1], index=idx, key=f"inp_{col}")
            inputs.append(val)
        else:
            # continuous or integer feature: use median and shift slightly so it's not identical to CSV values
            med = float(series.median()) if not series.dropna().empty else 0.0
            # small shift to be "slightly different" from values in CSV
            shift = 0.13 if series.dtype.kind in "f" else 0.5
            default = med + shift
            min_v = float(series.min())
            max_v = float(series.max())
            # clamp default into range
            if default < min_v:
                default = min_v
            if default > max_v:
                default = max_v
            val = st.number_input(col, value=float(round(default, 2)), min_value=min_v, max_value=max_v, step=0.1, format="%.2f", key=f"inp_{col}")
            inputs.append(val)
    else:
        # Non-numeric columns (e.g., Over18 with 'Y') - present choices from unique values
        uniques = series.dropna().unique().tolist()
        if len(uniques) == 0:
            # fallback to text input
            v = st.text_input(col, value="", key=f"inp_{col}")
            inputs.append(v)
        else:
            # pick a default slightly different from the most common value when possible
            mode = series.mode().iat[0] if not series.mode().empty else uniques[0]
            default_choice = None
            for u in uniques:
                if u != mode:
                    default_choice = u
                    break
            if default_choice is None:
                default_choice = uniques[0]
            try:
                default_index = uniques.index(default_choice)
            except ValueError:
                default_index = 0
            val = st.selectbox(col, options=uniques, index=default_index, key=f"inp_{col}")
            inputs.append(val)

# When predicting, convert inputs to numeric array matching the CSV column order
if st.button("Predict"):
    row = []
    for v in inputs:
        # Convert common text flags to numeric where appropriate
        if isinstance(v, str):
            if v.upper() == "Y":
                row.append(1.0)
            elif v.upper() == "N":
                row.append(0.0)
            else:
                # try cast to float, fallback to 0.0
                try:
                    row.append(float(v))
                except Exception:
                    row.append(0.0)
        else:
            try:
                row.append(float(v))
            except Exception:
                row.append(0.0)

    input_data = np.array([row], dtype=float)
    try:
        pred = model.predict(input_data)
        st.success(f"Prediction: {pred[0]}")
    except Exception as e:
        st.error(f"Model prediction failed: {e}")
