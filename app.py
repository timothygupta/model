import streamlit as st
import joblib
import numpy as np

st.title("Simple ML Prediction App")

model = joblib.load("model.pkl")

val1 = st.number_input("Feature 1")
val2 = st.number_input("Feature 2")

if st.button("Predict"):
    input_data = np.array([[val1, val2]])
    pred = model.predict(input_data)
    st.write("Prediction:", pred[0])
