import streamlit as st
import numpy as np
import pickle

# ---------------------------
# Load trained model & scaler
# ---------------------------
model = pickle.load(open("diabetes_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

# ---------------------------
# App Title
# ---------------------------
st.set_page_config(page_title="Diabetes Prediction System")
st.title("Diabetes Prediction System")
st.write("Enter patient details below to predict diabetes.")

# ---------------------------
# User Input Section
# ---------------------------
preg = st.number_input("Pregnancies", min_value=0)
glucose = st.number_input("Glucose Level", min_value=0)
bp = st.number_input("Blood Pressure", min_value=0)
skin = st.number_input("Skin Thickness", min_value=0)
insulin = st.number_input("Insulin Level", min_value=0)
bmi = st.number_input("BMI", min_value=0.0)
dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0)
age = st.number_input("Age", min_value=0)

# ---------------------------
# Predict Button
# ---------------------------
if st.button("Predict"):

    # Create feature engineering (same as training)
    bmi_age = bmi * age
    glucose_bmi = glucose * bmi
    age_squared = age ** 2

    # Arrange input in correct order
    input_data = np.array([[
        preg, glucose, bp, skin, insulin,
        bmi, dpf, age,
        bmi_age, glucose_bmi, age_squared
    ]])

    # Scale input
    input_scaled = scaler.transform(input_data)

    # Make prediction
    prediction = model.predict(input_scaled)

    # Display result
    if prediction[0] == 1:
        st.error("The patient is likely Diabetic.")
    else:
        st.success("The patient is likely Not Diabetic.")
