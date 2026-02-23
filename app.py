import streamlit as st
import numpy as np
import pickle

# Load model and scaler
model = pickle.load(open("models/model.pkl", "rb"))
scaler = pickle.load(open("models/scaler.pkl", "rb"))

st.title("🫁 Lung Cancer Prediction App")

st.write("Enter patient details:")


# Inputs
gender = st.selectbox("Gender", ["Male", "Female"])
age = st.number_input("Age", 1, 120, 30)
smoking = st.selectbox("Smoking", ["Yes", "No"])
yellow_fingers = st.selectbox("Yellow Fingers", ["Yes", "No"])
anxiety = st.selectbox("Anxiety", ["Yes", "No"])
peer_pressure = st.selectbox("Peer Pressure", ["Yes", "No"])
chronic_disease = st.selectbox("Chronic Disease", ["Yes", "No"])
fatigue = st.selectbox("Fatigue", ["Yes", "No"])
allergy = st.selectbox("Allergy", ["Yes", "No"])
wheezing = st.selectbox("Wheezing", ["Yes", "No"])
alcohol = st.selectbox("Alcohol Consuming", ["Yes", "No"])
coughing = st.selectbox("Coughing", ["Yes", "No"])
short_breath = st.selectbox("Shortness of Breath", ["Yes", "No"])
swallowing = st.selectbox("Swallowing Difficulty", ["Yes", "No"])
chest_pain = st.selectbox("Chest Pain", ["Yes", "No"])


# Encoding function
def encode(val):
    return 1 if val == "Yes" else 0

# Convert to numeric
gender = 1 if gender == "Male" else 0

if st.button("Predict"):
    features = np.array([[
        gender,
        age,
        encode(smoking),
        encode(yellow_fingers),
        encode(anxiety),
        encode(peer_pressure),
        encode(chronic_disease),
        encode(fatigue),
        encode(allergy),
        encode(wheezing),
        encode(alcohol),
        encode(coughing),
        encode(short_breath),
        encode(swallowing),
        encode(chest_pain)
    ]])

# SCALE before prediction

    features_scaled = scaler.transform(features)

    proba = model.predict_proba(features_scaled)[0][1]

    st.write(f"Risk Probability: {proba:.2f}")

    if proba >= 0.5:
        st.error("High Risk of Lung Cancer")
    else:
        st.success("Low Risk of Lung Cancer")

    st.progress(float(proba))