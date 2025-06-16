import streamlit as st
import numpy as np
import pickle

st.set_page_config(page_title="Heart Disease Prediction", layout="centered")

st.title("Heart Disease Prediction App")
st.write("Fill the form below to predict the likelihood of heart disease:")

# Input fields
age = st.number_input("Age", min_value=1, max_value=120)
sex = st.selectbox("Sex", ["Male", "Female"])
cp = st.slider("Chest Pain Type (0–3)", 0, 3)
trestbps = st.number_input("Resting Blood Pressure", min_value=80, max_value=200)
chol = st.number_input("Cholesterol Level", min_value=100, max_value=600)
fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", ["No", "Yes"])
restecg = st.slider("Resting ECG results (0–2)", 0, 2)
thalach = st.number_input("Maximum Heart Rate Achieved", min_value=50, max_value=250)
exang = st.selectbox("Exercise Induced Angina", ["No", "Yes"])
oldpeak = st.number_input("ST Depression", min_value=0.0, max_value=6.0, step=0.1)
slope = st.slider("Slope of Peak Exercise ST Segment (0–2)", 0, 2)
ca = st.slider("Number of Major Vessels Colored by Fluoroscopy", 0, 4)
thal = st.slider("Thalassemia (1–3)", 1, 3)

# Encoding
sex = 1 if sex == "Male" else 0
fbs = 1 if fbs == "Yes" else 0
exang = 1 if exang == "Yes" else 0

# Load model
with open("heart_model.pkl", "rb") as f:
    model = pickle.load(f)

if st.button("Predict"):
    data = np.array([[age, sex, cp, trestbps, chol, fbs, restecg,
                      thalach, exang, oldpeak, slope, ca, thal]])
    pred = model.predict(data)[0]
    if pred == 1:
        st.error("⚠️ High likelihood of heart disease.")
    else:
        st.success("✅ Low likelihood of heart disease.")
