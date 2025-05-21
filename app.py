import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt

st.set_page_config(page_title="PNDM Prediction App", layout="wide")
st.title("🔬 Permanent Neonatal Diabetes Mellitus (PNDM) Prediction App")

model = joblib.load("model.pkl")
explainer = joblib.load("shap_explainer.pkl")

def predict_pndm(input_data):
    prediction = model.predict(input_data)
    probability = model.predict_proba(input_data)[0][1]
    return prediction[0], probability

def display_shap_values(input_data):
    shap_values = explainer.shap_values(input_data)
    plt.title("Feature Importance (SHAP)")
    shap.summary_plot(shap_values, input_data, plot_type="bar", show=False)
    st.pyplot(bbox_inches='tight')

st.sidebar.header("Patient Information")
glucose = st.sidebar.slider("Glucose Level", 50, 200, 120)
insulin = st.sidebar.slider("Insulin Level", 0, 400, 80)
bmi = st.sidebar.slider("BMI", 10.0, 50.0, 25.0)
age = st.sidebar.slider("Age", 0, 100, 30)
skin_thickness = st.sidebar.slider("Skin Thickness", 0, 100, 20)
blood_pressure = st.sidebar.slider("Blood Pressure", 50, 150, 80)

input_data = pd.DataFrame({
    "Glucose": [glucose],
    "Insulin": [insulin],
    "BMI": [bmi],
    "Age": [age],
    "SkinThickness": [skin_thickness],
    "BloodPressure": [blood_pressure]
})

st.subheader("Input Data")
st.dataframe(input_data)

if st.button("Predict PNDM"):
    prediction, probability = predict_pndm(input_data)
    st.subheader("Prediction Result")
    if prediction == 1:
        st.error(f"🔴 High Risk of PNDM Detected (Probability: {probability:.2f})")
    else:
        st.success(f"🟢 Low Risk of PNDM Detected (Probability: {probability:.2f})")
    
    st.markdown("### Model Explanation with SHAP")
    display_shap_values(input_data)
