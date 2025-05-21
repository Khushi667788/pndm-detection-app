# pndm-detection-app
A Machine Learning-powered web application to predict **Permanent Neonatal Diabetes Mellitus (PNDM)** using clinical features. Built with **Streamlit**, this app helps visualize model predictions and understand feature contributions through **SHAP values**.
# PNDM Detection App 🧬

A Machine Learning-powered web application to predict **Permanent Neonatal Diabetes Mellitus (PNDM)** using clinical features. Built with **Streamlit**, this app helps visualize model predictions and understand feature contributions through **SHAP values**.

## 🔍 Project Overview

- **Goal**: Early detection of PNDM for better clinical outcomes.
- **Tech Stack**: Python, Streamlit, Scikit-learn, SHAP, Pandas
- **ML Model**: Logistic Regression / Random Forest (based on performance)
- **Visualization**: SHAP Explainer
- **Deployment**: Streamlit Cloud / Hugging Face Spaces

## 🛠️ How It Works

- Input patient values (glucose, insulin, BMI, age, etc.)
- Model predicts whether PNDM is likely
- SHAP values explain model decision-making

## 📁 File Structure

pndm-detection-app/
│
├── app.py # Streamlit web app
├── model.pkl # Trained ML model
├── requirements.txt # Python dependencies
├── data/
│ └── pndm_dataset.csv # Clinical dataset
├── notebooks/
│ └── model_training.ipynb # Jupyter notebook for training
├── images/
│ └── shap_summary.png # SHAP visualization
└── README.md # Project documentation
