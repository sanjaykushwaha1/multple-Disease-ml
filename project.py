import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle

# Load and train models only once
@st.cache_resource
def train_models():
    # DIABETES
    diabetes = pd.read_csv('diabetes.csv')
    X_d = diabetes.drop('Outcome', axis=1)
    y_d = diabetes['Outcome']
    X_d_train, X_d_test, y_d_train, y_d_test = train_test_split(X_d, y_d, test_size=0.2, random_state=42)
    model_diabetes = RandomForestClassifier()
    model_diabetes.fit(X_d_train, y_d_train)

    # HEART
    heart = pd.read_csv('heart_disease_data.csv')
    X_h = heart.drop('target', axis=1)
    y_h = heart['target']
    X_h_train, X_h_test, y_h_train, y_h_test = train_test_split(X_h, y_h, test_size=0.2, random_state=42)
    model_heart = RandomForestClassifier()
    model_heart.fit(X_h_train, y_h_train)

    # PARKINSON’S
    parkinson = pd.read_csv('Parkinsson disease.csv')
    parkinson = parkinson.drop(['name'], axis=1)
    X_p = parkinson.drop(['status'], axis=1)
    y_p = parkinson['status']
    X_p_train, X_p_test, y_p_train, y_p_test = train_test_split(X_p, y_p, test_size=0.2, random_state=42)
    model_parkinson = RandomForestClassifier()
    model_parkinson.fit(X_p_train, y_p_train)

    return model_diabetes, model_heart, model_parkinson, X_d.columns, X_h.columns, X_p.columns

# Load models
model_diabetes, model_heart, model_parkinson, diabetes_cols, heart_cols, parkinson_cols = train_models()

# Streamlit UI
st.title("🧠🫀🩺 Multi-Disease Detection System")
st.sidebar.title("Choose Disease to Predict")

disease = st.sidebar.radio("Select", ["Diabetes", "Heart Disease", "Parkinson's"])

def input_features(features):
    user_data = {}
    for feat in features:
        val = st.number_input(f"Enter value for {feat}", value=0.0)
        user_data[feat] = val
    return pd.DataFrame([user_data])

if disease == "Diabetes":
    st.subheader("Diabetes Prediction")
    input_df = input_features(diabetes_cols)
    if st.button("Predict Diabetes"):
        pred = model_diabetes.predict(input_df)[0]
        st.success("Diabetic" if pred == 1 else "Not Diabetic")

elif disease == "Heart Disease":
    st.subheader("Heart Disease Prediction")
    input_df = input_features(heart_cols)
    if st.button("Predict Heart Disease"):
        pred = model_heart.predict(input_df)[0]
        st.success("Heart Disease Detected" if pred == 1 else "Healthy Heart")

elif disease == "Parkinson's":
    st.subheader("Parkinson's Disease Prediction")
    input_df = input_features(parkinson_cols)
    if st.button("Predict Parkinson's"):
        pred = model_parkinson.predict(input_df)[0]
        st.success("Parkinson's Detected" if pred == 1 else "No Parkinson's")