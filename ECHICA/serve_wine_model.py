import os
import pandas as pd
import joblib
import streamlit as st
from sklearn.preprocessing import PolynomialFeatures

# Load model and feature names
model_path = os.path.join(os.path.dirname(__file__), "trained_pipeline.pkl")
data_path = os.path.join(os.path.dirname(__file__), "../data/winequality-red.csv")
if not os.path.exists(model_path):
    st.error("Trained model not found. Please train and save the model first.")
else:
    pipeline = joblib.load(model_path)
    feature_names = list(pd.read_csv(data_path).drop('quality', axis=1).columns)
    st.title("Wine Quality Prediction")
    st.write("Enter the wine characteristics below:")
    user_input = []
    for feature in feature_names:
        val = st.number_input(f"{feature}", value=0.0, format="%f")
        user_input.append(val)
    # Polynomial features (degree 2, to match your training)
    poly = PolynomialFeatures(degree=2, include_bias=False)
    X_user = poly.fit_transform([user_input])
    if st.button("Predict Quality"):
        pred = pipeline.predict(X_user)
        st.success(f"Predicted Wine Quality: {pred[0]}") 