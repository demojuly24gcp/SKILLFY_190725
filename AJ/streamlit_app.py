import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load the wine quality dataset
data = pd.read_csv(r'C:\Users\ankit_aj\Desktop\MLOPS-case_studies\Demo_050725_DVC\SKILLFY_190725\data\winequality-red.csv')

# Features and target
X = data.drop('quality', axis=1)
y = data['quality']

# Train the model
clf = RandomForestClassifier(random_state=42)
clf.fit(X, y)

# Save the model for reuse
joblib.dump(clf, 'wine_rf_model.pkl')

# Streamlit UI
st.title("Wine Quality Prediction")

st.write("Enter wine characteristics to predict quality:")

input_data = {}
for col in X.columns:
    input_data[col] = st.number_input(col, value=float(X[col].mean()))

if st.button("Predict Quality"):
    model = joblib.load('wine_rf_model.pkl')
    input_df = pd.DataFrame([input_data])
    prediction = model.predict(input_df)[0]
    st.success(f"Predicted Wine Quality: {prediction}")