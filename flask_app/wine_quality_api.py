from flask import Flask, request, jsonify
import pickle
import numpy as np
import os

# Load the trained model
model_path = os.path.join(os.path.dirname(__file__), '../model_output/wine_rf_model.pkl')
with open(model_path, 'rb') as f:
    model = pickle.load(f)

app = Flask(__name__)

@app.route('/')
def home():
    return "Wine Quality Prediction API. Use POST /predict with wine features."

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    features = np.array(data['features']).reshape(1, -1)
    prediction = model.predict(features)
    return jsonify({'prediction': int(prediction[0])})

if __name__ == '__main__':
    app.run(debug=True)
