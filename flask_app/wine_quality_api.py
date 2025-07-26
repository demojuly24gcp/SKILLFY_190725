from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np
import os
from pyngrok import ngrok

# Load the trained model
model_path = os.path.join(os.path.dirname(__file__), '../model_output/wine_rf_model.pkl')
with open(model_path, 'rb') as f:
    model = pickle.load(f)

app = Flask(__name__)

# Wine feature names and their ranges (based on actual data ranges)
FEATURES = [
    {"name": "fixed_acidity", "min": 4.6, "max": 15.9, "step": 0.1},
    {"name": "volatile_acidity", "min": 0.12, "max": 1.58, "step": 0.01},
    {"name": "citric_acid", "min": 0.0, "max": 1.0, "step": 0.01},
    {"name": "residual_sugar", "min": 0.9, "max": 15.5, "step": 0.1},
    {"name": "chlorides", "min": 0.012, "max": 0.611, "step": 0.001},
    {"name": "free_sulfur_dioxide", "min": 1.0, "max": 72.0, "step": 1},
    {"name": "total_sulfur_dioxide", "min": 6.0, "max": 289.0, "step": 1},
    {"name": "density", "min": 0.990, "max": 1.004, "step": 0.0001},
    {"name": "pH", "min": 2.74, "max": 4.01, "step": 0.01},
    {"name": "sulphates", "min": 0.33, "max": 2.0, "step": 0.01},
    {"name": "alcohol", "min": 8.4, "max": 14.9, "step": 0.1}
]

@app.route('/')
def home():
    return render_template('index.html', features=FEATURES)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    features = np.array(data['features']).reshape(1, -1)
    prediction = model.predict(features)
    return jsonify({'prediction': int(prediction[0])})

if __name__ == '__main__':
    try:
        # Start ngrok
        port = 3000
        print(f"Starting ngrok tunnel to port {port}...")
        # Kill any existing ngrok processes before starting new one
        os.system("taskkill /im ngrok.exe /f")
        public_url = ngrok.connect(port).public_url
        print(f" * ngrok tunnel \"{public_url}\" -> \"http://127.0.0.1:{port}\"")
        
        # Start Flask
        print(f"Starting Flask server on port {port}...")
        app.run(host='127.0.0.1', port=port, debug=True)
    except Exception as e:
        print(f"Error starting server: {e}")
