from flask import Flask, request, jsonify
import pickle
import numpy as np

model_filename = 'SKILLFY_190725\model_output\wine_rf_model.pkl'

with open(model_filename, 'rb') as model_file:
    model = pickle.load(model_file)

app = Flask(__name__)

@app.route('/')
def home():
    return "Welcome to the wine Prediction API! Use the /predict endpoint to make predictions."

@app.route('/predict', methods=['POST'])
def predict():

    data = request.get_json(force=True)
    

    features = np.array(data['features']).reshape(1, -1)
    

    prediction = model.predict(features)
    

    return jsonify({'prediction': int(prediction[0])})


if __name__ == '__main__':
    app.run(debug=True)
