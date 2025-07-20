from flask import Flask, request, jsonify
from model_loader import load_model

app = Flask(__name__)
model = load_model()

@app.route("/")
def home():
    return "Welcome to the Wine Quality Prediction API!"


@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    features = [list(data.values())]  # assumes flat JSON with feature values
    prediction = model.predict(features)
    return jsonify({"prediction": prediction[0]})

if __name__ == "__main__":
    app.run(debug=True)
