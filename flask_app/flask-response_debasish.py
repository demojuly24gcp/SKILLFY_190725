import requests
import json

# URL of the Flask API
url = 'http://127.0.0.1:5000/predict'

# Sample input data
data = {
    "features":[7.8,0.88,0.0,2.6,0.098,25.0,67.0,0.9968,3.2,0.68,9.8]
}

# Send POST request
response = requests.post(url, json=data)

# Print response
if response.status_code == 200:
    print('Prediction:', response.json())
else:
    print(f'Error: {response.status_code}, {response.text}')