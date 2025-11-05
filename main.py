
import flask_cors
import joblib
from flask import Flask, request, jsonify
import numpy as np
model = joblib.load("model/isolation_forest_model.pkl")

app = Flask(__name__)
flask_cors.CORS(app)


@app.route('/')
def home():
    return "Anomaly Detection Service is Running"

app = Flask(__name__)

model = joblib.load("model/isolation_forest_model.pkl")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    
    # Convert JSON to feature array
    features = np.array([[
        data["mean"],
        data["std"],
        data["peak2peak"],
        data["crest_factor"],
        data["skew"],
        data["kurt"]
    ]], dtype=float)
    
    # Predict
    pred = model.predict(features)[0]   # 1 = normal, -1 = anomaly

    return jsonify({
        "prediction": 1 if pred == 1 else -1
    })

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000)



