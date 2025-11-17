import os
import flask_cors
import joblib
from flask import Flask, request, jsonify
import numpy as np

import json
import time
import threading
from datetime import datetime
import paho.mqtt.client as mqtt
import firebase_admin
from firebase_admin import credentials, firestore

firebase_json = os.getenv("FIREBASE_CRED")
cred_dict = json.loads(firebase_json)

cred = credentials.Certificate(cred_dict)
firebase_admin.initialize_app(cred)

buffer = []

def save_batch():
    global buffer
    while True:
        time.sleep(60)  # save every 60 seconds

        if len(buffer) == 0:
            continue

        data_to_save = buffer.copy()
        buffer.clear()

        doc_ref = db.collection("sensors").document()
        doc_ref.set({
            "timestamp": datetime.now().isoformat(),
            "data": data_to_save
        })

        print(f"Saved batch: {len(data_to_save)} records")

threading.Thread(target=save_batch, daemon=True).start()

BROKER_URL = os.getenv("MQTT_BROKER")
USERNAME = os.getenv("MQTT_USERNAME")
PASSWORD = os.getenv("MQTT_PASSWORD")
TOPIC = "test/inbound"

def on_connect(client, userdata, flags, rc):
    print("Connected with code:", rc)
    client.subscribe(TOPIC)

def on_message(client, userdata, msg):
    try:
        payload = json.loads(msg.payload.decode())
        payload["received_at"] = datetime.now().isoformat()

        print("Received:", payload)
        buffer.append(payload)

    except Exception as e:
        print("Error:", e)

client = mqtt.Client()
client.username_pw_set(USERNAME, PASSWORD)
client.tls_set() 
client.on_connect = on_connect
client.on_message = on_message

print("Connecting to MQTT...")
client.connect(BROKER_URL, 8883)
client.loop_forever()

app = Flask(__name__)
flask_cors.CORS(app)


@app.route('/')
def home():
    return "Anomaly Detection Service is Running"

app = Flask(__name__)

model = joblib.load("isolation_forest_model.pkl")

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



