import flask_cors
import joblib
from flask import Flask, request, jsonify
import numpy as np
from fcm_function import send_anomaly_notification

import os
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
db = firestore.client()

buffer = []

def save_batch():
    global buffer
    while True:
        time.sleep(30)  

        if len(buffer) == 0:
            continue

        data_to_save = buffer.copy()
        buffer.clear()
        
        batch = db.batch()
        totalCounter = 0
        
        while totalCounter < len(data_to_save):
            batchCounter = 0
            while batchCounter < 500 and totalCounter < len(data_to_save):
                doc_ref = db.collection("read_datas").document()
                batch.set(doc_ref, data_to_save[totalCounter])
                batchCounter += 1
                totalCounter += 1
            batch.commit()
        
        

        print(f"Saved batch: {len(data_to_save)} records")

threading.Thread(target=save_batch, daemon=True).start()

BROKER_URL = os.getenv("MQTT_BROKER")
USERNAME = os.getenv("MQTT_USERNAME")
PASSWORD = os.getenv("MQTT_PASSWORD")
TOPIC = "get/data/sensors"

def on_connect(client, userdata, flags, rc):
    print("Connected with code:", rc)
    client.subscribe(TOPIC)

def on_message(client, userdata, msg):
    try:
        payload = json.loads(msg.payload.decode())
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
threading.Thread(target=client.loop_forever, daemon=True).start()


app = Flask(__name__)
flask_cors.CORS(app)

model = joblib.load("isolation_forest_model.pkl")

@app.route('/')
def home():
    return "Anomaly Detection Service is Running"

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    
    feature_order = [
    'busV_crest', 'busV_entropy', 'busV_kurt', 'busV_rms', 
    'current_crest','current_entropy', 'current_kurt', 'current_rms', 'current_skew', 'current_thd', 'current_zcr', 
    'power_crest', 'power_entropy','power_error_abs_mean', 'power_error_mean', 'power_kurt', 'power_rms','power_skew', 'power_thd', 
    'shuntV_crest', 'shuntV_entropy','shuntV_kurt', 'shuntV_rms', 'shuntV_skew', 'shuntV_thd', 'shuntV_zcr'
    ]
    missing = [f for f in feature_order if f not in data]
    if missing:
        return jsonify({"error": "Missing fields", "missing": missing}), 400

    features = np.array([[ data[f] for f in feature_order ]], dtype=float)

    pred = model.predict(features)[0] 
    if pred == 1:
        send_anomaly_notification("device_123", data['current_rms'])
    return jsonify({
        "prediction": int(pred)
    })

@app.route("/test-fcm", methods=["GET"])
def test_fcm():
    send_anomaly_notification("device_123", 15.7)
    
if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=True)



