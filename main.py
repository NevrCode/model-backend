from collections import deque
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
from feature_extractions import extract_features

app = Flask(__name__)
flask_cors.CORS(app)
model = joblib.load("isolation_forest_model_20_feature.pkl")
scaler = joblib.load("scaler.pkl")

firebase_json = os.getenv("FIREBASE_CRED")
cred_dict = json.loads(firebase_json)

cred = credentials.Certificate(cred_dict)
firebase_admin.initialize_app(cred)
db = firestore.client()

BROKER_URL = os.getenv("MQTT_BROKER")
USERNAME = os.getenv("MQTT_USERNAME")
PASSWORD = os.getenv("MQTT_PASSWORD")
TOPIC = "get/data/sensors"

buffer = []
BUFFER_SIZE = 10
buffer_shuntV = deque(maxlen=BUFFER_SIZE)
buffer_busV = deque(maxlen=BUFFER_SIZE)
buffer_current = deque(maxlen=BUFFER_SIZE)

anomaly_sent = False   
def prediction_loop():
    global anomaly_sent
    while True:
        time.sleep(1)

        if len(buffer_shuntV) == BUFFER_SIZE:
            X = extract_features(list(buffer_shuntV), list(buffer_current))
            x_scaled = scaler.transform(X)
            prediction = model.predict(x_scaled)[0]

            normal_counter = 0
            if prediction == -1:
                normal_counter = 0
                if not anomaly_sent:    # Hanya kirim sekali
                    send_anomaly_notification()
                    anomaly_sent = True
                    print("⚠️ Notif dikirim")
            else:
                normal_counter += 1
                if normal_counter >= 3:
                    anomaly_sent = False
                if anomaly_sent:
                    print("System kembali normal, reset flag")
                anomaly_sent = False

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

def on_connect(client, userdata, flags, rc):
    print("Connected with code:", rc)
    client.subscribe(TOPIC)

def on_message(client, userdata, msg):
    try:
        payload = json.loads(msg.payload.decode())
        buffer_shuntV.append(payload['shuntV'])
        buffer_busV.append(payload['busV'])
        buffer_current.append(payload['current'])
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
    send_anomaly_notification()
    return jsonify({"status": "Notification sent"})
    
    
if __name__ == '__main__':
    threading.Thread(target=prediction_loop, daemon=True).start()
    threading.Thread(target=save_batch, daemon=True).start()
    threading.Thread(target=client.loop_forever, daemon=True).start()
    app.run(host="0.0.0.0", port=5000, debug=True)



