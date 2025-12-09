from collections import deque
import flask_cors
import joblib
from flask import Flask, request, jsonify
import numpy as np
from fcm_function import send_anomaly_notification
from dotenv import load_dotenv
import os
import json
import logging
import time
import threading
from datetime import datetime
import paho.mqtt.client as mqtt
import firebase_admin
from firebase_admin import credentials, firestore
from feature_engineering import get_features

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

load_dotenv()
app = Flask(__name__)
flask_cors.CORS(app)
model = joblib.load(os.getenv("MODEL_PATH"))
scaler = joblib.load(os.getenv("SCALER_PATH"))

firebase_json = os.getenv("FIREBASE_CRED")
cred_dict = json.loads(firebase_json)
# firebase_path = os.getenv("FIREBASE_CRED")

# with open(firebase_path, "r") as f:
#     cred_dict = json.load(f)


cred = credentials.Certificate(cred_dict)
firebase_admin.initialize_app(cred)
db = firestore.client()

BROKER_URL = os.getenv("MQTT_BROKER")
USERNAME = os.getenv("MQTT_USERNAME")
PASSWORD = os.getenv("MQTT_PASSWORD")
TOPIC = os.getenv("MQTT_TOPIC")
ALERT_TOPIC = os.getenv("ALERT_TOPIC")
WARNING_TRESHOLD = float(os.getenv("WARNING_THRESHOLD"))
THRESHOLD = float(os.getenv("ANOMALY_THRESHOLD"))
buffer = []
BUFFER_SIZE = 10
buffer_shuntV = deque(maxlen=BUFFER_SIZE)
buffer_current = deque(maxlen=BUFFER_SIZE)

    
anomaly_sent = False   
def prediction_loop():
    global anomaly_sent
    normal_counter = 0
    while True:
        time.sleep(1)

        if len(buffer_shuntV) == BUFFER_SIZE:
            X = get_features(list(buffer_shuntV), list(buffer_current))
            x_scaled = scaler.transform(X)
            score = -model.score_samples(x_scaled)[0]
            
            
            is_warning = score >= WARNING_TRESHOLD
            is_anomaly = score >= THRESHOLD
            if is_anomaly:
                normal_counter = 0
                if not anomaly_sent:    
                    logging.info(f"Anomaly detected with score: {score}")
                    client.publish(ALERT_TOPIC, "1")
                    send_anomaly_notification()
                    anomaly_sent = True
            else:
                normal_counter += 1
                if normal_counter >= 5:
                    logging.info("System back to normal.")
                    client.publish(ALERT_TOPIC, "0") 
                    anomaly_sent = False

def save_batch(data):
    doc_ref = db.collection("read_datas").document().set(data)

def on_connect(client, userdata, flags, rc):
    client.subscribe(TOPIC)

def on_message(client, userdata, msg):
    try:
        print("Message received on topic: " + msg.topic)
        payload = json.loads(msg.payload.decode())
        buffer_shuntV.append(payload['shuntV'])
        buffer_current.append(payload['current'])
        save_batch(data=payload)

    except Exception as e:
        print("Error:", e)
        

client = mqtt.Client()
client.username_pw_set(USERNAME, PASSWORD)
client.tls_set() 
client.on_connect = on_connect
client.on_message = on_message

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
        logging.info("Anomaly detected via /predict endpoint.")
    return jsonify({
        "prediction": int(pred)
    })

@app.route("/test-fcm", methods=["GET"])
def test_fcm():
    send_anomaly_notification()
    return jsonify({"status": "Notification sent"})
    

def start_background_services():
    threading.Thread(target=prediction_loop, daemon=True).start()
    client.loop_start()
    logging.info("Background services started.")
# if __name__ == '__main__':
    # threading.Thread(target=prediction_loop, daemon=True).start()
    # client.loop_start()
    # logging.info("Starting Flask app...")
    # app.run(host="0.0.0.0", port=5000, debug=False)

start_background_services()


