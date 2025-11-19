import json
import os
import firebase_admin
from firebase_admin import credentials, messaging


firebase_json = os.getenv("FIREBASE_CRED")
cred_dict = json.loads(firebase_json)

cred = credentials.Certificate(cred_dict)
firebase_admin.initialize_app(cred)

def send_anomaly_notification(device_id, current_value):
    message = messaging.Message(
        notification=messaging.Notification(
            title="⚠️ Electromonitor",
            body=f"Arus tidak normal pada {device_id}. Nilai arus: {current_value} A",
        ),
        android=messaging.AndroidConfig(
            priority="high",
            notification=messaging.AndroidNotification(
                channel_id="anomaly_alert",
                priority="high",
            )
        ),
        topic="electromoniton-alert",
    )

    response = messaging.send(message)
    print("Notification sent:", response)
