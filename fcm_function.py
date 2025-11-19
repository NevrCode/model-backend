import json
import os
import firebase_admin
from firebase_admin import messaging

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
