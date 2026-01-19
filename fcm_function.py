import logging
from firebase_admin import messaging

def send_anomaly_notification():
    message = messaging.Message(
        notification=messaging.Notification(
            title="Electromonitor",
            body=f"Arus tidak normal pada instalasi listrik anda. cek segera!",
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
    logging.info(f"Notification Sent: {response}")
