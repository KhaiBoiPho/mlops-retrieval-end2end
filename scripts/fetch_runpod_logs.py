"""
Fetch RunPod logs and push to Loki
Run this as a cron job or in Docker
"""
import os
import time
import requests
# from datetime import datetime

RUNPOD_API_KEY = os.getenv('RUNPOD_API_KEY')
BI_ENCODER_ENDPOINT_ID = os.getenv('BI_ENCODER_ENDPOINT_ID')
LOKI_URL = os.getenv('LOKI_URL', 'http://localhost:3100')

def fetch_logs():
    """Fetch logs from RunPod"""
    # This is pseudo-code - RunPod API for logs varies
    # Check RunPod docs for actual log API
    url = f"https://api.runpod.ai/v2/{BI_ENCODER_ENDPOINT_ID}/logs"
    headers = {"Authorization": f"Bearer {RUNPOD_API_KEY}"}
    
    response = requests.get(url, headers=headers)
    return response.json()

def push_to_loki(logs):
    """Push logs to Loki"""
    url = f"{LOKI_URL}/loki/api/v1/push"
    
    streams = []
    for log in logs:
        streams.append({
            "stream": {
                "job": "runpod",
                "endpoint": BI_ENCODER_ENDPOINT_ID,
                "level": log.get('level', 'info')
            },
            "values": [
                [str(int(time.time() * 1e9)), log.get('message', '')]
            ]
        })
    
    payload = {"streams": streams}
    requests.post(url, json=payload)

if __name__ == "__main__":
    while True:
        logs = fetch_logs()
        push_to_loki(logs)
        time.sleep(60)  # Every minute