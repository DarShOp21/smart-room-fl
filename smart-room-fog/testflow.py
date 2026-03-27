import requests
import json

# Test fog service directly
print("Testing Fog Service...")
fog_response = requests.post(
    "http://localhost:8001/explain",
    json={
        "trust_score": 0.35,
        "deviations": [
            {"feature": "motion_rate", "deviation": 0.25},
            {"feature": "entry_time", "deviation": 0.18}
        ]
    }
)
print("Fog Service Response:")
print(json.dumps(fog_response.json(), indent=2))
print()

# Test edge API
print("Testing Edge API...")
edge_response = requests.post(
    "http://192.168.2.255:5000/predict",
    json={
        "motion_rate": 0.7,
        "motion_variance": 0.3,
        "entry_time": 0.5,
        "entry_deviation": 0.2,
        "session_duration": 0.4,
        "device_usage": 0.6,
        "sequence_similarity": 0.5
    }
)
print("Edge API Response:")
print(json.dumps(edge_response.json(), indent=2))