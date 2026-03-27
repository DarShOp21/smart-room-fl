import torch
import numpy as np
import requests
import os
from flask import Flask, request, jsonify
from flask_cors import CORS  # optional, if UI is on different origin

from fl_client.model.autoencoder import AutoEncoder
from fl_client.trust.trust_score import compute_reconstruction_error, compute_trust_score

app = Flask(__name__)
CORS(app)  # enable CORS if needed

MODEL_PATH = "model_weights.pt"
FOG_URL = "http://0.0.0.0:8000/explain"   # replace with actual Jetson IP
DEVICE = torch.device("cpu")

# Load model (if exists)
model = AutoEncoder().to(DEVICE)
if os.path.exists(MODEL_PATH):
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
else:
    print("Warning: No model found, using random weights.")

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    # Extract the 7 features
    features = [
        float(data["motion_rate"]),
        float(data["motion_variance"]),
        float(data["entry_time"]),
        float(data["entry_deviation"]),
        float(data["session_duration"]),
        float(data["device_usage"]),
        float(data["sequence_similarity"])
    ]
    x = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        reconstructed = model(x)
        error = compute_reconstruction_error(x, reconstructed)
        trust = compute_trust_score(error)

    # Per‑feature deviations (absolute difference)
    original = x.cpu().numpy().flatten()
    recon = reconstructed.cpu().numpy().flatten()
    deviations = np.abs(original - recon)
    feature_names = [
        "motion_rate", "motion_variance", "entry_time", "entry_deviation",
        "session_duration", "device_usage", "sequence_similarity"
    ]
    # Get top‑3 indices (largest deviations)
    top_indices = np.argsort(deviations)[-3:][::-1]
    top_deviations = [{"feature": feature_names[i], "deviation": float(deviations[i])}
                      for i in top_indices]

    # Forward to fog explanation service
    explanation = ""
    try:
        resp = requests.post(FOG_URL, json={
            "trust_score": trust,
            "deviations": top_deviations
        }, timeout=5)
        explanation = resp.json().get("explanation", "No explanation available.")
    except Exception as e:
        explanation = f"Explanation service unavailable: {e}"

    # Determine status based on trust (mirror server logic)
    if trust > 0.8:
        status = "NORMAL"
    elif trust > 0.5:
        status = "SLIGHT ANOMALY"
    elif trust > 0.2:
        status = "SUSPICIOUS"
    else:
        status = "HIGH RISK"

    return jsonify({
        "trust_score": trust,
        "status": status,
        "explanation": explanation
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)