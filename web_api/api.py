from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import torch
import numpy as np

from fl_client.model.autoencoder import AutoEncoder
from fl_client.trust.trust_score import (
    compute_reconstruction_error,
    compute_trust_score
)

app = FastAPI()

# CORS CONFIGURATION
origins = [
    "*"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DEVICE = torch.device("cpu")

model = AutoEncoder().to(DEVICE)
model.eval()


@app.get("/")
def root():
    return {"message": "Smart Room Prediction API"}


@app.post("/predict")
def predict(data: dict):

    features = [
        float(data["motion_rate"]),
        float(data["motion_variance"]),
        float(data["entry_time"]),
        float(data["entry_deviation"]),
        float(data["session_duration"]),
        float(data["device_usage"]),
        float(data["sequence_similarity"]),
    ]

    x = torch.tensor(
        np.array(features),
        dtype=torch.float32
    ).unsqueeze(0)

    with torch.no_grad():

        output = model(x)

        error = compute_reconstruction_error(x, output)

        trust = compute_trust_score(error)

    if trust > 0.8:
        status = "NORMAL"
    elif trust > 0.5:
        status = "SLIGHT ANOMALY"
    elif trust > 0.2:
        status = "SUSPICIOUS"
    else:
        status = "HIGH RISK"

    return {
        "reconstruction_error": float(error),
        "trust_score": float(trust),
        "status": status
    }