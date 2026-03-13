import requests
import numpy as np
from fl_server.config import CLOUD_URL


def upload_model(parameters):

    # convert numpy arrays → python lists
    parameters_list = [p.tolist() for p in parameters]

    data = {
        "parameters": parameters_list
    }

    try:
        r = requests.post(
            f"{CLOUD_URL}/upload_model",
            json=data
        )

        print("Model uploaded to cloud:", r.status_code)

    except Exception as e:
        print("Cloud upload failed:", e)


def upload_trust(avg_trust):

    data = {
        "avg_trust": float(avg_trust)
    }

    try:
        r = requests.post(
            f"{CLOUD_URL}/upload_trust",
            json=data
        )

        print("Trust uploaded:", r.status_code)

    except Exception as e:
        print("Trust upload failed:", e)