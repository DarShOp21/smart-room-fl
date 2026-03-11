from fastapi import FastAPI
from cloud.database import store_model, store_trust, get_models, get_trust

app = FastAPI()

model_version = 0


@app.get("/")
def root():
    return {"message": "Smart Room Federated Cloud Server"}


@app.post("/upload_model")
def upload_model(data: dict):

    global model_version

    parameters = data["parameters"]

    store_model(model_version, parameters)

    model_version += 1

    return {"status": "model stored", "version": model_version}


@app.post("/upload_trust")
def upload_trust(data: dict):

    trust = data["avg_trust"]

    store_trust(trust)

    return {"status": "trust stored"}


@app.get("/models")
def models():

    return get_models()


@app.get("/trust")
def trust():

    return get_trust()