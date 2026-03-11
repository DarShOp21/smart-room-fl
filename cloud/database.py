model_versions = []
trust_logs = []


def store_model(version, parameters):

    model_versions.append({
        "version": version,
        "parameters": parameters
    })


def store_trust(avg_trust):

    trust_logs.append(avg_trust)


def get_models():
    return model_versions


def get_trust():
    return trust_logs