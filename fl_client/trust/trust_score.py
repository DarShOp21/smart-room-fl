import numpy as np


def compute_reconstruction_error(x, reconstructed):

    error = np.mean((x.cpu().numpy() - reconstructed.cpu().numpy()) ** 2)

    return float(error)


def compute_trust_score(error):

    # expected error range
    max_expected_error = 0.5

    trust = 1 - (error / max_expected_error)

    trust = max(0.0, min(1.0, trust))

    return float(trust)