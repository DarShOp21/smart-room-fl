import torch
import torch.nn.functional as F


def compute_reconstruction_error(input_data, reconstructed_data):
    """
    Computes the reconstruction error between
    original input and model output.
    """

    error = F.mse_loss(
        reconstructed_data,
        input_data,
        reduction="mean"
    )

    return error.item()


def compute_trust_score(error, max_error=0.05):
    """
    Converts reconstruction error into
    a normalized trust score between 0 and 1.
    """

    trust = 1 - (error / max_error)

    trust = max(0.0, min(1.0, trust))

    return trust