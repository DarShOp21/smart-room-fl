import torch.nn.functional as F


def compute_reconstruction_error(input_data, reconstructed_data):

    error = F.mse_loss(
        reconstructed_data,
        input_data,
        reduction="mean"
    )

    return error.item()


def compute_trust_score(error, max_error=0.3):
    """
    Convert reconstruction error to trust score.
    """

    trust = 1 - (error / max_error)

    trust = max(0.0, min(1.0, trust))

    return trust