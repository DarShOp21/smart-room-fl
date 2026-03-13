import numpy as np

def compute_reconstruction_error(x, reconstructed):
    error = np.mean((x.cpu().numpy() - reconstructed.cpu().numpy()) ** 2)
    return float(error)

def compute_trust_score(error, normal_error_threshold=0.15):
    """
    Map reconstruction error to trust score in [0,1].
    error <= threshold -> trust = 1.0
    error > threshold -> trust decays exponentially.
    """
    if error <= normal_error_threshold:
        return 1.0
    else:
        # Decay: trust = exp(- (error - threshold) / scale)
        scale = 0.2  # adjust based on expected error range
        return float(np.exp(-(error - normal_error_threshold) / scale))