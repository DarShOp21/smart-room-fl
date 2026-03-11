import numpy as np


def generate_normal_session():
    """
    Simulates normal smart-room behavior
    """

    avg_motion_rate = np.random.normal(0.6, 0.05)
    motion_variance = np.random.normal(0.2, 0.05)

    entry_time_hour = np.random.normal(0.4, 0.02)  # around morning
    entry_time_deviation = np.random.normal(0.1, 0.05)

    session_duration = np.random.normal(0.6, 0.1)
    num_device_activations = np.random.normal(0.5, 0.1)

    device_sequence_similarity = np.random.normal(0.8, 0.05)

    return np.clip([
        avg_motion_rate,
        motion_variance,
        entry_time_hour,
        entry_time_deviation,
        session_duration,
        num_device_activations,
        device_sequence_similarity
    ], 0, 1)


def generate_suspicious_session():
    """
    Simulates unusual / suspicious behavior
    """

    avg_motion_rate = np.random.normal(0.3, 0.2)
    motion_variance = np.random.normal(0.6, 0.2)

    entry_time_hour = np.random.uniform(0, 1)
    entry_time_deviation = np.random.normal(0.6, 0.2)

    session_duration = np.random.normal(0.2, 0.2)
    num_device_activations = np.random.normal(0.2, 0.2)

    device_sequence_similarity = np.random.normal(0.3, 0.2)

    return np.clip([
        avg_motion_rate,
        motion_variance,
        entry_time_hour,
        entry_time_deviation,
        session_duration,
        num_device_activations,
        device_sequence_similarity
    ], 0, 1)


def generate_dataset(num_samples=50, suspicious_ratio=0.2):
    """
    Creates a dataset mixing normal and suspicious sessions
    """

    data = []

    for _ in range(num_samples):
        if np.random.rand() < suspicious_ratio:
            data.append(generate_suspicious_session())
        else:
            data.append(generate_normal_session())

    return np.array(data, dtype=np.float32)
