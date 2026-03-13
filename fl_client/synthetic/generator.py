import numpy as np
import csv
import os


def generate_normal_session():
    """
    Simulates normal smart-room behavior
    """

    avg_motion_rate = np.random.normal(0.6, 0.05)
    motion_variance = np.random.normal(0.2, 0.05)

    entry_time_hour = np.random.normal(0.4, 0.02)
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


def generate_dataset(num_samples=2000, suspicious_ratio=0.3, save_csv=True):
    """
    Creates a dataset mixing normal and suspicious sessions
    and optionally saves it as a CSV file
    """

    data = []

    for _ in range(num_samples):
        if np.random.rand() < suspicious_ratio:
            data.append(generate_suspicious_session())
        else:
            data.append(generate_normal_session())

    dataset = np.array(data, dtype=np.float32)

    if save_csv:
        save_dataset_to_csv(dataset)

    return dataset


def save_dataset_to_csv(dataset, filename="smart_room_dataset.csv"):

    headers = [
        "motion_rate",
        "motion_variance",
        "entry_time",
        "entry_deviation",
        "session_duration",
        "device_usage",
        "sequence_similarity"
    ]

    os.makedirs("data", exist_ok=True)

    filepath = os.path.join("data", filename)

    with open(filepath, "w", newline="") as f:
        writer = csv.writer(f)

        writer.writerow(headers)
        writer.writerows(dataset)

    print(f"Dataset saved to {filepath}")