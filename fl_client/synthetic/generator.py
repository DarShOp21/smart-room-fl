import numpy as np
import csv
import os
from typing import List, Tuple, Optional

# ================== Generation functions for each status ==================

def generate_normal() -> np.ndarray:
    """
    Simulates normal smart-room behavior (expected trust > 0.8)
    """
    return np.clip([
        np.random.normal(0.6, 0.05),   # motion_rate
        np.random.normal(0.2, 0.05),   # motion_variance
        np.random.normal(0.4, 0.02),   # entry_time
        np.random.normal(0.1, 0.05),   # entry_deviation
        np.random.normal(0.6, 0.1),    # session_duration
        np.random.normal(0.5, 0.1),    # device_usage
        np.random.normal(0.8, 0.05)    # sequence_similarity
    ], 0, 1)


def generate_slight_anomaly() -> np.ndarray:
    """
    Mild deviations from normal (trust ≈ 0.5–0.8)
    """
    return np.clip([
        np.random.normal(0.7, 0.1),    # motion_rate slightly high
        np.random.normal(0.3, 0.1),    # motion_variance elevated
        np.random.normal(0.35, 0.1),   # entry_time a bit off
        np.random.normal(0.2, 0.1),    # entry_deviation increased
        np.random.normal(0.5, 0.15),   # session_duration moderate shift
        np.random.normal(0.4, 0.15),   # device_usage reduced
        np.random.normal(0.7, 0.1)     # sequence_similarity slightly lower
    ], 0, 1)


def generate_suspicious() -> np.ndarray:
    """
    Clearly unusual patterns (trust ≈ 0.2–0.5)
    """
    return np.clip([
        np.random.normal(0.4, 0.2),    # motion_rate erratic
        np.random.normal(0.5, 0.2),    # motion_variance high
        np.random.uniform(0.2, 0.8),   # entry_time unpredictable
        np.random.normal(0.4, 0.15),   # entry_deviation large
        np.random.normal(0.3, 0.2),    # session_duration short
        np.random.normal(0.3, 0.2),    # device_usage low
        np.random.normal(0.4, 0.2)     # sequence_similarity low
    ], 0, 1)


def generate_high_risk() -> np.ndarray:
    """
    Extreme or random values (trust < 0.2)
    """
    return np.clip([
        np.random.uniform(0.0, 0.3) if np.random.rand() < 0.5 else np.random.uniform(0.7, 1.0),  # bimodal
        np.random.uniform(0.3, 1.0),   # high variance
        np.random.uniform(0.0, 1.0),   # completely random
        np.random.uniform(0.3, 1.0),   # large deviation
        np.random.uniform(0.0, 0.3),   # very short or long? here short
        np.random.uniform(0.0, 0.3),   # very low usage
        np.random.uniform(0.0, 0.4)    # low similarity
    ], 0, 1)


# ================== Balanced dataset generator ==================

def generate_balanced_dataset(
    samples_per_class: int = 500,
    statuses: Optional[List[str]] = None,
    include_labels: bool = False,
    save_csv: bool = True,
    filename: str = "smart_room_balanced.csv"
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Generates a dataset with equal number of samples for each requested status.

    Parameters
    ----------
    samples_per_class : int
        Number of samples per status category.
    statuses : list of str, optional
        Which statuses to include. Default: all four.
        Allowed values: 'normal', 'slight_anomaly', 'suspicious', 'high_risk'.
    include_labels : bool
        If True, returns labels (0,1,2,3) corresponding to status order.
    save_csv : bool
        Whether to save the dataset as CSV.
    filename : str
        Name of the CSV file (saved inside 'data/' folder).

    Returns
    -------
    features : np.ndarray of shape (N, 7)
    labels : np.ndarray of shape (N,) if include_labels else None
    """
    if statuses is None:
        statuses = ['normal', 'slight_anomaly', 'suspicious', 'high_risk']

    # Map status names to generator functions
    generator_map = {
        'normal': generate_normal,
        'slight_anomaly': generate_slight_anomaly,
        'suspicious': generate_suspicious,
        'high_risk': generate_high_risk
    }

    data = []
    labels = [] if include_labels else None

    for idx, status in enumerate(statuses):
        func = generator_map.get(status)
        if func is None:
            raise ValueError(f"Unknown status: {status}. Allowed: {list(generator_map.keys())}")

        for _ in range(samples_per_class):
            data.append(func())
            if include_labels:
                labels.append(idx)

    features = np.array(data, dtype=np.float32)
    labels_arr = np.array(labels, dtype=np.int32) if include_labels else None

    if save_csv:
        save_dataset_to_csv(features, labels_arr, filename, statuses)

    return features, labels_arr


def save_dataset_to_csv(
    features: np.ndarray,
    labels: Optional[np.ndarray],
    filename: str,
    status_names: List[str]
) -> None:
    """
    Saves features and optionally labels to CSV inside 'data/' folder.
    """
    os.makedirs("data", exist_ok=True)
    filepath = os.path.join("data", filename)

    headers = [
        "motion_rate", "motion_variance", "entry_time", "entry_deviation",
        "session_duration", "device_usage", "sequence_similarity"
    ]
    if labels is not None:
        headers.append("status_label")
        # Add a human-readable column if desired
        headers.append("status_name")

    with open(filepath, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers)

        for i in range(len(features)):
            row = list(features[i])
            if labels is not None:
                row.append(labels[i])
                row.append(status_names[labels[i]])
            writer.writerow(row)

    print(f"Balanced dataset saved to {filepath}")


# ================== Example usage ==================
if __name__ == "__main__":
    # Generate 2000 samples (500 per status) with labels
    X, y = generate_balanced_dataset(
        samples_per_class=500,
        statuses=['normal', 'slight_anomaly', 'suspicious', 'high_risk'],
        include_labels=True,
        save_csv=True,
        filename="smart_room_balanced_2000.csv"
    )

    print(f"Generated dataset shape: {X.shape}")
    print(f"Label distribution: {np.bincount(y)}")