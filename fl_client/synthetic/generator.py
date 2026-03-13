import numpy as np
import csv
import os
from typing import List, Optional, Union

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
) -> Union[np.ndarray, tuple]:
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
        If True, returns (features, labels) where labels are integers 0..(n-1).

    Returns
    -------
    features : np.ndarray of shape (N, 7)
    labels : np.ndarray of shape (N,) if include_labels else None
    """
    if statuses is None:
        statuses = ['normal', 'slight_anomaly', 'suspicious', 'high_risk']

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

    if include_labels:
        return features, labels_arr
    return features


# ================== Original (legacy) dataset generator ==================

def generate_normal_session() -> np.ndarray:
    """Legacy normal session generator (used by old generate_dataset)."""
    return np.clip([
        np.random.normal(0.6, 0.05),
        np.random.normal(0.2, 0.05),
        np.random.normal(0.4, 0.02),
        np.random.normal(0.1, 0.05),
        np.random.normal(0.6, 0.1),
        np.random.normal(0.5, 0.1),
        np.random.normal(0.8, 0.05)
    ], 0, 1)


def generate_suspicious_session() -> np.ndarray:
    """Legacy suspicious session generator (used by old generate_dataset)."""
    return np.clip([
        np.random.normal(0.3, 0.2),
        np.random.normal(0.6, 0.2),
        np.random.uniform(0, 1),
        np.random.normal(0.6, 0.2),
        np.random.normal(0.2, 0.2),
        np.random.normal(0.2, 0.2),
        np.random.normal(0.3, 0.2)
    ], 0, 1)


# ================== Main user-facing function ==================

def generate_dataset(
    num_samples: int = 2000,
    suspicious_ratio: float = 0.3,
    balanced: bool = False,
    statuses: Optional[List[str]] = None,
    save_csv: bool = False,
    filename: str = "smart_room_dataset.csv"
) -> np.ndarray:
    """
    Generate a synthetic smart-room dataset.

    Two modes:
      - Balanced mode (balanced=True): produces equal samples per status.
      - Legacy mode (balanced=False): uses normal vs. suspicious ratio.

    Parameters
    ----------
    num_samples : int
        Total number of samples (ignored in balanced mode; use samples_per_class).
    suspicious_ratio : float
        Fraction of suspicious samples (only in legacy mode).
    balanced : bool
        If True, use balanced generation across statuses.
    statuses : list of str, optional
        List of statuses to include (only in balanced mode). Defaults to all four.
    save_csv : bool
        If True, save dataset to a CSV file in the 'data/' folder.
    filename : str
        Name of the CSV file (only used if save_csv=True).

    Returns
    -------
    np.ndarray of shape (N, 7)
    """
    if balanced:
        # In balanced mode, num_samples is interpreted as samples per class
        samples_per_class = num_samples if num_samples > 0 else 500
        features = generate_balanced_dataset(
            samples_per_class=samples_per_class,
            statuses=statuses,
            include_labels=False
        )
    else:
        # Legacy mode: mix of normal and suspicious
        data = []
        for _ in range(num_samples):
            if np.random.rand() < suspicious_ratio:
                data.append(generate_suspicious_session())
            else:
                data.append(generate_normal_session())
        features = np.array(data, dtype=np.float32)

    if save_csv:
        _save_dataset_to_csv(features, filename)

    return features


def _save_dataset_to_csv(dataset: np.ndarray, filename: str) -> None:
    """Internal helper to save dataset to CSV."""
    headers = [
        "motion_rate", "motion_variance", "entry_time", "entry_deviation",
        "session_duration", "device_usage", "sequence_similarity"
    ]
    os.makedirs("data", exist_ok=True)
    filepath = os.path.join("data", filename)
    with open(filepath, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(dataset)
    print(f"Dataset saved to {filepath}")


# ================== Example usage (if run directly) ==================
if __name__ == "__main__":
    # Example 1: balanced dataset (250 samples per status → total 1000)
    X_bal = generate_dataset(balanced=True, num_samples=250, save_csv=True, filename="balanced_1000.csv")
    print(f"Balanced dataset shape: {X_bal.shape}")

    # Example 2: legacy dataset (2000 samples, 20% suspicious)
    X_leg = generate_dataset(num_samples=2000, suspicious_ratio=0.2, save_csv=True, filename="legacy_2000.csv")
    print(f"Legacy dataset shape: {X_leg.shape}")