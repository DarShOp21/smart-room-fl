import numpy as np

from fl_client.sensors.pir_sensor import read_motion_events
from fl_client.sensors.door_sensor import get_entry_time
from fl_client.sensors.smart_plug import read_device_usage


def extract_features():

    motion_events = read_motion_events()

    avg_motion_rate = np.mean(motion_events)

    motion_variance = np.var(motion_events)

    entry_time = get_entry_time()

    entry_time_deviation = abs(entry_time - 0.4)

    session_duration = 0.6

    device_activations = read_device_usage()

    device_sequence_similarity = np.random.normal(0.8, 0.05)

    features = [
        avg_motion_rate,
        motion_variance,
        entry_time,
        entry_time_deviation,
        session_duration,
        device_activations,
        device_sequence_similarity
    ]

    features = np.clip(features, 0, 1)

    return np.array(features, dtype=np.float32)