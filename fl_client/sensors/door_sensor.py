import time


def get_entry_time():
    """
    Returns normalized hour of entry (0–1).
    """

    hour = time.localtime().tm_hour

    normalized_hour = hour / 24

    return normalized_hour