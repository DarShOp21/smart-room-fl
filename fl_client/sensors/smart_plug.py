import random


def read_device_usage():

    """
    Simulates how many smart devices
    were activated during the session
    """

    activations = random.randint(1, 5)

    normalized = activations / 5

    return normalized