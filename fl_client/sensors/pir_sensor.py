import random
import time


def read_motion_events(duration=10):
    """
    Simulates PIR sensor motion events
    over a time window.
    """

    events = []

    start = time.time()

    while time.time() - start < duration:

        motion_detected = random.choice([0, 1])

        events.append(motion_detected)

        time.sleep(0.5)

    return events