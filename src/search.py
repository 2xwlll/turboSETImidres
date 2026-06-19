# src/search.py

import numpy as np


def run_search(data, config):
    """
    Placeholder for drift search / FFT / detection logic.
    """

    candidates = []

    for i, item in enumerate(data):
        # fake detection logic for now
        snr = np.random.rand() * 20

        if snr > config.snr_threshold:
            candidates.append({
                "index": i,
                "snr": snr
            })

    return candidates