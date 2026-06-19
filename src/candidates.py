# src/candidates.py


def postprocess_candidates(candidates, config):
    """
    Filters / ranks / cleans raw detections.
    """

    filtered = [
        c for c in candidates
        if c["snr"] >= config.snr_threshold
    ]

    # sort strongest signals first
    filtered.sort(key=lambda x: x["snr"], reverse=True)

    return filtered