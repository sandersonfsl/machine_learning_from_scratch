import numpy as np

FEATURES = ["distance_m", "items"]
TRUE_W = np.array([0.003, 1.2])  # minutes per meter and per item
TRUE_B = 10.0  # fixed minutes
LATE_LIMIT = 35.0  # a delivery that takes more than this many minutes is late


def make_late_deliveries(m=100, seed=0, noise_std=3.0):
    """Synthetic deliveries labelled 1 (late) or 0 (on time).

    The delivery time is X @ TRUE_W + TRUE_B + noise, and it is late when that
    time goes beyond LATE_LIMIT. Only the 0/1 label is returned, not the time.
    """
    rng = np.random.default_rng(seed)
    distance_m = np.round(rng.uniform(500, 10_000, m), -2)
    items = rng.integers(1, 13, m)
    X = np.column_stack([distance_m, items]).astype(float)
    minutes = X @ TRUE_W + TRUE_B + rng.normal(0, noise_std, m)
    y = (minutes > LATE_LIMIT).astype(int)
    return X, y
