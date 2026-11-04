import numpy as np

FEATURES = ["distance_m", "items", "rush_hour"]
TRUE_W = np.array([0.003, 1.2, 7.0])  # minutes per meter, per item, and rush-hour penalty
TRUE_B = 10.0  # fixed minutes


def make_deliveries(m=100, seed=0, noise_std=3.0, rush_slowdown=0.0):
    """Synthetic deliveries: time = X @ TRUE_W + TRUE_B + noise.

    With rush_slowdown > 0, every meter also costs that many extra minutes in the
    rush hour (an interaction between distance and rush hour).
    """
    rng = np.random.default_rng(seed)
    distance_m = np.round(rng.uniform(500, 10_000, m), -2)
    items = rng.integers(1, 13, m)
    rush_hour = (rng.random(m) < 0.4).astype(int)
    X = np.column_stack([distance_m, items, rush_hour]).astype(float)
    y = X @ TRUE_W + TRUE_B + rush_slowdown * distance_m * rush_hour + rng.normal(0, noise_std, m)
    return X, np.round(y, 1)
