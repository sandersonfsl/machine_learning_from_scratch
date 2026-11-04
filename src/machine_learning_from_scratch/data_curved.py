import numpy as np

TRUE_COEFFICIENTS = (10.0, 5.0, -0.25)  # time = 10 + 5*km - 0.25*km**2


def make_curved_deliveries(m=60, seed=0, noise_std=2.0):
    """Synthetic deliveries whose time grows with the distance, but ever more slowly."""
    rng = np.random.default_rng(seed)
    km = np.round(rng.uniform(0.5, 10, m), 1)
    a, b, c = TRUE_COEFFICIENTS
    y = a + b * km + c * km**2 + rng.normal(0, noise_std, m)
    return km, np.round(y, 1)
