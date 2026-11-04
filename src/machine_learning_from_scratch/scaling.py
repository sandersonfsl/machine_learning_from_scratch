import numpy as np


def zscore_normalize(X):
    """Rescale every feature (column) to mean 0 and standard deviation 1.

    Returns the rescaled table, plus the mean and the standard deviation of each
    column. Both are needed later: to rescale new examples and to undo the scaling.
    """
    mu = X.mean(axis=0)
    sigma = X.std(axis=0)
    if np.any(sigma == 0):
        raise ValueError("a feature is constant (standard deviation 0); remove it before scaling")
    return (X - mu) / sigma, mu, sigma


def apply_zscore(x, mu, sigma):
    """Rescale new examples with the mean and standard deviation of the training set."""
    return (x - mu) / sigma


def unscale_parameters(w_scaled, b_scaled, mu, sigma):
    """Turn parameters learned on scaled features into parameters for the original ones."""
    w = w_scaled / sigma
    b = b_scaled - np.sum(w_scaled * mu / sigma)
    return w, b
