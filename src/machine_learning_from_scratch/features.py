import numpy as np


def add_interaction(X, i, j):
    """Append a column with the product of columns i and j."""
    return np.column_stack([X, X[:, i] * X[:, j]])


def polynomial_features(x, degree):
    """Turn a single feature into the columns x, x**2, ..., x**degree."""
    return np.column_stack([x**d for d in range(1, degree + 1)])
