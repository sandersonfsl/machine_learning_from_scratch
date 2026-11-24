import numpy as np


def add_interaction(X, i, j):
    """Append a column with the product of columns i and j."""
    return np.column_stack([X, X[:, i] * X[:, j]])


def polynomial_features(x, degree):
    """Turn a single feature into the columns x, x**2, ..., x**degree."""
    return np.column_stack([x**d for d in range(1, degree + 1)])


def polynomial_features_2d(X, degree):
    """Turn two features x1, x2 into every product x1**i * x2**j with 1 <= i + j <= degree.

    The columns come ordered by total degree: x1, x2, then x1**2, x1*x2, x2**2, and so on.
    """
    x1, x2 = X[:, 0], X[:, 1]
    return np.column_stack([x1 ** (d - j) * x2**j for d in range(1, degree + 1) for j in range(d + 1)])
