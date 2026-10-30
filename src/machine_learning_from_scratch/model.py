import numpy as np


def predict(x, w, b):
    """The model: a linear function. Works for one feature or many.

    x: (n,) one example, or (m, n) many examples; w: (n,) weights; b: scalar.
    With a single feature, x can be (m,) and w a scalar.
    """
    return np.dot(x, w) + b


def predict_loop(x, w, b):
    """Same result as predict for ONE example, computed element by element."""
    total = 0.0
    for j in range(len(x)):
        total += w[j] * x[j]
    return total + b
