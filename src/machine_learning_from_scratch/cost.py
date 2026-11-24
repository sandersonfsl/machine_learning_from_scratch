import numpy as np

from machine_learning_from_scratch.model import predict


def compute_cost(x, y, w, b, lambda_=0.0):
    """Squared error cost J(w, b): how badly the line w*x + b fits the data.

    lambda_ is the regularization strength: it adds lambda_/(2m) * sum(w**2), a
    penalty on large weights. With the default 0 there is no penalty. b is never penalized.
    """
    m = len(x)
    predictions = predict(x, w, b)
    errors = predictions - y  # prediction minus the right answer
    squared_errors = errors**2  # squaring removes the sign and punishes big errors
    cost = squared_errors.sum() / (2 * m)  # average (the 2 simplifies math later)
    return cost + lambda_ / (2 * m) * np.sum(np.asarray(w) ** 2)
