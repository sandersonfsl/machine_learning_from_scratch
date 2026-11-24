import numpy as np

from machine_learning_from_scratch.model import predict


def compute_gradient(x, y, w, b, lambda_=0.0):
    """Derivatives of the cost: how much J changes if w or b grows a little.

    With several features, dj_dw has one derivative per weight. lambda_ is the
    regularization strength: it adds lambda_/m * w to dj_dw (and nothing to dj_db).
    """
    m = len(x)
    errors = predict(x, w, b) - y
    dj_dw = np.dot(x.T, errors) / m + lambda_ / m * w
    dj_db = errors.sum() / m
    return dj_dw, dj_db
