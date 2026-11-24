import numpy as np


def sigmoid(z):
    """The sigmoid function g(z) = 1 / (1 + e^-z). Works for a number or an array.

    It maps any value to a number between 0 and 1: large negative values give
    almost 0, 0 gives exactly 0.5, and large positive values give almost 1.
    """
    # clip keeps e^-z from overflowing when z is very negative
    return 1 / (1 + np.exp(-np.clip(z, -500, 500)))


def predict_proba(X, w, b):
    """The logistic model f(x) = g(w . x + b): the probability that the label is 1.

    X: (n,) one example or (m, n) many examples; w: (n,) weights; b: scalar.
    """
    return sigmoid(np.dot(X, w) + b)


def predict_class(X, w, b, threshold=0.5):
    """The predicted label, 0 or 1: 1 when the probability reaches the threshold."""
    return (predict_proba(X, w, b) >= threshold).astype(int)


def compute_cost_logistic(X, y, w, b, lambda_=0.0):
    """Logistic (log loss) cost J(w, b): how badly the probabilities fit the 0/1 labels.

    For one example the loss is  -log(f)      when y = 1
                                 -log(1 - f)  when y = 0
    which is written in one line as  -y*log(f) - (1 - y)*log(1 - f).

    lambda_ is the regularization strength: it adds lambda_/(2m) * sum(w**2), a
    penalty on large weights. With the default 0 there is no penalty. b is never penalized.
    """
    z = np.dot(X, w) + b
    # The same loss, rewritten in terms of z:  log(1 + e^z) - y*z.
    # np.logaddexp(0, z) computes log(1 + e^z) without overflow, and it never
    # takes log(0), which f = 0 or f = 1 would make with the plain formula.
    loss = np.logaddexp(0, z) - y * z
    return loss.mean() + lambda_ / (2 * len(X)) * np.sum(w**2)


def compute_gradient_logistic(X, y, w, b, lambda_=0.0):
    """Derivatives of the logistic cost with respect to w and b.

    It has the same form as the gradient of linear regression: the only change is
    that the prediction f is the sigmoid of the linear part. lambda_ is the
    regularization strength: it adds lambda_/m * w to dj_dw (and nothing to dj_db).
    Returns (dj_dw, dj_db).
    """
    m = len(X)
    errors = predict_proba(X, w, b) - y
    dj_dw = np.dot(X.T, errors) / m + lambda_ / m * w
    dj_db = errors.sum() / m
    return dj_dw, dj_db


def accuracy(X, y, w, b):
    """The share of examples whose predicted class is the right one."""
    return np.mean(predict_class(X, w, b) == y)
