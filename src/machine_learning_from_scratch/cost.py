from machine_learning_from_scratch.model import predict


def compute_cost(x, y, w, b):
    """Squared error cost J(w, b): how badly the line w*x + b fits the data."""
    m = len(x)
    predictions = predict(x, w, b)
    errors = predictions - y  # prediction minus the right answer
    squared_errors = errors**2  # squaring removes the sign and punishes big errors
    return squared_errors.sum() / (2 * m)  # average (the 2 simplifies math later)
