from machine_learning_from_scratch.cost import compute_cost
from machine_learning_from_scratch.gradient import compute_gradient


def gradient_descent(x, y, w, b, alpha, num_iters):
    """Repeat the gradient step num_iters times, starting from w and b.

    Returns the learned w and b, and the cost after every iteration.
    """
    costs = []
    for _ in range(num_iters):
        dj_dw, dj_db = compute_gradient(x, y, w, b)

        # Simultaneous update: both use the old w and b
        w, b = w - alpha * dj_dw, b - alpha * dj_db

        costs.append(compute_cost(x, y, w, b))

    return w, b, costs
