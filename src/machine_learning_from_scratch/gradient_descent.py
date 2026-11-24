from machine_learning_from_scratch.cost import compute_cost
from machine_learning_from_scratch.gradient import compute_gradient


def gradient_descent(x, y, w, b, alpha, num_iters, cost_function=compute_cost, gradient_function=compute_gradient):
    """Repeat the gradient step num_iters times, starting from w and b.

    By default it trains linear regression. For another model, pass its cost and
    gradient functions, both called as function(x, y, w, b).

    Returns the learned w and b, and the cost after every iteration.
    """
    costs = []
    for _ in range(num_iters):
        dj_dw, dj_db = gradient_function(x, y, w, b)

        # Simultaneous update: both use the old w and b
        w, b = w - alpha * dj_dw, b - alpha * dj_db

        costs.append(cost_function(x, y, w, b))

    return w, b, costs
