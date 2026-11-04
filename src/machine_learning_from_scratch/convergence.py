def is_decreasing(costs, tolerance=1e-9):
    """True if the cost never went up (beyond rounding noise) between iterations.

    With a correct gradient and a small enough alpha this always holds. If it does
    not, alpha is too large or the gradient has a bug.
    """
    return all(later <= earlier + tolerance for earlier, later in zip(costs, costs[1:]))


def converged_at(costs, epsilon):
    """Index of the first iteration whose cost changed by less than epsilon.

    Returns None if the cost was still changing by epsilon or more at the end.
    """
    for i in range(1, len(costs)):
        if abs(costs[i - 1] - costs[i]) < epsilon:
            return i
    return None
