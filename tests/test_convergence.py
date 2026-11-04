import numpy as np

from machine_learning_from_scratch.convergence import converged_at, is_decreasing
from machine_learning_from_scratch.data_multiple import make_deliveries
from machine_learning_from_scratch.gradient_descent import gradient_descent
from machine_learning_from_scratch.scaling import zscore_normalize


def test_is_decreasing_on_hand_made_curves():
    assert is_decreasing([10.0, 5.0, 2.0, 1.0])
    assert not is_decreasing([10.0, 5.0, 6.0, 1.0])


def test_converged_at_finds_the_first_small_change():
    costs = [10.0, 5.0, 2.0, 1.9995, 1.9994]

    # 2.0 -> 1.9995 is the first change smaller than 0.001
    assert converged_at(costs, epsilon=1e-3) == 3


def test_converged_at_is_none_while_the_cost_still_moves():
    assert converged_at([10.0, 5.0, 2.0, 1.0], epsilon=1e-3) is None


def test_a_small_alpha_never_goes_up_and_a_large_one_does():
    X, y = make_deliveries()
    X_scaled, _, _ = zscore_normalize(X)

    _, _, small = gradient_descent(X_scaled, y, np.zeros(3), 0.0, alpha=0.1, num_iters=100)
    with np.errstate(all="ignore"):
        _, _, large = gradient_descent(X_scaled, y, np.zeros(3), 0.0, alpha=2.0, num_iters=100)

    assert is_decreasing(small)
    assert not is_decreasing(large)


def test_a_larger_alpha_converges_in_fewer_iterations():
    X, y = make_deliveries()
    X_scaled, _, _ = zscore_normalize(X)

    stops = []
    for alpha in (0.03, 0.1, 0.3):
        _, _, costs = gradient_descent(X_scaled, y, np.zeros(3), 0.0, alpha, num_iters=300)
        stops.append(converged_at(costs, epsilon=1e-3))

    assert stops[0] > stops[1] > stops[2]
