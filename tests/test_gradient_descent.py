import numpy as np
import pytest

from machine_learning_from_scratch.data import x, y
from machine_learning_from_scratch.gradient_descent import gradient_descent


def test_recovers_the_line_when_data_has_no_noise():
    x_line = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y_line = 3 * x_line + 12  # no noise: the best line is exactly w=3, b=12

    w, b, _ = gradient_descent(x_line, y_line, w=0.0, b=0.0, alpha=0.05, num_iters=20_000)

    assert w == pytest.approx(3, abs=0.01)
    assert b == pytest.approx(12, abs=0.01)


def test_cost_never_increases():
    _, _, costs = gradient_descent(x, y, w=0.0, b=0.0, alpha=0.01, num_iters=1_000)

    assert all(later <= earlier for earlier, later in zip(costs, costs[1:]))


def test_returns_one_cost_per_iteration():
    _, _, costs = gradient_descent(x, y, w=0.0, b=0.0, alpha=0.01, num_iters=50)

    assert len(costs) == 50


def test_one_step_updates_w_and_b_simultaneously():
    x_small = np.array([1.0, 2.0, 3.0])
    y_small = np.array([1.0, 2.0, 3.0])

    # From w=0, b=0 the gradient is dj_dw = -14/3 and dj_db = -2 (computed by hand).
    # With alpha=0.1: new w = 0.1 * 14/3 = 14/30, new b = 0.1 * 2 = 0.2.
    # If b were updated using the already-updated w, it would come out different.
    w, b, _ = gradient_descent(x_small, y_small, w=0.0, b=0.0, alpha=0.1, num_iters=1)

    assert w == pytest.approx(14 / 30)
    assert b == pytest.approx(0.2)
