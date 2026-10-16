import numpy as np
import pytest

from machine_learning_from_scratch.cost import compute_cost
from machine_learning_from_scratch.data import x, y
from machine_learning_from_scratch.gradient import compute_gradient


def test_gradient_matches_a_hand_calculation():
    x_small = np.array([1.0, 2.0, 3.0])
    y_small = np.array([1.0, 2.0, 3.0])

    # With w=0 and b=0 the errors are -1, -2, -3.
    # dj_dw = (-1*1 + -2*2 + -3*3) / 3 = -14 / 3
    # dj_db = (-1 + -2 + -3) / 3 = -2
    dj_dw, dj_db = compute_gradient(x_small, y_small, w=0, b=0)

    assert dj_dw == pytest.approx(-14 / 3)
    assert dj_db == pytest.approx(-2)


def test_gradient_is_zero_at_the_best_line():
    x_line = np.array([1.0, 2.0, 3.0])
    y_line = np.array([15.0, 18.0, 21.0])  # exactly 3 * x + 12

    dj_dw, dj_db = compute_gradient(x_line, y_line, w=3, b=12)

    assert dj_dw == 0.0
    assert dj_db == 0.0


def test_gradient_agrees_with_nudging_the_parameters():
    # Gradient checking: nudge a parameter slightly in both directions and
    # measure how much the cost changes. That slope must match the formula.
    w, b, h = 1.5, 4.0, 1e-5

    nudge_w = (compute_cost(x, y, w + h, b) - compute_cost(x, y, w - h, b)) / (2 * h)
    nudge_b = (compute_cost(x, y, w, b + h) - compute_cost(x, y, w, b - h)) / (2 * h)
    dj_dw, dj_db = compute_gradient(x, y, w, b)

    assert dj_dw == pytest.approx(nudge_w, rel=1e-4)
    assert dj_db == pytest.approx(nudge_b, rel=1e-4)
