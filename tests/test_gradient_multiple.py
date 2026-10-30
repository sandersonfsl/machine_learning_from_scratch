import numpy as np
import pytest

from machine_learning_from_scratch.cost import compute_cost
from machine_learning_from_scratch.gradient import compute_gradient
from machine_learning_from_scratch.gradient_descent import gradient_descent


def test_cost_with_several_features_matches_a_hand_calculation():
    # Each example uses a different feature, which keeps the arithmetic simple.
    X = np.array([[1.0, 0.0], [0.0, 1.0]])
    y = np.array([2.0, 4.0])

    # With w = 0 and b = 0 the errors are -2 and -4: (4 + 16) / (2 * 2) = 5
    assert compute_cost(X, y, w=np.zeros(2), b=0.0) == pytest.approx(5.0)


def test_gradient_with_several_features_matches_a_hand_calculation():
    X = np.array([[1.0, 0.0], [0.0, 1.0]])
    y = np.array([2.0, 4.0])

    dj_dw, dj_db = compute_gradient(X, y, w=np.zeros(2), b=0.0)

    # errors = [-2, -4]: dj_dw = [-2 / 2, -4 / 2] = [-1, -2], dj_db = -6 / 2 = -3
    np.testing.assert_allclose(dj_dw, [-1.0, -2.0])
    assert dj_db == pytest.approx(-3.0)


def test_gradient_agrees_with_nudging_each_parameter():
    rng = np.random.default_rng(0)
    X, y = rng.random((20, 3)), rng.random(20)
    w, b, h = np.array([0.5, -1.0, 2.0]), 0.3, 1e-6

    dj_dw, dj_db = compute_gradient(X, y, w, b)

    for j in range(3):
        step = np.zeros(3)
        step[j] = h
        nudge = (compute_cost(X, y, w + step, b) - compute_cost(X, y, w - step, b)) / (2 * h)
        assert dj_dw[j] == pytest.approx(nudge, rel=1e-5)
    nudge_b = (compute_cost(X, y, w, b + h) - compute_cost(X, y, w, b - h)) / (2 * h)
    assert dj_db == pytest.approx(nudge_b, rel=1e-5)


def test_recovers_the_parameters_when_data_has_no_noise():
    rng = np.random.default_rng(0)
    X = rng.random((50, 3))
    y = X @ np.array([2.0, -1.0, 0.5]) + 4.0

    w, b, _ = gradient_descent(X, y, np.zeros(3), 0.0, alpha=0.1, num_iters=30_000)

    np.testing.assert_allclose(w, [2.0, -1.0, 0.5], atol=0.01)
    assert b == pytest.approx(4.0, abs=0.01)
