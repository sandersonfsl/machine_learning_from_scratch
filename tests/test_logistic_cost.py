import numpy as np
import pytest

from machine_learning_from_scratch.data_classification import make_late_deliveries
from machine_learning_from_scratch.logistic import compute_cost_logistic, predict_proba

# six points of a small classification problem, with the costs worked out by hand
X_small = np.array([[0.5, 1.5], [1.0, 1.0], [1.5, 0.5], [3.0, 0.5], [2.0, 2.0], [1.0, 2.5]])
y_small = np.array([0, 0, 0, 1, 1, 1])


def cost_with_the_plain_formula(X, y, w, b):
    f = predict_proba(X, w, b)
    return np.mean(-y * np.log(f) - (1 - y) * np.log(1 - f))


def test_cost_of_a_single_example_matches_the_formula():
    # y = 1 and f = 0.5 gives -log(0.5) = log 2; the same holds for y = 0
    X = np.array([[1.0]])

    assert compute_cost_logistic(X, np.array([1]), np.array([0.0]), 0.0) == pytest.approx(np.log(2))
    assert compute_cost_logistic(X, np.array([0]), np.array([0.0]), 0.0) == pytest.approx(np.log(2))


def test_cost_of_a_confident_right_and_a_confident_wrong_answer():
    # z = 2 gives f = 0.8808: -log(0.8808) = 0.1269 if y = 1, -log(0.1192) = 2.1269 if y = 0
    X, w = np.array([[1.0]]), np.array([2.0])

    assert compute_cost_logistic(X, np.array([1]), w, 0.0) == pytest.approx(0.126928, rel=1e-5)
    assert compute_cost_logistic(X, np.array([0]), w, 0.0) == pytest.approx(2.126928, rel=1e-5)


def test_cost_matches_the_values_worked_out_for_the_small_problem():
    w = np.array([1.0, 1.0])

    assert compute_cost_logistic(X_small, y_small, w, -3.0) == pytest.approx(0.3669, abs=1e-4)
    assert compute_cost_logistic(X_small, y_small, w, -4.0) == pytest.approx(0.5037, abs=1e-4)


def test_cost_matches_the_plain_formula_on_the_deliveries():
    X, y = make_late_deliveries()
    X = X * [0.001, 1.0]
    w, b = np.array([0.9, 0.36]), -7.5

    assert compute_cost_logistic(X, y, w, b) == pytest.approx(cost_with_the_plain_formula(X, y, w, b))


def test_cost_is_never_negative_and_is_small_for_a_good_fit():
    X, y = make_late_deliveries()
    X = X * [0.001, 1.0]

    good = compute_cost_logistic(X, y, np.array([0.9, 0.36]), -7.5)
    bad = compute_cost_logistic(X, y, np.array([-0.9, -0.36]), 7.5)  # the opposite answers

    assert 0 < good < 0.5 < bad


def test_cost_of_all_zero_parameters_is_log_2():
    # f = 0.5 for everyone, so every example costs log 2 whatever its label
    assert compute_cost_logistic(X_small, y_small, np.zeros(2), 0.0) == pytest.approx(np.log(2))


def test_cost_does_not_break_with_extreme_probabilities():
    X, y = np.array([[1.0], [-1.0]]), np.array([1, 0])

    # a log(0) or an overflow would raise here (underflow to 0 is harmless)
    with np.errstate(over="raise", divide="raise", invalid="raise"):
        right = compute_cost_logistic(X, y, np.array([1000.0]), 0.0)  # f = 1 and 0, both right
        wrong = compute_cost_logistic(X, y, np.array([-1000.0]), 0.0)  # f = 0 and 1, both wrong

    assert right == pytest.approx(0.0, abs=1e-12)
    assert wrong == pytest.approx(1000.0)
