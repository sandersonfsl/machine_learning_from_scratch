import numpy as np
import pytest

from machine_learning_from_scratch.convergence import is_decreasing
from machine_learning_from_scratch.cost import compute_cost
from machine_learning_from_scratch.data_classification import make_late_deliveries
from machine_learning_from_scratch.gradient import compute_gradient
from machine_learning_from_scratch.gradient_descent import gradient_descent
from machine_learning_from_scratch.logistic import (
    accuracy,
    compute_cost_logistic,
    compute_gradient_logistic,
    sigmoid,
)
from machine_learning_from_scratch.scaling import zscore_normalize

X_small = np.array([[0.5, 1.5], [1.0, 1.0], [1.5, 0.5], [3.0, 0.5], [2.0, 2.0], [1.0, 2.5]])
y_small = np.array([0, 0, 0, 1, 1, 1])


def deliveries():
    X, y = make_late_deliveries()
    return X * [0.001, 1.0], y  # columns [km, items]


def test_gradient_of_a_single_example_matches_a_hand_calculation():
    # z = 0 so f = 0.5; error = 0.5 - 1 = -0.5; dj_dw = -0.5 * x, dj_db = -0.5
    dj_dw, dj_db = compute_gradient_logistic(np.array([[2.0, 4.0]]), np.array([1]), np.zeros(2), 0.0)

    np.testing.assert_allclose(dj_dw, [-1.0, -2.0])
    assert dj_db == pytest.approx(-0.5)


def test_gradient_matches_the_values_worked_out_for_the_small_problem():
    dj_dw, dj_db = compute_gradient_logistic(X_small, y_small, np.array([2.0, 3.0]), 1.0)

    np.testing.assert_allclose(dj_dw, [0.49833339, 0.49883943], rtol=1e-6)
    assert dj_db == pytest.approx(0.49861807, rel=1e-6)


def test_gradient_matches_a_numerical_derivative_of_the_cost():
    X, y = deliveries()
    w, b = np.array([0.7, -0.2]), -1.0
    h = 1e-6

    dj_dw, dj_db = compute_gradient_logistic(X, y, w, b)

    for j in range(2):
        step = np.zeros(2)
        step[j] = h
        numeric = (compute_cost_logistic(X, y, w + step, b) - compute_cost_logistic(X, y, w - step, b)) / (2 * h)
        assert dj_dw[j] == pytest.approx(numeric, rel=1e-5)
    numeric_b = (compute_cost_logistic(X, y, w, b + h) - compute_cost_logistic(X, y, w, b - h)) / (2 * h)
    assert dj_db == pytest.approx(numeric_b, rel=1e-5)


def test_gradient_is_zero_when_the_model_is_exactly_right():
    # labels equal to the probabilities themselves leave no error to correct
    X = np.array([[1.0], [2.0], [-1.0]])
    w, b = np.array([0.5]), 0.2
    y = sigmoid(np.dot(X, w) + b)

    dj_dw, dj_db = compute_gradient_logistic(X, y, w, b)

    np.testing.assert_allclose(dj_dw, 0.0, atol=1e-12)
    assert dj_db == pytest.approx(0.0, abs=1e-12)


def test_accuracy_counts_the_right_predictions():
    X = np.array([[-2.0], [-1.0], [1.0], [2.0]])
    y = np.array([0, 1, 1, 1])  # the model with w = 1, b = 0 answers 0, 0, 1, 1

    assert accuracy(X, y, np.array([1.0]), 0.0) == pytest.approx(0.75)


def test_gradient_descent_without_functions_still_trains_linear_regression():
    x, y = np.array([1.0, 2.0, 3.0]), np.array([3.0, 5.0, 7.0])  # y = 2x + 1

    default = gradient_descent(x, y, 0.0, 0.0, 0.1, 500)
    explicit = gradient_descent(x, y, 0.0, 0.0, 0.1, 500, compute_cost, compute_gradient)

    assert default[0] == explicit[0] and default[1] == explicit[1]
    assert default[0] == pytest.approx(2.0, abs=1e-2)


def test_logistic_cost_goes_down_at_every_step():
    X, y = deliveries()
    X_scaled, _, _ = zscore_normalize(X)

    _, _, costs = gradient_descent(
        X_scaled, y, np.zeros(2), 0.0, 0.5, 300, compute_cost_logistic, compute_gradient_logistic
    )

    assert is_decreasing(costs)
    assert costs[0] < np.log(2)  # already below the cost of knowing nothing


def test_training_reaches_the_minimum_and_classifies_most_deliveries():
    X, y = deliveries()
    X_scaled, _, _ = zscore_normalize(X)

    w, b, costs = gradient_descent(
        X_scaled, y, np.zeros(2), 0.0, 0.5, 3000, compute_cost_logistic, compute_gradient_logistic
    )

    # the gradient is almost zero at the minimum
    dj_dw, dj_db = compute_gradient_logistic(X_scaled, y, w, b)
    assert np.abs(dj_dw).max() < 1e-4 and abs(dj_db) < 1e-4
    assert costs[-1] < compute_cost_logistic(X, y, np.array([0.9, 0.36]), -7.5)  # beats the hand-picked
    assert accuracy(X_scaled, y, w, b) >= 0.9
