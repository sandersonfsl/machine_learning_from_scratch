from functools import partial

import numpy as np
import pytest

from machine_learning_from_scratch.cost import compute_cost
from machine_learning_from_scratch.data_classification import make_late_deliveries
from machine_learning_from_scratch.features import polynomial_features_2d
from machine_learning_from_scratch.gradient import compute_gradient
from machine_learning_from_scratch.gradient_descent import gradient_descent
from machine_learning_from_scratch.logistic import (
    accuracy,
    compute_cost_logistic,
    compute_gradient_logistic,
)
from machine_learning_from_scratch.scaling import apply_zscore, zscore_normalize

X_small = np.array([[0.5, 1.5], [1.0, 1.0], [1.5, 0.5], [3.0, 0.5], [2.0, 2.0], [1.0, 2.5]])
y_small = np.array([0, 0, 0, 1, 1, 1])


# --- the penalty in the cost ----------------------------------------------------


def test_linear_cost_adds_the_penalty_calculated_by_hand():
    # one example, x = 1, y = 0, w = [2], b = 0: error 2, J = 4/2 = 2
    # penalty = lambda/(2m) * w**2 = 1/(2*1) * 4 = 2
    x, y, w = np.array([[1.0]]), np.array([0.0]), np.array([2.0])

    assert compute_cost(x, y, w, 0.0) == pytest.approx(2.0)
    assert compute_cost(x, y, w, 0.0, lambda_=1.0) == pytest.approx(4.0)


def test_logistic_cost_adds_the_penalty_calculated_by_hand():
    # w = [3, 4], m = 6: penalty = lambda/(2*6) * (9 + 16) = lambda * 25/12
    w = np.array([3.0, 4.0])
    plain = compute_cost_logistic(X_small, y_small, w, 0.0)

    penalized = compute_cost_logistic(X_small, y_small, w, 0.0, lambda_=0.6)

    assert penalized - plain == pytest.approx(0.6 * 25 / 12)


def test_lambda_zero_changes_nothing():
    w, b = np.array([1.0, 1.0]), -3.0

    assert compute_cost_logistic(X_small, y_small, w, b, lambda_=0.0) == compute_cost_logistic(X_small, y_small, w, b)
    assert compute_cost(X_small, y_small, w, b, lambda_=0.0) == compute_cost(X_small, y_small, w, b)


def test_the_penalty_ignores_b():
    # with w = 0 only b is left, so lambda must have no effect
    w = np.zeros(2)

    assert compute_cost_logistic(X_small, y_small, w, 5.0, lambda_=100.0) == pytest.approx(
        compute_cost_logistic(X_small, y_small, w, 5.0)
    )


# --- the penalty in the gradient ------------------------------------------------


def test_gradient_adds_lambda_over_m_times_w_and_leaves_db_alone():
    w, b = np.array([2.0, 3.0]), 1.0

    for gradient in (compute_gradient, compute_gradient_logistic):
        dj_dw, dj_db = gradient(X_small, y_small, w, b)
        dj_dw_reg, dj_db_reg = gradient(X_small, y_small, w, b, lambda_=3.0)

        np.testing.assert_allclose(dj_dw_reg - dj_dw, 3.0 / 6 * w)
        assert dj_db_reg == dj_db


@pytest.mark.parametrize(
    "cost, gradient",
    [(compute_cost, compute_gradient), (compute_cost_logistic, compute_gradient_logistic)],
)
def test_regularized_gradient_matches_a_numerical_derivative(cost, gradient):
    w, b, lambda_, h = np.array([0.7, -0.4]), 0.3, 2.0, 1e-6

    dj_dw, _ = gradient(X_small, y_small, w, b, lambda_=lambda_)

    for j in range(2):
        step = np.zeros(2)
        step[j] = h
        numeric = (cost(X_small, y_small, w + step, b, lambda_) - cost(X_small, y_small, w - step, b, lambda_)) / (
            2 * h
        )
        assert dj_dw[j] == pytest.approx(numeric, rel=1e-5)


def test_a_larger_lambda_gives_smaller_weights():
    sizes = []
    for lambda_ in (0.0, 1.0, 10.0):
        w, _, _ = gradient_descent(
            X_small,
            y_small,
            np.zeros(2),
            0.0,
            0.1,
            5000,
            partial(compute_cost_logistic, lambda_=lambda_),
            partial(compute_gradient_logistic, lambda_=lambda_),
        )
        sizes.append(np.abs(w).sum())

    assert sizes[0] > sizes[1] > sizes[2]


# --- polynomial features with two variables ------------------------------------


def test_polynomial_features_2d_by_hand():
    # x1 = 2, x2 = 3: x1, x2, x1**2, x1*x2, x2**2
    result = polynomial_features_2d(np.array([[2.0, 3.0]]), 2)

    np.testing.assert_array_equal(result, [[2.0, 3.0, 4.0, 6.0, 9.0]])


def test_polynomial_features_2d_has_the_expected_number_of_columns():
    X = np.ones((4, 2))

    # degree d adds d + 1 columns: 2 + 3 + ... + (d + 1) = d(d + 3)/2
    for degree in (1, 2, 3, 6):
        assert polynomial_features_2d(X, degree).shape == (4, degree * (degree + 3) // 2)


def test_polynomial_features_2d_of_degree_1_is_the_original_table():
    np.testing.assert_array_equal(polynomial_features_2d(X_small, 1), X_small)


# --- overfitting ----------------------------------------------------------------


def test_regularization_reduces_overfitting():
    X_train, y_train = make_late_deliveries(m=30, seed=0)
    X_test, y_test = make_late_deliveries(m=1000, seed=1)
    features = polynomial_features_2d(X_train * [0.001, 1.0], 6)
    Z_train, mu, sigma = zscore_normalize(features)
    Z_test = apply_zscore(polynomial_features_2d(X_test * [0.001, 1.0], 6), mu, sigma)

    results = {}
    for lambda_ in (0.0, 1.0):
        w, b, _ = gradient_descent(
            Z_train,
            y_train,
            np.zeros(Z_train.shape[1]),
            0.0,
            0.1,
            20_000,
            partial(compute_cost_logistic, lambda_=lambda_),
            partial(compute_gradient_logistic, lambda_=lambda_),
        )
        results[lambda_] = (accuracy(Z_train, y_train, w, b), accuracy(Z_test, y_test, w, b))

    train_free, test_free = results[0.0]
    train_reg, test_reg = results[1.0]
    assert train_free == 1.0  # without the penalty it memorizes the 30 deliveries...
    assert test_free < 0.85  # ...and fails on new ones
    assert test_reg > test_free + 0.05  # the penalty gives up some training accuracy and wins on new data
    assert train_reg < train_free
