import numpy as np
import pytest

from machine_learning_from_scratch.cost import compute_cost
from machine_learning_from_scratch.data_multiple import make_deliveries
from machine_learning_from_scratch.gradient_descent import gradient_descent
from machine_learning_from_scratch.model import predict
from machine_learning_from_scratch.scaling import apply_zscore, unscale_parameters, zscore_normalize


def test_zscore_matches_a_hand_calculation():
    X = np.array([[1.0, 10.0], [3.0, 30.0]])

    X_scaled, mu, sigma = zscore_normalize(X)

    # means are [2, 20] and standard deviations are [1, 10]
    np.testing.assert_allclose(mu, [2.0, 20.0])
    np.testing.assert_allclose(sigma, [1.0, 10.0])
    np.testing.assert_allclose(X_scaled, [[-1.0, -1.0], [1.0, 1.0]])


def test_scaled_features_have_mean_zero_and_standard_deviation_one():
    X, _ = make_deliveries()

    X_scaled, _, _ = zscore_normalize(X)

    np.testing.assert_allclose(X_scaled.mean(axis=0), 0.0, atol=1e-12)
    np.testing.assert_allclose(X_scaled.std(axis=0), 1.0)


def test_new_examples_are_scaled_with_the_training_mean_and_std():
    X = np.array([[1.0, 10.0], [3.0, 30.0]])
    _, mu, sigma = zscore_normalize(X)

    # the average training example is exactly in the middle: [0, 0] once scaled
    np.testing.assert_allclose(apply_zscore(np.array([2.0, 20.0]), mu, sigma), [0.0, 0.0])


def test_a_constant_feature_cannot_be_scaled():
    X = np.array([[1.0, 5.0], [2.0, 5.0], [3.0, 5.0]])

    with pytest.raises(ValueError):
        zscore_normalize(X)


def test_unscaled_parameters_give_the_same_predictions():
    X, _ = make_deliveries()
    _, mu, sigma = zscore_normalize(X)
    w_scaled, b_scaled = np.array([5.0, -2.0, 1.5]), 30.0

    w, b = unscale_parameters(w_scaled, b_scaled, mu, sigma)

    from_scaled = predict(apply_zscore(X, mu, sigma), w_scaled, b_scaled)
    from_original = predict(X, w, b)
    np.testing.assert_allclose(from_original, from_scaled)


def test_scaling_reaches_the_minimum_where_raw_features_do_not():
    X, y = make_deliveries()
    A = np.column_stack([X, np.ones(len(X))])
    solution = np.linalg.lstsq(A, y, rcond=None)[0]
    cost_best = compute_cost(X, y, solution[:-1], solution[-1])
    X_scaled, _, _ = zscore_normalize(X)

    _, _, scaled_costs = gradient_descent(X_scaled, y, np.zeros(3), 0.0, alpha=0.1, num_iters=200)
    _, _, raw_costs = gradient_descent(X, y, np.zeros(3), 0.0, alpha=1e-8, num_iters=200)

    assert scaled_costs[-1] - cost_best < 1e-3
    assert raw_costs[-1] - cost_best > 10


def test_training_on_scaled_features_finds_the_exact_solution():
    X, y = make_deliveries()
    A = np.column_stack([X, np.ones(len(X))])
    solution = np.linalg.lstsq(A, y, rcond=None)[0]
    X_scaled, mu, sigma = zscore_normalize(X)

    w_scaled, b_scaled, _ = gradient_descent(X_scaled, y, np.zeros(3), 0.0, alpha=0.1, num_iters=1000)
    w, b = unscale_parameters(w_scaled, b_scaled, mu, sigma)

    np.testing.assert_allclose(w, solution[:-1], rtol=1e-3)
    assert b == pytest.approx(solution[-1], rel=1e-3)
