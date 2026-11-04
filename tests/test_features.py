import numpy as np
import pytest

from machine_learning_from_scratch.cost import compute_cost
from machine_learning_from_scratch.data_curved import make_curved_deliveries
from machine_learning_from_scratch.data_multiple import make_deliveries
from machine_learning_from_scratch.features import add_interaction, polynomial_features
from machine_learning_from_scratch.gradient_descent import gradient_descent
from machine_learning_from_scratch.scaling import unscale_parameters, zscore_normalize


def best_cost(X, y):
    """Lowest possible cost for these features (exact least squares)."""
    A = np.column_stack([X, np.ones(len(X))])
    solution = np.linalg.lstsq(A, y, rcond=None)[0]
    return compute_cost(X, y, solution[:-1], solution[-1])


def test_add_interaction_appends_the_product_of_two_columns():
    X = np.array([[2.0, 3.0, 1.0], [4.0, 5.0, 0.0]])

    result = add_interaction(X, 0, 2)

    np.testing.assert_allclose(result, [[2.0, 3.0, 1.0, 2.0], [4.0, 5.0, 0.0, 0.0]])


def test_polynomial_features_matches_a_hand_calculation():
    result = polynomial_features(np.array([1.0, 2.0, 3.0]), 3)

    np.testing.assert_allclose(result, [[1, 1, 1], [2, 4, 8], [3, 9, 27]])


def test_the_interaction_feature_captures_what_three_features_cannot():
    X, y = make_deliveries(rush_slowdown=0.002)

    plain = best_cost(X, y)
    engineered = best_cost(add_interaction(X, 0, 2), y)

    assert plain > 7
    assert engineered < 4.5  # close to the noise floor, 3.0**2 / 2


def test_a_curve_needs_the_squared_feature():
    km, y = make_curved_deliveries()

    line = best_cost(polynomial_features(km, 1), y)
    parabola = best_cost(polynomial_features(km, 2), y)

    assert parabola < line / 2


def test_gradient_descent_on_scaled_polynomial_features_finds_the_best_fit():
    km, y = make_curved_deliveries()
    features = polynomial_features(km, 2)
    X_scaled, mu, sigma = zscore_normalize(features)

    w_s, b_s, _ = gradient_descent(X_scaled, y, np.zeros(2), 0.0, alpha=0.1, num_iters=5000)
    w, b = unscale_parameters(w_s, b_s, mu, sigma)

    assert compute_cost(features, y, w, b) == pytest.approx(best_cost(features, y), abs=1e-3)


def test_raw_polynomial_columns_diverge_with_an_alpha_that_scaled_ones_handle():
    km, y = make_curved_deliveries()
    features = polynomial_features(km, 3)

    with np.errstate(all="ignore"):
        _, _, raw = gradient_descent(features, y, np.zeros(3), 0.0, alpha=1e-4, num_iters=100)

    assert raw[-1] > raw[0]
