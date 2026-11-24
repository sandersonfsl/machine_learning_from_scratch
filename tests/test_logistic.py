import numpy as np
import pytest

from machine_learning_from_scratch.logistic import predict_class, predict_proba, sigmoid


def test_sigmoid_of_zero_is_one_half():
    assert sigmoid(0) == 0.5


def test_sigmoid_matches_hand_calculated_values():
    # g(1) = 1 / (1 + e^-1) = 0.731..., g(-1) = 1 / (1 + e) = 0.268...
    result = sigmoid(np.array([-1.0, 0.0, 1.0, 2.0]))

    np.testing.assert_allclose(result, [0.26894142, 0.5, 0.73105858, 0.88079708], rtol=1e-6)


def test_sigmoid_is_symmetric_around_one_half():
    z = np.linspace(-10, 10, 21)

    np.testing.assert_allclose(sigmoid(-z), 1 - sigmoid(z))


def test_sigmoid_stays_between_0_and_1_and_always_grows():
    z = np.linspace(-20, 20, 201)

    g = sigmoid(z)

    assert np.all(g > 0)
    assert np.all(g < 1)
    assert np.all(np.diff(g) > 0)


def test_sigmoid_does_not_overflow_for_very_large_values():
    with np.errstate(all="raise"):  # any overflow warning becomes an error
        low, high = sigmoid(-1000), sigmoid(1000)

    assert low == pytest.approx(0.0, abs=1e-12)
    assert high == pytest.approx(1.0, abs=1e-12)


def test_sigmoid_keeps_the_shape_of_the_input():
    z = np.zeros((3, 4))

    assert sigmoid(z).shape == (3, 4)


# --- the logistic model ---------------------------------------------------------


def test_predict_proba_matches_a_hand_calculation():
    # z = 1*2 + 2*(-1) + 0.5 = 0.5, and g(0.5) = 0.62245933
    p = predict_proba(np.array([2.0, -1.0]), np.array([1.0, 2.0]), 0.5)

    assert p == pytest.approx(0.62245933, rel=1e-6)


def test_predict_proba_accepts_many_examples_at_once():
    X = np.array([[0.0, 0.0], [1.0, 1.0], [-1.0, 3.0]])
    w, b = np.array([0.5, -0.25]), 0.1

    together = predict_proba(X, w, b)
    one_by_one = [predict_proba(row, w, b) for row in X]

    assert together.shape == (3,)
    np.testing.assert_allclose(together, one_by_one)


def test_predict_proba_is_one_half_on_the_decision_boundary():
    # the boundary of w = [1, 1], b = -3 is the line x1 + x2 = 3
    w, b = np.array([1.0, 1.0]), -3.0
    on_the_line = np.array([[1.0, 2.0], [3.0, 0.0], [-1.0, 4.0]])

    np.testing.assert_allclose(predict_proba(on_the_line, w, b), 0.5)


def test_predict_class_is_1_exactly_when_z_is_not_negative():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 3))
    w, b = np.array([1.0, -2.0, 0.5]), 0.3

    z = np.dot(X, w) + b

    np.testing.assert_array_equal(predict_class(X, w, b), (z >= 0).astype(int))


def test_predict_class_on_the_boundary_is_1():
    assert predict_class(np.array([1.0, 2.0]), np.array([1.0, 1.0]), -3.0) == 1


def test_predict_class_threshold_changes_the_decision():
    X = np.array([[0.0], [1.0], [2.0]])  # probabilities 0.5, 0.73, 0.88
    w, b = np.array([1.0]), 0.0

    np.testing.assert_array_equal(predict_class(X, w, b), [1, 1, 1])
    np.testing.assert_array_equal(predict_class(X, w, b, threshold=0.8), [0, 0, 1])
