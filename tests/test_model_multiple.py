import numpy as np
import pytest

from machine_learning_from_scratch.model import predict, predict_loop


def test_predict_with_several_features_matches_a_hand_calculation():
    x = np.array([2000.0, 3.0, 1.0])
    w = np.array([0.003, 1.0, 7.0])

    # 0.003 * 2000 + 1.0 * 3 + 7.0 * 1 + 10 = 6 + 3 + 7 + 10 = 26
    assert predict(x, w, b=10.0) == pytest.approx(26.0)


def test_vectorized_predict_matches_the_loop():
    rng = np.random.default_rng(0)
    x, w = rng.random(50), rng.random(50)

    assert predict(x, w, b=3.0) == pytest.approx(predict_loop(x, w, b=3.0))


def test_predict_handles_many_examples_at_once():
    X = np.array([[1000.0, 2.0, 0.0], [2000.0, 3.0, 1.0]])
    w = np.array([0.003, 1.0, 7.0])

    np.testing.assert_allclose(predict(X, w, b=10.0), [15.0, 26.0])
