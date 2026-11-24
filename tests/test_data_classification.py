import numpy as np

from machine_learning_from_scratch.data_classification import make_late_deliveries


def test_shapes_and_labels_are_0_or_1():
    X, y = make_late_deliveries(m=100)

    assert X.shape == (100, 2)
    assert y.shape == (100,)
    assert set(np.unique(y)) == {0, 1}


def test_same_seed_gives_the_same_data():
    X1, y1 = make_late_deliveries(seed=3)
    X2, y2 = make_late_deliveries(seed=3)

    np.testing.assert_array_equal(X1, X2)
    np.testing.assert_array_equal(y1, y2)


def test_both_classes_are_reasonably_common():
    _, y = make_late_deliveries(m=100)

    assert 0.3 < y.mean() < 0.7


def test_long_deliveries_are_late_more_often_than_short_ones():
    X, y = make_late_deliveries(m=500)
    distance = X[:, 0]

    assert y[distance > 7_000].mean() > y[distance < 3_000].mean()
